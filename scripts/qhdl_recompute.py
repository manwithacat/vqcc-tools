#!/usr/bin/env python3
"""qhdl plugin: `recompute` -- can a value erased at one cut be restored at a later cut from other live values, and
what does it cost? (consumer of the `qhdl valuegraph` value graph and its JSON contract)

    qhdl recompute -c CIRCUIT --query query.json [--out DIR] [--rows N]
    qhdl recompute -c CIRCUIT --target q23 --erase-after 447 --restore-before 15918 \
                   --available q13:v6,q4:v7,... [--max-extra-qubits 0] [--max-and-cost 25]

Query (recompute_query): {source_sha256, target_value_id, erase_after_op, restore_before_op, available_value_ids,
max_extra_qubits, max_and_cost}. `target_value_id` may be one id, a bare wire/bit (`q23`: the version live at the
restore cut) or, as an extension, `target_value_ids` (list: jointly synthesised).

Option available_at='erase_cut' reads every available value that is live at the erase cut at that cut (the window-entry
state: splice semantics) instead of at the restore cut.

Cut convention (as in valuegraph): cut k = before op k. The erase cut is erase_after_op + 1, the restore cut is restore_before_op.
A value is evaluated at the restore cut when it is live there, else at the erase cut, else at its birth cut.

Determination (exact on the stated domain): every legal x (1..3,133,452) is simulated (qhdl_bitsim, bit-parallel) in the
all-HMR-outcome-0 branch; the available values and the target(s) are read at their cuts. The target is `determined` iff
no two legal inputs give identical available state and different target. If not, `witness` holds one such pair
(x0 with target 0, x1 with target 1; smallest such x of the first conflicting available-state class), re-simulated
alone on those two inputs as an independent check. Measurement outcomes are independent of x, so the zero branch is the
conditional-on-no-flip view; branches with outcome 1 are NOT enumerated (domain says so). The target is excluded from its own
available set. Classical bit values read at the cut in the zero branch.

When determined: dependency set (greedy minimal subset of the available values that still determines the target; order
dependent, not unique), then pricing with the window machinery (`qhdl_window.analyse`): pivots of the affine hull of the
reachable dependency states, exact ANF, AND floor lb/ub. price_status:
  exact       lb == ub (affine: 0; exactly-quadratic on a full affine hull: rank/2 by Dickson), placement unchecked;
  bounded     best constructive upper bound (monomial-cache AND network, or the quadratic-span basis), with the lower
              bound beside it; deg >= 3 outputs are always `bounded`;
  not_priced  too many pivots for an exact ANF (limit qhdl_window.ANF_FULL_MAX), or the dependency set needs > 64 values.
and_cost is the constructive count (Toffoli-like gates), scratch_peak the extra wires that construction holds at once,
cleanup the obligations (cached products are removed with HMR + CZ, 0 Toffolis; the target's own erase/restore
needs a phase repair that this tool does NOT prove: phase_repair.status = unknown). A KMB fragment (CX + CCX text ops
on the supplied wires, in-place operand forms undone afterwards) is built and exactly checked on all legal x for
affine and exactly-quadratic cases on quantum values; otherwise `fragment` is null with the reason.
Product-span route (run 2, scripts/qhdl_productspan.py + .c): when the available quantum values take at most 1024 distinct
states, a sequential k-product search ('fix-first-factors' linear algebra with Gray-code enumeration of the last factor
and free choice among the 2^r - 1 combinations of the still-missing targets) captures the targets one combination at a
time; optional helper products (query search.helpers, built first, held in scratch wires under a pool of
len(targets) + max_extra_qubits, removed by HMR + CZ/Z of their operand forms) enlarge the operand space.  The fragment
is emitted as CX/CCX text (plus HMR/CZ/Z-if lines for helpers), every row is checked, price_status stays `bounded`
(a construction, no lower bound).  Query search = {seconds, procs, helpers, seed, max_bits}.
A lower AND count is not an improvement by itself: `feasible_within_budget` compares and_cost / scratch_peak with
max_and_cost / max_extra_qubits, and `net` states what is and is not charged.

Tier: exact-all-x for determination over the stated domain (zero branch); pricing is a static construction bound plus the
exact fragment check where a fragment exists. Not a verifier PASS: nothing here is spliced into a circuit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_bitsim as B  # noqa: E402
import qhdl_valuegraph as G  # noqa: E402

SCHEMA = 'qhdl-recompute-result/v1'
MAX_DEPS_KEY = 64
BARE = re.compile(r'^([qb])(\d+)$')


# ---------------------------------------------------------------- value resolution and state reads
def live_at_cut(v, c):
    return v['birth_cut'] <= c and (v['kill_cut'] is None or c < v['kill_cut'])


def resolve_value(values, ref, restore_cut, erase_cut):
    """-> value record. ref: full id, or bare q23 / b5 (version live at the restore cut, else at the erase cut)."""
    if ref in values:
        return values[ref]
    m = BARE.match(ref)
    if not m:
        raise KeyError(f'unknown value id {ref!r}')
    key = 'wire' if m.group(1) == 'q' else 'bit'
    num = int(m.group(2))
    cands = [v for v in values.values() if v.get(key) == num and v['id'][0] == m.group(1)]
    for c in (restore_cut, erase_cut):
        hit = [v for v in cands if live_at_cut(v, c)]
        if hit:
            return hit[-1]
    raise KeyError(f'{ref}: no version live at cut {restore_cut} or {erase_cut}')


def eval_cut(v, restore_cut, erase_cut):
    for c in (restore_cut, erase_cut):
        if live_at_cut(v, c):
            return c, c == restore_cut
    return v['birth_cut'], False


def read_values(ops, xq, nq, recs, cuts, rows, outcome='zero'):
    """recs: value records; cuts: {id: cut}. -> {id: bool array over rows}, n."""
    sim = B.BitSim(ops, rows=rows, num_qubits=nq, x_qubits=xq, outcome=outcome)
    want = {}
    for v in recs:
        want.setdefault(cuts[v['id']], []).append(v)
    got = {}

    def cb(i, s):
        for v in want.get(i, []):
            if v['id'][0] == 'q':
                words = s.q[v['wire']]
            else:
                words = s._bit(v['bit'])
            got[v['id']] = B.unpack_bits(words.copy(), s.n)

    stop = max(want) if want else 0
    sim.run(stop=stop, snaps=sorted(want), on_snap=cb)
    return got, sim.n


# ---------------------------------------------------------------- determination
def _words(cols):
    """list of bool arrays -> list of uint64 key words (<= 64 bits each)."""
    n = len(cols[0]) if cols else 0
    out = []
    for s in range(0, len(cols), 64):
        w = np.zeros(n, np.uint64)
        for j, c in enumerate(cols[s:s + 64]):
            w |= c.astype(np.uint64) << np.uint64(j)
        out.append(w)
    return out or [np.zeros(n, np.uint64)]


def _order(words):
    return np.argsort(words[0], kind='stable') if len(words) == 1 else np.lexsort(words[::-1])


def conflicts(cols, targets):
    """-> None if every target is a function of cols; else (order, bad_first_index, sorted_words) of the first
    conflicting adjacent pair (in sorted order). cols: available value columns; targets: list of bool arrays."""
    words = _words(cols)
    order = _order(words)
    sw = [w[order] for w in words]
    same = np.ones(len(order) - 1, bool)
    for w in sw:
        same &= w[1:] == w[:-1]
    bad_any = np.zeros(len(order) - 1, bool)
    which = np.full(len(order) - 1, -1, np.int64)
    for ti, t in enumerate(targets):
        tv = t[order]
        b = same & (tv[1:] != tv[:-1]) & ~bad_any
        which[b] = ti
        bad_any |= b
    if not bad_any.any():
        return None
    first = int(np.flatnonzero(bad_any)[0])
    return dict(order=order, first=first, which=int(which[first]), sorted_words=sw)


def witness_pair(conf, cols, target, rows):
    """x0 (target 0) and x1 (target 1) from the first conflicting available-state class."""
    order, first = conf['order'], conf['first']
    sw = conf['sorted_words']
    lo = first
    while lo > 0 and all(w[lo - 1] == w[first] for w in sw):
        lo -= 1
    hi = first + 1
    while hi < len(order) - 1 and all(w[hi + 1] == w[first] for w in sw):
        hi += 1
    members = order[lo:hi + 1]
    t = target[members]
    r0 = members[~t]
    r1 = members[t]
    i0, i1 = int(r0[np.argmin(rows[r0])]), int(r1[np.argmin(rows[r1])])
    return int(rows[i0]), int(rows[i1]), len(members)


def minimise_deps(cols, ids, targets):
    """Greedy minimal sub-list of ids whose columns still determine all targets (drops in list order)."""
    keep = list(range(len(ids)))
    for j in list(keep):
        trial = [i for i in keep if i != j]
        if not trial:
            continue
        if conflicts([cols[i] for i in trial], targets) is None:
            keep = trial
    # a constant target needs no dependencies
    if all(t.all() or not t.any() for t in targets):
        keep = []
    return [ids[i] for i in keep], keep


# ---------------------------------------------------------------- pricing
def _cache_ub(monos_list):
    """AND count of a monomial-cache network: nonlinear monomial (mask) = parent (one fewer var) AND one variable.
    Greedy: reuse any present parent, else chain down by dropping the highest variable. -> (count, nodes list)."""
    need = sorted({m for ms in monos_list for m in ms if bin(m).count('1') >= 2}, key=lambda m: (bin(m).count('1'), m))
    built = set()

    def add(m):
        if bin(m).count('1') < 2 or m in built:
            return
        for j in range(m.bit_length()):
            if m >> j & 1 and (m ^ (1 << j) in built or bin(m ^ (1 << j)).count('1') == 1):
                built.add(m)
                return
        p = m ^ (1 << (m.bit_length() - 1))
        add(p)
        built.add(m)

    for m in need:
        add(m)
    return len(built), sorted(built)


def _dickson(monos, k):
    """quadratic part of an ANF (monomial masks over k pivots) -> list of (L1 mask, L2 mask) with
    quad = XOR of L1*L2 products up to affine terms (symplectic reduction)."""
    rows = [0] * k
    for m in monos:
        if bin(m).count('1') == 2:
            i = (m & -m).bit_length() - 1
            j = m.bit_length() - 1
            rows[i] ^= 1 << j
            rows[j] ^= 1 << i
    pairs = []
    while True:
        i = next((a for a in range(k) if rows[a]), None)
        if i is None:
            return pairs
        j = (rows[i] & -rows[i]).bit_length() - 1
        ci, cj = rows[i] & ~(1 << j), rows[j] & ~(1 << i)   # B_ic and B_jc over c not in {i, j}
        l1 = (1 << i) | cj                                    # x_i + sum_c B_jc x_c
        l2 = (1 << j) | ci                                    # x_j + sum_c B_ic x_c
        pairs.append((l1, l2))
        # remove i, j and update B'_cd = B_cd + B_jc B_id + B_jd B_ic
        for c in range(k):
            if c in (i, j):
                continue
            row = rows[c] & ~((1 << i) | (1 << j))
            if cj >> c & 1:
                row ^= ci
            if ci >> c & 1:
                row ^= cj
            row &= ~(1 << c)
            rows[c] = row
        rows[i] = rows[j] = 0


def _mask_wires(mask, pivot_wires):
    return [pivot_wires[j] for j in range(len(pivot_wires)) if mask >> j & 1]


def _apply(ops_text, state):
    """tiny evaluator for X/CX/CCX text ops on {wire: bool array}."""
    for line in ops_text:
        t = line.split()
        w = [int(a[1:]) for a in t[1:]]
        if t[0] == 'X':
            state[w[0]] = ~state[w[0]]
        elif t[0] == 'CX':
            state[w[1]] = state[w[1]] ^ state[w[0]]
        elif t[0] == 'CCX':
            state[w[2]] = state[w[2]] ^ (state[w[0]] & state[w[1]])
        else:
            raise ValueError(line)


def build_fragment(res_outputs, out_names, pivots_dep, dep_vals, target_wires, dep_cols, target_cols, k, nq_free):
    """CX/CCX fragment for affine and exactly-quadratic outputs. pivots_dep: list of dep value records (pivot order).
    -> (lines, info) or (None, reason). Verified exactly on the supplied rows."""
    if any(v['id'][0] != 'q' for v in pivots_dep):
        return None, 'a pivot is a classical bit: no CX/CCX fragment'
    pw = [v['wire'] for v in pivots_dep]
    lines = []
    ands = 0
    for nm, tw in zip(out_names, target_wires):
        o = res_outputs[nm]
        monos = o['monos']
        if monos is None or o['deg'] is None or o['deg'] > 2:
            return None, f'output degree {o.get("deg")} > 2: no fragment builder (cache network not emitted)'
        if tw in pw:
            return None, 'target wire is also a dependency wire'
        for l1, l2 in _dickson(monos, k):
            a_vars, b_vars = _mask_wires(l1, pw), _mask_wires(l2, pw)
            # operand wires: lowest-index var of each form; the other vars are added in place with CX and undone
            wa, wb = a_vars[0], b_vars[0]
            if wa == wb:
                return None, 'operand forms share a carrier wire'
            pre = [f'CX q{c} q{wa}' for c in a_vars[1:]] + [f'CX q{c} q{wb}' for c in b_vars[1:]]
            if any(c == wb for c in a_vars[1:]) or any(c == wa for c in b_vars[1:]):
                return None, 'operand carrier appears in the other form (in-place order conflict)'
            lines += pre + [f'CCX q{wa} q{wb} q{tw}'] + pre[::-1]
            ands += 1
        # affine remainder, solved against the true column below
        lines.append(('AFFINE', nm, tw))
    # resolve affine remainders numerically
    n = len(next(iter(dep_cols.values())))
    st = {w: dep_cols[v['id']].copy() for v, w in zip(pivots_dep, pw)}
    for tw in target_wires:
        st[tw] = np.zeros(n, bool)
    final = []
    from qhdl_cheap_function import solve_small
    for ln in lines:
        if isinstance(ln, tuple):
            _, nm, tw = ln
            tcol = target_cols[nm]
            resid = tcol ^ st[tw]
            feats = [(('1',), np.ones(n, bool))] + [((f'p{j}',), dep_cols[v['id']]) for j, v in enumerate(pivots_dep)]
            sol = _solve_affine(feats, resid)
            if sol is None:
                return None, 'quadratic part leaves a non-affine residual (not exactly quadratic over these pivots)'
            for term in sol:
                if term == '1':
                    final.append(f'X q{tw}'); _apply([final[-1]], st)
                else:
                    final.append(f'CX q{pw[int(term[1:])]} q{tw}'); _apply([final[-1]], st)
        else:
            final.append(ln)
            _apply([ln], st)
    for tw, nm in zip(target_wires, out_names):
        if not np.array_equal(st[tw], target_cols[nm]):
            return None, 'fragment check failed on the supplied rows'
    for v, w in zip(pivots_dep, pw):
        if not np.array_equal(st[w], dep_cols[v['id']]):
            return None, 'fragment leaves a dependency wire modified'
    return final, dict(ands=sum(1 for x in final if x.startswith('CCX')), rows_checked=int(n))


def _solve_affine(feats, target):
    from qhdl_window import bool_to_int, solve_gf2
    sol = solve_gf2([(f[0][0], bool_to_int(f[1])) for f in feats], bool_to_int(target))
    return sol



def quadratic_route(avail_recs, avail_cols, targets_named, sample=6000, seed=1):
    """Cheap-function route (qhdl_cheap_function.analyse_sample): each target as 1 + linear + pair products of the
    available values, solved on a row sample and re-checked exactly on every row. -> dict per target (kind, cost,
    terms, exact) or None when there are no available values. Cost per target = rank(B)/2 of its quadratic part after
    the cheap_function support reduction (not a MinRank optimum over the don't-cares)."""
    import qhdl_cheap_function as CF
    if not avail_recs:
        return None
    n = len(next(iter(targets_named.values())))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    vals = {i: CF.to_int(B.pack_bits(c[idx]), len(idx)) for i, c in enumerate(avail_cols)}
    out = {}
    for nm, col in targets_named.items():
        tint = CF.to_int(B.pack_bits(col[idx]), len(idx))
        f = CF.analyse_sample(tint, vals, -1, len(idx))
        row = dict(kind=f['kind'], cost=f.get('cost'), exact=None, terms=None)
        if f['kind'] in ('zero', 'one', 'affine', 'quadratic') and f.get('terms') is not None or f['kind'] in ('zero', 'one'):
            terms = f['terms']
            acc = np.zeros(n, bool)
            for t in terms:
                if t == ('1',):
                    acc ^= True
                elif len(t) == 1:
                    acc ^= avail_cols[int(t[0][1:])]
                else:
                    acc ^= avail_cols[int(t[0][1:])] & avail_cols[int(t[1][1:])]
            row['exact'] = bool(np.array_equal(acc, col))
            row['terms'] = [list(t) for t in terms]
            if not row['exact']:
                row['cost'] = None
        out[nm] = row
    return out


def quad_to_monos(terms, order):
    """terms over value indices -> monomial masks over `order` (list of value indices); drops the constant."""
    pos = {v: j for j, v in enumerate(order)}
    return [sum(1 << pos[int(a[1:])] for a in t) for t in terms if t != ('1',)]


def price(dep_recs, dep_cols, target_cols_named, target_wires=None):
    """Price a set of target columns as functions of dep values. -> dict (price_status, and_cost, ...), fragment info."""
    import qhdl_window as W
    k_raw = len(dep_recs)
    if k_raw == 0:
        return dict(price_status='exact', and_cost=0, scratch_peak=0, lower_bound=0, construction='constant target',
                    notes=['target is constant on the domain']), None
    if k_raw > MAX_DEPS_KEY:
        return dict(price_status='not_priced', reason=f'{k_raw} dependency values > {MAX_DEPS_KEY} key bits'), None
    names = [f'q{i}' for i in range(k_raw)]
    key = _words(dep_cols)[0]
    ukeys, first_idx, counts = np.unique(key, return_index=True, return_counts=True)
    outcols = {nm: c[first_idx] for nm, c in target_cols_named.items()}
    res = W.analyse(names, ukeys, counts.astype(np.int64), outcols, None, {nm: True for nm in outcols})
    out = dict(reachable_states=int(len(ukeys)), pivots=[dep_recs[int(p[1:])]['id'] for p in res['pivots']],
               affine_hull_dim=res['k'], dont_cares_in_hull=res['dont_cares_in_hull'], anf_kind=res['anf_kind'])
    outs = res['outputs']
    degs = {nm: o.get('deg') for nm, o in outs.items()}
    out['output_degrees'] = degs
    fl = res['floor']
    out['and_lower_bound'] = fl['lb']
    if any(o['monos'] is None for o in outs.values()):
        out.update(price_status='not_priced', reason=res['anf_kind'], floor=fl)
        return out, None
    max_deg = max(o['deg'] for o in outs.values())
    cache_n, cache_nodes = _cache_ub([o['monos'] for o in outs.values()])
    cand = [('monomial-cache network', cache_n, max(0, cache_n - sum(
        1 for o in outs.values() if o['deg'] >= 2 and len(o['monos']) == 1 and bin(o['monos'][0]).count('1') >= 2)))]
    if fl['ub'] is not None:
        cand.append(('quadratic-span basis (Dickson)', fl['ub'], 0 if len(outs) == 1 else 1))
    cons, ac, sp = min(cand, key=lambda t: (t[1], t[2]))
    exact_ok = max_deg <= 2 and fl['kind'].startswith('exact') and fl['ub'] == fl['lb'] == ac
    if max_deg <= 1:
        out.update(price_status='exact', and_cost=0, scratch_peak=0, construction='affine: CX only')
    else:
        out.update(price_status='exact' if exact_ok else 'bounded', and_cost=ac, scratch_peak=sp, construction=cons)
    out['floor'] = {k: fl[k] for k in ('lb', 'ub', 'kind')}
    out['anf'] = {nm: W.anf_text(o['monos'], res['pivot_names'], 300) for nm, o in outs.items()}
    out['_pivot_idx'] = [int(p[1:]) for p in res['pivots']]
    out['_res_outputs'] = outs
    out['_k'] = res['k']
    return out, res


# ---------------------------------------------------------------- the query
def productspan_route(avail, targets, got0, query, srch, olds=None, dep_recs=None):
    """Product-span synthesis over the available quantum values (needs <= 1024 distinct states).  -> dict (ok, ...)."""
    import qhdl_productspan as PS
    if any(t['id'][0] != 'q' for t in targets):
        return dict(ok=False, reason='a target is a classical bit')
    qa = [(v['wire'], got0[v['id']]) for v in avail if v['id'][0] == 'q']
    tg = [(t['wire'], got0[t['id']]) for t in targets]
    if dep_recs and olds is None and all(v['id'][0] == 'q' for v in dep_recs):
        # too many distinct states over the full available set (> kernel limit): use the minimal dependency subset,
        # whose reachable-state hull is small (still available wires, so the fragment is legal on them)
        import qhdl_productspan as _PS
        if _PS.instance_from_columns(qa, [c for _, c in tg]) is None and 'exceed' in _PS.LAST_FAIL.get('why', ''):
            qa = [(v['wire'], got0[v['id']]) for v in dep_recs]
    extra = int(query.get('max_extra_qubits') or 0)
    if olds is not None:
        return PS.synth_from_columns_inplace(qa, tg, olds, seed=int(srch.get('seed', 1)),
                                             seconds=float(srch.get('seconds', 20)), nproc=int(srch.get('procs', 1)),
                                             max_bits=int(srch.get('max_bits', 24)))
    return PS.synth_from_columns(qa, tg, extra=extra, helpers=int(srch.get('helpers', 0)), seed=int(srch.get('seed', 1)),
                                 seconds=float(srch.get('seconds', 20)), nproc=int(srch.get('procs', 1)),
                                 max_bits=int(srch.get('max_bits', 24)))


def run_query(ops, xq, nq, values, query, rows=None, outcomes=('zero',), progress=None):
    t0 = time.time()
    erase_cut = int(query['erase_after_op']) + 1
    restore_cut = int(query['restore_before_op'])
    tids = query.get('target_value_ids') or [query['target_value_id']]
    notes = []
    targets = [resolve_value(values, t, restore_cut, erase_cut) for t in tids]
    avail = []
    for a in query.get('available_value_ids', []):
        v = resolve_value(values, a, restore_cut, erase_cut)
        if v['id'] in [t['id'] for t in targets]:
            notes.append(f"available value {v['id']} is the target itself: dropped from the available set")
            continue
        if v['id'] not in [x['id'] for x in avail]:
            avail.append(v)
    cuts = {}
    at_erase = query.get('available_at') == 'erase_cut'
    for v in targets + avail:
        if at_erase and v in avail and live_at_cut(v, erase_cut):
            cuts[v['id']] = erase_cut          # splice semantics: inputs are the window-entry state
            continue
        c, live = eval_cut(v, restore_cut, erase_cut)
        cuts[v['id']] = c
        if not live and v in avail:
            notes.append(f"available value {v['id']} is not live at the restore cut {restore_cut}; read at cut {c}")
    # in-place model (run 3): the version each target wire holds at the erase cut is an extra operand and the start value
    oldrecs = []
    if query.get('target_start') == 'old_value_at_erase_cut':
        for t in targets:
            c = [v for v in values.values() if v.get('wire') == t.get('wire') and v['id'][0] == 'q'
                 and live_at_cut(v, erase_cut) and v['id'] != t['id']]
            oldrecs.append(c[-1] if c else None)
            if c:
                cuts[c[-1]['id']] = erase_cut
        if any(t['id'][0] != 'q' for t in targets):
            oldrecs = []
            notes.append('target_start ignored: a target is a classical bit')
    domain_rows = B.legal_rows() if rows is None else np.asarray(rows, dtype=np.int64)
    result = dict(schema=SCHEMA, type='recompute_result', source_sha256=query.get('source_sha256'),
                  target_value_ids=[t['id'] for t in targets], erase_after_op=int(query['erase_after_op']),
                  restore_before_op=restore_cut, available_value_ids=[v['id'] for v in avail],
                  max_extra_qubits=query.get('max_extra_qubits'), max_and_cost=query.get('max_and_cost'),
                  domain=dict(inputs=int(len(domain_rows)), x_range=[int(domain_rows.min()), int(domain_rows.max())],
                              outcomes=[str(o) for o in outcomes],
                              statement=f'{len(domain_rows)} inputs, all-HMR-outcome branches {list(map(str, outcomes))}; '
                                        'classical values read in those branches; outcome-1 branches not enumerated'),
                  evidence_tier='exact-all-x' if rows is None else 'exact on the supplied rows', notes=notes,
                  read_cuts=cuts, phase_repair=dict(status='unknown', note='erase/restore phase obligations are not '
                                                    'proved by this tool'),
                  certificate=dict(status='unknown', note='no source-relative circuit certificate at this stage'))
    determined = True
    all_recs = targets + avail + [o for o in oldrecs if o]
    colsets = []
    for oc in outcomes:
        got, n = read_values(ops, xq, nq, all_recs, cuts, domain_rows, oc)
        colsets.append(got)
        tcols = [got[t['id']] for t in targets]
        acols = [got[v['id']] for v in avail]
        if progress:
            progress(f'branch {oc}: simulated {n} inputs in {time.time() - t0:.1f}s')
        conf = conflicts(acols or [np.zeros(n, bool)], tcols)
        if conf is not None:
            determined = False
            ti = conf['which']
            x0, x1, size = witness_pair(conf, acols, tcols[ti], domain_rows)
            wit = dict(branch=str(oc), target_value_id=targets[ti]['id'], x_target0=x0, x_target1=x1,
                       conflicting_class_size=size)
            # independent re-simulation on the two inputs alone
            g2, _ = read_values(ops, xq, nq, all_recs, cuts, np.array([x0, x1]), oc)
            same = {v['id']: bool(g2[v['id']][0] == g2[v['id']][1]) for v in avail}
            wit['available_state_identical'] = all(same.values())
            wit['differing_available'] = [k for k, s in same.items() if not s]
            wit['target_values'] = [int(g2[targets[ti]['id']][0]), int(g2[targets[ti]['id']][1])]
            wit['reverified'] = bool(wit['available_state_identical'] and wit['target_values'] == [0, 1])
            result['witness'] = wit
            result['determined'] = False
            result['seconds'] = round(time.time() - t0, 1)
            return result
    result['determined'] = True
    # dependency set: minimal subset (zero branch columns; checked on every branch afterwards)
    got0 = colsets[0]
    tcols0 = [got0[t['id']] for t in targets]
    acols0 = [got0[v['id']] for v in avail]
    dep_ids, keep = minimise_deps(acols0, [v['id'] for v in avail], tcols0) if avail else ([], [])
    for gi, got in enumerate(colsets[1:], 1):
        d_cols = [got[i] for i in dep_ids]
        if dep_ids and conflicts(d_cols, [got[t['id']] for t in targets]) is not None:
            notes.append(f'minimal dependency set of branch 0 does not determine the target in branch {gi}')
    dep_recs = [v for v in avail if v['id'] in dep_ids]
    result['dependency_ids'] = dep_ids
    result['dependency_minimal'] = 'greedy, list-order dependent; not unique'
    dep_cols = {v['id']: got0[v['id']] for v in dep_recs}
    named = {f'o{i}': got0[t['id']] for i, t in enumerate(targets)}
    pr, res = price(dep_recs, [got0[v['id']] for v in dep_recs], named)
    frag_info = None
    fragment = None
    if res is not None and pr.get('price_status') in ('exact', 'bounded') and pr['and_cost'] is not None:
        piv = [dep_recs[i] for i in pr['_pivot_idx']]
        twires = [t['wire'] if t['id'][0] == 'q' else None for t in targets]
        if None in twires:
            fragment, frag_info = None, 'a target is a classical bit'
        else:
            fragment, frag_info = build_fragment(pr['_res_outputs'], list(named), piv, None, twires, dep_cols, named,
                                                 pr['_k'], 0)
    for key in ('_pivot_idx', '_res_outputs', '_k'):
        pr.pop(key, None)
    # second route: quadratic over the whole available set (no dependency minimisation)
    qr = quadratic_route(avail, acols0, named)
    if qr is not None:
        pr['quadratic_route'] = {nm: {k: v for k, v in r.items() if k != 'terms'} for nm, r in qr.items()}
        if all(r['exact'] and r['cost'] is not None for r in qr.values()):
            tot = sum(r['cost'] for r in qr.values())
            pr['quadratic_route_total'] = tot
            if pr.get('and_cost') is None or tot < pr['and_cost']:
                used = sorted({int(a[1:]) for r in qr.values() for t in r['terms'] for a in t if a != '1'})
                urecs = [avail[i] for i in used]
                fr2, fi2 = None, 'a used value is classical or a target is a classical bit'
                if all(v['id'][0] == 'q' for v in urecs) and all(t['id'][0] == 'q' for t in targets):
                    fake = {nm: dict(monos=quad_to_monos([tuple(t) for t in r['terms']], used), deg=2)
                            for nm, r in qr.items()}
                    fr2, fi2 = build_fragment(fake, list(named), urecs, None, [t['wire'] for t in targets],
                                              {v['id']: got0[v['id']] for v in urecs}, named, len(used), 0)
                pr.update(price_status='bounded', and_cost=tot, scratch_peak=0 if fr2 else None,
                          construction='quadratic over available values (sum of per-target rank/2; no sharing)',
                          dependency_ids_of_route=[avail[i]['id'] for i in used])
                if fr2:
                    fragment, frag_info = fr2, fi2
                else:
                    fragment, frag_info = None, fi2
    # third route: product-span synthesis: k products per captured target combination, helper products
    # under a wire pool, exact fragment check on every row.  Skipped when the price is already exact.
    srch = query.get('search') or {}
    if determined and avail and not srch.get('disable') and pr.get('price_status') != 'exact':
        olds = [None if o is None else got0[o['id']] for o in oldrecs] if oldrecs else None
        psr = productspan_route(avail, targets, got0, query, srch, olds, dep_recs=dep_recs)
        pr['product_span'] = {k: v for k, v in psr.items() if k != 'ops'}
        if psr.get('ok') and (pr.get('and_cost') is None or psr['and_cost'] < pr['and_cost']):
            pr.update(price_status='bounded', and_cost=psr['and_cost'], scratch_peak=psr['scratch_peak'],
                      construction='product-span: %d products (%d helpers) from the sequential k-product search'
                                   % (psr['and_cost'], psr['helper_products']),
                      fragment_start_model=psr.get('start_model', 'targets start at 0'))
            fragment, frag_info = psr['ops'], dict(ands=psr['and_cost'], rows_checked=psr['rows_checked'],
                                                   note='X/CX/CCX plus HMR/CZ/Z-if lines for helper removal; operand '
                                                        'carriers built in place; helper values checked at removal')
    result['price'] = pr
    result['price_status'] = pr.get('price_status')
    result['and_cost'] = pr.get('and_cost')
    result['scratch_peak'] = pr.get('scratch_peak')
    result['fragment'] = None if fragment is None else dict(format='kmb text ops', ops=fragment,
                                                            exact_check=frag_info)
    result['fragment_unavailable_reason'] = None if fragment is not None else frag_info
    result['cleanup'] = dict(
        cached_products='measure-based uncompute: HMR + CZ(parent, variable) in reverse order, 0 Toffolis'
        if pr.get('scratch_peak') else 'none (no cached products)',
        dependency_wires='operand forms are built in place with CX and undone, 0 Toffolis, no dirty residue',
        target_wire='erase (HMR) and restore (recompute + Z-by-outcome) carry the phase_repair obligation: unknown')
    bc, bq = query.get('max_and_cost'), query.get('max_extra_qubits')
    ok = None
    if pr.get('and_cost') is not None:
        ok = (bc is None or pr['and_cost'] <= bc) and (bq is None or pr['scratch_peak'] <= bq)
    result['feasible_within_budget'] = ok
    result['net'] = ('and_cost counts Toffoli-like gates of the recompute only; the erase HMR, the phase repair and any '
                     'width change at the plateau are not priced here. A cheaper recompute is an improvement only if '
                     'the schedule that needs it fits the width, which is the caller\'s check.')
    result['seconds'] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------- qhdl verb
def v_recompute(api, a):
    import qhdl_envelope as E
    meta = api.ensure_open(a.circuit)
    d = api.ws_dir(meta['sha256'])
    ops, xq, nq = G.load_ops(d / 'circuit.kmx')
    nq = max(nq, meta['declared_qubits'])
    if a.query:
        query = json.loads(Path(a.query).read_text())
    else:
        query = dict(target_value_id=a.target, erase_after_op=a.erase_after, restore_before_op=a.restore_before,
                     available_value_ids=[x for x in (a.available or '').split(',') if x],
                     max_extra_qubits=a.max_extra_qubits, max_and_cost=a.max_and_cost)
    if not query.get('available_value_ids'):
        # an empty available set makes every target "NOT determined", which agents misread as a width verdict
        raise api.QhdlError('recompute needs a non-empty available set (--available or available_value_ids); an empty '
                            'set cannot determine anything and is not a width verdict',
                            'list the live values to recompute from, e.g. every other live wire at the restore cut '
                            '(qhdl window / peak show which values are live)')
    srch = dict(query.get('search') or {})
    for key, val in (('seconds', a.search_seconds), ('helpers', a.helpers), ('procs', a.procs), ('max_bits', a.max_bits),
                     ('seed', a.search_seed)):
        if val is not None:
            srch[key] = val
    if srch:
        query['search'] = srch
    sha = query.get('source_sha256')
    if sha and not meta['sha256'].startswith(sha[:12]):
        raise api.QhdlError(f"query source_sha256 {sha[:12]} does not match the circuit {meta['sha12']}")
    query['source_sha256'] = meta['sha256']
    values, uses, info = G.build(ops, xq, nq, G.read_intervals(d))
    rows = None if not a.rows else np.random.default_rng(1).choice(np.arange(1, B.MAX_X + 1), a.rows, replace=False)
    outcomes = ['zero'] + ([int(s) for s in a.seed] if a.seed else [])
    t0 = time.time()
    res = run_query(ops, xq, nq, values, query, rows=rows, outcomes=outcomes)
    ev = Path(a.out) / f"recompute_{hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:12]}.json" \
        if a.out else d / 'recompute' / f"{hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:12]}.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    if ev.exists() and a.out:
        raise api.QhdlError(f'{ev} exists; outputs are append-only', 'choose a new --out directory')
    full = dict(query=query, **res)
    ev.write_text(json.dumps(full, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else str(o)) + '\n')
    led = E.Ledger()
    led.earn('static', 'value graph from the op stream')
    led.earn('exact-all-x' if rows is None else 'sampled(%d)' % len(rows),
             f'bit-parallel simulation of {res["domain"]["inputs"]} inputs, outcome branches {outcomes}')
    if res['determined']:
        claim = (f"{meta['sha12']}: target {','.join(res['target_value_ids'])} is determined by "
                 f"{len(res['dependency_ids'])} of {len(res['available_value_ids'])} available values; price "
                 f"{res['price_status']}" + (f", {res['and_cost']} ANDs, scratch {res['scratch_peak']}"
                                              if res.get('and_cost') is not None else ''))
    else:
        w = res['witness']
        claim = (f"{meta['sha12']}: target {','.join(res['target_value_ids'])} is NOT determined by the "
                 f"{len(res['available_value_ids'])} available values: x={w['x_target0']} and x={w['x_target1']} give "
                 'identical available state and opposite target')
    facts = {k: res[k] for k in ('determined', 'price_status', 'and_cost', 'scratch_peak', 'feasible_within_budget',
                                 'dependency_ids') if k in res}
    if not res['determined']:
        facts['witness'] = res['witness']
    rr = api.cmd('recompute', '-c', meta['sha12'], '--query', str(a.query)) if a.query else api.cmd(
        'recompute', '-c', meta['sha12'], '--target', a.target, '--erase-after', a.erase_after, '--restore-before',
        a.restore_before, '--available', a.available or '')
    return E.Envelope(led, claim, domain=res['domain']['statement'], evidence=api.rel(ev), rerun=rr,
                      tool_sha=api.tool_sha(), circuit_sha=meta['sha256'], facts=facts, rows=[], parts=[],
                      next_cmds=[])


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('recompute', help='can an erased value be restored from other live values, and at what '
                                                'AND cost (recompute_query)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--circuit', required=True)
    p.add_argument('--query', help='recompute_query JSON file')
    p.add_argument('--target'), p.add_argument('--erase-after', type=int), p.add_argument('--restore-before', type=int)
    p.add_argument('--available', help='comma list of value ids (required unless --query gives available_value_ids)')
    p.add_argument('--max-extra-qubits', type=int, default=0), p.add_argument('--max-and-cost', type=int)
    p.add_argument('--rows', type=int, default=0, help='sample this many legal x instead of all (tier sampled)')
    p.add_argument('--seed', action='append', help='extra random HMR-outcome branch (seed); repeatable')
    p.add_argument('--out', help='write the result JSON into this new directory (append-only)')
    p.add_argument('--search-seconds', type=float, help='product-span search time cap (default 20 s)')
    p.add_argument('--helpers', type=int, help='helper products built first (cached in scratch wires, removed by HMR)')
    p.add_argument('--max-bits', type=int, help='product-span search: largest 2^e test budget per step (default 24)')
    p.add_argument('--procs', type=int, help='worker processes for the product-span search (default 1)')
    p.add_argument('--search-seed', type=int)
    p.add_argument('--all', action='store_true')
    p.set_defaults(fn=lambda a: v_recompute(api, a))
