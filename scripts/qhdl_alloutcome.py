#!/usr/bin/env python3
"""All-outcome equivalence screen for invsqrt rewrites that touch conditioned gates.

    python3 scripts/qhdl_alloutcome.py PARENT CHILD [--out DIR] [--max-targets N] [--rand-runs K] [--json]

PARENT and CHILD are .kmb or .kmx invsqrt circuits (a .kmb is converted with the accepted kickmix venv into
--kmx-dir, default DIR/kmx). Exit code 0 = no counterexample and every conditioned changed op was activated,
1 = REJECT (a row differs), 2 = INCONCLUSIVE (clean, but a changed op was never activated or targets were
skipped), 3 = usage error.

Why. `qhdl fdiff --whole` compares the zero branch and one random branch. A gate nested under many PUSH_CONDITION
scopes (e.g. 13e23a0d71c2: 15 CCZ deletions) fires with probability ~2^-k under random outcomes, so those
branches never reach it. This tool drives the measurement outcomes instead of sampling them blindly.

Method.
  1. Align the two op streams (difflib on the op text). Aligned HMRs share one outcome variable; unaligned HMRs
     get independent variables. Both circuits are run on identical outcomes, so a correct rewrite must agree row by
     row, not only in distribution.
  2. A symbolic classical layer evaluates every classical bit as a GF(2) polynomial (ANF) in the outcome variables,
     including nested PUSH_CONDITION scopes, conditional HMR writes and BIT_* ops. For each changed op the
     activation condition is the AND of the scope bits and its own `if b`.
  3. The affine part of each activation condition is solved by Gaussian elimination. The solution space (not one
     point) is sampled independently per row, so one run covers all 3,133,452 legal x against about 3 million
     different activating outcome vectors. An inconsistent affine part proves the op never executes (reported).
  4. Per run: per-row phase (sign + Z_POW units), full final qubit state, x restored and scratch clean for both
     circuits. The activation and effect row counts of every changed op are recorded as coverage evidence.

Tier: local prescreen. The outcomes are sampled inside the activating subspace, not enumerated, and a nonaffine
part of a condition is handled by rejection sampling (its measured activation count is the coverage evidence).
This is stronger than three branch draws and it rejects the known 13e23a0d71c2 false positive, but it is not a
proof; the unmodified official verifier stays the authority. Library: check(ops1, ops2, xq, ...) -> dict.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_bitsim as B  # noqa: E402

import vqcc_config as CFG  # noqa: E402
KMPY = CFG.KMPY
POLY_CAP = 4000          # monomials; a larger classical bit is treated as unknown (None)


# ----------------------------------------------------------------------------------------------- loading
def to_kmx(path, kmx_dir):
    """Return a .kmx path for a .kmb or .kmx circuit (conversion needs the accepted kickmix venv)."""
    p = Path(path)
    if p.suffix == '.kmx':
        return p
    kmx_dir = Path(kmx_dir)
    kmx_dir.mkdir(parents=True, exist_ok=True)
    out = kmx_dir / (p.stem + '.kmx')
    if not out.exists():
        code = ('import sys\nfrom kickmix import Circuit\n'
                'c = Circuit.from_file(sys.argv[1])\nc.to_file(sys.argv[2], format="kmx")')
        subprocess.run([KMPY, '-I', '-c', code, str(p), str(out)], check=True, capture_output=True, text=True)
    return out


def load(path, kmx_dir):
    kmx = to_kmx(path, kmx_dir)
    ops, _hdr, xq = B.parse_kmx(str(kmx))
    nq = 1 + max([max(o.q) for o in ops if o.q] + list(xq))
    return ops, xq, nq


# ------------------------------------------------------------------------------------------ GF(2) polynomials
# A polynomial is a frozenset of monomials; a monomial is an int bitmask over outcome variables (0 = constant 1).
ONE = frozenset({0})
ZERO = frozenset()


def p_xor(a, b):
    if a is None or b is None:
        return None
    return a ^ b


def p_mul(a, b):
    if a is None or b is None:
        return None
    if not a or not b:
        return ZERO
    if a == ONE:
        return b
    if b == ONE:
        return a
    r = set()
    for x in a:
        for y in b:
            r ^= {x | y}
    if len(r) > POLY_CAP:
        return None
    return frozenset(r)


def p_eval(poly, assign_mask):
    """Scalar evaluation at an outcome assignment given as an int bitmask of the variables that are 1."""
    v = 0
    for m in poly:
        v ^= 1 if (m & ~assign_mask) == 0 else 0
    return v


def classical_layer(ops, hmr_var, wanted):
    """Run the classical bits symbolically. hmr_var: op index of each HMR -> variable id. wanted: op indices whose
    activation factors are returned: {i: [poly, ...]} (scope bits, then the op's own condition; None = unknown)."""
    bits = {}
    stack = []
    out = {}

    def bit(b):
        return bits.get(b, ZERO)

    for i, o in enumerate(ops):
        c = o.code
        if c == B.C_INFO:
            continue
        if c == B.C_PUSH:
            if i in wanted:
                out[i] = list(stack) + [bit(o.cond)]
            stack.append(bit(o.cond))
            continue
        if c == B.C_POP:
            if i in wanted:
                out[i] = list(stack)
            if stack:
                stack.pop()
            continue
        fac = list(stack) + ([bit(o.cond)] if o.cond >= 0 else [])
        if i in wanted:
            out[i] = fac
        if c in (B.C_HMR, B.C_BINV, B.C_BST0, B.C_BST1):
            act = ONE
            for f in fac:
                act = p_mul(act, f)
                if act is None:
                    break
            old = bit(o.bit)
            if c == B.C_HMR:
                m = frozenset({1 << hmr_var[i]})
                new = p_xor(old, p_mul(act, p_xor(old, m)))
            elif c == B.C_BINV:
                new = p_xor(old, act)
            elif c == B.C_BST0:
                new = p_xor(old, p_mul(act, old))
            else:
                new = p_xor(old, p_mul(act, p_xor(ONE, old)))
            bits[o.bit] = new
    return out


# ------------------------------------------------------------------------------------------------ solving
def solve_activation(factors, nvars):
    """Affine part of 'all factors = 1'. Returns dict(status, pivots {var: (freemask, const)}, nonaffine, unknown).
    status 'unreachable' when the affine equations are inconsistent (a necessary condition fails: op never runs)."""
    cbit = 1 << nvars
    rows, nonaff, unknown = [], 0, 0
    for f in factors:
        if f is None:
            unknown += 1
            continue
        if all(bin(m).count('1') <= 1 for m in f):
            r = 0
            for m in f:
                r ^= cbit if m == 0 else m
            rows.append(r ^ cbit)             # sum(vars) + const = 1  ->  sum(vars) = 1 + const
        else:
            nonaff += 1
    piv = {}
    for r in rows:
        for v, pr in piv.items():
            if (r >> v) & 1:
                r ^= pr
        lo = r & (cbit - 1)
        if lo == 0:
            if r & cbit:
                return dict(status='unreachable', pivots={}, nonaffine=nonaff, unknown=unknown)
            continue
        v = lo.bit_length() - 1
        for u in list(piv):
            if (piv[u] >> v) & 1:
                piv[u] ^= r
        piv[v] = r
    pivots = {v: ((r & (cbit - 1)) & ~(1 << v), (r >> nvars) & 1) for v, r in piv.items()}
    return dict(status='ok', pivots=pivots, nonaffine=nonaff, unknown=unknown)


# ------------------------------------------------------------------------------------------------ alignment
def align(ops1, ops2):
    t1 = [o.text for o in ops1]
    t2 = [o.text for o in ops2]
    sm = difflib.SequenceMatcher(None, t1, t2, autojunk=False)
    match = {}
    d1, d2 = [], []
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(a1 - a0):
                match[a0 + k] = b0 + k
        else:
            d1.extend(range(a0, a1))
            d2.extend(range(b0, b1))
    return match, d1, d2


def hmr_variables(ops1, ops2, match):
    """Variable ids: parent HMR ordinals; aligned child HMRs share them; the rest get fresh ids."""
    h1 = [i for i, o in enumerate(ops1) if o.code == B.C_HMR]
    h2 = [i for i, o in enumerate(ops2) if o.code == B.C_HMR]
    var1 = {i: k for k, i in enumerate(h1)}
    inv = {j: i for i, j in match.items()}
    var2, nxt = {}, len(h1)
    for j in h2:
        if j in inv and inv[j] in var1:
            var2[j] = var1[inv[j]]
        else:
            var2[j] = nxt
            nxt += 1
    return var1, var2, nxt


# ------------------------------------------------------------------------------------------------ simulation
class OSim(B.BitSim):
    """BitSim whose HMR outcomes come from a provider keyed by outcome variable."""

    def __init__(self, ops, hmr_var, provider, **kw):
        super().__init__(ops, outcome='zero', **kw)
        self.hmr_var, self.provider, self.cur = hmr_var, provider, -1

    def _outcome(self):
        return self.provider(self.hmr_var[self.cur])


def effect_mask(sim, o, act):
    q = sim.q
    a = o.q
    c = o.code
    if c == B.C_CCZ:
        m = q[a[0]] & q[a[1]] & q[a[2]]
    elif c == B.C_CCX:
        m = q[a[0]] & q[a[1]]
    elif c == B.C_CZ:
        m = q[a[0]] & q[a[1]]
    elif c in (B.C_Z, B.C_CX, B.C_ZPOW, B.C_HMR, B.C_R):
        m = q[a[0]]
    elif c == B.C_CSWAP:
        m = q[a[0]] & (q[a[1]] ^ q[a[2]])
    elif c == B.C_SWAP:
        m = q[a[0]] ^ q[a[1]]
    else:
        m = sim.valid
    return m if act is None else m & act


def activation(sim, o):
    act = None
    for s in sim.stack:
        act = s if act is None else act & s
    if o.cond >= 0:
        cb = sim._bit(o.cond)
        act = cb if act is None else act & cb
    return act


def execute(ops, hmr_var, provider, targets, xq, nq, rows):
    sim = OSim(ops, hmr_var, provider, rows=rows, num_qubits=nq, x_qubits=xq, count_toffoli=True)
    events = sorted(set(hmr_var) | set(targets))
    cov = {}
    for e in events:
        sim.run(stop=e)
        if e in targets:
            o = ops[e]
            act = activation(sim, o)
            n_act = sim.n if act is None else int(np.bitwise_count(act & sim.valid).sum())
            em = effect_mask(sim, o, act) if o.code not in (B.C_PUSH, B.C_POP) else None
            n_eff = None if em is None else int(np.bitwise_count(em & sim.valid).sum())
            cov[e] = (n_act, n_eff)
        if e in hmr_var:
            sim.cur = e
            sim.run(stop=e + 1)
    sim.run()
    return sim, cov


def make_provider(kind, seed, W, valid, sol=None):
    """kind 'zero' | 'rand' | 'sol'. All draws are keyed by (seed, variable) so both circuits see the same bits."""
    cache = {}

    def rnd(v):
        a = cache.get(v)
        if a is None:
            a = np.random.default_rng([seed, v]).integers(0, 1 << 64, W, dtype=np.uint64, endpoint=False) & valid
            cache[v] = a
        return a

    def prov(v):
        if kind == 'zero':
            return np.zeros(W, np.uint64)
        if kind == 'sol' and sol and v in sol['pivots']:
            free, const = sol['pivots'][v]
            a = valid.copy() if const else np.zeros(W, np.uint64)
            f = free
            while f:
                low = f & -f
                a = a ^ rnd(low.bit_length() - 1)
                f ^= low
            return a
        return rnd(v)
    return prov


def final_view(sim, xq, rows):
    bad = np.zeros(sim.W, np.uint64)
    for k, qq in enumerate(xq):
        bad |= sim.q[qq] ^ B.pack_bits(((rows >> k) & 1).astype(bool))
    xs = set(xq)
    dirty = np.zeros(sim.W, np.uint64)
    for w in range(sim.nq):
        if w not in xs:
            dirty |= sim.q[w]
    return bad, dirty


def compare_runs(s1, s2, xq, rows):
    state = np.zeros(s1.W, np.uint64)
    for w in range(max(s1.nq, s2.nq)):
        if w < s1.nq and w < s2.nq:
            state |= s1.q[w] ^ s2.q[w]
        else:
            state |= (s1.q[w] if w < s1.nq else s2.q[w])
    pd = s1.phase_units() != s2.phase_units()
    b1, d1 = final_view(s1, xq, rows)
    b2, d2 = final_view(s2, xq, rows)

    def cnt(a):
        return int(np.bitwise_count(a & s1.valid).sum())
    first = np.flatnonzero(pd)
    return dict(phase_differs=int(pd.sum()), state_differs=cnt(state),
                x_bad=[cnt(b1), cnt(b2)], scratch_dirty=[cnt(d1), cnt(d2)],
                first_row_phase_differs=int(rows[first[0]]) if len(first) else None,
                mean_toffoli=[round(s1.toffoli_rows / s1.n, 4), round(s2.toffoli_rows / s2.n, 4)])


# ------------------------------------------------------------------------------------------------ driver
def check(ops1, ops2, xq, nq1=None, nq2=None, rows=None, max_targets=60, rand_runs=2, log=None, seed=1):
    t0 = time.time()
    log = log or (lambda *a: None)
    rows = B.legal_rows() if rows is None else np.asarray(rows, dtype=np.int64)
    nq1 = nq1 or 1 + max(max(o.q) for o in ops1 if o.q)
    nq2 = nq2 or 1 + max(max(o.q) for o in ops2 if o.q)
    match, d1, d2 = align(ops1, ops2)
    var1, var2, nvars = hmr_variables(ops1, ops2, match)
    log(f'aligned: {len(match)} equal ops, {len(d1)} parent-only, {len(d2)} child-only, {nvars} outcome variables')
    f1 = classical_layer(ops1, var1, set(d1))
    f2 = classical_layer(ops2, var2, set(d2))

    targets, seen, n_uncond = [], {}, 0
    for side, dlist, fmap in ((1, d1, f1), (2, d2, f2)):
        for i in dlist:
            fac = fmap.get(i, [])
            if all(f == ONE for f in fac):
                n_uncond += 1
                continue
            sol = solve_activation(fac, nvars)
            key = (sol['status'], tuple(sorted((v, fm, c) for v, (fm, c) in sol['pivots'].items())), sol['nonaffine'])
            if key in seen:
                seen[key]['ops'][side].append(i)
                continue
            t = dict(sol=sol, ops={1: [], 2: []})
            t['ops'][side].append(i)
            seen[key] = t
            targets.append(t)
    unreachable = [t for t in targets if t['sol']['status'] == 'unreachable']
    live = [t for t in targets if t['sol']['status'] == 'ok']
    skipped = max(0, len(live) - max_targets)
    live = live[:max_targets]
    log(f'{n_uncond} unconditioned changed ops, {len(live) + skipped} distinct activation targets '
        f'({len(unreachable)} provably unreachable, {skipped} beyond --max-targets)')

    plans = [('zero', 'zero', 0, None)]
    plans += [(f'rand{k}', 'rand', seed + k, None) for k in range(rand_runs)]
    plans += [(f'target{k}', 'sol', 1000 + k, t) for k, t in enumerate(live)]
    runs, rejected, uncovered = [], False, []
    W = (len(rows) + 63) // 64
    valid = B.pack_bits(np.ones(len(rows), bool))
    for label, kind, sd, t in plans:
        prov = make_provider(kind, sd, W, valid, t['sol'] if t else None)
        c1 = set(t['ops'][1]) if t else set()
        c2 = set(t['ops'][2]) if t else set()
        s1, cov1 = execute(ops1, var1, prov, c1, xq, nq1, rows)
        s2, cov2 = execute(ops2, var2, prov, c2, xq, nq2, rows)
        r = compare_runs(s1, s2, xq, rows)
        ok = (r['phase_differs'] == 0 and r['state_differs'] == 0
              and not any(r['x_bad']) and not any(r['scratch_dirty']))
        r.update(label=label, kind=kind, equivalent=ok)
        if t:
            acts = [cov1[i] for i in c1] + [cov2[i] for i in c2]
            r['targets'] = dict(
                ops_parent=sorted(c1)[:6], ops_child=sorted(c2)[:6], n_ops=len(c1) + len(c2),
                nonaffine_factors=t['sol']['nonaffine'], unknown_factors=t['sol']['unknown'],
                min_activated_rows=min(a[0] for a in acts), max_activated_rows=max(a[0] for a in acts),
                max_effect_rows=max([a[1] or 0 for a in acts] + [0]))
            if r['targets']['max_activated_rows'] == 0:
                uncovered.append(label)
        runs.append(r)
        rejected |= not ok
        log(f"  {label}: {'equivalent' if ok else 'DIFFERS'} phase_diff={r['phase_differs']} "
            f"state_diff={r['state_differs']}"
            + (f" activated<={r['targets']['max_activated_rows']} effect<={r['targets']['max_effect_rows']}" if t else ''))
        if rejected and kind == 'sol':
            break  # a counterexample is enough
    verdict = ('REJECT' if rejected else
               'INCONCLUSIVE' if (uncovered or skipped) else 'EQUIVALENT_ON_ACTIVATING_SAMPLES')
    return dict(verdict=verdict, tier='local prescreen (sampled outcomes inside the activating subspace; not a proof)',
                n_rows=len(rows), aligned_ops=len(match), parent_only_ops=len(d1), child_only_ops=len(d2),
                unconditioned_changed_ops=n_uncond, conditioned_targets=len(live) + skipped,
                provably_unreachable_targets=len(unreachable), targets_skipped=skipped,
                uncovered_targets=uncovered, nonaffine_targets=sum(1 for t in live if t['sol']['nonaffine']),
                runs=runs, seconds=round(time.time() - t0, 1))


def sha12(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('parent')
    ap.add_argument('child')
    ap.add_argument('--out', help='directory for the JSON evidence file (never overwritten)')
    ap.add_argument('--kmx-dir', help='where .kmb conversions are written (default OUT/kmx or a temp dir)')
    ap.add_argument('--max-targets', type=int, default=60)
    ap.add_argument('--rand-runs', type=int, default=2)
    ap.add_argument('--rows', type=int, default=0, help='first N legal x only (debug)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    kd = a.kmx_dir or (str(Path(a.out) / 'kmx') if a.out else '/tmp/qhdl_alloutcome_kmx')

    def logf(*s):
        print(*s, file=sys.stderr, flush=True)
    try:
        ops1, xq1, nq1 = load(a.parent, kd)
        ops2, xq2, nq2 = load(a.child, kd)
    except Exception as exc:  # noqa: BLE001
        print(f'error: {exc!r}', file=sys.stderr)
        return 3
    if xq1 != xq2:
        print('error: parent and child x registers differ', file=sys.stderr)
        return 3
    rows = B.legal_rows()[:a.rows] if a.rows else None
    res = check(ops1, ops2, xq1, nq1, nq2, rows=rows, max_targets=a.max_targets, rand_runs=a.rand_runs, log=logf)
    res.update(parent=sha12(a.parent), child=sha12(a.child), parent_path=str(a.parent), child_path=str(a.child))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        f = out / f'alloutcome_{res["parent"]}_{res["child"]}.json'
        if f.exists():
            print(f'refusing to overwrite {f}', file=sys.stderr)
            return 3
        f.write_text(json.dumps(res, indent=1))
        res['evidence'] = str(f)
    summary = {k: v for k, v in res.items() if k != 'runs'}
    summary['runs'] = [{k: v for k, v in r.items() if k in ('label', 'equivalent', 'phase_differs', 'targets')}
                       for r in res['runs'] if (not r['equivalent'] or r['kind'] != 'sol')]
    print(json.dumps(res if a.json else summary, indent=1))
    return {'REJECT': 1, 'INCONCLUSIVE': 2}.get(res['verdict'], 0)


if __name__ == '__main__':
    sys.exit(main())
