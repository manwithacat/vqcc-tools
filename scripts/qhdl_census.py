#!/usr/bin/env python3
"""qhdl plugin: `census` — dirty-borrow chains and provably-|0> Toffoli targets.

    qhdl census -c CIRCUIT [--region R | --ops A:B] [--proof auto|track|local|structural] [--sampled N]
                [--sites mbu|chains|all] [--min-saving S]

The generalisation tool for one rewrite class (examples: the 4-Toffoli dirty-borrow AND3 uncompute and the 6-input
normalize-flag uncompute in public DQI decoders): an uncompute that pays full price on a borrowed dirty wire, replaced by measurement-based uncomputation (MBU) or a conditionally clean ancilla construction.

What it finds (one row per site):
  chain   a block of consecutive Toffolis (X gates allowed inside) whose net action is exactly
          t ^= AND(n controls, some negated) while every other wire it writes is restored for ANY input value: a
          multi-control X that borrows dirty wires (the 4-Toffoli "toggle twice" AND3, 4(n-2)-Toffoli Barenco
          ladders). Exact: the block's algebraic normal form over all its wire values is computed and compared.
          `pairs_with` names the earlier chain with the same target and controls (its compute).
  single  a lone Toffoli whose target is provably |0> right after it (the target held exactly a*b).
`zero` says whether the target is |0> right after the site, i.e. whether it is an MBU candidate:
  proved          in every shot (a theorem). --proof track: the symbolic tracker (qhdl_track) over the whole span,
                  every chain folded into one multi-control macro so a compute chain and its uncompute cancel even
                  when they borrow dirty wires (about an hour on a whole DQI circuit); --proof local: a structural rule
                  (compute/uncompute pair with target and controls untouched between, target |0> before) and then
                  the tracker on each compute..uncompute window from a cold start (chains <= 200,000 ops apart, lone
                  Toffolis <= 256; ~100 s on DQI); --proof structural: the rule alone; auto = track up to 2M ops,
                  else local.
  sampled(N)      DQI only: a paired chain uncompute that no proof covered, with the target |0> on all N simulated
                  valid syndromes (qhdl_simulate; --sampled N, default 1024). Evidence, not a proof: the 6-input
                  normalize flag is of this kind (the controls are conditionally swapped between compute and
                  uncompute, so value-identity proofs fail).
On yoyobar's DQI 6d58bd6d0232: the 12 dirty-borrow AND3 uncomputes are proved and the 12 6-input flag uncomputes are
sampled(1024); on a circuit with both rewritten to MBU neither appears.

Pricing (static model, Toffolis per shot if the site runs unconditionally; `conditional` sites run in some shots):
  now          Toffolis in the block today.
  mbu          only when zero_after: HMR t -> m, then a phase fix-up C^n Z on the controls if m (probability 1/2):
               0.5 * cost(C^n Z), cost = 0 (n=2), 1 (n=3), n-2 with a clean AND ladder (needs n-3 clean wires
               besides t, which is clean after its HMR), else 2n-3 (conditionally clean, below).
  clean        a clean AND ladder: n-1 Toffolis, the ladder itself uncomputed by MBU (Clifford fix-ups only); needs
               n-2 clean wires at the site.
  cc           a conditionally clean ancilla construction (Khattar & Gidney, "Rise of conditionally clean ancillae
               for optimizing quantum circuits", 2024): priced at 2n-3 Toffolis with the wires already borrowed, the
               figure from that paper's construction (not re-derived here; the construction is not built).
  saving       now - the cheapest option that fits in the free wires (`best`); `saving_with_width` ignores width.
  free_clean   wires free at the site: peak live width minus the largest live width over the block (allocator
               interval model of `qhdl peak`); `width` names a cheaper option that needs k wires more than are free.
               Example: the flag rewrite above reused the wire the AND3 rewrite freed, so it realised 16 -> 2.25, not
               the 4.5 priced at free 0.
Every priced row is a static model; only `zero` = proved and the chain's net function are exact.

Phases: chains are matched on values (bit flips); a Barenco ladder made of Toffolis has no phase side effect.
A rewrite still needs fdiff (invsqrt) or the DQI draw gate / official verifier.
Library: find_chains(k, A, B) -> list of chains; census(path, A, B, proof=..., sampled=N) -> dict.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_kmb as K  # noqa: E402

CCX, X_, TAG = 0x31, 0x11, 0x06
RUN_OPS = (CCX, X_, TAG)   # ops a chain may contain (INFO_TAGs pass through)
MAX_BLOCK = 400            # ops per chain block (a 100-control Barenco ladder)
MAX_TERMS = 4096           # ANF size cap per wire inside one block (abort the start position beyond it)
SITES = ('mbu', 'chains', 'all')
SAMPLED_DEFAULT = 1024


# ---------------------------------------------------------------- algebraic normal form of one block
def _mul(p, q):
    out = set()
    for a in p:
        for b in q:
            out ^= {a | b}
    return out


def _net_and(state, var):
    """If the block's state is 'one wire t gets t ^= AND(literals), all others restored', return (t, [(wire,
    neg)]), else None. state: {wire: set of monomials (int bitmasks over var ids)}; var: {wire: var id}."""
    moved = [w for w, p in state.items() if p != {1 << var[w]}]
    if len(moved) != 1:
        return None
    t = moved[0]
    delta = state[t] ^ {1 << var[t]}
    if not delta:
        return None
    top = max(delta, key=lambda m: bin(m).count('1'))
    deg = bin(top).count('1')
    if deg < 2 or top >> var[t] & 1 or sum(1 for m in delta if bin(m).count('1') == deg) != 1:
        return None
    inv = {v: w for w, v in var.items()}
    lits, prod = [], {0}
    for i in range(top.bit_length()):
        if top >> i & 1:
            neg = (top ^ (1 << i)) in delta
            lits.append((inv[i], neg))
            prod = _mul(prod, {1 << i} ^ ({0} if neg else set()))
    if prod != delta:
        return None
    return t, lits


def _block_from(ops, s, e_max):
    """Smallest block ops[s:e] (e <= e_max) that is a dirty-borrow multi-control X, or None."""
    var, state, written = {}, {}, set()
    toff = 0

    def val(w):
        if w not in var:
            var[w] = len(var)
            state[w] = {1 << var[w]}
        return state[w]

    for e in range(s, e_max):
        code, q = ops[e]
        if code == TAG:
            continue
        if code == X_:
            p = val(q[0])
            state[q[0]] = p ^ {0}
            written.add(q[0])
        else:
            a, b, c = q
            pa, pb, pc = val(a), val(b), val(c)
            add = _mul(pa, pb)
            if len(add) > MAX_TERMS:
                return None
            state[c] = pc ^ add
            written.add(c)
            toff += 1
        if toff >= 2:
            r = _net_and(state, var)
            if r is not None:
                t, lits = r
                borrowed = sorted(w for w in written if w != t and w not in {x for x, _ in lits})
                if len(lits) >= 3 or borrowed:
                    return dict(start=s, end=e + 1, target=t, controls=lits, borrowed=borrowed, toffolis=toff)
        if any(len(p) > MAX_TERMS for p in state.values()):
            return None
    return None


def _op_records(k, idxs):
    """[(code, qubits)] for the given sorted op indices (stream positions by searchsorted)."""
    t = np.asarray(k.op_types)
    cls = t & 0x30
    i3 = np.flatnonzero(cls == 0x30)
    i1 = np.flatnonzero(cls == 0x10)
    s3 = [k.stream(n) for n in ('qqq0', 'qqq1', 'qqq2')]
    s1 = k.stream('q0')
    out = {}
    idxs = np.asarray(idxs, np.int64)
    c = t[idxs]
    m3 = (c & 0x30) == 0x30
    m1 = (c & 0x30) == 0x10
    p3 = np.searchsorted(i3, idxs[m3])
    p1 = np.searchsorted(i1, idxs[m1])
    q3 = np.stack([np.asarray(s[p3]) for s in s3], 1) if len(p3) else np.zeros((0, 3), np.int64)
    q1 = np.asarray(s1[p1]) if len(p1) else np.zeros(0, np.int64)
    for i, q in zip(idxs[m3].tolist(), q3.tolist()):
        out[i] = (int(t[i]), tuple(q))
    for i, q in zip(idxs[m1].tolist(), q1.tolist()):
        out[i] = (int(t[i]), (int(q),))
    for i in idxs[~(m3 | m1)].tolist():
        out[i] = (int(t[i]), ())
    return out


def find_chains(k, A=0, B=None):
    """Dirty-borrow multi-control X blocks in ops [A, B): runs of CCX/X/INFO_TAG with >= 2 CCX, segmented greedily
    into the smallest exact blocks. Returns a list of dicts (start, end, target, controls [(wire, negated)],
    borrowed, toffolis, depth)."""
    t = np.asarray(k.op_types)
    B = len(t) if B is None else B
    seg = t[A:B]
    ok = np.isin(seg, np.array(RUN_OPS, np.uint8))
    # maximal runs of allowed ops
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    rs = np.flatnonzero(d == 1)
    re_ = np.flatnonzero(d == -1)
    ccx = np.r_[0, np.cumsum(seg == CCX)]
    keep = (ccx[re_] - ccx[rs]) >= 2
    rs, re_ = rs[keep] + A, re_[keep] + A
    if not len(rs):
        return []
    idxs = np.concatenate([np.arange(a, b) for a, b in zip(rs.tolist(), re_.tolist())])
    rec = _op_records(k, idxs)
    depth = K.condition_depth(t) if len(rs) else None
    out = []
    for a, b in zip(rs.tolist(), re_.tolist()):
        ops = {i: rec[i] for i in range(a, b)}
        s = a
        while s < b:
            if ops[s][0] != CCX and ops[s][0] != X_:
                s += 1
                continue
            r = _block_from(ops, s, min(b, s + MAX_BLOCK))
            if r is None:
                s += 1
                continue
            r['depth'] = int(depth[s])
            out.append(r)
            s = r['end']
    return out


# ---------------------------------------------------------------- pricing
def price(n, now, zero_after, free_clean):
    """Options {name: (Toffolis per shot, clean wires needed)} and the best saving (static model)."""
    opts = {}
    if zero_after:
        if n <= 2:
            cz, need = 0, 0
        elif n == 3:
            cz, need = 1, 0
        elif free_clean >= n - 3:
            cz, need = n - 2, n - 3
        else:
            cz, need = 2 * n - 3, 0
        opts['mbu'] = (0.5 * cz, need)
    if n >= 3:
        opts['clean'] = (n - 1, n - 2)
        opts['cc'] = (2 * n - 3, 0)
    ok = {kk: v for kk, v in opts.items() if v[1] <= free_clean}
    best = min(ok.items(), key=lambda kv: (kv[1][0], kv[1][1])) if ok else None
    return opts, best


def best_any(opts):
    return min(opts.items(), key=lambda kv: (kv[1][0], kv[1][1])) if opts else None


# ---------------------------------------------------------------- structural zero proofs (fast, any size)
W_SHIFT = 36


def _write_keys(k, chains):
    """Sorted keys wire << 36 | op for every op that may change a wire's value, with a reset flag (unconditional
    depth-0 R/HMR). Ops inside chain blocks are replaced by one write of the chain's target at its start (borrowed
    wires are restored by the block's proven net action)."""
    t = np.asarray(k.op_types)
    depth = K.condition_depth(t)
    cls = t & 0x30
    base = t & 0x7F
    keys, rst = [], []
    i3 = np.flatnonzero(cls == 0x30)
    i2 = np.flatnonzero(cls == 0x20)
    i1 = np.flatnonzero(cls == 0x10)
    for idx, names, writes in ((i3, ('qqq0', 'qqq1', 'qqq2'), {0x31: (2,), 0x32: (1, 2)}),
                               (i2, ('qq0', 'qq1'), {0x21: (1,), 0x22: (0, 1)}),
                               (i1, ('q0',), {0x11: (0,), 0x12: (0,), 0x50: (0,)})):
        b_ = base[idx]
        for code, slots in writes.items():
            m = b_ == code
            if not m.any():
                continue
            ops = idx[m].astype(np.int64)
            for sl in slots:
                q = np.asarray(k.stream(names[sl]))[m].astype(np.int64)
                keys.append((q << W_SHIFT) | ops)
                if code in (0x12, 0x50):
                    rst.append(((t[ops] & 0x80) == 0) & (depth[ops] == 0))
                else:
                    rst.append(np.zeros(len(ops), bool))
    keys = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    rst = np.concatenate(rst) if rst else np.zeros(0, bool)
    if chains:
        cs = np.array([c['start'] for c in chains], np.int64)
        ce = np.array([c['end'] for c in chains], np.int64)
        ops = keys & ((1 << W_SHIFT) - 1)
        j = np.searchsorted(cs, ops, 'right') - 1
        inside = (j >= 0) & (ops < ce[np.maximum(j, 0)])
        keys, rst = keys[~inside], rst[~inside]
        keys = np.concatenate([keys, (np.array([c['target'] for c in chains], np.int64) << W_SHIFT) | cs])
        rst = np.concatenate([rst, np.zeros(len(chains), bool)])
    o = np.argsort(keys, kind='stable')
    return keys[o], rst[o]


def _writes_between(keys, w, a, b):
    """Number of writes to wire w at ops in [a, b)."""
    return int(np.searchsorted(keys, (w << W_SHIFT) | b) - np.searchsorted(keys, (w << W_SHIFT) | a))


def structural_proofs(k, sites, keys, rst, regs):
    """Sound zero-after proofs without tracking: site S (target t, controls C, block [s, e)) with partner P (same t,
    same literal controls, block [ps, pe), earlier) is proved when both run unconditionally at depth 0, no op in
    [pe, s) writes t or a control, and t is |0> before P (last earlier write an unconditional reset, or t is scratch
    never written before, or that write is a site already proved). Then t = AND(C) after P and 0 after S."""
    proved_at = {}  # (target, write op) -> True for proved uncompute sites (write op = block start)
    out = {}
    for st in sorted(sites, key=lambda x: x['start']):
        p = st.get('partner')
        if p is None or st['depth'] or p['depth']:
            continue
        t = st['target']
        ws = [t] + [w for w, _ in st['controls']]
        if any(_writes_between(keys, w, p['end'], st['start']) for w in ws):
            continue
        j = int(np.searchsorted(keys, (t << W_SHIFT) | p['start'])) - 1
        if j >= 0 and keys[j] >> W_SHIFT == t:
            prev = int(keys[j] & ((1 << W_SHIFT) - 1))
            ok = bool(rst[j]) or proved_at.get((t, prev), False)
        else:
            ok = t not in regs
        if ok:
            out[st['start']] = True
            proved_at[(t, st['start'])] = True
    return out


def _singles_and_pairs(k, chains, A, B):
    """Unconditional depth-0 CCX ops outside chains, each paired with the previous CCX on the same (a, b, t)."""
    t = np.asarray(k.op_types)
    i3 = np.flatnonzero((t & 0x30) == 0x30)
    m = (t[i3] == CCX)
    ops = i3[m].astype(np.int64)
    a = np.asarray(k.stream('qqq0'))[m].astype(np.int64)
    b = np.asarray(k.stream('qqq1'))[m].astype(np.int64)
    tt = np.asarray(k.stream('qqq2'))[m].astype(np.int64)
    sel = (ops >= A) & (ops < B)
    if chains:
        cs = np.array([c['start'] for c in chains], np.int64)
        ce = np.array([c['end'] for c in chains], np.int64)
        j = np.searchsorted(cs, ops, 'right') - 1
        sel &= ~((j >= 0) & (ops < ce[np.maximum(j, 0)]))
    depth = K.condition_depth(t)[ops]
    ops, a, b, tt, depth = ops[sel], a[sel], b[sel], tt[sel], depth[sel]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = (lo << 42) | (hi << 21) | tt
    o = np.lexsort((ops, key))
    same = np.r_[False, key[o][1:] == key[o][:-1]]
    out = []
    for r, prev in zip(np.flatnonzero(same).tolist(), (np.flatnonzero(same) - 1).tolist()):
        i, pi_ = o[r], o[prev]
        mk = lambda x: dict(start=int(ops[x]), end=int(ops[x]) + 1, target=int(tt[x]),
                            controls=[(int(a[x]), False), (int(b[x]), False)], depth=int(depth[x]))
        site = mk(i)
        site['partner'] = mk(pi_)
        out.append(site)
    return out


def pair_chains(chains):
    """Attach to each chain the previous chain with the same target and literal controls (its compute)."""
    last = {}
    for c in chains:
        key = (c['target'], tuple(sorted(c['controls'])))
        if key in last:
            c['partner'] = last[key]
        last[key] = c
    return chains


def _cold_states(k, starts, regs):
    """{A: (zero list per wire, condition depth at A)} as qhdl_track._cold_state, for many cuts at once."""
    t = np.asarray(k.op_types)
    nq = int(k.num_qubits)
    ops, qs, (i3, i2, i1) = K.op_qubits(k)
    keys = (qs.astype(np.int64) << W_SHIFT) | ops
    del ops, qs
    depth = K.condition_depth(t)
    q1 = np.asarray(k.stream('q0')).astype(np.int64)
    isr = ((t[i1] & 0x7F) == 0x12) | ((t[i1] & 0x7F) == 0x50)
    isr &= ((t[i1] & 0x80) == 0) & (depth[i1] == 0)
    rkeys = np.sort((q1[isr] << W_SHIFT) | i1[isr])
    keys.sort()
    out = {}
    wires = np.arange(nq, dtype=np.int64) << W_SHIFT
    for A in sorted(set(starts)):
        j = np.searchsorted(keys, wires | A) - 1
        has = (j >= 0) & ((keys[np.maximum(j, 0)] >> W_SHIFT) == np.arange(nq))
        last = np.where(has, keys[np.maximum(j, 0)], -1)
        zero = ~has
        r = np.searchsorted(rkeys, last[has])
        hit = (r < len(rkeys)) & (rkeys[np.minimum(r, len(rkeys) - 1)] == last[has])
        zero[np.flatnonzero(has)] = hit
        zero[list(regs)] = False
        out[A] = (zero.tolist(), int(depth[A]))
    return out


def local_proofs(path, k, sites, regs, max_gap):
    """Zero-after proofs by tracking only [partner start, site end) for each paired site (cold start: a wire is |0>
    at the window start iff its last earlier touch is an unconditional depth-0 reset). Sound; weaker than a whole
    track only where relations from before the window are needed. Chains are folded into macros."""
    import qhdl_track as T
    wins = []
    for st in sites:
        p = st.get('partner')
        if p is None or st['end'] - p['start'] > max_gap.get(st['kind'], 0):
            continue
        wins.append([p['start'], st['end']])
    if not wins:
        return {}, 0, 0
    wins.sort()
    merged = [wins[0]]
    for a, b in wins[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cold = _cold_states(k, [a for a, _ in merged], regs)
    t = np.asarray(k.op_types)
    st_ = np.array([a for a, _ in merged], np.int64)
    offs = []
    for m in (lambda x: (x & 0x30) == 0x30, lambda x: (x & 0x30) == 0x20, lambda x: (x & 0x30) == 0x10,
              lambda x: (x & 0x40) > 0, lambda x: (x & 0x80) > 0):
        c = np.concatenate([[0], np.cumsum(m(t), dtype=np.int64)])
        offs.append(c[st_])
        del c
    offs = {a: tuple(int(o[i]) for o in offs) for i, a in enumerate(st_.tolist())}
    last_bit = T._last_bit_access(k)
    chains = [x for x in sites if x['kind'] == 'chain']
    proved = {}
    for a, b in merged:
        mac = {c['start']: (c['end'], c['target'], c['controls']) for c in chains if a <= c['start'] and c['end'] <= b}
        r = T.track(path, a, b, macros=mac, cold_state=cold[a], last_bit=last_bit, k=k, offsets=offs[a])
        for s_, _, z0, z1 in r['macro_events']:
            if z1 and not z0:
                proved[s_] = True
        for i, w, code, uncond, single in r['uncomputes']:
            if code == CCX and uncond:
                proved[i] = True
    return proved, len(merged), sum(b - a for a, b in merged)


# ---------------------------------------------------------------- whole census
PROOFS = ('auto', 'track', 'local', 'structural')
LOCAL_GAP = dict(chain=200_000, single=256)


def census(path, A=None, B=None, warm=False, width=True, progress=None, proof='auto', local_gap=None, sampled=0,
           sim_dir=None):
    """proof: 'track' (whole-span symbolic tracker with chain macros; strongest; ~3-10 min per 10M DQI ops),
    'local' (structural rule, then the tracker on each compute..uncompute window: chains up to 200,000 ops apart,
    lone Toffolis up to 256), 'structural' (paired compute/uncompute with untouched operands only; seconds),
    'auto' (track up to 2M ops in the span, else local).
    sampled: DQI only — lanes (valid syndromes, official sampler) on which a paired chain uncompute that no proof
    covered is checked by simulation (qhdl_simulate); a target |0> on every lane gives zero='sampled(N)'."""
    import qhdl_track as T
    t0 = time.time()
    k = K.read(str(path))
    N = len(k.op_types)
    A = 0 if A is None else A
    B = N if B is None else B
    chains = pair_chains(find_chains(k, A, B))
    t_find = time.time() - t0
    if proof == 'auto':
        proof = 'track' if (B - A) <= 2_000_000 else 'local'
    regs = {q for _, cs in k.registers for isq, q, _ in cs if isq}
    singles = []
    cold = None
    local_info = None
    if proof == 'track':
        macros = {c['start']: (c['end'], c['target'], [(w, neg) for w, neg in c['controls']]) for c in chains}
        r = T.track(path, A, B, warm=warm, macros=macros, progress=progress)
        cold = r['cold']
        mev = {s: (old_zero, new_zero) for s, _, old_zero, new_zero in r['macro_events']}
        for c in chains:
            oz, nz = mev.get(c['start'], (False, False))
            c['zero_after'] = bool(nz and not oz)
        inside = {i for c in chains for i in range(c['start'], c['end'])}
        for i, w, code, uncond, single in r['uncomputes']:
            if i in inside or code != CCX:
                continue
            singles.append(dict(start=i, end=i + 1, target=w, conditional=not uncond, zero_after=True))
        t_proof = r['seconds']
    else:
        t1 = time.time()
        keys, rst = _write_keys(k, chains)
        cand = _singles_and_pairs(k, chains, A, B)
        pr = structural_proofs(k, chains + cand, keys, rst, regs)
        del keys, rst
        if proof == 'local':
            gap = dict(LOCAL_GAP, **(local_gap or {}))
            todo = [dict(c, kind='chain') for c in chains if c['start'] not in pr] + \
                   [dict(c, kind='single') for c in cand if c['start'] not in pr]
            lp, nwin, nops = local_proofs(path, k, todo, regs, gap)
            pr.update(lp)
            local_info = dict(windows=nwin, ops=nops, gap=gap)
        for c in chains:
            c['zero_after'] = c['start'] in pr
        for s_ in cand:
            if s_['start'] in pr:
                singles.append(dict(start=s_['start'], end=s_['end'], target=s_['target'], conditional=False,
                                    zero_after=True))
        t_proof = round(time.time() - t1, 1)
    for c in chains:
        c['zero'] = 'proved' if c['zero_after'] else ''
    sampled_info = None
    open_pairs = [c for c in chains if c.get('partner') and not c['zero_after']]
    names = {nm for nm, _ in k.registers}
    if sampled and open_pairs and 'S0' in names and 'E0' in names:
        import qhdl_simulate as SIM
        t2 = time.time()
        sd = Path(sim_dir) if sim_dir else Path(path).parent
        sd.mkdir(parents=True, exist_ok=True)
        inp = sd / f'census_inputs_{sampled}_1.npz'
        if not inp.exists():
            SIM.gen_inputs(path, inp, sampled, 1)
        sim = SIM.simulate(path, inp, probes=[c['end'] for c in open_pairs])
        for c in open_pairs:
            if sim['state_at'][c['end']][c['target']] == 0:
                c['zero'] = f"sampled({sim['lanes']})"
        sampled_info = dict(lanes=sim['lanes'], checked=len(open_pairs),
                            zero=sum(1 for c in open_pairs if c['zero']), seconds=round(time.time() - t2, 1))
    peak, wmax = None, {}
    if width:
        peak, wmax = _site_widths(k, [(c['start'], c['end']) for c in chains] + [(s_['start'], s_['end'])
                                                                                   for s_ in singles])
    starts, paths = _paths(k)
    rows = []
    for c in chains:
        n = len(c['controls'])
        free = (peak - wmax[(c['start'], c['end'])]) if width else None
        opts, best = price(n, c['toffolis'], bool(c['zero']), free if free is not None else 0)
        rows.append(_row(c['start'], c['end'], 'chain', c['target'], n, c['toffolis'], c['zero_after'], opts, best,
                         free, c['depth'] > 0, paths[int(np.searchsorted(starts, c['start'], 'right')) - 1],
                         controls=c['controls'], borrowed=c['borrowed'],
                         partner=c['partner']['start'] if c.get('partner') else None, zero=c['zero']))
    depth = K.condition_depth(k.op_types) if singles else None
    for s_ in singles:
        free = (peak - wmax[(s_['start'], s_['end'])]) if width else None
        opts, best = price(2, 1, True, free or 0)
        rows.append(_row(s_['start'], s_['end'], 'single', s_['target'], 2, 1, True, opts, best, free,
                         s_['conditional'] or int(depth[s_['start']]) > 0,
                         paths[int(np.searchsorted(starts, s_['start'], 'right')) - 1]))
    rows.sort(key=lambda x: x['op'])
    return dict(path=str(path), num_ops=N, A=A, B=B, cold=cold, proof=proof, peak=peak, rows=rows,
                local=local_info, sampled=sampled_info,
                static_toffolis=int(np.isin(np.asarray(k.op_types[A:B]), np.array(K.TOFFOLI_LIKE, np.uint8)).sum()),
                seconds=round(time.time() - t0, 1), seconds_find=round(t_find, 1), seconds_proof=t_proof)


def _row(a, b, pattern, target, n, now, zero_after, opts, best, free, conditional, path, controls=(), borrowed=(),
         partner=None, zero=None):
    zero = ('proved' if zero_after else '') if zero is None else zero
    anyb = best_any(opts)
    need = anyb[1][1] if anyb else 0
    row = dict(op=a, ops=f'{a}..{b - 1}', path=path, pattern=pattern if pattern == 'single' else
               ('mbu-' if zero else '') + f'chain{n}', target=f'q{target}', n=n, now=now,
               zero_after=zero_after, zero=zero, conditional=conditional,
               mbu=opts['mbu'][0] if 'mbu' in opts else None,
               clean=opts['clean'][0] if 'clean' in opts else None, cc=opts['cc'][0] if 'cc' in opts else None,
               best=best[0] if best else None,
               saving=round(now - best[1][0], 2) if best and best[1][0] < now else 0,
               free_clean=free,
               saving_with_width=round(now - anyb[1][0], 2) if anyb and anyb[1][0] < now else 0,
               width=('ok' if free is None or need <= free else f'{anyb[0]} needs width +{need - free}'),
               pairs_with=partner)
    if controls:
        row['controls'] = ' '.join(('~' if neg else '') + f'q{w}' for w, neg in controls)
    if borrowed:
        row['borrowed'] = ' '.join(f'q{w}' for w in borrowed)
    return row


def _paths(k):
    import qhdl_peak as P
    return P.leaf_segments(k)


def _site_widths(k, spans):
    """Peak live width and the max live width over each [a, b) (allocator interval model, as `qhdl peak`)."""
    import qhdl_lifetime_allocator as LA
    t = k.op_types
    is_reset, _ = LA.reset_ops(k)
    ops, qs, order, _ = LA.references(k)
    iv = LA.build_intervals(k, ops, qs, order, is_reset)
    del ops, qs, order
    prof = LA.live_profile(iv, len(t), LA.register_qubits(k), register_gaps=False)
    del iv
    out = {}
    for a, b in spans:
        out[(a, b)] = int(prof[a:b].max()) if b > a else int(prof[a])
    return int(prof.max()), out


# ---------------------------------------------------------------- CLI verb
def v_census(api, a):
    import qhdl_envelope as E
    import qhdl_track as T
    led = E.Ledger()
    meta = api.ensure_open(a.circuit)
    sha, s12 = meta['sha256'], meta['sha12']
    src = api.circuit_path(meta)
    if src.suffix != '.kmb':
        raise api.QhdlError('census reads .kmb circuits', 'convert with qhdl_kmb.to_file or name the .kmb')
    A, B, span = T._span(api, a, src)
    d = api.ws_dir(sha) / 'census'
    tag = f"{A}_{B}{'_warm' if a.warm else ''}_{a.proof}_s{a.sampled}"
    ev = d / f'census_{tag}.json'
    key = dict(tool_sha=api.tool_sha(), verb='census', A=A, B=B, warm=bool(a.warm), proof=a.proof, sampled=a.sampled)
    res = api.cached(ev, key)
    if res is None:
        res = census(src, A, B, warm=a.warm, proof=a.proof, sampled=a.sampled, sim_dir=d)
        res['cache_key'] = key
        res['span'] = span
        d.mkdir(parents=True, exist_ok=True)
        cols = ['op', 'ops', 'path', 'pattern', 'target', 'n', 'now', 'zero', 'conditional', 'mbu', 'clean', 'cc',
                'best', 'saving', 'free_clean', 'saving_with_width', 'width', 'pairs_with', 'controls', 'borrowed']
        tsv = d / f'census_{tag}.tsv'
        api.write_tsv(tsv, cols, [[r_.get(c, '') for c in cols] for r_ in res['rows']])
        res['table'] = api.rel(tsv)
        api.write_json(ev, res)
    how = {'track': 'sound symbolic tracker over the span (qhdl_track, chains folded into macros)',
           'local': 'sound structural rule, then the symbolic tracker on each compute..uncompute window (cold start)',
           'structural': 'sound structural rule (compute/uncompute pair, operands untouched, target |0> before)'}
    led.earn('static', 'chains: exact net function of each block (algebraic normal form over all its wire values); '
                       f"zero 'proved': {how[res['proof']]}, true in every shot; prices and free wires: static model "
                       '(allocator interval liveness)')
    parts = [dict(part='chain functions, zero proved, prices', tier='static')]
    smp = res.get('sampled')
    if smp and smp['zero']:
        led.earn(f"sampled({smp['lanes']})", f"zero 'sampled({smp['lanes']})': paired chain uncomputes no proof "
                                             f"covered, target |0> on all {smp['lanes']} simulated valid syndromes")
        parts.append(dict(part="zero 'sampled'", tier=f"sampled({smp['lanes']})"))
    rows = res['rows']
    allr = rows
    if a.sites == 'mbu':
        rows = [r_ for r_ in rows if r_['zero']]
    elif a.sites == 'chains':
        rows = [r_ for r_ in rows if r_['pattern'] != 'single']
    rows = [r_ for r_ in rows if r_['saving'] >= a.min_saving]
    groups = {}
    for r_ in allr:
        g = groups.setdefault((r_['pattern'], r_['now'], r_['zero']),
                              dict(pattern=r_['pattern'], now=r_['now'], zero=r_['zero'], sites=0, saving=0.0,
                                   conditional=0, width_ok=0))
        g['sites'] += 1
        g['saving'] += r_['saving']
        g['conditional'] += bool(r_['conditional'])
        g['width_ok'] += r_['width'] == 'ok'
    summ = sorted(groups.values(), key=lambda g: -g['saving'])
    for g in summ:
        g['saving'] = round(g['saving'], 2)
    pr = [r_ for r_ in allr if r_['zero'] == 'proved']
    sm = [r_ for r_ in allr if r_['zero'].startswith('sampled')]
    ch = [r_ for r_ in allr if r_['pattern'] != 'single']
    tot = sum(r_['saving'] for r_ in allr)
    unc = sum(r_['saving'] for r_ in allr if not r_['conditional'])
    pct = 100 * tot / res['static_toffolis'] if res['static_toffolis'] else 0
    claim = (f"{s12} {res['span']}: {len(ch)} dirty-borrow multi-control chains and "
             f"{sum(1 for r_ in allr if r_['pattern'] == 'single')} lone Toffolis with a provably |0> target. "
             f"MBU candidates: {len(pr)} with the target proved |0> after (priced saving "
             f"{sum(r_['saving'] for r_ in pr):,.1f} Toffolis/shot)"
             + (f", {len(sm)} more |0> on {smp['lanes']} sampled syndromes only (saving "
                f"{sum(r_['saving'] for r_ in sm):,.1f})" if sm else '')
             + f". All sites: priced saving {tot:,.1f} static Toffolis ({pct:.3f}% of {res['static_toffolis']:,}; "
             f"{unc:,.1f} at unconditional sites); {sum(1 for r_ in allr if r_['width'] != 'ok')} sites have a cheaper "
             f"option that needs more width (saving {sum(r_['saving_with_width'] for r_ in allr):,.1f} if width were "
             'free). Static model; build and verify before claiming')
    facts = dict(table=res['table'], by_pattern=summ[:8], peak=res['peak'], proof=res['proof'],
                 local=res.get('local'), sampled=smp, cold_start=res['cold'], seconds=res['seconds'])
    rr = api.cmd('census', '-c', s12, *(['--region', a.region] if a.region else []),
                 *(['--ops', a.ops] if a.ops else []), '--warm' if a.warm else None,
                 *(['--sites', a.sites] if a.sites != 'all' else []),
                 *(['--min-saving', a.min_saving] if a.min_saving else []),
                 *(['--proof', a.proof] if a.proof != 'auto' else []),
                 *(['--sampled', a.sampled] if a.sampled != SAMPLED_DEFAULT else []))
    nxt = []
    mb = pr + sm
    if mb:
        r0 = min(mb, key=lambda r_: r_['op'])
        nxt.append(api.cmd('track', '-c', s12, '--ops', f"{max(0, r0['op'] - 200)}:{r0['op'] + 200}"))
    nxt.append(api.cmd('peak', '-c', s12))
    rows = [{kk: v for kk, v in r_.items() if kk not in ('controls', 'borrowed', 'zero_after')} for r_ in rows]
    return E.Envelope(led, claim, tier='static',
                      domain=f"{res['span']}: exact block functions and zero proofs (every shot); prices are a "
                             'static model', evidence=api.rel(ev), rerun=rr,
                      tool_sha=api.tool_sha(), circuit_sha=sha, facts=facts, rows=rows, table=res['table'],
                      parts=parts, next_cmds=nxt)


def qhdl_register(subparsers, api):
    import argparse
    p = subparsers.add_parser(
        'census', help='dirty-borrow multi-control chains and provably-|0> Toffoli targets, priced (MBU / clean / '
                       'conditionally clean); any size',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all', action='store_true', help='print every row')
    p.add_argument('-c', '--circuit', required=True, help='.kmb path or sha prefix')
    p.add_argument('--region', help='top-level INFO_TAG region (as in `peak` rows)')
    p.add_argument('--ops', help='A:B op span (B exclusive)')
    p.add_argument('--warm', action='store_true', help='track from op 0 even with --region/--ops (more zero proofs)')
    p.add_argument('--sites', choices=SITES, default='all', help='rows: mbu (target |0> after: proved or sampled), chains, all')
    p.add_argument('--min-saving', type=float, default=0, help='rows with at least this priced saving')
    p.add_argument('--proof', choices=PROOFS, default='auto',
                   help='zero-after proofs: track (whole-span symbolic tracker, strongest, ~1 h on a whole DQI circuit), '
                        'local (structural rule + tracker on each compute..uncompute window), structural (paired '
                        'compute/uncompute with untouched operands), auto (track up to 2M ops, else local)')
    p.add_argument('--sampled', type=int, default=SAMPLED_DEFAULT,
                   help='DQI: simulate this many valid syndromes to check paired chain uncomputes no proof covered '
                        f'(default {SAMPLED_DEFAULT}; 0 = off; about 20 s on a whole DQI circuit)')
    p.set_defaults(fn=lambda a: v_census(api, a))
