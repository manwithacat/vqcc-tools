#!/usr/bin/env python3
"""Cheap-function test for idle wires.

Question per gap (wire w, last touch a, next touch b): is the value w holds across the gap equal, on every legal
x, to a cheap function of the wires present at a recompute point r (a < r <= b)? If so, w can be parked:

    erase:      HMR w m        right after a (any value; m is a fresh classical bit)
    recompute:  w ^= g(...)    just before r  (affine part: CX; quadratic part: AND gates into w)
    repair:     Z w if m       after the recompute (the HMR flipped the sign by (-1)^(m*w); w is back)

Phases are per-trajectory scalars, so the repair can sit anywhere after w is recomputed. The cost is only the
AND count of g: affine 0; quadratic rank(B)/2 where B is g's alternating bilinear form (the multiplicative
complexity of a quadratic form). Cubic and above are reported as 'none' here.

Search uses a random sample of legal x (deterministic 'zero' measurement branch), solving over GF(2) with Python
big-int bitsets; every hit is then re-checked exactly on all 3,133,452 x, in the zero branch and in two random
measurement branches.

Usage: python qhdl_cheap_function.py CIRCUIT.kmx IDLE.json OUT.json [--top N] [--sample N] [--min-runs K]
       (IDLE.json from qhdl_idle_wires.py). Library: analyse_sample, solve, quad_cost, determined_target.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_bitsim as B  # noqa: E402


def to_int(words, n):
    """Packed uint64 words -> python int (bit r = row r)."""
    return int.from_bytes(words.tobytes(), 'little') & ((1 << n) - 1)


def solve(features, target):
    if len(features) <= 300:
        return solve_small(features, target)
    return solve_np(features, target)


def solve_small(features, target):
    """Big-int elimination; fast when there are few features."""
    basis = {}
    for idx, (_, v) in enumerate(features):
        combo = 1 << idx
        while v:
            h = v.bit_length() - 1
            if h in basis:
                bv, bc = basis[h]
                v ^= bv
                combo ^= bc
            else:
                basis[h] = (v, combo)
                break
    v, combo = target, 0
    while v:
        h = v.bit_length() - 1
        if h not in basis:
            return None
        bv, bc = basis[h]
        v ^= bv
        combo ^= bc
    return [features[i][0] for i in range(len(features)) if (combo >> i) & 1]


def solve_np(features, target):
    """features: list of (name, int over n sample rows). Return names whose XOR == target, or None.
    Packed-numpy Gaussian elimination over GF(2) with combination tracking."""
    m = len(features)
    n = max([target.bit_length()] + [v.bit_length() for _, v in features] + [1])
    W = (n + 63) // 64
    def words(v):
        return np.frombuffer(v.to_bytes(W * 8, 'little'), dtype='<u8')
    A = np.zeros((m + 1, W), np.uint64)
    for i, (_, v) in enumerate(features):
        A[i] = words(v)
    A[m] = words(target)
    Wc = (m + 1 + 63) // 64
    Cm = np.zeros((m + 1, Wc), np.uint64)
    idx = np.arange(m + 1)
    Cm[idx, idx >> 6] = np.uint64(1) << (idx & 63).astype(np.uint64)
    free = np.ones(m + 1, bool)
    free[m] = False  # target row never pivots
    for c in range(n):
        col = (A[:, c >> 6] >> np.uint64(c & 63)) & np.uint64(1)
        has = col.astype(bool)
        cand = np.flatnonzero(has & free)
        if len(cand) == 0:
            continue
        p = cand[0]
        free[p] = False
        others = np.flatnonzero(has)
        others = others[others != p]
        if len(others):
            A[others] ^= A[p]
            Cm[others] ^= Cm[p]
    if A[m].any():
        return None
    combo = Cm[m]
    out = []
    for i in range(m):
        if (int(combo[i >> 6]) >> (i & 63)) & 1:
            out.append(features[i][0])
    return out


def gf2_rank_matrix(rows):
    basis = {}
    r = 0
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                r += 1
                break
    return r


def quad_cost(terms):
    """terms: names like ('1',), ('a',), ('a','b'). AND cost = rank of alternating matrix / 2."""
    pairs = [t for t in terms if len(t) == 2]
    if not pairs:
        return 0
    vs = sorted({v for p in pairs for v in p})
    ix = {v: i for i, v in enumerate(vs)}
    rows = [0] * len(vs)
    for a, b in pairs:
        rows[ix[a]] ^= 1 << ix[b]
        rows[ix[b]] ^= 1 << ix[a]
    return gf2_rank_matrix(rows) // 2


def classes(vals, exclude):
    """Collapse wires equal/complementary on the sample; drop constants. vals: {wire: int}. Returns reps, notes."""
    reps, seen, notes = [], {}, {}
    full = None
    for w, v in sorted(vals.items()):
        if w in exclude:
            continue
        if full is None:
            full = (1 << max(1, max(x.bit_length() for x in vals.values()))) - 1
        if v == 0:
            notes[w] = 'zero'
            continue
        key = v
        if key in seen:
            notes[w] = f'=q{seen[key]}'
            continue
        reps.append(w)
        seen[key] = w
    return reps, notes


def analyse_sample(target, vals, w, n):
    """target int, vals {wire:int} at r. Return dict of findings on the sample."""
    allones = (1 << n) - 1
    if target == 0:
        return dict(kind='zero', terms=[], cost=0)
    if target == allones:
        return dict(kind='one', terms=[('1',)], cost=0)
    reps, notes = classes(vals, {w})
    feats = [(('1',), allones)] + [((f'q{v}',), vals[v]) for v in reps]
    sol = solve(feats, target)
    if sol is not None:
        return dict(kind='affine', terms=sol, cost=0)
    nfeat = len(feats) + len(reps) * (len(reps) - 1) // 2
    if 2 * nfeat > n:
        return dict(kind='unpriced', terms=None, cost=None, note=f'{nfeat} quadratic features vs {n} sample rows')
    quad = list(feats)
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            quad.append(((f'q{a}', f'q{b}'), vals[a] & vals[b]))
    sol = solve(quad, target)
    if sol is not None:
        sol2 = reduce_quadratic(sol, quad, target) if len(reps) <= 64 else sol
        return dict(kind='quadratic', terms=sol2, cost=quad_cost(sol2), cost_first=quad_cost(sol))
    return dict(kind='none', terms=None, cost=None)


def reduce_quadratic(sol, quad, target):
    """Greedy: try to find a representation with fewer variables by re-solving on restricted supports."""
    best = sol
    best_cost = quad_cost(sol)
    vars_ = sorted({v for t in sol for v in t if v != '1'})
    improved = True
    rounds = 0
    while improved and best_cost > 0 and rounds < 6:
        rounds += 1
        improved = False
        for drop in vars_:
            sub = [f for f in quad if drop not in f[0]]
            s = solve(sub, target)
            if s is not None and quad_cost(s) <= best_cost:
                best, best_cost = s, quad_cost(s)
                vars_ = sorted({v for t in s for v in t if v != '1'})
                improved = True
                break
    return best


def eval_terms(terms, qwords, W):
    acc = np.zeros(W, np.uint64)
    for t in terms:
        if t == ('1',):
            acc ^= B.ALL1
        elif len(t) == 1:
            acc ^= qwords[int(t[0][1:])]
        else:
            acc ^= qwords[int(t[0][1:])] & qwords[int(t[1][1:])]
    return acc


def snapshot_values(ops, xq, nq, rows, points, outcome):
    snaps = {}

    def cb(i, s):
        snaps[i] = s.q.copy()
    sim = B.BitSim(ops, rows=rows, num_qubits=nq, x_qubits=xq, outcome=outcome)
    sim.run(stop=max(points) + 1, snaps=sorted(set(points)), on_snap=cb)
    return snaps, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmx')
    ap.add_argument('idle_json')
    ap.add_argument('out')
    ap.add_argument('--top', type=int, default=60)
    ap.add_argument('--sample', type=int, default=6000)
    ap.add_argument('--min-runs', type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    ops, header, xq, nq = B.load(a.kmx)
    data = json.loads(Path(a.idle_json).read_text())
    gaps = [g for g in data['gaps'] if len(g['runs_closed']) >= a.min_runs][:a.top]
    runs = data['runs']
    rng = np.random.default_rng(1)
    rows = np.unique(np.concatenate([rng.integers(1, B.MAX_X + 1, a.sample),
                                     [1, 2, 3, 161, B.MAX_X, B.MAX_X - 1] + [1 << k for k in range(22) if (1 << k) <= B.MAX_X]]))
    pts = set()
    for g in gaps:
        last_run_end = max(runs[k][1] for k in g['runs_closed'])
        first_run_start = min(runs[k][0] for k in g['runs_closed'])
        g['r_points'] = sorted({g['b'], last_run_end + 1})
        g['value_point'] = g['a'] + 1
        pts.update(g['r_points'])
        pts.add(g['value_point'])
        g['erase_point'] = g['a'] + 1
        g['first_run'] = first_run_start
    snaps, _ = snapshot_values(ops, xq, nq, rows, pts, 'zero')
    n = len(rows)
    results = []
    for g in gaps:
        w = g['wire']
        target = to_int(snaps[g['value_point']][w], n)
        best = None
        per_r = []
        for r in g['r_points']:
            vals = {q: to_int(snaps[r][q], n) for q in range(nq)}
            f = analyse_sample(target, vals, w, n)
            per_r.append(dict(r=r, **{k: v for k, v in f.items()}))
            if f['cost'] is not None and (best is None or f['cost'] < best['cost']):
                best = dict(r=r, **f)
        results.append(dict(gap={k: g[k] for k in ('wire', 'a', 'b', 'idle_ops', 'runs_closed', 'peak_ops_closed',
                                                  'next_kind', 'dead_tail', 'last_touch', 'next_touch')},
                            per_r=per_r, best=best))
    # exact re-check of hits on all legal x, zero branch and two random branches
    hits = [r for r in results if r['best'] is not None]
    for oc in ('zero', 101, 202):
        pts = set()
        for h in hits:
            pts.add(h['gap']['a'] + 1)
            pts.add(h['best']['r'])
        if not pts:
            break
        full, sim = snapshot_values(ops, xq, nq, None, pts, oc)
        for h in hits:
            w = h['gap']['wire']
            tgt = full[h['gap']['a'] + 1][w]
            got = eval_terms(h['best']['terms'], full[h['best']['r']], sim.W) & sim.valid
            ok = bool(np.array_equal(tgt & sim.valid, got))
            h.setdefault('exact', {})[str(oc)] = ok
        del full
    for h in hits:
        h['exact_all'] = all(h['exact'].values())
    Path(a.out).write_text(json.dumps(dict(circuit=a.kmx, sample_rows=n, seconds=round(time.time() - t0, 1),
                                           results=results), indent=1, default=str))
    for r in results:
        g, b = r['gap'], r['best']
        desc = 'none' if b is None else f"{b['kind']} cost {b['cost']} at r={b['r']} terms={b['terms']} exact={r.get('exact_all')}"
        print(f"q{g['wire']} [{g['a']},{g['b']}] runs {len(g['runs_closed'])} peakops {g['peak_ops_closed']} next {g['next_kind']}: {desc}")


if __name__ == '__main__':
    main()


def determined(qwords, w, n, exclude=()):
    """Exact: is wire w a function (of any degree) of the other wires on all rows? Returns (bool, #conflicting keys).
    qwords: (nq, W) snapshot of the state at the recompute point; target: w's value bitset (same rows)."""
    return determined_target(qwords, qwords[w], n, exclude=set(exclude) | {w})


def determined_target(qwords, target, n, exclude=()):
    nq = qwords.shape[0]
    key = np.zeros(n, np.uint64)
    for q in range(nq):
        if q in exclude:
            continue
        key |= B.unpack_bits(qwords[q], n).astype(np.uint64) << np.uint64(q)
    t = B.unpack_bits(target, n)
    order = np.argsort(key, kind='stable')
    k, tv = key[order], t[order]
    same = k[1:] == k[:-1]
    bad = same & (tv[1:] != tv[:-1])
    return (not bad.any()), int(np.unique(k[1:][bad]).size)
