#!/usr/bin/env python3
"""Width-blocker census: can the peak width drop by one, and at what Toffoli price?

For every peak run (maximal stretch of ops at the peak width) list the wires live across it and price the
cheapest way to take one of them out for the run:

  dead-tail  the wire's next touch is an unconditional HMR/R: move the erase up to its last real use, 0 T;
  park       erase by HMR right after the last touch, recompute just after the run as a cheap function g of the
             wires present there, then `Z w if m` (see qhdl_cheap_function): zero/one/affine 0 T, quadratic
             rank/2 T;
  blocked    every live wire is either touched inside the run (genuine reads/writes), or holds information that
             is not a function of the other wires at the recompute point ('info'), or only a cubic-or-worse one.

Runs are covered greedily in op order, preferring to extend the park chosen for the previous run (one erase and
one recompute then cover a whole block of runs). The census total is the sum of recompute costs of the distinct
parks, an upper bound on what 'peak - 1' costs with these moves (static model / local analysis tier).
Exact re-checks on all legal x are left to qhdl_park_rewrite (the certified rewrite step).

Usage: python qhdl_width_census.py CIRCUIT.kmx OUT.json [--sample 4000] [--level L]
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
import qhdl_cheap_function as C  # noqa: E402
import qhdl_idle_wires as I  # noqa: E402


def sample_rows(n, seed=1):
    rng = np.random.default_rng(seed)
    extra = [1, 2, 3, 161, B.MAX_X, B.MAX_X - 1] + [1 << k for k in range(22)] + [(1 << k) - 1 for k in range(1, 22)]
    return np.unique(np.concatenate([rng.integers(1, B.MAX_X + 1, n), extra]))


def run_census(path, sample=4000, level=None, quad=True, exact_info=False, registers=True, topk_wide=3):
    t0 = time.time()
    ops, header, xq, nq = B.load(path)
    depth, touches, intervals, prof = I.structure(ops, xq, nq)
    mx, runs = I.peak_runs(prof, level)
    # gaps spanning each run (with margins); generalised to 'level' (ops >= level are the runs to close)
    run_s = np.array([r[0] for r in runs])
    run_e = np.array([r[1] for r in runs])
    peak = prof >= mx
    cands = [[] for _ in runs]
    reg_iv = []
    if registers:
        for w in xq:
            tl = [-1] + [t for t, _ in touches[w]]
            reg_iv.append((w, -1, tl[-1], tl))
    for w, s, e, tl in list(intervals) + reg_iv:
        for a, b in zip(tl[:-1], tl[1:]):
            if b - a < 2:
                continue
            if a >= 0 and w in xq and ops[a].code in (B.C_HMR, B.C_R):
                continue  # register wire already measured out here (borrowed slot)
            lo = int(np.searchsorted(run_s, a + 1))
            hi = int(np.searchsorted(run_e, b - 1, 'right'))
            for k in range(lo, hi):
                if run_s[k] <= a or run_e[k] >= b:
                    continue
                if run_e[k] + 1 >= b and peak[b - 1]:
                    continue
                nxt = ops[b]
                dead = bool(nxt.code in (B.C_HMR, B.C_R) and nxt.cond < 0 and depth[b] == 0 and w not in xq)
                cands[k].append((w, a, b, dead))
    rows = sample_rows(sample)
    n = len(rows)
    pts = set(int(e) + 1 for e in run_e)
    for k in range(len(runs)):
        for (w, a, b, dead) in cands[k]:
            pts.add(a + 1)
    snaps, _ = C.snapshot_values(ops, xq, nq, rows, sorted(pts), 'zero')
    ints = {}

    def iv(p, q):
        key = (p, q)
        if key not in ints:
            ints[key] = C.to_int(snaps[p][q], n)
        return ints[key]

    allones = (1 << n) - 1
    cache = {}

    def price1(w, a, r):
        """Stage 1 on the sample: zero / one / affine / info (sample conflict) / open."""
        key3 = (w, a, r)
        if key3 in cache:
            return cache[key3]
        target = iv(a + 1, w)
        if target == 0:
            res = dict(kind='zero', terms=[], cost=0)
        elif target == allones:
            res = dict(kind='one', terms=[('1',)], cost=0)
        else:
            vals = {q: iv(r, q) for q in range(nq)}
            reps, _ = C.classes(vals, {w})
            feats = [(('1',), allones)] + [((f'q{v}',), vals[v]) for v in reps]
            sol = C.solve(feats, target)
            if sol is not None:
                res = dict(kind='affine', terms=sol, cost=0)
            else:
                key = np.zeros(n, np.uint64)
                for q in range(nq):
                    if q != w:
                        key |= B.unpack_bits(snaps[r][q], n).astype(np.uint64) << np.uint64(q)
                tv = B.unpack_bits(snaps[a + 1][w], n)
                o_ = np.argsort(key, kind='stable')
                kk, tt = key[o_], tv[o_]
                conflict = bool(((kk[1:] == kk[:-1]) & (tt[1:] != tt[:-1])).any())
                res = dict(kind='info' if conflict else 'open', terms=None, cost=None)
        cache[key3] = res
        return res

    # stage 1 per run
    tried = []
    for k, (rs, re_) in enumerate(runs):
        r = re_ + 1
        opts = []
        for (w, a, b, dead) in sorted(cands[k], key=lambda c: (0 if c[3] else 1, -(c[2] - c[1]))):
            if dead:
                opts.append(dict(w=w, a=a, b=b, r=b, kind='dead-tail', cost=0, terms=[]))
                break
            rr = min(r, b)
            p = price1(w, a, rr)
            opts.append(dict(w=w, a=a, b=b, r=rr, kind=p['kind'], cost=p['cost'], terms=p['terms']))
            if p['cost'] == 0:
                break
        tried.append(opts)
    # stage 2: exact information test (all legal x, zero branch) for 'open' candidates of unresolved runs
    # wire keys: exact bit positions when nq <= 64; otherwise random 64-bit XOR hashing ('determined' stays
    # exact; an 'info' verdict could be a hash false positive with probability ~1e-6, i.e. conservative)
    if nq <= 64:
        HKEY = np.array([1 << q for q in range(nq)], dtype=np.uint64)
    else:
        HKEY = np.random.default_rng(12345).integers(0, 1 << 64, nq, dtype=np.uint64, endpoint=False)
    unresolved = [k for k in range(len(runs)) if not any(o['cost'] == 0 for o in tried[k])]
    todo = sorted({(o['w'], o['a'], o['r']) for k in unresolved for o in tried[k] if o['kind'] == 'open'},
                  key=lambda t: t[2])
    verdict = {}
    if exact_info and todo:
        rpts = sorted({t[2] for t in todo})
        for i0 in range(0, len(rpts), 24):
            chunk = set(rpts[i0:i0 + 24])
            keys = {}

            def cb(i, sim):
                kk = np.zeros(sim.n, np.uint64)
                for q in range(nq):
                    kk ^= B.unpack_bits(sim.q[q], sim.n).astype(np.uint64) * HKEY[q]
                keys[i] = np.unique(kk)
            sim = B.BitSim(ops, num_qubits=nq, x_qubits=xq, outcome='zero')
            sim.run(stop=max(chunk) + 1, snaps=sorted(chunk), on_snap=cb)
            for (w, a_, r_) in todo:
                if r_ not in chunk:
                    continue
                K = keys[r_]
                flip = K ^ HKEY[w]
                pos = np.searchsorted(K, flip)
                pos[pos >= len(K)] = 0
                hit = bool((K[pos] == flip).any())
                verdict[(w, a_, r_)] = 'info' if hit else 'determined'
            del keys
    # stage 3: quadratic search (sample) for exactly determined candidates; exact re-check of hits
    for k in unresolved:
        for o in tried[k]:
            v = verdict.get((o['w'], o['a'], o['r']))
            if v is None:
                continue
            o['exact_info'] = v
            if v == 'info':
                o['kind'] = 'info'
            elif quad:
                vals = {q: iv(o['r'], q) for q in range(nq)}
                f = C.analyse_sample(iv(o['a'] + 1, o['w']), vals, o['w'], n)
                o['kind'] = 'determined-' + f['kind'] if f['kind'] in ('none', 'unpriced') else f['kind']
                o['cost'], o['terms'] = f['cost'], f['terms']
    # stage 3b: wide circuits. Quadratic search with a larger sample for the top-K determined candidates per run
    big = []
    for k in unresolved:
        dets = [o for o in tried[k] if o['kind'] == 'determined-unpriced']
        dets.sort(key=lambda o: -(o['b'] - o['a']))
        big.extend(dets[:topk_wide])
    if quad and big:
        nfeat = 1 + nq + nq * (nq - 1) // 2
        rows2 = sample_rows(int(2.3 * nfeat), seed=2)
        n2 = len(rows2)
        pts2 = sorted({p for o in big for p in (o['a'] + 1, o['r'])})
        snaps2, _ = C.snapshot_values(ops, xq, nq, rows2, pts2, 'zero')
        done = {}
        for o in big:
            key3 = (o['w'], o['a'], o['r'])
            if key3 not in done:
                vals = {q: C.to_int(snaps2[o['r']][q], n2) for q in range(nq)}
                done[key3] = C.analyse_sample(C.to_int(snaps2[o['a'] + 1][o['w']], n2), vals, o['w'], n2)
            f = done[key3]
            o['kind'] = 'determined-' + f['kind'] if f['kind'] in ('none', 'unpriced') else f['kind']
            o['cost'], o['terms'] = f['cost'], f['terms']
    # exact re-check of every priced quadratic option on all legal x (zero branch); failures are demoted
    qopts = [o for k in unresolved for o in tried[k] if o['kind'] == 'quadratic']
    if qopts:
        pts3 = sorted({p for o in qopts for p in (o['a'] + 1, o['r'])})
        for i0 in range(0, len(pts3), 40):
            chunk = pts3[i0:i0 + 40]
            full, sim = C.snapshot_values(ops, xq, nq, None, chunk, 'zero')
            for o in qopts:
                if o['a'] + 1 in full and o['r'] in full and 'exact' not in o:
                    got = C.eval_terms([tuple(t) for t in o['terms']], full[o['r']], sim.W) & sim.valid
                    o['exact'] = bool(np.array_equal(full[o['a'] + 1][o['w']] & sim.valid, got))
            del full
        for o in qopts:
            if not o.get('exact', False):
                o['kind'], o['cost'] = 'quadratic-sample-only', None
    per_run = []
    chosen = []
    prev = None
    for k, (rs, re_) in enumerate(runs):
        best = None
        for o in tried[k]:
            if o['cost'] is not None and (best is None or o['cost'] < best['cost']):
                best = o
        per_run.append(dict(run=k, start=int(rs), end=int(re_), ops=int(re_ - rs + 1), n_candidates=len(cands[k]),
                            best=best, tried=[{kk: o.get(kk) for kk in ('w', 'a', 'b', 'r', 'kind', 'cost', 'exact')}
                                              for o in tried[k]],
                            all_candidates_info=bool(best is None and all(o['kind'] == 'info' for o in tried[k]))))
        if best is not None:
            if prev is not None and (prev['w'], prev['a']) == (best['w'], best['a']):
                if best['r'] >= prev['r']:
                    prev.update(r=best['r'], terms=best['terms'], cost=best['cost'], kind=best['kind'])
                prev['runs'].append(k)
            else:
                prev = dict(w=best['w'], a=best['a'], b=best['b'], r=best['r'], kind=best['kind'], cost=best['cost'],
                            terms=best['terms'], runs=[k])
                chosen.append(prev)
        else:
            prev = None
    blocked = [p['run'] for p in per_run if p['best'] is None]
    total = sum(c['cost'] for c in chosen)
    rep = dict(circuit=path, peak=mx, runs=len(runs), peak_ops=int(peak.sum()), sample_rows=n,
               runs_priced=len(runs) - len(blocked), runs_blocked=len(blocked), blocked_runs=blocked,
               blocked_runs_all_info=sum(1 for k in blocked if per_run[k].get('all_candidates_info')),
               blocked_peak_ops=int(sum(per_run[k]['ops'] for k in blocked)),
               parks=len(chosen), census_total_T_upper=total if not blocked else None,
               census_total_T_priced_runs=total,
               kinds=dict((kd, sum(1 for c in chosen if c['kind'] == kd)) for kd in sorted({c['kind'] for c in chosen})),
               seconds=round(time.time() - t0, 1))
    return rep, per_run, chosen, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmx')
    ap.add_argument('out')
    ap.add_argument('--sample', type=int, default=4000)
    ap.add_argument('--level', type=int, default=None)
    ap.add_argument('--no-quad', action='store_true')
    ap.add_argument('--exact-info', action='store_true')
    a = ap.parse_args()
    rep, per_run, chosen, runs = run_census(a.kmx, a.sample, a.level, not a.no_quad, a.exact_info)
    Path(a.out).write_text(json.dumps(dict(report=rep, parks=chosen, per_run=per_run, runs=runs), indent=1,
                                      default=str))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
