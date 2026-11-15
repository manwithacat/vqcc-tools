#!/usr/bin/env python3
"""Idle-wire finder for kickmix text circuits (.kmx).

Liveness follows qhdl_lifetime_allocator (syntactic): a scratch wire's interval runs from a touch that follows
an unconditional R/HMR (or its first touch) to the next unconditional R/HMR or its last touch. Width at an op
= 22 register wires + live scratch intervals covering it (inclusive), which reproduces the allocator's profile.

An *idle gap* of a wire is the stretch strictly between two consecutive touches inside one live interval. The
wire's value is constant across the gap. A gap closes a peak run when the run lies strictly inside it and the gap
has a non-peak op on each side of the run (where an erase and a recompute could sit without raising the peak).

Usage: python qhdl_idle_wires.py CIRCUIT.kmx OUT.json [--min-ops N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_bitsim as B  # noqa: E402

ROLE = {B.C_X: 'w', B.C_CX: 'rw', B.C_CCX: 'rrw', B.C_Z: 'p', B.C_CZ: 'pp', B.C_CCZ: 'ppp', B.C_SWAP: 'ss',
        B.C_CSWAP: 'rss', B.C_R: 'z', B.C_HMR: 'z', B.C_ZPOW: 'p'}


def structure(ops, x_qubits, nq, with_regs=False):
    """Per-op condition depth, touches per wire, intervals, width profile."""
    n = len(ops)
    depth = np.zeros(n, np.int32)
    d = 0
    for i, o in enumerate(ops):
        depth[i] = d
        if o.code == B.C_PUSH:
            d += 1
        elif o.code == B.C_POP:
            d = max(0, d - 1)
    touches = [[] for _ in range(nq)]  # (op, role)
    for i, o in enumerate(ops):
        r = ROLE.get(o.code)
        if r is None:
            continue
        for k, qq in enumerate(o.q):
            touches[qq].append((i, r[k]))
    regs = set(x_qubits)
    intervals = []  # (wire, start, end, [touch ops]); register wires live from op 0 and to the end, with gaps
    reg_intervals = []
    for w in range(nq):
        if not touches[w] and w not in regs:
            continue
        isreg = w in regs
        cur = [w, 0, 0, []] if isreg else None
        for (i, r) in touches[w]:
            if cur is None:
                cur = [w, i, i, [i]]
            else:
                cur[2] = i
                cur[3].append(i)
            uncond_reset = (r == 'z' and ops[i].cond < 0 and depth[i] == 0)
            if uncond_reset:
                (reg_intervals if isreg else intervals).append(tuple(cur))
                cur = None
        if cur is not None:
            if isreg:
                cur[2] = n - 1
            (reg_intervals if isreg else intervals).append(tuple(cur))
        elif isreg:
            pass  # register wire left reset at the end would fail the verifier; not expected
    delta = np.zeros(n + 1, np.int64)
    for w, s, e, _ in intervals + reg_intervals:
        delta[s] += 1
        delta[e + 1] -= 1
    prof = np.cumsum(delta)[:n]
    if with_regs:
        return depth, touches, intervals, prof, reg_intervals
    return depth, touches, intervals, prof


def peak_runs(prof, level=None):
    mx = int(prof.max()) if level is None else level
    runs = []
    i, n = 0, len(prof)
    while i < n:
        if prof[i] >= mx:
            j = i
            while j < n and prof[j] >= mx:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return mx, runs


def idle_gaps(ops, intervals, prof, runs, depth, min_ops=1):
    """Gaps (wire, a, b) with a<b touches, b-a-1 >= min_ops, closing >= 1 peak run."""
    run_s = np.array([r[0] for r in runs])
    run_e = np.array([r[1] for r in runs])
    mx = int(prof.max())
    out = []
    for w, s, e, tl in intervals:
        for a, b in zip(tl[:-1], tl[1:]):
            if b - a - 1 < min_ops:
                continue
            # runs strictly inside (a, b) with a non-peak op between a and the run, and between the run and b
            lo = int(np.searchsorted(run_s, a + 1))
            hi = int(np.searchsorted(run_e, b - 1, 'right'))
            closed = []
            for k in range(lo, hi):
                rs, re_ = int(run_s[k]), int(run_e[k])
                if rs <= a or re_ >= b:
                    continue
                if not (prof[a + 1:rs] < mx).any() or not (prof[re_ + 1:b] < mx).any():
                    continue
                closed.append(k)
            if not closed:
                continue
            nxt = ops[b]
            kind = B.KINDS[nxt.code] + ('_if' if (nxt.cond >= 0 or depth[b] > 0) else '')
            out.append(dict(wire=w, a=a, b=b, idle_ops=b - a - 1, runs_closed=closed,
                            peak_ops_closed=int(sum(run_e[k] - run_s[k] + 1 for k in closed)),
                            last_touch=ops[a].text, next_touch=nxt.text, next_kind=kind,
                            dead_tail=bool(nxt.code in (B.C_HMR, B.C_R) and nxt.cond < 0 and depth[b] == 0)))
    out.sort(key=lambda g: -len(g['runs_closed']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmx')
    ap.add_argument('out')
    ap.add_argument('--min-ops', type=int, default=1)
    a = ap.parse_args()
    ops, header, xq, nq = B.load(a.kmx)
    depth, touches, intervals, prof = structure(ops, xq, nq)
    mx, runs = peak_runs(prof)
    gaps = idle_gaps(ops, intervals, prof, runs, depth, a.min_ops)
    covered = set()
    for g in gaps:
        covered.update(g['runs_closed'])
    rep = dict(circuit=a.kmx, num_ops=len(ops), num_qubits=nq, peak=mx, peak_ops=int((prof >= mx).sum()),
               runs=len(runs), runs_with_any_idle_wire=len(covered), gaps=len(gaps),
               ops_at=dict((int(v), int((prof == v).sum())) for v in range(mx - 3, mx + 1)),
               top_gaps=gaps[:40])
    Path(a.out).write_text(json.dumps(dict(report=rep, runs=runs, gaps=gaps), indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'top_gaps'}, indent=1))
    for g in gaps[:15]:
        print(g['wire'], g['a'], g['b'], g['idle_ops'], len(g['runs_closed']), g['peak_ops_closed'], g['next_kind'],
              g['dead_tail'], '|', g['last_touch'], '->', g['next_touch'])


if __name__ == '__main__':
    main()
