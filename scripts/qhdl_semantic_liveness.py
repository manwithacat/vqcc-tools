#!/usr/bin/env python3
"""Semantic liveness: when is a scratch wire |0> on every legal x?

The lifetime allocator reasons syntactically (a wire is live from a touch after a reset to the next reset). A wire
that has been uncomputed to 0 by CX/CCX, but is reset only later, is still counted live. In kickmix every shot is
one computational-basis trajectory, so a wire whose value is 0 on every legal x and in every measurement branch is
exactly |0>; it can be handed to another value at no cost.

This tool runs the exact bitset simulator (qhdl_bitsim) op by op and records, for every touch, whether the touched
wire is 0 on all rows before and after. A semantic interval runs from a touch that makes the wire nonzero to the
touch that makes it zero again. Touches that see the wire 0 before and after ('null touches': controls on a zero
wire, CX a w with a = 0, an HMR of a zero wire) need no live wire.

The profile is computed in two measurement branches (all outcomes 0, and seeded random outcomes); a wire counts as
zero only if it is zero in both. Output: syntactic vs semantic peak, the semantic peak runs, and the list of
'early zero' points (op, wire) where a reset could be inserted for free.

Usage: python qhdl_semantic_liveness.py CIRCUIT.kmx OUT.json
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
import qhdl_idle_wires as I  # noqa: E402


def zero_trace(ops, xq, nq, outcome):
    """For each op i and each operand wire w: (zero_before, zero_after)."""
    sim = B.BitSim(ops, num_qubits=nq, x_qubits=xq, outcome=outcome)
    nz = np.array([sim.q[w].any() for w in range(nq)])
    out = []
    for i, o in enumerate(ops):
        sim.run(stop=i + 1)
        if not o.q:
            out.append(())
            continue
        rec = []
        for w in o.q:
            before = not nz[w]
            now = bool(sim.q[w].any())
            nz[w] = now
            rec.append((before, not now))
        out.append(tuple(rec))
    return out


def semantic_profile(ops, xq, nq, traces):
    regs = set(xq)
    n = len(ops)
    delta = np.zeros(n + 1, np.int64)
    live_since = {}
    null_touches = 0
    early_zero = []  # (op, wire): op leaves w zero on all rows/branches while the syntactic interval continues
    intervals = []
    for i, o in enumerate(ops):
        for k, w in enumerate(o.q):
            if w in regs:
                continue
            zb = all(t[i][k][0] for t in traces)
            za = all(t[i][k][1] for t in traces)
            if zb and za:
                null_touches += 1
                continue
            if w not in live_since:
                live_since[w] = i
            if za:
                s = live_since.pop(w)
                intervals.append((w, s, i))
                if o.code not in (B.C_R, B.C_HMR):
                    early_zero.append((i, w))
    for w, s in live_since.items():
        intervals.append((w, s, n - 1))
    for w, s, e in intervals:
        delta[s] += 1
        delta[e + 1] -= 1
    prof = np.cumsum(delta)[:n] + len(regs)
    return prof, intervals, early_zero, null_touches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmx')
    ap.add_argument('out')
    a = ap.parse_args()
    t0 = time.time()
    ops, header, xq, nq = B.load(a.kmx)
    depth, touches, intervals, sprof = I.structure(ops, xq, nq)
    traces = [zero_trace(ops, xq, nq, 'zero'), zero_trace(ops, xq, nq, 31)]
    prof, sint, early, nulls = semantic_profile(ops, xq, nq, traces)
    smx, sruns = I.peak_runs(prof)
    # early zeros that are followed (on that wire) by a long syntactic tail: how many ops of dead tail each frees
    rep = dict(circuit=a.kmx, num_qubits=nq, syntactic_peak=int(sprof.max()),
               syntactic_ops_at_peak=int((sprof == sprof.max()).sum()), semantic_peak=int(smx),
               semantic_ops_at_peak=int((prof == smx).sum()), semantic_runs=len(sruns),
               semantic_ops_at=dict((int(v), int((prof == v).sum())) for v in range(smx - 3, smx + 1)),
               null_touches=nulls, early_zero_points=len(early), seconds=round(time.time() - t0, 1))
    Path(a.out).write_text(json.dumps(dict(report=rep, early_zero=early, semantic_runs=sruns,
                                           semantic_intervals=sint), default=int))
    np.save(str(Path(a.out).with_suffix('.profile.npy')), prof)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
