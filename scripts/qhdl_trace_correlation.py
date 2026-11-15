#!/usr/bin/env python3
"""Trace-correlation pass (principle credited to hengck23, Kaggle discussion
753403, comment 3546385: treat the wire traces as data and share wires with the same trace pattern).

For every width-peak run, take the exact value bitsets (all 3,133,452 legal x, zero measurement branch, from
qhdl_bitsim) of the wires live just before the run's first op and report:
  * width W (registers included, register gaps honoured), evaluated just after the run's first op;
  * constant wires, and pairs equal or complementary on every x;
  * the affine rank: GF(2) rank of {1} + live bitsets, minus one. W - affine_rank live wires are affine functions of
    the others there (recomputable by CX alone).
Redundancy is necessary, not sufficient, for a free qubit: the wire must also be idle across the run, and phase uses
still need the certified rewrite (qhdl_park_rewrite) and the verifier. Quadratic dependencies are priced per run by
qhdl_width_census.

Usage: python qhdl_trace_correlation.py CIRCUIT.kmx OUT.json [--level L] [--chunk 24]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_bitsim as B  # noqa: E402
import qhdl_idle_wires as I  # noqa: E402


def gf2_rank(rows):
    """rows: (m, W) uint64 packed bit-vectors. Exact GF(2) rank."""
    A = rows.copy()
    m = A.shape[0]
    rank = 0
    for i in range(m):
        nz = np.flatnonzero(A[i])
        if len(nz) == 0:
            continue
        wi = nz[0]
        word = int(A[i, wi])
        bit = (word & -word).bit_length() - 1
        mask = np.uint64(1 << bit)
        rank += 1
        sel = np.flatnonzero((A[i + 1:, wi] & mask) != 0) + i + 1
        if len(sel):
            A[sel] ^= A[i]
    return rank


def analyse_state(qwords, live, valid):
    vecs = [qwords[w] & valid for w in live]
    const = [w for w, v in zip(live, vecs) if not v.any() or np.array_equal(v, valid)]
    seen = {}
    pairs = []
    for w, v in zip(live, vecs):
        if w in const:
            continue
        k1 = v.tobytes()
        k2 = (v ^ valid).tobytes()
        if k1 in seen:
            pairs.append((seen[k1], w, 'equal'))
        elif k2 in seen:
            pairs.append((seen[k2], w, 'complement'))
        else:
            seen[k1] = w
    M = np.stack([valid] + vecs)
    r = gf2_rank(M)
    return dict(width=len(live), affine_rank=r - 1, redundancy=len(live) - (r - 1), constants=const, pairs=pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmx')
    ap.add_argument('out')
    ap.add_argument('--level', type=int, default=None)
    ap.add_argument('--chunk', type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    ops, header, xq, nq = B.load(a.kmx)
    depth, touches, intervals, prof, regiv = I.structure(ops, xq, nq, with_regs=True)
    mx, runs = I.peak_runs(prof, a.level)
    allv = intervals + regiv
    starts = [r[0] for r in runs]
    live_at = {}
    # state just AFTER the run's first op (a wire born at that op then holds its value); wires whose live interval
    # ends at that op (reset by it) are 0 afterwards and are left out of the rank (still counted in the width)
    ending = {}
    for o in starts:
        cover = [(w, s, e) for (w, s, e, _) in allv if s <= o <= e]
        live_at[o + 1] = sorted({w for (w, s, e) in cover if e > o or s == o})
        ending[o + 1] = len(cover) - len(live_at[o + 1])
    starts = [o + 1 for o in starts]
    res = {}
    for i0 in range(0, len(starts), a.chunk):
        chunk = starts[i0:i0 + a.chunk]

        def cb(i, sim):
            res[i] = analyse_state(sim.q, live_at[i], sim.valid)
        sim = B.BitSim(ops, num_qubits=nq, x_qubits=xq, outcome='zero')
        sim.run(stop=max(chunk) + 1, snaps=chunk, on_snap=cb)
    per_run = [dict(run=k, start=int(r[0]), end=int(r[1]), ops=int(r[1] - r[0] + 1), ending_at_first_op=ending[r[0] + 1],
                    **res[r[0] + 1])
               for k, r in enumerate(runs)]
    red = Counter(p['redundancy'] for p in per_run)
    rep = dict(circuit=a.kmx, peak=mx, runs=len(runs), peak_ops=int(sum(p['ops'] for p in per_run)),
               redundancy_histogram=dict(sorted(red.items())),
               runs_with_redundancy=sum(1 for p in per_run if p['redundancy'] > 0),
               peak_ops_with_redundancy=sum(p['ops'] for p in per_run if p['redundancy'] > 0),
               runs_with_equal_or_complement_pair=sum(1 for p in per_run if p['pairs']),
               runs_with_constant_live_wire=sum(1 for p in per_run if p['constants']),
               width_check=sorted({p['width'] for p in per_run}), seconds=round(time.time() - t0, 1))
    Path(a.out).write_text(json.dumps(dict(report=rep, per_run=per_run), indent=1, default=int))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
