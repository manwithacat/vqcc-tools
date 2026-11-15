#!/usr/bin/env python3
"""Dirty-accumulator fan-out sites that a clean helper turns from 2 Toffolis into 1.

    python3 scripts/qhdl_dirty_fanout.py CIRCUIT.kmb [...] [--json OUT]

Identity: two CCX(a, b -> w) at c1 < c2 with, inside (c1, c2), w touched only as a CX control, a and b never written,
nothing touching a, b, w conditioned (flag or PUSH/POP block), both CCX unconditioned. Every CX(w -> X) in the window
XORs w ^ ab into X. Replace: drop both CCX; h = ab on a wire unallocated over [c1, c2]; CX(h -> X) next to each
CX(w -> X); at c2 HMR h with CZ(a, b) if the outcome is 1. Saves one Toffoli per pair, same width (h was free).
Free wires: live intervals as in qhdl_measure_first (first touch after a zeroing .. next zeroing). Tier: static.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402

CCX, CX = 0x31, 0x21
WRITE_SLOT = {0x31: 2, 0x21: 1}            # CCX target, CX target
ALWAYS_WRITES = {0x11, 0x12, 0x50, 0x22}   # X, R, HMR, SWAP write every operand
CSWAP = 0x32


def scan(path) -> dict:
    t0 = time.time()
    k = K.read(str(path))
    T = k.op_types
    t = T & np.uint8(0x7F)
    n = len(t)
    cond = (((T & np.uint8(0x80)) != 0) & (T != np.uint8(K.PUSH_CONDITION))) | (K.condition_depth(T) > 0)
    ops, qs, (i3, i2, i1) = K.op_qubits(k)
    n3, n2, n1 = len(i3), len(i2), len(i1)
    slot = np.r_[np.zeros(n3, np.int8), np.ones(n3, np.int8), np.full(n3, 2, np.int8),
                 np.zeros(n2, np.int8), np.ones(n2, np.int8), np.zeros(n1, np.int8)]
    code = t[ops]
    # per operand role: 'w' = writes the wire, 'c' = CX control (allowed for w), 'r' = other read
    writes = np.isin(code, list(ALWAYS_WRITES)) | ((code == CCX) & (slot == 2)) | ((code == CX) & (slot == 1)) | \
             ((code == CSWAP) & (slot > 0))
    cxctl = (code == CX) & (slot == 0)
    bad = cond[ops]
    order = np.lexsort((ops, qs))
    q_s, op_s = qs[order], ops[order]
    w_s, c_s, b_s = writes[order], cxctl[order], bad[order]
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    ends = np.r_[starts[1:], len(q_s)]
    wire = {int(q_s[s]): (s, e) for s, e in zip(starts, ends)}

    def window_ok_w(w, c1, c2):  # w touched only as CX control strictly inside (c1, c2), none conditioned
        s, e = wire[w]
        o = op_s[s:e]
        lo, hi = np.searchsorted(o, c1, 'right'), np.searchsorted(o, c2, 'left')
        sl = slice(s + lo, s + hi)
        return bool(c_s[sl].all() and not b_s[sl].any()), int(hi - lo)

    def window_ok_ab(q, c1, c2):  # never written, never conditioned inside (c1, c2)
        s, e = wire[q]
        o = op_s[s:e]
        lo, hi = np.searchsorted(o, c1, 'right'), np.searchsorted(o, c2, 'left')
        sl = slice(s + lo, s + hi)
        return not (w_s[sl].any() or b_s[sl].any())

    ccx = np.flatnonzero((t == CCX) & ~cond)
    q3 = {int(i): (int(a), int(b), int(c)) for i, a, b, c in
          zip(i3, k.stream('qqq0'), k.stream('qqq1'), k.stream('qqq2'))}
    # Episodes (generalises the pair case): consecutive unconditioned CCXs on the same target w whose gaps touch w only
    # as a CX control. Track the XOR-set of products held in w; each return to the empty set closes a segment whose last
    # CCX can be replaced by a measurement erase of a clean helper (h = that last product), if every control is unwritten
    # over the segment and a clean wire spans it.
    by_w = {}
    for i in ccx.tolist():
        by_w.setdefault(q3[i][2], []).append(i)
    sites = []
    for w, L in by_w.items():
        seg_start, held, ctrls = None, set(), set()
        prev = None
        for i in L:
            a, b, _ = q3[i]
            if prev is not None and held:
                okw, _n = window_ok_w(w, prev, i)
                if not okw:            # chain broken while products are held: abandon this segment
                    seg_start, held, ctrls = None, set(), set()
            if not held:
                seg_start, ctrls = i, set()
            p = (min(a, b), max(a, b))
            held ^= {p}
            ctrls |= {a, b}
            prev = i
            if not held:               # back to empty: candidate site (seg_start .. i)
                if all(window_ok_ab(c, seg_start, i) for c in ctrls):
                    nc = len([j for j in L if seg_start <= j <= i])
                    okw, ncx = window_ok_w(w, seg_start, i)
                    sites.append(dict(c1=seg_start, c2=i, w=w, ccx_in_segment=nc, cx_in_window=ncx,
                                      products=len(ctrls)))
                seg_start, ctrls = None, set()
    # live intervals for free helpers
    zero = np.zeros(len(ops), bool)
    o1 = 3 * n3 + 2 * n2
    zero[o1:o1 + n1] = np.isin(t[i1], [0x50, 0x12])
    z_s = zero[order]
    nw = np.ones(len(q_s), bool)
    nw[1:] = q_s[1:] != q_s[:-1]
    az = np.zeros(len(q_s), bool)
    az[1:] = z_s[:-1] & ~nw[1:]
    bi = np.flatnonzero(nw | az)
    ei = np.r_[bi[1:] - 1, len(q_s) - 1]
    busy = {}
    for b_, e_ in zip(bi, ei):
        busy.setdefault(int(q_s[b_]), []).append((int(op_s[b_]), int(op_s[e_])))
    bstart = {q: np.array([x[0] for x in L]) for q, L in busy.items()}
    bend = {q: np.array([x[1] for x in L]) for q, L in busy.items()}
    nq = k.num_qubits
    taken = {q: [] for q in range(nq)}

    def free(q, c1, c2):
        if q in bstart:  # intervals are disjoint and sorted: the only candidate is the last one starting <= c2
            j = int(np.searchsorted(bstart[q], c2, 'right')) - 1
            if j >= 0 and bend[q][j] >= c1:
                return False
        for b_, e_ in taken[q]:
            if b_ <= c2 and e_ >= c1:
                return False
        return True
    saved = 0
    for s in sorted(sites, key=lambda s: s['c2'] - s['c1']):
        h = next((q for q in range(nq - 1, -1, -1) if free(q, s['c1'], s['c2'])), None)
        s['helper'] = h
        if h is not None:
            taken[h].append((s['c1'], s['c2']))
            saved += 1
    return dict(circuit=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), ops=n,
                pair_sites=len(sites), with_free_helper=saved, toffoli_saving_static=saved,
                sites=sorted(sites, key=lambda s: s['c1']), seconds=round(time.time() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('circuits', nargs='+')
    ap.add_argument('--json', type=Path)
    a = ap.parse_args(argv)
    out = []
    for c in a.circuits:
        r = scan(c)
        out.append(r)
        print(json.dumps({k_: v for k_, v in r.items() if k_ != 'sites'}), flush=True)
    if a.json:
        if a.json.exists():
            raise SystemExit(f'{a.json} exists; outputs are never overwritten')
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
