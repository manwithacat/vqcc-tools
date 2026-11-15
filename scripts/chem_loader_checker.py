#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Exhaustive classical checker for a chemistry ANGLE_k loader (.kmb/.kmx).

Simulates the X/CX/CCX/R/HMR/CZ network over all 2**n addresses at once (each wire = a big-int bitset over addresses),
from several random initial word contents. Checks, on every used (valid) address:
  1 selection register restored;  2 word registers end = initial XOR delta(address) on all bits (or all active bits
  under an optional don't-care mask);  3 every non-selection, non-word wire is zero at the end;
  4 phase: for each HMR on wire w (outcome m) the phase (-1)^(m*value) must be cancelled by the CZs conditioned on that
  outcome, i.e. value(w) XOR sum(a_i AND b_i) == 0 at each CZ time (exact, every outcome).
Usage (kickmix interpreter):  python -I chem_loader_checker.py LOADER.kmb WORDS.npy K [--corrupt]
"""
import argparse, random, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1] / "tools/dfthc-refit"
sys.path.insert(0, str(HERE))


def simulate(regs, ops, nsel, init_words, nbits=None):
    n = 1 << nsel
    full = (1 << n) - 1
    val = {}
    for i, q in enumerate(regs["selection"]):
        m = 0
        for a in range(n):
            if (a >> i) & 1:
                m |= 1 << a
        val[q] = m
    for (j, b), bits in init_words.items():
        val[regs[f"word_{j}"][b]] = bits
    g = lambda q: val.get(q, 0)
    pend = {}          # measurement bit name -> list of (value bitset)
    phase_bad = 0
    for t in ops:
        o = t[0]
        if o == "X":
            q = int(t[1][1:]); val[q] = g(q) ^ full
        elif o == "CX":
            a, b = int(t[1][1:]), int(t[2][1:]); val[b] = g(b) ^ g(a)
        elif o == "CCX":
            a, b, c = (int(x[1:]) for x in t[1:4]); val[c] = g(c) ^ (g(a) & g(b))
        elif o == "R":
            val[int(t[1][1:])] = 0
        elif o == "HMR":
            q = int(t[1][1:]); phase_bad |= pend.get(t[2], 0); pend[t[2]] = g(q); val[q] = 0   # residual phase poly = value of measured wire
        elif o == "CZ":
            a, b = int(t[1][1:]), int(t[2][1:])
            if len(t) >= 5 and t[3] == "if":
                pend[t[4]] = pend.get(t[4], 0) ^ (g(a) & g(b))
            else:
                phase_bad |= g(a) & g(b)                       # unconditional CZ is a residual phase
        else:
            raise ValueError(f"unsupported op {o}")
    for k, v in pend.items():
        phase_bad |= v
    return val, phase_bad


def check(km, path, prev, now, nsel=11, dc=None):
    """prev, now: (A, J) stored 8-bit table words of slot k-1 (zeros for k=0) and slot k, for valid addresses 0..A-1.
    The register enters holding prev[a] (the pi-gauge top bit included, as the previous loader left it) and must leave
    holding now[a]. dc: optional (A, J) bit mask of don't-care bits (1 = free)."""
    import recover as R
    circ, regs, ops = R.read_loader(km, path)
    J = len([r for r in regs if r.startswith("word_")])
    n, A = 1 << nsel, len(now)
    valid = (1 << A) - 1
    res = dict(ok=True, fails=[], toffoli=int(circ.operation_statistics().num_toffoli_like), qubits=int(circ.num_qubits))
    init = {}
    for j in range(J):
        for b in range(8):
            init[(j, b)] = sum(((int(prev[a, j]) >> b) & 1) << a for a in range(A))
    val, pb = simulate(regs, ops, nsel, init)
    if pb & valid:
        res["ok"] = False; res["fails"].append(f"residual phase on {bin(pb & valid).count('1')} valid addresses")
    for i, q in enumerate(regs["selection"]):
        m = sum(1 << a for a in range(A) if (a >> i) & 1)
        if (val.get(q, 0) ^ m) & valid: res["ok"] = False; res["fails"].append(f"selection bit {i} not restored")
    bad = 0
    for j in range(J):
        for b in range(8):
            got = val.get(regs[f"word_{j}"][b], 0)
            for a in range(A):
                if dc is not None and (int(dc[a, j]) >> b) & 1: continue
                if ((got >> a) & 1) != ((int(now[a, j]) >> b) & 1): bad += 1
    if bad: res["ok"] = False; res["fails"].append(f"{bad} word-bit errors on {A} valid addresses x {J*8} bits")
    keep = {q for r in regs.values() for q in r}
    dirty = [q for q, v in val.items() if q not in keep and (v & valid)]
    if dirty: res["ok"] = False; res["fails"].append(f"scratch nonzero on {len(dirty)} wires")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kmb"); ap.add_argument("words"); ap.add_argument("k", type=int)
    ap.add_argument("--corrupt", action="store_true", help="flip one word bit of the target table (negative control)")
    a = ap.parse_args()
    import kickmix as km
    w = np.load(a.words)
    now = w[:, 5 * a.k:5 * a.k + 5].astype(np.int64)
    prev = w[:, 5 * (a.k - 1):5 * a.k].astype(np.int64) if a.k else np.zeros_like(now)
    if a.corrupt: now[17, 2] ^= 1 << 3
    print(check(km, a.kmb, prev, now))
