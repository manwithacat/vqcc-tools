#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Cold-start feature decomposition of a loader's mixed ANF tensor (high x low x output), for chem_c12_walk.py.

Builds the slot-k delta table from the words (dfthc_refit.loader_delta, Lera's split: 16 low addresses, 7 output bits
per word), pads to whole 16-row chunks with zeros, takes the mixed ANF tensor (recover.anf_rows) and writes the
plane-by-plane GF(2) rank decomposition: for each output plane, a row basis of its (high x low) matrix. Optionally
replaces the table by random bits with the same per-column density (control).
  python3 d7_naive_start.py WORDS.npy K OUT.json [--random SEED]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/dfthc-refit"))
import dfthc_refit as dr, recover as rc


def row_basis(rows):
    """rows: list of ints. Returns (basis, coeffs): rows[h] = XOR of basis[i] over the set bits i of coeffs[h]."""
    piv = {}                       # pivot bit -> (reduced vector, basis-index set)
    basis, coeffs = [], []
    for r in rows:
        v, s = r, 0
        while v:
            pb = v.bit_length() - 1
            if pb not in piv:
                break
            v ^= piv[pb][0]; s ^= piv[pb][1]
        if v:
            i = len(basis); basis.append(r)
            piv[v.bit_length() - 1] = (v, s ^ (1 << i))
            coeffs.append(1 << i)
        else:
            coeffs.append(s)
    return basis, coeffs


def main():
    wp, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    words = np.load(wp)
    delta, cw = dr.loader_delta(words, k, 5)
    A, J = delta.shape
    n = -(-A // 16) * 16
    tab = np.zeros((n, J), np.int64); tab[:A] = delta
    if "--random" in sys.argv:
        rng = np.random.default_rng(int(sys.argv[sys.argv.index("--random") + 1]))
        rnd = np.zeros_like(tab)
        for j in range(J):
            for b in range(7):
                p = ((tab[:A, j] >> b) & 1).mean()
                rnd[:A, j] |= (rng.random(A) < p).astype(np.int64) << b
        tab = rnd
    rows, m, Lm = rc.anf_rows(tab, 8, 4)
    H = len(rows)
    F, T = [], 0
    for o in range(m):
        plane = [sum(((rows[h] >> ((l - 1) * m + o)) & 1) << (l - 1) for l in range(1, Lm + 1)) for h in range(H)]
        basis, coeffs = row_basis(plane)
        for i, v in enumerate(basis):
            b = sum(1 << h for h in range(H) if coeffs[h] >> i & 1)
            F.append((v, b, 1 << o))
    # exact check: features reproduce the tensor
    chk = [0] * H
    for a, b, s in F:
        col = sum(s << ((l - 1) * m) for l in range(1, Lm + 1) if a >> (l - 1) & 1)
        for h in range(H):
            if b >> h & 1:
                chk[h] ^= col
    equal = chk == rows
    prods = sum((sum(1 for f in F if f[2] == 1 << o) + 1) // 2 for o in range(m))
    ranks = [sum(1 for f in F if f[2] == 1 << o) for o in range(m)]
    caches = dr.expected_cache_toffolis(A)
    print(f"k={k} A={A} H={H} Lm={Lm} m={m} coded={cw} features={len(F)} plane-rank-sum={sum(ranks)} "
          f"pair-products={prods} caches={caches} equal={equal}")
    json.dump(dict(m=m, Lm=Lm, H=H, rows=[int(r) for r in rows], features=[list(f) for f in F], products=prods,
                   equal=equal, caches=caches, coded=cw, ranks=ranks), open(out, "x"))


if __name__ == "__main__":
    main()
