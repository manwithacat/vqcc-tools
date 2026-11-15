#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Static analysis of a loader slot tensor: plane ranks with/without the pad-free row, dimension-count (generic rank)
bounds for alternative low/high splits, and the one-hot hybrid price. Usage: python3 chem_c12_bounds.py TENSOR.json"""
import json, sys, math
import numpy as np
d = json.load(open(sys.argv[1])); m, Lm, H = d["m"], d["Lm"], d["H"]
rows = d["rows"]
def rk(M):
    M = [int(x) for x in M]; r = 0; basis = []
    for v in M:
        for b in basis: v = min(v, v ^ b)
        if v: basis.append(v); r += 1
    return r
planes_full, planes_75 = [], []
for o in range(m):
    cols = []
    for h in range(H):
        c = 0
        for l in range(Lm):
            c |= ((rows[h] >> (l * m + o)) & 1) << l
        cols.append(c)
    planes_full.append(rk(cols)); planes_75.append(rk(cols[:75]))
print("plane ranks (15x76):", sorted(set(planes_full)), "sum", sum(planes_full), "| without row 76:", sorted(set(planes_75)), "sum", sum(planes_75))
# row-space rank of the 76 rows (as 525-bit vectors)
print("row-space rank", rk(rows), "of", H)
def generic(n1, n2, n3):
    return math.ceil(n1 * n2 * n3 / (n1 + n2 + n3 - 2))
A = 1220
print("split  H  lowmon  caches(mono)  caches(one-hot)  generic-features  products>=  total(mono)  total(one-hot hi+lo)")
for l in (2, 3, 4, 5, 6):
    L = 1 << l; Hh = -(-A // L); hb = (Hh - 1).bit_length()
    caches_m = (Hh - 1 - hb) + (L - 1 - l)           # high monomials <H beyond linear + low monomials beyond linear
    caches_oh = (Hh - 1) + (L - 1)                    # one-hot unary iteration ~ (values - 1) ANDs each side
    g = generic(Hh - 1, L - 1, m); prods = math.ceil(g / 2)
    print(f"l={l}  {Hh}  {L-1}  {caches_m}  {caches_oh}  {g}  {prods}  {caches_m + prods}  {caches_oh + prods}")
print("shipped l=4: caches 75, products 195 (features 390 = 1.21 x generic", generic(76, 15, 35), ")")
