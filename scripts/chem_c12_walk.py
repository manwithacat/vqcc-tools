#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Out-of-layout-ish heuristic: flip-graph walk over GF(2) rank-one feature decompositions (a x b x s) of the slot
tensor T (76 high x 15 low x 35 outputs), cost = number of Toffoli products when two features with equal s share one
CCX: cost = sum over s-classes of ceil(n_s/2). Optional don't-care move: toggle high bit 76 of a feature whose low form
touches none of the pinned low monomials 1..3 (row 76 only has 4 valid low rows).
  python3 chem_c12_walk.py TENSOR.json SEED SECONDS OUT.json [--dc] [--temp T]"""
import json, random, sys, time
from collections import defaultdict

def main():
    tj, seed, secs, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    dc = "--dc" in sys.argv
    temp = float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else 0.0
    rng = random.Random(seed)
    d = json.load(open(tj))
    F = [list(f) for f in d["features"]]
    n = len(F)
    idx = [defaultdict(set) for _ in range(3)]
    ns = defaultdict(int)
    for i, f in enumerate(F):
        for ax in range(3):
            idx[ax][f[ax]].add(i)
        ns[f[2]] += 1
    alive = set(range(n))
    rank_mode = "--rank" in sys.argv
    def cost():
        return len(alive) if rank_mode else sum((c + 1) // 2 for c in ns.values())
    def pcost():
        return sum((c + 1) // 2 for c in ns.values())
    cur = cost(); best = cur; best_F = [tuple(f) for f in F]; best_p = pcost()
    PIN = 0b111          # low monomials 1,2,3 (bits 0..2)
    HB = 1 << 75         # high monomial 76 (bit 75)
    def setf(i, ax, v):
        old = F[i][ax]
        idx[ax][old].discard(i)
        if not idx[ax][old]: del idx[ax][old]
        if ax == 2:
            ns[old] -= 1
            if ns[old] == 0: del ns[old]
            ns[v] += 1
        F[i][ax] = v
        idx[ax][v].add(i)
    def kill(i):
        for ax in range(3):
            idx[ax][F[i][ax]].discard(i)
            if not idx[ax][F[i][ax]]: del idx[ax][F[i][ax]]
        ns[F[i][2]] -= 1
        if ns[F[i][2]] == 0: del ns[F[i][2]]
        alive.discard(i)
    alist = list(alive)
    t0 = time.time(); steps = 0; hist = []; flips = reductions = merges = 0
    while time.time() - t0 < secs:
        steps += 1
        if steps % 4096 == 0:
            alist = list(alive)
        i = alist[rng.randrange(len(alist))]
        if i not in alive: continue
        if dc and rng.random() < 0.01:
            if not F[i][0] & PIN:
                old_cost = cur
                setf(i, 1, F[i][1] ^ HB)
                new = cost()
                if new <= cur: cur = new
                else: setf(i, 1, F[i][1] ^ HB)
            continue
        ax = rng.randrange(3)
        cand = idx[ax][F[i][ax]]
        if len(cand) < 2: continue
        j = rng.choice(tuple(cand))
        if j == i: continue
        Y, Z = [x for x in range(3) if x != ax]
        if rng.random() < 0.5: Y, Z = Z, Y
        oldYi, oldZj = F[i][Y], F[j][Z]
        zi = F[i][Z]
        setf(i, Y, oldYi ^ F[j][Y]); setf(j, Z, oldZj ^ zi)
        flips += 1
        if F[i][Y] == 0 or F[j][Z] == 0:        # a factor vanished: rank reduction
            for x in (i, j):
                if F[x][0] == 0 or F[x][1] == 0 or F[x][2] == 0: kill(x)
            cur = cost(); reductions += 1
        else:
            new = cost()
            if new <= cur or (temp and rng.random() < 2.718 ** (-(new - cur) / temp)):
                cur = new
            else:
                setf(i, Y, oldYi); setf(j, Z, oldZj)
        if steps % 20000 == 0:
            # merge scan: terms sharing two factors
            for (p1, p2, p3) in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                seen = {}
                for x in tuple(alive):
                    key = (F[x][p1], F[x][p2])
                    y = seen.get(key)
                    if y is None: seen[key] = x
                    else:
                        setf(y, p3, F[y][p3] ^ F[x][p3]); kill(x); merges += 1
                        if F[y][p3] == 0: kill(y)
            cur = cost()
        if cur < best:
            best = cur; best_F = [tuple(F[x]) for x in alive]; best_p = pcost()
            hist.append((round(time.time() - t0), steps, best, len(alive), best_p))
    json.dump(dict(seed=seed, steps=steps, best_cost=best, best_products=best_p, best_features=len(best_F), final_cost=cur, final_features=len(alive),
                   hist=hist, features=best_F), open(out, "w"))
    print(seed, "steps", steps, "flips", flips, "reductions", reductions, "merges", merges, "best cost", best, "products at best", best_p, "features", len(best_F), "final", cur, len(alive), hist[-3:])
main()
