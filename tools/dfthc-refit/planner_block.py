# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Block-partition planner for generic loaders (a pairing-scored planner, ported onto planner_shared).

Model and steps are planner_shared's (see its docstring): the L low monomials are split into blocks, the low forms
are all nonzero combinations inside a block, relations are block-supported, the never-read padded rows complete the
last high row, and `pair_lines` pairs W over the exact candidate lines. For generic loaders (no coded outputs) there
is no structural start; relations are admitted from nothing:

1. **Phase 1 (form-space cut ranking).** Blocks round-robin (fewest relations first); candidates are random combinations of the
   block's kernel outside the admitted span, ranked by the number of form spaces V_f they shrink (fewest first); a
   candidate is kept if W stays spanned and, once the block holds `check_from` relations, the pairing is perfect
   (products = ceil(dim W / 2)). The original planner checked every step with a full pairing; here the pairing is
   `pair_lines`, and the early steps (where perfect pairing is not in doubt) are checked for feasibility only.
2. **Phase 2.** planner_shared.admit_shared with template-protecting candidates: relations admitted
   while the product count does not rise.
3. **Exchange moves (optional).** planner_shared.exchange: drop j relations of a block, re-admit j+1.

The block partition is an input (for Syed Asad Ali's f4291619f2fd table: 1,5,13 / 2,6,14 / 3,7,15 / 4,8,11 /
9,10,12, recovered from the shipped loaders). Output: Lera-convention terms, exact against the rewritten delta, for
`tensorqrom.build_circuit(..., groups=None, scratch=True)`.

Credit: block partition recovered from Syed Asad Ali's generic loaders; relation model after
lightningv08's pairing method; padded rows after qwertykeypadapple's padded-row baseline; emitter
Lera's (marculera) tensorqrom, unmodified.
"""
from __future__ import annotations

import random
import time

import planner_shared as ps
from recover import set_bits


def cuts_score(P, r, spaces):
    """Number of form spaces V_f a new relation r shrinks (the phase-1 ranking)."""
    m = P.m
    ch = [(r >> (j * m)) & P.mask for j in range(P.L)]
    s = 0
    for f, V in spaces.items():
        e = 0
        for j in set_bits(f):
            e ^= ch[j]
        if e and any((e & v).bit_count() & 1 for v in V):
            s += 1
    return s


def admit_perfect(P, rng, rels=None, *, check_from=12, pool=60, patience=12, trials=3, max_fails=3, cpu_cap=900.0,
                  log=None):
    """Phase 1: (products, relations, pairing) after admitting relations one at a time while the pairing stays
    perfect (feasibility only while the block holds fewer than `check_from` relations)."""
    t0 = time.process_time()
    rels = list(rels or [])
    piv = {}
    for r in rels:
        ps.append_row(r, piv)
    nb = len(P.blocks)
    per_block = [len(ps.block_relations(P, rels, b)) for b in range(nb)]
    ok, spaces = P.feasible(rels)
    assert ok
    cur = None
    fails = 0
    while fails < max_fails * nb and time.process_time() - t0 < cpu_cap:
        b = min(range(nb), key=lambda i: (per_block[i], rng.random())) if fails == 0 else rng.randrange(nb)
        cands = ps.block_candidates(P, b, piv, rng, pool=pool, enum_dim=0)
        scored = sorted((cuts_score(P, r, spaces), rng.random(), r) for r in cands)
        added = False
        for _, _, r in scored[:patience]:
            trial = rels + [r]
            ok, sp = P.feasible(trial)
            if not ok:
                continue
            pr = None
            if per_block[b] + 1 >= check_from:
                pr = P.pair_lines(sp, P.dims(trial), rng, trials=trials, aug="quotient",
                                  seed_lines=cur[0] if cur is not None else None)
                if pr is None or ps.products_of(pr) > (P.N - len(trial) + 1) // 2:
                    continue
            rels, spaces = trial, sp
            if pr is not None:
                cur = pr
            ps.append_row(r, piv)
            per_block[b] += 1
            added = True
            break
        fails = 0 if added else fails + 1
    if cur is None or ps.products_of(cur) > (P.N - len(rels) + 1) // 2:
        cur = P.pair_lines(spaces, P.dims(rels), rng, trials=max(trials, 6), aug="quotient",
                           seed_lines=cur[0] if cur else None)
    if log:
        log(f"phase 1: {len(rels)} relations {per_block}, {ps.products_of(cur)} products "
            f"({time.process_time() - t0:.0f} CPU-s)")
    return ps.products_of(cur), rels, spaces, cur


def plan_generic(delta, real_rows, blocks, *, seed=0, pad_free=True, phase1_kw=None, admit_kw=None,
                 exchange_kw=None, log=None, backend=None):
    """Plan one generic loader on a given block partition (slot lists, monomial - 1). Returns a dict like
    planner_shared.plan (terms, rewritten delta, products, relations, per-phase counts)."""
    rng = random.Random(seed)
    P = ps.SharedPlanner(delta, real_rows, (), pad_free=pad_free, coded_block=False, backend=backend)
    P.setup(blocks)
    phases = {}
    n1, rels, sp, pr = admit_perfect(P, rng, log=log, **(phase1_kw or {}))
    phases["phase1"] = dict(products=n1, relations=len(rels))
    akw = dict(dict(protect=20, pool=60, patience=12, max_fails=4, cpu_cap=1500.0, aug="quotient"), **(admit_kw or {}))
    if exchange_kw is not None:
        akw.setdefault("prefer_more", True)
    n2, rels2, pr2 = ps.admit_shared(P, rels, sp, pr, rng, log=log, **akw)
    phases["phase2"] = dict(products=n2, relations=len(rels2))
    if log:
        log(f"phase 2: {len(rels2)} relations, {n2} products")
    if exchange_kw is not None:
        ex = ps.exchange(P, rels2, pr2, rng, log=log, **exchange_kw)
        phases["exchange"] = ex["stats"]
        if ex["best"][0] <= n2:
            n2, rels2, pr2 = ex["best"]
    pairs, singles = pr2
    rows, d, changes = P.complete(rels2)
    terms = P.terms(rows, pairs, singles)
    return dict(products=len(terms), pairs=len(pairs), singles=len(singles), relations=len(rels2),
                per_block=[len(ps.block_relations(P, rels2, b)) for b in range(len(P.blocks))],
                blocks=[[s + 1 for s in x] for x in blocks], terms=[tuple(int(x) for x in t) for t in terms],
                delta=d, pad_changed_outputs=sorted(changes), phases=phases,
                relations_hex=[hex(r) for r in rels2])
