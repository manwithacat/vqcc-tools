# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Block-partition loader planner with structural relations and shared products (any family, generic or coded).

Model (a whole-loader relation model, after lightningv08's pairing method): the loader's
mixed ANF tensor has one row per high monomial 1..H-1, a vector over (L low monomials) x (m outputs). A plan is a
subspace W containing every row and spanned by rank-one vectors (low form) x (output vector); every product of
Lera's emitter is a *pair* of such vectors with the same output vector, so a perfectly paired W costs
dim(W) / 2 products plus the cache Toffolis. W is fixed by its relations (W = relations-perp).

The construction this module implements was recovered from the coded loaders of "i want to run gemma 9b"'s
soft-word family 6566cd395128 (via recover.py):

1. **Blocks.** The L low monomials are split into blocks (triples by default); every low form lies inside one block.
   Relations are block-supported (coordinates: the block's monomials x all outputs).
2. **Coded outputs split off.** A coded output (bit 0 of a coded word, affine in the low address bits) is zero on
   every non-single-bit low monomial: those coordinates are admitted as trivial relations, so a coded output only
   ever pairs with a single-bit low form. The coded outputs at the single-bit monomials form the "coded LSB" part.
3. **Full structural annihilator.** Per block, every relation supported on the block's non-coded coordinates (the
   full annihilator of the rows restricted there) is admitted at once, so W restricted to those coordinates is the
   row span itself.
4. **Shared products.** Further block-kernel relations that mix coded and non-coded coordinates are admitted one at
   a time while the pairing does not get worse (pairing-scored admission). They let one product feed coded
   and non-coded outputs together.
5. **Padded rows as don't-cares.** The never-read addresses A .. 16*ceil(A/16)-1 change only the last high row, on
   low monomials >= A mod 16 (non-coded outputs here). That row is left out of the kernels and completed at the end
   so that it lies in W; the padded table entries are rewritten to match (dfthc_refit.apply_pad_changes).
6. **Pairing** (`pair_lines`, from this tool's earlier pairing-scored planner): enumerate every candidate line (form f, form g, v in
   V_f cap V_g), randomised greedy selection, 1-out/2-in augmentation, then singles.

The partition is searched (`search_partition`): random triple partitions are scored by the structural plan's
paired product count (steps 1-3, 5, 6), the best few are then run through step 4.

Inputs: the loader's delta (dfthc_refit.loader_delta: coded bit 0 already affine-padded) and its coded word
indices. Output: Lera-convention terms (a1, b1, a2, b2, s), exact against the (rewritten) delta's mixed tensor, for
`tensorqrom.build_circuit(..., groups=None, scratch=True)` (full low cache, borrowed top bits: 126 qubits on the
five-word layout).

Credit: construction recovered from "i want to run gemma 9b"'s coded loaders (dataset
iwanttorungemma9b/vqcc-dfthc-soft-word-loaders v2, Apache-2.0; their loaders descend from Ayodeji's and Syed Asad
Ali's coded loaders); block partition first recovered from Syed Asad Ali's loaders; pairing and relation model
after lightningv08's pairing method; padded rows as
qwertykeypadapple's padded-row baseline; emitter Lera's (marculera) tensorqrom, unmodified.
"""
from __future__ import annotations

import itertools
import random
import time

import numpy as np

from recover import Basis, anf_rows, column, gf2_backend, make_basis, nullspace, rank, rust_module, set_bits


def reduce_row(v, piv):
    while v:
        p = v.bit_length() - 1
        if p not in piv:
            return v
        v ^= piv[p]
    return 0


def append_row(v, piv):
    v = reduce_row(v, piv)
    if v:
        piv[v.bit_length() - 1] = v
        return True
    return False


def span_vectors(D, cap, rng, samples):
    if len(D) <= cap:
        out = []
        for code in range(1, 1 << len(D)):
            v = 0
            for i in range(len(D)):
                if code >> i & 1:
                    v ^= D[i]
            out.append(v)
        return out
    out = list(D)
    seen = set(out)
    for _ in range(samples):
        v = 0
        for i in range(len(D)):
            if rng.getrandbits(1):
                v ^= D[i]
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def intersect(V1, V2, m, backend=None):
    """Basis of span(V1) cap span(V2) in GF(2)^m."""
    if gf2_backend(backend) == "rust":
        return rust_module().gf2_intersect(V1, V2, m)
    cons = nullspace(V2, m) if len(V2) < m else []
    if not cons:
        return list(V1)
    eqs = [sum(((V1[i] & e).bit_count() & 1) << i for i in range(len(V1))) for e in cons]
    out = []
    for c in nullspace(eqs, len(V1)):
        v = 0
        for i in set_bits(c):
            v ^= V1[i]
        out.append(v)
    return out


def decompose(B, v):
    """(residual, coords): v = (sum of B's vectors in coords) + residual, residual on non-pivot bits only (a linear
    projection, unlike Basis.reduce)."""
    res, c = 0, 0
    while v:
        p = v.bit_length() - 1
        if p in B.piv:
            pv, pc = B.piv[p]
            v ^= pv
            c ^= pc
        else:
            res |= 1 << p
            v ^= 1 << p
    return res, c


def _independent(items):
    """items: (block, vector) pairs; True if the vectors of each block are linearly independent."""
    piv = {}
    for b, x in items:
        d = piv.setdefault(b, {})
        if not append_row(x, d):
            return False
    return True


def augment_quotient(ends, sel, nb, total_dim, rng, max_pass, l1_cap=40):
    """1-out/2-in augmentation of a line selection (indices into `ends`, each a pair of (block, vector)): remove a
    selected line p and add two lines whose four ends are independent modulo the other selected lines. In that
    quotient an end's image is its residual outside the selected span plus its coordinates on p's two ends, so a
    candidate line need not lie in p's span. Stops when the selection spans `total_dim` dimensions, after
    `max_pass` successful swaps, or when no swap exists."""
    sel = list(sel)
    for _ in range(max_pass):
        if 2 * len(sel) >= total_dim - 1:
            break
        bases = [Basis() for _ in range(nb)]
        pos = [{} for _ in range(nb)]
        for li in sel:
            for e, (b, vec) in enumerate(ends[li]):
                assert bases[b].add(vec)
                pos[b][(li, e)] = len(bases[b]) - 1
        selset = set(sel)
        dec = []
        for li in range(len(ends)):
            if li in selset:
                continue
            (b1, a), (b2, c) = ends[li]
            r1, c1 = decompose(bases[b1], a)
            r2, c2 = decompose(bases[b2], c)
            dec.append((li, b1, r1, c1, b2, r2, c2))
        order = list(sel)
        rng.shuffle(order)
        found = None
        for p in order:
            (bx, _), (by, _) = ends[p]
            px, py = pos[bx][(p, 0)], pos[by][(p, 1)]

            def img(b, r, c):
                x = r << 2
                if b == bx and c >> px & 1:
                    x |= 1
                if b == by and c >> py & 1:
                    x |= 2
                return x
            cands = []
            for li, b1, r1, c1, b2, r2, c2 in dec:
                i1, i2 = img(b1, r1, c1), img(b2, r2, c2)
                if i1 and i2 and (b1 != b2 or i1 != i2):
                    cands.append((li, b1, i1, b2, i2))
            if len(cands) < 2:
                continue
            rng.shuffle(cands)
            for i, (l1, a1, x1, a2, x2) in enumerate(cands[:l1_cap]):
                for l2, d1, y1, d2, y2 in cands[i + 1:]:
                    if _independent([(a1, x1), (a2, x2), (d1, y1), (d2, y2)]):
                        found = (p, l1, l2)
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
        p, l1, l2 = found
        sel = [x for x in sel if x != p] + [l1, l2]
    return sel


def _augment_quotient_rust(gs, sel, total_dim, rng, max_pass, l1_cap=40):
    """augment_quotient on a kickmix_analysis.GroupSet holding the lines' ends (backend='rust'); same random calls,
    same result."""
    sel = list(sel)
    for _ in range(max_pass):
        if 2 * len(sel) >= total_dim - 1:
            break
        gs.build(sel)
        pos = {li: gs.positions(li) for li in sel}
        selset = set(sel)
        gs.compute_decomp([li for li in range(len(gs)) if li not in selset])
        order = list(sel)
        rng.shuffle(order)
        found = None
        for p in order:
            (bx, px), (by, py) = pos[p]
            cands = gs.quotient_cands(bx, px, by, py)
            if len(cands) < 2:
                continue
            rng.shuffle(cands)
            hit = gs.quotient_pair(cands, l1_cap)
            if hit is not None:
                found = (p, hit[0], hit[1])
                break
        if not found:
            break
        p, l1, l2 = found
        sel = [x for x in sel if x != p] + [l1, l2]
    return sel


def canonical(vectors):
    """Reduced row echelon basis of span(vectors) as a sorted tuple (a hashable key for the span)."""
    piv = {}
    for v in vectors:
        for p in sorted(piv, reverse=True):
            if v >> p & 1:
                v ^= piv[p]
        if v:
            p = v.bit_length() - 1
            for q in list(piv):
                if piv[q] >> p & 1:
                    piv[q] ^= v
            piv[p] = v
    return tuple(sorted(piv.values()))


class SharedPlanner:
    def __init__(self, delta, real_rows, coded_words=(), *, brot=8, lowbits=4, pad_free=True, coded_block=True,
                 singles_forms="any", backend=None):
        self.backend = gf2_backend(backend)      # GF(2) primitives: "python" (reference) or "rust"
        delta = np.asarray(delta, np.int64)
        Lb = 1 << lowbits
        if len(delta) % Lb:
            delta = np.concatenate([delta, np.zeros((-len(delta) % Lb, delta.shape[1]), np.int64)])
        self.delta0 = delta
        self.A = int(real_rows)
        self.brot, self.lowbits = brot, lowbits
        self.coded_block_on = bool(coded_block)
        assert singles_forms in ("any", "coded")
        self.singles_mode = singles_forms
        self.rows, self.m, self.L = anf_rows(delta, brot, lowbits)
        m, L = self.m, self.L
        self.H = len(self.rows)
        self.N = L * m
        self.mask = (1 << m) - 1
        self.coded = sum(1 << (j * (brot - 1)) for j in coded_words)
        self.single_slots = [s for s in range(L) if (s + 1) & s == 0]
        self.trivial = [s * m + o for s in range(L) if s not in self.single_slots for o in set_bits(self.coded)]
        for r in self.rows:
            for c in self.trivial:
                assert not (r >> c) & 1, "coded output nonzero on a non-single-bit low monomial"
        n0 = self.A % Lb
        self.pad_free = bool(pad_free and n0)
        self.last = self.H - 1
        if self.pad_free:
            self.free_slots = list(range(n0 - 1, L))
            noncoded = self.mask & ~self.coded
            self.free_mask = sum(noncoded << (s * m) for s in self.free_slots)
            self.rows_k = self.rows[:self.last]
        else:
            self.free_slots, self.free_mask, self.rows_k = [], 0, list(self.rows)

    # ------------------------------------------------------------------ partition-dependent setup
    def setup(self, blocks):
        """blocks: lists of low slots (monomial - 1). Builds forms and per-block kernels."""
        m = self.m
        assert sorted(s for b in blocks for s in b) == list(range(self.L))
        self.blocks = [list(b) for b in blocks]
        self.forms_by_block, self.blk = [], {}
        for bi, b in enumerate(self.blocks):
            fs = []
            for code in range(1, 1 << len(b)):
                f = sum(1 << b[i] for i in range(len(b)) if code >> i & 1)
                fs.append(f)
                self.blk[f] = bi
            self.forms_by_block.append(fs)
        # singles block: the coded outputs live on the single-bit monomials only; any combination of two or more
        # single-bit monomials is a further form (the singles themselves stay in their triples). With
        # singles_forms="coded" these forms carry coded outputs only; with "any" (recovered from loader 14 of
        # 6566cd395128) they may also carry non-coded outputs, across triples.
        self.coded_only_forms = set()
        self.pair_global = False
        if self.coded and self.coded_block_on:
            ss = self.single_slots
            fs = []
            for code in range(1, 1 << len(ss)):
                if code.bit_count() >= 2:
                    f = sum(1 << ss[i] for i in range(len(ss)) if code >> i & 1)
                    fs.append(f)
                    self.blk[f] = len(self.blocks)
                    self.coded_only_forms.add(f)
            self.forms_by_block.append(fs)
            self.pair_global = True          # features of the coded block overlap the triples' coded coordinates
        self.forms = [f for fs in self.forms_by_block for f in fs]
        triv = set(self.trivial)
        coded_coords = [s * m + o for s in self.single_slots for o in set_bits(self.coded)]
        self.coded_coords_mask = sum(1 << c for c in coded_coords)
        self.block_coords, self.block_masks, self.struct_kernels, self.mixed_kernels = [], [], [], []
        self.own_kernels = []          # the kernel on the block's own coordinates (shipped relations live here)
        for b in self.blocks:
            coords = [s * m + o for s in b for o in range(m) if s * m + o not in triv]
            nc = [c for c in coords if not (self.coded >> (c % m)) & 1]
            self.block_coords.append(coords)
            self.block_masks.append(sum(1 << c for c in coords))
            self.struct_kernels.append(self._kernel(nc))
            self.own_kernels.append(self._kernel(coords))
            full = self._kernel(sorted(set(coords) | set(coded_coords)) if self.pair_global else coords)
            piv = {}
            for r in self.struct_kernels[-1]:
                append_row(r, piv)
            self.mixed_kernels.append([r for r in full if reduce_row(r, piv)])   # completes struct to full kernel
            # (mixed_kernels spans full kernel modulo struct; random combos below add struct parts, harmless)
        self.trivial_rels = [1 << c for c in self.trivial]

    def _kernel(self, coords):
        idx = {c: i for i, c in enumerate(coords)}
        brows = []
        for row in self.rows_k:
            v = 0
            for c, i in idx.items():
                if (row >> c) & 1:
                    v |= 1 << i
            brows.append(v)
        out = []
        for r in nullspace(brows, len(coords), self.backend):
            out.append(sum(1 << coords[i] for i in set_bits(r)))
        return out

    # ------------------------------------------------------------------ relation model
    def col(self, f, v):
        return column(f, v, self.m)

    def form_spaces(self, relations):
        m = self.m
        chunks = [[(r >> (j * m)) & self.mask for j in range(self.L)] for r in relations]
        out = {}
        for f in self.forms:
            piv, cons = {}, []
            for ch in chunks:
                e = 0
                for j in set_bits(f):
                    e ^= ch[j]
                if e and append_row(e, piv):
                    cons.append(e)
            if f in self.coded_only_forms and self.singles_mode == "coded":
                for o in set_bits(self.mask & ~self.coded):
                    if append_row(1 << o, piv):
                        cons.append(1 << o)
            out[f] = nullspace(cons, m, self.backend) if cons else [1 << i for i in range(m)]
        return out

    def feasible(self, relations):
        spaces = self.form_spaces(relations)
        piv = {}
        for f, V in spaces.items():
            for v in V:
                append_row(self.col(f, v), piv)
        ok = len(piv) == self.N - len(relations)
        if ok and self.last_row_completion(relations) is None:
            ok = False
        return ok, spaces

    def last_row_completion(self, relations):
        """pad free: delta supported on the free coordinates with <r, last + delta> = 0 for all relations."""
        if not self.pad_free:
            return 0 if all(((r & self.rows[self.last]).bit_count() & 1) == 0 for r in relations) else None
        last = self.rows[self.last]
        piv = {}
        for r in relations:
            rhs = (r & last).bit_count() & 1
            v = ((r & self.free_mask) << 1) | rhs
            v = reduce_row(v, piv)
            if v == 1:
                return None
            if v:
                piv[v.bit_length() - 1] = v
        sol = 0
        for p in sorted(piv):
            v = piv[p]
            rest = (v >> 1) & ~(1 << (p - 1))
            if (v & 1) ^ ((rest & sol).bit_count() & 1):
                sol |= 1 << (p - 1)
        return sol

    def dims(self, relations):
        """dim W restricted to each block (every non-trivial relation is block-supported); one global dimension
        when the coded block is on."""
        if self.pair_global:
            return [self.N - len(relations)]
        out = []
        for bi in range(len(self.blocks)):
            bm = self.block_masks[bi]
            out.append(len(self.block_coords[bi]) - sum(1 for r in relations if r & bm and not r & ~bm))
        return out

    # ------------------------------------------------------------------ pairing over exact candidate lines
    def lines(self, spaces, rng, cap=4, samples=12, cache=None):
        """Candidate pair lines (f, g, v), v in V_f cap V_g (all of it when dim <= cap, else a sample).
        `cache` (a dict, optional) memoises the intersections by the two spaces' canonical bases, so states that
        differ in one block recompute only that block's form pairs."""
        out = []
        keys = None if cache is None else {f: canonical(spaces[f]) for f in self.forms}
        for f, g in itertools.combinations(self.forms, 2):
            if cache is None:
                D = intersect(spaces[f], spaces[g], self.m, self.backend)
            else:
                key = (keys[f], keys[g])
                D = cache.get(key)
                if D is None:
                    D = cache[key] = intersect(spaces[f], spaces[g], self.m, self.backend)
            if D:
                for v in span_vectors(D, cap, rng, samples):
                    out.append((f, g, v))
        return out

    def pair_lines(self, spaces, dims, rng, trials=3, max_pass=400, lines=None, seed_lines=None, aug="rebuild"):
        """(pairs, singles) spanning W (per-block dims `dims`) with as many pairs as found, or None.
        Randomised greedy over the candidate lines, 1-out/2-in
        augmentation, then singles. `seed_lines`: lines of a previous pairing; those still inside W are
        taken first in the first trial (warm start after a local change of the relations). `aug="quotient"`
        runs the augmentation in the quotient by the remaining pairs (any line whose ends are independent there
        qualifies, a superset of the "rebuild" rule) and skips it once the pairs span W; "rebuild" is the original rule."""
        if self.backend == "rust":
            return self._pair_lines_rust(spaces, dims, rng, trials, max_pass, lines, seed_lines, aug)
        if lines is None:
            lines = self.lines(spaces, rng)
        seed_idx = []
        if seed_lines:
            lines = list(lines)
            index = {ln: i for i, ln in enumerate(lines)}
            Bf = {}
            for ln in seed_lines:
                f, g, v = ln
                ok = True
                for x in (f, g):
                    if x not in Bf:
                        Bf[x] = Basis()
                        for u in spaces[x]:
                            Bf[x].add(u)
                    if Bf[x].solve(v) is None:
                        ok = False
                if not ok:
                    continue
                if ln not in index:
                    index[ln] = len(lines)
                    lines.append(ln)
                seed_idx.append(index[ln])
        if self.pair_global:
            nb = 1
            ends = [((0, self.col(f, v)), (0, self.col(g, v))) for f, g, v in lines]
            single_groups = [list(self.forms)]
        else:
            nb = len(self.blocks)
            ends = [((self.blk[f], self.col(f, v)), (self.blk[g], self.col(g, v))) for f, g, v in lines]
            single_groups = self.forms_by_block
        best = None
        for trial in range(trials):
            order = list(range(len(lines)))
            rng.shuffle(order)
            if trial == 0 and seed_idx:
                first = set(seed_idx)
                order = list(seed_idx) + [li for li in order if li not in first]
            sel = []

            def build(selected):
                bases = [Basis() for _ in range(nb)]
                owner = [[] for _ in range(nb)]
                for li in selected:
                    for e, (b, vec) in enumerate(ends[li]):
                        assert bases[b].add(vec)
                        owner[b].append((li, e))
                return bases, owner

            def addable(li, bases):
                (b1, a), (b2, c) = ends[li]
                if b1 != b2:
                    return bases[b1].solve(a) is None and bases[b2].solve(c) is None
                B = bases[b1]
                return B.solve(a) is None and B.solve(c) is None and B.solve(a ^ c) is None

            bases, owner = build([])
            for li in order:
                if addable(li, bases):
                    for e, (b, vec) in enumerate(ends[li]):
                        bases[b].add(vec)
                        owner[b].append((li, e))
                    sel.append(li)
            if aug == "quotient":
                sel = augment_quotient(ends, sel, nb, sum(dims), rng, max_pass)
            for _ in range(max_pass if aug == "rebuild" else 0):
                bases, owner = build(sel)
                pos = [{(li, e): i for i, (li, e) in enumerate(owner[b])} for b in range(nb)]
                selset = set(sel)
                status = {}
                for li in range(len(lines)):
                    if li not in selset:
                        status[li] = [(b, bases[b].solve(vec)) for b, vec in ends[li]]
                improved = False
                sel_order = list(sel)
                rng.shuffle(sel_order)
                for p in sel_order:
                    (bx, _), (by, _) = ends[p]
                    px, py = pos[bx][(p, 0)], pos[by][(p, 1)]
                    enabled = []
                    for li, st in status.items():
                        ok = True
                        for b, coords in st:
                            if coords is None:
                                continue
                            if (b == bx and coords >> px & 1) or (b == by and coords >> py & 1):
                                continue
                            ok = False
                            break
                        if ok:
                            enabled.append(li)
                    if len(enabled) < 2:
                        continue
                    rest = [x for x in sel if x != p]
                    b0, _ = build(rest)
                    rng.shuffle(enabled)
                    done = False
                    for i, l1 in enumerate(enabled):
                        if not addable(l1, b0):
                            continue
                        b1_, _ = build(rest + [l1])
                        for l2 in enabled[i + 1:]:
                            if addable(l2, b1_):
                                sel = rest + [l1, l2]
                                done = True
                                break
                        if done:
                            break
                    if done:
                        improved = True
                        break
                if not improved:
                    break
            bases, owner = build(sel)
            singles = []
            ok = True
            for bi, fs in enumerate(single_groups):
                fs = list(fs)
                rng.shuffle(fs)
                B = bases[bi]
                n = len(B)
                for f in fs:
                    if n >= dims[bi]:
                        break
                    for v in spaces[f]:
                        if B.add(self.col(f, v)):
                            singles.append((f, v))
                            n += 1
                            if n >= dims[bi]:
                                break
                if n != dims[bi]:
                    ok = False
                    break
            if not ok:
                return None
            cost = len(sel) + len(singles)
            if best is None or cost < best[0]:
                best = (cost, [lines[li] for li in sel], singles)
            if cost <= (sum(dims) + 1) // 2:
                break
        return None if best is None else (best[1], best[2])

    def _pair_lines_rust(self, spaces, dims, rng, trials, max_pass, lines, seed_lines, aug):
        """pair_lines with the GF(2) work in kickmix_analysis (backend='rust'): the same control flow and
        random calls as the Python above; the lines' ends live in one GroupSet, so each pass is a few calls."""
        ka = rust_module()
        if lines is None:
            lines = self.lines(spaces, rng)
        seed_idx = []
        if seed_lines:
            lines = list(lines)
            index = {ln: i for i, ln in enumerate(lines)}
            Bf = {}
            for ln in seed_lines:
                f, g, v = ln
                ok = True
                for x in (f, g):
                    if x not in Bf:
                        Bf[x] = ka.Gf2Basis(self.m)
                        Bf[x].add_many(spaces[x])
                    if not Bf[x].contains(v):
                        ok = False
                if not ok:
                    continue
                if ln not in index:
                    index[ln] = len(lines)
                    lines.append(ln)
                seed_idx.append(index[ln])
        if self.pair_global:
            nb = 1
            ends = [((0, self.col(f, v)), (0, self.col(g, v))) for f, g, v in lines]
            single_groups = [list(self.forms)]
        else:
            nb = len(self.blocks)
            ends = [((self.blk[f], self.col(f, v)), (self.blk[g], self.col(g, v))) for f, g, v in lines]
            single_groups = self.forms_by_block
        gs = ka.GroupSet(nb, self.N, ends)
        best = None
        for trial in range(trials):
            order = list(range(len(lines)))
            rng.shuffle(order)
            if trial == 0 and seed_idx:
                first = set(seed_idx)
                order = list(seed_idx) + [li for li in order if li not in first]
            gs.clear()
            sel = gs.greedy(order)
            if aug == "quotient":
                sel = _augment_quotient_rust(gs, sel, sum(dims), rng, max_pass)
            for _ in range(max_pass if aug == "rebuild" else 0):
                gs.build(sel)
                pos = {li: gs.positions(li) for li in sel}
                selset = set(sel)
                gs.compute_status([li for li in range(len(lines)) if li not in selset])
                improved = False
                sel_order = list(sel)
                rng.shuffle(sel_order)
                for p in sel_order:
                    (bx, px), (by, py) = pos[p]
                    enabled = gs.enabled(bx, px, by, py)
                    if len(enabled) < 2:
                        continue
                    rest = [x for x in sel if x != p]
                    gs.build_values(rest)
                    rng.shuffle(enabled)
                    hit = gs.first_pair(enabled)
                    if hit is not None:
                        sel = rest + [enabled[hit[0]], enabled[hit[1]]]
                        improved = True
                        break
                if not improved:
                    break
            gs.build(sel)
            singles = []
            ok = True
            for bi, fs in enumerate(single_groups):
                fs = list(fs)
                rng.shuffle(fs)
                n = gs.size(bi)
                for f in fs:
                    if n >= dims[bi]:
                        break
                    for v in spaces[f]:
                        if gs.add_vec(bi, self.col(f, v)):
                            singles.append((f, v))
                            n += 1
                            if n >= dims[bi]:
                                break
                if n != dims[bi]:
                    ok = False
                    break
            if not ok:
                return None
            cost = len(sel) + len(singles)
            if best is None or cost < best[0]:
                best = (cost, [lines[li] for li in sel], singles)
            if cost <= (sum(dims) + 1) // 2:
                break
        return None if best is None else (best[1], best[2])

    # ------------------------------------------------------------------ terms and table rewrite
    def complete(self, relations):
        """(rows with the completed last row, rewritten full-block delta, {output: low-monomial change})."""
        rows = list(self.rows)
        d = self.delta0
        changes = {}
        if self.pad_free:
            sol = self.last_row_completion(relations)
            assert sol is not None
            rows[self.last] ^= sol
            for o in range(self.m):
                v = sum(((sol >> (s * self.m + o)) & 1) << s for s in range(self.L))
                if v:
                    changes[o] = v
            d = rewrite_pad(self.delta0, changes, self.A, self.brot, self.lowbits)
            assert anf_rows(d, self.brot, self.lowbits)[0] == rows, "padded-row rewrite"
            assert np.array_equal(d[:self.A], self.delta0[:self.A])
        return rows, d, changes

    def terms(self, rows, pairs, singles):
        B = make_basis(self.N, self.backend)
        cols = []
        for f1, f2, v in pairs:
            assert B.add(self.col(f1, v)) and B.add(self.col(f2, v))
            cols.append(("pair", f1, f2, v))
        for f, v in singles:
            assert B.add(self.col(f, v))
            cols.append(("single", f, v))
        sols = [B.solve(r) for r in rows]
        assert all(s is not None for s in sols), "rows outside W"
        out, idx = [], 0
        for c in cols:
            h1 = sum(((s >> idx) & 1) << h for h, s in enumerate(sols))
            if c[0] == "pair":
                h2 = sum(((s >> (idx + 1)) & 1) << h for h, s in enumerate(sols))
                out.append((c[1], h1, c[2], h2, c[3]))
                idx += 2
            else:
                out.append((c[1], h1, 0, 0, c[2]))
                idx += 1
        return out


def rewrite_pad(d, changes, A, brot=8, lowbits=4):
    """As dfthc_refit.apply_pad_changes (any lowbits): rewrite never-read padded entries so that the last high row of
    the mixed ANF tensor changes by `changes` {output: low-monomial mask, bit s = monomial s+1}."""
    d = np.array(d, dtype=np.int64, copy=True)
    L = 1 << lowbits
    h, n0 = A // L, A % L
    for o, v in changes.items():
        w, b = o // (brot - 1), o % (brot - 1)
        for mono in range(1, n0):
            assert not (v >> (mono - 1)) & 1, "change on a low monomial the padded addresses cannot reach"
        for l in range(n0, L):
            f = 0
            for mono in range(1, L):
                if (mono & l) == mono and (v >> (mono - 1)) & 1:
                    f ^= 1
            if f:
                d[L * h + l, w] ^= (1 << b) | ((1 << (brot - 1)) if b == brot - 2 else 0)
    return d


# ---------------------------------------------------------------------- driver

def random_triples(L, rng, size=3, spread_singles=True):
    """A random partition of the L low slots into blocks of `size`. With `spread_singles` the single-bit monomials
    (slots 2**i - 1) go to different blocks, as in the recovered coded loaders (their coded LSB part then spreads
    over the blocks)."""
    while True:
        s = list(range(L))
        rng.shuffle(s)
        blocks = [sorted(s[i:i + size]) for i in range(0, L, size)]
        if not spread_singles:
            return blocks
        if all(sum(1 for x in b if (x + 1) & x == 0) <= 1 for b in blocks):
            return blocks


def structural(P, blocks, rng, trials=2, struct_drop=0):
    """Steps 1-3, 5, 6 for one partition: (products, relations, spaces, pairing) or None if infeasible.
    `struct_drop` > 0 admits a random codimension-`struct_drop` subspace of each block's structural annihilator
    instead of all of it (the shipped loaders admit 12-15 of 14-15 there), leaving room for shared relations."""
    P.setup(blocks)
    rels = list(P.trivial_rels)
    for K in P.struct_kernels:
        K = list(K)
        if struct_drop and len(K) > struct_drop:
            # random invertible recombination, then drop the last struct_drop vectors
            mixed = []
            for _ in range(len(K)):
                while True:
                    v = 0
                    for r in K:
                        if rng.getrandbits(1):
                            v ^= r
                    if v and rank(mixed + [v], P.backend) == len(mixed) + 1:
                        mixed.append(v)
                        break
            K = mixed[:len(K) - struct_drop]
        rels += K
    ok, sp = P.feasible(rels)
    if not ok:
        return None
    pr = P.pair_lines(sp, P.dims(rels), rng, trials=trials)
    if pr is None:
        return None
    return len(pr[0]) + len(pr[1]), rels, sp, pr


def admit_shared(P, rels, sp, pr, rng, *, pool=40, patience=8, max_fails=2, trials=2, cpu_cap=600.0, enum_dim=6,
                 repair_trials=6, polish_trials=16, log=None, prefer_more=False, protect=0, aug="rebuild"):
    """Step 4: admit block-kernel relations (beyond the structural ones) while the paired product count does not
    rise. Returns the best (products, relations, pairing); with `prefer_more` ties in products go to the
    state with more relations (the better start for exchange moves). `protect` adds template-protecting
    candidates (block_candidates)."""
    t0 = time.process_time()
    piv = {}
    for r in rels:
        append_row(r, piv)
    cur_n = len(pr[0]) + len(pr[1])
    best = (cur_n, list(rels), pr)
    lines_cur = P.lines(sp, rng)
    nb = len(P.blocks)
    fails = 0
    last_state = None
    while fails < max_fails * nb and time.process_time() - t0 < cpu_cap:
        b = rng.randrange(nb)
        # the block's kernel modulo the admitted relations (struct parts are already admitted): enumerate all
        # combinations when the quotient is small, else sample
        qpiv, Q = dict(piv), []
        for r in P.mixed_kernels[b] + P.struct_kernels[b]:
            rr = reduce_row(r, qpiv)
            if rr:
                qpiv[rr.bit_length() - 1] = rr
                Q.append(rr)
        cands = []
        if 0 < len(Q) <= enum_dim:
            for code in range(1, 1 << len(Q)):
                v = 0
                for i in set_bits(code):
                    v ^= Q[i]
                cands.append(v)
        elif Q:
            for _ in range(pool):
                v = 0
                for r in Q:
                    if rng.getrandbits(1):
                        v ^= r
                if v:
                    cands.append(v)
        if protect:
            cands += block_candidates(P, b, piv, rng, pool=0, enum_dim=0, protect=protect, pairing=pr)
        if not cands:
            fails += 1
            continue
        sel = set(pr[0])

        def kills(r):
            w = 0
            for f, g, v in lines_cur:
                if ((r & P.col(f, v)).bit_count() & 1) or ((r & P.col(g, v)).bit_count() & 1):
                    w += 3 if (f, g, v) in sel else 1
            return w
        scored = sorted((kills(r), rng.random(), r) for r in cands)
        cand = None
        for _, _, r in scored[:patience]:
            trial = rels + [r]
            ok, sp2 = P.feasible(trial)
            if not ok:
                continue
            lt = P.lines(sp2, rng)
            pr2 = P.pair_lines(sp2, P.dims(trial), rng, trials=trials, lines=lt, aug=aug)
            if pr2 is None:
                continue
            n = len(pr2[0]) + len(pr2[1])
            tgt = (P.N - len(trial) + 1) // 2
            if n > tgt and n <= cur_n + 1 and repair_trials:
                # an imperfect pairing may be the pairing heuristic, not the relation: try harder once
                pr3 = P.pair_lines(sp2, P.dims(trial), rng, trials=repair_trials, lines=lt, aug=aug)
                if pr3 is not None and len(pr3[0]) + len(pr3[1]) < n:
                    pr2, n = pr3, len(pr3[0]) + len(pr3[1])
            if n <= cur_n and (cand is None or n < cand[0]):
                cand = (n, r, sp2, pr2, lt)
                if n <= (P.N - len(trial) + 1) // 2:
                    break
        if cand is None:
            fails += 1
            continue
        n, r, sp, pr, lines_cur = cand
        rels = rels + [r]
        append_row(r, piv)
        cur_n = n
        fails = 0
        if log:
            log(f"shared + block {b}: {len(rels)} relations, {n} products ({time.process_time() - t0:.0f} CPU-s)")
        if n < best[0] or (prefer_more and n == best[0] and len(rels) > len(best[1])):
            best = (n, list(rels), pr)
        last_state = (list(rels), sp, lines_cur, n)
    # polish: re-pair the final state (most relations) with more trials
    if last_state is not None and polish_trials:
        rl, sp_, lt_, n_ = last_state
        if n_ > (P.N - len(rl) + 1) // 2:
            pr4 = P.pair_lines(sp_, P.dims(rl), rng, trials=polish_trials, lines=lt_, aug=aug)
            if pr4 is not None and len(pr4[0]) + len(pr4[1]) < best[0]:
                best = (len(pr4[0]) + len(pr4[1]), rl, pr4)
                if log:
                    log(f"polish: {len(rl)} relations, {best[0]} products")
    return best


# ---------------------------------------------------------------------- exchange moves
#
# Monotone admission (admit_shared) stalls one or two relations short of the shipped plans: on loader 4 of
# 6566cd395128 at 131-132 relations (197 products) against 133 (196), on Syed Asad Ali's generic loaders at 89 (218)
# against 91 (217). The shipped plans trade relations: they admit a slightly different subspace of each block's
# kernel. An exchange move drops j admitted relations of one block (a random codimension-j subspace of the block's
# admitted span) and re-admits up to j+1 relations from the block's kernel, each scored by the achieved pairing.
# A move that ends with one more relation and no more products is progress; one that ends with as many relations
# and no more products is a neutral (plateau) step, accepted when the relation span is new.

def products_of(pr):
    return len(pr[0]) + len(pr[1])


def block_relations(P, rels, b):
    """The admitted relations owned by block b: non-trivial and supported on b's coordinates (plus the coded
    coordinates when the coded block is on)."""
    ext = P.block_masks[b] | (P.coded_coords_mask if P.pair_global else 0)
    triv = set(P.trivial_rels)
    return [r for r in rels if r not in triv and not r & ~ext]


def recombine(vectors, rng):
    """A random basis of span(vectors) (random invertible recombination; vectors assumed independent)."""
    out, piv = [], {}
    while len(out) < len(vectors):
        v = 0
        for r in vectors:
            if rng.getrandbits(1):
                v ^= r
        if v and append_row(v, piv):
            out.append(v)
    return out


def block_candidates(P, b, piv, rng, *, pool=40, enum_dim=6, exclude=None, protect=0, pairing=None, scope="global"):
    """Relations from block b's kernel outside the admitted span `piv` (a pivot dict): the whole quotient when its
    dimension is <= enum_dim, else `pool` random combinations, plus `protect` combinations orthogonal to a random
    subset of the current pairing's templates in the block. `exclude` (pivot dict) drops candidates inside
    that span. scope "global": the block's kernel over its coordinates plus every coded coordinate (the shared
    admission); "own": over its own coordinates only (where the shipped 6566cd395128 relations lie; the same as
    "global" on generic loaders)."""
    qpiv, Q = dict(piv), []
    for r in (P.own_kernels[b] if scope == "own" else P.mixed_kernels[b] + P.struct_kernels[b]):
        rr = reduce_row(r, qpiv)
        if rr:
            qpiv[rr.bit_length() - 1] = rr
            Q.append(rr)
    cands = []
    if 0 < len(Q) <= enum_dim:
        for code in range(1, 1 << len(Q)):
            v = 0
            for i in set_bits(code):
                v ^= Q[i]
            cands.append(v)
    elif Q:
        for _ in range(pool):
            v = 0
            for r in Q:
                if rng.getrandbits(1):
                    v ^= r
            if v:
                cands.append(v)
    if protect and pairing is not None and len(Q) >= 2:
        temps = []
        for f, g, v in pairing[0]:
            for x in (f, g):
                if P.blk.get(x) == b:
                    temps.append(P.col(x, v))
        for f, v in pairing[1]:
            if P.blk.get(f) == b:
                temps.append(P.col(f, v))
        for _ in range(protect if temps else 0):
            S = rng.sample(temps, min(len(Q) - 1, len(temps)))
            eqs = [sum(((Q[i] & t).bit_count() & 1) << i for i in range(len(Q))) for t in S]
            ns = nullspace(eqs, len(Q), P.backend)
            v = 0
            for c in ns:
                if rng.getrandbits(1):
                    for i in set_bits(c):
                        v ^= Q[i]
            if v:
                cands.append(v)
    out, seen = [], set()
    for v in cands:
        if v in seen:
            continue
        seen.add(v)
        if exclude is not None and not reduce_row(v, exclude):
            continue
        out.append(v)
    return out


def kills_score(P, r, lines_, selected):
    """Weighted number of candidate lines a new relation r removes (selected lines weigh 3), the pairing-scored ranking."""
    sel = set(selected)
    w = 0
    for f, g, v in lines_:
        if ((r & P.col(f, v)).bit_count() & 1) or ((r & P.col(g, v)).bit_count() & 1):
            w += 3 if (f, g, v) in sel else 1
    return w


def evaluate(P, rels, rng, *, trials=2, cache=None, seed_lines=None, aug="quotient"):
    """(products, spaces, pairing, lines) of a relation set, or None if W is not spanned / the padded row cannot
    be completed / no pairing is found."""
    ok, sp = P.feasible(rels)
    if not ok:
        return None
    lt = P.lines(sp, rng, cache=cache)
    pr = P.pair_lines(sp, P.dims(rels), rng, trials=trials, lines=lt, seed_lines=seed_lines, aug=aug)
    if pr is None:
        return None
    return products_of(pr), sp, pr, lt


def exchange(P, rels, pr, rng, *, drops=(0, 1, 1, 2), cpu_cap=600.0, pool=40, protect=20, patience=6, enum_dim=8,
             trials=2, max_excess=1, neutral=True, max_stall=None, polish_trials=16, scope="own", log=None):
    """Exchange-move local search from a paired state (rels, pr). Each move: pick a block b and j in `drops`; drop a
    random codimension-j subspace of b's admitted relations; re-admit up to j+1 relations from b's kernel (the
    first one outside the dropped span), each the candidate with the best achieved pairing among the `patience`
    lowest-ranked by kills_score, kept only while products <= the state's products and products - ceil(dim W / 2)
    <= max_excess. A move ending with more relations is progress; one ending with as many relations and a new span
    is a neutral step (if `neutral`). A progress state whose pairing is not perfect is re-paired with
    `polish_trials` trials at once. Candidates come from the block's kernel in `scope` (block_candidates). Stops
    after cpu_cap CPU-s or max_stall moves without progress.
    Returns dict(best=(products, relations, pairing), final=(...), stats)."""
    t0 = time.process_time()
    cache = {}
    nb = len(P.blocks)
    max_stall = max_stall or 40 * nb

    def ideal(nrel):
        return (P.N - nrel + 1) // 2

    cur_rels, cur_pr = list(rels), pr
    cur_n = products_of(pr)
    ev = evaluate(P, cur_rels, rng, trials=1, cache=cache, seed_lines=cur_pr[0])
    cur_lines = ev[3] if ev else P.lines(P.feasible(cur_rels)[1], rng, cache=cache)
    best = (cur_n, list(cur_rels), cur_pr)
    seen = {canonical(cur_rels)}
    st = dict(moves=0, progress=0, neutral=0, rejected=0, evals=0, start=(cur_n, len(cur_rels)))
    stall = 0
    while time.process_time() - t0 < cpu_cap and stall < max_stall:
        b = rng.randrange(nb)
        j = rng.choice(drops)
        mine = block_relations(P, cur_rels, b)
        if scope == "own":
            mine = [r for r in mine if not r & ~P.block_masks[b]]
        if len(mine) < j:
            stall += 1
            continue
        st["moves"] += 1
        mine_set = set(mine)
        others = [r for r in cur_rels if r not in mine_set]
        mixed = recombine(mine, rng) if j else list(mine)
        dropped, kept = mixed[:j], mixed[j:]
        work = others + kept
        piv = {}
        for r in work:
            append_row(r, piv)
        excl = dict(piv)
        for r in dropped:
            append_row(r, excl)
        if j:
            evb = evaluate(P, work, rng, trials=1, cache=cache, seed_lines=cur_pr[0])
            st["evals"] += 1
            if evb is None:
                st["rejected"] += 1
                stall += 1
                continue
            work_n, _, work_pr, work_lines = evb
        else:
            work_n, work_pr, work_lines = cur_n, cur_pr, cur_lines
        target = len(cur_rels) + 1
        first = True
        while len(work) < target and time.process_time() - t0 < cpu_cap:
            cands = block_candidates(P, b, piv, rng, pool=pool, enum_dim=enum_dim,
                                     exclude=excl if first else None, protect=protect, pairing=work_pr,
                                     scope=scope)
            if not cands:
                break
            sel = list(work_pr[0])
            scored = sorted((kills_score(P, r, work_lines, sel), rng.random(), r) for r in cands)
            pick = None
            for _, _, r in scored[:patience]:
                trial = work + [r]
                e = evaluate(P, trial, rng, trials=trials, cache=cache, seed_lines=work_pr[0])
                st["evals"] += 1
                if e is None:
                    continue
                n = e[0]
                if n > cur_n or n - ideal(len(trial)) > max_excess:
                    continue
                key = (n - ideal(len(trial)), n)
                if pick is None or key < pick[0]:
                    pick = (key, r, e)
                    if key[0] <= 0:
                        break
            if pick is None:
                break
            _, r, (work_n, _, work_pr, work_lines) = pick
            work = work + [r]
            append_row(r, piv)
            first = False
        kind = None
        if len(work) > len(cur_rels) and work_n <= cur_n:
            kind = "progress"
        elif neutral and len(work) == len(cur_rels) and work_n <= cur_n and j:
            c = canonical(work)
            if c not in seen:
                seen.add(c)
                kind = "neutral"
        if kind is None:
            st["rejected"] += 1
            stall += 1
            continue
        st[kind] += 1
        if kind == "progress":
            stall = 0
            seen.add(canonical(work))
            if polish_trials and work_n > ideal(len(work)):
                e = evaluate(P, work, rng, trials=polish_trials, cache=cache, seed_lines=work_pr[0])
                st["evals"] += 1
                if e is not None and e[0] < work_n:
                    work_n, _, work_pr, work_lines = e
        cur_rels, cur_pr, cur_n, cur_lines = work, work_pr, work_n, work_lines
        if (cur_n, -len(cur_rels)) < (best[0], -len(best[1])):
            best = (cur_n, list(cur_rels), cur_pr)
        if log and kind == "progress":
            log(f"exchange {kind}: block {b}, drop {j}: {len(cur_rels)} relations, {cur_n} products "
                f"(ideal {ideal(len(cur_rels))}, {time.process_time() - t0:.0f} CPU-s)")
    # polish: re-pair the final state with more trials when its pairing is not perfect
    if polish_trials and cur_n > ideal(len(cur_rels)):
        e = evaluate(P, cur_rels, rng, trials=polish_trials, cache=cache, seed_lines=cur_pr[0])
        if e is not None and e[0] < cur_n:
            cur_n, cur_pr = e[0], e[2]
            if (cur_n, -len(cur_rels)) < (best[0], -len(best[1])):
                best = (cur_n, list(cur_rels), cur_pr)
    st["cpu_s"] = round(time.process_time() - t0, 1)
    st["final"] = (cur_n, len(cur_rels))
    st["best"] = (best[0], len(best[1]))
    return dict(best=best, final=(cur_n, cur_rels, cur_pr), stats=st)


def plan(delta, real_rows, coded_words=(), *, seed=0, blocks=None, partitions=0, keep=1, pad_free=True,
         coded_block=True, singles_forms="any", struct_drop=0, admit_kw=None, exchange_kw=None, log=None,
         backend=None):
    """Plan one loader. With `blocks` (slot lists) that partition is used; else `partitions` random triple
    partitions are scored structurally and the best `keep` go through shared admission (and, with `exchange_kw`,
    exchange moves from the admitted state). Returns a dict with terms, the rewritten full-block
    delta, products and diagnostics (exactness is checked by the caller)."""
    rng = random.Random(seed)
    P = SharedPlanner(delta, real_rows, coded_words, pad_free=pad_free, coded_block=coded_block,
                      singles_forms=singles_forms, backend=backend)
    cand = []
    tried = []
    exchange_stats = []
    if blocks is not None:
        part_list = [blocks]
    else:
        part_list, seen = [], set()
        while len(part_list) < partitions:
            b = random_triples(P.L, rng)
            key = tuple(sorted(tuple(x) for x in b))
            if key not in seen:
                seen.add(key)
                part_list.append(b)
    for b in part_list:
        res = structural(P, b, rng, struct_drop=struct_drop)
        tried.append(dict(blocks=[[s + 1 for s in x] for x in b], products=None if res is None else res[0]))
        if res is not None:
            cand.append((res[0], rng.random(), b, res))
    cand.sort(key=lambda x: (x[0], x[1]))
    best = None
    for n0, _, b, (n, rels, sp, pr) in cand[:keep]:
        P.setup(b)
        akw = dict(admit_kw or {})
        if exchange_kw is not None:
            akw.setdefault("prefer_more", True)
        n2, rels2, pr2 = admit_shared(P, rels, sp, pr, rng, log=log, **akw)
        if log:
            log(f"partition {[[s + 1 for s in x] for x in b]}: structural {n0}, shared {n2}")
        if exchange_kw is not None:
            ex = exchange(P, rels2, pr2, rng, log=log, **exchange_kw)
            exchange_stats.append(dict(blocks=[[s + 1 for s in x] for x in b], **ex["stats"]))
            if ex["best"][0] <= n2:
                n2, rels2, pr2 = ex["best"]
            if log:
                log(f"partition {[[s + 1 for s in x] for x in b]}: exchange {n2} ({len(rels2)} relations)")
        if best is None or n2 < best[0]:
            best = (n2, b, rels2, pr2)
    if best is None:
        return dict(products=None, tried=tried)
    n, b, rels, (pairs, singles) = best
    P.setup(b)
    rows, d, changes = P.complete(rels)
    terms = P.terms(rows, pairs, singles)
    out = dict(products=len(terms), pairs=len(pairs), singles=len(singles), relations=len(rels),
               blocks=[[s + 1 for s in x] for x in b], terms=[tuple(int(x) for x in t) for t in terms],
               delta=d, pad_changed_outputs=sorted(changes), tried=tried)
    if exchange_kw is not None:
        out["exchange"] = exchange_stats
    return out


def exact(tq, delta, terms, brot=8, lowbits=4):
    C = tq.anf_tensor(np.asarray(delta, np.int64), brot, lowbits)
    return bool(np.array_equal(tq.mixed_of(terms, C.shape[0] - 1, C.shape[1] - 1, C.shape[2]), C[1:, 1:, :]))


def exact_rows(delta, terms, brot=8, lowbits=4):
    """Exactness without Lera's module: the terms' mixed tensor equals the delta's mixed ANF rows."""
    rows, m, L = anf_rows(np.asarray(delta, np.int64), brot, lowbits)
    got = [0] * len(rows)
    for a1, b1, a2, b2, s in terms:
        for h in set_bits(b1):
            got[h] ^= column(a1, s, m)
        for h in set_bits(b2):
            got[h] ^= column(a2, s, m)
    return got == rows
