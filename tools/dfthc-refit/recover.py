# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Recover the plan structure of a shipped ANGLE_k loader from its circuit and the words it loads.

Works for any loader emitted in the cached-product style of Lera's (marculera) `tensorqrom.build_circuit` (two
monomial caches, then one CCX per product whose controls are a low-monomial form plus a high-monomial form): generic
and coded loaders, any table size, any family. Nothing here is specific to one table.

Steps (`recover_loader`):

1. **Terms.** The circuit is executed symbolically over GF(2): every qubit holds a set of atoms (address monomials,
   input word bits, opaque product atoms). A CCX onto a zero qubit with two single-monomial controls is a cache; any
   other CCX is a product, whose two controls must be pure-low + pure-high linear forms, which gives the term
   (a1, b1, a2, b2, s) in Lera's convention (a = low-form mask over low monomials 1..L, b = high-form mask over high
   monomials 1..H-1, s = the outputs receiving the product). A leading selection-only CX/X run undone by its mirror
   is read as an affine address frame. Scratch must end clean and the words must be restored (asserted).
2. **Function.** The recovered loader's truth table on every addressed row (the table rounded up to whole low
   blocks) is compared with the table delta on the real rows; the never-read padded rows are reported separately,
   since the emitter's planner may treat them as free.
3. **Structure** (`structure`, exact GF(2) algebra on the recovered terms): the feature space W (span of the
   rank-one (low form) x (output set) vectors), the number of relations (L*m - dim W), whether W contains every row
   of the recovered mixed ANF tensor, the coordinate blocks (connected components of the features' (low, output)
   supports), relations per block against the block's full row annihilator, the low-monomial and output partitions,
   the pairing (which blocks each product's two features lie in), products shared between coded and non-coded
   outputs, and per-block rank-2 product counts.

The re-emission (`reemit`) rebuilds the recovered terms with Lera's unmodified emitter and runs the native check.

Credit: the structure census generalises earlier single-family analyses, which first found Syed Asad Ali's block
partition. Lera's emitter is loaded unmodified by path.

Usage (kickmix interpreter, isolated mode):

    python -I recover.py LOADER.kmb WORDS.npy K --lera-src DIR [--emit OUT_DIR] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

WORD_ATOM = 1 << 12      # atom base for input word bits (address monomials use 0 .. 2**11 - 1)
PROD_ATOM = 1 << 13      # atom base for product atoms


# ----------------------------------------------------------------------------------------------- GF(2) helpers

def set_bits(x: int):
    while x:
        b = (x & -x).bit_length() - 1
        yield b
        x &= x - 1


# GF(2) backend: "python" (the reference, this module's code) or "rust" (kickmix_analysis, crate
# kmx-analysis in the engine repository; same pivots, reduced vectors and combination masks, byte-identical plans).
# The default is "auto" (after a bit-for-bit parity check): rust when kickmix_analysis imports, python otherwise.
# DFTHC_GF2_BACKEND (or the CLI's --gf2-backend, which sets it for worker processes too) overrides the default.
GF2_BACKENDS = ("python", "rust", "auto")
GF2_BACKEND = os.environ.get("DFTHC_GF2_BACKEND", "auto")
_KA = None
_AUTO = None


def gf2_backend(backend=None) -> str:
    """Resolve a backend choice to "python" or "rust"; "auto" picks rust only if kickmix_analysis loads."""
    global _AUTO
    b = backend or GF2_BACKEND
    if b not in GF2_BACKENDS:
        raise ValueError(f"GF(2) backend must be one of {GF2_BACKENDS}, not {b!r}")
    if b == "auto":
        if _AUTO is None:
            try:
                rust_module()
                _AUTO = "rust"
            except ImportError:
                _AUTO = "python"
        return _AUTO
    return b


def rust_module():
    """kickmix_analysis with the GF(2) objects. Found on sys.path (the analysis venv has a .pth) or in
    KICKMIX_ANALYSIS_PATH (a directory holding the kickmix_analysis package; read even under python -I)."""
    global _KA
    if _KA is None:
        extra = os.environ.get("KICKMIX_ANALYSIS_PATH")
        if extra and extra not in sys.path:
            sys.path.append(extra)
        try:
            import kickmix_analysis as ka
        except ImportError as e:
            raise ImportError("backend='rust' needs kickmix_analysis (crate kmx-analysis): run with "
                              "$QHDL_ANALYSIS_PYTHON (an interpreter with the engine analysis extension) or set "
                              "KICKMIX_ANALYSIS_PATH to .../crates/kmx-analysis/python") from e
        if not hasattr(ka, "Gf2Basis"):
            raise ImportError(f"kickmix_analysis at {ka.__file__} has no Gf2Basis (needs a build with the GF(2) objects)")
        _KA = ka
    return _KA


def make_basis(width, backend=None):
    """A Basis (python) or kickmix_analysis.Gf2Basis(width) (rust): add, reduce, solve, len, vectors."""
    if gf2_backend(backend) == "rust":
        return rust_module().Gf2Basis(int(width))
    return Basis()


class Basis:
    """Incremental GF(2) row echelon basis over Python ints, with coordinates (solve) in the added vectors."""

    def __init__(self):
        self.piv = {}            # pivot bit -> (reduced vector, combination mask over added vectors)
        self.vectors = []

    def reduce(self, v):
        c = 0
        while v:
            p = v.bit_length() - 1
            if p not in self.piv:
                break
            pv, pc = self.piv[p]
            v ^= pv
            c ^= pc
        return v, c

    def add(self, v) -> bool:
        r, c = self.reduce(v)
        if not r:
            return False
        i = len(self.vectors)
        self.vectors.append(v)
        self.piv[r.bit_length() - 1] = (r, c ^ (1 << i))
        return True

    def solve(self, v):
        """Mask of added vectors summing to v, or None if v is outside the span."""
        # full reduction (pivots by descending bit)
        c = 0
        while v:
            p = v.bit_length() - 1
            if p not in self.piv:
                return None
            pv, pc = self.piv[p]
            v ^= pv
            c ^= pc
        return c

    def __len__(self):
        return len(self.vectors)


def rank(vectors, backend=None) -> int:
    if gf2_backend(backend) == "rust":
        return rust_module().gf2_rank(vectors)
    B = Basis()
    for v in vectors:
        B.add(v)
    return len(B)


def nullspace(rows, width, backend=None):
    """Basis of {x in GF(2)^width : parity(x & r) = 0 for every r}."""
    if gf2_backend(backend) == "rust":
        return rust_module().gf2_nullspace(rows, width)
    piv = {}
    for r in rows:
        v = r
        for p in sorted(piv, reverse=True):
            if v >> p & 1:
                v ^= piv[p]
        if v:
            p = v.bit_length() - 1
            for q in list(piv):
                if piv[q] >> p & 1:
                    piv[q] ^= v
            piv[p] = v
    free = [i for i in range(width) if i not in piv]
    out = []
    for f in free:
        x = 1 << f
        for p, v in piv.items():
            if v >> f & 1:
                x |= 1 << p
        out.append(x)
    return out


# ----------------------------------------------------------------------------------------------- circuit reading

def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_kmx_text(text: str):
    """(registers {name: [qubit ids]}, ops [[op, args...]]) from kickmix .kmx text."""
    regs, pending, ops = {}, [], []
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "APPEND_TO_REGISTER":
            pending.append(int(t[1][1:]))
        elif t[0] == "REGISTER":
            regs[t[2].strip('"')] = pending
            pending = []
        else:
            ops.append(t)
    return regs, ops


def read_loader(km, path, tmp_dir=None):
    """Read a .kmb/.kmx loader with kickmix; returns (circuit, registers, ops)."""
    path = Path(path)
    fmt = "kmb" if path.suffix == ".kmb" else "kmx"
    circ = km.Circuit.from_file(str(path), format=fmt)
    if fmt == "kmx":
        text = path.read_text()
    else:
        import tempfile
        with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
            p = Path(td) / "loader.kmx"
            circ.to_file(str(p), format="kmx")
            text = p.read_text()
    regs, ops = parse_kmx_text(text)
    return circ, regs, ops


def split_frame(regs, ops):
    """Leading selection-only CX/X run undone by its mirrored trailing run -> (frame, core ops)."""
    selset = set(regs["selection"])

    def is_sel(t):
        return t[0] in ("CX", "X") and all(int(x[1:]) in selset for x in t[1:])
    nf = 0
    while nf < len(ops) and is_sel(ops[nf]):
        nf += 1
    frame = ops[:nf]
    if nf:
        assert ops[len(ops) - nf:] == frame[::-1], "selection frame not undone by its mirror"
        ops = ops[nf:len(ops) - nf]
    return frame, ops


def run_symbolic(regs, ops, brot=8, lowbits=4, gauge_in=None):
    """Symbolic GF(2) execution. Returns (word registers, final contents, caches, products). `gauge_in` (v1.5): per word, the incoming gauge p_in(low address bits) as a mask of low-monomial atoms; the top bit then starts
    as bit brot-2 XOR p_in (pi gauge) instead of a copy of bit brot-2 (signed gauge)."""
    sel = regs["selection"]
    nwords = len([r for r in regs if r.startswith("word_")])
    words = [regs[f"word_{j}"] for j in range(nwords)]
    content = {}
    for i, q in enumerate(sel):
        content[q] = 1 << (1 << i)
    for w in words:
        for b, q in enumerate(w):
            src = w[brot - 2] if b == brot - 1 else q          # signed gauge: top bit = bit brot-2
            content[q] = 1 << (WORD_ATOM + src)
    if gauge_in is not None:
        for w, g in zip(words, gauge_in):
            content[w[brot - 1]] ^= int(g)                   # pi gauge: XOR p_in(low bits)
    caches, products = [], []
    mono_only = (1 << WORD_ATOM) - 1
    for idx, t in enumerate(ops):
        op = t[0]
        if op in ("R", "HMR"):
            content[int(t[1][1:])] = 0
        elif op == "CZ":
            pass
        elif op == "X":
            q = int(t[1][1:])
            content[q] = content.get(q, 0) ^ 1          # atom 0 = monomial 0 = constant 1
        elif op == "CX":
            a, b = int(t[1][1:]), int(t[2][1:])
            content[b] = content.get(b, 0) ^ content.get(a, 0)
        elif op == "CCX":
            a, b, c = (int(x[1:]) for x in t[1:4])
            ca, cb = content.get(a, 0), content.get(b, 0)
            assert ca & ~mono_only == 0 and cb & ~mono_only == 0, f"op {idx}: control holds non-monomial atoms"
            if content.get(c, 0) == 0 and ca.bit_count() == 1 and cb.bit_count() == 1:
                m = (ca.bit_length() - 1) | (cb.bit_length() - 1)
                content[c] = 1 << m
                caches.append({"op": idx, "qubit": c, "mask": m, "low": m < (1 << lowbits)})
            else:
                i = len(products)
                products.append({"op": idx, "f1": list(set_bits(ca)), "f2": list(set_bits(cb)), "target": c})
                content[c] = content.get(c, 0) ^ (1 << (PROD_ATOM + i))
        else:
            raise ValueError(f"unknown op {op}")
    return sel, words, content, caches, products


def _form_split(f, lowbits):
    L = 1 << lowbits
    lo = hi = 0
    for m in f:
        ml, mh = m & (L - 1), m >> lowbits
        if ml and mh:
            raise ValueError("mixed monomial in a product form")
        if ml:
            lo |= 1 << (ml - 1)
        elif mh:
            hi |= 1 << (mh - 1)
        else:
            raise ValueError("constant in a product form")
    return lo, hi


def recover_terms(regs, ops, brot=8, lowbits=4, gauge_in=None):
    """Lera-convention terms (a1, b1, a2, b2, s) plus the final linear parts and the selection frame. `gauge_in`: see
    run_symbolic; the top bit's final linear part is then the delta's (the incoming p_in removed)."""
    frame, core = split_frame(regs, ops)
    if gauge_in is not None and any(gauge_in) and frame:
        raise ValueError("an incoming pi gauge with a selection frame is not supported")
    sel, wregs, content, caches, products = run_symbolic(regs, core, brot, lowbits, gauge_in)
    outs_of = [[] for _ in products]
    final_lin = {}
    for j, w in enumerate(wregs):
        for b, q in enumerate(w):
            c = content[q]
            if gauge_in is not None and b == brot - 1:
                c ^= int(gauge_in[j])
            final_lin[(j, b)] = c & ((1 << WORD_ATOM) - 1)
            wa = (c >> WORD_ATOM) & ((1 << (PROD_ATOM - WORD_ATOM)) - 1)
            src = w[brot - 2] if b == brot - 1 else q
            assert wa == 1 << src, f"word {j} bit {b}: input not restored as identity"
            for i in set_bits(c >> PROD_ATOM):
                outs_of[i].append((j, b))
    dirty = [q for q in content if content[q] and q not in sel and all(q not in w for w in wregs)]
    assert not dirty, f"scratch not clean: {dirty[:5]}"
    terms, top_ok = [], True
    for i, p in enumerate(products):
        os_ = set(outs_of[i])
        for (j, b) in os_:
            if b == brot - 1 and (j, brot - 2) not in os_:
                top_ok = False
        a1, b2 = _form_split(p["f1"], lowbits)
        a2, b1 = _form_split(p["f2"], lowbits)
        s = sum(1 << (j * (brot - 1) + b) for (j, b) in os_ if b < brot - 1)
        terms.append((a1, b1, a2, b2, s))
    low_groups, cur, last = [], [], None
    events = sorted([(c["op"], "c", c["mask"]) for c in caches if c["low"]] +
                    [(p["op"], "p", i) for i, p in enumerate(products)])
    for _, kind, v in events:
        if kind == "c":
            if last == "p" and cur:
                low_groups.append(cur)
                cur = []
            cur.append(v)
        last = kind
    if cur:
        low_groups.append(cur)
    return dict(terms=terms, frame=frame, products=products, caches=caches, final_lin=final_lin, outs_of=outs_of,
                n_words=len(wregs), low_groups=low_groups, top_bit_follows=top_ok, selection=regs["selection"])


def truth_table(rec, n_pts, brot=8):
    """Recovered function on addresses 0..n_pts-1 (in the core's address coordinates): (n_pts, J) words."""
    x = np.arange(n_pts)

    def form_val(f):
        v = np.zeros(n_pts, np.uint8)
        for m in f:
            v ^= ((x & m) == m).astype(np.uint8)
        return v
    pv = [form_val(p["f1"]) & form_val(p["f2"]) for p in rec["products"]]
    J = rec["n_words"]
    out = np.zeros((n_pts, J), np.int64)
    member = {}
    for i, os_ in enumerate(rec["outs_of"]):
        for jb in os_:
            member.setdefault(jb, []).append(i)
    for j in range(J):
        for b in range(brot):
            v = form_val(list(set_bits(rec["final_lin"][(j, b)])))
            for i in member.get((j, b), []):
                v = v ^ pv[i]
            out[:, j] |= v.astype(np.int64) << b
    return out


def frame_map(frame, selection, n_pts):
    """y = F(x) for the affine address frame (identity when there is none)."""
    pos = {q: i for i, q in enumerate(selection)}
    v = np.arange(n_pts)
    for t in frame:
        if t[0] == "X":
            v = v ^ (1 << pos[int(t[1][1:])])
        else:
            c, d = pos[int(t[1][1:])], pos[int(t[2][1:])]
            v = v ^ (((v >> c) & 1) << d)
    return v


# ----------------------------------------------------------------------------------------------- tensor model

def anf_rows(table, brot=8, lowbits=4):
    """Mixed ANF rows of a full-block table: one int per high monomial 1..H-1, bit (l-1)*m + o for low monomial l,
    output o = j*(brot-1) + bit (equal to Lera's anf_tensor[1:, 1:, :])."""
    table = np.asarray(table, np.int64)
    A, J = table.shape
    L = 1 << lowbits
    assert A % L == 0
    H = A // L
    m = J * (brot - 1)
    bits = np.stack([(table[:, j] >> b) & 1 for j in range(J) for b in range(brot - 1)], axis=1).astype(np.uint8)
    t = bits.reshape(H, L, m)
    for axis, n in ((0, H), (1, L)):
        t = np.swapaxes(t, axis, 0).copy()
        ids = np.arange(n)
        for bit in range(max(1, (n - 1).bit_length())):
            rows = ids[(ids & (1 << bit)) != 0]
            rows = rows[(rows ^ (1 << bit)) < n]
            t[rows] ^= t[rows ^ (1 << bit)]
        t = np.swapaxes(t, axis, 0)
    tensor = t[1:, 1:, :]
    return [int(sum(int(v) << i for i, v in enumerate(tensor[h].reshape(-1)))) for h in range(H - 1)], m, L - 1


def column(form, v, m):
    """Rank-one vector (low form) x (output vector v) in the (L*m)-bit coordinate space."""
    return sum(v << (j * m) for j in set_bits(form))


def coded_output_mask(coded_words, brot=8):
    return sum(1 << (j * (brot - 1)) for j in coded_words)


def structure(terms, rows, m, L, coded_outputs=0, brot=8, backend=None):
    """Exact census of a recovered plan (see module docstring). `coded_outputs` = output mask of the coded bit-0
    outputs (bit j*(brot-1) for a coded word j)."""
    feats = []
    for a1, b1, a2, b2, s in terms:
        feats.append((a1, s))
        if a2:
            feats.append((a2, s))
    W = make_basis(L * m, backend)
    for a, s in feats:
        W.add(column(a, s, m))
    dimW = len(W)
    rows_in_W = all(W.solve(r) is not None for r in rows)
    # coordinate blocks: union-find over (slot, output) touched by each feature
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for a, s in feats:
        coords = [l * m + o for l in set_bits(a) for o in set_bits(s)]
        for c in coords[1:]:
            union(coords[0], c)
        if coords:
            find(coords[0])
    comps = {}
    for c in list(parent):
        comps.setdefault(find(c), set()).add(c)
    blocks = sorted(comps.values(), key=lambda cs: min(cs))
    block_of = {c: bi for bi, cs in enumerate(blocks) for c in cs}
    all_coords = set(range(L * m))
    uncovered = sorted(all_coords - set(block_of))
    binfo = []
    for bi, cs in enumerate(blocks):
        cmask = sum(1 << c for c in cs)
        proj = rank([w & cmask for w in W.vectors], backend)
        rk = rank([r & cmask for r in rows], backend)
        slots = sorted({c // m for c in cs})
        outs = sorted({c % m for c in cs})
        full_product = len(cs) == len(slots) * len(outs)
        binfo.append(dict(block=bi, coords=len(cs), low_monomials=[s + 1 for s in slots], outputs=outs,
                          coded_outputs_only=all((coded_outputs >> o) & 1 for o in outs),
                          rectangle=full_product, dimW=proj, relations=len(cs) - proj,
                          row_rank=rk, annihilator_dim=len(cs) - rk, full_annihilator=(len(cs) - proj) == (len(cs) - rk)))
    block_sum = sum(b["dimW"] for b in binfo)
    # product pairing by blocks
    pair_kind = {"cross_block": 0, "same_block": 0, "single": 0}
    per_block_products = [0] * len(blocks)
    shared = {"coded_only": 0, "noncoded_only": 0, "mixed": 0}
    rank2 = [0] * len(blocks)
    for a1, b1, a2, b2, s in terms:
        b_1 = block_of.get((next(set_bits(a1))) * m + next(set_bits(s))) if a1 and s else None
        if a2:
            b_2 = block_of.get((next(set_bits(a2))) * m + next(set_bits(s))) if s else None
            if b_1 == b_2:
                pair_kind["same_block"] += 1
                rank2[b_1] += 1
            else:
                pair_kind["cross_block"] += 1
            for b in {b_1, b_2}:
                if b is not None:
                    per_block_products[b] += 1
        else:
            pair_kind["single"] += 1
            if b_1 is not None:
                per_block_products[b_1] += 1
        if s & coded_outputs and s & ~coded_outputs:
            shared["mixed"] += 1
        elif s & coded_outputs:
            shared["coded_only"] += 1
        else:
            shared["noncoded_only"] += 1
    for bi in range(len(blocks)):
        binfo[bi]["products_touching"] = per_block_products[bi]
        binfo[bi]["rank2_within_block"] = rank2[bi]
    # low-monomial partition (components of the low forms' supports) and output partition (of the output sets)
    lowpar = _components([a for a, _ in feats])
    outpar = _components([s for _, s in feats])
    forms = sorted({a for a, _ in feats})
    # cells: (low-partition block) x (non-coded / coded outputs), the write-up's "coded LSB block" view
    full_mask = (1 << m) - 1
    classes = [("noncoded", full_mask & ~coded_outputs)] + ([("coded", coded_outputs)] if coded_outputs else [])
    cells, cell_sum = [], 0
    for lp in lowpar:
        for name, omask in classes:
            cmask = column(lp, omask, m)
            proj = rank([w & cmask for w in W.vectors], backend)
            rk = rank([r & cmask for r in rows], backend)
            # coordinates where some row is nonzero (the rest are trivially annihilated)
            live = 0
            for r in rows:
                live |= r & cmask
            cell_sum += proj
            cells.append(dict(low_monomials=[s + 1 for s in set_bits(lp)], outputs=name,
                              coords=cmask.bit_count(), live_coords=live.bit_count(), dimW=proj, row_rank=rk,
                              relations=cmask.bit_count() - proj, annihilator_dim=cmask.bit_count() - rk,
                              live_annihilator_dim=live.bit_count() - rk,
                              live_relations=live.bit_count() - proj))
    # products by the cells their features fall in
    cell_of = {}
    for ci, c in enumerate(cells):
        lp = sum(1 << (s - 1) for s in c["low_monomials"])
        for l in set_bits(lp):
            cls = 1 if c["outputs"] == "coded" else 0
            cell_of[(l, cls)] = ci

    def fcells(a, s):
        out = set()
        for l in set_bits(a):
            if s & ~coded_outputs:
                out.add(cell_of[(l, 0)])
            if s & coded_outputs:
                out.add(cell_of[(l, 1)])
        return out
    touching = [0] * len(cells)
    for a1, b1, a2, b2, s in terms:
        cs = fcells(a1, s) | (fcells(a2, s) if a2 else set())
        for c in cs:
            touching[c] += 1
    for ci, c in enumerate(cells):
        c["products_touching"] = touching[ci]
    return dict(products=len(terms), features=len(feats), dimW=dimW, relations=L * m - dimW,
                rows_in_W=rows_in_W, row_rank=rank(rows, backend), coordinate_blocks=binfo,
                blocks_direct_sum=block_sum == dimW, uncovered_coords=len(uncovered),
                low_partition=[[s + 1 for s in set_bits(c)] for c in lowpar],
                output_partition=[sorted(set_bits(c)) for c in outpar],
                distinct_low_forms=len(forms), max_low_form_weight=max((f.bit_count() for f in forms), default=0),
                distinct_output_sets=len({s for _, s in feats}), pairing=pair_kind, output_sharing=shared,
                cells=cells, cells_direct_sum=cell_sum == dimW,
                lower_bound_pairs=(dimW + 1) // 2)


def _components(masks):
    comps = []
    for x in masks:
        if not x:
            continue
        merged = x
        rest = []
        for c in comps:
            if c & merged:
                merged |= c
            else:
                rest.append(c)
        # repeat merge until stable
        changed = True
        while changed:
            changed = False
            nxt = []
            for c in rest:
                if c & merged:
                    merged |= c
                    changed = True
                else:
                    nxt.append(c)
            rest = nxt
        comps = rest + [merged]
    return sorted(comps, key=lambda c: (c & -c))


# ----------------------------------------------------------------------------------------------- driver

def recover_loader(km, kmb, words, k, *, J=5, brot=8, lowbits=4, gauge_in=None):
    """Recover loader k's terms and structure. `words` = the (A, n) words the family loads (official
    evaluate_circuits). Returns a JSON-able report plus the raw terms and recovered padded table. `gauge_in`
    (v1.5): (2**lowbits, J) incoming gauge p_in of a pi-gauge family (gauge.slot_gauge), None for the signed gauge."""
    import dfthc_refit as dr
    words = np.asarray(words, np.int64)
    A = len(words)
    L = 1 << lowbits
    H = -(-A // L)
    n_pts = L * H
    circ, regs, ops = read_loader(km, kmb)
    gmask = None
    if gauge_in is not None and np.asarray(gauge_in).any():
        anf = np.array(gauge_in, np.uint8)                     # ANF over the low monomials, per word (Moebius)
        bit = 1
        while bit < L:
            for a in range(L):
                if a & bit:
                    anf[a] ^= anf[a ^ bit]
            bit <<= 1
        gmask = [sum(1 << m for m in range(L) if anf[m, jj]) for jj in range(anf.shape[1])]
    rec = recover_terms(regs, ops, brot, lowbits, gmask)
    tt = truth_table(rec, n_pts, brot)
    ymap = frame_map(rec["frame"], rec["selection"], n_pts)
    assert sorted(ymap.tolist()) == list(range(n_pts)), "frame does not preserve the addressed range"
    # v1.8.1: the loader's own word count wins over the caller's default. J = 5 is Lera's layout;
    # families with J = 3 or 4 words per loader were compared against the wrong table slice before.
    if rec.get("n_words") and rec["n_words"] != J:
        J = int(rec["n_words"])
    delta, cw = dr.loader_delta(words, k, J)
    full = dr.full_block_delta(delta)
    real_ok = bool(np.array_equal(tt[ymap[:A]], delta[:A]))
    rec_x = tt[ymap]                       # recovered function in table address order
    pad_diff = [int(a) for a in range(A, n_pts) if not np.array_equal(rec_x[a], full[a])]
    rows, m, Lm = anf_rows(tt, brot, lowbits)      # rows of the core (frame coordinates)
    co = coded_output_mask(cw, brot)
    st = structure(rec["terms"], rows, m, Lm, co, brot)
    stats = circ.operation_statistics()
    report = dict(k=k, kmb=str(kmb), kmb_sha256=sha256_file(kmb), num_qubits=int(circ.num_qubits),
                  num_toffoli_like=int(stats.num_toffoli_like), caches=len(rec["caches"]),
                  low_caches=sum(c["low"] for c in rec["caches"]), products=len(rec["terms"]),
                  low_groups_in_order=rec["low_groups"], coded_words=cw, rows=A, addressed_rows=n_pts,
                  selection_frame=[" ".join(t) for t in rec["frame"]],
                  recovered_equals_delta_on_rows=real_ok, top_bit_follows=rec["top_bit_follows"],
                  padded_rows_differing_from_default_pad=pad_diff, structure=st)
    return report, rec, rec_x, np.asarray(delta[:A], np.int64)     # real rows only (coded deltas arrive padded)


def reemit(km, tq, rec, rec_x, delta, out_path, *, brot=8, lowbits=4, scratch=True, check_inputs=None):
    """Rebuild the recovered terms with Lera's unmodified emitter (low groups as recovered, the selection frame
    wrapped around the core) and run the native check on the real rows. With a free-source gauge emitter
    (`gauge.GaugeEmitter(..., free_sources=True)`, v1.8.0-dev) the low cache is the emitter's (the
    recovered groups are not passed), which re-emits partial low caches built from the incoming gauge monomials.
    `check_inputs` (the gauge check inputs, `gauge.check_inputs`) selects the gauge-aware native check."""
    import dfthc_refit as dr
    A = len(delta)
    n_sel = (A - 1).bit_length()
    ymap = frame_map(rec["frame"], rec["selection"], len(rec_x))
    dpad = np.zeros_like(rec_x)
    dpad[ymap] = rec_x                       # core-coordinate table
    # a non-signed gauge emitter builds its own low cache (full, or with free sources): recovered groups are not passed
    own_cache = getattr(tq, "p_in", None) is not None and not getattr(tq, "signed", True)
    rb = tq.build_circuit(dpad, [tuple(t) for t in rec["terms"]], n_sel, brot, lowbits,
                          groups=None if own_cache else [sorted(g) for g in rec["low_groups"]], scratch=scratch)
    out_path = Path(out_path)
    if rec["frame"]:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            core = Path(td) / "core.kmx"
            rb.to_file(str(core), format="kmx")
            lines = core.read_text().splitlines()
            head = [x for x in lines if x.startswith(("APPEND_TO_REGISTER", "REGISTER"))]
            body = [x for x in lines if x.strip() and not x.startswith(("APPEND_TO_REGISTER", "REGISTER"))]
            fl = [" ".join(t) for t in rec["frame"]]
            framed = Path(td) / "framed.kmx"
            framed.write_text("\n".join(head + fl + body + fl[::-1]) + "\n")
            rb = km.Circuit.from_file(str(framed), format="kmx")
    rb.to_file(str(out_path), format="kmb")
    rb = km.Circuit.from_file(str(out_path), format="kmb")
    if check_inputs is not None:
        import gauge as gz
        check = gz.native_check(km, rb, np.asarray(delta[:A], np.int64), check_inputs, brot)
    else:
        check = dr.native_check(km, rb, np.asarray(delta[:A], np.int64), brot)
    return dict(num_toffoli_like=int(rb.operation_statistics().num_toffoli_like), num_qubits=int(rb.num_qubits),
                native_check=check, kmb_sha256=sha256_file(out_path), scratch=scratch)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kmb")
    ap.add_argument("words", help="the words the family loads (.npy, e.g. from dfthc_refit.recover_words)")
    ap.add_argument("k", type=int)
    ap.add_argument("--lera-src", help="directory with Lera's tensorqrom.py (needed for --emit)")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--emit", help="directory to write the re-emitted loader into (created; must not exist)")
    ap.add_argument("--json", help="write the report here (must not exist)")
    a = ap.parse_args(argv)
    import dfthc_refit as dr
    km = dr.find_kickmix(a.kickmix_dir)
    if km is None:
        raise SystemExit("kickmix not found")
    words = np.load(a.words, allow_pickle=False)
    report, rec, rec_x, delta = recover_loader(km, a.kmb, words, a.k)
    report["terms"] = [list(map(int, t)) for t in rec["terms"]]
    if a.emit:
        tq, tq_sha = dr._load_tensorqrom(a.lera_src)
        out = Path(a.emit)
        out.mkdir(parents=True, exist_ok=False)
        report["reemit"] = reemit(km, tq, rec, rec_x, delta, out / f"angle_k_{a.k}.kmb")
        report["reemit"]["tensorqrom_sha256"] = tq_sha
    text = json.dumps(report, indent=1) + "\n"
    if a.json:
        p = Path(a.json)
        assert not p.exists(), f"refusing to overwrite {p}"
        p.write_text(text)
    summary = {k: v for k, v in report.items() if k != "terms"}
    print(json.dumps(summary))
    return report


if __name__ == "__main__":
    main()
