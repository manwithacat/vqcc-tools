# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Exact word symmetries of a Givens-tree angle table and a joint search over them with the loader planners
(dfthc_refit 1.5.0-dev add-on).

The official decoder maps the words w_e of one row (theta_e / pi = w_e / 2**(brot-1)) on a Givens tree (edges
(p_e, c_e) in order, applied to e_0) to a vector u; only u u^T enters the score. Two word moves per column e:

- **pi flip** h_e: w_e -> w_e + 2**(brot-1) (top bit flipped). R(theta + pi) = -R(theta) on (p_e, c_e), so the flip
  multiplies by -1 every component that descends from p_e from edge e on: u -> D_h(e) u, a sign diagonal.
- **negation** g_c (one per node c != 0): w -> -w on the edge that creates c and on every later edge that touches c.
  R(-theta) = D_c R(theta) D_c and c is still zero when its edge runs, so g_c gives u -> D_c u (component c only).

Both compose by XOR (an abelian group Z_2^(2E)), u -> D(n, f) u with D linear in (n, f). A move is an exact symmetry
(u u^T unchanged on every row, whatever the angles) iff D(n, f) is +I or -I. On a tree with E edges:

- every pi-flip pattern f has exactly one negation set n(f) that completes it to a symmetry (the D_g are unit
  vectors on nodes 1..E, so n(f) = D_g^-1 of D_h(f), complemented when node 0 is flipped): the group is 2**E per row;
- the pure pi-gauge symmetries (n(f) = 0) are the kernel of f -> D_h(f) mod +-I.

The planners' tensor (`recover.anf_rows`, Lera's `anf_tensor`) reads bits 0..brot-2 of each XOR delta, and a pi
flip touches bit brot-1 only, so a pure pi-gauge choice cannot change any plan's product count (Toffolis). A
negation changes bits 1..brot-2 (never bit 0: -w = w mod 2): -w XOR w = the bits above w's lowest set bit.

The gauge-aware emitter (`gauge.build_circuit_gauge`) needs the final top-bit pattern (bit 7 XOR bit 6) to be a
function of the low address bits per word. Searching inside that constraint: per low address l (16 slices) choose
f_l, i.e. the negation set n_l = n(f_l), for every row of the slice. The search scores a slot by the planners'
structural lower bound: for a block partition of the low monomials, sum over blocks of the GF(2) rank of the mixed
ANF rows restricted to the block's coordinates (W restricted to a block contains those rows; products >=
dim W / 2). The last high row is left out (it holds the never-read padded addresses, which the planners complete).

Credit: the table is Syed Asad Ali's words in Atakan Aldemir's pi-gauge tree and loaders (family aba448aeadb5),
with Heliosli's weights and Akmal Xodarev's refit, integrated by Kirill Tymchishin; the emitter model is Lera's
(marculera) tensorqrom. The symmetry analysis and search are this tool's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gauge as gz  # noqa: E402
from recover import anf_rows  # noqa: E402

BROT = 8
J = 5
LOWBITS = 4


# ------------------------------------------------------------------------------------------- the symmetry group


def gf2_rank_rows(vs) -> int:
    """GF(2) rank of a list of Python ints."""
    piv = {}
    r = 0
    for v in vs:
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p]
            else:
                piv[p] = v
                r += 1
                break
    return r


def _bits_to_int(a) -> int:
    return int(sum(1 << i for i, x in enumerate(np.asarray(a).ravel()) if x))


def apply_move(words, n, f, brot=BROT):
    """Words with negation set n and pi-flip set f (0/1 arrays broadcastable to words), unsigned storage."""
    w = np.asarray(words, np.int64)
    M = 1 << brot
    w = np.where(np.asarray(n, bool), (-w) % M, w)
    return np.where(np.asarray(f, bool), w ^ (1 << (brot - 1)), w)


def sign_diagonal(words_a, words_b, edges, brot=BROT, tol=1e-9):
    """(rows, N) 0/1 sign pattern d with u_b = (-1)^d u_a on every component (None if not a sign diagonal)."""
    ua = gz.decode(words_a, edges, brot)
    ub = gz.decode(words_b, edges, brot)
    if np.abs(np.abs(ua) - np.abs(ub)).max() > tol:
        return None
    return (np.sign(ua) != np.sign(ub)).astype(np.uint8)


def group(edges, brot=BROT, *, samples=64, seed=0) -> dict:
    """The exact symmetry group of a Givens tree, computed from generators on generic words and checked.

    Returns n_g (E, E): negation set of g_c for c = 1..E (row c-1); D_g (E, N); D_h (E, N): sign pattern of the pi
    flip on each column; Nf (E, E) the linear map f -> n(f) (n = f @ Nf mod 2); dims."""
    E = np.asarray(edges, np.int64)
    nE, N = len(E), len(E) + 1
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 1 << brot, size=(samples, nE))
    G = np.where(G % (1 << (brot - 2)) == 0, G + 7, G)          # generic: no zero component anywhere
    child_edge = {int(c): i for i, (_, c) in enumerate(E)}
    n_g = np.zeros((nE, nE), np.uint8)
    D_g = np.zeros((nE, N), np.uint8)
    for c in range(1, N):
        e = child_edge[c]
        n_g[c - 1, e] = 1
        for i in range(e + 1, nE):
            if c in (int(E[i, 0]), int(E[i, 1])):
                n_g[c - 1, i] ^= 1
        d = sign_diagonal(G, apply_move(G, n_g[c - 1], 0, brot), E, brot)
        assert d is not None and (d == d[0]).all(), f"g_{c} is not a sign diagonal"
        D_g[c - 1] = d[0]
    D_h = np.zeros((nE, N), np.uint8)
    for e in range(nE):
        f = np.zeros(nE, np.uint8)
        f[e] = 1
        d = sign_diagonal(G, apply_move(G, 0, f, brot), E, brot)
        assert d is not None and (d == d[0]).all(), f"h_{e} is not a sign diagonal"
        D_h[e] = d[0]
    # D_g must be the unit vectors on nodes 1..E (then n(f) is unique)
    assert np.array_equal(D_g[:, 1:], np.eye(nE, dtype=np.uint8)) and not D_g[:, 0].any()
    # f -> n(f): D_h(f), complemented when node 0 is flipped, restricted to nodes 1..E, through D_g^-1 = n_g rows
    Dc = D_h ^ D_h[:, :1]                                   # complement rows whose node-0 bit is set
    Nf = (Dc[:, 1:].astype(np.int64) @ n_g.astype(np.int64)) % 2
    # check: random f, its n(f) leaves u u^T unchanged
    for t in range(16):
        f = rng.integers(0, 2, nE).astype(np.uint8)
        n = (f.astype(np.int64) @ Nf) % 2
        d = sign_diagonal(G, apply_move(G, n, f, brot), E, brot)
        assert d is not None and all(row.all() or not row.any() for row in d), f"n(f) check {t}"
    rank_h = gf2_rank_rows([_bits_to_int(r) for r in D_h])
    rank_hp = gf2_rank_rows([_bits_to_int(r) for r in D_h] + [(1 << N) - 1])
    rank_n = gf2_rank_rows([_bits_to_int(r) for r in Nf])
    return dict(n_g=n_g, D_g=D_g, D_h=D_h, Nf=Nf.astype(np.uint8), edges=nE, nodes=N,
                dims=dict(group=nE, pure_pi=nE - rank_hp + 1, negation_projection=rank_n,
                          rank_D_h=rank_h, ones_in_D_h=rank_h == rank_hp))


def projector_change(words_a, words_b, edges, brot=BROT) -> float:
    ua = gz.decode(words_a, edges, brot)
    ub = gz.decode(words_b, edges, brot)
    return float(np.abs(ua[:, :, None] * ua[:, None, :] - ub[:, :, None] * ub[:, None, :]).max())


# ------------------------------------------------------------------------------------- slot tensors and bounds


def slot_rows(words, k, coded_fn, j=J, brot=BROT, lowbits=LOWBITS):
    """Mixed ANF rows of slot k's delta (full-block padded with zeros; coded bit 0 affine-padded) as in replan."""
    delta, cw = coded_fn(words, k, j)
    A = len(delta)
    L = 1 << lowbits
    full = np.zeros((-(-A // L) * L, delta.shape[1]), np.int64)
    full[:A] = delta
    rows, m, Lm = anf_rows(full, brot, lowbits)
    return rows, m, Lm, cw


def block_masks(blocks, m):
    """Coordinate masks (bit (l-1)*m + o) of each block of low monomials (1-based)."""
    out = []
    for b in blocks:
        mk = 0
        for l in b:
            mk |= ((1 << m) - 1) << ((l - 1) * m)
        out.append(mk)
    return out


def block_bound(rows, masks, drop_last=True) -> int:
    """Sum over blocks of rank(rows restricted to the block): a lower bound on dim W (products >= half of it)."""
    rs = rows[:-1] if drop_last else rows
    return sum(gf2_rank_rows([r & mk for r in rs]) for mk in masks)


SYED_BLOCKS = [[1, 5, 13], [2, 6, 14], [3, 7, 15], [4, 8, 11], [9, 10, 12]]


# ------------------------------------------------------------------------------------------------ the search


def slice_table(words, Nf, F, brot=BROT, lowbits=LOWBITS):
    """Apply per-slice moves: F is a (2**lowbits, E) 0/1 array of pi flips f_l; every row of slice l (row mod
    2**lowbits) gets the negation set n(f_l) = f_l @ Nf and the pi flips f_l. An exact symmetry on every row."""
    F = np.asarray(F, np.int64)
    n = (F @ np.asarray(Nf, np.int64)) % 2
    low = np.arange(len(words)) % (1 << lowbits)
    return apply_move(words, n[low], F[low], brot)


def allowed_spaces(words, Nf, brot=BROT, lowbits=LOWBITS):
    """Per slice l: a basis (rows) of the pi-flip vectors f_l whose negation set n(f_l) avoids every column holding
    a word of value 2**(brot-2) or 3 * 2**(brot-2) in that slice (negating those flips their gauge bit, which would
    make the top-bit pattern depend on the high address). Every f in these spaces keeps the pattern low-only."""
    q = 1 << (brot - 2)
    bad = (np.asarray(words, np.int64) % (2 * q)) == q
    low = np.arange(len(words)) % (1 << lowbits)
    out = []
    for l in range(1 << lowbits):
        forb = bad[low == l].any(axis=0)
        if forb.any():
            out.append(np.array(_kernel_basis(np.asarray(Nf, np.int64)[:, forb]), np.int64))
        else:
            out.append(np.eye(Nf.shape[0], dtype=np.int64))
    return out


def random_variant(words, Nf, spaces, rng, brot=BROT, lowbits=LOWBITS):
    """(F, words'): a uniformly random element of the per-slice allowed spaces, applied."""
    F = np.stack([(rng.integers(0, 2, len(B)) @ B) % 2 if len(B) else np.zeros(Nf.shape[0], np.int64)
                  for B in spaces])
    return F, slice_table(words, Nf, F, brot, lowbits)


def structural_products(words, k, blocks, coded_fn, seeds=(0,), trials=2, j=J):
    """planner_shared steps 1-3, 5, 6 (structural annihilator, pairing) for slot k with a fixed partition
    (1-based low monomials); one product count per rng seed (None if infeasible)."""
    import planner_shared as ps
    delta, cw = coded_fn(words, k, j)
    P = ps.SharedPlanner(delta, len(words), cw, pad_free=True)
    b0 = [[x - 1 for x in b] for b in blocks]
    out = []
    for sd in seeds:
        res = ps.structural(P, b0, random.Random(sd), trials=trials)
        out.append(None if res is None else int(res[0]))
    return out


def gauge_ok(words, brot=BROT, lowbits=LOWBITS) -> bool:
    return gz.low_pattern(gz.off_gauge(words, brot), lowbits) is not None


def _coded_fn():
    import dfthc_refit as dr
    return dr.loader_delta


def load_family_words(npz):
    z = np.load(npz)
    return np.asarray(z["angle_words"], np.int64), np.asarray(z["tree_edges"], np.int64), int(z["brot"])


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_group(args) -> int:
    words, edges, brot = load_family_words(args.npz)
    t0 = time.time()
    g = group(edges, brot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    # pi-gauge invariance of every slot's planner tensor: shipped vs canonical vs random pure-pi per row
    coded = _coded_fn()
    can = gz.canonical_words(words, brot)
    rng = np.random.default_rng(args.seed)
    Nf = g["Nf"].astype(np.int64)
    # pure-pi kernel basis: f with n(f) = 0
    ker = _kernel_basis(Nf)
    R = np.zeros_like(words)
    for r in range(len(words)):
        coef = rng.integers(0, 2, len(ker))
        R[r] = (coef @ np.array(ker, np.int64)) % 2 if ker else 0
    rand_pi = apply_move(words, 0, R, brot)
    inv = []
    for k in range(-(-words.shape[1] // J)):
        r0 = slot_rows(words, k, coded)[0]
        inv.append(dict(k=k, canonical_equal=slot_rows(can, k, coded)[0] == r0,
                        random_pure_pi_equal=slot_rows(rand_pi, k, coded)[0] == r0))
    # shipped pattern group: which rows' patterns are pure-pi kernel elements
    off = gz.off_gauge(words, brot)
    pats = {tuple(np.flatnonzero(o)) for o in off}
    in_kernel = []
    for p in sorted(pats):
        f = np.zeros(words.shape[1], np.int64)
        f[list(p)] = 1
        in_kernel.append(dict(columns=[int(x) for x in p], n_of_f_zero=not ((f @ Nf) % 2).any()))
    # zero components per row (data-dependent extra symmetries; not used by the search)
    u = gz.decode(words, edges, brot)
    zeros = (np.abs(u) < 1e-12).sum(axis=1)
    rep = dict(tool="gauge_search", npz=str(args.npz), npz_sha256=sha256(args.npz), words_shape=list(words.shape),
               dims=g["dims"], pure_pi_kernel_basis=len(ker),
               max_projector_change=dict(canonical=projector_change(words, can, edges, brot),
                                         random_pure_pi=projector_change(words, rand_pi, edges, brot)),
               planner_tensor_invariance=inv, shipped_patterns=in_kernel,
               rows_with_zero_components=int((zeros > 0).sum()), zero_components_mean=float(zeros.mean()),
               words_parity_odd_share=float((words % 2).mean()),
               words_multiple_of_64_share=float((words % 64 == 0).mean()), wall_s=round(time.time() - t0, 2))
    np.savez(out / "group.npz", n_g=g["n_g"], D_g=g["D_g"], D_h=g["D_h"], Nf=g["Nf"])
    (out / "group.json").write_text(json.dumps(rep, indent=1, default=int) + "\n")
    print(json.dumps({k: rep[k] for k in ("dims", "pure_pi_kernel_basis", "max_projector_change")}, default=int))
    return 0


def _kernel_basis(Nf):
    """Basis (list of 0/1 int arrays) of {f : f @ Nf = 0 mod 2}."""
    E = Nf.shape[0]
    # row-reduce [Nf | I]
    M = np.concatenate([Nf % 2, np.eye(E, dtype=np.int64)], axis=1).astype(np.uint8)
    r = 0
    for c in range(Nf.shape[1]):
        piv = [i for i in range(r, E) if M[i, c]]
        if not piv:
            continue
        M[[r, piv[0]]] = M[[piv[0], r]]
        for i in range(E):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return [M[i, Nf.shape[1]:].astype(np.int64) for i in range(r, E)]


def cmd_bound(args) -> int:
    """Block bounds of every slot for given partitions (JSON {k: blocks}) and Syed Asad Ali's default."""
    words, edges, brot = load_family_words(args.npz)
    coded = _coded_fn()
    blocks = json.loads(Path(args.blocks).read_text()) if args.blocks else {}
    rep = []
    for k in range(-(-words.shape[1] // J)):
        rows, m, Lm, cw = slot_rows(words, k, coded)
        b = blocks.get(str(k), SYED_BLOCKS)
        rep.append(dict(k=k, coded_words=cw, blocks=b, bound=block_bound(rows, block_masks(b, m)),
                        bound_all_rows=block_bound(rows, block_masks(b, m), drop_last=False),
                        rank_rows=gf2_rank_rows(rows[:-1]), coords=Lm * m))
    print(json.dumps(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


def _variants(args, words, Nf):
    spaces = allowed_spaces(words, Nf)
    rng = np.random.default_rng(args.variant_seed)
    out = [("shipped", np.zeros((16, Nf.shape[0]), np.int64), words)]
    for v in range(args.variants):
        F, w = random_variant(words, Nf, spaces, rng)
        out.append((f"v{v}", F, w))
    return out


def cmd_structural(args) -> int:
    """Structural product counts (planner_shared steps 1-3, 5, 6; fixed partition) of the coded slots on the shipped
    table and on random per-slice symmetry variants."""
    words, edges, brot = load_family_words(args.npz)
    Nf = np.load(args.group)["Nf"].astype(np.int64)
    coded = _coded_fn()
    blocks = json.loads(args.blocks_json)
    slots = [int(x) for x in args.slots.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rep = []
    t0 = time.time()
    for name, F, w in _variants(args, words, Nf):
        rec = dict(variant=name, gauge_low_pattern=gauge_ok(w, brot),
                   max_projector_change=projector_change(words, w, edges, brot),
                   entries_changed=int((w != words).sum()), slots={})
        for k in slots:
            r0, r1 = slot_rows(words, k, coded)[0], slot_rows(w, k, coded)[0]
            rec["slots"][k] = dict(tensor_bits_changed=sum(bin(a ^ b).count("1") for a, b in zip(r0, r1)),
                                   structural=structural_products(w, k, blocks, coded,
                                                                  seeds=range(args.seeds)))
        rec["t"] = round(time.time() - t0, 1)
        rep.append(rec)
        print(json.dumps(rec, default=int), flush=True)
        np.save(out / f"F_{name}.npy", F)
    (out / "report.json").write_text(json.dumps(dict(tool="gauge_search", npz=str(args.npz), npz_sha256=sha256(args.npz),
                                                     blocks=blocks, variant_seed=args.variant_seed, runs=rep,
                                                     wall_s=round(time.time() - t0, 1)), indent=1, default=int)
                                     + "\n")
    return 0


def _plan_job(a):
    name, k, seed, delta, cw, A, out_json = a
    import dfthc_refit as dr
    r = dr.shared_plan("shared", delta, A, cw, seed, blocks=None, pad_free=True, exchange=True,
                       settings=dr.PLAN_DEFAULTS["shared"], out_json=out_json)
    return dict(variant=name, k=k, seed=seed, products=r["products"], exact=r["exact"], cpu_s=r["cpu_s"],
                blocks=r.get("blocks"))


def cmd_plan_variants(args) -> int:
    """Full shared-planner runs (replan's shared + pad-rows + exchange settings) of one coded slot on the shipped
    table (control) and on random per-slice symmetry variants."""
    from concurrent.futures import ProcessPoolExecutor
    words, edges, brot = load_family_words(args.npz)
    Nf = np.load(args.group)["Nf"].astype(np.int64)
    coded = _coded_fn()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    k = args.slot
    jobs = []
    meta = []
    for name, F, w in _variants(args, words, Nf):
        np.save(out / f"F_{name}.npy", F)
        np.save(out / f"words_{name}.npy", w.astype(np.int16))
        delta, cw = coded(w, k)
        seeds = [int(x) for x in (args.control_seeds if name == "shipped" else args.seeds).split(",") if x.strip()]
        meta.append(dict(variant=name, gauge_low_pattern=gauge_ok(w, brot),
                         max_projector_change=projector_change(words, w, edges, brot),
                         entries_changed=int((w != words).sum()), seeds=seeds))
        for sd in seeds:
            jobs.append((name, k, sd, delta, list(cw), len(words), str(out / f"plan_{name}_s{sd}.json")))
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for r in ex.map(_plan_job, jobs):
            res.append(r)
            print(json.dumps(r, default=int), flush=True)
    (out / "report.json").write_text(json.dumps(dict(tool="gauge_search", npz=str(args.npz), npz_sha256=sha256(args.npz),
                                                     slot=k, variant_seed=args.variant_seed, variants=meta,
                                                     runs=res, wall_s=round(time.time() - t0, 1)),
                                                indent=1, default=int) + "\n")
    return 0


# ------------------------------------------------------------------------- re-gauge to cut gauge changes (qubits)


def gauge_changes(off, k, j=J, lowbits=LOWBITS):
    """Words of slot k whose low-bit gauge pattern differs between batch k-1 (incoming) and batch k (outgoing): each
    costs the gauge-aware emitter one kept top bit (+1 qubit). None if a pattern is not a function of the low bits."""
    A, n = off.shape
    S = -(-n // j)
    P = np.zeros((A, S * j), np.uint8)
    P[:, :n] = off
    qo = gz.low_pattern(P[:, j * k:j * (k + 1)], lowbits)
    qi = gz.low_pattern(P[:, j * (k - 1):j * k], lowbits) if k else np.zeros_like(qo)
    if qo is None or qi is None:
        return None
    return [int(x) for x in np.flatnonzero((qo != qi).any(axis=0))]


def gauge_change_entries(off, k, j=J, lowbits=LOWBITS) -> int:
    """Number of (low slice, word) pairs of slot k whose gauge bit changes between batch k-1 and batch k."""
    A, n = off.shape
    P = np.zeros((A, -(-n // j) * j), np.uint8)
    P[:, :n] = off
    qo = gz.low_pattern(P[:, j * k:j * (k + 1)], lowbits)
    qi = gz.low_pattern(P[:, j * (k - 1):j * k], lowbits) if k else np.zeros_like(qo)
    return int((qo != qi).sum())


def regauge(words, Nf, window_cols, slots, *, j=J, brot=BROT, lowbits=LOWBITS, restarts=20, seed=0):
    """Choose one pure pi-gauge kernel element per low slice, supported on `window_cols` (columns whose loaders on
    both sides are re-emitted), minimising the total number of gauge-changing words over `slots` (lexicographically:
    the worst slot first). Exact symmetry (n(f) = 0); the planners' tensors are unchanged. Returns the best table."""
    Nf = np.asarray(Nf, np.int64)
    E = Nf.shape[0]
    out_cols = np.ones(E, bool)
    out_cols[list(window_cols)] = False
    # f with n(f) = 0 and f = 0 outside the window: kernel of [Nf ; I_outside]
    M = np.concatenate([Nf, np.eye(E, dtype=np.int64)[:, out_cols]], axis=1)
    Kw = np.array(_kernel_basis(M), np.int64).reshape(-1, E)
    d = len(Kw)
    low = np.arange(len(words)) % (1 << lowbits)
    L = 1 << lowbits
    if d <= 10:   # every combination per slice
        combos = np.array([[(c >> i) & 1 for i in range(d)] for c in range(1, 1 << d)], np.int64)
    else:         # single basis flips and pairs
        eye = np.eye(d, dtype=np.int64)
        combos = np.array([eye[a] ^ eye[b] if a != b else eye[a] for a in range(d) for b in range(a, d)], np.int64)

    def table(C):
        F = (C @ Kw) % 2 if d else np.zeros((L, E), np.int64)
        return apply_move(words, 0, F[low], brot)

    def score(w):
        off = gz.off_gauge(w, brot)
        ch = [gauge_changes(off, k, j, lowbits) for k in slots]
        if any(c is None for c in ch):
            return (10 ** 6,), ch
        cnt = [len(c) for c in ch]
        # tiebreak: (slice, word) entries whose gauge changes, so single-slice moves can make progress
        fine = sum(gauge_change_entries(off, k, j, lowbits) for k in slots)
        return (max(cnt), sum(cnt), fine), ch

    best = None
    rng = np.random.default_rng(seed)
    for r in range(restarts if d else 1):
        C = np.zeros((L, d), np.int64) if r == 0 else rng.integers(0, 2, (L, d))
        cur = score(table(C))[0]
        improved = True
        while improved and d:
            improved = False
            for l in range(L):
                for cmb in combos:     # XOR moves on slice l's coefficient vector
                    C2 = C.copy()
                    C2[l] = C[l] ^ cmb
                    sc = score(table(C2))[0]
                    if sc < cur:
                        C, cur, improved = C2, sc, True
        if best is None or cur < best[0]:
            best = (cur, C.copy())
    w = table(best[1])
    sc, ch = score(w)
    return dict(kernel_dim_in_window=d, score=list(sc), changes={int(k): c for k, c in zip(slots, ch)},
                start_changes={int(k): gauge_changes(gz.off_gauge(words, brot), k, j, lowbits) for k in slots},
                words=w, C=best[1])


def cmd_regauge(args) -> int:
    words = np.load(args.words).astype(np.int64)
    z = np.load(args.npz)
    edges, brot = np.asarray(z["tree_edges"], np.int64), int(z["brot"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    g = group(edges, brot)
    slots = [int(x) for x in args.slots.split(",")]
    batches = sorted({b for k in slots for b in (k - 1, k) if b >= 0} - {int(x) for x in args.fixed_batches.split(",")
                                                                         if x.strip()})
    window = [c for b in batches for c in range(J * b, min(J * (b + 1), words.shape[1]))]
    res = regauge(words, g["Nf"], window, slots, restarts=args.restarts, seed=args.seed)
    w = res.pop("words")
    rep = dict(tool="gauge_search", words=str(args.words), words_sha256=sha256(args.words), npz=str(args.npz),
               npz_sha256=sha256(args.npz), slots=slots, rebatched=batches, window_cols=window,
               max_projector_change=projector_change(words, w, edges, brot),
               planner_tensors_equal={k: slot_rows(words, k, _coded_fn())[0] == slot_rows(w, k, _coded_fn())[0]
                                      for k in slots},
               entries_changed=int((w != words).sum()), **{k: v for k, v in res.items() if k != "C"},
               wall_s=round(time.time() - t0, 1))
    np.save(out / "words_regauged.npy", w.astype(np.int16))
    np.save(out / "C.npy", res["C"])
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=int) + "\n")
    print(json.dumps({k: rep[k] for k in ("kernel_dim_in_window", "start_changes", "changes", "max_projector_change",
                                          "planner_tensors_equal", "entries_changed")}, default=int))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("group")
    a.add_argument("npz")
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_group)
    b = sub.add_parser("bound")
    b.add_argument("npz")
    b.add_argument("--blocks")
    b.add_argument("--out")
    b.set_defaults(fn=cmd_bound)
    d = sub.add_parser("structural")
    d.add_argument("npz")
    d.add_argument("--group", required=True)
    d.add_argument("--blocks-json", default=json.dumps([[7, 8, 13], [1, 3, 11], [6, 9, 14], [4, 10, 12], [2, 5, 15]]))
    d.add_argument("--slots", default="4,5,6,7,8,9,10,11,12,13,14")
    d.add_argument("--variants", type=int, default=3)
    d.add_argument("--variant-seed", type=int, default=0)
    d.add_argument("--seeds", type=int, default=1)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_structural)
    e = sub.add_parser("plan-variants")
    e.add_argument("npz")
    e.add_argument("--group", required=True)
    e.add_argument("--slot", type=int, default=8)
    e.add_argument("--variants", type=int, default=6)
    e.add_argument("--variant-seed", type=int, default=0)
    e.add_argument("--seeds", default="0,1")
    e.add_argument("--control-seeds", default="0,1,2,3")
    e.add_argument("--procs", type=int, default=3)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_plan_variants)
    g = sub.add_parser("regauge")
    g.add_argument("npz", help="family submission.npz (tree_edges, brot)")
    g.add_argument("--words", required=True, help="the loaded words (.npy, e.g. replan's work/words.npy)")
    g.add_argument("--slots", required=True, help="slots whose loaders are re-emitted, e.g. 11,12,13,14")
    g.add_argument("--fixed-batches", default="", help="batches a kept (shipped) loader reads or writes")
    g.add_argument("--restarts", type=int, default=20)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_regauge)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
