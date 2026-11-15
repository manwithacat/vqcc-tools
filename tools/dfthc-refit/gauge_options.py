# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Per-loader gauge characterisation: which exact word symmetries matter to one ANGLE_k loader
(dfthc_refit 1.6.0-dev).

The exact symmetries of a Givens-tree angle table (gauge_search.py) are, per row, the group Z_2^E of pi-flip
sets f on the E columns, each paired with its unique negation set n(f). Slot k's loader XORs batch k - 1 into batch k
(J = 5 words each), so it sees a move only through the columns W of those two batches. Every move f is classified
for slot k as

- **costed**: n(f) restricted to W is nonzero. A negation changes bits 1..brot-2 of a word, so it changes the
  planners' tensor (bits 0..brot-2 of the XOR delta): Toffolis can move. Quotient dimension rank(Nf[:, W]).
- **constrained**: n(f)|W = 0 (the tensor is unchanged: Toffoli-neutral) but the pi flips on W differ between the two
  batches on some word: the loader's gauge change g = p_in XOR p_out moves, which changes the emitter's width (the
  qubit price below) and, above 128 qubits, legality.
- **free**: n(f)|W = 0 and the pi flips on W are equal on both batches word by word: neither the tensor nor g changes
  (p_in and p_out move together; only CNOTs in the loader change). Free for this slot; neighbours may still see it.

The emitter adds a constraint on top of the group: the final top-bit pattern of every word must be a function of
the low address bits (one move per low slice, avoiding words of value 2**(brot-2) or 3 * 2**(brot-2) where a negation
would flip the gauge bit; `gauge_search.allowed_spaces`). The per-slice dimensions of the costed space are reported
under that constraint.

The loader's **gauge contract**: p_in, p_out (top-bit pattern, as functions of the low address bits), whether both
are address-only, the change g, the changing words and rank(g). The qubit price of a gauge change:

- the 1.5 layout keeps one top bit per changing word: width = base + #changing;
- the in-place layout (gauge.build_circuit_gauge(inplace=True), v1.6) keeps rank(g) top bits: width = base + rank.

**Why rank(g) is the minimum for CNOT schedules.** After bit 7 ^= bit 6, the top bits hold p_in(l) XOR dir g(l)
(dir = 0 forward, 1 reverse). Any schedule of CNOTs among the top bits and X gates controlled by address functions
maps them to M (p_in(l) XOR dir g(l)) XOR c(l) with M invertible; the dir-dependent part dir M g(l) spans M applied
to span{g(l)}, an r-dimensional space, which needs at least r nonzero coordinates: at least r top bits hold the
direction bit on some address and cannot be lent clean. The in-place layout attains r. (Toffoli schedules that
recompute dir from the word bits are outside this class and would cost Toffolis.)

Credit: the in-place construction is read from Dmitrii Gluzdov's loaders (family 726299a7f65b, Apache-2.0); the
table symmetry analysis is gauge_search.py's; the emitter model is Lera's (marculera) tensorqrom; the pi-gauge convention is
Atakan Aldemir's.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gauge as gz  # noqa: E402
import gauge_search as gs  # noqa: E402

BROT, J, LOWBITS = 8, 5, 4
QUBIT_CAP = gz.QUBIT_CAP


# ------------------------------------------------------------------------------------------------ GF(2) helpers


def _int(v) -> int:
    return int(sum(1 << i for i, x in enumerate(np.asarray(v).ravel()) if x))


def _vec(x: int, n: int) -> np.ndarray:
    return np.array([(x >> i) & 1 for i in range(n)], np.int64)


def kernel(M) -> list:
    """Basis (0/1 arrays) of {f : f @ M = 0 mod 2} for an (E, m) 0/1 matrix (m may be 0)."""
    M = np.asarray(M, np.int64).reshape(np.asarray(M).shape[0], -1)
    if M.shape[1] == 0:
        return list(np.eye(M.shape[0], dtype=np.int64))
    return gs._kernel_basis(M % 2)


def extend_basis(sub, ambient) -> list:
    """Vectors of `ambient` (0/1 arrays) that extend the span of `sub` to the span of sub + ambient (a complement)."""
    piv = {}

    def add(v):
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p]
            else:
                piv[p] = v
                return True
        return False
    for s in sub:
        add(_int(s))
    return [a for a in ambient if add(_int(a))]


def span_basis(vectors) -> list:
    """An independent subset spanning the same space."""
    return extend_basis([], vectors)


# ----------------------------------------------------------------------------------------------- slot structure


def window(n_cols, k, j=J):
    """(prev, cur): column indices of batch k - 1 (empty for k = 0) and batch k, padded positions as None."""
    prev = [j * (k - 1) + i if k and j * (k - 1) + i < n_cols else None for i in range(j)]
    cur = [j * k + i if j * k + i < n_cols else None for i in range(j)]
    return prev, cur


def contract_matrix(E, k, n_cols, j=J):
    """(E, j) 0/1: f -> the change of g on word i = f[prev_i] XOR f[cur_i] (slot 0: f[cur_i])."""
    prev, cur = window(n_cols, k, j)
    C = np.zeros((E, j), np.int64)
    for i in range(j):
        for c in (prev[i], cur[i]):
            if c is not None:
                C[c, i] ^= 1
    return C


def classify(Nf, k, n_cols, j=J) -> dict:
    """Free / constrained / costed subspaces of the per-row move group for slot k (bases as 0/1 arrays)."""
    Nf = np.asarray(Nf, np.int64)
    E = Nf.shape[0]
    prev, cur = window(n_cols, k, j)
    W = [c for c in prev + cur if c is not None]
    K0 = kernel(Nf[:, W])                                    # tensor-neutral: n(f)|W = 0
    C = contract_matrix(E, k, n_cols, j)
    if K0:
        K0m = np.array(K0, np.int64)
        coef = kernel((K0m @ C) % 2)                          # combinations of K0 with no contract change
        free = span_basis([(np.asarray(c, np.int64) @ K0m) % 2 for c in coef]) if coef else []
    else:
        free = []
    constrained = extend_basis(free, K0)
    costed = extend_basis(K0, list(np.eye(E, dtype=np.int64)))
    pure = kernel(Nf)                                         # global pure-pi kernel (n(f) = 0 everywhere)
    pure_constrained = len(span_basis(free + [p for p in pure])) - len(free) if pure else 0
    return dict(window=W, free=free, constrained=constrained, costed=costed,
                dims=dict(group=E, free=len(free), constrained=len(constrained), costed=len(costed),
                          tensor_neutral=len(K0), pure_pi_kernel=len(pure),
                          pure_pi_not_free_for_slot=pure_constrained))


def describe(f, Nf, W) -> dict:
    """A move as its pi-flip columns and negation columns (all, and within the slot's window)."""
    f = np.asarray(f, np.int64)
    n = (f @ np.asarray(Nf, np.int64)) % 2
    Ws = set(W)
    return dict(pi=[int(c) for c in np.flatnonzero(f)], neg=[int(c) for c in np.flatnonzero(n)],
                pi_in_window=[int(c) for c in np.flatnonzero(f) if c in Ws],
                neg_in_window=[int(c) for c in np.flatnonzero(n) if c in Ws])


def gauge_contract(words, k, j=J, brot=BROT, lowbits=LOWBITS) -> dict:
    """p_in, p_out, address-only test, g, changing words, rank(g), keepers and both layouts' static widths."""
    B = gz.batches(words, j)
    A = B.shape[1]
    P = gz.off_gauge(B.transpose(1, 0, 2).reshape(A, -1), brot).reshape(A, -1, j).transpose(1, 0, 2)
    pin_rows = P[k - 1] if k else np.zeros_like(P[0])
    pin = gz.low_pattern(pin_rows, lowbits)
    pout = gz.low_pattern(P[k], lowbits)
    rep = dict(k=k, address_only=pin is not None and pout is not None,
               incoming_off_gauge_entries=int(pin_rows.sum()), outgoing_off_gauge_entries=int(P[k].sum()))
    if not rep["address_only"]:
        rep["note"] = ("the gauge pattern is not a function of the low address bits: the gauge-aware emitter cannot "
                       "clear the borrowed top bits (refused)")
        return rep
    g = pin ^ pout
    keepers, combos = gz.gf2_basis_columns(g)
    changing = [int(x) for x in range(j) if g[:, x].any()]
    base = gz.gauge_width(len(words), j, brot, lowbits, 0)
    rep.update(signed=not (pin.any() or pout.any()),
               p_in=["".join(map(str, pin[:, x])) for x in range(j)],
               p_out=["".join(map(str, pout[:, x])) for x in range(j)],
               g=["".join(map(str, g[:, x])) for x in range(j)],
               changing_words=changing, gauge_change_rank=len(keepers), keepers=[int(x) for x in keepers],
               dependents={int(a): [int(x) for x in b] for a, b in combos.items()},
               width=dict(base=base, kept_layout=gz.gauge_width(len(words), j, brot, lowbits, len(changing)),
                          inplace_layout=gz.gauge_width(len(words), j, brot, lowbits, len(keepers)), cap=QUBIT_CAP,
                          model="static model: selection + words + low cache + high cache - lent top bits"))
    return rep


def _g_rank_after(words, k, F_slice, Nf=None, j=J, brot=BROT, lowbits=LOWBITS):
    """(changing, rank) of slot k after the exact move with pi flips F_slice (16, E) per low slice (and their
    negation sets n(f), when Nf is given); None if the pattern is no longer address-only."""
    low = np.arange(len(words)) % (1 << lowbits)
    F = np.asarray(F_slice, np.int64)
    n = (F @ np.asarray(Nf, np.int64)) % 2 if Nf is not None else np.zeros_like(F)
    w = gs.apply_move(words, n[low], F[low], brot)
    c = gauge_contract(w, k, j, brot, lowbits)
    if not c["address_only"]:
        return None
    return len(c["changing_words"]), c["gauge_change_rank"]


def price_constrained(words, Nf, k, constrained, n_slots, j=J, brot=BROT, lowbits=LOWBITS) -> list:
    """For each constrained basis direction e: apply the exact move (its pi flips and n(e); n(e)|W = 0, so only the
    top bits move in the window; the negations outside W, listed, change other slots' tensors) on one low slice l; the new (changing, rank) of slots k - 1, k, k + 1 and the
    in-place width of slot k. Reports the best slice per direction (smallest slot-k rank, then neighbours)."""
    L = 1 << lowbits
    E = np.asarray(Nf).shape[0]
    nb = [s for s in (k - 1, k, k + 1) if 0 <= s < n_slots]
    start = {s: _g_rank_after(words, s, np.zeros((L, E), np.int64), Nf, j, brot, lowbits) for s in nb}
    out = []
    for e in constrained:
        e = np.asarray(e, np.int64)
        n_out = [int(c) for c in np.flatnonzero((e @ np.asarray(Nf, np.int64)) % 2)]
        best = None
        for l in range(L):
            F = np.zeros((L, E), np.int64)
            F[l] = e
            res = {s: _g_rank_after(words, s, F, Nf, j, brot, lowbits) for s in nb}
            if any(v is None for v in res.values()):
                continue
            key = (res[k][1], sum(v[1] for v in res.values()), res[k][0])
            if best is None or key < best[0]:
                best = (key, l, res)
        rec = dict(pi=[int(c) for c in np.flatnonzero(e)], neg_outside_window=n_out)
        if best is not None:
            _, l, res = best
            rec.update(best_slice=l, slot_rank={int(s): dict(before=start[s][1], after=res[s][1]) for s in nb},
                       slot_changing={int(s): dict(before=start[s][0], after=res[s][0]) for s in nb},
                       inplace_width_delta=res[k][1] - start[k][1], kept_width_delta=res[k][0] - start[k][0])
        out.append(rec)
    return out


def allowed_costed(words, Nf, k, n_cols, j=J, brot=BROT, lowbits=LOWBITS) -> list:
    """Per low slice: dimension of the moves the emitter allows (gauge_search.allowed_spaces) and of their costed
    part for slot k (rank of n(f)|W over the allowed space)."""
    spaces = gs.allowed_spaces(words, Nf, brot, lowbits)
    prev, cur = window(n_cols, k, j)
    W = [c for c in prev + cur if c is not None]
    out = []
    for l, Bs in enumerate(spaces):
        Bs = np.asarray(Bs, np.int64).reshape(-1, np.asarray(Nf).shape[0])
        nW = (Bs @ np.asarray(Nf, np.int64))[:, W] % 2 if len(Bs) else np.zeros((0, len(W)), np.int64)
        out.append(dict(slice=l, allowed=int(len(Bs)), costed=gs.gf2_rank_rows([_int(r) for r in nW])))
    return out


# ------------------------------------------------------------------------------------------------- plans


def find_plan(plans_dir, k):
    """The best plan JSON for slot k under `plans_dir` (a replan OUT_DIR, its work/plans, or a k<k> directory):
    fewest products, ties to the first file in sorted order (replan's choice). None if absent."""
    d = Path(plans_dir)
    for cand in (d / "work" / "plans" / f"k{k}", d / "plans" / f"k{k}", d / f"k{k}", d):
        files = sorted(cand.glob("*.json")) if cand.is_dir() else []
        files = [f for f in files if f.name not in ("report.json", "build.json")]
        if files:
            best = None
            for f in files:
                kp = json.loads(f.read_text())
                if "terms" not in kp:
                    continue
                n = len(kp["terms"])
                if best is None or n < best[0]:
                    best = (n, f, kp)
            if best:
                return best[1], best[2]
    return None


def plan_delta(words, k, kp, loader_delta):
    """The emitted delta of a plan: the slot's delta (coded padding as replan) with the plan's padded rows."""
    delta, _ = loader_delta(words, k)
    new = delta.copy()
    for r, vals in (kp.get("pad_rows") or {}).items():
        assert int(r) >= len(words), "a plan changes a real table row"
        new[int(r)] = np.asarray(vals, np.int64)
    return new


def emit_plan(km, tq, words, k, kp, loader_delta, emit_dir=None, *, sha256=None) -> dict:
    """Emit a plan in both gauge layouts (1.5 kept / v1.6 in place), native-check each on the gauge inputs."""
    import tempfile
    delta = plan_delta(words, k, kp, loader_delta)
    n_sel = (len(words) - 1).bit_length()
    pin, pout = gz.slot_gauge(words, k, J)
    rec = dict(products=len(kp["terms"]))
    td = tempfile.TemporaryDirectory()
    changing = bool((np.asarray(pin) ^ np.asarray(pout)).any())
    for pol in (("never", "always") if changing else ("never",)):   # no gauge change: one layout
        em = gz.GaugeEmitter(tq, pin, pout, inplace=pol)
        circ = em.build_circuit(delta, [tuple(t) for t in kp["terms"]], n_sel, BROT, LOWBITS, groups=None,
                                scratch=True)
        layout = em.layout or "lera"
        if emit_dir is not None:
            p = Path(emit_dir) / f"angle_k_{k}.{layout}.kmb"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise FileExistsError(p)
            circ.to_file(str(p), format="kmb")
        else:
            p = Path(td.name) / f"angle_k_{k}.{layout}.kmb"
            circ.to_file(str(p), format="kmb")
        circ = km.Circuit.from_file(str(p), format="kmb")
        chk = gz.native_check(km, circ, delta[:len(words)], gz.check_inputs(words, k))
        rec["inplace" if pol == "always" else "kept"] = dict(
            qubits=int(circ.num_qubits), toffolis=int(circ.operation_statistics().num_toffoli_like),
            native_check=chk["status"], file=str(p) if emit_dir is not None else None,
            kmb_sha256=sha256(p) if sha256 else None, emitter=em.record())
    td.cleanup()
    kq, iq = rec["kept"]["qubits"], rec.get("inplace", {}).get("qubits")
    rec["auto_layout"] = "inplace" if (iq is not None and kq > QUBIT_CAP and iq < kq) else "kept"
    return rec


# ----------------------------------------------------------------------------------------------- the report


def slot_options(words, edges, k, *, group=None, shipped=None, plan=None, structural=0, structural_seed=0,
                 brot=BROT, j=J, lowbits=LOWBITS, max_basis=None) -> dict:
    """The gauge-options report for slot k (JSON-ready). `shipped`: the loader's class record (gauge.loader_classes);
    `plan`: the emit_plan record; `structural`: number of random allowed per-slice variants priced with
    planner_shared's structural count (gauge_search.structural_products)."""
    words = np.asarray(words, np.int64)
    n_cols = words.shape[1]
    n_slots = -(-n_cols // j)
    g = group if group is not None else gs.group(edges, brot)
    Nf = np.asarray(g["Nf"], np.int64)
    cl = classify(Nf, k, n_cols, j)
    W = cl["window"]
    con = gauge_contract(words, k, j, brot, lowbits)
    rep = dict(k=k, window_columns=W, dims=cl["dims"], contract=con)
    lim = (lambda xs: xs[:max_basis]) if max_basis else (lambda xs: xs)
    rep["bases"] = {name: [describe(f, Nf, W) for f in lim(cl[name])] for name in ("free", "constrained", "costed")}
    if max_basis and any(len(cl[n]) > max_basis for n in ("free", "constrained", "costed")):
        rep["bases_truncated_to"] = max_basis
    rep["allowed_per_slice"] = allowed_costed(words, Nf, k, n_cols, j, brot, lowbits)
    rep["allowed_costed_max"] = max(s["costed"] for s in rep["allowed_per_slice"])
    if con["address_only"]:
        rep["constrained_prices"] = price_constrained(words, Nf, k, cl["constrained"], n_slots, j, brot, lowbits)
    if structural:
        rep["costed_prices"] = price_costed_structural(words, Nf, k, structural, structural_seed)
    if shipped is not None:
        rep["shipped_loader"] = {x: shipped.get(x) for x in ("file", "loader", "toffolis", "qubits",
                                                              "needs_incoming_gauge", "gauge_change")}
    if plan is not None:
        rep["plan"] = plan
    rep["repairs"] = repairs(rep)
    return rep


def price_costed_structural(words, Nf, k, n, seed) -> dict:
    """planner_shared's structural count (fixed partition, one rng seed) on the shipped table and on n random
    allowed per-slice variants (gauge_search.random_variant): a static model of what the costed moves can do."""
    import dfthc_refit as dr
    blocks = [[7, 8, 13], [1, 3, 11], [6, 9, 14], [4, 10, 12], [2, 5, 15]]
    spaces = gs.allowed_spaces(words, Nf)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    base = gs.structural_products(words, k, blocks, dr.loader_delta)[0]
    runs = []
    for v in range(n):
        F, w = gs.random_variant(words, Nf, spaces, rng)
        d0, _ = dr.loader_delta(words, k)
        d1, _ = dr.loader_delta(w, k)
        runs.append(dict(variant=v, tensor_entries_changed=int((d0 != d1).sum()),
                         structural=gs.structural_products(w, k, blocks, dr.loader_delta)[0]))
    return dict(blocks=blocks, shipped=base, variants=runs, seed=seed, cpu_s=round(time.time() - t0, 1),
                tier="static model (planner_shared structural count, fixed partition)")


def repairs(rep) -> list:
    """Suggested repairs, most direct first."""
    con, out = rep["contract"], []
    if not con.get("address_only"):
        out.append("gauge pattern is not address-only: re-gauge per low slice (gauge_search.regauge) before any "
                   "gauge-aware emission")
        return out
    w = con["width"]
    plan = rep.get("plan") or {}
    kept_q = plan.get("kept", {}).get("qubits", w["kept_layout"])
    inpl_q = plan.get("inplace", {}).get("qubits", w["inplace_layout"])
    if con["changing_words"]:
        if kept_q > QUBIT_CAP >= inpl_q:
            out.append(f"in-place gauge change: keepers {con['keepers']} (rank {con['gauge_change_rank']} of "
                       f"{len(con['changing_words'])} changing words), dependents {con['dependents']}; width "
                       f"{kept_q} -> {inpl_q} <= {QUBIT_CAP} (replan --gauge-inplace auto does this)")
        elif inpl_q > QUBIT_CAP:
            better = [p for p in rep.get("constrained_prices", []) if p.get("inplace_width_delta", 0) < 0]
            out.append(f"in-place width {inpl_q} > {QUBIT_CAP}: lower rank(g) by re-gauging "
                       + (f"({len(better)} constrained directions lower slot {rep['k']}'s rank on one slice; "
                          f"they move neighbours' contracts)" if better else
                          "(no single-slice constrained direction lowers it; gauge_search.regauge over the window)"))
        elif inpl_q < kept_q:
            out.append(f"in-place layout saves {kept_q - inpl_q} qubits (not needed: the 1.5 layout fits; width does "
                       f"not enter the score below the cap)")
    else:
        out.append("no gauge change across this slot: the contract is p_in = p_out (lent top bits, no extra qubits)")
    if rep.get("allowed_costed_max", 0) == 0:
        out.append("no costed move is allowed under the emitter's per-slice constraint: this slot's tensor is fixed")
    else:
        out.append(f"costed moves exist (up to {rep['allowed_costed_max']} dims per slice): price them with the "
                   f"planner (gauge_search plan-variants); none was found below the shipped plans on aba448aeadb5 slot 8")
    return out


def summary_lines(rep) -> list:
    """One screen per slot."""
    d, con = rep["dims"], rep["contract"]
    L = [f"slot {rep['k']}: window columns {rep['window_columns'][0]}..{rep['window_columns'][-1]} "
         f"({len(rep['window_columns'])})   group 2^{d['group']} per row",
         f"  free {d['free']}   constrained {d['constrained']}   costed {d['costed']}   (tensor-neutral "
         f"{d['tensor_neutral']}; pure-pi kernel {d['pure_pi_kernel']}, of which not free here "
         f"{d['pure_pi_not_free_for_slot']})",
         f"  allowed per low slice (emitter): costed dims max {rep['allowed_costed_max']}, "
         f"allowed dims {min(s['allowed'] for s in rep['allowed_per_slice'])}.."
         f"{max(s['allowed'] for s in rep['allowed_per_slice'])}"]
    if con.get("address_only"):
        w = con["width"]
        L.append(f"  contract: {'signed' if con['signed'] else 'pi'} gauge, address-only; changing words "
                 f"{con['changing_words']} rank {con['gauge_change_rank']} keepers {con['keepers']} dependents "
                 f"{con['dependents']}")
        L.append(f"  width (static): base {w['base']}, 1.5 kept layout {w['kept_layout']}, in place "
                 f"{w['inplace_layout']}, cap {w['cap']}")
    else:
        L.append("  contract: NOT address-only")
    sl = rep.get("shipped_loader")
    if sl:
        L.append(f"  shipped loader: {sl['loader']}, {sl['toffolis']} Toffolis, {sl['qubits']} Q")
    pl = rep.get("plan")
    if pl:
        for lay in ("kept", "inplace"):
            if lay in pl:
                r = pl[lay]
                L.append(f"  plan ({pl['products']} products) {lay:7s}: {r['toffolis']} Toffolis, {r['qubits']} Q, "
                         f"native check {r['native_check']}")
    cp = rep.get("costed_prices")
    if cp:
        L.append(f"  costed (structural, fixed partition): shipped {cp['shipped']}, variants "
                 f"{[v['structural'] for v in cp['variants']]}")
    for r in rep["repairs"]:
        L.append(f"  -> {r}")
    return L
