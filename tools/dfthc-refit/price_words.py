#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the author's direction.
"""price_words: exact accuracy pricing of angle-word moves for a dfthcsos family (dfthc_refit 1.3.0-dev module).

A *word move* changes some entries of the angle-word table (row, column) -> new word. The ANGLE_k loaders must then
load the new words, which may let a planner use fewer products (Toffolis); the accuracy side pays: error_1, error_2,
E_gap and Lambda move at fixed weights, and a constrained re-refit of the weights (dfthc_refit.refit) has to restore
error_2 and the E_gap margin at some lambda_eff cost. This module prices that cost, per move, in three tiers:

  first_order  per-entry model (exact decode of the changed row, e2^2 to second order, exact rank-2 update of h1'
               for Lambda, E_gap and error_1 with the official matching) converted to lambda_eff with the base's KKT
               multipliers. Vectorised over a whole column; used to screen many entries.
  exact_fixed  official dfthc_math metrics of the moved table at the base weights, converted with the KKT
               multipliers (refit-corrected first-order price of the exact change).
  refit        a bounded constrained refit (dfthc_refit.refit, error_2 and the E_gap margin held at the base's
               values, error_1 free below the cap) from the base weights on the moved table; the price is the
               official lambda_eff after the refit against the better of the base and a same-budget control refit
               on the unmoved table. This is the tier a decision uses.

Prices are reported as relative lambda_eff changes and in **score units** (chemistry cost units, lambda_eff x Q x T):
one loader Toffoli per pass is 2 lambda_eff Q units (T counts the angle loaders twice). `rank_moves` combines a
price with a planner callback's loader saving (Toffolis per pass) into net = saving - price.

Planner callbacks: any callable (old_words, new_words, move) -> saving in Toffolis per pass. `coded_rank_planner`
is the static one shipped here: per loader, the GF(2) rank of the coded bit-0 outputs' mixed ANF tensor (rows =
output x low monomial, columns = high monomials); ceil(rank / 2) products is the lower bound a pairing planner meets
on these tables (the "low-bit mixed rank" of the soft-word write-up by "i want to run gemma 9b"). It is a static
model: a loader only realises it when re-planned and emitted.

`rank_drop_moves` constructs, for one loader, the cheapest bit-0 toggles (word XOR 1: only bit 0 changes, so the
signed gauge and every free output are untouched) that put a chosen combination of the coded outputs' slope vectors
into the constant space, which lowers that loader's coded mixed rank by one.

Every metric is the official verifier code (dfthcsos.scoreinfra.dfthc_math, unmodified), imported from the
competition's verifiers directory. Needs NumPy, SciPy and PyTorch (the refit tier); no kickmix (pass the words the
loaders load, e.g. from dfthc_refit.recover_words in a kickmix interpreter).

    import price_words as pw
    fam = pw.load_family(FAMILY_DIR, VERIFIER_DIR, words=words, qubits=427, toffolis=15175)
    pw.shadow_prices(fam)                       # score units per uHa of error_2 / error_1, uHa per loader Toffoli
    moves = [pw.Move("m0", ((row, col, new_word), ...))]
    table = pw.rank_moves(fam, moves, pw.coded_rank_planner(), refit_top=10, max_evals=300)
"""
from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dfthc_refit as dr  # noqa: E402

__version__ = "0.1.0"

UHA = 1e-6
E1_SLACK_MHA = 0.1   # error_1 further than this below the limit counts as an inactive inequality


# ------------------------------------------------------------------------------------------------------ moves


@dataclass(frozen=True)
class Move:
    """A named set of table edits ((row, column, new_word), ...)."""
    name: str
    changes: tuple
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def gauge_ok(words, brot=8):
    """Signed gauge: the top bit copies bit brot - 2."""
    w = np.asarray(words, np.int64)
    return ((w >> (brot - 1)) & 1) == ((w >> (brot - 2)) & 1)


def apply_move(words, move: Move, brot=8, check_gauge=True):
    """New table with the move's edits; raises on out-of-range words or a broken signed gauge."""
    w = np.array(words, dtype=np.int64, copy=True)
    for r, c, v in move.changes:
        v = int(v)
        if not 0 <= v < (1 << brot):
            raise ValueError(f"{move.name}: word {v} out of range at ({r}, {c})")
        w[int(r), int(c)] = v
    if check_gauge and not gauge_ok(w, brot).all():
        raise ValueError(f"{move.name}: signed gauge broken")
    return w


def parent_amplitude(words, tree, brot=8):
    """|parent amplitude| at each Givens rotation of every row (the rotation of word (row, e) acts on it). Zero means
    the word is an exact don't-care for every metric (a dead word)."""
    th = np.pi * np.asarray(words, np.float64) / (1 << (brot - 1))
    A, n = th.shape
    v = np.zeros((A, n + 1))
    v[:, 0] = 1
    amp = np.zeros((A, n))
    for e, (p, c) in enumerate(np.asarray(tree, np.int64)):
        pv = v[:, p].copy()
        cv = v[:, c].copy()
        amp[:, e] = np.abs(pv)
        v[:, p] = np.cos(th[:, e]) * pv - np.sin(th[:, e]) * cv
        v[:, c] = np.sin(th[:, e]) * pv + np.cos(th[:, e]) * cv
    return amp


# ------------------------------------------------------------------------------------------------------ family


@dataclass
class Family:
    name: str
    npz: Path
    words: np.ndarray
    brot: int
    tree: np.ndarray
    qubits: int
    toffolis: float
    P: object            # dfthc_refit.Problem at the base words
    y0: np.ndarray
    base: dict           # official metrics at the base
    mu: dict             # KKT multipliers (d log lambda / d c), c = (e1^2, e2^2, E_gap)
    diag: dict
    vdir: Path
    official_ctx: dict

    @property
    def cost(self):
        return self.base["lambda_eff"] * self.qubits * self.toffolis

    @property
    def loader_toffoli_units(self):
        """Score units of one loader Toffoli per pass (T counts each loader twice)."""
        return 2.0 * self.base["lambda_eff"] * self.qubits

    def units(self, rel):
        """Relative lambda_eff change -> score units (cost x rel; Q and T fixed)."""
        r = np.asarray(rel, np.float64) * self.cost
        return float(r) if r.ndim == 0 else r

    def problem_for(self, words):
        """dfthc_refit.Problem on another table, sharing the base's integrals (no rebuild of the large tensors)."""
        ctx = self.official_ctx
        dm = ctx["dm"]
        words = np.asarray(words, np.int64)
        u = dm.reconstruct_u_vectors(words / (1 << (self.brot - 1)), self.tree)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        P = copy.copy(self.P)
        torch = P.torch
        N = P.N
        P.V = torch.tensor(u[:N])
        P.ub = torch.tensor(u[N:].reshape(P.R, P.B, N))
        rr, cc = np.tril_indices(N)
        sc = np.where(rr == cc, 1.0, math.sqrt(2.0))
        P.PB = P.ub[..., torch.tensor(rr)] * P.ub[..., torch.tensor(cc)] * torch.tensor(sc)
        P.kink_basis = None
        brot, edges, S, beta1, rd, eup = (self.brot, self.tree, ctx["S"], ctx["beta1"], ctx["rd"], ctx["eup"])

        def official(w, W):
            return dm.evaluate_dfthc_angle_words(words, w, W, S, beta1, rd, brot=int(brot), tree_edges=edges,
                                                 energy_upper=eup)

        P.official_fn = official
        return P


def load_family(family_dir, verifier_dir, *, words=None, qubits, toffolis, name=None, threads=2) -> Family:
    """A family (submission.npz + the words its loaders load). `words` defaults to the npz angle_words (pass the
    recovered words when they can differ). `qubits` / `toffolis` are the official resources (Q_total, T_total)."""
    import torch
    torch.set_num_threads(int(threads))
    d = Path(family_dir)
    npz = d / "submission.npz" if d.is_dir() else d
    vdir = Path(verifier_dir)
    z = np.load(npz, allow_pickle=False)
    if words is None:
        words = np.asarray(z["angle_words"], np.int64)
    words = np.asarray(words, np.int64)
    brot = int(z["brot"]) if "brot" in z.files else 8
    tree = np.asarray(z["tree_edges"], np.int64)
    P = dr.femoco_problem(npz, words, brot, vdir)
    dm, ref = dr._official_modules(vdir)
    shifts = ref.load_fixed_shifts()
    N = int(ref.ORBITALS)
    S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), np.float64)
    beta1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))
    rd = ref.load_reference(ref.check_fcidump())
    ctx = dict(dm=dm, ref=ref, S=S, beta1=beta1, rd=rd, eup=float(ref.UPPER_ENERGY))
    y0 = P.y0()
    diag = dr.diagnose(P, y0)
    mu = dict(zip(("e1", "e2", "gap"), diag["kkt_mu"]))
    # error_1 is an inequality; with slack (> E1_SLACK_MHA) its multiplier is 0 and the fit uses (e2, gap) only
    if diag["official"]["error_1"] * 1e3 < dr.ERROR_LIMIT * 1e3 - E1_SLACK_MHA:
        z = np.where(np.abs(y0) >= dr.ZERO_HOLD, y0, 0.0)
        q = P.evaluate(z)
        m2, res2, _, viol2 = dr.kkt(q, np.abs(y0) >= dr.ZERO_HOLD, act=(1, 2))
        diag["kkt_e1_inactive"] = dict(mu_e2=float(m2[0]), mu_gap=float(m2[1]), rel_residual=float(res2),
                                       held_violations=int((viol2 > 0).sum()))
        mu = dict(e1=0.0, e2=float(m2[0]), gap=float(m2[1]))
    return Family(name=name or d.name, npz=npz, words=words, brot=brot, tree=tree, qubits=int(qubits),
                  toffolis=float(toffolis), P=P, y0=y0, base=diag["official"], mu=mu, diag=diag, vdir=vdir,
                  official_ctx=ctx)


# ---------------------------------------------------------------------------------------------- shadow prices


def first_order_rel(fam: Family, dLam, dgap, de2sq, de1sq, e1_binding=False):
    """Refit-corrected first-order relative lambda_eff change of a metric change (dLambda, dE_gap, d(e2^2),
    d(e1^2)) at the base: the change at fixed weights plus what a constrained refit pays (KKT multipliers) to restore
    error_2 and E_gap. error_1 is an inequality: priced only when `e1_binding` (its multiplier is used then)."""
    m = fam.base
    g, Lam, lam = m["E_gap"], m["Lambda"], m["lambda_eff"]
    fixed = (g * np.asarray(dLam) + (Lam - g) * np.asarray(dgap)) / lam ** 2
    corr = -fam.mu["gap"] * np.asarray(dgap) - fam.mu["e2"] * np.asarray(de2sq)
    if e1_binding:
        corr = corr - fam.mu["e1"] * np.asarray(de1sq)
    return fixed + corr


def shadow_prices(fam: Family) -> dict:
    """KKT shadow prices at the base, in score units: per uHa of error_2 and error_1 (d cost / d e at the current e),
    per uHa of E_gap margin, and the uHa of error_2 one loader Toffoli per pass buys."""
    m = fam.base
    c = fam.cost
    p_e2 = c * abs(fam.mu["e2"]) * 2 * m["error_2"] * UHA
    p_e1 = c * abs(fam.mu["e1"]) * 2 * m["error_1"] * UHA
    p_gap = c * abs(fam.mu["gap"]) * UHA
    lt = fam.loader_toffoli_units
    return dict(cost=c, lambda_eff=m["lambda_eff"], error_1_mHa=m["error_1"] * 1e3, error_2_mHa=m["error_2"] * 1e3,
                E_gap_margin_Ha=m["E_gap"] - dr.E_GAP_FLOOR, kkt_mu=fam.mu, kkt_mu_all3=fam.diag["kkt_mu"],
                kkt_e1_inactive=fam.diag.get("kkt_e1_inactive"),
                kkt_rel_residual=fam.diag["kkt_rel_residual"], wrong_sign=fam.diag["wrong_sign"],
                units_per_uHa_e2=p_e2, units_per_uHa_e1=p_e1, units_per_uHa_gap=p_gap,
                loader_toffoli_units=lt, uHa_e2_per_loader_toffoli=lt / p_e2 if p_e2 else math.inf)


def set_e2_price(fam: Family, units_per_uHa: float, *, source="finite response") -> dict:
    """Replace the error_2 multiplier by one implied by a measured price (score units per uHa of error_2), e.g. when
    the KKT fit is ill-conditioned or has the wrong sign (E_gap and error_2 normals nearly anti-parallel). The
    E_gap multiplier is kept. Returns the old and new multipliers."""
    old = dict(fam.mu)
    mu2 = -float(units_per_uHa) / (fam.cost * 2 * fam.base["error_2"] * UHA)
    fam.mu = dict(fam.mu, e2=mu2)
    fam.diag["mu_override"] = dict(old=old, new=dict(fam.mu), units_per_uHa=float(units_per_uHa), source=source)
    return fam.diag["mu_override"]


# ------------------------------------------------------------------------------------------------------ refits


def _targets(fam: Family, P, y, m_start, e2_official_mHa=None, gap_margin=None):
    """dfthc_refit.refit targets in P's model units: error_2 and E_gap held at official values (default: the
    base's), error_1 free unless the start is above the cap (then held at the cap)."""
    z = np.where(np.abs(y) >= dr.ZERO_HOLD, y, 0.0)
    q = P.evaluate(z)
    e2o = fam.base["error_2"] * 1e3 if e2_official_mHa is None else float(e2_official_mHa)
    gapo = fam.base["E_gap"] if gap_margin is None else dr.E_GAP_FLOOR + float(gap_margin)
    e2m = q["e2"] + (e2o - m_start["error_2"] * 1e3)
    gapm = q["gap"] + (gapo - m_start["E_gap"])
    e1m = None
    if m_start["error_1"] * 1e3 > dr.E1_CAP_MHA:
        e1m = q["e1"] + (dr.E1_CAP_MHA - 1e-3 - m_start["error_1"] * 1e3)
    return (e1m, e2m, gapm)


def penalty_refit(P, y_start, *, e2_mHa, gap_Ha, e1_cap_mHa=dr.E1_CAP_MHA, max_evals=300, rho=1e-3):
    """Quadratic-penalty refit (scipy L-BFGS-B on the free coordinates; held zeros stay 0): minimise
    log lambda_eff + rho * (v_e2^2 + v_gap^2 + v_e1^2), each v a one-sided violation in uHa (official units are
    set by the caller through the model-unit targets). Robust where the Gauss-Newton retraction of
    dfthc_refit.refit fails (e2 and E_gap normals nearly anti-parallel, as at a kink of E_gap). The optimal
    violation is about price / (2 rho): ~3e-3 uHa at rho = 1e-3 and the prices seen here."""
    from scipy.optimize import minimize
    y0 = np.asarray(y_start, np.float64)
    free = np.abs(y0) >= dr.ZERO_HOLD
    base = np.where(free, y0, 0.0)
    idx = np.flatnonzero(free)
    count = [0]
    best = [math.inf, None]

    def f(x):
        y = base.copy()
        y[idx] = x
        q = P.evaluate(y)
        count[0] += 1
        v2 = max(0.0, q["e2"] - e2_mHa) * 1e3          # uHa
        vg = max(0.0, gap_Ha - q["gap"]) * 1e6
        v1 = max(0.0, q["e1"] - e1_cap_mHa) * 1e3
        val = q["obj"] + rho * (v2 * v2 + vg * vg + v1 * v1)
        g = q["g"][0].copy()
        # d e2 / d c2 = 1 / (2 e2) (Ha); v2 in uHa: d v2 / d c2 = 1e6 / (2 e2_Ha)
        if v2 > 0:
            g += rho * 2 * v2 * (1e6 / (2 * q["e2"] * 1e-3)) * q["g"][2]
        if vg > 0:
            g -= rho * 2 * vg * 1e6 * q["g"][3]
        if v1 > 0:
            g += rho * 2 * v1 * (1e6 / (2 * q["e1"] * 1e-3)) * q["g"][1]
        if val < best[0]:
            best[0], best[1] = val, y.copy()
        return val, g[idx]

    try:
        r = minimize(f, base[idx], jac=True, method="L-BFGS-B",
                     options=dict(maxfun=max_evals, maxiter=max_evals, maxcor=20, ftol=1e-15, gtol=1e-12))
        stop = "PENALTY_" + ("CONVERGED" if r.success else "BUDGET")
    except Exception as ex:  # noqa: BLE001
        stop = f"PENALTY_ERROR {ex}"
    return dict(y=best[1], stop=stop, evals=count[0])


def bounded_refit(fam: Family, words=None, *, max_evals=300, e2_official_mHa=None, gap_margin=None, log=None,
                  kink_tol=dr.KINK_TOL, start=None, method="retract", rho=1e-3, steps=6):
    """One constrained refit from the base weights (or `start`) on `words` (default: the base table), error_2 and
    E_gap held at the base's official values (or `e2_official_mHa` / `gap_margin`), error_1 free below the cap.
    method 'retract': dfthc_refit.refit (projected L-BFGS with a Gauss-Newton retraction); 'penalty':
    penalty_refit (L-BFGS-B, robust at E_gap kinks); 'ladder': penalty refits along `steps` targets from the
    start's own error_2 / E_gap to the goal (continuation, for moves that cost tens of uHa). Returns dict(lambda_eff, metrics, passes, stop, evals,
    seconds, y); only points passing the official gates (and the e2 target within 0.2 uHa) count; the start counts
    when it passes and is better."""
    t0 = time.time()
    P = fam.P if words is None else fam.problem_for(words)
    y_s = fam.y0 if start is None else np.asarray(start, np.float64)
    m0 = P.official(y_s)
    tg = _targets(fam, P, y_s, m0, e2_official_mHa, gap_margin)
    if method == "penalty":
        rf = penalty_refit(P, y_s, e2_mHa=tg[1], gap_Ha=tg[2], max_evals=max_evals, rho=rho)
    elif method in ("ladder", "ladder+retract"):
        # continuation: walk the error_2 / E_gap targets from the start's own values to the goal in `steps`
        q = P.evaluate(np.where(np.abs(y_s) >= dr.ZERO_HOLD, y_s, 0.0))
        e2a, ga = max(q["e2"], tg[1]), min(q["gap"], tg[2])
        y_c, ev, stop = y_s, 0, None
        for t in range(1, steps + 1):
            f = t / steps
            rr = penalty_refit(P, y_c, e2_mHa=e2a + f * (tg[1] - e2a), gap_Ha=ga + f * (tg[2] - ga),
                               max_evals=max(1, (max_evals // 2 if method == "ladder+retract" else max_evals) // steps),
                               rho=rho)
            ev += rr["evals"]
            stop = rr["stop"]
            if rr["y"] is not None:
                y_c = rr["y"]
        rf = dict(y=y_c, stop="LADDER_" + str(stop), evals=ev)
        if method == "ladder+retract":
            # polish with dfthc_refit.refit from the ladder's end point (same targets)
            r2 = dr.refit(P, y_c, tg, max_evals=max(1, max_evals - ev), log=log, kink_tol=kink_tol)
            mm = P.official(y_c)
            if r2["y"] is not None and (not P.passes(mm) or P.official(r2["y"])["lambda_eff"] < mm["lambda_eff"]):
                y_c = r2["y"]
            rf = dict(y=y_c, stop=rf["stop"] + "+" + r2["stop"], evals=ev + r2["evals"])
    else:
        rf = dr.refit(P, y_s, tg, max_evals=max_evals, log=log, kink_tol=kink_tol)
    cands = []
    e2_ok = e2_official_mHa is None or m0["error_2"] * 1e3 <= float(e2_official_mHa) + 1e-9
    if P.passes(m0) and e2_ok:
        cands.append((m0["lambda_eff"], m0, y_s, "start"))
    if rf["y"] is not None:
        m1 = P.official(rf["y"])
        if P.passes(m1) and (e2_official_mHa is None or m1["error_2"] * 1e3 <= float(e2_official_mHa) + 2e-4):
            cands.append((m1["lambda_eff"], m1, rf["y"], "refit"))
    out = dict(stop=rf["stop"], evals=rf["evals"], seconds=round(time.time() - t0, 1), targets=tg, fixed=m0,
               last=None if rf["y"] is None else {k: v for k, v in P.official(rf["y"]).items()})
    if not cands:
        out.update(lambda_eff=math.inf, metrics=None, passes=False, y=None, source=None)
    else:
        lam, m, y, how = min(cands, key=lambda c: c[0])
        out.update(lambda_eff=lam, metrics=m, passes=True, y=y, source=how)
    return out


def finite_response(fam: Family, deltas_uHa=(0.0, 1.0, 5.0, 20.0), *, max_evals=300, log=None,
                    kink_tol=dr.KINK_TOL, method="retract", ladder=False, polish_evals=300) -> list:
    """lambda_eff after a bounded refit at fixed words with error_2 held delta uHa below the base (delta 0 is the
    fixed-word reproduction check). Returns rows with the finite price in score units per uHa (against the base).

    ladder=True: first polish the base at delta 0 with dfthc_refit.refit (`polish_evals`), then walk the deltas in
    increasing order with `method` refits each started from the previous point; prices are then also reported
    against the polished point (`units_vs_polished`). Every point is checked with the official metrics."""
    rows = []
    lam0 = fam.base["lambda_eff"]
    start, lam_ref = None, lam0
    if ladder:
        r0 = bounded_refit(fam, None, max_evals=polish_evals, log=log, kink_tol=kink_tol, method="retract")
        start, lam_ref = r0["y"], r0["lambda_eff"]
        rows.append(dict(delta_uHa=0.0, polish=True, lambda_eff=r0["lambda_eff"], rel=(lam_ref - lam0) / lam0,
                         units=fam.units((lam_ref - lam0) / lam0), units_per_uHa=None, passes=r0["passes"],
                         stop=r0["stop"], evals=r0["evals"], seconds=r0["seconds"],
                         error_2_mHa=r0["metrics"]["error_2"] * 1e3, error_1_mHa=r0["metrics"]["error_1"] * 1e3,
                         E_gap_margin_Ha=r0["metrics"]["E_gap"] - dr.E_GAP_FLOOR))
        deltas_uHa = sorted(d for d in deltas_uHa if d > 0)
    for d in deltas_uHa:
        r = bounded_refit(fam, None, max_evals=max_evals, e2_official_mHa=fam.base["error_2"] * 1e3 - d * 1e-3,
                          log=log, kink_tol=kink_tol, method=method, start=start)
        rel = (r["lambda_eff"] - lam0) / lam0
        row = dict(delta_uHa=d, lambda_eff=r["lambda_eff"], rel=rel, units=fam.units(rel),
                   units_per_uHa=fam.units(rel) / d if d else None, passes=r["passes"], stop=r["stop"],
                   evals=r["evals"], seconds=r["seconds"], source=r.get("source"),
                   error_2_mHa=None if r["metrics"] is None else r["metrics"]["error_2"] * 1e3,
                   error_1_mHa=None if r["metrics"] is None else r["metrics"]["error_1"] * 1e3,
                   E_gap_margin_Ha=None if r["metrics"] is None else r["metrics"]["E_gap"] - dr.E_GAP_FLOOR)
        if ladder and r["metrics"] is not None:
            de = fam.base["error_2"] * 1e6 - r["metrics"]["error_2"] * 1e6     # realised tightening, uHa
            rp = (r["lambda_eff"] - lam_ref) / lam_ref
            row.update(units_vs_polished=fam.units(rp), realised_delta_uHa=de,
                       units_per_realised_uHa=fam.units(rp) / de if de > 0 else None)
            if r["y"] is not None:
                start = r["y"]
        rows.append(row)
        if log:
            log(event="finite_row", **row)
    return rows


# --------------------------------------------------------------------------------------------- per-entry model


class RowPricer:
    """Vectorised first-order price of replacing one column's words, row by row, against a frame table (the
    per-row model: exact decode, e2^2 to second order in the row's projector change with
    cross terms between rows left out, exact rank-2 h1' update for Lambda, E_gap and e1 with the official matching).
    Rows whose word is unchanged price 0."""

    def __init__(self, fam: Family, words=None):
        ctx = fam.official_ctx
        dm, rd = ctx["dm"], ctx["rd"]
        self.fam = fam
        self.words = np.asarray(fam.words if words is None else words, np.int64)
        self.tree = fam.tree
        z = np.load(fam.npz, allow_pickle=False)
        w = np.asarray(z["W"], np.float64)
        W = np.asarray(z["H2_I"], np.float64) - w.sum(1)
        self.w = w
        R, B, C = w.shape
        oei, tei = np.asarray(rd["oei"], float), np.asarray(rd["tei"], float)
        N = oei.shape[0]
        self.N, self.R, self.B = N, R, B
        S, beta1 = ctx["S"], ctx["beta1"]
        h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
        self.kap = np.einsum("rc,rbc->rb", W, w).ravel()
        u = self._decode(self.words)
        self.u = u
        self.one, self.ub = u[:N], u[N:].reshape(R, B, N)
        X = dm.reconstruct_x(self.ub, w)
        self.h1p = dm.build_h1_prime(h1_thc, S, beta1, rd["electrons"], W, X)
        lam, V = np.linalg.eigh(self.h1p)
        qi, ii = linear_sum_assignment(-np.abs(self.one @ V))
        m = np.empty(N)
        m[qi] = lam[ii]
        self.m, self.lam0 = m, lam
        self.R1 = np.einsum("j,jp,jq->pq", m, self.one, self.one) - self.h1p
        self.e1sq = float(np.sum(self.R1 ** 2))
        self.sabs0 = np.abs(lam).sum()
        D = dm.reconstruct_h2(X)
        D -= tei
        dg = np.arange(N)
        D[:, :, dg, dg] -= 0.5 * S[:, :, None]
        D[dg, dg, :, :] -= 0.5 * S[None, :, :]
        Xf = X.reshape(R * C, N * N)
        M = (D.reshape(N * N, N * N) @ Xf.T).reshape(N * N, R, C)
        del D
        self.A = np.einsum("rbc,prc->rbp", w, M).reshape(R, B, N, N)
        del M
        Xr = X.reshape(R, C, N * N)
        G = np.einsum("rcp,rdp->rcd", Xr, Xr)
        self.gw = np.einsum("rbc,rcd,rbd->rb", w, G, w)
        self.Y = np.einsum("rbc,rcp->rbp", w, Xr).reshape(R, B, N, N)
        self.uA = np.einsum("rbp,rbpq,rbq->rb", self.ub, self.A, self.ub)
        self.uY = np.einsum("rbp,rbpq,rbq->rb", self.ub, self.Y, self.ub)
        self.r1one = np.einsum("jp,pq,jq->j", self.one, self.R1, self.one)

    def _decode(self, words):
        dm = self.fam.official_ctx["dm"]
        u = dm.reconstruct_u_vectors(np.asarray(words) / (1 << (self.fam.brot - 1)), self.tree)
        return u / np.linalg.norm(u, axis=1, keepdims=True)

    def components(self, col, cand):
        """Per row (dLambda, dE_gap, d e2^2, d e1^2) of setting column `col` to `cand` (one value per row)."""
        N, N1, B = self.N, self.N, self.B
        cand = np.asarray(cand, np.int64)
        w2 = self.words.copy()
        w2[:, col] = cand
        chg = np.flatnonzero(cand != self.words[:, col])
        comp = np.zeros((len(cand), 4))
        if not len(chg):
            return comp
        u2 = self._decode(w2)
        dot = np.einsum("ip,ip->i", self.u, u2)
        ob = chg[chg < N1]
        if len(ob):
            o2 = u2[ob]
            ip = np.einsum("jp,pq,jq->j", o2, self.R1, o2) - self.r1one[ob]
            comp[ob, 3] = 2 * self.m[ob] * ip + self.m[ob] ** 2 * (2 - 2 * dot[ob] ** 2)
        tb = chg[chg >= N1] - N1
        if len(tb):
            ri, bi = tb // B, tb % B
            t2 = u2[N1 + tb]
            At, Yt = self.A[ri, bi], self.Y[ri, bi]
            lin = np.einsum("np,npq,nq->n", t2, At, t2) - self.uA[ri, bi]
            yq = np.einsum("np,npq,nq->n", t2, Yt, t2) - self.uY[ri, bi]
            dp2 = 2 - 2 * dot[N1 + tb] ** 2
            comp[N1 + tb, 2] = 4 * lin + 2 * dp2 * self.gw[ri, bi] + 2 * yq ** 2
            old = self.ub.reshape(-1, N)[tb]
            dP = t2[:, :, None] * t2[:, None, :] - old[:, :, None] * old[:, None, :]
            H = self.h1p[None] - self.kap[tb, None, None] * dP
            lam, V = np.linalg.eigh(H)
            sabs = np.abs(lam).sum(1)
            comp[N1 + tb, 0] = sabs - self.sabs0
            comp[N1 + tb, 1] = (sabs - lam.sum(1)) - (self.sabs0 - self.lam0.sum())
            ov = np.abs(np.einsum("jp,npk->njk", self.one, V))
            for t in range(len(tb)):
                qi, ii = linear_sum_assignment(-ov[t])
                m = np.empty(N)
                m[qi] = lam[t, ii]
                comp[N1 + tb[t], 3] = np.sum((self.one.T @ (m[:, None] * self.one) - H[t]) ** 2) - self.e1sq
        return comp

    def prices(self, col, cand):
        """Per-row relative lambda_eff price (refit-corrected first order) of setting column `col` to `cand`, and
        the components. E_gap = E_up - E_SOS with E_SOS = const - (lambda_h1 - sum ev) - ..., so d E_gap is the change
        of (lambda_h1 - sum ev) at fixed weights; d Lambda is the change of lambda_h1 = sum |ev|."""
        c = self.components(col, cand)
        return first_order_rel(self.fam, c[:, 0], c[:, 1], c[:, 2], c[:, 3]), c


# ------------------------------------------------------------------------------------------ pricing of moves


def exact_fixed(fam: Family, words) -> dict:
    """Official metrics of `words` at the base weights and their refit-corrected first-order relative price."""
    P = fam.problem_for(words)
    m = P.official(fam.y0)
    b = fam.base
    e1_bind = m["error_1"] > dr.ERROR_LIMIT - 5e-7
    rel = float(first_order_rel(fam, m["Lambda"] - b["Lambda"], m["E_gap"] - b["E_gap"],
                                m["error_2"] ** 2 - b["error_2"] ** 2, m["error_1"] ** 2 - b["error_1"] ** 2,
                                e1_binding=e1_bind))
    return dict(metrics=m, rel=rel, e1_binding=bool(e1_bind),
                unchanged=all(m[k] == b[k] for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff")))


def price_move(fam: Family, move: Move, *, refit=True, max_evals=300, control=None, log=None,
               kink_tol=dr.KINK_TOL, method="retract") -> dict:
    """Price one move: exact fixed-weight metrics (+ first-order correction) and, with `refit`, the bounded-refit
    price against `control` (a bounded_refit of the base table at the same budget; computed if None)."""
    words = apply_move(fam.words, move, fam.brot)
    ex = exact_fixed(fam, words)
    b = fam.base
    row = dict(move=move.name, n_changes=len(move.changes), meta=move.meta,
               fixed=dict(error_1_mHa=ex["metrics"]["error_1"] * 1e3, error_2_mHa=ex["metrics"]["error_2"] * 1e3,
                          E_gap=ex["metrics"]["E_gap"], Lambda=ex["metrics"]["Lambda"],
                          lambda_eff=ex["metrics"]["lambda_eff"],
                          d_error_2_uHa=(ex["metrics"]["error_2"] - b["error_2"]) * 1e6,
                          d_error_1_uHa=(ex["metrics"]["error_1"] - b["error_1"]) * 1e6),
               metrics_unchanged=ex["unchanged"], first_order_rel=ex["rel"],
               first_order_units=fam.units(ex["rel"]), e1_binding=ex["e1_binding"])
    if ex["unchanged"]:
        row.update(refit_rel=0.0, refit_units=0.0, refit=dict(note="metrics bit-identical: exact zero price"))
        return row
    if refit:
        if control is None:
            control = bounded_refit(fam, None, max_evals=max_evals, kink_tol=kink_tol, method=method)
        ref = min(b["lambda_eff"], control["lambda_eff"])
        r = bounded_refit(fam, words, max_evals=max_evals, log=log, kink_tol=kink_tol, method=method)
        rel = (r["lambda_eff"] - ref) / ref
        row.update(refit_rel=rel, refit_units=fam.units(rel) if math.isfinite(rel) else math.inf,
                   refit=dict(lambda_eff=r["lambda_eff"], reference_lambda_eff=ref, passes=r["passes"],
                              stop=r["stop"], evals=r["evals"], seconds=r["seconds"], source=r["source"],
                              error_1_mHa=None if r["metrics"] is None else r["metrics"]["error_1"] * 1e3,
                              error_2_mHa=None if r["metrics"] is None else r["metrics"]["error_2"] * 1e3,
                              E_gap=None if r["metrics"] is None else r["metrics"]["E_gap"],
                              last_point=r["last"]))
    return row


def rank_moves(fam: Family, moves, planner=None, *, refit_top=10, max_evals=300, log=None,
               kink_tol=dr.KINK_TOL, method="retract") -> list:
    """Price every move (exact fixed + first order), ask `planner(old_words, new_words, move)` for its loader saving
    in Toffolis per pass, and run the bounded refit on the `refit_top` moves with the best first-order net. Rows are
    sorted by net = saving_units - price_units (refit price where computed, else first order); best first."""
    rows = []
    for mv in moves:
        r = price_move(fam, mv, refit=False)
        sv = 0.0
        if planner is not None:
            sv = float(planner(fam.words, apply_move(fam.words, mv, fam.brot), mv))
        r["saving_toffolis_per_pass"] = sv
        r["saving_units"] = sv * fam.loader_toffoli_units
        r["net_first_order_units"] = r["saving_units"] - r["first_order_units"]
        rows.append((r, mv))
    rows.sort(key=lambda t: -t[0]["net_first_order_units"])
    control = None
    for r, mv in rows[:refit_top]:
        if r["metrics_unchanged"]:
            r["refit_units"] = 0.0
            continue
        if control is None:
            control = bounded_refit(fam, None, max_evals=max_evals, kink_tol=kink_tol, method=method)
        rr = price_move(fam, mv, refit=True, max_evals=max_evals, control=control, log=log, kink_tol=kink_tol,
                        method=method)
        r.update({k: rr[k] for k in ("refit_rel", "refit_units", "refit")})
    out = []
    for r, _ in rows:
        pu = r.get("refit_units", r["first_order_units"])
        r["price_units"] = pu
        r["price_tier"] = "refit" if "refit_units" in r else "first_order"
        r["net_units"] = r["saving_units"] - pu
        out.append(r)
    out.sort(key=lambda r: -r["net_units"])
    return out


# ------------------------------------------------------------------------------------------- GF(2) planner side


def mobius(x, axis=0):
    """GF(2) Moebius (ANF) transform along `axis` (length a power of two)."""
    x = np.array(x, dtype=np.uint8, copy=True)
    n = x.shape[axis]
    assert n & (n - 1) == 0
    x = np.moveaxis(x, axis, 0)
    h = 1
    while h < n:
        y = x.reshape(n // (2 * h), 2, h, *x.shape[1:])
        y[:, 1] ^= y[:, 0]
        x = y.reshape(x.shape)
        h *= 2
    return np.moveaxis(x, 0, axis)


def anf_tensor(delta, brot=8, lowbits=4):
    """C (2**lowbits, H, outputs): ANF over (low monomial, high monomial < H, output), output (brot - 1) j + k is
    bit k of word j (the signed-gauge top bit is dropped). Rows past the table are zero; {0..H-1} is closed under
    bit subsets, so the high monomials below H are exact. (Same convention as Lera's tensorqrom.anf_tensor.)"""
    delta = np.asarray(delta, np.int64)
    count, J = delta.shape
    L = 1 << lowbits
    H = -(-count // L)
    hb = max(1, (H - 1).bit_length())
    bits = np.zeros((L * H, J * (brot - 1)), np.uint8)
    for j in range(J):
        for k in range(brot - 1):
            bits[:count, j * (brot - 1) + k] = (delta[:, j] >> k) & 1
    truth = bits.reshape(H, L, -1)
    padded = np.zeros((1 << hb, L, truth.shape[2]), np.uint8)
    padded[:H] = truth
    C = mobius(mobius(padded, 0), 1)[:H]
    return C.transpose(1, 0, 2).copy()


def gf2_rank(rows, backend=None) -> int:
    """Rank over GF(2) of integer bit-rows. backend: "python" (reference) or "rust" (kickmix_analysis);
    default recover.GF2_BACKEND."""
    from recover import gf2_backend, rust_module
    if gf2_backend(backend) == "rust":
        return rust_module().gf2_rank(rows)
    piv = {}
    r = 0
    for v in rows:
        v = int(v)
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p]
            else:
                piv[p] = v
                r += 1
                break
    return r


def _rowbits(vec):
    return int("".join("1" if b else "0" for b in vec[::-1]) or "0", 2)


def mixed_rank(C, outputs) -> int:
    """GF(2) rank of the mixed part of `outputs`: rows (output, low monomial != 1), columns high monomials != 1."""
    rows = [_rowbits(C[l, 1:, o]) for o in outputs for l in range(1, C.shape[0])]
    return gf2_rank(rows)


def coded_outputs(coded_words, brot=8):
    return [(brot - 1) * j for j in coded_words]


def _free_pad_slopes(real_rows, lowbits=4):
    """Low bits i whose slope on the last, partial block is free (no real row there has bit i set)."""
    n = real_rows % (1 << lowbits)
    if n == 0:
        return []
    return [i for i in range(lowbits) if (1 << i) >= n]


def coded_mixed_rank(delta, coded_words, *, brot=8, lowbits=4, real_rows=None, pad_free=False):
    """GF(2) mixed rank of the coded bit-0 outputs of one loader delta. With `pad_free`, the slopes of the last,
    partial block that no real row fixes are chosen to minimise the rank (the never-read padded rows are free; their
    change touches only the top high monomial's linear low terms). Returns (rank, chosen free bits or None)."""
    C = anf_tensor(delta, brot, lowbits)
    outs = coded_outputs(coded_words, brot)
    rows = {(o, l): _rowbits(C[l, 1:, o]) for o in outs for l in range(1, C.shape[0])}
    base = gf2_rank(rows.values())
    if not pad_free or real_rows is None:
        return base, None
    free = [(o, 1 << i) for o in outs for i in _free_pad_slopes(real_rows, lowbits)]
    if not free:
        return base, None
    H = C.shape[1]
    bit = 1 << (H - 2)          # column of high monomial H - 1 in the mixed rows (monomial 1.. -> bit 0..)
    best = (base, 0)
    keys = list(rows)
    for mask in range(1, 1 << len(free)):
        rr = dict(rows)
        for t, key in enumerate(free):
            if mask >> t & 1:
                rr[key] ^= bit
        r = gf2_rank(rr[k] for k in keys)
        if r < best[0]:
            best = (r, mask)
    return best[0], best[1]


def loader_coded_ranks(words, *, j=dr.LERA_J, brot=8, lowbits=4, loaders=None, pad_free=False):
    """Per loader k: (coded word slots, GF(2) mixed rank of their bit-0 outputs) on dfthc_refit.loader_delta; with
    `pad_free` the rank is minimised over the free padded slopes (see coded_mixed_rank)."""
    words = np.asarray(words, np.int64)
    n = -(-words.shape[1] // j)
    out = {}
    for k in (range(n) if loaders is None else loaders):
        delta, cw = dr.loader_delta(words, k, j)
        if not cw:
            out[k] = (cw, 0)
            continue
        r, _ = coded_mixed_rank(delta, cw, brot=brot, lowbits=lowbits, real_rows=len(words), pad_free=pad_free)
        out[k] = (list(cw), r)
    return out


def coded_rank_planner(*, j=dr.LERA_J, brot=8, lowbits=4, pad_free=True):
    """Planner callback (static model): saving in Toffolis per pass = sum over loaders of ceil(r_old / 2) -
    ceil(r_new / 2), r the coded bit-0 mixed rank. Only loaders whose columns the move touches are recomputed; a
    loader whose coded set changes (a column stops being coded) counts its whole coded rank change."""
    def planner(old, new, move):
        cols = sorted({int(c) for _, c, _ in move.changes})
        ks = sorted({c // j for c in cols} | {c // j + 1 for c in cols})
        n = -(-np.asarray(old).shape[1] // j)
        ks = [k for k in ks if k < n]
        a = loader_coded_ranks(old, j=j, brot=brot, lowbits=lowbits, loaders=ks, pad_free=pad_free)
        b = loader_coded_ranks(new, j=j, brot=brot, lowbits=lowbits, loaders=ks, pad_free=pad_free)
        sv = 0
        for k in ks:
            if a[k][0] != b[k][0]:
                return -math.inf   # the coded set changed: the planner's structure breaks (not priced here)
            sv += -(-a[k][1] // 2) - (-(-b[k][1] // 2))
        return sv
    return planner


# ------------------------------------------------------------------------------------- rank-drop construction


def block_slopes(delta, coded_words, lowbits=4):
    """Slopes s[o, h, i] of the affine bit 0 of each coded delta word o on 16-address block h (i = low bit)."""
    delta = np.asarray(delta, np.int64)
    L = 1 << lowbits
    H = len(delta) // L
    out = np.zeros((len(coded_words), H, lowbits), np.uint8)
    for a, wj in enumerate(coded_words):
        b = (delta[:H * L, wj] & 1).reshape(H, L)
        for i in range(lowbits):
            out[a, :, i] = b[:, 1 << i] ^ b[:, 0]
    return out


def rank_drop_moves(fam: Family, k, *, row_price=None, j=dr.LERA_J, lowbits=4, max_terms=4, keep=64,
                    protect=(), words=None, restarts=0, seed=0, frozen_terms=()):
    """Candidate moves lowering loader k's coded mixed rank by one with bit-0 toggles (word XOR 1).

    For a nonzero combination x of the coded outputs' slope vectors (output o, low bit i), the rank falls when
    g_h = sum_x s[o, h, i] is constant over the high blocks h: on each block where g_h differs from the chosen
    constant, one slope in x is flipped by toggling bit 0 on the 8 rows of the block with l_i = 1 (or the 8 with
    l_i = 0, which also flips the block constant, a free change). A delta word is B_k xor B_(k-1), so either word
    column may carry the toggle; the cheaper is used (columns of loaders in `protect` are not touched). The last,
    partial block is never edited (its padded slopes are fixed by the affine padding).

    `row_price` is a (rows, columns) array, or a mapping column -> per-row array, of the relative price of toggling
    bit 0 of each entry (RowPricer.prices on words ^ 1); default: unit price per row. Combinations x with up to `max_terms` terms are enumerated; returns the
    `keep` cheapest as Move objects (meta: loader, x, constant, blocks, predicted price). With `restarts` > 0, a
    local search over combinations of any size (one term in or out per step, random restarts) is added: larger
    combinations give more choice per mismatched block. `frozen_terms` ((o, i), ...) are slopes that are never
    edited (pass a previous drop's x so that a second drop keeps the first's dependency). The drop is guaranteed when the rank is full (no existing
    dependency the toggles could break); in general check it with the planner callback."""
    import itertools
    words = np.asarray(fam.words if words is None else words, np.int64)
    A, ncol = words.shape
    L = 1 << lowbits
    delta, cw = dr.loader_delta(words, k, j)
    if not cw:
        return []
    s = block_slopes(delta, cw, lowbits)
    nfull = A // L               # blocks with all 16 real rows (editable)
    O = len(cw)
    # cost of flipping slope (o, i) on block h: cheaper word column and polarity
    INF = math.inf
    cost = np.full((O, nfull, lowbits), INF)
    how = {}
    lowpos = np.arange(L)
    for a, wj in enumerate(cw):
        cols = [j * k + wj] + ([j * (k - 1) + wj] if k > 0 else [])
        for col in cols:
            feeds = {col // j, col // j + 1} - {k}
            if col >= ncol or feeds & set(protect):
                continue
            if row_price is None:
                rp = np.ones(A)
            elif isinstance(row_price, np.ndarray) and row_price.ndim == 2:
                rp = row_price[:, col]
            else:
                rp = np.asarray(row_price[col])
            for i in range(lowbits):
                for pol in (1, 0):
                    sel = ((lowpos >> i) & 1) == pol
                    pr = rp[:nfull * L].reshape(nfull, L)[:, sel].sum(1)
                    better = pr < cost[a, :, i]
                    cost[a, better, i] = pr[better]
                    for h in np.flatnonzero(better):
                        how[(a, int(h), i)] = (col, pol)
    for a_, i_ in frozen_terms:               # slopes another drop relies on: never edited
        cost[int(a_), :, int(i_)] = INF
    terms = [(a, i) for a in range(O) for i in range(lowbits)]
    sv = np.stack([s[a, :nfull, i] for a, i in terms])     # (terms, nfull)
    last = s[:, nfull:, :]                                   # the partial block's slopes (fixed)
    def score(x):
        x = tuple(sorted(x))
        g = np.bitwise_xor.reduce(sv[list(x)], axis=0)
        if last.shape[1]:
            consts = [int(np.bitwise_xor.reduce([last[terms[q][0], 0, terms[q][1]] for q in x]))]
        else:
            consts = [0, 1]
        out = []
        for cst in consts:
            bad = np.flatnonzero(g != cst)
            if not len(bad):
                continue
            cmat = np.stack([cost[terms[q][0], bad, terms[q][1]] for q in x])
            best = cmat.argmin(0)
            tot = float(cmat[best, np.arange(len(bad))].sum())
            out.append((tot, x, cst, bad, best))
        return min(out, key=lambda c: c[0]) if out else (math.inf, x, None, None, None)

    seen = {}
    for t in range(1, max_terms + 1):
        for x in itertools.combinations(range(len(terms)), t):
            seen[x] = score(x)
    # local search over subsets of any size (flip one term in or out while the price falls), random restarts
    rng = np.random.default_rng(seed)
    nt = len(terms)
    for _ in range(restarts):
        cur = tuple(sorted(int(q) for q in np.flatnonzero(rng.random(nt) < rng.uniform(0.1, 0.6)))) or (0,)
        sc = seen.get(cur) or score(cur)
        seen[cur] = sc
        improved = True
        while improved:
            improved = False
            for q in rng.permutation(nt):
                y = tuple(sorted(set(cur) ^ {int(q)}))
                if not y:
                    continue
                sy = seen.get(y)
                if sy is None:
                    sy = score(y)
                    seen[y] = sy
                if sy[0] < sc[0]:
                    cur, sc, improved = y, sy, True
    cands = [c for c in seen.values() if math.isfinite(c[0])]
    cands.sort(key=lambda c: c[0])
    moves = []
    for tot, x, cst, bad, best in cands[:keep]:
        changes = {}
        for h, q in zip(bad, best):
            a, i = terms[x[q]]
            col, pol = how[(a, int(h), i)]
            for l in range(L):
                if ((l >> i) & 1) == pol:
                    r = int(h) * L + l
                    key = (r, col)
                    changes[key] = changes.get(key, 0) ^ 1
        ch = tuple((r, c, int(words[r, c] ^ 1)) for (r, c), f in sorted(changes.items()) if f)
        moves.append(Move(f"k{k}_x{'-'.join(str(q) for q in x)}_c{cst}", ch,
                          dict(loader=k, x=[list(terms[q]) for q in x], constant=cst, blocks=[int(b) for b in bad],
                               predicted_rel=tot if row_price is not None else None)))
    return moves
