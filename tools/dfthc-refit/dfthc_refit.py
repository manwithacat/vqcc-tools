#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the author's direction.
"""dfthc_refit: refit the DF-THC weights of a dfthcsos submission on its fixed angle words.

For the Kaggle competition "Google Quantum AI - Verifiable Quantum Circuit Compilation", task dfthcsos (FeMoCo).
Method and results: technical note https://doi.org/10.5281/zenodo.23930466.
The ANGLE_k loaders and the words they load stay fixed, so the qubit and Toffoli counts stay fixed; only the two-body
weights w (R, B, C) and the offsets W (R, C) of submission.npz move. The tool minimises

    lambda_eff = sqrt(E_gap (2 Lambda - E_gap))

subject to error_1 <= 20 mHa, error_2 <= 20 mHa and E_gap above its floor, by alternating

  1. a constrained refit: projected L-BFGS on log(lambda_eff) with the three constraints held as equalities at
     targets just inside the limits, a Gauss-Newton retraction onto the constraint set after every step, Armijo
     backtracking, and a KKT check that releases weights held at zero when that lowers lambda_eff;
  2. a gauge step: for each rank, right-multiplying [w_r ; w_B,r] by an orthogonal C x C matrix leaves error_1,
     error_2 and E_gap unchanged but changes Lambda; exact Jacobi rotation sweeps (plus a smoothed-L1 L-BFGS start)
     minimise Lambda over that freedom;

until the first-order gain predicted from the KKT multipliers and the realised gain of a round are both below 1e-5
relative, or a round limit. It is a local method on a non-convex problem: no global optimality is claimed.

Every reported metric comes from the official verifier code (dfthcsos.scoreinfra.dfthc_math, unmodified), imported
from the competition's verifiers directory. The words come from what the loaders actually load: the official
dfthcsos.verify.evaluate_circuits (needs a kickmix build); only if no kickmix build is available does the tool fall
back to the npz's angle_words, with a loud warning.

Usage:
    python dfthc_refit.py SUBMISSION_DIR OUT_DIR [--verifier-dir PATH] [--kickmix-dir PATH] [--max-evals N]
        [--rounds N] [--arm A|B] [--e1t start|MHA] [--e2t MHA] [--gap-margin HA] [--threads N] [--seed N]

    import dfthc_refit
    report = dfthc_refit.run("in/dfthcsos", "out", verifier_dir=".../verifiers", max_evals=300, rounds=3)

Output: OUT_DIR/dfthcsos/ (the loaders copied unchanged + the refit submission.npz) and OUT_DIR/report.json.
Same inputs and settings (including --threads) give the same submission.npz bytes.

Loader re-planning (v1.3.0-dev, `replan` mode; words and weights never change):
    python dfthc_refit.py replan FAMILY_DIR OUT_DIR --planner QROM_PLAN_BINARY --lera-src DIR_WITH_tensorqrom.py
        [--verifier-dir PATH] [--kickmix-dir PATH] [--recipes R.json] [--slots 0-3] [--procs 6] [--verify]
    python dfthc_refit.py replan FAMILY_DIR OUT_DIR --backend kernel --kernel-src DIR_WITH_kernel_search.py
        --lera-src DIR_WITH_tensorqrom.py [--kernel-seeds 0,1] [--pad-rows [--no-pad-screen]] [--verify]
re-plans each ANGLE_k loader with the qrom-plan relation planner (a separate Rust binary) or qwertykeypadapple's
kernel-relation search (loaded unmodified by path; --pad-rows lets the never-read padded table rows take free
values), emits it with Lera's unmodified emitter and keeps, per slot, the re-planned loader only if it has strictly
fewer Toffolis (see replan()).
    python dfthc_refit.py replan FAMILY_DIR OUT_DIR --backend shared|block --lera-src DIR [--seeds 0,1]
        [--blocks recover|search|1,5,13/...] [--exchange] [--pad-rows] [--verify]
runs this tool's block-partition planners (planner_shared.py / planner_block.py) instead (v1.4.0-dev).
    python dfthc_refit.py replan FAMILY_DIR OUT_DIR --plans-from PLANS_DIR --lera-src DIR [--slots 8,11] [--verify]
replays saved plan JSONs instead of searching (v1.7.0-dev): no planner, no CPU caps, deterministic.

Family subcommands (v1.4.0-dev; each reads one family, none changes it):
    python dfthc_refit.py conventions FAMILY_DIR [--out FILE.json]    (v1.5.0-dev) word conventions: storage range,
                                           tree, gauge class (signed / pi / free range), loader class per slot
                                           (XOR-in / chained); every family subcommand runs it first (gauge.py)
    python dfthc_refit.py plan FAMILY_DIR --planner kernel|shared|block [--pad-rows] [--exchange] [--seeds 0,1]
        [--out DIR]                        per-slot plans and predicted Toffolis (static model / projection)
    python dfthc_refit.py recover FAMILY_DIR [--slots 0-3] [--emit-check --lera-src DIR] [--out DIR]
                                           structure census of the shipped loaders (recover.py)
    python dfthc_refit.py price-words FAMILY_DIR [--moves FILE | --rank-drop K] [--refit-method retract|penalty]
        [--recover-dir DIR] [--out DIR]    shadow prices and priced word moves (price_words.py; needs PyTorch)
    python dfthc_refit.py chained-and TARGET.json bounds|verify|encode ...   (chained_and.py)
    python dfthc_refit.py gauge-options FAMILY_DIR --slot K [--plans DIR --lera-src DIR [--emit-dir D]]
        [--structural N] [--out FILE.json]  (v1.6.0-dev, gauge_options.py) per-loader gauge characterisation: the
                                           slot's exact word symmetries split into free / constrained / costed
                                           directions, its gauge contract (p_in, p_out, address-only, rank of the
                                           gauge change), qubit prices, suggested repairs; --plans emits a replan's
                                           plans in both gauge layouts (kept top bits / in place) and checks them
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # thread caps must be set before NumPy/BLAS load
    _t = "3"
    for _i, _a in enumerate(sys.argv):
        if _a == "--threads" and _i + 1 < len(sys.argv):
            _t = sys.argv[_i + 1]
        elif _a.startswith("--threads="):
            _t = _a.split("=", 1)[1]
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_k, _t)

import argparse
import glob
import hashlib
import importlib
import json
import math
import platform
import shutil
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

__version__ = "1.8.0-dev"

ERROR_LIMIT = 0.020        # Ha, error_1 and error_2 (official)
E_GAP_FLOOR = 3.914898     # Ha, strict (official)
LAMBDA_FLOOR = 49.0        # strict (official)
REL_TIE = 1e-5             # relative lambda_eff: below this a gain cannot change a ranking
ZERO_HOLD = 1e-15          # coordinates below this start held at exactly 0 (v1.1.0: was 1e-6, which zeroed thousands
                           # of deliberate tiny weights in public refits and cost up to ~1e-5 in lambda_eff)
FEAS_E_TOL_MHA = 2e-4      # refit-internal feasibility slack on the targets (targets sit inside the limits)
FEAS_GAP_TOL = 2e-6
KINK_TOL = 1e-6            # v1.1.0: hold an h1' eigenvalue this close to 0 (the |ev| kink) as a constraint
E1_CAP_MHA = 19.9995        # v1.1.0: cap for an inequality error_1 / for holding a start's error_2 (official limit 20)


# ---------------------------------------------------------------------------------------------------------- locating


def find_verifier_dir(explicit: str | os.PathLike | None = None) -> Path:
    """The competition's `verifiers` directory (holding dfthcsos/verify.py): `explicit`, $DFTHC_REFIT_VERIFIER_DIR,
    $QHDL_OFFICIAL_ROOT/verifiers, <repo>/data/official_full/verifiers, kickmix_tools.competition_dir()/verifiers
    when kickmix_tools is importable, else a search under /kaggle/input. The verifiers are not shipped with this
    tool: download the competition data."""
    cands = []
    for p in (explicit, os.environ.get("DFTHC_REFIT_VERIFIER_DIR")):
        if p:
            cands.append(Path(p))
    if os.environ.get("QHDL_OFFICIAL_ROOT"):
        cands.append(Path(os.environ["QHDL_OFFICIAL_ROOT"]) / "verifiers")
    cands.append(Path(__file__).resolve().parents[2] / "data" / "official_full" / "verifiers")
    try:
        import kickmix_tools as kt  # optional
        cands.append(Path(kt.competition_dir()) / "verifiers")
    except Exception:
        pass
    cands += [Path(p).parent.parent for p in sorted(glob.glob("/kaggle/input/**/verifiers/dfthcsos/verify.py",
                                                              recursive=True))]
    for c in cands:
        if (c / "dfthcsos" / "verify.py").is_file():
            return c.resolve()
        if c.name == "dfthcsos" and (c / "verify.py").is_file():
            return c.parent.resolve()
    raise FileNotFoundError("official verifiers directory not found: pass --verifier-dir (the directory that holds "
                            "dfthcsos/verify.py) or attach the competition data")


def _official_modules(verifier_dir: Path):
    """dfthc_math and reference from the official package (neither imports kickmix)."""
    pkg = Path(verifier_dir) / "dfthcsos"
    loaded = sys.modules.get("dfthcsos")
    if loaded is not None and Path(loaded.__file__).resolve().parent != pkg.resolve():
        raise ImportError(f"a different dfthcsos package is already imported: {loaded.__file__}")
    if str(verifier_dir) not in sys.path:
        sys.path.insert(0, str(verifier_dir))
    dm = importlib.import_module("dfthcsos.scoreinfra.dfthc_math")
    ref = importlib.import_module("dfthcsos.scoreinfra.reference")
    return dm, ref


def find_kickmix(explicit: str | os.PathLike | None = None, verifier_dir: Path | None = None):
    """Import kickmix (needed only to recover words from the loaders and to verify). Tries an existing import, then
    `explicit`, $DFTHC_REFIT_KICKMIX_DIR, kickmix_tools' official and fork builds, and <competition>/kickmix."""
    def _real():
        m = sys.modules.get("kickmix")
        return m is not None and hasattr(m, "Circuit")

    def _forget():
        # v1.1.1: on Kaggle a bare `kickmix` directory (a namespace package with no extension) can shadow the build
        for k in [k for k in sys.modules if k == "kickmix" or k.startswith("kickmix.")]:
            del sys.modules[k]

    try:
        import kickmix  # noqa: F401
        if _real():
            return sys.modules["kickmix"]
        _forget()
    except ImportError:
        pass
    cands = [explicit, os.environ.get("DFTHC_REFIT_KICKMIX_DIR")]
    try:
        import kickmix_tools as kt
        cands += [kt.official_build(), kt.fork_build()]
    except Exception:
        pass
    if verifier_dir is not None:
        cands.append(Path(verifier_dir).parent / "kickmix")
    for c in cands:
        if not c or not Path(c).is_dir():
            continue
        if not (list(Path(c).glob("kickmix*.so")) or list(Path(c).glob("kickmix*.pyd"))):
            continue
        sys.path.insert(0, str(c))
        try:
            import kickmix  # noqa: F401
            if _real():
                return sys.modules["kickmix"]
            _forget()
        except ImportError:
            pass
        sys.path.remove(str(c))
    return None


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def loader_files(sub_dir: Path) -> list[Path]:
    fs = [p for p in Path(sub_dir).iterdir() if p.is_file() and p.name.startswith("angle_k_")
          and p.suffix in (".kmb", ".kmx")]
    return sorted(fs, key=lambda p: (int(p.stem.split("_")[-1]), p.suffix))


# -------------------------------------------------------------------------------------------- deterministic output


def write_npz(path, arrays: dict) -> None:
    """np.savez layout (one stored .npy member per key, in order) with fixed zip timestamps, so equal arrays give
    equal bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for k, v in arrays.items():
            zi = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(v), allow_pickle=False)


# ------------------------------------------------------------------------------------------------------- the problem


class Problem:
    """The fixed-word weight problem. Coordinates y = (w (R,B,C) raveled, w_B (R,C) raveled), w_B = W + sum_b w, so
    both L1 cusps of Lambda are coordinate cusps. `official(w, W)` returns the official metrics dict
    (error_1, error_2, Lambda, E_gap, lambda_eff, gap_in_domain); the torch model below re-implements the same
    formulas (error_2 in the packed pair basis) for gradients only."""

    def __init__(self, u_all, w0, wB0, S, beta1, oei, tei, electrons, ecore, energy_upper, official, *,
                 gap_floor=E_GAP_FLOOR, lambda_floor=LAMBDA_FLOOR, error_limit=ERROR_LIMIT):
        import torch
        torch.set_default_dtype(torch.float64)
        self.torch = torch
        self.w0 = np.asarray(w0, np.float64)
        self.wB0 = np.asarray(wB0, np.float64)
        self.R, self.B, self.C = self.w0.shape
        self.NW = self.R * self.B * self.C
        S = np.asarray(S, np.float64)
        oei, tei = np.asarray(oei, np.float64), np.asarray(tei, np.float64)
        N = oei.shape[0]
        self.N = N
        self.official_fn = official
        self.kink_basis = None
        self.gap_floor, self.lambda_floor, self.error_limit = float(gap_floor), float(lambda_floor), float(error_limit)
        h1 = oei - 0.5 * np.einsum("prrq->pq", tei)
        self.A = torch.tensor(h1 - 0.5 * electrons * S + beta1 * np.eye(N))
        self.CONST = float(ecore - electrons * beta1)
        self.EUP = float(energy_upper)
        t = tei.copy()
        d_ = np.arange(N)
        t[:, :, d_, d_] += 0.5 * S[:, :, None]
        t[d_, d_, :, :] += 0.5 * S[None, :, :]
        rr, cc = np.tril_indices(N)
        sc = np.where(rr == cc, 1.0, math.sqrt(2.0))
        self.T = torch.tensor(t.reshape(N * N, N * N)[np.ix_(rr * N + cc, rr * N + cc)] * np.outer(sc, sc))
        del t
        u = np.asarray(u_all, np.float64)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        self.V = torch.tensor(u[:N])
        self.ub = torch.tensor(u[N:].reshape(self.R, self.B, N))
        self.PB = self.ub[..., torch.tensor(rr)] * self.ub[..., torch.tensor(cc)] * torch.tensor(sc)

    # coordinates
    def y0(self):
        return np.concatenate([self.w0.ravel(), self.wB0.ravel()])

    def from_weights(self, w, wB):
        return np.concatenate([np.asarray(w, np.float64).ravel(), np.asarray(wB, np.float64).ravel()])

    def split(self, y):
        y = np.asarray(y)
        w = y[:self.NW].reshape(self.R, self.B, self.C)
        wB = y[self.NW:].reshape(self.R, self.C)
        return w, wB

    def h1p_np(self, y):
        """h1' at y (NumPy), for picking the kink block."""
        w, wB = self.split(y)
        off = wB - w.sum(1)
        s = np.einsum("rbc,rc->rb", w, off)
        ub = self.ub.numpy()
        h = self.A.numpy() - np.einsum("rb,rbp,rbq->pq", s, ub, ub)
        return 0.5 * (h + h.T)

    def official(self, y):
        w, wB = self.split(y)
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                for k, v in self.official_fn(w, wB - w.sum(1)).items()}

    def passes(self, m) -> bool:
        return bool(m["error_1"] <= self.error_limit and m["error_2"] <= self.error_limit and m["gap_in_domain"]
                    and np.isfinite(m["lambda_eff"]) and m["E_gap"] > self.gap_floor
                    and m["Lambda"] > self.lambda_floor)

    def evaluate(self, y):
        """Torch metrics and gradients (log lam, c1 = e1^2, c2 = e2^2, gap) at y. torch's |x|' = 0 at x = 0, so
        gradients on zero coordinates are the smooth part only; `allow` is the L1 subgradient half-width in log-lam
        units."""
        torch = self.torch
        R, B, Cc, NW = self.R, self.B, self.C, self.NW
        yt = torch.tensor(y, requires_grad=True)
        w = yt[:NW].reshape(R, B, Cc)
        wB = yt[NW:].reshape(R, Cc)
        off = wB - w.sum(1)
        X = torch.einsum("rbm,rbc->rcm", self.PB, w).reshape(R * Cc, -1)
        E = X.T @ X - self.T
        c2 = (E * E).sum()
        s = torch.einsum("rbc,rc->rb", w, off)
        h1p = self.A - torch.einsum("rb,rbp,rbq->pq", s, self.ub, self.ub)
        h1p = 0.5 * (h1p + h1p.T)
        ev, Evec = torch.linalg.eigh(h1p)
        with torch.no_grad():
            qi, ii = linear_sum_assignment(-(self.V @ Evec).abs().numpy())
            perm = ii[np.argsort(qi)]
        lm = ev[torch.tensor(perm)]
        D = self.V.T @ (lm[:, None] * self.V) - h1p
        c1 = (D * D).sum()
        lam_h1 = ev.abs().sum()
        trc = w.abs().sum(1) + wB.abs()
        Lam = lam_h1 + 0.25 * (trc ** 2).sum()
        gap = self.EUP - (self.CONST - (lam_h1 - ev.sum()) - 0.5 * (off * off).sum())
        obj = 0.5 * torch.log(gap * (2 * Lam - gap))
        # v1.1.0: h1' eigenvalues at ~0 are kinks of lambda_h1 = sum |ev| (and of E_gap); refit() holds the block of
        # h1' on their (fixed) start eigenvectors, which keeps those eigenvalues where they are to first order.
        kv = []
        if self.kink_basis is not None:
            Vs = torch.tensor(self.kink_basis)
            Mk = Vs.T @ h1p @ Vs
            iu = np.triu_indices(Mk.shape[0])
            kv = [Mk[i, j] for i, j in zip(*iu)]
        g = []
        for qq in (obj, c1, c2, gap, *kv):
            (gq,) = torch.autograd.grad(qq, yt, retain_graph=True)
            g.append(gq.numpy().copy())
        G, L, lam = float(gap.detach()), float(Lam.detach()), math.exp(float(obj.detach()))
        kappa = G / lam ** 2                       # d log(lam) / d Lambda
        tr = trc.detach().numpy()
        allow = np.concatenate([np.broadcast_to(0.5 * kappa * tr[:, None, :], (R, B, Cc)).ravel(),
                                0.5 * kappa * tr.ravel()])
        c1, c2 = float(c1.detach()), float(c2.detach())
        return dict(obj=float(obj.detach()), e1=math.sqrt(c1) * 1e3, e2=math.sqrt(c2) * 1e3, Lam=L, gap=G,
                    lam=lam, c1=c1, c2=c2, perm=perm, min_abs_ev=float(ev.detach().abs().min()), g=g,
                    kink=[float(x.detach()) for x in kv],
                    allow=allow)


def femoco_problem(npz_path, words, brot, verifier_dir) -> Problem:
    """The competition problem: official reference integrals, fixed shifts and Givens tree; u from `words`."""
    dm, ref = _official_modules(Path(verifier_dir))
    z = np.load(npz_path, allow_pickle=False)
    edges = np.asarray(z["tree_edges"], np.int64)
    w0, wB0 = np.asarray(z["W"], np.float64), np.asarray(z["H2_I"], np.float64)
    shifts = ref.load_fixed_shifts()
    N = int(ref.ORBITALS)
    S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), np.float64)
    beta1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))
    rd = ref.load_reference(ref.check_fcidump())
    words = np.asarray(words).astype(np.int64)
    u_all = dm.reconstruct_u_vectors(words / (1 << (int(brot) - 1)), edges)
    eup = float(ref.UPPER_ENERGY)

    def official(w, W):
        return dm.evaluate_dfthc_angle_words(words, w, W, S, beta1, rd, brot=int(brot), tree_edges=edges,
                                             energy_upper=eup)

    return Problem(u_all, w0, wB0, S, beta1, rd["oei"], rd["tei"], rd["electrons"], rd["ecore"], eup, official)


# ------------------------------------------------------------------------------------------- the constrained refit
# Projected L-BFGS on the constraint manifold with a Gauss-Newton retraction: the idea follows anazemcev's public
# Apache-2.0 complete-H projected L-BFGS refit; re-implemented here for three equality constraints and exact L1.


class _Budget(Exception):
    pass


def refit(P: Problem, y_start, targets, *, max_evals=1000, memory=12, release_every=20, step0=1e-5, log=None,
          kink_tol=KINK_TOL):
    """One constrained refit from y_start. targets = (e1t_mHa, e2t_mHa, gap_target_Ha). Returns dict(y (best feasible
    point by the torch model, or None), lam, stop, evals, iterations)."""
    log = log or (lambda **kw: None)
    e1t, e2t, gapt = targets
    e1_free = e1t is None  # v1.1.0: error_1 as an inequality (must stay <= E1_CAP_MHA), not a held equality
    E1T, E2T = (None if e1_free else e1t * 1e-3), e2t * 1e-3
    act = [1, 2] if e1_free else [0, 1, 2]   # which of (c1, c2, gap) are held
    count = [0]

    def evaluate(y):
        if count[0] >= max_evals:
            raise _Budget()
        count[0] += 1
        return P.evaluate(y)

    kt = []  # v1.1.0: held entries of the kink block (act indices 3, 4, ...)

    def residual(q):
        r = np.array([0.0 if e1_free else q["c1"] - E1T ** 2, q["c2"] - E2T ** 2, q["gap"] - gapt]
                     + [a - b for a, b in zip(q["kink"], kt)])
        return r[act]

    def normals(q, free):
        return np.stack([q["g"][1 + i] for i in act]) * free

    def tangent(d, Nm):
        return d - Nm.T @ np.linalg.solve(Nm @ Nm.T, Nm @ d)

    def retract(y, free, q=None, iters=8):
        for _ in range(iters):
            if q is None:
                q = evaluate(y)
            r = residual(q)
            tol = np.array([1e-11, 1e-11, 1e-8] + [1e-12] * len(kt))[act]
            if np.all(np.abs(r) < tol):
                return y, q, True
            Nm = normals(q, free)
            y = y - Nm.T @ np.linalg.solve(Nm @ Nm.T, r)
            q = None
        return y, q, False

    def feasible(q):
        if e1_free:
            return q["e1"] <= E1_CAP_MHA and q["e2"] <= e2t + FEAS_E_TOL_MHA and q["gap"] >= gapt - FEAS_GAP_TOL
        return q["e1"] <= e1t + FEAS_E_TOL_MHA and q["e2"] <= e2t + FEAS_E_TOL_MHA and q["gap"] >= gapt - FEAS_GAP_TOL

    y0 = np.asarray(y_start, np.float64)
    free = np.abs(y0) >= ZERO_HOLD
    y = np.where(free, y0, 0.0)
    best = (math.inf, None)
    stop, it, stall = "BUDGET", 0, 0
    try:
        if kink_tol:
            evs, vecs = np.linalg.eigh(P.h1p_np(y))
            small = np.flatnonzero(np.abs(evs) < kink_tol)
            if small.size:
                P.kink_basis = vecs[:, small]
                log(event="kink_hold", eigenvalues=[float(x) for x in evs[small]])
        q = evaluate(y)
        if P.kink_basis is not None:
            kt.extend(q["kink"])
            act.extend(range(3, 3 + len(kt)))
        y, q, ok = retract(y, free, q)
        log(event="retract0", ok=ok, evals=count[0])
        if not ok:
            P.kink_basis = None
            return dict(y=None, lam=math.inf, stop="RETRACTION_FAILED", evals=count[0], iterations=0)
        if feasible(q):
            best = (q["lam"], y.copy())
        ss, ys = [], []
        while True:
            it += 1
            Nm = normals(q, free)
            g = tangent(q["g"][0] * free, Nm)
            pg = float(np.linalg.norm(g))
            if stall >= 2 or pg < 1e-9 or it % release_every == 1:
                mu, kres, rZ, viol = kkt(q, free, act)
                idx = np.flatnonzero(viol > 0)
                log(event="release_check", it=it, evals=count[0], violations=int(idx.size), kkt_res=kres)
                if idx.size == 0 and it % release_every == 1 and stall < 2 and pg >= 1e-9:
                    pass
                elif idx.size == 0:
                    stop = "KKT_SATISFIED" if pg < 1e-9 else "STALLED_NO_RELEASE"
                    break
                if idx.size:
                    idx = idx[np.argsort(-viol[idx])][:2000]
                    free = free.copy()
                    free[idx] = True
                    y = y.copy()
                    y[idx] = -np.sign(rZ[idx]) * 1e-9
                    y, q, ok = retract(y, free)
                    ss, ys, stall = [], [], 0
                    if q is not None and feasible(q) and q["lam"] < best[0]:
                        best = (q["lam"], y.copy())
                    if q is None:
                        q = evaluate(y)
                    continue
            work = g.copy()
            alphas = []
            for s_, y_ in reversed(list(zip(ss, ys))):
                a = float(s_ @ work) / float(y_ @ s_)
                alphas.append(a)
                work -= a * y_
            if ss:
                work *= float(ss[-1] @ ys[-1]) / float(ys[-1] @ ys[-1])
            else:
                work *= step0 * np.linalg.norm(y) / pg
            for (s_, y_), a in zip(zip(ss, ys), reversed(alphas)):
                work += s_ * (a - float(y_ @ work) / float(y_ @ s_))
            d = tangent(-work * free, Nm)
            slope = float(g @ d)
            if slope >= 0:
                ss, ys = [], []
                d = -g * step0 * np.linalg.norm(y) / pg
                slope = float(g @ d)
            loss = q["obj"]
            step, found = 1.0, False
            for _ in range(10):
                trial = y + step * d
                cross = free & (y != 0) & (np.sign(trial) != np.sign(y))
                tfree = free & ~cross
                trial = np.where(cross, 0.0, trial)
                yt, qt, okr = retract(trial, tfree)
                if okr and qt["obj"] <= loss + 1e-4 * step * slope and (not e1_free or qt["e1"] <= E1_CAP_MHA):
                    found = True
                    break
                step *= 0.5
            if not found:
                ss, ys = [], []
                stall += 1
                log(event="backtrack_exhausted", it=it, evals=count[0], stall=stall)
                continue
            gn = tangent(qt["g"][0] * tfree, normals(qt, tfree))
            s_, y_ = (yt - y) * tfree, (gn - g) * tfree
            if float(s_ @ y_) > 1e-12 * np.linalg.norm(s_) * np.linalg.norm(y_):
                ss.append(s_)
                ys.append(y_)
                if len(ss) > memory:
                    ss.pop(0)
                    ys.pop(0)
            if cross.any():
                ss = [a * tfree for a in ss]
                ys = [a * tfree for a in ys]
            gain = loss - qt["obj"]
            stall = stall + 1 if gain < 1e-9 else 0
            y, q, free = yt, qt, tfree
            if feasible(q) and q["lam"] < best[0]:
                best = (q["lam"], y.copy())
            log(event="iter", it=it, evals=count[0], lam=q["lam"], e1=q["e1"], e2=q["e2"], gap=q["gap"],
                step=step, held=int((~free).sum()), crossed=int(cross.sum()))
    except _Budget:
        stop = "BUDGET"
    P.kink_basis = None
    log(event="refit_stop", reason=stop, evals=count[0], iterations=it, best_lambda_model=best[0])
    return dict(y=best[1], lam=best[0], stop=stop, evals=count[0], iterations=it)


def kkt(q, free, act=(0, 1, 2)):
    """Multipliers mu (grad log lam = sum mu_i grad c_i over the held constraints `act` of c = (e1^2, e2^2, E_gap))
    from the free coordinates; relative residual there; worst subgradient violation on held zeros."""
    Ns = np.stack([q["g"][1 + i] for i in act])
    Nm = Ns * free
    g0 = q["g"][0] * free
    mu, *_ = np.linalg.lstsq(Nm.T, g0, rcond=None)
    res = float(np.linalg.norm(g0 - Nm.T @ mu) / np.linalg.norm(g0))
    rZ = (q["g"][0] - Ns.T @ mu) * (~free)
    viol = (np.abs(rZ) - q["allow"]) * (~free)
    return mu, res, rZ, viol


def diagnose(P: Problem, y) -> dict:
    """Official metrics at y, KKT multipliers (at y with |y| < 1e-6 held at 0) and the first-order relative
    lambda_eff gain from spending all remaining slack: -mu_1 (e_lim^2 - e1^2) - mu_2 (e_lim^2 - e2^2)
    + mu_gap (E_gap - floor), each term counted only when the multiplier has the sign that makes slack useful.
    A multiplier with the other sign means the point is not a constrained optimum (`wrong_sign`)."""
    m = P.official(y)
    free = np.abs(y) >= ZERO_HOLD
    q = P.evaluate(np.where(free, y, 0.0))
    mu, kres, rZ, viol = kkt(q, free)
    lim = P.error_limit
    slack = (lim ** 2 - m["error_1"] ** 2, lim ** 2 - m["error_2"] ** 2, m["E_gap"] - P.gap_floor)
    gain = (max(0.0, -mu[0]) * max(0.0, slack[0]) + max(0.0, -mu[1]) * max(0.0, slack[1])
            + max(0.0, mu[2]) * max(0.0, slack[2]))
    wrong = [n for n, x, bad in (("error_1", mu[0], mu[0] > 0), ("error_2", mu[1], mu[1] > 0),
                                 ("E_gap", mu[2], mu[2] < 0)) if bad]
    return dict(official=m, passes=P.passes(m), kkt_mu=[float(x) for x in mu], kkt_rel_residual=kres,
                held_violations=int((viol > 0).sum()), first_order_gain_rel=float(gain), wrong_sign=wrong)


# ---------------------------------------------------------------------------------------------------- the gauge step

_Q4 = np.pi / 4


def _term(M):
    return 0.25 * float((np.abs(M).sum(0) ** 2).sum())


def _schedule(C):
    """Round-robin tournament: C-1 rounds of C/2 disjoint column pairs (odd C: a dummy column sits one out)."""
    Ce = C + (C % 2)
    idx = list(range(Ce))
    out = []
    for _ in range(Ce - 1):
        out.append(np.array([(idx[k], idx[Ce - 1 - k]) for k in range(Ce // 2)
                             if idx[k] < C and idx[Ce - 1 - k] < C]))
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return out


def _best_angles(a, b):
    """Exact minimiser over theta in [-pi/4, pi/4) of ||a c - b s||_1^2 + ||a s + b c||_1^2 for each row pair:
    between breakpoints (an entry crossing zero) the objective is p + q cos 2t + s sin 2t, so every interval's
    endpoints and interior stationary point are evaluated."""
    Pn, n = a.shape
    bp = (np.arctan2(a, b) + _Q4) % (2 * _Q4) - _Q4
    edges = np.concatenate([np.full((Pn, 1), -_Q4), np.sort(bp, 1), np.full((Pn, 1), _Q4)], 1)
    lo, hi = edges[:, :-1], edges[:, 1:]
    mid = 0.5 * (lo + hi)
    cm, sm = np.cos(mid)[..., None], np.sin(mid)[..., None]
    su = np.sign(a[:, None, :] * cm - b[:, None, :] * sm)
    sv = np.sign(a[:, None, :] * sm + b[:, None, :] * cm)
    al = np.einsum("pik,pk->pi", su, a)
    be = -np.einsum("pik,pk->pi", su, b)
    ga = np.einsum("pik,pk->pi", sv, a)
    de = np.einsum("pik,pk->pi", sv, b)
    q = 0.5 * (al * al + ga * ga - be * be - de * de)
    s = al * be + ga * de
    th = 0.5 * (np.arctan2(s, q) + np.pi)
    th = th - np.pi * np.round((th - mid) / np.pi)
    th = np.where((th >= lo) & (th <= hi), th, mid)
    cand = np.concatenate([edges, th], 1)
    c, s_ = np.cos(cand)[..., None], np.sin(cand)[..., None]
    U = np.abs(a[:, None, :] * c - b[:, None, :] * s_).sum(2)
    V = np.abs(a[:, None, :] * s_ + b[:, None, :] * c).sum(2)
    f = U * U + V * V
    k = np.argmin(f, 1)
    r = np.arange(Pn)
    return cand[r, k], f[r, k]


def _jacobi(M, O, max_sweeps=500, tol=1e-12):
    M = M @ O
    O = O.copy()
    sch = _schedule(M.shape[1])
    for _ in range(max_sweeps):
        f_start = _term(M)
        for pr in sch:
            i, j = pr[:, 0], pr[:, 1]
            a, b = M[:, i].T, M[:, j].T
            f0 = np.abs(a).sum(1) ** 2 + np.abs(b).sum(1) ** 2
            th, f = _best_angles(a, b)
            ok = f < f0 * (1 - 1e-15)
            if not ok.any():
                continue
            i, j, th = i[ok], j[ok], th[ok]
            c, s = np.cos(th), np.sin(th)
            for X in (M, O):
                xi, xj = X[:, i].copy(), X[:, j].copy()
                X[:, i] = xi * c - xj * s
                X[:, j] = xi * s + xj * c
        if f_start - _term(M) < tol:
            break
    U_, _, Vt = np.linalg.svd(O)
    return U_ @ Vt


def _polish(M, O, eps_list=(3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6)):
    """Smoothed-L1 L-BFGS on O = expm(A - A^T) O0 with a decreasing smoothing schedule (single torch thread)."""
    import torch
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.set_default_dtype(torch.float64)
        Mt, Ot = torch.tensor(M), torch.tensor(O)
        med = float(Mt.abs().median())
        C = M.shape[1]
        for er in eps_list:
            eps = er * med
            A = torch.zeros(C, C, requires_grad=True)
            opt = torch.optim.LBFGS([A], lr=1, max_iter=300, history_size=30, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-14, tolerance_change=1e-16)

            def cl():
                opt.zero_grad()
                X = Mt @ (torch.linalg.matrix_exp(A - A.T) @ Ot)
                loss = 0.25 * (torch.sqrt(X * X + eps * eps).sum(0) ** 2).sum()
                loss.backward()
                return loss
            opt.step(cl)
            with torch.no_grad():
                Ot = torch.linalg.matrix_exp(A - A.T) @ Ot
        U_, _, Vt = np.linalg.svd(Ot.numpy())
        return U_ @ Vt
    finally:
        torch.set_num_threads(nt)


def _gauge_task(M, O0, deterministic):
    cands = []
    if deterministic:
        Oa = _jacobi(M, O0, max_sweeps=200)
        cands.append((_term(M @ Oa), Oa))
    Ob = _jacobi(M, _polish(M, O0), max_sweeps=60)
    cands.append((_term(M @ Ob), Ob))
    return min(cands, key=lambda x: x[0])


def _expm_skew(rng, C, kappa):
    from scipy.linalg import expm
    G = rng.normal(size=(C, C)) * kappa / math.sqrt(2 * C)
    return expm(G - G.T)


def _gauge_rank(M, kicks):
    """One rank: exact Jacobi and smoothed-L1 L-BFGS + Jacobi from the identity, then each kick around the best."""
    C = M.shape[1]
    best = (_term(M), np.eye(C))
    f, O = _gauge_task(M, np.eye(C), True)
    if f < best[0]:
        best = (f, O)
    for K in kicks:
        f, O = _gauge_task(M, K @ best[1], False)
        if f < best[0]:
            best = (f, O)
    return best


def _single_thread():
    """Context limiting BLAS to one thread (threadpoolctl when installed), so the gauge step gives the same bits in
    the main process and in pool workers."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def _gauge_worker(args):
    with _single_thread():
        return _gauge_rank(*args)


def _worker_init():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"


def gauge(w, wB, *, seed=0, kappas=(0.01, 0.05), procs=1, kicks_on=True, log=None):
    """Per rank r, minimise 1/4 sum_c ||(M_r O)_{:,c}||_1^2 over orthogonal O, M_r = [w_r ; w_B,r]: exact Jacobi and
    smoothed-L1 L-BFGS + Jacobi from the identity, then one kicked start per kappa around the best. The kicks are
    drawn up front in a fixed order (rank, kappa), so the result does not depend on `procs` (ranks run in a spawn
    pool when procs > 1). Returns (w O, w_B O, O stack, per-rank Lambda_2 terms). Never worse than the identity."""
    log = log or (lambda **kw: None)
    w, wB = np.asarray(w, np.float64), np.asarray(wB, np.float64)
    R, B, C = w.shape
    Ms = [np.vstack([w[r], wB[r][None]]) for r in range(R)]
    rng = np.random.default_rng(seed)
    kicks = [[_expm_skew(rng, C, kp) for kp in kappas] for _ in range(R)]
    if not kicks_on:  # warm rounds: the start already carries the previous rounds' rotations (v1.1.0)
        kicks = [[] for _ in range(R)]
    jobs = list(zip(Ms, kicks))
    if procs > 1 and R > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(procs, R), mp_context=mp.get_context("spawn"),
                                 initializer=_worker_init) as ex:
            best = list(ex.map(_gauge_worker, jobs))
    else:
        best = [_gauge_worker(j) for j in jobs]
    start = [_term(M) for M in Ms]
    best = [b if b[0] < s else (s, np.eye(C)) for b, s in zip(best, start)]
    Os = np.stack([b[1] for b in best])
    log(event="gauge", lambda2_before=sum(start), lambda2_after=sum(b[0] for b in best))
    return (np.stack([w[r] @ Os[r] for r in range(R)]), np.stack([wB[r] @ Os[r] for r in range(R)]), Os,
            dict(before=start, after=[b[0] for b in best]))


# ---------------------------------------------------------------------------------------------------- alternation


E2_AT_LIMIT_MHA = 1e-6     # v1.5.0: an error_2 within this of the 20 mHa limit is held at its own value
E1T_DEFAULT, E2T_DEFAULT = "free", 19.999   # v1.7.0: the defaults; an explicitly set target is never auto-held


def default_targets(o0, q0, *, e1t, e2t, gap_margin, gap_floor, e1t_explicit=False, e2t_explicit=False):
    """(targets (e1, e2, gap) in the refit model's units, notes). v1.1.0: a start tighter than a target keeps its own
    value, and targets are the model's start value plus the wanted OFFICIAL change. v1.5.0: a start that is already past a target the first step could only reach by giving ground is held at
    its own value instead of stalling the retraction: error_2 within E2_AT_LIMIT_MHA of the limit (any e2t), and,
    with e1t='free', error_1 above the inequality cap E1_CAP_MHA. `o0` are official metrics (Ha), `q0` the model's
    (mHa for e1/e2). v1.7.0: the two holds apply only to a target left at its default. An explicit
    `e2t` (`e2t_explicit`) is the error_2 target even at the limit (the 1.1.0-1.4.0 rule), and an
    explicit `--e1t free` (`e1t_explicit`) is the plain inequality."""
    notes = []
    e2_start = o0["error_2"] * 1e3
    if e2_start < ERROR_LIMIT * 1e3 - E2_AT_LIMIT_MHA:
        e2o = max(float(e2t), e2_start)
    elif e2t_explicit:
        e2o = float(e2t)
        if e2_start > float(e2t):
            notes.append("e2_explicit_at_limit")
    else:
        e2o = e2_start
        if e2_start > float(e2t):
            notes.append("e2_held_at_limit")
    gmo = float(gap_margin)
    m0 = o0["E_gap"] - gap_floor
    if m0 > 0:
        gmo = min(gmo, m0)
    e2m = q0["e2"] + (e2o - e2_start)
    gapm = q0["gap"] + (gap_floor + gmo - o0["E_gap"])
    if e1t == "start":
        e1m = q0["e1"]
    elif e1t == "free":
        e1m = None
        if o0["error_1"] * 1e3 > E1_CAP_MHA and not e1t_explicit:
            e1m = q0["e1"]
            notes.append("e1_held_past_cap")
    else:
        e1m = q0["e1"] + (float(e1t) - o0["error_1"] * 1e3)
    return (e1m, e2m, gapm), notes


def alternate(P: Problem, y_base, *, arm="B", rounds=10, max_evals=1000, e1t=None, e2t=None, gap_margin=1e-5,
              seed=0, kappas=(0.01, 0.05), procs=1, min_gain=1e-7, patience=3, kick_rounds=1, log=None) -> dict:
    """Arm A: one constrained refit. Arm B: refit, then rounds of (gauge step, refit from the gauged point), keeping
    the lowest official lambda_eff among points that pass the official accuracy gates. Stops when a round's realised
    relative gain and the first-order gain predicted at its best candidate are both below REL_TIE, or after `rounds`
    rounds. The input point is the initial incumbent when it passes, so the result is never worse than the input.
    e1t='start' holds error_1 at its starting value (capped at e2t's default 19.999 mHa). e1t / e2t None means the
    default ('free' / 19.999 mHa) with the v1.5.0 holds; a value given is explicit and never held (v1.7.0)."""
    log = log or (lambda **kw: None)
    y_base = np.asarray(y_base, np.float64)
    d0 = diagnose(P, y_base)
    o0 = d0["official"]
    # v1.1.0 targets. (1) Never force the first step to give ground: a start tighter than a default target keeps its
    # own value. (2) Targets are set in the refit model's units, as the model's start value plus the wanted OFFICIAL
    # change: near a cusp the model and the official evaluation can differ (seen: E_gap by 7e-5 Ha), and official
    # targets would then ask the retraction to move. The official check on every candidate still decides.
    z0 = np.where(np.abs(y_base) >= ZERO_HOLD, y_base, 0.0)
    q0 = P.evaluate(z0)
    e1x, e2x = e1t is not None, e2t is not None
    targets, held = default_targets(o0, q0, e1t=E1T_DEFAULT if e1t is None else e1t,
                                    e2t=E2T_DEFAULT if e2t is None else e2t, gap_margin=gap_margin,
                                    gap_floor=P.gap_floor, e1t_explicit=e1x, e2t_explicit=e2x)
    log(event="base", diag=d0, targets=targets, held_at_start=held)
    inc = dict(y=y_base, lam=d0["official"]["lambda_eff"], round=0, diag=d0) if d0["passes"] else None
    hist = []
    stop = None
    low = 0  # consecutive rounds whose realised gain is below min_gain (v1.1.0 stop rule)
    for k in range(1, (1 if arm == "A" else rounds) + 1):
        cands = []
        rec = dict(round=k)
        start = inc["y"] if inc is not None else y_base
        if k > 1:
            w, wB = P.split(start)
            gw, gwB, _, gi = gauge(w, wB, seed=seed + k, kappas=kappas, procs=procs,
                                   kicks_on=(k - 1) <= kick_rounds, log=log)
            yg = P.from_weights(gw, gwB)
            dg = diagnose(P, yg)
            rec["gauge"] = dict(lambda2_before=sum(gi["before"]), lambda2_after=sum(gi["after"]),
                                lambda_eff=dg["official"]["lambda_eff"], passes=dg["passes"])
            cands.append((yg, dg, "gauge"))
            start = yg
        rf = refit(P, start, targets, max_evals=max_evals, log=log)
        rec["refit"] = dict(stop=rf["stop"], evals=rf["evals"], iterations=rf["iterations"])
        if rf["y"] is not None:
            dr = diagnose(P, rf["y"])
            rec["refit"].update(lambda_eff=dr["official"]["lambda_eff"], passes=dr["passes"])
            cands.append((rf["y"], dr, "refit"))
        feas = [c for c in cands if c[1]["passes"]]
        prev = inc["lam"] if inc is not None else math.inf
        if feas:
            yb, db, how = min(feas, key=lambda c: c[1]["official"]["lambda_eff"])
            lam_new = db["official"]["lambda_eff"]
            fo = db["first_order_gain_rel"]
        else:
            yb, db, how, lam_new, fo = None, None, None, prev, 0.0
        realised = (prev - lam_new) / prev if math.isfinite(prev) else (math.inf if feas else 0.0)
        if yb is not None and lam_new < prev:
            inc = dict(y=np.array(yb), lam=lam_new, round=k, diag=db)
        rec.update(best_candidate=how, lambda_eff=lam_new, realised_gain_rel=realised, first_order_gain_rel=fo,
                   incumbent_lambda_eff=inc["lam"] if inc is not None else None)
        hist.append(rec)
        log(event="round", **rec)
        if arm == "A":
            stop = "ARM_A"
            break
        if k > 1 and realised < REL_TIE and fo < REL_TIE:
            stop = "GAIN_BELOW_REL_TIE"
            break
        # The first-order gain is only collectable by the refit step; when the refit cannot move (e.g. a stalled
        # retraction) it stays large while rounds gain ~1e-7. Stop on realised gains instead (v1.1.0).
        low = low + 1 if (k > 1 and realised < min_gain) else 0
        if low >= patience:
            stop = "REALISED_GAIN_BELOW_MIN"
            break
    if stop is None:
        stop = "ROUNDS"
    return dict(stop=stop, targets=targets, held=held, base=d0, rounds=hist,
                result=None if inc is None else dict(y=inc["y"], lambda_eff=inc["lam"], from_round=inc["round"],
                                                     diag=inc["diag"]))


# ----------------------------------------------------------------------------------------- words and verification


def resolve_brot(z, brot=None, circuit_brot=None) -> tuple:
    """(brot, source): bits per angle word (v1.7.0). The official verifier reads brot from the
    circuits' word registers (register_layout), and some public npz files have no 'brot' field (Syed Asad Ali's
    eff6a4863f08). Order: the npz field; then an explicit `brot` (--brot); then the circuits' word-register width
    (`circuit_brot`, known only with a kickmix build); else a clear error. Any two sources that are given must agree."""
    npz_b = int(z["brot"]) if "brot" in z.files else None
    given = [(v, src) for v, src in ((npz_b, "npz field 'brot'"), (None if brot is None else int(brot), "--brot"),
                                     (None if circuit_brot is None else int(circuit_brot),
                                      "the circuits' word-register width")) if v is not None]
    if not given:
        raise RuntimeError("submission.npz has no 'brot' field and no kickmix build is available to read the "
                           "circuits' word-register width: pass --brot (bits per angle word, e.g. 8)")
    for v, src in given[1:]:
        if v != given[0][0]:
            raise RuntimeError(f"brot {v} from {src} contradicts brot {given[0][0]} from {given[0][1]}")
    return given[0]


def recover_words(sub_dir: Path, verifier_dir: Path, kickmix_dir=None, brot=None) -> dict:
    """The words the loaders load, via the official dfthcsos.verify.evaluate_circuits (unmodified; default seed).
    Returns dict(words, brot, brot_source, source, before=official evaluation summary) or, without kickmix, the npz
    words with source='npz' and a warning. `brot` (--brot) is used when the npz has no 'brot' field and there is no
    kickmix build; see resolve_brot."""
    km = find_kickmix(kickmix_dir, verifier_dir)
    z = np.load(Path(sub_dir) / "submission.npz", allow_pickle=False)
    if km is not None:
        _official_modules(verifier_dir)
        V = importlib.import_module("dfthcsos.verify")
        inputs = V.load_inputs(Path(sub_dir) / "submission.npz")
        edges = V.load_tree_edges(None, sub_dir)
        r = V.evaluate_circuits(inputs, sub_dir, edges)
        words = np.asarray(r["angle_words"]).astype(np.int64)
        npz_match = bool("angle_words" in z.files and np.array_equal(np.asarray(z["angle_words"]), words))
        if "angle_words" in z.files and not npz_match:
            warnings.warn("submission.npz angle_words differ from the words the loaders load; using the loaders' "
                          "words (the npz copy is stale)", stacklevel=2)
        rb, rb_src = resolve_brot(z, brot, int(r["brot"]))
        return dict(words=words, brot=rb, brot_source=rb_src, source="loaders (official evaluate_circuits)",
                    kickmix=str(getattr(km, "__file__", "?")), npz_angle_words_match=npz_match,
                    before=dict(metrics=_jsonable(r["metrics"]), resources=_jsonable(r["resources"]),
                                score=float(r["score"]), submission_sha256=r["submission_sha256"]))
    msg = ("WARNING: no kickmix build found, so the words were NOT recovered from the loaders. Using the "
           "submission.npz angle_words instead: these can be stale (differ from what the circuits load), and the "
           "refit is then only as good as that copy. Attach the competition data or pass --kickmix-dir.")
    print("\n" + "!" * 100 + "\n" + msg + "\n" + "!" * 100 + "\n", file=sys.stderr, flush=True)
    warnings.warn(msg, stacklevel=2)
    if "angle_words" not in z.files:
        raise RuntimeError("no kickmix build and no angle_words in submission.npz: cannot determine the words")
    words = np.asarray(z["angle_words"])
    brot, brot_src = resolve_brot(z, brot)
    if not np.issubdtype(words.dtype, np.integer) or words.min() < 0 or words.max() >= (1 << brot):
        raise RuntimeError(f"npz angle_words are not integers in [0, 2**{brot}) (brot from {brot_src})")
    return dict(words=words.astype(np.int64), brot=brot, brot_source=brot_src,
                source="npz angle_words (NOT recovered from the loaders)",
                kickmix=None, npz_angle_words_match=None, before=None)


def verify_dir(sub_dir: Path, verifier_dir: Path, kickmix_dir=None) -> dict | None:
    """The unmodified official dfthcsos.verify.verify on a submission directory (default salt), in this process.
    None without kickmix; a FAIL is returned as {"status": "FAIL", "error": ...}."""
    if find_kickmix(kickmix_dir, verifier_dir) is None:
        return None
    _official_modules(verifier_dir)
    V = importlib.import_module("dfthcsos.verify")
    try:
        return _jsonable(V.verify(str(sub_dir)))
    except (AssertionError, ValueError, RuntimeError) as ex:
        return {"status": "FAIL", "error": str(ex)}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _versions(verifier_dir, kickmix_file):
    import scipy
    v = dict(dfthc_refit=__version__, python=sys.version.split()[0], platform=platform.platform(),
             numpy=np.__version__, scipy=scipy.__version__, kickmix=kickmix_file)
    try:
        import torch
        v["torch"] = torch.__version__
        v["torch_threads"] = torch.get_num_threads()
    except ImportError:
        v["torch"] = None
    pkg = Path(verifier_dir) / "dfthcsos"
    v["verifier_sha256"] = {p: sha256_file(pkg / p) for p in ("verify.py", "scoreinfra/dfthc_math.py",
                                                               "scoreinfra/reference.py", "scoreinfra/circuits.py")
                            if (pkg / p).is_file()}
    return v


def _metrics_row(m):
    return dict(error_1_mHa=m["error_1"] * 1e3, error_2_mHa=m["error_2"] * 1e3, E_gap=m["E_gap"],
                E_gap_margin_Ha=m["E_gap"] - E_GAP_FLOOR, Lambda=m["Lambda"], lambda_eff=m["lambda_eff"])


# ------------------------------------------------------------------------------- loader re-planning (replan, v1.2)
#
# `dfthc_refit.py replan FAMILY_DIR OUT_DIR --planner QROM_PLAN --lera-src DIR` re-plans every ANGLE_k loader of a
# family on its own words with the qrom-plan relation planner (a separate Rust binary), emits each plan with Lera's
# unmodified emitter (tensorqrom.build_circuit, 4 low / 7 high address bits, grouped low cache, borrowed top bits),
# checks it natively, and per slot keeps the cheaper of shipped and re-planned. Words and weights never change:
# submission.npz is copied byte for byte. A slot changes only if its re-planned loader has strictly fewer Toffolis,
# passes the native check, uses <= 128 qubits and has the shipped register layout; the assembled family must then
# load exactly the input words with unchanged accuracy metrics (no chain break) and one register layout across all
# slots (no layout mismatch), else the shipped family is written unchanged.

QUBIT_CAP = 128                      # per ANGLE_k circuit (dfthcsos.verify.QUBIT_CAP)
PLANNER_CACHE_TOFFOLIS = 89          # qrom-plan's hard-coded cache count (1,363-row tables); corrected per table
LERA_BROT, LERA_LOWBITS, LERA_J = 8, 4, 5
GROUPS_8_3 = [[3, 5, 6, 7, 11, 13, 14, 15], [9, 10, 12]]
GROUPS_ONE = [[3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]]
DESIGN_D0 = [[3, 5, 6], [0, 1, 2], [4, 10, 12], [7, 13, 14], [8, 9, 11]]
# Default recipes: block designs for generic loaders
# (no coded words), structured draws for coded loaders. `values` are block `per` or structured `count`; `seeds` is an
# inclusive range. Jobs run in this order; a later plan replaces the incumbent only with strictly fewer products.
DEFAULT_RECIPES = {
    "generic": [dict(name="block_8+3_d0", mode="block", groups=GROUPS_8_3, design=DESIGN_D0,
                     values=[13, 14, 15, 16, 17], seeds=[0, 63])],
    "coded": [dict(name="structured_8+3", mode="structured", groups=GROUPS_8_3, values=list(range(109, 122)),
                   seeds=[0, 95]),
              dict(name="structured_one_group", mode="structured", groups=GROUPS_ONE,
                   values=[111, 113, 115, 117, 119], seeds=[0, 63])],
}


def _coded_columns(words):
    """Word columns whose bit 0 is affine on every full 16-address block (the partial last block is always fine)."""
    A, n = words.shape
    affine = set()
    for c in range(2):
        for mask in range(16):
            affine.add(tuple(c ^ (bin(mask & a).count("1") & 1) for a in range(16)))
    out = []
    for j in range(n):
        bits = words[:, j] & 1
        if all(tuple(int(x) for x in bits[b0:b0 + 16]) in affine for b0 in range(0, A - A % 16, 16)):
            out.append(j)
    return out


def _pad_affine(delta, coded_words):
    """As Lera's pad_affine: extend bit 0 of the coded words over the partial last block by its affine pattern."""
    A = len(delta)
    out = np.concatenate([delta, np.zeros((-A % 16, delta.shape[1]), delta.dtype)])
    b0 = A - A % 16
    n = A - b0
    if 0 < n < 16:
        affs = [[c ^ (bin(m & a).count("1") & 1) for a in range(16)] for c in (0, 1) for m in range(16)]
        for j in coded_words:
            bits = out[b0:b0 + n, j] & 1
            fit = [f for f in affs if all(f[i] == bits[i] for i in range(n))]
            assert fit, f"bit 0 of word {j} is not affine on the last block"
            out[b0:b0 + 16, j] = (out[b0:b0 + 16, j] & ~1) | np.array(fit[0])
    return out


def loader_delta(words, k, j=LERA_J):
    """delta_k = B_k XOR B_(k-1) (slot k turns block k-1 into block k), with the coded bit-0 affine padding.
    Returns (delta, coded word indices)."""
    words = np.asarray(words, dtype=np.int64)
    A, n = words.shape
    padded = np.zeros((A, -(-n // j) * j), dtype=np.int64)
    padded[:, :n] = words
    bk = padded[:, j * k:j * (k + 1)]
    delta = bk.copy() if k == 0 else bk ^ padded[:, j * (k - 1):j * k]
    coded = set(_coded_columns(words))
    cw = [i for i in range(j) if (j * k + i) in coded and (k == 0 or (j * (k - 1) + i) in coded)]
    if cw:
        delta = _pad_affine(delta, cw)
    return delta, cw


def expected_cache_toffolis(rows: int, lowbits: int = LERA_LOWBITS) -> int:
    """Lera's cache Toffolis for a table of `rows` addresses: (L - 1 - lowbits) low ANDs plus (H - 1 - bits(H - 1))
    high ANDs, L = 2**lowbits, H = ceil(rows / L) high chunks. 1,220 rows: 11 + 69 = 80; 1,363 rows: 11 + 78 = 89."""
    L = 1 << lowbits
    H = -(-int(rows) // L)
    return (L - 1 - (L - 1).bit_length()) + (H - 1 - (H - 1).bit_length())


def cache_toffolis(tq, delta) -> int:
    """Toffolis of Lera's two monomial caches for this table (low: 2**4 - 1 - 4 = 11; high: H - 1 - bits(H - 1) with
    H = ceil(rows / 16) high chunks: 69 at 1,220 rows, 78 at 1,363). qrom-plan always adds 89."""
    C = tq.anf_tensor(np.asarray(delta, dtype=np.int64), LERA_BROT, LERA_LOWBITS)
    return int(tq.cost([], C.shape[0], C.shape[1]))


def free_low_sources(words, k, gauged) -> int:
    """v1.8.0-dev: low-cache ANDs the free-source gauge emitter saves on slot k (the incoming gauge
    monomials of words whose gauge does not change; `gauge.free_monomials`). 0 for a signed-gauge family."""
    if not gauged:
        return 0
    gz = _local("gauge")
    pin, pout = gz.slot_gauge(words, k, LERA_J)[:2]
    return len(gz.free_monomials(pin, pout))


def native_check(km, circuit, delta, brot=LERA_BROT, seed=4242) -> dict:
    """Run the loader twice on every table address with zero, table and random signed-gauge words: the first pass
    must XOR delta in, the second restore the input; no phase, selection unchanged, scratch clean. Counts Toffolis
    per application."""
    count, J = delta.shape
    n_sel = (count - 1).bit_length()
    rng = np.random.default_rng(seed)
    rand = rng.integers(0, 1 << (brot - 1), size=(count, J))
    rand = rand | ((rand >> (brot - 2)) & 1) << (brot - 1)
    initial = np.concatenate([np.zeros((count, J), np.int64), delta, rand]).astype(np.int64)
    addresses = np.tile(np.arange(count), 3)
    sim = km.Simulator(batch_size=len(addresses), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots("selection", ((addresses[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for j in range(J):
        sim.write_across_shots(f"word_{j}", ((initial[:, j][None, :] >> np.arange(brot)[:, None]) & 1).astype(bool))
    powers = np.int64(1) << np.arange(brot)
    toffolis = []
    try:
        for application in range(2):
            before = sim.operation_statistics().num_toffoli_like
            sim.do(circuit)
            toffolis.append((sim.operation_statistics().num_toffoli_like - before) / len(addresses))
            got = np.stack([powers @ sim.read_across_shots(f"word_{j}").astype(np.int64) for j in range(J)], axis=1)
            expected = initial ^ delta[addresses] if application == 0 else initial
            assert np.array_equal(got, expected), "incorrect XOR"
            assert not np.any(sim.read_phase_flipped_across_shots()), "phase"
            decoded = (np.int64(1) << np.arange(n_sel)) @ sim.read_across_shots("selection").astype(np.int64)
            assert np.array_equal(decoded, addresses), "selection changed"
            total = sim.read_across_shots(km.array([km.q(i) for i in range(circuit.num_qubits)]))
            assert not np.any(total[n_sel + brot * J:]), "scratch not clean"
    except AssertionError as ex:
        return {"status": "FAIL", "error": str(ex), "shots": int(len(addresses)), "toffolis_per_application": toffolis}
    return {"status": "PASS", "shots": int(len(addresses)), "toffolis_per_application": toffolis}


def _load_tensorqrom(lera_src):
    """Lera's emitter module (tensorqrom.py, Apache-2.0, unmodified) from a directory, loaded by path."""
    import importlib.util
    p = Path(lera_src) / "tensorqrom.py"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found: pass --lera-src (the directory holding Lera's tensorqrom.py)")
    spec = importlib.util.spec_from_file_location("dfthc_refit_tensorqrom", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha256_file(p)


def _circuit_layout(C, circuit):
    """(selection qubit ids, [word_j qubit ids]) as the official circuits module names the registers."""
    regs = circuit.register_data
    if C.SELECTION_REGISTER not in regs:
        return None
    sel = [q.id for q in regs[C.SELECTION_REGISTER]]
    idx = sorted(int(m.group(1)) for n in regs if (m := C._WORD_REGISTER_PATTERN.fullmatch(n)))
    return [sel, [[q.id for q in regs[C.word_register(j)]] for j in idx]]


def _read_plan_txt(path) -> dict:
    """A qrom-plan text plan: k, groups, toffolis, rel 0x.., term a1 b1 a2 b2 s."""
    plan = {"relations": 0, "terms": []}
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "k":
            plan["k"] = int(t[1])
        elif t[0] == "groups":
            plan["groups"] = [[int(x) for x in g.split(",")] for g in t[1].split(";")]
        elif t[0] == "toffolis":
            plan["planner_toffolis"] = int(t[1])
        elif t[0] == "rel":
            plan["relations"] += 1
        elif t[0] == "term":
            plan["terms"].append(tuple(int(x) for x in t[1:6]))
    return plan


def _seed_range(s):
    return (int(s[0]), int(s[1]) + 1)


def plan_jobs(work: Path, k: int, recipes: list, chunk: int = 32) -> list:
    """Template files and planner jobs for slot k, in selection order."""
    kd = work / "plans" / f"k{k}"
    kd.mkdir(parents=True, exist_ok=False)
    jobs = []
    for r in recipes:
        key = "per" if r["mode"] == "block" else "count"
        s0, s1 = _seed_range(r["seeds"])
        for val in r["values"]:
            lines = [f"k {k}", "groups " + ";".join(",".join(str(x) for x in g) for g in r["groups"])]
            if r["mode"] == "block":
                lines.append("design " + ";".join(",".join(str(x) for x in g) for g in r["design"]))
            lines.append(f"{key} {val}")
            tpl = kd / f"{r['name']}_{key}{val}.txt"
            tpl.write_text("\n".join(lines) + "\n")
            for c0 in range(s0, s1, chunk):
                c1 = min(c0 + chunk, s1)
                jobs.append(dict(k=k, recipe=r["name"], mode=r["mode"], key=key, value=val, seeds=[c0, c1 - 1],
                                 template=tpl, prefix=kd / f"{r['name']}_{key}{val}_s{c0}"))
    return jobs


def _run_planner(planner, words_npy, job, trials, samples, repair):
    import subprocess
    cmd = [str(planner), "replan", str(words_npy), str(job["template"]), job["mode"], str(job["seeds"][0]),
           str(job["seeds"][1] + 1), str(trials), str(samples), str(repair), str(job["prefix"])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[-2000:]}
    Path(str(job["prefix"]) + ".json").write_text(r.stdout)
    d = json.loads(r.stdout)
    return {"best_products": d.get("best_products"), "best_toffolis": d.get("best_toffolis"),
            "best_relations": d.get("best_relations"), "best_seed": d.get("best_seed"),
            "attempts": len(d.get("attempts", [])), "plan": str(job["prefix"]) + "_best.txt"}


def family_sha256(entries: dict) -> str:
    """Family hash as the VQCC public-circuits dataset defines it: SHA-256 of sorted '<file>\\t<sha256>\\n' lines."""
    return hashlib.sha256("".join(f"{n}\t{entries[n]}\n" for n in sorted(entries)).encode("utf-8")).hexdigest()


def _parse_slots(text, n):
    if text is None:
        return list(range(n))
    out = []
    for part in str(text).split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return sorted(set(out))


# ------------------------------------------------------------- kernel-relation planner backend (qwertykeypadapple)
#
# `--backend kernel` replaces the qrom-plan binary with qwertykeypadapple's kernel-relation search
# (`kernel_search.shared_kernel`, dataset qwertykeypadapple/qcc-kernel-search-portfolio v1, src/kernel_search.py,
# Apache-2.0, Copyright qwertykeypadapple; loaded unmodified from --kernel-src, never copied into this tool). It admits
# annihilator relations one at a time, each scored by the greedy pairing it leaves, over the full low cache (all 11
# nonlinear low monomials in one phase, borrowed top bits), in output groups of six (coded bit-0 outputs last).
# Each plan is emitted with Lera's emitter with groups=None (full cache) and scratch=True, as his
# marculera_plan_build.py does, then goes through the same native check and never-worse rule as qrom-plan plans.


def kernel_layout(n_out, coded_words, brot=LERA_BROT, size=6):
    """His `sixes` layout: free outputs in groups of `size` (in index order), then the coded bit-0 outputs."""
    coded = [(brot - 1) * j for j in coded_words]
    free = [o for o in range(n_out) if o not in coded]
    sizes = [size] * (len(free) // size) + ([len(free) % size] if len(free) % size else [])
    return free + coded, sizes + ([len(coded)] if coded else [])


def _load_kernel_modules(lera_src, kernel_src):
    """(tensorqrom, kernel_search): his module imports `tensorqrom` by name, so Lera's file is registered under that
    name first. Both are loaded by path, unmodified."""
    import importlib.util
    mods = {}
    for name, d in (("tensorqrom", lera_src), ("kernel_search", kernel_src)):
        if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(Path(d) / f"{name}.py"):
            mods[name] = sys.modules[name]
            continue
        spec = importlib.util.spec_from_file_location(name, Path(d) / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["tensorqrom"], mods["kernel_search"]


def kernel_plan(lera_src, kernel_src, delta, coded_words, seed, out_json=None, size=6, perm=None, pad=None):
    """One run of his search on a loader's padded delta: {products, terms, perm, sizes, seed, exact, pad_rows}.
    `perm`: output order (default kernel_layout); `pad`: padded-row options (see padded-row section below), or None
    for his search unchanged."""
    tq, ks = _load_kernel_modules(lera_src, kernel_src)
    delta = np.asarray(delta, dtype=np.int64)
    C = tq.anf_tensor(delta, LERA_BROT, LERA_LOWBITS)
    perm0, sizes = kernel_layout(C.shape[2], coded_words, size=size)
    perm = list(perm0 if perm is None else perm)
    assert sorted(perm) == sorted(perm0), "perm is not a permutation of the outputs"
    back = {i: o for i, o in enumerate(perm)}
    pad_info, changes = None, {}
    if pad is None:
        terms = ks.shared_kernel(C[:, :, perm], list(sizes), trials=1, seed=seed, margin=0, groups=None)
    else:
        coded = {(LERA_BROT - 1) * j for j in coded_words}
        free = {i for i, o in enumerate(perm) if o not in coded}
        state = dict(calls=0, g0=0, how=None)
        orig = ks.kernel_features
        ks.kernel_features = _pad_kernel_features(ks, tq, orig, C.shape[0] - 1, free, int(pad["n0"]), pad,
                                                  changes, state)
        try:
            terms = ks.shared_kernel(C[:, :, perm], list(sizes), trials=1, seed=seed, margin=0, groups=None)
        finally:
            ks.kernel_features = orig
        pad_info = dict(first_group=state["how"])
        if changes:
            delta = apply_pad_changes(delta, {back[o]: v for o, v in changes.items()}, int(pad["real_rows"]))
            C = tq.anf_tensor(delta, LERA_BROT, LERA_LOWBITS)
    terms = [(a1, b1, a2, b2, sum(1 << back[i] for i in tq._bits(s))) for a1, b1, a2, b2, s in terms]
    exact = bool(np.array_equal(tq.mixed_of(terms, C.shape[0] - 1, C.shape[1] - 1, C.shape[2]), C[1:, 1:, :]))
    res = dict(products=len(terms), terms=[list(t) for t in terms], perm=perm, sizes=sizes, seed=seed, exact=exact)
    if pad is not None:
        A = int(pad["real_rows"])
        rows = {int(r): [int(x) for x in delta[r]] for r in range(A, len(delta))}
        res.update(pad=dict(pad_info, real_rows=A, addressed_rows=len(delta), n0=int(pad["n0"]),
                            changed=bool(changes)), pad_rows=rows)
    if out_json is not None:
        Path(out_json).write_text(json.dumps(res))
    return res


def _kernel_job(args):
    lera_src, kernel_src, delta, coded_words, seed, out_json = args[:6]
    perm, pad = (args[6], args[7]) if len(args) > 6 else (None, None)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    r = kernel_plan(lera_src, kernel_src, delta, coded_words, seed, out_json, perm=perm, pad=pad)
    return {"best_products": r["products"] if r["exact"] else None, "best_seed": seed, "plan": str(out_json),
            "exact": r["exact"]}


# ------------------------------------------------------------------------- padded rows (never-read table rows)
#
# `--pad-rows` (kernel backend). The emitter addresses whole 16-address blocks, so a table of A rows is
# loaded as ceil(A / 16) * 16 rows; rows A.. of the last block are never read by the verifier (selection < A) and
# their values are free. They reach only the last high-monomial row of the ANF tensor, and there only the low
# monomials >= A % 16. For the first output group: his search runs with that row left out, then an exhaustive
# one-step extension of the row span (`extend_features`) is tried, then the row is set to a vector of the found
# feature span that agrees with the original on the low monomials the padded addresses cannot change; the padded
# table entries are rewritten to match (`apply_pad_changes`). A per-loader screen first ranks each word's bits 1..6
# as the first group (`pad_screen`). His `shared_kernel` is called unmodified; only its `kernel_features` lookup is
# wrapped for the duration of the call (the file on disk is not changed; its SHA-256 is recorded). The functions here
# are this tool's own code; they call his `_Group`, `rank_packed`,
# `_packed` and Lera's `nullspace`, `_products`.

PAD_SCREEN_TOP, PAD_EXTEND_TOP, PAD_RESTARTS = 64, 128, 200


def padded_row_range(real_rows: int, block: int = 1 << LERA_LOWBITS) -> tuple[int, int]:
    """(first, end) of the never-read padded rows of a table with `real_rows` rows: the emitter's addressed size is
    whole `block`-address blocks, so rows real_rows .. ceil(real_rows / block) * block - 1 are never read."""
    A = int(real_rows)
    return A, -(-A // block) * block


def full_block_delta(delta, block: int = 1 << LERA_LOWBITS):
    """delta zero-padded to whole blocks (coded deltas are already affine-padded by loader_delta)."""
    delta = np.asarray(delta, dtype=np.int64)
    return np.concatenate([delta, np.zeros((-len(delta) % block, delta.shape[1]), np.int64)])


def apply_pad_changes(d, changes, A, brot=LERA_BROT):
    """Rewrite never-read padded entries (rows A.. of the last block) so that the mixed ANF row of the last high chunk
    changes by `changes` {output: low-monomial vector, bit m-1 = monomial m}. A flip at low address l >= A % 16
    changes the coefficients of every low monomial m containing l, so the flips are the Moebius transform of the
    change. Bit 6 flips carry bit 7 (the signed-gauge copy, not an ANF output)."""
    d = np.array(d, dtype=np.int64, copy=True)
    L = 1 << LERA_LOWBITS
    h, n0 = A // L, A % L
    for o, v in changes.items():
        w, b = o // (brot - 1), o % (brot - 1)
        for m in range(1, n0):
            assert not (v >> (m - 1)) & 1, "change on a low monomial that the padded addresses cannot reach"
        for l in range(n0, L):
            f = 0
            for m in range(1, L):
                if (m & l) == m and (v >> (m - 1)) & 1:
                    f ^= 1
            if f:
                d[L * h + l, w] ^= (1 << b) | ((1 << (brot - 1)) if b == brot - 2 else 0)
    assert len(d) == -(-A // L) * L
    return d


def _solve_fixed(feats, target, mask):
    """A combination w of the feature vectors with w & mask == target & mask, or None."""
    piv = {}
    for f in feats:
        v, c = f & mask, f
        while v:
            p = v.bit_length() - 1
            if p not in piv:
                piv[p] = (v, c)
                break
            v ^= piv[p][0]
            c ^= piv[p][1]
    t, w = target & mask, 0
    while t:
        p = t.bit_length() - 1
        if p not in piv:
            return None
        t ^= piv[p][0]
        w ^= piv[p][1]
    return w


def extend_features(ks, tq, rows, Y, m, rng, top=PAD_EXTEND_TOP, restarts=PAD_RESTARTS):
    """Exhaustive one-step extension: every W = R + <v> (v over the quotient by the row span R, one class at a time)
    and W = R itself, ranked by the pairing bound, the best `top` evaluated with his pairing (`restarts` restarts).
    Returns (products, feats) or None."""
    n = Y * m
    N = tq.nullspace(rows, n)
    G = ks._Group(Y, m, [(1 << Y) - 1])
    V = G.V
    for r in N:
        V, _ = G.apply(r, V)
    nq = len(N)
    k_s = {s: len(V[(s, 0)]) for s in range(1, 1 << m)}
    cons = [G.cons(r) for r in N]
    member = np.zeros((1 << m, 1 << nq), bool)
    for s in range(1, 1 << m):
        piv, basis = {}, []
        for j in range(Y):
            v = sum(((cons[i][s] >> j) & 1) << i for i in range(nq))
            while v:
                p = v.bit_length() - 1
                if p not in piv:
                    piv[p] = v
                    basis.append(v)
                    break
                v ^= piv[p]
        span = np.zeros(1, np.int64)
        for b in basis:
            span = np.concatenate([span, span ^ b])
        member[s, span] = True
    ks_arr = np.array([0] + [k_s[s] for s in range(1, 1 << m)])
    dims = ks_arr[:, None] + member.astype(np.int64)
    dims[0] = 0
    dims[:, 0] = ks_arr
    pairs = (2 * (dims // 2)).sum(axis=0)
    total = dims.sum(axis=0)
    wdim = np.full(1 << nq, n - nq + 1)
    wdim[0] = n - nq
    bound = np.maximum(-(-wdim // 2), wdim - pairs // 2)
    bound = np.where(total >= wdim, bound, 10 ** 6)
    order = np.lexsort((-pairs, bound))[:top]
    best = None
    for q in order:
        if bound[q] >= 10 ** 6:
            break
        q = int(q)
        if q == 0:
            rel = list(N)
        else:
            i0 = (q & -q).bit_length() - 1
            rel = [N[i] ^ (N[i0] if (q >> i) & 1 else 0) for i in range(nq) if i != i0]
        Vw = G.V
        for r in rel:
            Vw, _ = G.apply(r, Vw)
        dim = n - len(rel)
        if ks.rank_packed(ks._packed(G.vectors(Vw), n), n) < dim:
            continue
        pr = G.pairing(Vw, dim, rng, restarts)
        if pr is not None and (best is None or pr[0] < best[0]):
            best = pr
            if pr[0] <= -(-(n - nq) // 2):
                break
    return best


def _pad_kernel_features(ks, tq, orig, Y, free, n0, pad, changes, state):
    """A stand-in for his kernel_features during one shared_kernel call: groups after the first, or a first group
    that does not use all Y low monomials or holds a coded output, go to his function unchanged. For the first
    group: his search with the last high row left out, the extension, then the padded-row fit (rows[-1] is updated
    in place, which shared_kernel then expresses); with no fit the group is searched as usual."""
    def kf(rows, Ys, mm, rng, **kw):
        gi, g0 = state["calls"], state["g0"]
        state["calls"] += 1
        state["g0"] += mm
        if gi != 0 or n0 == 0 or Ys != Y or not all(o in free for o in range(g0, g0 + mm)):
            return orig(rows, Ys, mm, rng, **kw)
        nr = len(rows)
        search_rows = list(rows)
        search_rows[nr - 1] = 0
        feats = orig(search_rows, Ys, mm, rng, **kw)
        how = "kernel"
        if pad.get("extend", True):
            ex = extend_features(ks, tq, search_rows, Ys, mm, rng, top=pad.get("top", PAD_EXTEND_TOP),
                                 restarts=pad.get("restarts", PAD_RESTARTS))
            if ex is not None and ex[0] < tq._products(feats):
                feats, how = ex[1], "kernel+extend"
        fixed = sum(1 << (y * mm + o) for y in range(n0 - 1) for o in range(mm))
        w = _solve_fixed([f[2] for f in feats], rows[nr - 1], fixed)
        if w is None:
            state["how"] = "no padded-row fit: searched as usual"
            return orig(rows, Ys, mm, rng, **kw)
        old_row = rows[nr - 1]
        for i in range(mm):
            new = sum(((w >> (y * mm + i)) & 1) << y for y in range(Y))
            old = sum(((old_row >> (y * mm + i)) & 1) << y for y in range(Y))
            if new != old:
                changes[g0 + i] = new ^ old
        rows[nr - 1] = w
        state["how"] = how + " +pad"
        return feats
    return kf


def _group_rows(C, outs):
    """Mixed ANF rows of output group `outs` as his shared_kernel builds them for a first group (no fan-in)."""
    X = C[1:, 1:].transpose(1, 0, 2)
    nr, Y, _ = X.shape
    m = len(outs)
    supp = [y for y in range(Y) if X[:, y, outs].any()]
    sub = {y: i for i, y in enumerate(supp)}
    rows = [sum(1 << (sub[y] * m + i) for y in supp for i, o in enumerate(outs) if X[r, y, o]) for r in range(nr)]
    return rows, len(supp), m


def pad_screen(lera_src, kernel_src, delta, coded_words, seed=0, top=PAD_SCREEN_TOP, restarts=PAD_RESTARTS):
    """Screen each word's bits 1..6 (when all free) as the first group with the last high row left free: products
    of the best one-step extension. Returns {groups: [...], pick: outs or None, perm}. The pick (fewest products,
    ties to the lower word) goes first; the other free outputs follow in index order, then the coded outputs.
    (`chem_l27a_firstgroup.py --pad-extend --words --samples 0`: one RNG over the words in order.)"""
    import random
    tq, ks = _load_kernel_modules(lera_src, kernel_src)
    C = tq.anf_tensor(np.asarray(delta, dtype=np.int64), LERA_BROT, LERA_LOWBITS)
    n_out = C.shape[2]
    coded = {(LERA_BROT - 1) * j for j in coded_words}
    rng = random.Random(seed)
    groups = []
    for w in range(n_out // (LERA_BROT - 1)):
        outs = [(LERA_BROT - 1) * w + b for b in range(1, LERA_BROT - 1)]
        if any(o in coded for o in outs):
            continue
        c0 = time.process_time()
        rows, Ys, m = _group_rows(C, outs)
        rows[len(rows) - 1] = 0
        ex = extend_features(ks, tq, rows, Ys, m, rng, top=top, restarts=restarts)
        groups.append(dict(word=w, outs=outs, products=ex[0] if ex else None,
                           cpu_s=round(time.process_time() - c0, 2)))
    ok = [g for g in groups if g["products"] is not None]
    pick = min(ok, key=lambda g: (g["products"], g["outs"]))["outs"] if ok else None
    perm0, _ = kernel_layout(n_out, coded_words)
    if pick is None:
        perm = perm0
    else:
        perm = pick + [o for o in range(n_out) if o not in coded and o not in pick] + [o for o in perm0 if o in coded]
    return dict(groups=groups, pick=pick, perm=[int(x) for x in perm])


def _pad_screen_job(args):
    lera_src, kernel_src, delta, coded_words, out_json = args
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    r = pad_screen(lera_src, kernel_src, delta, coded_words)
    Path(out_json).write_text(json.dumps(r))
    return r


# ------------------------------------------------- block-partition planner backends (shared / block), v1.4.0-dev
#
# `--backend shared` / `--backend block` (replan) and `plan --planner shared|block` run this tool's own planner
# modules next to this file: planner_shared.py (block partition, structural annihilator, shared
# products, exchange moves; construction recovered from the coded loaders of "i want to run gemma 9b"'s soft-word
# family) and planner_block.py (the pairing-scored generic-loader planner on a given block partition, first
# recovered from Syed Asad Ali's loaders). Pairing after lightningv08's method. Plans are emitted like kernel plans
# (Lera's emitter, groups=None, scratch=True) and go through the same native check and never-worse rule.
#
# Block partitions (`blocks`): None (shared: partition search; block: "recover"), "recover" (each slot's own
# shipped partition, read with recover.py, else the family's most common valid partition for slots of the same kind,
# else a partition search), or an explicit "1,5,13/2,6,14/..." (low monomials 1..15, every slot).

BACKENDS = ("qrom-plan", "kernel", "shared", "block")
PLAN_DEFAULTS = {"shared": dict(partitions=6, keep=2, cpu_cap=90.0, ex_cap=300.0),
                 "block": dict(partitions=6, keep=2, cpu_cap=None, ex_cap=600.0)}
MAX_BLOCK = 4                        # a recovered low block wider than this is not used (2**w - 1 forms per block)


def _local(name):
    """Import a sibling module of this file (planner_shared, planner_block, recover, price_words, chained_and);
    works under `python -I`, which leaves the script's directory off sys.path."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module(name)


def parse_blocks(text, L=(1 << LERA_LOWBITS) - 1):
    """'1,5,13/2,6,14/...' (low monomials 1..L) -> [[0, 4, 12], ...] (slots, monomial - 1); must partition 1..L."""
    blocks = [[int(x) - 1 for x in b.split(",") if x.strip()] for b in str(text).split("/") if b.strip()]
    if sorted(s for b in blocks for s in b) != list(range(L)):
        raise ValueError(f"blocks {text!r} do not partition the low monomials 1..{L}")
    return blocks


def valid_partition(blocks, L=(1 << LERA_LOWBITS) - 1, max_block=MAX_BLOCK) -> bool:
    return (blocks is not None and sorted(s for b in blocks for s in b) == list(range(L))
            and max(len(b) for b in blocks) <= max_block)


def recovered_partitions(km, loaders, words, slots) -> dict:
    """Per slot: the low-monomial partition of the shipped loader (recover.py structure census), 0-based slots."""
    rc = _local("recover")
    gz = _local("gauge")
    out = {}
    for k in slots:
        try:
            sg = gz.slot_gauge(words, k, LERA_J) if gz.off_gauge(words).any() else None   # v1.5: pi-gauge families
            rep, *_ = rc.recover_loader(km, loaders[k], words, k, J=LERA_J, gauge_in=None if sg is None else sg[0])
            lp = [[s - 1 for s in b] for b in rep["structure"]["low_partition"]]
            out[k] = dict(blocks=lp, valid=valid_partition(lp), coded=bool(rep["coded_words"]),
                          products=rep["products"])
        except Exception as ex:  # noqa: BLE001  (a loader not in the cached-product style: recorded, search instead)
            out[k] = dict(blocks=None, valid=False, error=f"{type(ex).__name__}: {ex}"[:300])
    return out


def resolve_blocks(spec, backend, km, loaders, words, slots, coded_of) -> tuple:
    """(per-slot partition or None, report). See the section comment for the rules."""
    if spec is None:
        spec = "recover" if backend == "block" else "search"
    if spec == "search":
        return {k: None for k in slots}, dict(source="search")
    if spec != "recover":
        b = parse_blocks(spec) if isinstance(spec, str) else [list(x) for x in spec]
        return {k: b for k in slots}, dict(source="given", blocks=[[s + 1 for s in x] for x in b])
    if km is None or not loaders:
        raise RuntimeError("blocks 'recover' needs a kickmix build and the family's loaders")
    per = recovered_partitions(km, loaders, words, slots)
    common = {}
    for k, r in per.items():
        if r["valid"]:
            key = (bool(coded_of[k]), tuple(sorted(tuple(sorted(b)) for b in r["blocks"])))
            common[key] = common.get(key, 0) + 1
    chosen, how = {}, {}
    for k in slots:
        if per[k]["valid"]:
            chosen[k], how[k] = per[k]["blocks"], "own"
            continue
        same = [(n, key) for key, n in common.items() if key[0] == bool(coded_of[k])]
        if same:
            chosen[k], how[k] = [list(b) for b in max(same)[1][1]], "family"
        else:
            chosen[k], how[k] = None, "search"
    rep = dict(source="recover", per_slot={str(k): dict(recovered=None if per[k]["blocks"] is None else
                                                        [[s + 1 for s in b] for b in per[k]["blocks"]],
                                                        valid=per[k]["valid"], error=per[k].get("error"),
                                                        used=how[k],
                                                        blocks=None if chosen[k] is None else
                                                        [[s + 1 for s in b] for b in chosen[k]])
                                           for k in slots})
    return chosen, rep


def shared_plan(kind, delta, real_rows, coded_words, seed, *, blocks=None, pad_free=False, exchange=False,
                settings=None, out_json=None) -> dict:
    """One planner run on a loader's delta. kind 'block' with a generic loader and a partition: planner_block
    .plan_generic; otherwise planner_shared.plan (partition search when `blocks` is None). `pad_free`: the
    never-read padded rows are don't-cares (the plan then rewrites them). Exactness is checked here against the
    (rewritten) delta's mixed ANF rows."""
    ps = _local("planner_shared")
    s = dict(PLAN_DEFAULTS["block" if kind == "block" else "shared"], **(settings or {}))
    t0 = time.process_time()
    delta = np.asarray(delta, dtype=np.int64)
    if kind == "block" and not coded_words and blocks is not None:
        pb = _local("planner_block")
        ex = dict(cpu_cap=float(s["ex_cap"])) if exchange else None
        res = pb.plan_generic(delta, real_rows, blocks, seed=seed, pad_free=pad_free,
                              admit_kw=None if s.get("cpu_cap") is None else dict(cpu_cap=float(s["cpu_cap"])),
                              exchange_kw=ex)
        module = "planner_block.plan_generic"
    else:
        ex = dict(cpu_cap=float(s["ex_cap"])) if exchange else None
        res = ps.plan(delta, real_rows, list(coded_words), seed=seed, blocks=blocks, partitions=int(s["partitions"]),
                      keep=int(s["keep"]), pad_free=pad_free,
                      admit_kw=dict(cpu_cap=float(s["cpu_cap"] if s.get("cpu_cap") is not None else 90.0)),
                      exchange_kw=ex)
        module = "planner_shared.plan"
    out = dict(planner=module, kind=kind, seed=int(seed), products=res["products"], exact=False, pad_free=pad_free,
               exchange=bool(exchange), settings=s, blocks_given=None if blocks is None else
               [[x + 1 for x in b] for b in blocks])
    if res["products"] is not None:
        d = np.asarray(res["delta"], np.int64)
        out.update(exact=bool(ps.exact_rows(d, res["terms"])), terms=[list(map(int, t)) for t in res["terms"]],
                   relations=int(res["relations"]), blocks=res["blocks"], pairs=int(res["pairs"]),
                   singles=int(res["singles"]), pad_changed_outputs=res["pad_changed_outputs"],
                   real_rows_unchanged=bool(np.array_equal(d[:real_rows], delta[:real_rows])),
                   pad_rows={int(r): [int(x) for x in d[r]] for r in range(real_rows, len(d))})
        out["exact"] = out["exact"] and out["real_rows_unchanged"]
    for key in ("tried", "phases", "exchange", "per_block"):
        if key in res:
            out[key] = _jsonable(res[key])
    out["cpu_s"] = round(time.process_time() - t0, 1)
    if out_json is not None:
        Path(out_json).write_text(json.dumps(_jsonable(out)))
    return out


def _shared_job(args):
    kind, delta, real_rows, coded_words, seed, blocks, pad_free, exchange, settings, out_json = args
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    r = shared_plan(kind, delta, real_rows, coded_words, seed, blocks=blocks, pad_free=pad_free, exchange=exchange,
                    settings=settings, out_json=out_json)
    return {"best_products": r["products"] if r["exact"] else None, "best_seed": int(seed), "plan": str(out_json),
            "exact": r["exact"]}


def shared_jobs(backend, todo, deltas, A, work, *, seeds, slot_blocks, pad_on, exchange, settings):
    """(jobs, job args) for the shared / block backends, one job per slot and seed, in selection order."""
    jobs, args = [], []
    for k in todo:
        kd = work / "plans" / f"k{k}"
        kd.mkdir(parents=True, exist_ok=False)
        delta, cw = deltas[k]
        kind = backend
        for sd in seeds:
            name = backend + ("_padrows" if pad_on else "") + ("_exchange" if exchange else "")
            jobs.append(dict(k=k, recipe=name, mode=backend, key="seed", value=int(sd), seeds=[int(sd), int(sd)],
                             prefix=kd / f"{backend}_s{sd}"))
            args.append((kind, delta, A, list(cw), int(sd), slot_blocks.get(k), bool(pad_on), bool(exchange),
                         settings, str(jobs[-1]["prefix"]) + ".json"))
    return jobs, args


def _module_record(name):
    p = Path(__file__).resolve().parent / f"{name}.py"
    return dict(path=str(p), sha256=sha256_file(p) if p.is_file() else None)


def replay_plan_files(plans_dir, k) -> list:
    """The saved plan JSONs for slot k under `plans_dir` (v1.7.0, `replan --plans-from`): a replan OUT_DIR, its
    work/plans, a directory of k<k>/ subdirectories, or a k<k>/ directory itself; the first of these that holds plan
    files is used. Plan files are the JSONs the shared, block and kernel backends write (each carries 'terms')."""
    d = Path(plans_dir)
    for cand in (d / "work" / "plans" / f"k{k}", d / "plans" / f"k{k}", d / f"k{k}"):
        files = sorted(f for f in cand.glob("*.json") if f.name not in ("report.json", "build.json")) \
            if cand.is_dir() else []
        if files:
            return files
    return []


def replay_choice(files) -> tuple:
    """(file, plan, skipped) for one slot: among plans marked exact (or not marked) with terms, the fewest products;
    ties go to the first file in sorted order, as replan selects among its seeds. `skipped` lists the rest."""
    best, skipped = None, []
    for f in files:
        kp = json.loads(Path(f).read_text())
        if "terms" not in kp or kp.get("exact") is False or ("products" in kp and kp["products"] is None):
            skipped.append(dict(file=Path(f).name, reason="no exact plan in the file"))
            continue
        n = len(kp["terms"])
        if best is None or n < best[0]:
            best = (n, Path(f), kp)
    return (None, None, skipped) if best is None else (best[1], best[2], skipped)


def replan(family_dir, out_dir, *, planner=None, lera_src, verifier_dir=None, kickmix_dir=None, recipes=None,
           slots=None, procs=6, trials=12, samples=24, repair=60.0, verify=False, verbose=True, backend="qrom-plan",
           kernel_src=None, kernel_seeds=(0, 1), pad_rows=False, pad_screen=True, seeds=None, blocks=None,
           exchange=False, planner_settings=None, gauge_inplace="auto", plans_from=None,
           gauge_free_sources=False) -> dict:
    """Re-plan the loaders of FAMILY_DIR into OUT_DIR/dfthcsos/ (never worse per slot) and write OUT_DIR/report.json.
    Work files (words.npy, planner templates/plans/JSON, every emitted loader and its check) go to OUT_DIR/work/.
    backend "qrom-plan" (default) runs the Rust planner; "kernel" runs qwertykeypadapple's kernel-relation search
    from `kernel_src` (his kernel_search.py, Apache-2.0) with seeds `kernel_seeds` and the full low cache; "shared"
    and "block" run planner_shared.py / planner_block.py (v1.4.0-dev) with `seeds` (default 0), block partitions
    `blocks` (see resolve_blocks), optional `exchange` moves and `planner_settings` (PLAN_DEFAULTS keys).
    v1.7.0 `plans_from` (backend "replay"): no planner runs; each slot's plan is read from saved plan JSONs (see
    replay_plan_files / replay_choice), checked exact against this family's words (real rows unchanged), then emitted
    and checked exactly as a searched plan (gauge-aware emitter, native check, strict-gain and family checks).
    Deterministic: no planner, no CPU caps, no seeds.
    v1.8.0-dev `gauge_free_sources`: pi-gauge slots are emitted with the incoming gauge monomials as free
    low sources (`gauge.GaugeEmitter(free_sources=True)`), and the predicted cache Toffolis drop by their count."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    t0 = time.time()
    if plans_from is not None:
        backend = "replay"
        if not Path(plans_from).is_dir():
            raise FileNotFoundError(f"plans directory not found: {plans_from}")
    if pad_rows and backend == "qrom-plan":
        raise ValueError("pad_rows needs the kernel, shared or block backend")
    fam, out = Path(family_dir), Path(out_dir)
    if backend == "replay":
        pass
    elif backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend}")
    if backend == "qrom-plan":
        planner = Path(planner) if planner else None
        if planner is None or not planner.is_file():
            raise FileNotFoundError(f"qrom-plan binary not found: {planner}")
    elif backend == "kernel":
        if kernel_src is None or not (Path(kernel_src) / "kernel_search.py").is_file():
            raise FileNotFoundError(f"kernel_search.py not found in {kernel_src}: pass --kernel-src")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    vdir = find_verifier_dir(verifier_dir)
    km = find_kickmix(kickmix_dir, vdir)
    if km is None:
        raise RuntimeError("replan needs a kickmix build (word recovery, emission, native checks): pass --kickmix-dir")
    tq, tq_sha = _load_tensorqrom(lera_src)
    _official_modules(vdir)
    C = importlib.import_module("dfthcsos.scoreinfra.circuits")
    recipes = recipes or DEFAULT_RECIPES
    loaders = loader_files(fam)
    npz_in = fam / "submission.npz"
    if not loaders or not npz_in.is_file():
        raise FileNotFoundError(f"{fam} must hold angle_k_*.kmb/.kmx and submission.npz")
    ks = [int(p.stem.split("_")[-1]) for p in loaders]
    if ks != list(range(len(loaders))):
        raise RuntimeError(f"loader files are not one per slot 0..{len(loaders) - 1}: {[p.name for p in loaders]}")
    work, dest = out / "work", out / "dfthcsos"
    work.mkdir(parents=True, exist_ok=False)
    logf = open(out / "log.jsonl", "w")

    def log(**kw):
        kw["t"] = round(time.time() - t0, 2)
        s = json.dumps(_jsonable(kw))
        logf.write(s + "\n")
        logf.flush()
        if verbose:
            print(s[:400], flush=True)

    inputs_sha = {p.name: sha256_file(p) for p in loaders + [npz_in]}
    wr = recover_words(fam, vdir, kickmix_dir)
    if wr["before"] is None:
        raise RuntimeError("words were not recovered from the loaders")
    words = wr["words"]
    if wr["brot"] != LERA_BROT or -(-words.shape[1] // LERA_J) != len(loaders):
        raise RuntimeError(f"unsupported family: brot {wr['brot']}, {words.shape[1]} word columns, {len(loaders)} "
                           f"slots (the emitter layout needs brot 8 and 5 words per slot)")
    words_npy = work / "words.npy"
    np.save(words_npy, words.astype(np.int16 if words.max() < (1 << 15) else np.int64))
    log(event="words", source=wr["source"], shape=list(words.shape), sha256=sha256_file(words_npy))
    gz = _local("gauge")
    conv = family_conventions(fam, words, km, wr["brot"])
    log(event="conventions", gauge=conv["gauge"], loader_summary=conv.get("loader_summary"))
    why = gz.refusal("replan", conv)
    if why:
        (out / "conventions.json").write_text(json.dumps(_jsonable(conv), indent=1) + "\n")
        logf.close()
        raise SystemExit(f"REFUSED {why} (conventions written to {out / 'conventions.json'})")
    gauged = conv["gauge"] == "pi"
    n_sel = (len(words) - 1).bit_length()
    todo = _parse_slots(slots, len(loaders))

    # shipped loaders
    shipped = []
    for k, p in enumerate(loaders):
        c = C.read_circuit(p)
        shipped.append(dict(file=p.name, sha256=inputs_sha[p.name], toffolis=int(c.operation_statistics()
                                                                                   .num_toffoli_like),
                            qubits=int(c.num_qubits), layout=_circuit_layout(C, c)))

    # never-read padded rows, from the family's own row count
    A = len(words)
    pad_first, pad_end = padded_row_range(A)
    pad_rep = None
    pad_on = bool(pad_rows) and pad_end > pad_first
    if pad_rows:
        pad_rep = dict(real_rows=A, addressed_rows=pad_end, padded_rows=[pad_first, pad_end - 1] if pad_on else [],
                       n0=A % (1 << LERA_LOWBITS), active=pad_on,
                       screen=bool(pad_screen) and pad_on and backend == "kernel",
                       screen_top=PAD_SCREEN_TOP, extend_top=PAD_EXTEND_TOP, restarts=PAD_RESTARTS, screen_seed=0,
                       note=None if pad_on else "no never-read rows (row count is a whole number of blocks): "
                                                "his search unchanged")
        log(event="pad_rows", **pad_rep)

    # planner jobs, all slots in one pool; selection in job order (strict improvement only)
    t_plan = time.time()
    deltas, jobs, screens, best = {}, [], {}, {}
    for k in todo:
        deltas[k] = loader_delta(words, k)
        if pad_on:
            deltas[k] = (full_block_delta(deltas[k][0]), deltas[k][1])
            assert len(deltas[k][0]) == pad_end
    if pad_on and pad_screen and backend == "kernel":
        sd = work / "screen"
        sd.mkdir(parents=True, exist_ok=False)
        sargs = [(str(lera_src), str(kernel_src), deltas[k][0], deltas[k][1], str(sd / f"k{k}.json")) for k in todo]
        t_scr = time.time()
        with ProcessPoolExecutor(max(1, int(procs))) as ex:
            screens = dict(zip(todo, ex.map(_pad_screen_job, sargs)))
        pad_rep["screen_wall_s"] = round(time.time() - t_scr, 1)
        log(event="pad_screen", seconds=pad_rep["screen_wall_s"],
            picks={k: (v["pick"], min([g["products"] for g in v["groups"] if g["products"] is not None],
                                      default=None)) for k, v in screens.items()})
    slot_blocks, blocks_rep = {}, None
    if backend in ("shared", "block"):
        slot_blocks, blocks_rep = resolve_blocks(blocks, backend, km, loaders, words, todo,
                                                 {k: deltas[k][1] for k in todo})
        log(event="blocks", **blocks_rep)
        jobs, sargs = shared_jobs(backend, todo, deltas, A, work, seeds=list(seeds or (0,)), slot_blocks=slot_blocks,
                                  pad_on=pad_on, exchange=exchange, settings=planner_settings)
    replay_rep = {}
    if backend == "replay":
        for k in todo:
            files = replay_plan_files(plans_from, k)
            f, kp, skipped = replay_choice(files)
            rr = dict(files=[x.name for x in files], skipped=skipped)
            replay_rep[k] = rr
            if f is None:
                rr["error"] = "no saved plan for this slot" if not files else "no exact plan among the files"
                continue
            delta, cw = deltas[k]
            slot_pad = bool(kp.get("pad_free") or kp.get("pad"))
            if slot_pad:
                delta = full_block_delta(delta)
            d = delta.copy()
            bad = [int(r) for r in (kp.get("pad_rows") or {}) if int(r) < A or int(r) >= len(d)]
            for r, vals in (kp.get("pad_rows") or {}).items():
                if int(r) not in bad:
                    d[int(r)] = np.asarray(vals, dtype=np.int64)
            ok = not bad and bool(_local("planner_shared").exact_rows(d, [tuple(t) for t in kp["terms"]]))
            rr.update(chosen=f.name, sha256=sha256_file(f), planner=kp.get("planner"), kind=kp.get("kind"),
                      seed=kp.get("seed"), products=len(kp["terms"]), pad_free=slot_pad,
                      exact_on_this_family=ok, bad_pad_rows=bad or None)
            if not ok:
                rr["error"] = "the plan does not compute this family's delta (words differ, or a pad row is wrong)"
                continue
            deltas[k] = (delta, cw)
            sd = kp.get("seed")
            jobs.append(dict(k=k, recipe="replay", mode=kp.get("kind") or "replay", key="seed", value=sd,
                             seeds=[sd, sd], prefix=f.with_suffix("")))
            best[k] = dict(best_products=len(kp["terms"]), best_seed=sd, plan=str(f), exact=True, recipe="replay",
                           mode=kp.get("kind") or "replay", value=sd, seeds=[sd, sd], replay_pad=slot_pad)
        log(event="replay", plans_from=str(plans_from), slots={k: {x: v.get(x) for x in
                                                                    ("chosen", "products", "exact_on_this_family",
                                                                     "error")} for k, v in replay_rep.items()})
    for k in (todo if backend in ("qrom-plan", "kernel") else ()):
        if backend == "qrom-plan":
            kind = "coded" if deltas[k][1] else "generic"
            jobs += plan_jobs(work, k, recipes[kind])
        else:
            kd = work / "plans" / f"k{k}"
            kd.mkdir(parents=True, exist_ok=False)
            for sd in kernel_seeds:
                name = "kernel_sixes_full_cache" + ("_padrows" if pad_on else "")
                jobs.append(dict(k=k, recipe=name, mode="kernel", key="seed", value=int(sd),
                                 seeds=[int(sd), int(sd)], prefix=kd / (f"kernel_pad_s{sd}" if pad_on
                                                                        else f"kernel_s{sd}")))
    if backend == "qrom-plan":
        with ThreadPoolExecutor(max(1, int(procs))) as ex:
            results = list(ex.map(lambda j: _run_planner(planner, words_npy, j, trials, samples, repair), jobs))
    elif backend in ("shared", "block"):
        with ProcessPoolExecutor(max(1, int(procs))) as ex:
            results = list(ex.map(_shared_job, sargs))
    elif backend == "replay":
        results = []
    else:
        padopt = dict(n0=A % (1 << LERA_LOWBITS), real_rows=A, extend=True, top=PAD_EXTEND_TOP,
                      restarts=PAD_RESTARTS) if pad_on else None
        args = [(str(lera_src), str(kernel_src), deltas[j["k"]][0], deltas[j["k"]][1], j["value"],
                 str(j["prefix"]) + ".json", screens[j["k"]]["perm"] if j["k"] in screens else None, padopt)
                for j in jobs]
        with ProcessPoolExecutor(max(1, int(procs))) as ex:
            results = list(ex.map(_kernel_job, args))
    errors = []
    for job, res in zip(jobs, results):
        if "error" in res:
            errors.append(dict(k=job["k"], recipe=job["recipe"], value=job["value"], seeds=job["seeds"],
                               error=res["error"]))
            continue
        if res["best_products"] is None:
            continue
        cur = best.get(job["k"])
        if cur is None or res["best_products"] < cur["best_products"]:
            best[job["k"]] = dict(res, recipe=job["recipe"], mode=job["mode"], value=job["value"],
                                  seeds=job["seeds"])
    plan_s = round(time.time() - t_plan, 1)
    log(event="planned", jobs=len(jobs), errors=len(errors), seconds=plan_s,
        best={k: v["best_products"] for k, v in best.items()})

    # emit, check, choose
    slots_rep, chosen = [], {}
    for k in range(len(loaders)):
        sh = shipped[k]
        row = dict(k=k, shipped={x: sh[x] for x in ("file", "sha256", "toffolis", "qubits")}, kept="shipped")
        if k not in todo:
            row["reason"] = "not re-planned (--slots)"
            slots_rep.append(row)
            continue
        delta, cw = deltas[k]
        row["coded_words"] = cw
        caches = cache_toffolis(tq, delta)
        assert caches == expected_cache_toffolis(len(delta)), "cache count model differs from the emitter's"
        free_k = free_low_sources(words, k, gauged) if gauge_free_sources else 0
        caches -= free_k
        b = best.get(k)
        if b is None:
            row["reason"] = (f"replay: {replay_rep.get(k, {}).get('error')}" if backend == "replay"
                             else "the planner found no plan")
            slots_rep.append(row)
            continue
        if backend == "qrom-plan":
            plan = _read_plan_txt(b["plan"])
            emit_groups = [list(g) for g in plan["groups"]]
        else:
            kp = json.loads(Path(b["plan"]).read_text())
            plan = dict(relations=kp.get("relations", "unavailable"), terms=[tuple(t) for t in kp["terms"]])
            emit_groups = None                      # full low cache (their plans)
            b = dict(b, best_toffolis=len(plan["terms"]) + caches)
            if (b["replay_pad"] if backend == "replay" else pad_on):
                # the plan's never-read padded rows; real rows must be unchanged
                new = delta.copy()
                for r, vals in kp["pad_rows"].items():
                    assert int(r) >= A, "a plan changes a real table row"
                    new[int(r)] = np.asarray(vals, dtype=np.int64)
                assert np.array_equal(new[:A], delta[:A]), "real rows changed"
                pad_changed = sorted(int(r) for r in np.flatnonzero((new != delta).any(axis=1)))
                delta = new
        rp = dict(recipe=b["recipe"], mode=b["mode"], value=b["value"], seeds=b["seeds"], seed=b["best_seed"],
                  relations=plan["relations"], products=len(plan["terms"]), planner_toffolis=b["best_toffolis"],
                  planner_cache_toffolis=b["best_toffolis"] - b["best_products"], cache_toffolis=caches,
                  predicted_toffolis=caches + len(plan["terms"]), plan_txt=b["plan"],
                  plan_sha256=sha256_file(b["plan"]))
        rp["planner_cache_offset"] = rp["planner_cache_toffolis"] - caches
        rp["free_low_sources"] = free_k
        if backend == "kernel":
            rp["perm_first_group"] = kp["perm"][:kp["sizes"][0]]
            if pad_on:
                rp.update(pad=kp.get("pad"), pad_rows_changed=pad_changed,
                          pad_rows={str(r): kp["pad_rows"][str(r)] for r in pad_changed},
                          screen=screens.get(k))
        elif backend in ("shared", "block", "replay"):
            rp.update(planner_module=kp.get("planner"), blocks=kp.get("blocks"), exchange=kp.get("exchange"))
            if backend == "replay":
                rp["replayed_plan"] = {x: replay_rep[k].get(x) for x in ("chosen", "sha256", "planner", "kind",
                                                                         "seed", "products", "pad_free")}
            if (b["replay_pad"] if backend == "replay" else pad_on):
                rp.update(pad_rows_changed=pad_changed, pad_rows={str(r): kp["pad_rows"][str(r)] for r in pad_changed})
        ed = work / "emit" / f"k{k}"
        ed.mkdir(parents=True)
        path = ed / f"angle_k_{k}.kmb"
        try:
            if gauged:      # v1.5: pi gauge, gauge-aware borrowed top bits (gauge.py); signed slots call Lera's
                em = gz.GaugeEmitter(tq, *gz.slot_gauge(words, k), inplace=gauge_inplace, cap=QUBIT_CAP,
                                     free_sources=gauge_free_sources)
            else:
                em = tq
            circ = em.build_circuit(delta, list(plan["terms"]), n_sel, LERA_BROT, LERA_LOWBITS,
                                    groups=emit_groups, scratch=True)
            if gauged:      # v1.6: recorded after the build (the layout chosen: kept top bits or in place)
                rp["gauge_emitter"] = em.record()
            circ.to_file(path, format="kmb")
            circ = km.Circuit.from_file(path, format="kmb")
            rp.update(emitted_toffolis=int(circ.operation_statistics().num_toffoli_like),
                      qubits=int(circ.num_qubits), kmb_sha256=sha256_file(path),
                      layout_equal_shipped=_circuit_layout(C, circ) == sh["layout"],
                      native_check=(gz.native_check(km, circ, delta[:len(words)], gz.check_inputs(words, k))
                                    if gauged else native_check(km, circ, delta[:len(words)])))
        except Exception as ex:  # noqa: BLE001  (recorded: the slot keeps the shipped loader)
            rp["emit_error"] = f"{type(ex).__name__}: {ex}"[:600]
        (ed / "build.json").write_text(json.dumps(_jsonable(rp), indent=1) + "\n")
        row["replanned"] = rp
        if "emit_error" in rp:
            row["reason"] = "emit failed"
        elif rp["native_check"]["status"] != "PASS":
            row["reason"] = "native check failed"
        elif rp["emitted_toffolis"] != rp["predicted_toffolis"]:
            row["reason"] = "emitted Toffolis differ from caches + products (model mismatch)"
        elif rp["qubits"] > QUBIT_CAP:
            row["reason"] = f"{rp['qubits']} qubits > {QUBIT_CAP}"
        elif not rp["layout_equal_shipped"]:
            row["reason"] = "register layout differs from the shipped loader"
        elif rp["emitted_toffolis"] >= sh["toffolis"]:
            row["reason"] = "not strictly fewer Toffolis than shipped"
        else:
            row["kept"], row["reason"] = "replanned", "strictly fewer Toffolis"
            chosen[k] = path
        slots_rep.append(row)
        log(event="slot", k=k, shipped=sh["toffolis"], replanned=rp.get("emitted_toffolis"), kept=row["kept"],
            reason=row["reason"])

    # assemble a candidate, check it as a family, then write the result (the shipped family if a check fails)
    def assemble(target, use):
        target.mkdir(parents=True, exist_ok=False)
        for k, p in enumerate(loaders):
            if k in use:
                shutil.copyfile(use[k], target / f"angle_k_{k}.kmb")
            else:
                shutil.copyfile(p, target / p.name)
        shutil.copyfile(npz_in, target / "submission.npz")

    checks, before = {}, wr["before"]
    after_eval = None
    if chosen:
        cand = work / "candidate"
        assemble(cand, chosen)
        lays = []
        for p in loader_files(cand):
            lays.append(json.dumps(_circuit_layout(C, C.read_circuit(p))))
        checks["layout_mismatch"] = len(set(lays)) != 1
        checks["max_loader_qubits"] = max(int(C.read_circuit(p).num_qubits) for p in loader_files(cand))
        V = importlib.import_module("dfthcsos.verify")
        try:
            r = V.evaluate_circuits(V.load_inputs(cand / "submission.npz"), cand, V.load_tree_edges(None, cand))
            got = np.asarray(r["angle_words"]).astype(np.int64)
            checks["chain_break"] = not (got.shape == words.shape and np.array_equal(got, words))
            mb, ma = before["metrics"], _jsonable(r["metrics"])
            checks["metrics_unchanged"] = all(mb[x] == ma[x] for x in ("error_1", "error_2", "Lambda", "E_gap",
                                                                         "lambda_eff"))
            checks["rejection_reason"] = V.rejection_reason(r["metrics"], r["resources"])
            after_eval = dict(metrics=ma, resources=_jsonable(r["resources"]), score=float(r["score"]))
            checks["cost_not_worse"] = after_eval["score"] <= before["score"]
        except Exception as ex:  # noqa: BLE001
            checks["evaluate_error"] = f"{type(ex).__name__}: {ex}"[:600]
        ok = (not checks.get("layout_mismatch") and checks["max_loader_qubits"] <= QUBIT_CAP
              and checks.get("chain_break") is False and checks.get("metrics_unchanged") is True
              and checks.get("rejection_reason") is None and checks.get("cost_not_worse") is True)
        checks["family_ok"] = bool(ok)
        if not ok:
            log(event="family_check_failed", checks=checks)
            for row in slots_rep:
                if row["kept"] == "replanned":
                    row["kept"], row["reason"] = "shipped", "family check failed: whole family reverted"
            chosen = {}
            after_eval = None
    assemble(dest, chosen)
    outputs_sha = {p.name: sha256_file(p) for p in sorted(dest.iterdir())}
    if not chosen:
        assert all(outputs_sha[p.name] == inputs_sha[p.name] for p in loaders + [npz_in]), "unchanged copy mismatch"

    per_pass = dict(shipped=sum(s["toffolis"] for s in shipped),
                    replanned=sum(r["replanned"]["emitted_toffolis"] if r.get("replanned", {}).get(
                        "emitted_toffolis") is not None else r["shipped"]["toffolis"] for r in slots_rep),
                    kept=sum(r["replanned"]["emitted_toffolis"] if r["kept"] == "replanned"
                             else r["shipped"]["toffolis"] for r in slots_rep))
    per_pass["saving"] = per_pass["shipped"] - per_pass["kept"]
    rs = before["resources"]
    lam, tof, qub = float(before["metrics"]["lambda_eff"]), float(rs["toffoli"]), int(rs["qubits"])
    cost = dict(before_official_evaluation=before["score"], lambda_eff=lam, toffoli_before=tof, qubits_before=qub,
                toffoli_after_projected=tof - 2 * per_pass["saving"],
                after_projected=lam * (tof - 2 * per_pass["saving"]) * qub,
                after_official_evaluation=after_eval["score"] if after_eval else before["score"],
                toffoli_after_official_evaluation=(after_eval["resources"]["toffoli"] if after_eval else tof),
                qubits_after_official_evaluation=(after_eval["resources"]["qubits"] if after_eval else qub),
                tier_note="before / after_official_evaluation: the official evaluate_circuits (default seed), a "
                          "static model / projection until the local official verifier passes (--verify); "
                          "after_projected: lambda_eff x (Toffolis - 2 x per-pass saving: forward and reverse "
                          "pass) x qubits, arithmetic")
    rep = dict(
        tool=f"dfthc_refit {__version__}", mode="replan", family_dir=str(fam), out_dir=str(out),
        rule="a slot changes only if its re-planned loader has strictly fewer Toffolis, passes the native check, "
             "uses <= 128 qubits and keeps the shipped register layout; words and weights never change (with "
             "pad_rows only never-read padded rows >= the table's row count may differ in the emitted delta)",
        planner=(dict(backend="qrom-plan", path=str(planner), sha256=sha256_file(planner),
                      cache_constant=PLANNER_CACHE_TOFFOLIS, trials=trials, samples=samples, repair_s=repair,
                      procs=procs, jobs=len(jobs), errors=errors) if backend == "qrom-plan" else
                 dict(backend="kernel", credit="qwertykeypadapple, kernel_search.py (Apache-2.0)",
                      path=str(Path(kernel_src) / "kernel_search.py"),
                      sha256=sha256_file(Path(kernel_src) / "kernel_search.py"), seeds=list(kernel_seeds),
                      layout="sixes, coded bit-0 outputs last; full low cache", procs=procs, jobs=len(jobs),
                      errors=errors, pad_rows=pad_rep) if backend == "kernel" else
                 dict(backend="replay", plans_from=str(plans_from), per_slot=replay_rep,
                      rule="no planner run; per slot the fewest-product exact plan among the saved JSONs (ties: first "
                           "file in sorted order), re-checked exact on this family's words, then emitted and checked "
                           "as a searched plan", procs=None, jobs=len(jobs), errors=errors)
                 if backend == "replay" else
                 dict(backend=backend, credit="planner_shared.py / planner_block.py (this tool; construction after "
                                              "\"i want to run gemma 9b\"'s coded loaders and Syed Asad Ali's block "
                                              "partition, pairing after lightningv08)",
                      planner_shared=_module_record("planner_shared"), planner_block=_module_record("planner_block"),
                      recover=_module_record("recover"), seeds=list(seeds or (0,)), exchange=bool(exchange),
                      settings=dict(PLAN_DEFAULTS[backend], **(planner_settings or {})), blocks=blocks_rep,
                      layout="full low cache", procs=procs, jobs=len(jobs), errors=errors, pad_rows=pad_rep)),
        emitter=dict(module="tensorqrom.build_circuit (Lera, unmodified)", path=str(Path(lera_src) / "tensorqrom.py"),
                     sha256=tq_sha, brot=LERA_BROT, lowbits=LERA_LOWBITS, scratch=True,
                     gauge=None if not gauged else dict(
                         module="gauge.build_circuit_gauge (this tool, after Lera's build_circuit; full low cache)",
                         sha256=_module_record("gauge"), inplace_policy=gauge_inplace,
                         free_sources=bool(gauge_free_sources))),
        conventions=conv,
        recipes=recipes if backend == "qrom-plan" else None, slots_replanned=todo,
        words=dict(source=wr["source"], shape=list(words.shape), npy_sha256=sha256_file(words_npy)),
        slots=slots_rep, toffolis_per_pass=per_pass, changed_slots=sorted(chosen), family_checks=checks,
        cost=cost, inputs_sha256=inputs_sha, outputs_sha256=outputs_sha,
        input_family_sha256=family_sha256(inputs_sha), output_family_sha256=family_sha256(outputs_sha),
    )
    t_ver = time.time()
    if verify:
        va = verify_dir(dest, vdir, kickmix_dir)
        rep["verify_after"] = va if va is not None else "unavailable (no kickmix build)"
        log(event="verify_after", result={x: va.get(x) for x in ("status", "score", "lambda_eff", "error")}
            if va else None)
    rep["wall_s"] = dict(plan=plan_s, verify=round(time.time() - t_ver, 1),
                         total=round(time.time() - t0, 1))
    rep["versions"] = _versions(vdir, wr["kickmix"])
    (out / "report.json").write_text(json.dumps(_jsonable(rep), indent=1) + "\n")
    log(event="done", per_pass=per_pass, changed=sorted(chosen))
    logf.close()
    return _jsonable(rep)


def replan_main(argv):
    ap = argparse.ArgumentParser(prog="dfthc_refit.py replan",
                                 description="Re-plan the ANGLE_k loaders of a dfthcsos family on its fixed words and "
                                             "keep, per slot, the cheaper of shipped and re-planned.")
    ap.add_argument("family_dir", type=Path, help="angle_k_*.kmb/.kmx + submission.npz")
    ap.add_argument("out_dir", type=Path, help="writes OUT_DIR/dfthcsos/, OUT_DIR/report.json, OUT_DIR/work/")
    ap.add_argument("--planner", default=os.environ.get("DFTHC_REFIT_QROM_PLAN") or shutil.which("qrom-plan"),
                    help="the qrom-plan binary (or $DFTHC_REFIT_QROM_PLAN, else `qrom-plan` on PATH)")
    ap.add_argument("--lera-src", default=os.environ.get("DFTHC_REFIT_LERA_SRC"),
                    help="directory holding Lera's tensorqrom.py (or $DFTHC_REFIT_LERA_SRC)")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--recipes", type=Path, help="JSON {generic: [...], coded: [...]} replacing the default recipes")
    ap.add_argument("--slots", help="slots to re-plan, e.g. 0-3,7 (default all)")
    ap.add_argument("--procs", type=int, default=6, help="parallel planner processes")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--repair", type=float, default=60.0, help="planner repair budget, seconds")
    ap.add_argument("--verify", action="store_true", help="run the unmodified official verify() on the output")
    ap.add_argument("--backend", choices=list(BACKENDS), default="qrom-plan",
                    help="planner: the qrom-plan binary (default), qwertykeypadapple's kernel-relation search "
                         "(--kernel-src), or this tool's block-partition planners: 'shared' (planner_shared: "
                         "structural annihilator + shared products) or 'block' (planner_block for generic loaders on "
                         "a block partition; coded slots use planner_shared)")
    ap.add_argument("--kernel-src", default=os.environ.get("DFTHC_REFIT_KERNEL_SRC"),
                    help="directory holding qwertykeypadapple's kernel_search.py (or $DFTHC_REFIT_KERNEL_SRC)")
    ap.add_argument("--kernel-seeds", default="0,1", help="seeds for the kernel backend (comma list)")
    ap.add_argument("--pad-rows", action="store_true",
                    help="kernel/shared/block backends: treat the never-read padded rows (rows >= the table's row "
                         "count up to the emitter's whole 16-address blocks) as controllability don't-cares (opt-in)")
    ap.add_argument("--no-pad-screen", action="store_true",
                    help="kernel backend with --pad-rows: skip the per-loader word-group screen (keep the default "
                         "sixes order)")
    _shared_args(ap)
    ap.add_argument("--gauge-inplace", choices=("auto", "never", "always"), default="auto",
                    help="pi-gauge slots whose gauge changes: 'never' keeps one top bit per changing word (1.5 layout); "
                         "'always' changes the gauge in place (rank(g) kept top bits; v1.6); 'auto' (default) "
                         "uses the 1.5 layout unless it exceeds 128 qubits")
    ap.add_argument("--gauge-free-sources", action="store_true",
                    help="v1.8.0-dev: pi-gauge slots use the incoming gauge monomials of words whose gauge "
                         "does not change as free low sources (Syed Asad Ali's construction): fewer low-cache ANDs, "
                         "predicted and emitted alike. Off by default (earlier emissions are byte-identical)")
    ap.add_argument("--plans-from", type=Path,
                    help="v1.7.0 replay: emit each slot from saved plan JSONs instead of searching (a replan OUT_DIR, "
                         "its work/plans, or a directory of k<slot>/ subdirectories). No planner, no CPU caps, "
                         "deterministic; the emitter, native check, strict-gain rule and family checks are unchanged")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.plans_from is not None:
        if (a.exchange or a.blocks or a.seeds or a.partitions is not None or a.keep is not None
                or a.cpu_cap is not None or a.ex_cap is not None or a.pad_rows or a.recipes
                or a.backend != "qrom-plan"):
            ap.error("--plans-from replays saved plans: planner options (--backend, --seeds, --blocks, --exchange, "
                     "--partitions, --keep, --cpu-cap, --ex-cap, --pad-rows, --recipes) do not apply")
        if not a.lera_src:
            ap.error("--lera-src is required")
        rep = replan(a.family_dir, a.out_dir, lera_src=a.lera_src, verifier_dir=a.verifier_dir,
                     kickmix_dir=a.kickmix_dir, slots=a.slots, verify=a.verify, verbose=not a.quiet,
                     gauge_inplace=a.gauge_inplace, plans_from=a.plans_from,
                     gauge_free_sources=a.gauge_free_sources)
        print(json.dumps(dict(toffolis_per_pass=rep["toffolis_per_pass"], changed_slots=rep["changed_slots"],
                              cost=rep["cost"], output_family_sha256=rep["output_family_sha256"],
                              verify_after=rep.get("verify_after")), indent=1))
        return
    if not a.lera_src or (a.backend == "qrom-plan" and not a.planner):
        ap.error("--lera-src is required, and --planner for the qrom-plan backend")
    if a.pad_rows and a.backend == "qrom-plan":
        ap.error("--pad-rows needs --backend kernel, shared or block")
    if a.no_pad_screen and a.backend != "kernel":
        ap.error("--no-pad-screen needs --backend kernel")
    if a.backend not in ("shared", "block") and (a.exchange or a.blocks or a.seeds or a.partitions is not None
                                                  or a.keep is not None or a.cpu_cap is not None
                                                  or a.ex_cap is not None):
        ap.error("--seeds/--blocks/--exchange/--partitions/--keep/--cpu-cap/--ex-cap need --backend shared or block")
    recipes = json.loads(a.recipes.read_text()) if a.recipes else None
    rep = replan(a.family_dir, a.out_dir, planner=a.planner, lera_src=a.lera_src, verifier_dir=a.verifier_dir,
                 kickmix_dir=a.kickmix_dir, recipes=recipes, slots=a.slots, procs=a.procs, trials=a.trials,
                 samples=a.samples, repair=a.repair, verify=a.verify, verbose=not a.quiet, backend=a.backend,
                 kernel_src=a.kernel_src, kernel_seeds=[int(x) for x in a.kernel_seeds.split(",")],
                 pad_rows=a.pad_rows, pad_screen=not a.no_pad_screen, gauge_inplace=a.gauge_inplace,
                 gauge_free_sources=a.gauge_free_sources, **_shared_kwargs(a))
    print(json.dumps(dict(toffolis_per_pass=rep["toffolis_per_pass"], changed_slots=rep["changed_slots"],
                          cost=rep["cost"], verify_after=rep.get("verify_after")), indent=1))


# ------------------------------------------------------------------------- family subcommands (v1.4.0-dev)
#
# `plan`, `recover`, `price-words` and `chained-and` make the standalone modules next to this file (planner_shared,
# planner_block, recover, price_words, chained_and) first-class subcommands, one command per family. They never
# change a family: `replan` is the only subcommand that writes a new dfthcsos directory.


def _shared_args(ap):
    """Arguments of the shared / block planners (replan and plan)."""
    ap.add_argument("--seeds", help="planner seeds, comma list (shared/block default 0; kernel default 0,1)")
    ap.add_argument("--blocks", help="block partition of the low monomials: 'recover' (each shipped loader's own, "
                                     "read with recover.py; block default), 'search' (shared default) or an explicit "
                                     "'1,5,13/2,6,14/3,7,15/4,8,11/9,10,12'")
    ap.add_argument("--exchange", action="store_true",
                    help="shared/block: run exchange moves (drop j relations of a block, re-admit j+1; zero-cost "
                         "replacements allowed) after admission")
    ap.add_argument("--partitions", type=int, help="shared: random triple partitions scored structurally (default 6)")
    ap.add_argument("--keep", type=int, help="shared: best partitions taken through shared admission (default 2)")
    ap.add_argument("--cpu-cap", type=float,
                    help="admission CPU-seconds per partition (shared default 90; block: module defaults)")
    ap.add_argument("--ex-cap", type=float, help="exchange-move CPU-seconds (shared default 300, block 600)")


def _shared_kwargs(a) -> dict:
    settings = {k: getattr(a, k) for k in ("partitions", "keep", "cpu_cap", "ex_cap") if getattr(a, k) is not None}
    return dict(seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None, blocks=a.blocks,
                exchange=bool(a.exchange), planner_settings=settings or None)


def _family_words(fam, words, verifier_dir, kickmix_dir):
    """(words, word-source record, official evaluation before or None). `words`: a .npy path or an array; else the
    words the loaders load, recovered with the official evaluate_circuits (needs a kickmix build)."""
    if words is not None:
        if isinstance(words, (str, os.PathLike)):
            p = Path(words)
            return (np.load(p, allow_pickle=False).astype(np.int64), dict(source=f"given ({p})",
                                                                          sha256=sha256_file(p)), None)
        return np.asarray(words, np.int64), dict(source="given (array)"), None
    vdir = find_verifier_dir(verifier_dir)
    if find_kickmix(kickmix_dir, vdir) is None:
        raise RuntimeError("the words must be recovered from the loaders (needs a kickmix build: --kickmix-dir) or "
                           "given with --words")
    wr = recover_words(Path(fam), vdir, kickmix_dir)
    return wr["words"], dict(source=wr["source"], npz_angle_words_match=wr["npz_angle_words_match"]), wr["before"]


def family_conventions(fam, words, km=None, brot=None) -> dict:
    """The convention detector (gauge.py) on a family: word conventions from `words` and the npz tree, plus per-slot
    loader classes when a kickmix module is given and the family's loaders are present."""
    gz = _local("gauge")
    fam = Path(fam)
    npz = fam / "submission.npz" if fam.is_dir() else fam
    z = np.load(npz, allow_pickle=False) if npz.is_file() else {}
    files = getattr(z, "files", [])
    brot = int(brot if brot is not None else (z["brot"] if "brot" in files else LERA_BROT))
    tree = z["tree_edges"] if "tree_edges" in files else None
    loaders = loader_files(fam) if (km is not None and fam.is_dir()) else []
    j = LERA_J
    if loaders:
        j = -(-np.asarray(words).shape[1] // len(loaders))
    return _jsonable(gz.conventions(km, loaders, np.asarray(words, np.int64), tree, brot, j=j))


def conventions_family(family_dir, out=None, *, words=None, verifier_dir=None, kickmix_dir=None, loaders=True,
                       verbose=True) -> dict:
    """`conventions` subcommand: storage range, tree, gauge class with its pattern group, and (with kickmix) the
    loader class per slot. Words: given, else recovered from the loaders (kickmix), else the npz angle_words."""
    t0 = time.time()
    fam = Path(family_dir)
    km = _optional_kickmix(kickmix_dir, verifier_dir) if loaders else None
    if words is None and km is None:
        z = np.load(fam / "submission.npz", allow_pickle=False)
        w, wsrc = np.asarray(z["angle_words"], np.int64), dict(source="npz angle_words (no kickmix)")
    else:
        w, wsrc, _ = _family_words(fam, words, verifier_dir, kickmix_dir)
    conv = family_conventions(fam, w, km)
    gz = _local("gauge")
    rep = _jsonable(dict(tool=f"dfthc_refit {__version__}", mode="conventions", family_dir=str(fam),
                         words=dict(wsrc, shape=list(w.shape)), conventions=conv,
                         support={c: dict(handles=gz.SUPPORT[c], refused=gz.refusal(c, conv))
                                  for c in gz.SUPPORT},
                         gauge_module=_module_record("gauge"), wall_s=round(time.time() - t0, 1)))
    if out is not None:
        out = Path(out)
        if out.exists():
            raise FileExistsError(f"{out} exists")
        out.write_text(json.dumps(rep, indent=1) + "\n")
    if verbose:
        c = conv
        print(f"storage {c['storage_kind']} {c['storage']}  tree {c['tree']}  gauge {c['gauge']}  off-gauge "
              f"{c['off_gauge_entries']} ({c['off_gauge_share']:.1%})  row patterns {c['patterns']}  exact symmetry "
              f"{c['exact_symmetry']}  pattern = f(low address bits) {c['pattern_low_bits']}")
        if "slots" in c:
            _print_table(c["slots"], [("k", "k"), ("loader", "loader"), ("Toffolis", "toffolis"), ("qubits", "qubits"),
                                      ("fwd", lambda r: r["runs"]["forward"]["correct"]),
                                      ("rev", lambda r: r["runs"]["reverse"]["correct"]),
                                      ("random-gauge rows wrong", lambda r: r["runs"]["random_gauge"]["rows_wrong"]),
                                      ("gauge change", "gauge_change")])
        for cmd, v in rep["support"].items():
            print(f"  {cmd:12s} {'REFUSED: ' + v['refused'] if v['refused'] else 'ok'}")
    return rep


def conventions_main(argv):
    ap = argparse.ArgumentParser(
        prog="dfthc_refit.py conventions",
        description="Word conventions of a family (gauge.py): storage range, Givens tree, gauge class (signed: top "
                    "bit = bit below; pi: off-gauge words whose canonical form leaves every u u^T unchanged; free "
                    "range: not a symmetry), the gauge pattern group, and per slot whether the loader is XOR-in or "
                    "chained. Every family subcommand runs this first and refuses what it cannot handle.")
    ap.add_argument("family_dir", type=Path)
    ap.add_argument("--out", type=Path, help="write the report here (a new .json file)")
    ap.add_argument("--words", type=Path, help="the words the loaders load (.npy); default: recovered (kickmix)")
    ap.add_argument("--no-loaders", action="store_true", help="word conventions only (no loader simulation)")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    return conventions_family(a.family_dir, a.out, words=a.words, verifier_dir=a.verifier_dir,
                              kickmix_dir=a.kickmix_dir, loaders=not a.no_loaders, verbose=not a.quiet)


def gauge_options_family(family_dir, slots, out=None, *, words=None, plans=None, lera_src=None, emit_dir=None,
                         structural=0, structural_seed=0, verifier_dir=None, kickmix_dir=None, loaders=True,
                         max_basis=None, verbose=True) -> dict:
    """`gauge-options` subcommand (v1.6.0-dev; gauge_options.py): per slot, the exact word symmetries split
    into free / constrained / costed directions (bases, dimensions), the loader's gauge contract (p_in, p_out,
    address-only, g, rank), prices (qubits per constrained direction; optional structural Toffoli model for costed
    variants) and suggested repairs. With --plans (a replan OUT_DIR or its plans dir) and --lera-src, the best plan
    of each slot is emitted in both gauge layouts and native-checked."""
    t0 = time.time()
    go = _local("gauge_options")
    gz = _local("gauge")
    fam = Path(family_dir)
    km = _optional_kickmix(kickmix_dir, verifier_dir) if (loaders or plans) else None
    if words is None and km is None:
        z = np.load(fam / "submission.npz", allow_pickle=False)
        w, wsrc = np.asarray(z["angle_words"], np.int64), dict(source="npz angle_words (no kickmix)")
    else:
        w, wsrc, _ = _family_words(fam, words, verifier_dir, kickmix_dir)
    z = np.load(fam / "submission.npz", allow_pickle=False)
    edges, brot = np.asarray(z["tree_edges"], np.int64), int(z["brot"]) if "brot" in z.files else LERA_BROT
    conv = gz.word_conventions(w, edges, brot)
    n_slots = -(-w.shape[1] // LERA_J)
    ks = _parse_slots(slots, n_slots)
    grp = _local("gauge_search").group(edges, brot)
    shipped = {}
    if km is not None and loaders and loader_files(fam):
        for r in gz.loader_classes(km, loader_files(fam), w, brot=brot, j=LERA_J, slots=set(ks)):
            shipped[r["k"]] = r
    tq = None
    if plans is not None:
        if km is None or lera_src is None:
            raise RuntimeError("--plans needs a kickmix build and --lera-src (the plans are emitted and checked)")
        tq, tq_sha = _load_tensorqrom(lera_src)
    reps = []
    for k in ks:
        plan = None
        if plans is not None:
            found = go.find_plan(plans, k)
            if found is not None:
                pf, kp = found
                plan = dict(plan_file=str(pf), plan_sha256=sha256_file(pf),
                            **go.emit_plan(km, tq, w, k, kp, loader_delta,
                                           None if emit_dir is None else Path(emit_dir), sha256=sha256_file))
        reps.append(go.slot_options(w, edges, k, group=grp, shipped=shipped.get(k), plan=plan, structural=structural,
                                    structural_seed=structural_seed, brot=brot, max_basis=max_basis))
    rep = _jsonable(dict(tool=f"dfthc_refit {__version__}", mode="gauge-options", family_dir=str(fam),
                         words=dict(wsrc, shape=list(w.shape)),
                         conventions={x: conv.get(x) for x in ("gauge", "exact_symmetry", "tree", "storage_kind",
                                                                "off_gauge_entries", "patterns", "pattern_low_bits")},
                         group_dims=grp["dims"], slots=reps,
                         emitter=None if tq is None else dict(tensorqrom=str(Path(lera_src) / "tensorqrom.py"),
                                                              sha256=tq_sha),
                         modules={m: _module_record(m) for m in ("gauge", "gauge_search", "gauge_options")},
                         tiers=dict(classification="exact (GF(2) linear algebra on the tree's symmetry group)",
                                    widths="static model (emitted widths under plan.* are the circuits' own)",
                                    native_check="local prescreen (gauge inputs; the official verifier decides)",
                                    costed_prices="static model (structural count)"),
                         wall_s=round(time.time() - t0, 1)))
    if out is not None:
        out = Path(out)
        if out.exists():
            raise FileExistsError(f"{out} exists")
        out.write_text(json.dumps(rep, indent=1) + "\n")
    if verbose:
        print(f"family {fam.name}: gauge {conv['gauge']}, tree {conv['tree']}, group 2^{grp['dims']['group']} per row "
              f"(pure-pi kernel {grp['dims']['pure_pi']})")
        for r in rep["slots"]:
            for line in go.summary_lines(r):
                print(line)
    return rep


def gauge_options_main(argv):
    ap = argparse.ArgumentParser(
        prog="dfthc_refit.py gauge-options",
        description="Per-loader gauge characterisation (gauge_options.py): the slot's exact word symmetries as free / "
                    "constrained / costed directions, its gauge contract and widths, prices and suggested repairs.")
    ap.add_argument("family_dir", type=Path)
    ap.add_argument("--slot", required=True, help="slot(s), e.g. 11 or 11-13 or 4,8")
    ap.add_argument("--plans", type=Path, help="replan OUT_DIR (or its work/plans) whose best plan per slot is emitted "
                                               "in both gauge layouts and native-checked (needs --lera-src)")
    ap.add_argument("--lera-src", default=os.environ.get("DFTHC_REFIT_LERA_SRC"))
    ap.add_argument("--emit-dir", type=Path, help="write the emitted loaders here (angle_k_K.<layout>.kmb)")
    ap.add_argument("--structural", type=int, default=0,
                    help="price N random allowed costed variants with planner_shared's structural count")
    ap.add_argument("--structural-seed", type=int, default=0)
    ap.add_argument("--max-basis", type=int, help="list at most this many basis vectors per class in the JSON")
    ap.add_argument("--words", type=Path, help="the words the loaders load (.npy); default: recovered (kickmix)")
    ap.add_argument("--no-loaders", action="store_true", help="skip the shipped-loader classification")
    ap.add_argument("--out", type=Path, help="write the JSON report here (a new file)")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    gauge_options_family(a.family_dir, a.slot, a.out, words=a.words, plans=a.plans, lera_src=a.lera_src,
                         emit_dir=a.emit_dir, structural=a.structural, structural_seed=a.structural_seed,
                         verifier_dir=a.verifier_dir, kickmix_dir=a.kickmix_dir, loaders=not a.no_loaders,
                         max_basis=a.max_basis, verbose=not a.quiet)
    return 0


def _optional_kickmix(kickmix_dir, verifier_dir):
    try:
        vdir = find_verifier_dir(verifier_dir)
    except FileNotFoundError:
        vdir = None
    return find_kickmix(kickmix_dir, vdir)


def _shipped_counts(km, loaders) -> list:
    out = []
    for p in loaders:
        c = km.Circuit.from_file(str(p), format="kmb" if p.suffix == ".kmb" else "kmx")
        out.append(dict(file=p.name, sha256=sha256_file(p), toffolis=int(c.operation_statistics().num_toffoli_like),
                        qubits=int(c.num_qubits)))
    return out


def _new_out(out_dir):
    if out_dir is None:
        return None
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _print_table(rows, cols):
    """Plain fixed-width table of `cols` [(header, key or callable)] over dict rows."""
    cells = [[h for h, _ in cols]]
    for r in rows:
        line = []
        for _, k in cols:
            v = k(r) if callable(k) else r.get(k)
            line.append("-" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
        cells.append(line)
    w = [max(len(c[i]) for c in cells) for i in range(len(cols))]
    for c in cells:
        print("  ".join(x.rjust(w[i]) for i, x in enumerate(c)))


# ---------------------------------------------------------------------------------------------------------- plan


def plan_family(family_dir, out_dir=None, *, planner, words=None, lera_src=None, kernel_src=None, verifier_dir=None,
                kickmix_dir=None, slots=None, seeds=None, pad_rows=False, pad_screen=True, exchange=False,
                blocks=None, procs=2, planner_settings=None, verbose=True, gauge_free_sources=False) -> dict:
    """Per-slot plans of a family's loaders with one planner (kernel / shared / block), without emitting anything:
    products, predicted Toffolis (cache + products) and, with a kickmix build, the shipped Toffolis. A static
    model / projection; `replan --backend ...` emits, checks and applies positive-gain acceptance per slot.
    Writes OUT_DIR/plans.json and OUT_DIR/work/ (every plan) when `out_dir` is given."""
    import tempfile
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    if planner not in ("kernel", "shared", "block"):
        raise ValueError(f"unknown planner {planner}")
    if planner == "kernel":
        if lera_src is None or not (Path(lera_src) / "tensorqrom.py").is_file():
            raise FileNotFoundError("the kernel planner needs Lera's tensorqrom.py: pass --lera-src")
        if kernel_src is None or not (Path(kernel_src) / "kernel_search.py").is_file():
            raise FileNotFoundError(f"kernel_search.py not found in {kernel_src}: pass --kernel-src")
    fam = Path(family_dir)
    out = _new_out(out_dir)
    words, wsrc, before = _family_words(fam, words, verifier_dir, kickmix_dir)
    n = -(-words.shape[1] // LERA_J)
    loaders = loader_files(fam) if fam.is_dir() else []
    km = _optional_kickmix(kickmix_dir, verifier_dir)
    conv = family_conventions(fam, words, km if len(loaders) == n else None)
    why = _local("gauge").refusal("plan", conv)
    if why:
        raise SystemExit(f"REFUSED {why}")
    shipped = _shipped_counts(km, loaders) if km is not None and len(loaders) == n else None
    todo = _parse_slots(slots, n)
    seeds = list(seeds) if seeds else ([0, 1] if planner == "kernel" else [0])
    A = len(words)
    pad_first, pad_end = padded_row_range(A)
    pad_on = bool(pad_rows) and pad_end > pad_first
    deltas = {}
    for k in todo:
        deltas[k] = loader_delta(words, k)
        if pad_on:
            deltas[k] = (full_block_delta(deltas[k][0]), deltas[k][1])
    tmp = tempfile.TemporaryDirectory() if out is None else None
    work = Path(tmp.name) if tmp is not None else out / "work"
    work.mkdir(parents=True, exist_ok=True)
    blocks_rep, screens = None, {}
    try:
        if planner == "kernel":
            if pad_on and pad_screen:
                sd = work / "screen"
                sd.mkdir(parents=True, exist_ok=False)
                sargs = [(str(lera_src), str(kernel_src), deltas[k][0], deltas[k][1], str(sd / f"k{k}.json"))
                         for k in todo]
                with ProcessPoolExecutor(max(1, int(procs))) as ex:
                    screens = dict(zip(todo, ex.map(_pad_screen_job, sargs)))
            padopt = dict(n0=A % (1 << LERA_LOWBITS), real_rows=A, extend=True, top=PAD_EXTEND_TOP,
                          restarts=PAD_RESTARTS) if pad_on else None
            jobs, args = [], []
            for k in todo:
                kd = work / "plans" / f"k{k}"
                kd.mkdir(parents=True, exist_ok=False)
                for sd in seeds:
                    jobs.append(dict(k=k, recipe="kernel_sixes_full_cache" + ("_padrows" if pad_on else ""),
                                     mode="kernel", value=int(sd), prefix=kd / (f"kernel_pad_s{sd}" if pad_on
                                                                               else f"kernel_s{sd}")))
                    args.append((str(lera_src), str(kernel_src), deltas[k][0], deltas[k][1], int(sd),
                                 str(jobs[-1]["prefix"]) + ".json", screens[k]["perm"] if k in screens else None,
                                 padopt))
            fn = _kernel_job
        else:
            slot_blocks, blocks_rep = resolve_blocks(blocks, planner, km, loaders, words, todo,
                                                     {k: deltas[k][1] for k in todo})
            jobs, args = shared_jobs(planner, todo, deltas, A, work, seeds=seeds, slot_blocks=slot_blocks,
                                     pad_on=pad_on, exchange=exchange, settings=planner_settings)
            fn = _shared_job
        with ProcessPoolExecutor(max(1, int(procs))) as ex:
            results = list(ex.map(fn, args))
        best, per_seed = {}, {}
        for job, res in zip(jobs, results):
            per_seed.setdefault(job["k"], []).append(dict(seed=job["value"], products=res["best_products"]))
            if res["best_products"] is None:
                continue
            cur = best.get(job["k"])
            if cur is None or res["best_products"] < cur["best_products"]:
                best[job["k"]] = dict(res, seed=job["value"])
        rows = []
        for k in todo:
            delta, cw = deltas[k]
            cache = expected_cache_toffolis(len(delta))
            free_k = free_low_sources(words, k, conv["gauge"] == "pi") if gauge_free_sources else 0
            cache -= free_k                    # v1.8: the free-source emitter's low cache
            row = dict(k=k, coded_words=cw, cache_toffolis=cache, free_low_sources=free_k, seeds=per_seed.get(k, []))
            if shipped is not None:
                row.update(shipped_toffolis=shipped[k]["toffolis"], shipped_qubits=shipped[k]["qubits"])
            b = best.get(k)
            if b is not None:
                kp = json.loads(Path(b["plan"]).read_text())
                row.update(products=int(kp["products"]), predicted_toffolis=cache + int(kp["products"]),
                           seed=b["seed"], relations=kp.get("relations"), blocks=kp.get("blocks"),
                           planner_module=kp.get("planner", "kernel_search.shared_kernel"))
                if pad_on:
                    pr = {int(r): v for r, v in kp.get("pad_rows", {}).items()}
                    row["pad_rows_changed"] = sorted(r for r, v in pr.items() if list(map(int, delta[r])) != v)
                if out is not None:
                    row.update(plan=str(Path(b["plan"]).relative_to(out)), plan_sha256=sha256_file(b["plan"]))
                else:
                    row["terms"] = kp["terms"]
                if shipped is not None:
                    row["saving_vs_shipped"] = row["shipped_toffolis"] - row["predicted_toffolis"]
            rows.append(row)
    finally:
        if tmp is not None:
            tmp.cleanup()
    per_pass = None
    if shipped is not None:
        sh = sum(r["shipped_toffolis"] for r in rows)
        kept = sum(min(r["shipped_toffolis"], r.get("predicted_toffolis", r["shipped_toffolis"])) for r in rows)
        per_pass = dict(shipped=sh, best_of=kept, saving=sh - kept, slots=todo)
    rep = dict(tool=f"dfthc_refit {__version__}", mode="plan", planner=planner, family_dir=str(fam),
               out_dir=None if out is None else str(out), words=dict(wsrc, shape=list(words.shape)),
               seeds=seeds, pad_rows=dict(requested=bool(pad_rows), active=pad_on, real_rows=A, addressed_rows=pad_end),
               exchange=bool(exchange), gauge_free_sources=bool(gauge_free_sources),
               settings=None if planner == "kernel" else
               dict(PLAN_DEFAULTS[planner], **(planner_settings or {})), blocks=blocks_rep,
               modules=dict(kernel_search=None if kernel_src is None else
                            dict(path=str(Path(kernel_src) / "kernel_search.py"),
                                 sha256=sha256_file(Path(kernel_src) / "kernel_search.py")),
                            planner_shared=_module_record("planner_shared"),
                            planner_block=_module_record("planner_block")),
               slots=rows, toffolis_per_pass=per_pass, shipped_available=shipped is not None,
               tier_note="static model / projection: plans are not emitted or natively checked here; predicted "
                         "Toffolis = the table's cache Toffolis + products. `replan --backend` emits, checks and "
                         "keeps a slot only on a strict gain (positive-gain acceptance).",
               before=before, wall_s=round(time.time() - t0, 1))
    rep = _jsonable(rep)
    if out is not None:
        (out / "plans.json").write_text(json.dumps(rep, indent=1) + "\n")
    if verbose:
        _print_table(rows, [("k", "k"), ("coded", lambda r: len(r["coded_words"])), ("shipped", "shipped_toffolis"),
                            ("products", "products"), ("predicted", "predicted_toffolis"),
                            ("saving", "saving_vs_shipped"), ("seed", "seed")])
        if per_pass:
            print(json.dumps(dict(toffolis_per_pass=per_pass)))
    return rep


def plan_main(argv):
    ap = argparse.ArgumentParser(
        prog="dfthc_refit.py plan",
        description="Plan every ANGLE_k loader of a family with one AND-count-minimisation planner (relations: "
                    "linear relations among ANF rows; products paired two per Toffoli) and report products and "
                    "predicted Toffolis per slot against the shipped loaders. Nothing is emitted: a static model / "
                    "projection. Use `replan --backend ...` to emit, check and keep strict gains.")
    ap.add_argument("family_dir", type=Path, help="angle_k_*.kmb/.kmx (+ submission.npz unless --words is given)")
    ap.add_argument("--planner", choices=["kernel", "shared", "block"], required=True,
                    help="kernel: qwertykeypadapple's kernel-relation search (--kernel-src); shared: planner_shared "
                         "(structural annihilator + shared products); block: planner_block on a block partition "
                         "(generic loaders; coded slots use planner_shared)")
    ap.add_argument("--out", type=Path, help="write OUT/plans.json and every plan under OUT/work/ (new or empty)")
    ap.add_argument("--words", type=Path, help="the words the loaders load (.npy); default: recovered with the "
                                               "official evaluate_circuits (kickmix build)")
    ap.add_argument("--lera-src", default=os.environ.get("DFTHC_REFIT_LERA_SRC"),
                    help="directory holding Lera's tensorqrom.py (kernel planner; or $DFTHC_REFIT_LERA_SRC)")
    ap.add_argument("--kernel-src", default=os.environ.get("DFTHC_REFIT_KERNEL_SRC"),
                    help="directory holding qwertykeypadapple's kernel_search.py (or $DFTHC_REFIT_KERNEL_SRC)")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--slots", help="slots to plan, e.g. 0-3,7 (default all)")
    ap.add_argument("--procs", type=int, default=2, help="parallel planner processes (default 2)")
    ap.add_argument("--pad-rows", action="store_true",
                    help="treat the never-read padded rows as external input don't-cares (the plan rewrites them)")
    ap.add_argument("--no-pad-screen", action="store_true", help="kernel with --pad-rows: skip the word-group screen")
    _shared_args(ap)
    ap.add_argument("--gauge-free-sources", action="store_true",
                    help="v1.8.0-dev: predict pi-gauge slots' cache Toffolis with the free-source emitter "
                         "(incoming gauge monomials as low sources), as `replan --gauge-free-sources` emits them")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.planner == "kernel" and (a.exchange or a.blocks or a.partitions is not None or a.keep is not None
                                  or a.cpu_cap is not None or a.ex_cap is not None):
        ap.error("--blocks/--exchange/--partitions/--keep/--cpu-cap/--ex-cap need --planner shared or block")
    kw = _shared_kwargs(a)
    rep = plan_family(a.family_dir, a.out, planner=a.planner, words=a.words, lera_src=a.lera_src,
                      kernel_src=a.kernel_src, verifier_dir=a.verifier_dir, kickmix_dir=a.kickmix_dir, slots=a.slots,
                      seeds=kw["seeds"], pad_rows=a.pad_rows, pad_screen=not a.no_pad_screen, exchange=kw["exchange"],
                      blocks=kw["blocks"], procs=a.procs, planner_settings=kw["planner_settings"],
                      verbose=not a.quiet, gauge_free_sources=a.gauge_free_sources)
    return rep


# ------------------------------------------------------------------------------------------------------- recover


def recover_family(family_dir, out_dir=None, *, words=None, verifier_dir=None, kickmix_dir=None, slots=None,
                   emit_check=False, lera_src=None, verbose=True) -> dict:
    """Structure census of every shipped loader (recover.py): terms read back from the circuit, function check
    against the table delta, feature space and relations, low-monomial and output partitions, pairing, products
    shared between coded and non-coded outputs. With `emit_check`, each loader is rebuilt from its recovered terms
    with Lera's unmodified emitter and natively checked. Writes OUT/recover.json, OUT/words.npy and OUT/family.json
    (the input of `price-words --recover-dir`) when `out_dir` is given."""
    import tempfile
    rc = _local("recover")
    t0 = time.time()
    fam = Path(family_dir)
    try:
        vdir = find_verifier_dir(verifier_dir)
    except FileNotFoundError:
        if words is None:
            raise
        vdir = None
    km = find_kickmix(kickmix_dir, vdir)
    if km is None:
        raise RuntimeError("recover needs a kickmix build (it reads the loaders): pass --kickmix-dir")
    tq = tq_sha = None
    if emit_check:
        if lera_src is None:
            raise FileNotFoundError("--emit-check needs Lera's tensorqrom.py: pass --lera-src")
        tq, tq_sha = _load_tensorqrom(lera_src)
    out = _new_out(out_dir)
    loaders = loader_files(fam)
    if not loaders or [int(p.stem.split("_")[-1]) for p in loaders] != list(range(len(loaders))):
        raise RuntimeError(f"{fam} must hold one angle_k_*.kmb/.kmx per slot 0..n-1")
    words, wsrc, before = _family_words(fam, words, vdir, kickmix_dir)
    todo = _parse_slots(slots, len(loaders))
    conv = family_conventions(fam, words, km)          # v1.5: loader classes first; chained slots are not read
    cls = {s["k"]: s for s in conv.get("slots", [])}
    gz = _local("gauge")
    gauged = conv["gauge"] == "pi" and conv["pattern_low_bits"]
    tmp = tempfile.TemporaryDirectory() if (emit_check and out is None) else None
    emit_root = (Path(tmp.name) if tmp is not None else out / "emit") if emit_check else None
    rows, full = [], {}
    try:
        for k in todo:
            p = loaders[k]
            if k in cls and cls[k]["loader"] != "xor-in":
                rows.append(dict(k=k, file=p.name, sha256=sha256_file(p), loader=cls[k]["loader"],
                                 toffolis=cls[k]["toffolis"], qubits=cls[k]["qubits"],
                                 error=f"skipped: {cls[k]['loader']} loader (its action depends on the incoming "
                                       f"words; recover reads XOR-in cached-product loaders only)"))
                continue
            try:
                gin = gz.slot_gauge(words, k, LERA_J)[0] if gauged else None
                report, rec, rec_x, delta = rc.recover_loader(km, p, words, k, J=LERA_J, gauge_in=gin)
            except Exception as ex:  # noqa: BLE001  (not a cached-product loader, or a mismatch: recorded)
                rows.append(dict(k=k, file=p.name, sha256=sha256_file(p), error=f"{type(ex).__name__}: {ex}"[:400]))
                continue
            st = report["structure"]
            row = dict(k=k, file=p.name, sha256=report["kmb_sha256"], toffolis=report["num_toffoli_like"],
                       qubits=report["num_qubits"], caches=report["caches"], products=report["products"],
                       coded_words=report["coded_words"], function_ok=report["recovered_equals_delta_on_rows"],
                       selection_frame=bool(report["selection_frame"]),
                       padded_rows_differing=len(report["padded_rows_differing_from_default_pad"]),
                       low_partition=st["low_partition"], relations=st["relations"], dimW=st["dimW"],
                       rows_in_W=st["rows_in_W"], lower_bound_pairs=st["lower_bound_pairs"], pairing=st["pairing"],
                       output_sharing=st["output_sharing"], distinct_low_forms=st["distinct_low_forms"],
                       max_low_form_weight=st["max_low_form_weight"])
            report["terms"] = [list(map(int, t)) for t in rec["terms"]]
            if emit_check:
                ed = emit_root / f"k{k}"
                ed.mkdir(parents=True, exist_ok=False)
                try:
                    if gauged:   # v1.5: the gauge-aware emitter; v1.8: partial low caches with free sources
                        partial = report["low_caches"] < (1 << LERA_LOWBITS) - 1 - LERA_LOWBITS
                        re_ = rc.reemit(km, gz.GaugeEmitter(tq, *gz.slot_gauge(words, k, LERA_J), free_sources=partial),
                                        rec, rec_x, delta, ed / f"angle_k_{k}.kmb",
                                        check_inputs=gz.check_inputs(words, k))
                    else:
                        re_ = rc.reemit(km, tq, rec, rec_x, delta, ed / f"angle_k_{k}.kmb")
                    re_.update(tensorqrom_sha256=tq_sha, toffolis_equal_shipped=re_["num_toffoli_like"] == row["toffolis"],
                               byte_identical_shipped=p.suffix == ".kmb" and re_["kmb_sha256"] == row["sha256"])
                except Exception as ex:  # noqa: BLE001
                    re_ = dict(error=f"{type(ex).__name__}: {ex}"[:400])
                report["reemit"] = re_
                row["reemit"] = dict(status=re_.get("native_check", {}).get("status", "ERROR"),
                                     toffolis=re_.get("num_toffoli_like"), qubits=re_.get("num_qubits"),
                                     toffolis_equal_shipped=re_.get("toffolis_equal_shipped"),
                                     byte_identical_shipped=re_.get("byte_identical_shipped"))
            rows.append(row)
            full[str(k)] = report
    finally:
        if tmp is not None:
            tmp.cleanup()
    parts = {}
    for r in rows:
        if "low_partition" in r:
            key = json.dumps(sorted(sorted(b) for b in r["low_partition"]))
            parts.setdefault(key, []).append(r["k"])
    ok = [r for r in rows if "error" not in r]
    family = dict(slots=todo, loaders=len(loaders), recovered=len(ok), errors=len(rows) - len(ok),
                  coded_slots=[r["k"] for r in ok if r["coded_words"]],
                  generic_slots=[r["k"] for r in ok if not r["coded_words"]],
                  function_ok_all=all(r["function_ok"] for r in ok) if ok else None,
                  toffolis_per_pass=sum(r["toffolis"] for r in ok), products=sum(r["products"] for r in ok),
                  low_partitions=[dict(blocks=json.loads(k), slots=v)
                                  for k, v in sorted(parts.items(), key=lambda kv: (-len(kv[1]), kv[0]))],
                  mixed_products=sum(r["output_sharing"]["mixed"] for r in ok))
    rep = dict(tool=f"dfthc_refit {__version__}", mode="recover", family_dir=str(fam),
               out_dir=None if out is None else str(out), words=dict(wsrc, shape=list(words.shape)), family=family,
               slots=rows, recover_module=_module_record("recover"), conventions=conv,
               emitter=None if not emit_check else dict(module="tensorqrom.build_circuit (Lera, unmodified)",
                                                        path=str(Path(lera_src) / "tensorqrom.py"), sha256=tq_sha),
               before=before, wall_s=round(time.time() - t0, 1))
    rep = _jsonable(rep)
    if out is not None:
        (out / "recover.json").write_text(json.dumps(dict(rep, per_slot=_jsonable(full)), indent=1) + "\n")
        with open(out / "words.npy", "xb") as f:
            np.save(f, words.astype(np.int64))
        fj = dict(family_dir=str(fam), words_npy_sha256=sha256_file(out / "words.npy"), words=rep["words"],
                  inputs_sha256={p.name: sha256_file(p) for p in loaders + ([fam / "submission.npz"]
                                                                            if (fam / "submission.npz").is_file()
                                                                            else [])},
                  before=before)
        (out / "family.json").write_text(json.dumps(_jsonable(fj), indent=1) + "\n")
    if verbose:
        _print_table(rows, [("k", "k"), ("Toffolis", "toffolis"), ("qubits", "qubits"), ("products", "products"),
                            ("coded", lambda r: len(r["coded_words"]) if "coded_words" in r else None),
                            ("relations", "relations"), ("pairs>=", "lower_bound_pairs"),
                            ("cross", lambda r: r["pairing"]["cross_block"] if "pairing" in r else None),
                            ("singles", lambda r: r["pairing"]["single"] if "pairing" in r else None),
                            ("mixed", lambda r: r["output_sharing"]["mixed"] if "output_sharing" in r else None),
                            ("fn", "function_ok"),
                            ("reemit", lambda r: r["reemit"]["status"] if "reemit" in r else None),
                            ("partition", lambda r: "/".join(",".join(map(str, b)) for b in r["low_partition"])
                             if "low_partition" in r else r.get("error"))])
    return rep


def recover_main(argv):
    ap = argparse.ArgumentParser(
        prog="dfthc_refit.py recover",
        description="Structure census of a family's shipped ANGLE_k loaders (cached-product style of Lera's "
                    "emitter): products read back from the circuit, function check, feature space and linear "
                    "relations among ANF rows, low-monomial and output partitions, pairing, products shared "
                    "between coded and non-coded outputs. Optionally re-emit and check natively.")
    ap.add_argument("family_dir", type=Path, help="angle_k_*.kmb/.kmx (+ submission.npz unless --words is given)")
    ap.add_argument("--out", type=Path, help="write OUT/recover.json, OUT/words.npy, OUT/family.json (and "
                                             "OUT/emit/ with --emit-check); new or empty")
    ap.add_argument("--words", type=Path, help="the words the loaders load (.npy); default: recovered with the "
                                               "official evaluate_circuits")
    ap.add_argument("--slots", help="slots to recover, e.g. 0-3,7 (default all)")
    ap.add_argument("--emit-check", action="store_true",
                    help="rebuild each loader from its recovered terms with Lera's emitter and run the native check")
    ap.add_argument("--lera-src", default=os.environ.get("DFTHC_REFIT_LERA_SRC"),
                    help="directory holding Lera's tensorqrom.py (for --emit-check; or $DFTHC_REFIT_LERA_SRC)")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.emit_check and not a.lera_src:
        ap.error("--emit-check needs --lera-src")
    return recover_family(a.family_dir, a.out, words=a.words, verifier_dir=a.verifier_dir, kickmix_dir=a.kickmix_dir,
                          slots=a.slots, emit_check=a.emit_check, lera_src=a.lera_src, verbose=not a.quiet)


# --------------------------------------------------------------------------------------------------- price-words


def load_moves(path, pw):
    """Moves from JSON: a list of {name, changes: [[row, column, new_word], ...], meta?}, or {"moves": [...]}, or
    {name: changes}."""
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict) and "moves" in d:
        d = d["moves"]
    if isinstance(d, dict):
        d = [dict(name=k, changes=v) for k, v in d.items()]
    return [pw.Move(str(m.get("name", f"m{i}")), tuple((int(r), int(c), int(w)) for r, c, w in m["changes"]),
                    dict(m.get("meta", {}))) for i, m in enumerate(d)]


def rank_drop_candidates(fam, pw, k, *, drops=2, row_prices=True, max_terms=3, keep=8, restarts=0, protect=()):
    """Bit-0 toggle moves lowering loader k's coded mixed rank by `drops` (1 or 2; two successive single drops,
    the second keeping the first's dependency, as in the original word-move pricing study). With `row_prices`, toggles are chosen
    by the per-entry first-order price (RowPricer, no credit for negative prices); else every row costs 1."""
    words = fam.words
    _, cw = loader_delta(words, k)
    if not cw:
        return []
    price = None
    if row_prices:
        rp = pw.RowPricer(fam)
        cols = {LERA_J * k + w for w in cw} | ({LERA_J * (k - 1) + w for w in cw} if k > 0 else set())
        price = {}
        for c in sorted(x for x in cols if x < words.shape[1]):
            p, _ = rp.prices(c, words[:, c] ^ 1)
            price[c] = np.maximum(np.asarray(p, np.float64), 0.0)
    m1 = pw.rank_drop_moves(fam, k, row_price=price, max_terms=max_terms, keep=keep, restarts=restarts,
                            protect=tuple(protect))
    if drops == 1:
        return m1
    moves = []
    for mv1 in m1:
        w1 = pw.apply_move(words, mv1, fam.brot)
        pr1 = None
        if price is not None:
            pr1 = {c: v.copy() for c, v in price.items()}
            for r, c, _ in mv1.changes:
                if c in pr1:
                    pr1[c][r] = -price[c][r]          # toggling back undoes the change (first order)
        m2 = pw.rank_drop_moves(fam, k, row_price=pr1, max_terms=max_terms, keep=1, words=w1, restarts=restarts,
                                seed=1, protect=tuple(protect), frozen_terms=mv1.meta["x"])
        if not m2:
            continue
        ch = {(r, c): v for r, c, v in mv1.changes}
        for r, c, v in m2[0].changes:
            if (r, c) in ch:
                del ch[(r, c)]                        # toggled twice: back to the original word
            else:
                ch[(r, c)] = v
        moves.append(pw.Move(f"k{k}_d2_{mv1.name}_{m2[0].name}", tuple((r, c, v) for (r, c), v in sorted(ch.items())),
                             dict(loader=k, first=mv1.meta, second=m2[0].meta)))
    return moves


def price_words_family(family_dir, out_dir=None, *, words=None, recover_dir=None, qubits=None, toffolis=None,
                       verifier_dir=None, kickmix_dir=None, moves=None, rank_drop=None, drops=2,
                       refit_method="retract", max_evals=300, refit_top=10, keep=8, max_terms=3, restarts=0,
                       row_prices=True, protect=(), threads=2, verbose=True) -> dict:
    """Shadow prices (KKT multipliers of error_1, error_2 and the E_gap margin, in score units) and coded mixed
    ranks of a family; with `moves` (a JSON file) or `rank_drop` (a loader), a priced-move table: per move the
    loader saving from the static coded-rank planner, the accuracy price (first order; bounded re-refit for the
    `refit_top` best) and net = saving - price, best first. Approximate logic synthesis of the angle table: a word
    move spends accuracy budget for AND count. Needs PyTorch (load_family)."""
    pw = _local("price_words")
    t0 = time.time()
    fam_dir = Path(family_dir)
    out = _new_out(out_dir)
    wsrc, res_before = None, None
    if recover_dir is not None:
        rd = Path(recover_dir)
        fj = json.loads((rd / "family.json").read_text())
        if sha256_file(rd / "words.npy") != fj["words_npy_sha256"]:
            raise RuntimeError(f"{rd}/words.npy does not match family.json")
        words = np.load(rd / "words.npy", allow_pickle=False).astype(np.int64)
        wsrc = dict(source=f"recover dir {rd}", sha256=fj["words_npy_sha256"], recovered=fj.get("words"))
        res_before = (fj.get("before") or {}).get("resources")
    elif words is not None:
        words, wsrc, _ = _family_words(fam_dir, words, verifier_dir, kickmix_dir)
    if qubits is None or toffolis is None or words is None:
        if res_before is None:
            w2, ws2, before = _family_words(fam_dir, None, verifier_dir, kickmix_dir)
            if words is None:
                words, wsrc = w2, ws2
            res_before = before["resources"]
        qubits = int(res_before["qubits"]) if qubits is None else int(qubits)
        toffolis = float(res_before["toffoli"]) if toffolis is None else float(toffolis)
    vdir = find_verifier_dir(verifier_dir)
    try:
        import torch  # noqa: F401
    except ImportError as ex:
        raise RuntimeError("price-words needs PyTorch (the kickmix venv has none): run `recover --out DIR` in the "
                           "kickmix interpreter, then `price-words --recover-dir DIR` with a NumPy/SciPy/PyTorch "
                           "interpreter") from ex
    gz = _local("gauge")
    conv = family_conventions(fam_dir, words)               # v1.5: word conventions (no loader simulation here)
    why = gz.refusal("price-words", conv, moves=moves is not None or rank_drop is not None)
    if why:
        raise SystemExit(f"REFUSED {why}")
    working = None
    if conv["gauge"] == "pi":
        # exact symmetry: price on the canonical signed form; moves are translated in and out
        pattern = gz.off_gauge(words, int(conv["brot"]))
        words_family = words
        words = gz.canonical_words(words, int(conv["brot"]))
        working = dict(form="canonical signed gauge (pi gauge is an exact symmetry: u u^T unchanged on every row)",
                       off_gauge_entries=int(pattern.sum()), max_projector_change=conv["max_projector_change"])
    fam = pw.load_family(fam_dir, vdir, words=words, qubits=qubits, toffolis=toffolis, threads=threads)
    if working is not None:
        Pf = femoco_problem(fam.npz, words_family, fam.brot, vdir)
        mf = Pf.official(Pf.y0())
        working["metrics_family_gauge"] = _metrics_row(mf)
        working["metrics_working_form"] = _metrics_row(fam.base)
        working["lambda_eff_rel_diff"] = float((fam.base["lambda_eff"] - mf["lambda_eff"]) / mf["lambda_eff"])
    rep = dict(tool=f"dfthc_refit {__version__}", mode="price-words", family_dir=str(fam_dir),
               out_dir=None if out is None else str(out), words=dict(wsrc or {}, shape=list(words.shape)),
               conventions=conv, working_form=working,
               qubits=qubits, toffolis=toffolis, base=_metrics_row(fam.base), shadow=pw.shadow_prices(fam),
               coded_ranks={str(k): dict(coded_words=v[0], rank=v[1])
                            for k, v in pw.loader_coded_ranks(fam.words, pad_free=True).items()},
               price_words_module=_module_record("price_words"),
               tier_note="static model / projection: savings are the coded-rank planner's lower-bound model "
                         "(realised only when a loader is re-planned and emitted); prices are first order or a "
                         "bounded re-refit of the official dfthc_math metrics")
    mv = None
    if moves is not None:
        mv = load_moves(moves, pw)
        if working is not None:   # family-gauge words -> working form
            mv = [pw.Move(m.name, tuple((r, c, int(gz.canonical_words(np.array([[v]]), fam.brot)[0, 0]))
                                        for r, c, v in m.changes), m.meta) for m in mv]
        rep["moves_source"] = dict(path=str(moves), sha256=sha256_file(moves), n=len(mv))
    elif rank_drop is not None:
        mv = rank_drop_candidates(fam, pw, int(rank_drop), drops=drops, row_prices=row_prices, max_terms=max_terms,
                                  keep=keep, restarts=restarts, protect=protect)
        rep["moves_source"] = dict(rank_drop=int(rank_drop), drops=drops, row_prices=row_prices, max_terms=max_terms,
                                   keep=keep, restarts=restarts, protect=list(protect), n=len(mv))
    if mv is not None:
        rows = pw.rank_moves(fam, mv, pw.coded_rank_planner(), refit_top=refit_top, max_evals=max_evals,
                             method=refit_method)
        rep.update(refit=dict(method=refit_method, max_evals=max_evals, refit_top=refit_top), moves=rows)
        if working is not None:   # the moves in the family's own gauge (pattern re-applied)
            rep["moves_family_gauge"] = {m.name: [[int(r), int(c), int(v ^ ((1 << (fam.brot - 1)) * pattern[r, c]))]
                                                  for r, c, v in m.changes] for m in mv}
    rep["wall_s"] = round(time.time() - t0, 1)
    rep = _jsonable(rep)
    if out is not None:
        (out / "price_words.json").write_text(json.dumps(rep, indent=1) + "\n")
    if verbose:
        sp = rep["shadow"]
        print(json.dumps({k: sp[k] for k in ("cost", "lambda_eff", "units_per_uHa_e2", "units_per_uHa_e1",
                                               "units_per_uHa_gap", "loader_toffoli_units",
                                               "uHa_e2_per_loader_toffoli", "wrong_sign")}))
        if mv is not None:
            _print_table(rep["moves"], [("move", "move"), ("changes", "n_changes"),
                                        ("saving", "saving_toffolis_per_pass"), ("price", "price_units"),
                                        ("tier", "price_tier"), ("net", "net_units")])
    return rep


def price_words_main(argv):
    ap = argparse.ArgumentParser(
        prog="dfthc_refit.py price-words",
        description="Price word moves (edits of the angle-word table) for a family: approximate logic synthesis of "
                    "the angle table under the official accuracy constraints. Prints the shadow prices (KKT "
                    "multipliers, score units per uHa) and, for --moves or --rank-drop, a table of loader saving "
                    "(AND count, Toffolis per pass), accuracy price and net. Needs PyTorch.")
    ap.add_argument("family_dir", type=Path, help="the dfthcsos directory (submission.npz + loaders)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--moves", type=Path, help="JSON moves: [{name, changes: [[row, column, new_word], ...]}]")
    g.add_argument("--rank-drop", type=int, metavar="K",
                   help="build bit-0 toggle moves that lower loader K's coded mixed rank (soft-word idea of "
                        "\"i want to run gemma 9b\")")
    ap.add_argument("--drops", type=int, choices=(1, 2), default=2, help="rank drops per move (default 2: one "
                                                                         "product needs rank 20 -> 18)")
    ap.add_argument("--refit-method", choices=("retract", "penalty", "ladder"), default="retract",
                    help="bounded re-refit: retract (dfthc_refit.refit), penalty (L-BFGS-B, robust where the "
                         "retraction fails), ladder (penalty continuation)")
    ap.add_argument("--max-evals", type=int, default=300, help="metric evaluations per bounded re-refit")
    ap.add_argument("--refit-top", type=int, default=10, help="moves (best first-order net) that get a re-refit")
    ap.add_argument("--keep", type=int, default=8, help="rank-drop candidates kept per drop")
    ap.add_argument("--max-terms", type=int, default=3, help="rank-drop slope combinations up to this size")
    ap.add_argument("--restarts", type=int, default=0, help="rank-drop local-search restarts")
    ap.add_argument("--unit-row-prices", action="store_true",
                    help="rank-drop: every toggled row costs 1 (skip the first-order per-entry prices)")
    ap.add_argument("--protect", default="", help="loaders whose columns rank-drop moves must not touch, e.g. 3,5")
    ap.add_argument("--out", type=Path, help="write OUT/price_words.json (new or empty)")
    ap.add_argument("--recover-dir", type=Path, help="a `recover --out` directory: words.npy + family.json "
                                                     "(words and official resources, SHA-256 checked)")
    ap.add_argument("--words", type=Path, help="the words the loaders load (.npy)")
    ap.add_argument("--qubits", type=int, help="official Q_total (default: from --recover-dir or evaluate_circuits)")
    ap.add_argument("--toffolis", type=float, help="official T_total (default: as --qubits)")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.recover_dir and a.words:
        ap.error("--recover-dir and --words are alternatives")
    protect = tuple(int(x) for x in a.protect.split(",") if x.strip())
    return price_words_family(a.family_dir, a.out, words=a.words, recover_dir=a.recover_dir, qubits=a.qubits,
                              toffolis=a.toffolis, verifier_dir=a.verifier_dir, kickmix_dir=a.kickmix_dir,
                              moves=a.moves, rank_drop=a.rank_drop, drops=a.drops, refit_method=a.refit_method,
                              max_evals=a.max_evals, refit_top=a.refit_top, keep=a.keep, max_terms=a.max_terms,
                              restarts=a.restarts, row_prices=not a.unit_row_prices, protect=protect,
                              threads=a.threads, verbose=not a.quiet)


# --------------------------------------------------------------------------------------------------- chained-and


def _load_circuit_json(path, ca):
    d = json.loads(Path(path).read_text())
    for key in ("circuit", "rank2_construction"):
        if isinstance(d, dict) and key in d and "gates" not in d:
            d = d[key]
    if "circuit" in d and "gates" not in d:
        d = d["circuit"]
    return ca.Circuit.from_dict(d)


def chained_and_main(argv):
    """Thin wrapper over chained_and.py (AND-count and AND-depth questions for a multi-output GF(2) block)."""
    ap = argparse.ArgumentParser(
        prog="dfthc_refit.py chained-and",
        description="Multi-output GF(2) blocks under the cached-monomial model (chained_and.py): lower bounds on "
                    "the AND count (rank of the alternating matrix for depth-1 products; output dimension for any "
                    "AND-depth), an exhaustive verifier, and CNF encodings for a SAT solver.")
    ap.add_argument("target", type=Path, help="target JSON (ybits, outputs, rows[{care, tt}]; one per "
                                              "loader block)")
    sub = ap.add_subparsers(dest="action", required=True)
    b = sub.add_parser("bounds", help="rank-2 (depth-1) and general AND-count lower bounds + a verified rank-2 "
                                      "construction")
    b.add_argument("--depth1-exact", type=float, metavar="SECONDS",
                   help="also decide the tight depth-1 case exactly (pure-tensor DFS, time limit)")
    b.add_argument("--out", type=Path, help="write the JSON report here (must not exist)")
    v = sub.add_parser("verify", help="verify a circuit on every row and care point (exhaustive)")
    v.add_argument("--circuit", type=Path, required=True,
                   help="circuit JSON (Circuit.to_dict, or a report holding 'circuit' / 'rank2_construction')")
    v.add_argument("--out", type=Path, help="write the JSON report here (must not exist)")
    e = sub.add_parser("encode", help="write the DIMACS CNF for 'k ANDs meet these rows' (no solver is run)")
    e.add_argument("--k", type=int, required=True, help="AND count")
    e.add_argument("--depth", type=int, help="AND-depth cap (1 = rank-2 products only); plain encode_cnf")
    e.add_argument("--chained", type=int, help="R6 shape: the last CHAINED gates carry chain variables "
                                               "(encode_shape with gauge and lex symmetry breaking)")
    e.add_argument("--rows", choices=("all", "spanning"), default="all",
                   help="rows to encode: all, or a basis of the mixed row space plus partial rows (CEGAR start)")
    e.add_argument("--cnf", type=Path, required=True, help="DIMACS output path (must not exist)")
    a = ap.parse_args(argv)
    if getattr(a, "out", None) is not None and a.out.exists():
        ap.error(f"refusing to overwrite {a.out}")
    ca = _local("chained_and")
    T = ca.Target.from_json(a.target)
    rep = dict(tool=f"dfthc_refit {__version__}", chained_and=ca.__version__, target=str(a.target),
               target_sha256=sha256_file(a.target), action=a.action, ybits=T.n, outputs=T.m, rows=len(T.rows))
    if a.action == "bounds":
        circ, rb = ca.construct_rank2(T)
        ver = ca.verify(circ, T, rowbits=None if any(x is None for x in rb) else rb)
        if not ver["all_pass"]:
            ver = ca.verify(circ, T)
        rep.update(rank2=ca.rank2_lower_bound(T), general={k: x for k, x in ca.general_lower_bound(T).items()
                                                           if k != "combo_ranks"},
                   mixed_rank=ca.rank(ca.mixed_matrix(T)),
                   rank2_construction=dict(ands=circ.k, verify={k: x for k, x in ver.items() if k != "rowbits"},
                                           circuit=circ.to_dict()))
        if a.depth1_exact is not None:
            d1 = ca.depth1_exact(T, time_limit_s=a.depth1_exact)
            rep["depth1_exact"] = {k: (x.to_dict() if hasattr(x, "to_dict") else x) for k, x in d1.items()}
        print(json.dumps(dict(rank2_bound=rep["rank2"]["bound"], general_bound=rep["general"]["bound"],
                              mixed_rank=rep["mixed_rank"], rank2_construction_ands=circ.k,
                              verified=ver["all_pass"],
                              depth1_exact=rep.get("depth1_exact", {}).get("status"))))
    elif a.action == "verify":
        circ = _load_circuit_json(a.circuit, ca)
        ver = ca.verify(circ, T)
        rep.update(circuit=str(a.circuit), circuit_sha256=sha256_file(a.circuit),
                   verify={k: x for k, x in ver.items() if k != "rowbits"})
        print(json.dumps(dict(all_pass=ver["all_pass"], rows_met=ver["rows_met"], rows=ver["rows"], ands=ver["ands"],
                              depth=ver["depth"])))
    else:
        if a.depth is not None and a.chained is not None:
            ap.error("--depth and --chained are alternatives")
        if a.cnf.exists():
            ap.error(f"refusing to overwrite {a.cnf}")
        rows = ca.spanning_rows(T) if a.rows == "spanning" else None
        if a.chained is not None:
            cnf = ca.encode_shape(T, a.k, a.chained, rows=rows)
        else:
            cnf = ca.encode_cnf(T, a.k, depth=a.depth, rows=rows)
        a.cnf.write_text(cnf.dimacs())
        rep.update(cnf=dict(path=str(a.cnf), sha256=sha256_file(a.cnf), k=a.k, depth=a.depth, chained=a.chained,
                            rows=a.rows, n_rows=len(rows) if rows is not None else len(T.rows), vars=cnf.nv,
                            clauses=len(cnf.clauses)))
        print(json.dumps(rep["cnf"]))
    out = getattr(a, "out", None)
    if out is not None:
        out.write_text(json.dumps(_jsonable(rep), indent=1) + "\n")
    return _jsonable(rep)


def lowcache_family(family_dir, out=None, *, words=None, verifier_dir=None, kickmix_dir=None, loaders=True,
                    verbose=True) -> dict:
    """`lowcache` subcommand (v1.8.0-dev; lowcache.py): per slot, the free gauge sources (incoming top bits
    of words whose gauge does not change), the low-cache floor and whether it is proved (rank test on the real
    rows), the shipped loader's low caches and the gap; the columns that lose a gauge pattern and whether any pure
    pi-gauge symmetry can touch them. Exact (GF(2) algebra; recovered loaders)."""
    t0 = time.time()
    lc = _local("lowcache")
    rc = _local("recover")
    gz = _local("gauge")
    fam = Path(family_dir)
    words, wsrc, before = _family_words(fam, words, verifier_dir, kickmix_dir)
    n = -(-words.shape[1] // LERA_J)
    files = loader_files(fam) if fam.is_dir() else []
    km = _optional_kickmix(kickmix_dir, verifier_dir) if loaders else None
    conv = family_conventions(fam, words, km if len(files) == n else None)
    gauged = conv["gauge"] == "pi" and bool(conv["pattern_low_bits"])
    deltas = {k: full_block_delta(loader_delta(words, k)[0]) for k in range(n)}
    shipped_low, errors = None, {}
    if km is not None and len(files) == n:
        shipped_low = {}
        for k in range(n):
            try:
                gin = gz.slot_gauge(words, k, LERA_J)[0] if gauged else None
                rep_k, *_ = rc.recover_loader(km, files[k], words, k, J=LERA_J, gauge_in=gin)
                shipped_low[k] = rep_k["low_caches"]
            except Exception as ex:  # noqa: BLE001  (not a cached-product loader: recorded)
                errors[k] = f"{type(ex).__name__}: {ex}"[:300]
    edges = None
    npz = fam / "submission.npz"
    if npz.is_file():
        z = np.load(npz)
        if "tree_edges" in z.files:
            edges = np.asarray(z["tree_edges"], np.int64)
    rep = lc.family_lowcache(words, deltas, edges=edges if gauged else None, shipped_low=shipped_low, j=LERA_J)
    rep = _jsonable(dict(tool=f"dfthc_refit {__version__}", mode="lowcache", family_dir=str(fam),
                         words=dict(wsrc, shape=list(words.shape)), gauge=conv["gauge"], recover_errors=errors,
                         tier="exact (GF(2) algebra on the table and the recovered loaders)",
                         module=_module_record("lowcache"), **rep, before=before, wall_s=round(time.time() - t0, 1)))
    if out is not None:
        out = Path(out)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(json.dumps(rep, indent=1) + "\n")
    if verbose:
        _print_table(rep["slots"], [("k", "k"), ("free", lambda r: len(r["free_monomials"])), ("floor", "floor"),
                                    ("proved", "floor_proved"), ("shipped", "shipped_low_caches"), ("gap", "gap")])
        print(json.dumps({x: rep.get(x) for x in ("floor_total", "gap_total", "changing_columns_untouchable",
                                                  "plain_columns_untouchable")}))
    return rep


def lowcache_main(argv):
    ap = argparse.ArgumentParser(
        prog="dfthc_refit.py lowcache",
        description="Low-cache floor per loader (v1.8.0-dev): free gauge sources, the floor and its proof "
                    "(rank of the mixed ANF rows on real addresses), shipped low caches and the gap, and which "
                    "gauge-losing columns a pure pi-gauge symmetry could touch. Read only; exact.")
    ap.add_argument("family_dir", type=Path)
    ap.add_argument("--out", type=Path, help="write the report here (a new .json file)")
    ap.add_argument("--words", type=Path, help="the words the loaders load (.npy); default: recovered (kickmix)")
    ap.add_argument("--no-loaders", action="store_true", help="table only (no shipped low-cache counts)")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--kickmix-dir")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    return lowcache_family(a.family_dir, a.out, words=a.words, verifier_dir=a.verifier_dir,
                           kickmix_dir=a.kickmix_dir, loaders=not a.no_loaders, verbose=not a.quiet)


SUBCOMMANDS = {"replan": replan_main, "plan": plan_main, "recover": recover_main, "price-words": price_words_main,
               "chained-and": chained_and_main, "conventions": conventions_main, "gauge-options": gauge_options_main,
               "lowcache": lowcache_main}


# ---------------------------------------------------------------------------------------------------------- run


def run(submission_dir, out_dir, *, verifier_dir=None, kickmix_dir=None, arm="B", rounds=10, max_evals=1000,
        e1t=None, e2t=None, gap_margin=1e-5, threads=3, seed=0, verify_after=True, verbose=True,
        min_gain=1e-7, patience=3, kick_rounds=1, gauge_procs=None, brot=None) -> dict:
    """Refit SUBMISSION_DIR (15 angle_k_* loaders + submission.npz) into OUT_DIR/dfthcsos/ and write
    OUT_DIR/report.json. Returns the report. e1t / e2t None: the defaults ('free' / 19.999 mHa) with the automatic
    holds at the limits; any value given is an explicit target and is not held (v1.7.0)."""
    import torch
    t0 = time.time()
    torch.set_num_threads(int(threads))
    sub, out = Path(submission_dir), Path(out_dir)
    vdir = find_verifier_dir(verifier_dir)
    loaders = loader_files(sub)
    npz_in = sub / "submission.npz"
    if not loaders or not npz_in.is_file():
        raise FileNotFoundError(f"{sub} must hold angle_k_*.kmb/.kmx and submission.npz")
    dest = out / "dfthcsos"
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} exists and is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "w")

    def log(**kw):
        kw["t"] = round(time.time() - t0, 2)
        s = json.dumps(_jsonable(kw))
        logf.write(s + "\n")
        logf.flush()
        if verbose and kw.get("event") in ("words", "base", "round", "done", "verify_after"):
            print(s[:400], flush=True)

    inputs_sha = {p.name: sha256_file(p) for p in loaders + [npz_in]}
    wr = recover_words(sub, vdir, kickmix_dir, brot=brot)
    log(event="words", source=wr["source"], brot=wr["brot"], brot_source=wr["brot_source"], npz_angle_words_match=wr["npz_angle_words_match"])
    z = dict(np.load(npz_in, allow_pickle=False))
    if "brot" in z and int(z["brot"]) != wr["brot"]:
        raise RuntimeError(f"npz brot {int(z['brot'])} differs from the loaders' {wr['brot']}")
    conv = _local("gauge").word_conventions(wr["words"], z["tree_edges"], wr["brot"])   # v1.5: recorded, no refusal
    log(event="conventions", gauge=conv["gauge"], off_gauge_entries=conv["off_gauge_entries"])
    t_setup = time.time()
    P = femoco_problem(npz_in, wr["words"], wr["brot"], vdir)
    if wr["before"] is not None:
        mb = wr["before"]["metrics"]
        mo = P.official(P.y0())
        for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff"):
            if mb[k] != mo[k]:
                raise RuntimeError(f"official metric {k} differs between evaluate_circuits and the refit's evaluator")
    res = alternate(P, P.y0(), arm=arm, rounds=rounds, max_evals=max_evals, e1t=e1t, e2t=e2t,
                    gap_margin=gap_margin, seed=seed, log=log, min_gain=min_gain, patience=patience,
                    kick_rounds=kick_rounds,
                    procs=int(gauge_procs) if gauge_procs else max(int(threads), min(9, (os.cpu_count() or 2) - 1)))
    t_opt = time.time()
    if res["result"] is None:
        raise RuntimeError("no point passing the official accuracy gates was found (the input does not pass either)")
    y = res["result"]["y"]
    w, wB = P.split(y)
    new = dict(z)
    new["W"] = np.ascontiguousarray(w, dtype=np.float64)       # float64 always: no rounding of the new point
    new["H2_I"] = np.ascontiguousarray(wB, dtype=np.float64)
    if "offset_W" in new:
        new["offset_W"] = np.ascontiguousarray(wB - w.sum(1), dtype=np.float64)
    for p in loaders:
        shutil.copyfile(p, dest / p.name)
    npz_out = dest / "submission.npz"
    write_npz(npz_out, new)
    # re-read: the metrics reported are those of the bytes written
    zz = np.load(npz_out, allow_pickle=False)
    after = P.official(P.from_weights(zz["W"], zz["H2_I"]))
    outputs_sha = {p.name: sha256_file(p) for p in sorted(dest.iterdir())}
    assert all(outputs_sha[p.name] == inputs_sha[p.name] for p in loaders), "loader copy mismatch"
    before = res["base"]["official"]
    rep = dict(
        tool=f"dfthc_refit {__version__}", submission_dir=str(sub), out_dir=str(out),
        settings=dict(arm=arm, rounds=rounds, max_evals=max_evals, e1t=E1T_DEFAULT if e1t is None else e1t,
                      e2t=E2T_DEFAULT if e2t is None else e2t, e1t_explicit=e1t is not None,
                      e2t_explicit=e2t is not None, gap_margin=gap_margin,
                      threads=threads, seed=seed, targets=res["targets"]),
        words=dict(source=wr["source"], brot=wr["brot"], brot_source=wr["brot_source"], kickmix=wr["kickmix"],
                   npz_angle_words_match_loaders=wr["npz_angle_words_match"]),
        before=_metrics_row(before), after=_metrics_row(after), after_passes=P.passes(after),
        lambda_eff_change_rel=(after["lambda_eff"] - before["lambda_eff"]) / before["lambda_eff"],
        stop=res["stop"], result_from_round=res["result"]["from_round"], rounds=res["rounds"],
        targets_held_at_start=res.get("held", []), conventions=conv,
        kkt_after=dict(mu=res["result"]["diag"]["kkt_mu"],
                       first_order_gain_rel=res["result"]["diag"]["first_order_gain_rel"],
                       wrong_sign=res["result"]["diag"]["wrong_sign"]),
        inputs_sha256=inputs_sha, outputs_sha256=outputs_sha,
    )
    if wr["before"] is not None:
        rs = wr["before"]["resources"]
        tof, qub = float(rs["toffoli"]), int(rs["qubits"])
        rep["cost"] = dict(
            before_official_evaluation=wr["before"]["score"], toffoli=tof, qubits=qub,
            after_projected=after["lambda_eff"] * tof * qub,
            tier_note="before: official evaluate_circuits (default seed); after_projected: static model / "
                      "projection (after lambda_eff x before Toffolis x qubits; the per-shot Toffoli mean is "
                      "re-drawn under the new submission hash)")
    else:
        rep["cost"] = dict(before_official_evaluation=None, after_projected=None,
                           tier_note="unavailable: no kickmix build, so no circuit simulation")
    t_ver = time.time()
    if verify_after:
        va = verify_dir(dest, vdir, kickmix_dir)
        rep["verify_after"] = va if va is not None else "unavailable (no kickmix build)"
        log(event="verify_after", result={k: va.get(k) for k in ("status", "score", "lambda_eff", "error")}
            if va else None)
    rep["wall_s"] = dict(words=round(t_setup - t0, 1), refit=round(t_opt - t_setup, 1),
                         verify_after=round(time.time() - t_ver, 1), total=round(time.time() - t0, 1))
    rep["versions"] = _versions(vdir, wr["kickmix"])
    (out / "report.json").write_text(json.dumps(_jsonable(rep), indent=1) + "\n")
    log(event="done", lambda_before=before["lambda_eff"], lambda_after=after["lambda_eff"],
        npz_sha256=outputs_sha["submission.npz"])
    logf.close()
    return _jsonable(rep)


SUBCOMMAND_EPILOG = """subcommands (v1.7.0-dev; `dfthc_refit.py SUBCOMMAND --help` for each):
  conventions FAMILY_DIR           word conventions (storage, tree, gauge class) and loader class per slot (v1.5)
  [refit] SUBMISSION_DIR OUT_DIR   refit the DF-THC weights on the fixed words (the default mode; 'refit' optional)
  replan FAMILY_DIR OUT_DIR        re-plan, emit and check every loader; keep strict gains (positive-gain acceptance)
                                   --backend qrom-plan | kernel | shared | block, or --plans-from DIR (replay)
  plan FAMILY_DIR --planner P      per-slot plans and predicted Toffolis, nothing emitted (kernel | shared | block)
  recover FAMILY_DIR               structure census of the shipped loaders [--emit-check]
  price-words FAMILY_DIR           shadow prices and priced word moves [--moves FILE | --rank-drop K] (PyTorch)
  chained-and TARGET.json ACTION   AND-count bounds / exhaustive verify / CNF encode (bounds | verify | encode)
global option: --gf2-backend python|rust|auto (default auto: Rust kickmix_analysis if it imports; plans identical)
"""


def _take_gf2_backend(argv):
    """Remove a global `--gf2-backend {python,rust,auto}` (any position, any subcommand) and apply it. It sets
    DFTHC_GF2_BACKEND so spawned planner processes resolve the same backend; plans are identical on both."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        if t == "--gf2-backend" or t.startswith("--gf2-backend="):
            if "=" in t:
                v = t.split("=", 1)[1]
            elif i + 1 < len(argv):
                v = argv[i + 1]
                i += 1
            else:
                raise SystemExit("--gf2-backend needs a value: python, rust or auto")
            if v not in ("python", "rust", "auto"):
                raise SystemExit(f"--gf2-backend must be python, rust or auto, not {v!r}")
            os.environ["DFTHC_GF2_BACKEND"] = v
            if "recover" in sys.modules:
                sys.modules["recover"].GF2_BACKEND = v
                sys.modules["recover"]._AUTO = None
        else:
            out.append(t)
        i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Refit DF-THC weights on the fixed words of a dfthcsos submission.",
                                 epilog=SUBCOMMAND_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission_dir", type=Path, help="angle_k_*.kmb/.kmx + submission.npz")
    ap.add_argument("out_dir", type=Path, help="writes OUT_DIR/dfthcsos/ and OUT_DIR/report.json")
    ap.add_argument("--verifier-dir", help="the competition's verifiers directory (holds dfthcsos/verify.py)")
    ap.add_argument("--kickmix-dir", help="directory with a kickmix build (for word recovery and verification)")
    ap.add_argument("--arm", choices=("A", "B"), default="B", help="A: refit only; B: refit <-> gauge (default)")
    ap.add_argument("--max-evals", type=int, default=1000, help="metric evaluations per refit round")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--e1t", default=None,
                    help="error_1 target in mHa, 'start' (hold its input value) or 'free' (inequality, <= 19.9995). "
                         "Default 'free', which holds a start already past 19.9995 at its own value; a value given "
                         "here is never held (v1.7.0)")
    ap.add_argument("--e2t", type=float, default=None,
                    help="error_2 target in mHa. Default 19.999, which holds a start within 1e-6 mHa of the 20 mHa "
                         "limit at its own value; a value given here is the target even then (v1.7.0)")
    ap.add_argument("--gap-margin", type=float, default=1e-5, help="E_gap target above the floor, Ha")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="gauge-step kicks")
    ap.add_argument("--min-gain", type=float, default=1e-7,
                    help="stop after --patience consecutive rounds whose realised relative lambda gain is below this")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--kick-rounds", type=int, default=1, help="gauge rounds that also try random kicks")
    ap.add_argument("--gauge-procs", type=int, default=None, help="processes for the per-rank gauge (default: cores-1, max 9)")
    ap.add_argument("--brot", type=int, default=None,
                    help="bits per angle word, used only when submission.npz has no 'brot' field and no kickmix build "
                         "can read the circuits' word registers (v1.7.0); must agree with both when they exist")
    ap.add_argument("--no-verify-after", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    argv = sys.argv[1:] if argv is None else list(argv)
    argv = _take_gf2_backend(argv)
    if argv and argv[0] == "replan":   # v1.2: loader re-planning mode
        return replan_main(argv[1:])
    # v1.4: family subcommands; a submission directory that happens to carry a subcommand's name still refits
    if argv and argv[0] in SUBCOMMANDS and not (Path(argv[0]) / "submission.npz").is_file():
        return SUBCOMMANDS[argv[0]](argv[1:])
    if argv and argv[0] == "refit" and not (Path(argv[0]) / "submission.npz").is_file():
        argv = argv[1:]
    if not argv or argv == ["--version"]:   # e.g. Kaggle running the utility script on save
        print(f"dfthc_refit {__version__} (run with --help for usage)")
        return
    a = ap.parse_args(argv)
    e1t = None if a.e1t is None else a.e1t if a.e1t in ("start", "free") else float(a.e1t)
    rep = run(a.submission_dir, a.out_dir, verifier_dir=a.verifier_dir, kickmix_dir=a.kickmix_dir, arm=a.arm,
              rounds=a.rounds, max_evals=a.max_evals, e1t=e1t, e2t=a.e2t, gap_margin=a.gap_margin,
              threads=a.threads, seed=a.seed, verify_after=not a.no_verify_after, verbose=not a.quiet,
              min_gain=a.min_gain, patience=a.patience, kick_rounds=a.kick_rounds, gauge_procs=a.gauge_procs,
              brot=a.brot)
    print(json.dumps(dict(before=rep["before"], after=rep["after"], cost=rep["cost"], stop=rep["stop"],
                          npz_sha256=rep["outputs_sha256"]["submission.npz"]), indent=1))


if __name__ == "__main__":
    main()
