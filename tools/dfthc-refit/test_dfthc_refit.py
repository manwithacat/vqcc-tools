# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dfthc_refit on a small synthetic problem (N = 6 orbitals, R = 2, B = 4, C = 3).

The synthetic "official" metrics below follow the formulas of the official dfthcsos.scoreinfra.dfthc_math
(evaluate_dfthc_vectors) in plain NumPy. If $DFTHC_REFIT_VERIFIER_DIR points at the competition's verifiers directory,
the official function itself is also checked against this reference.

    python -m unittest test_dfthc_refit -v
"""
import io
import math
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dfthc_refit as dr  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

N, R, B, C = 6, 2, 4, 3
ETA = 4


def reference_metrics(u_all, w, W, S, beta1, oei, tei, ecore, eup):
    """NumPy transcription of the official evaluate_dfthc_vectors (without its Decimal cross-check)."""
    u = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
    V, ub = u[:N], u[N:].reshape(R, B, N)
    X = np.einsum("rbp,rbq,rbc->rcpq", ub, ub, w)
    h1 = oei - 0.5 * np.einsum("prrq->pq", tei)
    h1p = h1 - 0.5 * ETA * S + beta1 * np.eye(N) - np.einsum("rc,rcpq->pq", W, X)
    ev, Ev = np.linalg.eigh(h1p)
    from scipy.optimize import linear_sum_assignment
    qi, ii = linear_sum_assignment(-np.abs(V @ Ev))
    lm = np.empty_like(ev)
    lm[qi] = ev[ii]
    e1 = np.linalg.norm(np.einsum("j,jp,jq->pq", lm, V, V) - h1p)
    d = np.einsum("rcpq,rcst->pqst", X, X) - tei
    i = np.arange(N)
    d[:, :, i, i] -= 0.5 * S[:, :, None]
    d[i, i, :, :] -= 0.5 * S[None, :, :]
    e2 = np.linalg.norm(d.ravel())
    lam1 = np.abs(ev).sum()
    Lam = lam1 + 0.25 * ((np.abs(w).sum(1) + np.abs(W + w.sum(1))) ** 2).sum()
    esos = ecore - ETA * beta1 - (lam1 - ev.sum()) - 0.5 * (W ** 2).sum()
    gap = eup - esos
    ok = 0 < gap <= 2 * Lam
    return dict(error_1=float(e1), error_2=float(e2), Lambda=float(Lam), E_SOS=float(esos), E_gap=float(gap),
                lambda_eff=float(math.sqrt(gap * (2 * Lam - gap))) if ok else float("nan"), gap_in_domain=bool(ok))


def synthetic():
    """A planted factorization: tei = sum X (x) X - (S(x)I + I(x)S)/2 + small noise, one-body rows = eigenvectors
    of h1_prime plus a small rotation; limits and floors scaled so the start is feasible with slack."""
    rng = np.random.default_rng(1234)
    ub = rng.normal(size=(R, B, N))
    ub /= np.linalg.norm(ub, axis=2, keepdims=True)
    w = rng.normal(size=(R, B, C)) * 0.5
    W = rng.normal(size=(R, C)) * 0.3
    S = rng.normal(size=(N, N)) * 0.05
    S = 0.5 * (S + S.T)
    X = np.einsum("rbp,rbq,rbc->rcpq", ub, ub, w)
    tei = np.einsum("rcpq,rcst->pqst", X, X)
    i = np.arange(N)
    tei[:, :, i, i] -= 0.5 * S[:, :, None]
    tei[i, i, :, :] -= 0.5 * S[None, :, :]
    noise = rng.normal(size=(N,) * 4) * 2e-3
    noise = noise + noise.transpose(1, 0, 2, 3)
    noise = noise + noise.transpose(0, 1, 3, 2)
    noise = 0.25 * (noise + noise.transpose(2, 3, 0, 1))   # chemist-order 8-fold symmetry, as real integrals
    tei = tei + noise
    oei = rng.normal(size=(N, N))
    oei = 0.5 * (oei + oei.T)
    beta1 = 0.1
    h1 = oei - 0.5 * np.einsum("prrq->pq", tei)
    h1p = h1 - 0.5 * ETA * S + beta1 * np.eye(N) - np.einsum("rc,rcpq->pq", W, X)
    _, Ev = np.linalg.eigh(h1p)
    G = rng.normal(size=(N, N)) * 0.004
    from scipy.linalg import expm
    Vrows = (expm(G - G.T) @ Ev).T
    u_all = np.vstack([Vrows, ub.reshape(R * B, N)])
    ecore, eup = 1.0, 0.0
    wB = W + w.sum(1)
    m0 = reference_metrics(u_all, w, W, S, beta1, oei, tei, ecore, eup)
    lim = 1.0105 * max(m0["error_1"], m0["error_2"])   # small moves: a Gauss-Newton retraction converges fast
    floor = m0["E_gap"] - 0.0021

    def official(ww, WW):
        return reference_metrics(u_all, ww, WW, S, beta1, oei, tei, ecore, eup)

    P = dr.Problem(u_all, w, wB, S, beta1, oei, tei, ETA, ecore, eup, official,
                   gap_floor=floor, lambda_floor=0.0, error_limit=lim)
    return P, dict(u_all=u_all, S=S, beta1=beta1, oei=oei, tei=tei, ecore=ecore, eup=eup)


def targets_for(P, y):
    m = P.official(y)
    return (m["error_1"] * 1e3, m["error_2"] * 1e3 * 1.01, P.gap_floor + 1e-4)


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch (the refit model is a torch model)")
class TestSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.P, cls.data = synthetic()

    def test_torch_model_matches_reference(self):
        P = self.P
        y = P.y0()
        q = P.evaluate(y)
        m = P.official(y)
        self.assertAlmostEqual(q["e1"] * 1e-3, m["error_1"], delta=1e-10)
        self.assertAlmostEqual(q["e2"] * 1e-3, m["error_2"], delta=1e-10)
        self.assertAlmostEqual(q["Lam"], m["Lambda"], delta=1e-10)
        self.assertAlmostEqual(q["gap"], m["E_gap"], delta=1e-10)
        self.assertAlmostEqual(q["lam"], m["lambda_eff"], delta=1e-9)

    def test_gauge_invariance(self):
        P = self.P
        w, wB = P.split(P.y0())
        gw, gwB, Os, info = dr.gauge(w, wB, seed=0)
        for O in Os:
            np.testing.assert_allclose(O.T @ O, np.eye(C), atol=1e-12)
        m0, m1 = P.official(P.y0()), P.official(P.from_weights(gw, gwB))
        for k in ("error_1", "error_2", "E_gap"):
            self.assertAlmostEqual(m0[k], m1[k], delta=1e-11, msg=k)
        self.assertLessEqual(sum(info["after"]), sum(info["before"]) + 1e-12)
        self.assertLessEqual(m1["Lambda"], m0["Lambda"] + 1e-12)

    def test_gauge_same_with_process_pool(self):
        w, wB = self.P.split(self.P.y0())
        a = dr.gauge(w, wB, seed=0, procs=1)
        b = dr.gauge(w, wB, seed=0, procs=2)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_refit_holds_constraints_and_lowers_lambda(self):
        P = self.P
        y0 = P.y0()
        t = targets_for(P, y0)
        r = dr.refit(P, y0, t, max_evals=150)
        self.assertIsNotNone(r["y"])
        m0, m1 = P.official(y0), P.official(r["y"])
        self.assertTrue(P.passes(m1))
        self.assertLess(m1["lambda_eff"], m0["lambda_eff"])
        self.assertLessEqual(m1["error_2"] * 1e3, t[1] + dr.FEAS_E_TOL_MHA)
        self.assertLessEqual(m1["error_1"] * 1e3, t[0] + dr.FEAS_E_TOL_MHA)
        self.assertGreaterEqual(m1["E_gap"], t[2] - dr.FEAS_GAP_TOL)
        self.assertGreater(m1["E_gap"], P.gap_floor)

    def test_alternation_deterministic_and_feasible(self):
        P = self.P
        kw = dict(arm="B", rounds=3, max_evals=80, e1t="start", e2t=targets_for(P, P.y0())[1], gap_margin=1e-4)
        a = dr.alternate(P, P.y0(), **kw)
        b = dr.alternate(P, P.y0(), **kw)
        self.assertIsNotNone(a["result"])
        self.assertTrue(np.array_equal(a["result"]["y"], b["result"]["y"]))
        m = P.official(a["result"]["y"])
        self.assertTrue(P.passes(m))
        self.assertLessEqual(m["lambda_eff"], P.official(P.y0())["lambda_eff"])
        # same arrays -> same npz bytes
        with tempfile.TemporaryDirectory() as d:
            w, wB = P.split(a["result"]["y"])
            arrs = dict(W=w, H2_I=wB, tree_edges=np.arange(10).reshape(5, 2))
            dr.write_npz(Path(d) / "a.npz", arrs)
            dr.write_npz(Path(d) / "b.npz", dict(arrs))
            self.assertEqual(dr.sha256_file(Path(d) / "a.npz"), dr.sha256_file(Path(d) / "b.npz"))
            z = np.load(Path(d) / "a.npz")
            self.assertTrue(np.array_equal(z["W"], w) and np.array_equal(z["H2_I"], wB))

    def test_arm_a_never_worse_than_input(self):
        P = self.P
        r = dr.alternate(P, P.y0(), arm="A", max_evals=5, e2t=targets_for(P, P.y0())[1], gap_margin=1e-4)
        self.assertLessEqual(r["result"]["lambda_eff"], P.official(P.y0())["lambda_eff"])

    def test_npz_is_numpy_readable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.npz"
            dr.write_npz(p, dict(a=np.ones(3), b=np.int64(7)))
            with zipfile.ZipFile(p) as zf:
                self.assertEqual([i.date_time for i in zf.infolist()], [(1980, 1, 1, 0, 0, 0)] * 2)
            z = np.load(p, allow_pickle=False)
            self.assertEqual(int(z["b"]), 7)

    @unittest.skipUnless(os.environ.get("DFTHC_REFIT_VERIFIER_DIR"), "official verifier not given")
    def test_reference_matches_official_math(self):
        dm, _ = dr._official_modules(Path(os.environ["DFTHC_REFIT_VERIFIER_DIR"]))
        P, d = self.P, self.data
        w, wB = P.split(P.y0())
        W = wB - w.sum(1)
        ref = dict(oei=d["oei"], tei=d["tei"], electrons=ETA, ecore=d["ecore"])
        mo = dm.evaluate_dfthc_vectors(d["u_all"], w, W, d["S"], d["beta1"], ref, energy_upper=d["eup"])
        mr = P.official(P.y0())
        for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff"):
            self.assertAlmostEqual(float(mo[k]), mr[k], delta=1e-12 * max(1.0, abs(mr[k])), msg=k)


if __name__ == "__main__":
    unittest.main()
