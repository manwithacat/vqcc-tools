#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Exhaustive, exact single-word local search on the chemistry two-body angle words.

F = ||sum_rc x_rc x_rc^T - T'||_F^2 with x_rc = sum_b w_b,rc vec(u_rb u_rb^T). Changing one word of vector (r, b)
changes vec(u u^T) by delta, and X^T X by y delta^T + delta y^T + s delta delta^T (y = sum_c w_c x_rc, s = sum_c w_c^2),
so dF = 4 y.D.delta + 2 s delta.D.delta + 2 (y.y)(d.d) + 2 (y.d)^2 + s^2 (d.d)^2 + 4 s (y.d)(d.d): all 150 moves of a
vector cost one product with the residual D. Greedy: apply the best improving move, repeat up to REP times per vector;
after each sweep refit the weights (L-BFGS) and recompute D exactly.
Run (a Python that can import kickmix): $QHDL_KICKMIX_PYTHON -I scripts/d6_chem_word_search.py FAMILY_DIR OUT_DIR [sweeps] [refit_iters] [resume.npz]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import blas
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402
CFG.add_verifiers_path()  # the official verifiers (not shipped): QHDL_OFFICIAL_ROOT
from dfthcsos import verify as V                      # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as DM      # noqa: E402

N, REP = 76, 3
import os
SCALE = float(os.environ.get('D6_SCALE', '1'))
T0, T1, HOT = float(os.environ.get('D6_T0', '0')), float(os.environ.get('D6_T1', '0')), int(os.environ.get('D6_HOT', '0'))   # annealing: heat-bath over a vector's 150 moves + staying, temperature T0 -> T1 (units of F) over the first HOT sweeps, then greedy
fam, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
refit_iters = int(sys.argv[4]) if len(sys.argv) > 4 else 80
inp = V.load_inputs(fam / "submission.npz"); z = np.load(fam / "submission.npz")
edges = z["tree_edges"]; words = z["angle_words"].astype(np.int64).copy()
w = np.asarray(inp.w, dtype=np.float64).copy(); R, B, C = w.shape
start_sweep = 0; ALIVE = None
if len(sys.argv) > 5:
    st = np.load(sys.argv[5]); words, w, start_sweep = st["words"].astype(np.int64), st["w"], int(st["sweep"])
    if "alive" in st.files: ALIVE = st["alive"]                    # deleted vectors (d6_chem_vector_removal.py) stay at zero weight
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
dg = np.arange(N); T[:, :, dg, dg] += 0.5 * S[:, :, None]; T[dg, dg, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(N * N, N * N)

def decode(rows):
    u = np.asarray(DM.reconstruct_u_vectors(rows / (1 << 7), edges), dtype=np.float64)
    return u / np.linalg.norm(u, axis=1, keepdims=True)
def proj(u): return np.einsum("kp,kq->kpq", u, u).reshape(len(u), N * N)
_, u_chk = DM.split_u_vectors(DM.reconstruct_u_vectors(words / (1 << 7), edges), inp.w)
assert np.allclose(np.asarray(u_chk).reshape(R * B, N), np.asarray(DM.reconstruct_u_vectors(words[N:] / (1 << 7), edges))), "row order"

def build():
    P = proj(decode(words[N:])).reshape(R, B, N * N)
    X = np.einsum("rbk,rbc->rck", P, w).reshape(R * C, N * N)
    D = np.asfortranarray(X.T @ X - Tm)
    return P, X, D
def refit(P, iters):
    def fun(x):
        ww = x.reshape(R, B, C); X = np.einsum("rbk,rbc->rck", P, ww).reshape(R * C, N * N)
        D = X.T @ X - Tm; G = 4.0 * (X @ D); g = np.einsum("rbk,rck->rbc", P, G.reshape(R, C, N * N))
        if ALIVE is not None: g = g * ALIVE[:, :, None]            # zero weights have zero gradient, so L-BFGS leaves them at zero
        return float(np.sum(D * D)), g.ravel()
    if SCALE != 1:                                                 # scaled objective and tight tolerances (D6_SCALE)
        f0 = fun
        fun = lambda x: tuple(v * SCALE for v in f0(x))
        opts = dict(maxiter=iters, maxcor=20, ftol=1e-12, gtol=1e-6)
    else:
        opts = dict(maxiter=iters, maxcor=20)
    res = minimize(fun, w.ravel(), jac=True, method="L-BFGS-B", options=opts)
    return res.x.reshape(R, B, C)
mHa = lambda F: 1e3 * float(F) ** 0.5

def moves(row):
    cand = np.repeat(words[N + row][None, :], 150, axis=0)
    j = np.arange(75); cand[j, j] = (cand[j, j] + 1) & 255; cand[75 + j, j] = (cand[75 + j, j] - 1) & 255
    return cand

P, X, D = build(); F = float(np.sum(D * D)); print(f"start: error_2 {mHa(F):.4f} mHa (sweep {start_sweep})", flush=True)
rng = np.random.default_rng(20261115 + start_sweep)
# self-test: predicted dF against direct recomputation
row = int(rng.integers(R * B)); r, b = divmod(row, B); Xr = X[r * C:(r + 1) * C]; wv = w[r, b]; y = wv @ Xr; s = wv @ wv
cand = moves(row); delta = proj(decode(cand)) - P[r, b]; G = delta @ D; dd = np.einsum("kn,kn->k", delta, delta); yd = delta @ y
dF = 4 * (G @ y) + 2 * s * np.einsum("kn,kn->k", G, delta) + 2 * (y @ y) * dd + 2 * yd ** 2 + s * s * dd ** 2 + 4 * s * yd * dd
for k in (0, 77, 149):
    X2 = X.copy(); X2[r * C:(r + 1) * C] += np.outer(wv, delta[k]); D2 = X2.T @ X2 - Tm
    print(f"self-test move {k}: predicted dF {dF[k]:.6e}, direct {float(np.sum(D2 * D2)) - F:.6e}", flush=True)
del X2, D2

for sweep in range(start_sweep, start_sweep + sweeps):
    t0 = time.time(); acc = 0; tried = 0; h = sweep - start_sweep
    temp = T0 * (T1 / T0) ** (h / max(HOT - 1, 1)) if (T0 > 0 and h < HOT) else 0.0
    for row in rng.permutation(R * B):
        r, b = divmod(int(row), B); Xr = X[r * C:(r + 1) * C]; wv = w[r, b]; s = wv @ wv
        for _ in range(REP):
            y = wv @ Xr; cand = moves(row); delta = proj(decode(cand)) - P[r, b]
            G = delta @ D; dd = np.einsum("kn,kn->k", delta, delta); yd = delta @ y
            dF = 4 * (G @ y) + 2 * s * np.einsum("kn,kn->k", G, delta) + 2 * (y @ y) * dd + 2 * yd ** 2 + s * s * dd ** 2 + 4 * s * yd * dd
            tried += 150
            if temp > 0:                                           # heat bath: P(move k) ~ exp(-dF_k / temp), P(stay) ~ 1
                pr = np.exp(-np.maximum(dF, -50 * temp) / temp); pr = np.append(pr, 1.0); k = int(rng.choice(151, p=pr / pr.sum()))
                if k == 150: break
            else:
                k = int(np.argmin(dF))
                if dF[k] >= -1e-12 * F: break
            d = np.ascontiguousarray(delta[k]); words[N + row] = cand[k]; P[r, b] += d; Xr += np.outer(wv, d)
            D = blas.dger(1.0, y, d, a=D, overwrite_a=1); D = blas.dger(1.0, d, y, a=D, overwrite_a=1); D = blas.dger(s, d, d, a=D, overwrite_a=1)
            F += dF[k]; acc += 1
    e_fixed = mHa(F); w = refit(P, refit_iters); P, X, D = build(); F = float(np.sum(D * D))
    np.savez_compressed(out / f"state_sweep{sweep + 1:03d}.npz", words=words, w=w, sweep=sweep + 1, **({} if ALIVE is None else dict(alive=ALIVE)))
    rec = dict(sweep=sweep + 1, temperature=temp, moves_tried=tried, moves_accepted=acc, error_2_before_refit_mHa=e_fixed, error_2_mHa=mHa(F),
               words_changed_vs_pick=int((words != z["angle_words"]).sum()), seconds=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)
    with open(out / "log.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
