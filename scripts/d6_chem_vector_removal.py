#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""What does removing vectors from the chemistry two-body table cost in accuracy?

From a word-search state of scripts/d6_chem_word_search.py (words + weights): exact change of the squared two-body error for deleting each vector alone (same
rank-two formula as d6_chem_word_search.py with delta = -vec(u u^T)), then delete the K cheapest per group (greedy,
recomputing after each deletion), refit the remaining weights, and report the error. Deleted vectors keep zero weights.
Run (a Python that can import kickmix): $QHDL_KICKMIX_PYTHON -I scripts/d6_chem_vector_removal.py FAMILY_DIR STATE.npz OUT.json K [refit_iters]
"""
import json, sys
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
N = 76
fam = Path(sys.argv[1]); st = np.load(sys.argv[2]); K = int(sys.argv[4]); iters = int(sys.argv[5]) if len(sys.argv) > 5 else 150
inp = V.load_inputs(fam / "submission.npz"); edges = np.load(fam / "submission.npz")["tree_edges"]
words, w = st["words"].astype(np.int64), st["w"].astype(np.float64).copy(); R, B, C = w.shape
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
dg = np.arange(N); T[:, :, dg, dg] += 0.5 * S[:, :, None]; T[dg, dg, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(N * N, N * N)
u = np.asarray(DM.reconstruct_u_vectors(words[N:] / (1 << 7), edges), dtype=np.float64); u /= np.linalg.norm(u, axis=1, keepdims=True)
P = np.einsum("kp,kq->kpq", u, u).reshape(R, B, N * N)
mHa = lambda F: 1e3 * float(F) ** 0.5
alive = np.ones((R, B), bool)
def state():
    X = np.einsum("rbk,rbc->rck", P, w).reshape(R * C, N * N); D = np.asfortranarray(X.T @ X - Tm); return X, D
def removal_costs(X, D):
    Pf = P.reshape(R * B, N * N); G = Pf @ D                       # (RB, N^2)
    Y = np.einsum("rbc,rck->rbk", w, X.reshape(R, C, N * N)).reshape(R * B, N * N)
    s = np.einsum("rbc,rbc->rb", w, w).ravel()
    dd = np.einsum("kn,kn->k", Pf, Pf); yd = -np.einsum("kn,kn->k", Y, Pf)
    return (-4 * np.einsum("kn,kn->k", G, Y) + 2 * s * np.einsum("kn,kn->k", G, Pf) + 2 * np.einsum("kn,kn->k", Y, Y) * dd
            + 2 * yd ** 2 + s * s * dd ** 2 + 4 * s * yd * dd).reshape(R, B)
def refit():
    def fun(x):
        ww = x.reshape(R, B, C) * alive[:, :, None]; X = np.einsum("rbk,rbc->rck", P, ww).reshape(R * C, N * N)
        D = X.T @ X - Tm; G = 4.0 * (X @ D)
        return float(np.sum(D * D)), (np.einsum("rbk,rck->rbc", P, G.reshape(R, C, N * N)) * alive[:, :, None]).ravel()
    res = minimize(fun, w.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=iters, maxcor=20))
    return res.x.reshape(R, B, C) * alive[:, :, None]
X, D = state(); F0 = float(np.sum(D * D)); dF = removal_costs(X, D)
single = np.sqrt(np.maximum(F0 + dF, 0)) * 1e3
out = dict(start_error_2_mHa=mHa(F0), weight_norm_quantiles=np.quantile(np.sqrt(np.einsum("rbc,rbc->rb", w, w)), [0, .01, .1, .5, .9, 1]).tolist(),
           error_after_deleting_one_vector_mHa=dict(min=float(single.min()), q01=float(np.quantile(single, .01)), q10=float(np.quantile(single, .1)),
                                                    median=float(np.median(single)), max=float(single.max())),
           cheapest_per_group_mHa=[float(single[r].min()) for r in range(R)], rounds=[])
print(json.dumps({k: out[k] for k in list(out)[:4]}), flush=True)
for k in range(1, K + 1):
    for r in range(R):
        c = np.where(alive[r], dF[r], np.inf); b = int(np.argmin(c)); alive[r, b] = False; w[r, b] = 0.0
    X, D = state(); e_before = mHa(np.sum(D * D)); w = refit(); X, D = state(); F = float(np.sum(D * D)); dF = removal_costs(X, D)
    out["rounds"].append(dict(vectors_per_group=B - k, error_2_before_refit_mHa=e_before, error_2_after_refit_mHa=mHa(F)))
    print(json.dumps(out["rounds"][-1]), flush=True); Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
np.savez_compressed(Path(sys.argv[3]).with_suffix(".state.npz"), words=words, w=w, alive=alive, sweep=0)
