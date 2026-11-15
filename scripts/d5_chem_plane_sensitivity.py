#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""How much accuracy each BIT PLANE and each TREE DEPTH of the chemistry angle words carries.

For a public family: clear one bit plane (bit p of every two-body angle word, p = 0 least significant), or clear the
low `k` bits of every angle at tree depth >= D, decode the vectors, refit the weights only for the two-body error
(L-BFGS from the family's weights), and report the error. A cell that can be cleared cheaply could instead be forced
into a cheap code (affine or quadratic in the address) at little accuracy cost. Control: no change -> 19.86 mHa.
Run (a Python that can import kickmix): $QHDL_KICKMIX_PYTHON -I scripts/d5_chem_plane_sensitivity.py FAMILY_DIR OUT.json [maxiter]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402
CFG.add_verifiers_path()  # the official verifiers (not shipped): QHDL_OFFICIAL_ROOT
from dfthcsos import verify as V                      # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as DM      # noqa: E402

N = 76
fam = Path(sys.argv[1]); maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 120
inp = V.load_inputs(fam / "submission.npz")
z = np.load(fam / "submission.npz")
words0 = z["angle_words"].astype(np.int64); edges = z["tree_edges"]
depth = {0: 0}
for p_, c_ in edges: depth[int(c_)] = depth[int(p_)] + 1
col_depth = np.array([depth[int(c_)] for _, c_ in edges])            # depth of the child node of each angle column
w0 = np.asarray(inp.w, dtype=np.float64); R, B, C = w0.shape
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
d = np.arange(N); T[:, :, d, d] += 0.5 * S[:, :, None]; T[d, d, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(N * N, N * N)

def refit(words):
    u_all = DM.reconstruct_u_vectors(words / (1 << 7), edges)
    _, u = DM.split_u_vectors(u_all, inp.w)
    u = np.asarray(u, dtype=np.float64); u = u / np.linalg.norm(u, axis=2, keepdims=True)
    P = np.einsum("rbp,rbq->rbpq", u, u).reshape(R, B, N * N)
    def fun(x):
        w = x.reshape(R, B, C)
        X = np.einsum("rbk,rbc->rck", P, w).reshape(R * C, N * N)
        D = X.T @ X - Tm
        G = 4.0 * (X @ D)
        return float(np.sum(D * D)), np.einsum("rbk,rck->rbc", P, G.reshape(R, C, N * N)).ravel()
    f0 = fun(w0.ravel())[0]
    res = minimize(fun, w0.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxcor=20))
    return 1e3 * f0 ** 0.5, 1e3 * res.fun ** 0.5

out = dict(family=fam.name, maxiter=maxiter, column_depth_histogram={int(k): int((col_depth == k).sum()) for k in np.unique(col_depth)}, rows=[])
def run(label, words):
    t0 = time.time(); a, b = refit(words)
    out["rows"].append(dict(change=label, error_before_refit_mHa=a, error_after_refit_mHa=b, seconds=round(time.time() - t0, 1)))
    print(out["rows"][-1], flush=True); Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
run("none (control)", words0)
for p in range(0, 4):
    wd = words0.copy(); wd[N:] &= ~(1 << p); run(f"clear bit {p} of every two-body angle", wd)
for D_ in (5, 4, 3):
    for k in (1, 2, 3):
        wd = words0.copy(); cols = np.flatnonzero(col_depth >= D_)
        wd[N:, cols] &= ~((1 << k) - 1); run(f"clear the low {k} bit(s) of angles at tree depth >= {D_} ({len(cols)} columns)", wd)
