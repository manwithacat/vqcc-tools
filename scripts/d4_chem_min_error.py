#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""How much two-body accuracy slack hides behind a chemistry family whose refit pushes error_2 to the 20 mHa cap?

Keeps the family's decoded 8-bit vectors fixed and minimises error_2 = ||sum_rc X_rc (x) X_rc - T'||_F over the weights
w only (X_rc = sum_b w_b,rc u_b u_b^T), by L-BFGS. The minimum is the best two-body error this set of vectors can
reach; the gap to 20 mHa is slack that a smaller shape or coarser words could spend. lambda_eff is ignored here on
purpose: the one-norm bounds of scripts/d4_chem_lambda_*.py show it has an absolute floor (46.786) that the best
public family already sits on.
Run with a Python that can import kickmix (the official verifiers are needed too):
    $QHDL_KICKMIX_PYTHON -I scripts/d4_chem_min_error.py FAMILY_DIR OUT.json [maxiter]
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
fam = Path(sys.argv[1]); maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 150
inp = V.load_inputs(fam / "submission.npz")
z = np.load(fam / "submission.npz")
words, edges = z["angle_words"], z["tree_edges"]
u_all = DM.reconstruct_u_vectors(words / (1 << 7), edges)
one_body, u = DM.split_u_vectors(u_all, inp.w)
u = np.asarray(u, dtype=np.float64); u = u / np.linalg.norm(u, axis=2, keepdims=True)      # (R, B, N)
SRC = sys.argv[4] if len(sys.argv) > 4 else "words"
if SRC == "npzU":                                    # the continuous vectors stored in the archive (not used by the verifier)
    Uc = np.asarray(z["U"], dtype=np.float64); Uc = Uc / np.linalg.norm(Uc, axis=2, keepdims=True)
    print("mean |cos| between decoded and stored vectors:", float(np.abs(np.sum(u * Uc, axis=2)).mean()), flush=True)
    u = Uc
w0 = np.asarray(inp.w, dtype=np.float64)                                                    # (R, B, C)
R, B, C = w0.shape
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
d = np.arange(N); T[:, :, d, d] += 0.5 * S[:, :, None]; T[d, d, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(N * N, N * N)
P = np.einsum("rbp,rbq->rbpq", u, u).reshape(R, B, N * N)          # projectors as vectors

def fun(x):
    w = x.reshape(R, B, C)
    X = np.einsum("rbk,rbc->rck", P, w).reshape(R * C, N * N)       # generators as vectors (R*C, N^2)
    D = X.T @ X - Tm                                                # residual (N^2, N^2)
    f = float(np.sum(D * D))
    G = 4.0 * (X @ D)                                               # d f / d X  (D symmetric up to rounding)
    g = np.einsum("rbk,rck->rbc", P, G.reshape(R, C, N * N))
    return f, g.ravel()

t0 = time.time()
f0, _ = fun(w0.ravel())
hist = []
res = minimize(fun, w0.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxcor=20),
               callback=lambda xk: hist.append(1))
rep = dict(family=fam.name, shape=[R, B, C], error2_start_mHa=1e3 * f0 ** 0.5, error2_min_mHa=1e3 * res.fun ** 0.5,
           iterations=int(res.nit), converged=bool(res.success), seconds=round(time.time() - t0, 1),
           vectors=SRC, note="weights only, vectors fixed; lambda_eff not constrained")
print(json.dumps(rep, indent=1))
Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
