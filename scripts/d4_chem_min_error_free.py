#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""The best two-body error a chemistry SHAPE can reach with CONTINUOUS vectors (no 8-bit rounding).

Starts from a family's decoded vectors and weights and minimises error_2 = ||sum_rc X_rc (x) X_rc - T'||_F over both the
unit vectors u and the weights w (L-BFGS). Optional: drop groups or rank columns first (--groups R --rank C keep the
first R groups / C columns) to probe smaller shapes from the same start. The result separates what the vector
structure costs from what rounding the Givens angles to 8 bits costs (the best public family at the time of writing sat at 19.9 mHa).
Run with a Python that can import kickmix (the official verifiers are needed too):
    $QHDL_KICKMIX_PYTHON -I scripts/d4_chem_min_error_free.py FAMILY_DIR OUT.json [maxiter] [groups] [rank] [vectors per group]
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
fam = Path(sys.argv[1]); maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 300
inp = V.load_inputs(fam / "submission.npz")
z = np.load(fam / "submission.npz")
u_all = DM.reconstruct_u_vectors(z["angle_words"] / (1 << 7), z["tree_edges"])
_, u0 = DM.split_u_vectors(u_all, inp.w)
u0 = np.asarray(u0, dtype=np.float64); w0 = np.asarray(inp.w, dtype=np.float64)
Rk = int(sys.argv[4]) if len(sys.argv) > 4 else w0.shape[0]
Ck = int(sys.argv[5]) if len(sys.argv) > 5 else w0.shape[2]
u0, w0 = u0[:Rk], w0[:Rk, :, :Ck]
Bk = int(sys.argv[6]) if len(sys.argv) > 6 else w0.shape[1]
if Bk < w0.shape[1]:                                  # keep the Bk vectors per group with the largest weight norm
    keep = np.sort(np.argsort(-np.linalg.norm(w0, axis=2), axis=1)[:, :Bk], axis=1)
    u0 = np.take_along_axis(u0, keep[:, :, None], axis=1)
    w0 = np.take_along_axis(w0, keep[:, :, None], axis=1)
if Ck > w0.shape[2]:                                  # pad extra rank columns with small noise (a new, larger rank)
    pad = np.random.default_rng(7).standard_normal((w0.shape[0], w0.shape[1], Ck - w0.shape[2])) * 1e-3
    w0 = np.concatenate([w0, pad], axis=2)
R, B, C = w0.shape
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
d = np.arange(N); T[:, :, d, d] += 0.5 * S[:, :, None]; T[d, d, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(N * N, N * N)
nu, nw = R * B * N, R * B * C

def fun(x):
    uraw = x[:nu].reshape(R, B, N); w = x[nu:].reshape(R, B, C)
    nrm = np.linalg.norm(uraw, axis=2, keepdims=True); u = uraw / nrm
    P = np.einsum("rbp,rbq->rbpq", u, u).reshape(R, B, N * N)
    X = np.einsum("rbk,rbc->rck", P, w).reshape(R * C, N * N)
    D = X.T @ X - Tm
    f = float(np.sum(D * D))
    G = (4.0 * (X @ D)).reshape(R, C, N, N)
    gw = np.einsum("rbk,rck->rbc", P, G.reshape(R, C, N * N))
    A = np.einsum("rbc,rcpq->rbpq", w, G); A = A + A.transpose(0, 1, 3, 2)      # dF/dP symmetrised (x2 for u u^T)
    gu_hat = np.einsum("rbpq,rbq->rbp", A, u)
    gu = (gu_hat - u * np.sum(gu_hat * u, axis=2, keepdims=True)) / nrm
    return f, np.concatenate([gu.ravel(), gw.ravel()])

x0 = np.concatenate([u0.ravel(), w0.ravel()])
STATE = Path(sys.argv[2]).with_suffix(".state.npz")        # resume from a saved state of the same shape, if present
if STATE.exists():
    sx = np.load(STATE)["x"]
    if sx.shape == x0.shape:
        x0 = sx; print("resumed from", STATE.name, flush=True)
f0, g0 = fun(x0)
# finite-difference check of the gradient on a few coordinates
rng = np.random.default_rng(1); chk = []
for i in rng.choice(len(x0), 4, replace=False):
    e = np.zeros_like(x0); e[i] = 1e-6
    chk.append(((fun(x0 + e)[0] - fun(x0 - e)[0]) / 2e-6, g0[i]))
t0 = time.time()
trace = []
def cb(xk):
    trace.append(1)
    if len(trace) % 200 == 0:
        np.savez(STATE, x=xk); print(len(trace), "iterations, error_2", 1e3 * fun(xk)[0] ** 0.5, "mHa", flush=True)
import os
SCALE = float(os.environ.get("D4_SCALE", "1"))         # scale f and g (e.g. 1e6) and tighten tolerances
opts = dict(maxiter=maxiter, maxcor=30)
if SCALE != 1:
    opts.update(ftol=float(os.environ.get("D4_FTOL", "1e-12")), gtol=float(os.environ.get("D4_GTOL", "1e-6")))
fun_s = (lambda x: tuple(v * SCALE for v in fun(x))) if SCALE != 1 else fun
res = minimize(fun_s, x0, jac=True, method="L-BFGS-B", callback=cb, options=opts)
res.fun = res.fun / SCALE
np.savez(STATE, x=res.x)
rep = dict(family=fam.name, shape=[R, B, C], generators=R * C, vectors=R * B, error2_start_mHa=1e3 * f0 ** 0.5,
           error2_min_mHa=1e3 * res.fun ** 0.5, iterations=int(res.nit), converged=bool(res.success),
           message=str(res.message), nfev=int(res.nfev), max_abs_grad=float(np.abs(fun(res.x)[1]).max()), scale=SCALE, options={k: v for k, v in opts.items()},
           gradient_check=[[float(a), float(b)] for a, b in chk], seconds=round(time.time() - t0, 1),
           note="continuous unit vectors and weights; no rounding; lambda_eff, error_1 and E_gap not constrained")
print(json.dumps(rep, indent=1)); Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
