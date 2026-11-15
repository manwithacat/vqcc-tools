#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Dual lower bounds on the two-body one-norm lambda_h2 of ANY DF-THC-SOS factorization of the fixed target.

Any accepted factorization has sum_c X_c (x) X_c = T' (to 20 mHa), and its lambda_h2 >= 1/4 sum_c ||X_c||_*^2.
For PSD M = sum_k a_k v_k v_k^T (v_k eigenvectors of T', as N x N matrices V_k, eigenvalues s_k):
    1/4 sum_c ||X_c||_*^2  >=  1/4 <M, T'> / mu(M),     mu(M) = max_{|u|=1} sum_k a_k (u^T V_k u)^2.
mu is a quartic maximisation over the sphere. Reported: (i) the rigorous single-mode bounds s_k / (4 ||V_k||_op^2);
(ii) multi-mode bounds with mu ESTIMATED by projected gradient ascent from many starts (an estimate of mu from below,
so bound (ii) is indicative, not rigorous).
Run with a Python that can import kickmix (the official verifiers are needed too):
    $QHDL_KICKMIX_PYTHON -I scripts/d4_chem_lambda_dual.py FAMILY_DIR [OUT.json]
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402
CFG.add_verifiers_path()  # the official verifiers (not shipped): QHDL_OFFICIAL_ROOT
from dfthcsos import verify as V  # noqa: E402

N = 76
inp = V.load_inputs(Path(sys.argv[1]) / "submission.npz")
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
d = np.arange(N); T[:, :, d, d] += 0.5 * S[:, :, None]; T[d, d, :, :] += 0.5 * S[None, :, :]
Mx = T.reshape(N * N, N * N); Mx = 0.5 * (Mx + Mx.T)
s, vec = np.linalg.eigh(Mx); order = np.argsort(-s); s = s[order]; vec = vec[:, order]
K = 200
Vk = np.stack([0.5 * (vec[:, k].reshape(N, N) + vec[:, k].reshape(N, N).T) for k in range(K)])
opn = np.array([np.abs(np.linalg.eigvalsh(Vk[k])).max() for k in range(K)])
single = s[:K] / (4 * opn ** 2)
ident = np.eye(N) / np.sqrt(N)
rep = dict(top_eigenvalues=[float(x) for x in s[:8]], overlap_of_mode0_with_identity=float(abs(np.sum(Vk[0] * ident))),
           opnorm_mode0=float(opn[0]), rigorous_single_mode_bounds=[float(x) for x in single[:8]],
           best_rigorous_single_mode=float(single.max()), trace_bound=float(s[s > 0].sum() / 4))

rng = np.random.default_rng(0)
def mu_est(a, starts=400, iters=300):
    U = rng.standard_normal((starts, N)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    for _ in range(iters):
        q = np.einsum('si,kij,sj->sk', U, Vk[:len(a)], U)                # (starts, K) values u^T V_k u
        G = np.einsum('sk,kij,sj->si', q * a, Vk[:len(a)], U)            # gradient direction (up to 4x)
        U = G / np.linalg.norm(G, axis=1, keepdims=True)
    q = np.einsum('si,kij,sj->sk', U, Vk[:len(a)], U)
    return float(((q ** 2) * a).sum(axis=1).max())
out = []
for Kuse in (1, 5, 20, 60, 200):
    for p in (0.0, -0.5, -1.0):
        a = s[:Kuse] ** p
        mu = mu_est(a)
        out.append(dict(modes=Kuse, weight_power=p, mu_estimate=mu, bound_estimate=float((a * s[:Kuse]).sum() / (4 * mu))))
        print(out[-1], flush=True)
rep["multi_mode_estimates"] = out
rep["best_estimate"] = max(o["bound_estimate"] for o in out)
print(json.dumps({k: v for k, v in rep.items() if k != "multi_mode_estimates"}, indent=1))
if len(sys.argv) > 2: Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
