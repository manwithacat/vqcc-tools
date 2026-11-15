#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""How a chemistry family's LCU one-norm Lambda splits, and rigorous lower bounds on its two-body part.

Lambda = lambda_h1 + lambda_h2, lambda_h2 = 1/4 sum_rc (||w_rc||_1 + |w_B,rc|)^2 (official dfthc_math.lambda_h2).
With X_rc = sum_b w_b u_b u_b^T (unit u_b), ||w_rc||_1 >= ||X_rc||_* >= ||X_rc||_F, and the accepted factorization
satisfies sum_rc X_rc (x) X_rc = T' up to 20 mHa, where T' = tei + (S (x) I + I (x) S)/2 is fixed by the organisers
(the shift S is replaced by the baseline's). For any symmetric PSD M on pairs,
    <M, T'> = sum_rc vec(X_rc)^T M vec(X_rc) <= mu(M) sum_rc ||X_rc||_*^2,   mu(M) = max_{|u|=1} M[uu,uu] <= sigma_max(M),
so lambda_h2 >= <M, T'> / (4 sigma_max(M)). M = I gives lambda_h2 >= Tr(T')/4; M = T' gives ||T'||_F^2/(4 sigma_max).
Run with a Python that can import kickmix (the official verifiers are needed too):
    $QHDL_KICKMIX_PYTHON -I scripts/d4_chem_lambda_bounds.py FAMILY_DIR [OUT.json]
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402
CFG.add_verifiers_path()  # the official verifiers (not shipped): QHDL_OFFICIAL_ROOT
from dfthcsos import verify as V                      # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as DM      # noqa: E402

fam = Path(sys.argv[1])
inp = V.load_inputs(fam / "submission.npz")
ref = inp.ref
tei = np.asarray(ref["tei"], dtype=np.float64) if "tei" in ref else None
N = 76
lam2 = DM.lambda_h2(inp.w, inp.W)
w = np.asarray(inp.w, dtype=np.float64); W = np.asarray(inp.W, dtype=np.float64)
l1 = np.linalg.norm(w, ord=1, axis=1); wB = np.abs(W + w.sum(axis=1))
rep = dict(family=fam.name, shapes=dict(w=list(w.shape), W=list(W.shape)), lambda_h2=lam2,
           lambda_h2_from_l1_only=float(0.25 * np.sum(l1 ** 2)), lambda_h2_identity_term_share=float(1 - 0.25 * np.sum(l1 ** 2) / lam2),
           ref_keys=sorted(ref.keys()) if isinstance(ref, dict) else None)
S = np.asarray(inp.S, dtype=np.float64)
if tei is not None:
    T = tei.copy()
    d = np.arange(N)
    T[:, :, d, d] += 0.5 * S[:, :, None]
    T[d, d, :, :] += 0.5 * S[None, :, :]
    Mx = T.reshape(N * N, N * N); Mx = 0.5 * (Mx + Mx.T)
    ev = np.linalg.eigvalsh(Mx)
    rep.update(target_trace=float(np.trace(Mx)), target_fro=float(np.linalg.norm(Mx)), target_sigma_max=float(ev[-1]),
               target_min_eig=float(ev[0]), target_negative_mass=float(-ev[ev < 0].sum()),
               bound_M_identity=float(np.trace(Mx) / 4), bound_M_target=float(np.linalg.norm(Mx) ** 2 / (4 * ev[-1])),
               nuclear_norm_target_over_4=float(np.abs(ev).sum() / 4))
print(json.dumps(rep, indent=1))
if len(sys.argv) > 2: Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
