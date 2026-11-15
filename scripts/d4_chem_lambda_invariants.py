#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Invariant lower bound on the chemistry two-body one-norm.

Per channel c (a group/rank pair): l1_c = ||w_c||_1, t_c = sum_b w_b,c = tr X_c, iota_c = W_c + t_c (the identity
coefficient of the generator), and lambda_h2 = 1/4 sum_c (l1_c + |iota_c|)^2 (official formula).
Since l1_c >= |t_c| and sum_c |t_c||iota_c| >= |<t, iota>|,
    lambda_h2 >= 1/4 (|t|^2 + 2 |<t, iota>| + |iota|^2),
and the three quantities are invariant under any orthogonal re-mixing of the channels. |t|^2 = vec(I)^T T' vec(I) is
fixed by the organisers' target; <t, W> = tr(sum_c W_c X_c) and |W|^2 are set by the one-body term and the SOS energy.
Run with a Python that can import kickmix (the official verifiers are needed too):
    $QHDL_KICKMIX_PYTHON -I scripts/d4_chem_lambda_invariants.py FAMILY_DIR [OUT.json]
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

N = 76
inp = V.load_inputs(Path(sys.argv[1]) / "submission.npz")
w = np.asarray(inp.w, dtype=np.float64); W = np.asarray(inp.W, dtype=np.float64)   # w: (R, B, C); W: (R, C)
l1 = np.abs(w).sum(axis=1).ravel(); t = w.sum(axis=1).ravel(); Wv = W.ravel(); iota = Wv + t
lam2 = DM.lambda_h2(inp.w, inp.W)
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
d = np.arange(N); T[:, :, d, d] += 0.5 * S[:, :, None]; T[d, d, :, :] += 0.5 * S[None, :, :]
ITI = float(T[d][:, d][:, :, d, d].sum()) if False else float(np.einsum('ppqq->', T))
rep = dict(
    lambda_h2=lam2, channels=int(len(t)),
    quarter_sum_l1_sq=float(0.25 * (l1 ** 2).sum()), quarter_sum_trace_sq=float(0.25 * (t ** 2).sum()),
    quarter_target_identity_form=ITI / 4,
    fraction_of_weights_negative=float((w < 0).sum() / w.size), l1_over_abs_trace=float(l1.sum() / np.abs(t).sum()),
    quarter_sum_iota_sq=float(0.25 * (iota ** 2).sum()), half_sum_l1_abs_iota=float(0.5 * (l1 * np.abs(iota)).sum()),
    half_abs_inner_t_iota=float(0.5 * abs(np.dot(t, iota))),
    cosine_t_iota=float(np.dot(t, iota) / (np.linalg.norm(t) * np.linalg.norm(iota))),
    mean_iota_over_l1=float(np.abs(iota).sum() / l1.sum()),
    sum_W_sq_half=float(0.5 * (Wv ** 2).sum()), inner_t_W=float(np.dot(t, Wv)),
    invariant_lower_bound=float(0.25 * ((t ** 2).sum() + 2 * abs(np.dot(t, iota)) + (iota ** 2).sum())),
    electrons=int(inp.ref["electrons"]), filling_per_orbital=float(inp.ref["electrons"] / N))
rep["slack_above_invariant_bound_pct"] = 100 * (lam2 / rep["invariant_lower_bound"] - 1)
print(json.dumps(rep, indent=1))
if len(sys.argv) > 2: Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
