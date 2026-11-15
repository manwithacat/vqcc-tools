#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""A cold start for the chemistry two-body fit, built from the target's own eigenstructure (not from a public family).

T' (N^2 x N^2) = sum_k lam_k g_k g_k^T. The top R*C generators sqrt(lam_k) g_k (N x N, symmetrised) are dealt to
groups round-robin (generator k -> group k % R, column k // R). Each group's B unit vectors are the eigenvectors of its
generators with the largest |eigenvalue| (near-duplicates skipped), and the weights solve each generator's least
squares in the span of {u_b u_b^T}. Writes OUT.state.npz in the layout scripts/d4_chem_min_error_free.py resumes from
(x = [u (R,B,N), w (R,B,C)]), plus OUT.init.json.
Run (a Python that can import kickmix): $QHDL_KICKMIX_PYTHON -I scripts/d10_cold_init.py FAMILY_DIR OUT_STEM R B C
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402
CFG.add_verifiers_path()  # the official verifiers (not shipped): QHDL_OFFICIAL_ROOT
from dfthcsos import verify as V                      # noqa: E402

N = 76
fam, stem = Path(sys.argv[1]), Path(sys.argv[2]); R, B, C = (int(a) for a in sys.argv[3:6])
inp = V.load_inputs(fam / "submission.npz")
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
d = np.arange(N); T[:, :, d, d] += 0.5 * S[:, :, None]; T[d, d, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(N * N, N * N); Tm = 0.5 * (Tm + Tm.T)
lam, vec = np.linalg.eigh(Tm)
order = np.argsort(-lam)[:R * C]
gens = [0.5 * (g + g.T) for g in (np.sqrt(max(lam[k], 0.0)) * vec[:, k].reshape(N, N) for k in order)]
U = np.zeros((R, B, N)); W = np.zeros((R, B, C))
for r in range(R):
    gi = [r + R * c for c in range(C)]
    cand = []
    for c, k in enumerate(gi):
        ev, evec = np.linalg.eigh(gens[k])
        cand += [(abs(ev[i]), evec[:, i]) for i in range(N)]
    cand.sort(key=lambda t: -t[0])
    chosen = []
    for _, v in cand:
        if all(abs(v @ x) < 0.98 for x in chosen):
            chosen.append(v)
        if len(chosen) == B:
            break
    U[r] = np.array(chosen)
    A = np.einsum("bp,bq->pqb", U[r], U[r]).reshape(N * N, B)
    for c, k in enumerate(gi):
        W[r, :, c] = np.linalg.lstsq(A, gens[k].reshape(-1), rcond=None)[0]
X = np.einsum("rbk,rbc->rck", np.einsum("rbp,rbq->rbpq", U, U).reshape(R, B, N * N), W).reshape(R * C, N * N)
F = float(np.sum((X.T @ X - Tm) ** 2))
eckart = float(np.sqrt(np.sum(np.sort(lam)[::-1][R * C:] ** 2 * (np.sort(lam)[::-1][R * C:] > 0)) +
                       np.sum(lam[lam < 0] ** 2)))
np.savez(str(stem) + ".state.npz", x=np.concatenate([U.ravel(), W.ravel()]))
rep = dict(shape=[R, B, C], init_error2_mHa=1e3 * F ** 0.5, eckart_young_floor_mHa=1e3 * eckart,
           note="cold start from the target eigenstructure; no public-family vectors used")
print(json.dumps(rep)); Path(str(stem) + ".init.json").write_text(json.dumps(rep, indent=1))
