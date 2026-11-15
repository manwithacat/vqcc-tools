#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Weight-profiled discrete block search on the chemistry angle words.

Gauss-Newton model of the two-body residual D = X^T X - T' in 16 binary word moves plus the 44 weights of each
vector involved, with the weights eliminated by a damped Schur complement; all 2^16 move assignments are enumerated
in the surrogate, the best 8 are applied exactly and given an equal scaled weight refit, against two controls (the
unchanged state and the best single move) with the same refit.
Every residual column is a sum of symmetric outer products S(x, a) = x a^T + a x^T (x a row of X), so
<S(x,a), S(y,b)> = 2[(x.y)(a.b) + (x.b)(a.y)] and <D, S(x,a)> = 2 a^T D x: no 33M-row Jacobian is formed.
Run (a Python that can import kickmix): $QHDL_KICKMIX_PYTHON -I scripts/d11_chem_profiled_block.py FAMILY_DIR STATE.npz OUT_DIR [nvars] [refit_iters]
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

N, NN, SCALE = 76, 76 * 76, 1e6
fam, start, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NV = int(sys.argv[4]) if len(sys.argv) > 4 else 16
RI = int(sys.argv[5]) if len(sys.argv) > 5 else 150
out.mkdir(parents=False, exist_ok=False)
inp = V.load_inputs(fam / "submission.npz"); z = np.load(fam / "submission.npz"); edges = z["tree_edges"]
st = np.load(start); words0 = st["words"].astype(np.int64); w0 = st["w"].astype(np.float64)
alive = st["alive"].astype(bool); R, B, C = w0.shape
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
dg = np.arange(N); T[:, :, dg, dg] += 0.5 * S[:, :, None]; T[dg, dg, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(NN, NN)
mHa = lambda F: 1e3 * float(F) ** 0.5
log = open(out / "log.jsonl", "a")
def emit(**k):
    print(json.dumps(k), flush=True); log.write(json.dumps(k) + "\n"); log.flush()


def decode(rows):
    u = np.asarray(DM.reconstruct_u_vectors(rows / (1 << 7), edges), dtype=np.float64)
    return u / np.linalg.norm(u, axis=1, keepdims=True)
def proj(u): return np.einsum("kp,kq->kpq", u, u).reshape(len(u), NN)
def build(words, w):
    P = proj(decode(words[N:])).reshape(R, B, NN)
    X = np.einsum("rbk,rbc->rck", P, w).reshape(R * C, NN)
    return P, X, X.T @ X - Tm


def refit(P, w, iters):
    mask = alive[:, :, None].astype(float)
    def fun(x):
        ww = x.reshape(R, B, C); X = np.einsum("rbk,rbc->rck", P, ww).reshape(R * C, NN)
        D = X.T @ X - Tm; G = 4.0 * (X @ D); g = np.einsum("rbk,rck->rbc", P, G.reshape(R, C, NN)) * mask
        return SCALE * float(np.sum(D * D)), SCALE * g.ravel()
    res = minimize(fun, w.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=iters, maxcor=20, ftol=1e-12, gtol=1e-6))
    return res.x.reshape(R, B, C), res.fun / SCALE


def moves(row, words):
    cand = np.repeat(words[N + row][None, :], 150, axis=0)
    j = np.arange(75); cand[j, j] = (cand[j, j] + 1) & 255; cand[75 + j, j] = (cand[75 + j, j] - 1) & 255
    return cand


t0 = time.time()
P, X, D = build(words0, w0); F0 = float(np.sum(D * D))
emit(stage="start", error2_mHa=mHa(F0), state=str(start))

# --- variable selection: top-40 vectors by unit-vector gradient, best exact single move each, keep NV cheapest
u = decode(words0[N:]).reshape(R, B, N)
G4 = (4.0 * (X @ D)).reshape(R, C, N, N)
A = np.einsum("rbc,rcpq->rbpq", w0, G4); A = A + A.transpose(0, 1, 3, 2)
gh = np.einsum("rbpq,rbq->rbp", A, u); gu = gh - u * np.sum(gh * u, axis=2, keepdims=True)
score = np.linalg.norm(gu, axis=2) * alive
top = [divmod(int(k), B) for k in np.argsort(-score.ravel())[:40]]
del G4, A
cands = []
for (r, b) in top:
    row = r * B + b; Xr = X[r * C:(r + 1) * C]; wv = w0[r, b]; s = wv @ wv; y = wv @ Xr
    cm = moves(row, words0); delta = proj(decode(cm)) - P[r, b]
    Gd = delta @ D; dd = np.einsum("kn,kn->k", delta, delta); yd = delta @ y
    dF = 4 * (Gd @ y) + 2 * s * np.einsum("kn,kn->k", Gd, delta) + 2 * (y @ y) * dd + 2 * yd ** 2 + s * s * dd ** 2 + 4 * s * yd * dd
    k = int(np.argmin(dF)); cands.append((float(dF[k]), r, b, k, cm[k], delta[k]))
cands.sort(key=lambda t: t[0]); sel = cands[:NV]
emit(stage="selected", vars=[dict(r=r, b=b, move=k, exact_single_dF_mHa2=1e6 * dF) for dF, r, b, k, _, _ in sel],
     seconds=round(time.time() - t0, 1))

# --- Gauss-Newton model over [moves (NV); weights of the NV vectors (NV*C)]
avec = np.array([t[5] for t in sel] + [P[t[1], t[2]] for t in sel])          # a-vectors: d_i then P_i
XX = X @ X.T; XA = X @ avec.T; AA = avec @ avec.T; ADX = avec @ (D @ X.T)        # (352,352),(352,2NV),(2NV,2NV),(2NV,352)
cols = []                                                                      # (rows, a-index, coeffs)
for i, (_, r, b, _, _, _) in enumerate(sel):
    cols.append((r * C + np.arange(C), np.full(C, i), w0[r, b].copy()))
for i, (_, r, b, _, _, _) in enumerate(sel):
    for c in range(C):
        cols.append((np.array([r * C + c]), np.array([NV + i]), np.array([1.0])))
m = len(cols); H = np.zeros((m, m)); g = np.zeros(m)
for j, (Rj, Aj, Cj) in enumerate(cols):
    g[j] = 2.0 * np.sum(Cj * ADX[Aj, Rj])
    for l in range(j, m):
        Rl, Al, Cl = cols[l]
        M = XX[np.ix_(Rj, Rl)] * AA[np.ix_(Aj, Al)] + XA[np.ix_(Rj, Al)] * XA[np.ix_(Rl, Aj)].T
        H[j, l] = H[l, j] = 2.0 * (Cj @ M @ Cl)
q, wv_ = slice(0, NV), slice(NV, m)
pred_single = 2 * g[:NV] + np.diag(H)[:NV]
exact_single = np.array([t[0] for t in sel])
emit(stage="surrogate_check", pred_single_mHa2=(1e6 * pred_single).round(4).tolist(),
     exact_single_mHa2=(1e6 * exact_single).round(4).tolist(),
     sign_agree=int(np.sum(np.sign(pred_single) == np.sign(exact_single))),
     corr=float(np.corrcoef(pred_single, exact_single)[0, 1]))
Hww = H[wv_, wv_]; mu = 1e-6 * np.trace(Hww) / Hww.shape[0]
K = np.linalg.solve(Hww + mu * np.eye(Hww.shape[0]), np.column_stack([H[wv_, q], g[wv_]]))
Heff = H[q, q] - H[q, wv_] @ K[:, :NV]; geff = g[q] - H[q, wv_] @ K[:, NV]
refit_only = -float(g[wv_] @ K[:, NV])                                         # surrogate gain of weights alone
Q = ((np.arange(2 ** NV)[:, None] >> np.arange(NV)) & 1).astype(float)
vals = 2 * Q @ geff + np.einsum("ki,ij,kj->k", Q, Heff, Q)
order = np.argsort(vals)[:8]
emit(stage="enumerated", mu=mu, refit_only_surrogate_mHa2=1e6 * refit_only,
     best_surrogate_mHa2=(1e6 * vals[order]).round(4).tolist(),
     best_sets=[[int(i) for i in np.flatnonzero(Q[k])] for k in order], seconds=round(time.time() - t0, 1))

# --- exact checks with equal refits; controls first
def evaluate(label, idxs):
    words = words0.copy()
    for i in idxs:
        _, r, b, _, row, _ = sel[i]; words[N + r * B + b] = row
    Pn, _, Dn = build(words, w0); Fpre = float(np.sum(Dn * Dn))
    wn, Fpost = refit(Pn, w0, RI)
    emit(stage="exact", label=label, moves=[int(i) for i in idxs], before_refit_mHa=mHa(Fpre), after_refit_mHa=mHa(Fpost))
    return Fpost, words, wn
Fc0, _, _ = evaluate("control_unchanged", [])
best1 = int(np.argmin(exact_single))
Fc1, _, _ = evaluate("control_best_single", [best1])
ctrl = min(Fc0, Fc1); best = (ctrl, None, None, None)
for rank, k in enumerate(order):
    idxs = [int(i) for i in np.flatnonzero(Q[k])]
    if not idxs:
        continue
    Fk, wk, wwk = evaluate(f"block_rank{rank}", idxs)
    if Fk < best[0]:
        best = (Fk, idxs, wk, wwk)
gain = mHa(ctrl) - mHa(best[0])
emit(stage="verdict", control_mHa=mHa(ctrl), best_mHa=mHa(best[0]), gain_mHa=gain, gate_0p1=bool(gain >= 0.1),
     seconds=round(time.time() - t0, 1))
if best[1] is not None:
    np.savez_compressed(out / "best_state.npz", words=best[2], w=best[3], sweep=0, alive=alive.astype(np.int8))
