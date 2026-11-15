#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Thaw a few survivors of a discrete chemistry state, re-round them, polish with single-word moves.

Starts from a state written by scripts/d6_chem_word_search.py (words, w, sweep, alive). Each round: pick the M alive
vectors with the largest unit-vector gradient, optimise those vectors continuously plus all alive weights (others
pinned), round them back to 8-bit words through the exact inverse Givens tree (an independent inverse, checked < 1e-12), refit
weights, then run the unchanged word-search engine (d6_chem_word_search.py) for POLISH sweeps. A round is kept only if it lowers error_2.
Run (a Python that can import kickmix): $QHDL_KICKMIX_PYTHON -I scripts/d9_chem_thaw.py FAMILY_DIR START_STATE.npz OUT_DIR [rounds] [M] [thaw_iters] [polish]
"""
import json, subprocess, sys, time
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
fam, start, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 8
M = int(sys.argv[5]) if len(sys.argv) > 5 else 24
thaw_iters = int(sys.argv[6]) if len(sys.argv) > 6 else 300
polish = int(sys.argv[7]) if len(sys.argv) > 7 else 2
out.mkdir(parents=False, exist_ok=False)
inp = V.load_inputs(fam / "submission.npz"); z = np.load(fam / "submission.npz"); edges = z["tree_edges"]
st = np.load(start); words = st["words"].astype(np.int64).copy(); w = st["w"].copy(); alive = st["alive"].astype(bool)
R, B, C = w.shape
T = np.asarray(inp.ref["tei"], dtype=np.float64).copy(); S = np.asarray(inp.S, dtype=np.float64)
dg = np.arange(N); T[:, :, dg, dg] += 0.5 * S[:, :, None]; T[dg, dg, :, :] += 0.5 * S[None, :, :]
Tm = T.reshape(N * N, N * N)
mHa = lambda F: 1e3 * float(F) ** 0.5
SCALE = 1e6


def decode(rows):
    u = np.asarray(DM.reconstruct_u_vectors(rows / (1 << 7), edges), dtype=np.float64)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def proj(u): return np.einsum("kp,kq->kpq", u, u).reshape(len(u), N * N)


def to_words(u):
    """Exact inverse Givens tree then nearest 8-bit words; returns (words, roundtrip error)."""
    v = u.copy(); ang = np.zeros((len(v), len(edges)))
    for j in reversed(range(len(edges))):
        pa, ch = edges[j]; ang[:, j] = np.arctan2(v[:, ch], v[:, pa]) / np.pi
        v[:, pa] = np.hypot(v[:, pa], v[:, ch]); v[:, ch] = 0
    err = float(np.max(np.abs(np.asarray(DM.reconstruct_u_vectors(ang, edges)) - u)))
    return (np.rint(ang * 128).astype(np.int64) % 256), err


def error2(words, w):
    P = proj(decode(words[N:])).reshape(R, B, N * N)
    X = np.einsum("rbk,rbc->rck", P, w).reshape(R * C, N * N)
    D = X.T @ X - Tm
    return float(np.sum(D * D)), P, X, D


def refit(P, w, iters=120):
    mask = alive[:, :, None].astype(float)
    def fun(x):
        ww = x.reshape(R, B, C); X = np.einsum("rbk,rbc->rck", P, ww).reshape(R * C, N * N)
        D = X.T @ X - Tm; G = 4.0 * (X @ D); g = np.einsum("rbk,rck->rbc", P, G.reshape(R, C, N * N)) * mask
        return SCALE * float(np.sum(D * D)), SCALE * g.ravel()
    res = minimize(fun, w.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=iters, maxcor=20, ftol=1e-12, gtol=1e-6))
    return res.x.reshape(R, B, C)


def select(P, X, D, w):
    """Unit-vector gradient norm of every alive vector under the current residual."""
    u = decode(words[N:]).reshape(R, B, N)
    G = (4.0 * (X @ D)).reshape(R, C, N, N)
    A = np.einsum("rbc,rcpq->rbpq", w, G); A = A + A.transpose(0, 1, 3, 2)
    gh = np.einsum("rbpq,rbq->rbp", A, u)
    gu = gh - u * np.sum(gh * u, axis=2, keepdims=True)
    score = np.linalg.norm(gu, axis=2) * alive * ~TRIED
    flat = np.argsort(-score.ravel())[:M]
    return [divmod(int(k), B) for k in flat], score


CHECKED = []


def thaw(sel, w):
    """Optimise the selected unit vectors and all alive weights; everything else pinned."""
    u_all = decode(words[N:]).reshape(R, B, N); P0 = proj(u_all.reshape(-1, N)).reshape(R, B, N * N)
    idx = np.array(sel); mask = alive[:, :, None].astype(float)
    x0 = np.concatenate([u_all[idx[:, 0], idx[:, 1]].ravel(), w.ravel()]); nu = M * N
    def fun(x):
        uraw = x[:nu].reshape(M, N); nrm = np.linalg.norm(uraw, axis=1, keepdims=True); us = uraw / nrm
        ww = x[nu:].reshape(R, B, C); P = P0.copy(); P[idx[:, 0], idx[:, 1]] = proj(us)
        X = np.einsum("rbk,rbc->rck", P, ww).reshape(R * C, N * N); D = X.T @ X - Tm
        G = (4.0 * (X @ D)).reshape(R, C, N * N)
        gw = np.einsum("rbk,rck->rbc", P, G) * mask
        Gs = G.reshape(R, C, N, N)
        A = np.einsum("kc,kcpq->kpq", ww[idx[:, 0], idx[:, 1]], Gs[idx[:, 0]]); A = A + A.transpose(0, 2, 1)
        gh = np.einsum("kpq,kq->kp", A, us); gu = (gh - us * np.sum(gh * us, axis=1, keepdims=True)) / nrm
        return SCALE * float(np.sum(D * D)), SCALE * np.concatenate([gu.ravel(), gw.ravel()])
    if not CHECKED:                                    # finite-difference check of the gradient, first round only
        f0, g0 = fun(x0); rng = np.random.default_rng(3)
        for i in list(rng.choice(nu, 2, replace=False)) + list(nu + rng.choice(len(x0) - nu, 2, replace=False)):
            e = np.zeros_like(x0); e[i] = 1e-6
            print(json.dumps(dict(grad_check=int(i), fd=(fun(x0 + e)[0] - fun(x0 - e)[0]) / 2e-6, analytic=float(g0[i]))), flush=True)
        CHECKED.append(1)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=thaw_iters, maxcor=30, ftol=1e-12, gtol=1e-6))
    us = res.x[:nu].reshape(M, N); us = us / np.linalg.norm(us, axis=1, keepdims=True)
    return us, res.x[nu:].reshape(R, B, C), res.fun / SCALE


def polish_d6(words, w, k):
    d = out / f"round{k:02d}"; d.mkdir()
    s = d / "start.npz"; np.savez_compressed(s, words=words, w=w, sweep=0, alive=alive.astype(np.int8))
    subprocess.run([sys.executable, "-I", str(ROOT / "scripts/d6_chem_word_search.py"), str(fam), str(d / "d6"),
                    str(polish), "120", str(s)], check=True, stdout=open(d / "d6.log", "w"), stderr=subprocess.STDOUT)
    last = sorted((d / "d6").glob("state_sweep*.npz"))[-1]; e = np.load(last)
    return e["words"].astype(np.int64), e["w"]


TRIED = np.zeros((R, B), bool)                         # vectors of a rejected round are not selected again
F, P, X, D = error2(words, w)
best = F
log = open(out / "log.jsonl", "a")
print(json.dumps(dict(start=str(start), start_mHa=mHa(F), M=M, thaw_iters=thaw_iters, polish=polish)), flush=True)
for k in range(1, rounds + 1):
    t0 = time.time()
    sel, score = select(P, X, D, w)
    us, w_t, F_cont = thaw(sel, w)
    new_rows, err = to_words(us)
    assert err < 1e-12, err
    words2 = words.copy()
    for (r, b), row in zip(sel, new_rows):
        words2[N + r * B + b] = row
    F_round, P2, _, _ = error2(words2, w_t)
    w2 = refit(P2, w_t)
    F_refit = error2(words2, w2)[0]
    words3, w3 = polish_d6(words2, w2, k)
    F3, P3, X3, D3 = error2(words3, w3)
    keep = F3 < best
    rec = dict(round=k, selected=[list(x) for x in sel], score_top=float(score.max()), before_mHa=mHa(best),
               thawed_continuous_mHa=mHa(F_cont), rerounded_mHa=mHa(F_round), rerounded_refit_mHa=mHa(F_refit),
               polished_mHa=mHa(F3), kept=bool(keep), roundtrip_err=err, seconds=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if not keep:
        for r, b in sel: TRIED[r, b] = True
    if keep:
        words, w, best = words3, w3, F3
        P, X, D = P3, X3, D3
        np.savez_compressed(out / f"best_round{k:02d}.npz", words=words, w=w, sweep=k, alive=alive.astype(np.int8))
print(json.dumps(dict(final_mHa=mHa(best))), flush=True)
