#!/usr/bin/env python3
"""Product-span AND minimiser: multi-output synthesis over the reachable-state hull.

Setting.  All functions live on N <= 1024 points (the distinct reachable states of the available values; a `mask`
marks the valid points).  U is the span of the available values and the constant.  A *product* is L*M with L, M in the
current operand space V (a subspace of functions).  A synthesis is a sequence of steps; step s adds the products
P_1..P_k, after which one nonzero combination g_s of the still-missing targets satisfies
        g_s = sum_j L_j * M_j + u,     L_j, M_j, u in V_s,
and g_s joins V_{s+1} (it is now a live wire).  Cost = total number of products (Toffoli-like gates).
Sharing between targets is free: the target wires hold g_1..g_n and a CX network turns them into the targets at the
end.  Helper products (cached in scratch wires) enlarge V and are measured away again (HMR + CZ of their operand
forms; free when the operands are still live forms).

Search.  For k products the first k-1 factors are random elements of V; the last one is enumerated in Gray-code order
by the C kernel (qhdl_productspan.c): the products L*u_i modulo Q = U + sum_j L_j V form a linear map, and the step
succeeds when a combination of the missing targets lies in its column span.  This is the "fix-first-factors" method of
an earlier invsqrt MinRank search with two additions: the combination freedom (2^r - 1 target combinations are
tested at once) and the growth of V with every finished target.  Every hit is re-solved and checked exactly in Python.
Evidence tier of a result: static construction, exact on the stated point set; a fragment is a separate exact check.
"""
from __future__ import annotations

import ctypes
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
M64 = (1 << 64) - 1


# ------------------------------------------------------------------ library
def _lib(nw=4):
    work = Path(os.environ.get('QHDL_WORK', ROOT / 'work')) / 'productspan'
    work.mkdir(parents=True, exist_ok=True)
    so = work / ('libqhdl_productspan.so' if nw == 4 else f'libqhdl_productspan_w{nw}.so')
    src = HERE / 'qhdl_productspan.c'
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(['cc', '-O3', '-march=native', f'-DNW={nw}', '-shared', '-fPIC', '-o', str(so), str(src)],
                       check=True)
    lib = ctypes.CDLL(str(so))
    lib.ps_search.restype = ctypes.c_int
    lib.ps_search.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p]
    lib.ps_search2.restype = ctypes.c_int
    lib.ps_search2.argtypes = lib.ps_search.argtypes + [ctypes.c_int]
    return lib


_LIBS = {}
MAX_POINTS = 1024      # widest kernel: NW = 16 words of 64 bits


def _nw_for(*vecsets):
    """Words per vector: 4 (N <= 256, the original kernel) or 16 (N <= 1024)."""
    bits = max([v.bit_length() for vs in vecsets for v in vs] + [1])
    if bits > MAX_POINTS:
        raise ValueError(f'function on {bits} points exceeds the {MAX_POINTS}-point kernel limit')
    return 4 if bits <= 256 else 16


def words(vs, nw=4):
    arr = (ctypes.c_uint64 * (nw * max(1, len(vs))))()
    for i, v in enumerate(vs):
        for t in range(nw):
            arr[nw * i + t] = (v >> (64 * t)) & M64
    return arr


def popc(x):
    return bin(x).count('1')


def xor_mask(vecs, mask):
    v = 0
    b = 0
    while mask:
        if mask & 1:
            v ^= vecs[b]
        mask >>= 1
        b += 1
    return v


# ------------------------------------------------------------------ exact linear algebra helpers
class Echelon:
    """GF(2) echelon basis on int vectors with optional label tracking (int bitmask of source labels)."""

    def __init__(self):
        self.b = {}

    def reduce(self, v, lab=0):
        while v:
            h = v.bit_length() - 1
            e = self.b.get(h)
            if e is None:
                return v, lab
            v ^= e[0]
            lab ^= e[1]
        return 0, lab

    def add(self, v, lab=0):
        v, lab = self.reduce(v, lab)
        if v:
            self.b[v.bit_length() - 1] = (v, lab)
            return True
        return False


def independent_basis(vecs):
    """Greedy independent sub-list (indices) of vecs."""
    ech = Echelon()
    return [i for i, v in enumerate(vecs) if ech.add(v)]


def solve_step(V, Ls, R, force_last=False):
    """Given operand basis V (list of int functions), k factor functions Ls (elements of span V) and missing targets R,
    find a nonzero combination c of R, masks M_j and u over V with  sum_i c_i R_i = sum_j L_j*M_j + u.
    -> (cmask, [M_j masks], umask) or None. Exact."""
    D = len(V)
    k = len(Ls)
    ech = Echelon()
    lab = 0
    for j in range(k):
        for b in range(D):
            ech.add(Ls[j] & V[b], 1 << (j * D + b))
    base = k * D
    for b in range(D):
        ech.add(V[b], 1 << (base + b))
    tb = base + D
    for i, t in enumerate(R):
        v, l = ech.reduce(t, 1 << (tb + i))
        if v == 0 and force_last and i < len(R) - 1:
            continue                    # a dependency among the earlier targets: skip, the last must be in the combination
        if v == 0:
            # l = combination of columns and targets whose XOR is zero:  sum(cols) + sum(c_i t_i) = 0
            cmask = (l >> tb) & ((1 << len(R)) - 1)
            Ms = [(l >> (j * D)) & ((1 << D) - 1) for j in range(k)]
            u = (l >> base) & ((1 << D) - 1)
            return cmask, Ms, u
        ech.b[v.bit_length() - 1] = (v, l)
    return None


def check_step(V, Ls, Ms, u, R, cmask):
    acc = xor_mask(V, u)
    for L, m in zip(Ls, Ms):
        acc ^= L & xor_mask(V, m)
    tgt = 0
    for i, t in enumerate(R):
        if cmask >> i & 1:
            tgt ^= t
    return acc == tgt and cmask != 0


# ------------------------------------------------------------------ one step
def find_step(V, R, k, rng, max_iter=1 << 22, tries=50, deadline=None, log=None, force_last=False):
    """Random-restart search for k products capturing a combination of the missing targets R.
    -> dict(cmask, Ls (functions), Lmasks, Ms, u, tests, tries) or None."""
    nw = _nw_for(V, R)
    if nw not in _LIBS:
        _LIBS[nw] = _lib(nw)
    lib = _LIBS[nw]
    D = len(V)
    Uw = words(V, nw)
    Rw = words(R, nw)
    total = 0
    for t in range(tries):
        if deadline and time.time() > deadline:
            break
        masks = [rng.getrandbits(D) or 1 for _ in range(k - 1)]
        Ls = [xor_mask(V, m) for m in masks]
        Lw = words(Ls, nw)
        out = ctypes.c_uint64(0)
        done = ctypes.c_uint64(0)
        qd = ctypes.c_int(0)
        start = rng.getrandbits(D)
        rc = lib.ps_search2(D, Uw, len(R), Rw, k - 1, Lw, start, max_iter, ctypes.byref(out), ctypes.byref(done),
                             ctypes.byref(qd), len(R) - 1 if force_last else 0)
        total += done.value
        if rc == 1:
            Llast = xor_mask(V, out.value)
            LL = Ls + [Llast]
            sol = solve_step(V, LL, R, force_last)
            if sol is None:
                continue
            cmask, Ms, u = sol
            assert check_step(V, LL, Ms, u, R, cmask)
            assert not force_last or (cmask >> (len(R) - 1)) & 1
            return dict(cmask=cmask, Ls=LL, Lmasks=masks + [out.value], Ms=Ms, u=u, tests=total, tries=t + 1,
                        qdim=qd.value)
    if log:
        log(f'  k={k}: no hit in {total} tests')
    return None


# ------------------------------------------------------------------ operand forms and product emission
def _wires(mask, Vw):
    cst = False
    ws = []
    b = 0
    while mask:
        if mask & 1:
            if Vw[b] is None:
                cst = True
            else:
                ws.append(Vw[b])
        mask >>= 1
        b += 1
    return ws, cst


def emit_product(A, cA, B, cB, pw):
    """Ops adding (A + cA)*(B + cB) (A, B sets of wires, cA/cB constants) into wire pw; the operand carriers are built
    in place with CX and undone, so no scratch is used.  -> (ops, n_ccx)."""
    A, B = list(dict.fromkeys(A)), list(dict.fromkeys(B))
    ops = []
    ccx = 0
    sa, sb = set(A), set(B)
    pure = None
    extra_lin = []
    if sa and sb:
        if sa == sb:
            extra_lin = A
        elif sa <= sb:
            extra_lin = A
            pure = (A, [w for w in B if w not in sa])
        elif sb <= sa:
            extra_lin = B
            pure = ([w for w in A if w not in sb], B)
        else:
            pure = (A, B)
    if pure:
        A2, B2 = pure
        s2a, s2b = set(A2), set(B2)
        a = next(w for w in A2 if w not in s2b)
        b = next(w for w in B2 if w not in s2a)
        pre = [f'CX q{w} q{a}' for w in A2 if w != a] + [f'CX q{w} q{b}' for w in B2 if w != b]
        ops += pre + [f'CCX q{a} q{b} q{pw}'] + pre[::-1]
        ccx = 1
    ops += [f'CX q{w} q{pw}' for w in extra_lin]
    if cB:
        ops += [f'CX q{w} q{pw}' for w in A]
    if cA:
        ops += [f'CX q{w} q{pw}' for w in B]
    if cA and cB:
        ops.append(f'X q{pw}')
    return ops, ccx


# ------------------------------------------------------------------ parallel step search
def _worker(a):
    V, R, k, iters, tries, seed, deadline, fl = a
    return find_step(V, R, k, random.Random(seed), iters, tries, deadline, None, fl)


def find_step_par(V, R, k, rng, iters, tries, nproc=1, deadline=None, log=None, force_last=False):
    if nproc <= 1:
        return find_step(V, R, k, rng, iters, tries, deadline, log, force_last)
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    per = max(1, tries // nproc)
    jobs = [(V, R, k, iters, 1, rng.getrandbits(62), deadline, force_last) for _ in range(tries)]
    with ctx.Pool(nproc) as pool:
        tot = 0
        for res in pool.imap_unordered(_worker, jobs):
            if res is not None:
                pool.terminate()
                return res
    if log:
        log(f'  k={k}: no hit in {tries} tries x {iters} tests')
    return None


# ------------------------------------------------------------------ the synthesis (helpers, width, steps)
def _forms(mask, Vw):
    ws, c = _wires(mask, Vw)
    return ws, c


def random_helper(V, Vw, rng, wt=3):
    """A random product of two low-weight forms over the current operand space -> (Lmask, Mmask) or None."""
    D = len(V)
    for _ in range(200):
        def rm():
            m = 0
            for b in rng.sample(range(D), min(D, rng.randint(1, wt))):
                m |= 1 << b
            return m
        L, M = rm(), rm()
        p = xor_mask(V, L) & xor_mask(V, M)
        e = Echelon()
        for v in V:
            e.add(v)
        if e.add(p):
            return L, M, p
    return None


def synthesise(U, Uw, targets, tw, extra=0, helpers=0, rng=None, policy=None, nproc=1, log=print, deadline=None,
               helper_fn=random_helper):
    """Sequential product-span synthesis with optional helper products and a wire budget.

    U, Uw        operand basis (ints) and its wires (None for the constant);
    targets, tw  target functions and their wires;
    extra        wires available beyond the target wires; helper wires and started targets must fit in
                 len(targets) + extra at all times (helpers sitting on not-yet-started target wires count as free);
    helpers      number of helper products built first (each 1 product, removed again by HMR + CZ, 0 products);
    policy       callable (D, r, step) -> list of (k, max_iter, tries) to try in order.
    -> dict(events, missing, products, helper_products, scratch_peak, pool)."""
    rng = rng or random.Random(1)
    V, Vw = list(U), list(Uw)
    n = len(targets)
    pool = n + extra
    events = []
    live = []                               # live helper events (stack)
    R = list(targets)
    orig = list(range(n))
    done = 0
    products = 0
    peak = 0
    next_h = 5000
    for _ in range(helpers):
        h = helper_fn(V, Vw, rng)
        if h is None:
            break
        Lm, Mm, p = h
        ev = dict(type='helper', wire=next_h, Lmask=Lm, Mmask=Mm, Vw=list(Vw), fn=p)
        next_h += 1
        events.append(ev)
        live.append(ev)
        V.append(p)
        Vw.append(ev['wire'])
        products += 1
        peak = max(peak, len(live))
    s = 0
    while R:
        r = len(R)
        while len(live) + done + 1 > pool:
            ev = live.pop()
            j = Vw.index(ev['wire'])
            V.pop(j)
            Vw.pop(j)
            events.append(dict(type='remove', helper=ev))
        peak = max(peak, len(live) + done + 1)
        got = None
        ladder = policy(len(V), r, s) if policy else [(5, 1 << 20, 20)]
        late = bool(deadline and time.time() > deadline)
        if late:
            ladder = ladder[-3:]                # past the time cap: only the cheap fallbacks, no more optimism
        for k, iters, tries in ladder:
            got = find_step_par(V, R, k, rng, iters, tries, nproc, None if late else deadline, log)
            if got:
                got['k'] = k
                break
        if not got:
            return dict(events=events, missing=list(orig), products=products, helper_products=len(
                [e for e in events if e['type'] == 'helper']), pool=pool, pool_peak=peak, scratch_peak=max(0, peak - n), done=done)
        cmask = got['cmask']
        g = 0
        for i in range(r):
            if cmask >> i & 1:
                g ^= R[i]
        piv = (cmask & -cmask).bit_length() - 1
        got.update(type='step', V=list(V), Vw=list(Vw), g=g, pivot=piv, combo=[i for i in range(r) if cmask >> i & 1],
                   orig=list(orig))
        events.append(got)
        products += got['k']
        done += 1
        if log:
            log(f'step {s}: D={len(V)} r={r} k={got["k"]} helpers_live={len(live)} tests={got["tests"]}')
        Vn = V + [g]
        if independent_basis(Vn)[-1:] == [len(Vn) - 1]:
            V = Vn
            Vw.append(tw[orig[piv]])
        R = [t for i, t in enumerate(R) if i != piv]
        orig = [o for i, o in enumerate(orig) if i != piv]
        s += 1
        peak = max(peak, len(live) + done)
    while live:
        ev = live.pop()
        events.append(dict(type='remove', helper=ev))
    return dict(events=events, missing=[], products=products,
                helper_products=len([e for e in events if e['type'] == 'helper']), pool=pool,
                pool_peak=peak, scratch_peak=max(0, peak - n), done=done)


def emit_events(events, tw, state=None, hmr_bit0=9000):
    """Ops (kmx-style text) for a synthesis.  If `state` ({wire: numpy bool array}) is given the ops are applied as
    they are emitted and each helper removal is checked exactly (helper value == product of its operand forms read
    from the live wires).  -> (ops, n_ccx)."""
    import numpy as np
    ops = []
    ccx = 0
    bit = hmr_bit0

    def run(lines):
        if state is not None:
            apply_ops(lines, state)

    for ev in events:
        if ev['type'] == 'helper':
            A, cA = _wires(ev['Lmask'], ev['Vw'])
            B, cB = _wires(ev['Mmask'], ev['Vw'])
            ev['forms'] = (A, cA, B, cB)
            if state is not None:
                n = len(next(iter(state.values())))
                state[ev['wire']] = np.zeros(n, bool)
            o, c = emit_product(A, cA, B, cB, ev['wire'])
            ops += o
            ccx += c
            run(o)
        elif ev['type'] == 'remove':
            h = ev['helper']
            A, cA, B, cB = h['forms']
            if state is not None:
                va = np.full_like(state[h['wire']], cA)
                for w in A:
                    va = va ^ state[w]
                vb = np.full_like(state[h['wire']], cB)
                for w in B:
                    vb = vb ^ state[w]
                if not np.array_equal(state[h['wire']], va & vb):
                    raise AssertionError('helper %d is not the product of its operand forms at removal' % h['wire'])
                state[h['wire']] = np.zeros_like(state[h['wire']])
            b = bit
            bit += 1
            ops.append(f'HMR q{h["wire"]} b{b}')
            sa, sb = set(A), set(B)
            pure_a, pure_b = list(A), list(B)
            lin = []
            if sa and sb:
                if sa == sb:
                    lin, pure_a, pure_b = list(A), [], []
                elif sa <= sb:
                    lin, pure_b = list(A), [w for w in B if w not in sa]
                elif sb <= sa:
                    lin, pure_a = list(B), [w for w in A if w not in sb]
            else:
                pure_a, pure_b = [], []
            if pure_a and pure_b:
                a = next(w for w in pure_a if w not in set(pure_b))
                bb = next(w for w in pure_b if w not in set(pure_a))
                pre = [f'CX q{w} q{a}' for w in pure_a if w != a] + [f'CX q{w} q{bb}' for w in pure_b if w != bb]
                ops += pre + [f'CZ q{a} q{bb} if b{b}'] + pre[::-1]
            ops += [f'Z q{w} if b{b}' for w in lin]
            if cB:
                ops += [f'Z q{w} if b{b}' for w in A]
            if cA:
                ops += [f'Z q{w} if b{b}' for w in B]
            if cA and cB:
                ops.append(f'NEG if b{b}')      # (-1)^m: a global sign per shot, kept so the sign bit is exactly zero
        else:
            pw = tw[ev['orig'][ev['pivot']]]
            Vw = ev['Vw']
            for L, M in zip(ev['Lmasks'], ev['Ms']):
                A, cA = _wires(L, Vw)
                B, cB = _wires(M, Vw)
                o, c = emit_product(A, cA, B, cB, pw)
                ops += o
                ccx += c
                run(o)
            ws, cst = _wires(ev['u'], Vw)
            o = (['X q%d' % pw] if cst else []) + [f'CX q{w} q{pw}' for w in ws]
            ops += o
            run(o)
    steps = [e for e in events if e['type'] == 'step']
    fin = []
    for st in reversed(steps):
        pw = tw[st['orig'][st['pivot']]]
        for i in st['combo']:
            if i != st['pivot']:
                fin.append(f'CX q{tw[st["orig"][i]]} q{pw}')
    ops += fin
    run(fin)
    return ops, ccx


def apply_ops(ops, state):
    """Evaluate X/CX/CCX text ops on {wire: bool array}.  HMR zeroes its wire; conditional/phase lines are skipped."""
    for line in ops:
        t = line.split()
        if 'if' in t or t[0] in ('Z', 'CZ', 'HMR'):
            if t[0] == 'HMR':
                state[int(t[1][1:])] = state[int(t[1][1:])] & False
            continue
        w = [int(a[1:]) for a in t[1:]]
        if t[0] == 'X':
            state[w[0]] = ~state[w[0]]
        elif t[0] == 'CX':
            state[w[1]] = state[w[1]] ^ state[w[0]]
        elif t[0] == 'CCX':
            state[w[2]] = state[w[2]] ^ (state[w[0]] & state[w[1]])
        else:
            raise ValueError(line)


# ------------------------------------------------------------------ instance from simulated columns
LAST_FAIL = {}      # why the last instance_from_columns returned None


def instance_from_columns(avail, targets):
    """avail: list of (wire, bool column over all rows); targets: list of bool columns.  Collapse the rows to the
    distinct available-value states (<= 1024 points) and return the function vectors on those points:
    dict(N, U, Uw, T, rows_index) or None when there are more than 1024 states / a target is not determined."""
    import numpy as np
    n = len(targets[0])
    # distinct available-value states: pack the columns 64 per uint64 key word and take unique rows
    nk = max(1, (len(avail) + 63) // 64)
    key = np.zeros((n, nk), np.uint64)
    for j, (_, c) in enumerate(avail):
        key[:, j // 64] |= c.astype(np.uint64) << np.uint64(j % 64)
    u, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    N = len(u)
    LAST_FAIL.clear()
    LAST_FAIL.update(N=N, avail=len(avail))
    if N > MAX_POINTS:
        LAST_FAIL['why'] = f'{N} distinct available-value states exceed the {MAX_POINTS}-point kernel limit'
        return None
    for t in targets:
        if not np.array_equal(t, t[idx][inv]):
            LAST_FAIL['why'] = f'a target/old value is not determined by the {len(avail)} available values ({N} states)'
            return None
    ones = (1 << N) - 1
    vec = lambda c: sum(int(b) << i for i, b in enumerate(c[idx]))
    U = [ones] + [vec(c) for _, c in avail]
    Uw = [None] + [w for w, _ in avail]
    ind = independent_basis(U)
    return dict(N=N, U=[U[i] for i in ind], Uw=[Uw[i] for i in ind], T=[vec(t) for t in targets], dropped=len(U) - len(ind))


# ------------------------------------------------------------------ policy and the one-call entry point
def auto_policy(N, max_bits=24, per_try=1 << 22):
    """Which (k, max_iter, tries) to try at operand dimension D with r targets missing.  Counting model, calibrated on
    the 2026-11-09 runs (a hit needs about 2^e tests): a factor L*V adds about D-3.5 dimensions modulo the span, so k
    products leave N - D - k(D-3.5) dimensions uncovered and the r-fold combination freedom saves r bits:
    e(k) = N - D - k(D-3.5) - r.  Candidates are k with e(k) <= max_bits, smallest k first, each given 8 * 2^e tests (32 * 2^e for the first)
    (at most 2^(max_bits+3), at least 2^18); then the next larger k (which is nearly free) as the fallback."""
    def pol(D, r, s):
        def e(k):
            return N - D - k * (D - 3.5) - r
        ks = [k for k in range(1, 100) if e(k) <= max_bits]
        k0 = ks[0] if ks else 100
        out = []
        for k in (k0 - 1, k0, k0 + 1, k0 + 2, k0 + 4, k0 + 8, 2 * k0 + 4):
            if k < 1:
                continue
            bits = max(18, int(min(max_bits, e(k))) + (5 if k == k0 else 3))
            budget = 1 << bits
            if k < k0 and e(k) > max_bits:
                continue
            it = min(per_try, budget)
            out.append((k, it, max(1, budget // it)))
        return out
    return pol


def synth_from_columns(avail, targets, extra=0, helpers=0, seed=1, seconds=30, nproc=1, policy=None, log=None,
                       restarts=3, max_bits=24):
    """avail: [(wire, bool column)]; targets: [(wire, bool column)] (columns over the same rows, targets determined by
    avail).  Runs the product-span synthesis, emits the fragment and checks it exactly on every row.
    -> dict(ok, reason | and_cost, products, helper_products, scratch_peak, ops, ...)."""
    import numpy as np
    awires = [w for w, _ in avail]
    twires = [w for w, _ in targets]
    if set(awires) & set(twires):
        return dict(ok=False, reason='a target wire is also an available wire')
    inst = instance_from_columns(avail, [c for _, c in targets])
    if inst is None:
        return dict(ok=False, reason=LAST_FAIL.get('why', 'more than 1024 distinct available-value states (kernel limit) or not determined'))
    if len(inst['U']) > 64:
        return dict(ok=False, reason='operand space larger than 64 dimensions (kernel limit)')
    t0 = time.time()
    deadline = t0 + seconds if seconds else None
    res = None
    for attempt in range(max(1, restarts)):
        res = synthesise(inst['U'], inst['Uw'], inst['T'], twires, extra=extra, helpers=helpers,
                         rng=random.Random(seed + 1000 * attempt), policy=policy or auto_policy(inst['N'], max_bits),
                         nproc=nproc, log=log, deadline=deadline)
        if not res['missing'] or (deadline and time.time() > deadline):
            break
    out = dict(ok=False, N=inst['N'], operand_dim=len(inst['U']), seconds=round(time.time() - t0, 1),
               products=res['products'], helper_products=res['helper_products'], scratch_peak=res['scratch_peak'],
               pool=res['pool'], pool_peak=res['pool_peak'], missing=len(res['missing']))
    if res['missing']:
        out['reason'] = f'{len(res["missing"])} targets not captured within the search budget'
        return out
    n = len(targets[0][1])
    st = {w: c.copy() for w, c in avail}
    for w in twires:
        st[w] = np.zeros(n, bool)
    ops, ccx = emit_events(res['events'], twires, state=st)
    good = all(np.array_equal(st[w], c) for w, c in targets) and all(np.array_equal(st[w], c) for w, c in avail)
    out.update(ok=bool(good), and_cost=ccx, ops=ops, rows_checked=int(n),
               steps=[(e['k'], len(e['V']), len(e['combo'])) for e in res['events'] if e['type'] == 'step'])
    if not good:
        out['reason'] = 'fragment check failed'
    return out


def bitsim_check(ops, avail, targets, seeds=(11, 12, 13), target_init=None):
    """Independent re-check of a fragment on the project's bit-parallel simulator (qhdl_bitsim): wires are renumbered,
    the available wires are loaded with their columns, targets and helpers start at 0, and the fragment runs on every
    row under several random HMR outcome seeds.  Passes iff every target ends equal to its column, every available wire
    is unchanged, every helper wire is 0, and the accumulated sign is 0 on every row (the helper measure-aways leave no
    phase).  -> dict(ok, seeds, rows)."""
    import re
    import numpy as np
    import qhdl_bitsim as B
    wires = sorted({int(m) for line in ops for m in re.findall(r'\bq(\d+)', line)} |
                   {w for w, _ in avail} | {w for w, _ in targets})
    idx = {w: i for i, w in enumerate(wires)}
    text = [re.sub(r'\bq(\d+)', lambda m: 'q%d' % idx[int(m.group(1))], line) for line in ops]
    parsed = [B.parse_line(t) for t in text]
    n = len(avail[0][1]) if avail else len(targets[0][1])
    out = dict(ok=True, seeds=list(seeds), rows=int(n))
    for sd in seeds:
        sim = B.BitSim(parsed, rows=np.arange(1, n + 1), num_qubits=len(wires), x_qubits=[], outcome=sd)
        for w, c in avail:
            sim.q[idx[w]] = B.pack_bits(c)
        for w, c in (target_init or []):
            sim.q[idx[w]] = B.pack_bits(c)          # in-place model: the target wires start at their old values
        sim.run()
        for w, c in targets:
            out['ok'] &= bool(np.array_equal(B.unpack_bits(sim.q[idx[w]].copy(), n), c))
        for w, c in avail:
            out['ok'] &= bool(np.array_equal(B.unpack_bits(sim.q[idx[w]].copy(), n), c))
        for w in wires:
            if w >= 5000:
                out['ok'] &= not sim.q[idx[w]].any()
        out['ok'] &= not (sim.sign & sim.valid).any()
    return out


# ------------------------------------------------------------------ in-place update model
def synthesise_inplace(U, Uw, targets, tw, olds, order=None, rng=None, policy=None, nproc=1, log=print, deadline=None,
                       force_pivot=True):
    """Like `synthesise` (no helpers, no extra wires) but each target wire starts at an OLD value (olds[i], a function on the
    same points, or None for a wire that starts at 0) instead of 0, and the step *updates* the wire in place:
        wire <- old + sum L_j*M_j + u   must equal g,   i.e.  g + old = sum L_j*M_j + u  over the operand space.
    The operand space of a step is  U + finished g's + the old values of the OTHER unfinished targets (their wires still
    hold them); the pivot's own wire cannot be an operand of its own update (a Toffoli target cannot be a control).  The
    pivot is fixed before the search (order[0] of the unfinished) and forced into the combination with the kernel's minj
    option, with right-hand side t_pivot + old_pivot.  Old values are what they are: no restoration of them is claimed.
    -> same dict as `synthesise` (events usable by emit_events)."""
    rng = rng or random.Random(1)
    n = len(targets)
    order = list(order) if order is not None else list(range(n))
    fin_g, fin_w = [], []                # finished g functions and their wires
    unfinished = list(order)
    events = []
    products = 0
    s = 0
    while unfinished:
        p = unfinished[0]
        others = unfinished[1:]
        V = list(U) + list(fin_g)
        Vw = list(Uw) + list(fin_w)
        for j in others:
            if olds[j] is not None:
                V.append(olds[j])
                Vw.append(tw[j])
        # keep an independent operand basis (an old value may already lie in the span)
        ind = independent_basis(V)
        V = [V[i] for i in ind]
        Vw = [Vw[i] for i in ind]
        r = len(unfinished)
        # right-hand sides: the other targets as they are, the pivot as t_p + old_p, pivot last
        R = [targets[j] for j in others] + [targets[p] ^ (olds[p] or 0)]
        got = None
        ladder = policy(len(V), r, s) if policy else [(5, 1 << 20, 20)]
        late = bool(deadline and time.time() > deadline)
        if late:
            ladder = ladder[-2:]
        for k, iters, tries in ladder:
            got = find_step_par(V, R, k, rng, iters, tries, nproc, None if late else deadline, log, force_pivot)
            if got:
                got['k'] = k
                break
        if not got:
            return dict(events=events, missing=list(unfinished), products=products, helper_products=0, pool=n,
                        pool_peak=n, scratch_peak=0, done=s)
        cmask = got['cmask']
        g = 0
        orig_R = others + [p]
        for i in range(r):
            if cmask >> i & 1:
                g ^= targets[orig_R[i]]
        got.update(type='step', V=list(V), Vw=list(Vw), g=g, pivot=r - 1,
                   combo=[i for i in range(r) if cmask >> i & 1], orig=list(orig_R), D=len(V))
        events.append(got)
        products += got['k']
        if log:
            log(f'inplace step {s}: pivot target {p} D={len(V)} r={r} k={got["k"]} tests={got["tests"]}')
        fin_g.append(g)
        fin_w.append(tw[p])
        unfinished = others
        s += 1
    return dict(events=events, missing=[], products=products, helper_products=0, pool=n, pool_peak=n, scratch_peak=0,
                done=s)


def emit_events_inplace(events, tw, state):
    """Emit + check the in-place steps.  The step emission is the same as emit_events (products into the pivot wire, then
    the u carrier), but u is taken over the step's operand wires and the wire already holds old: the RHS solved above was
    g + old, so nothing is removed here.  The final CX network is emit_events' (steps in reverse)."""
    return emit_events(events, tw, state=state)


def inplace_check(ops, avail, targets, olds, twires, awires_state=None):
    """Exact check of an in-place fragment on every row (numpy): targets end equal to their columns, available wires
    unchanged.  avail: [(wire, col)], targets: [(wire, col)], olds: [col or None]."""
    import numpy as np
    n = len(targets[0][1])
    st = {w: c.copy() for w, c in avail}
    for (w, _), o in zip(targets, olds):
        st[w] = np.zeros(n, bool) if o is None else o.copy()
    apply_ops(ops, st)
    return all(np.array_equal(st[w], c) for w, c in targets) and all(np.array_equal(st[w], c) for w, c in avail)


def synth_from_columns_inplace(avail, targets, olds, seed=1, seconds=30, nproc=1, policy=None, log=None, restarts=3,
                               max_bits=24, order=None):
    """In-place model.  avail [(wire, col)], targets [(wire, col)], olds [col | None] (the value each target wire holds
    at the window entry; None = the wire starts at 0).  Old values must be determined by the available state (checked).
    -> dict like synth_from_columns, plus start_model='old value in place'."""
    import numpy as np
    twires = [w for w, _ in targets]
    if set(w for w, _ in avail) & set(twires):
        return dict(ok=False, reason='a target wire is also an available wire')
    nt = len(targets)
    cols = [c for _, c in targets] + [o for o in olds if o is not None]
    inst = instance_from_columns(avail, cols)
    if inst is None:
        return dict(ok=False, reason=LAST_FAIL.get('why', 'more than 1024 distinct available-value states (kernel limit) or a target/old value is not determined'))
    if len(inst['U']) > 64:
        return dict(ok=False, reason='operand space larger than 64 dimensions (kernel limit)')
    Tf = inst['T'][:nt]
    it = iter(inst['T'][nt:])
    Of = [None if o is None else next(it) for o in olds]
    t0 = time.time()
    deadline = t0 + seconds if seconds else None
    res = None
    for attempt in range(max(1, restarts)):
        res = synthesise_inplace(inst['U'], inst['Uw'], Tf, twires, Of, order=order,
                                 rng=random.Random(seed + 1000 * attempt),
                                 policy=policy or auto_policy(inst['N'], max_bits), nproc=nproc, log=log,
                                 deadline=deadline)
        if not res['missing'] or (deadline and time.time() > deadline):
            break
    out = dict(ok=False, start_model='old value in place', N=inst['N'], operand_dim=len(inst['U']),
               seconds=round(time.time() - t0, 1), products=res['products'], helper_products=0, scratch_peak=0,
               missing=len(res['missing']))
    if res['missing']:
        out['reason'] = f'{len(res["missing"])} targets not captured within the search budget'
        return out
    n = len(targets[0][1])
    st = {w: c.copy() for w, c in avail}
    for (w, _), o in zip(targets, olds):
        st[w] = np.zeros(n, bool) if o is None else o.copy()
    ops, ccx = emit_events(res['events'], twires, state=st)
    good = all(np.array_equal(st[w], c) for w, c in targets) and all(np.array_equal(st[w], c) for w, c in avail)
    out.update(ok=bool(good), and_cost=ccx, ops=ops, rows_checked=int(n),
               steps=[(e['k'], e['D'], len(e['combo'])) for e in res['events'] if e['type'] == 'step'])
    if not good:
        out['reason'] = 'fragment check failed'
    return out
