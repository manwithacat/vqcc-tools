#!/usr/bin/env python3
"""Exact-domain bitset simulator for kickmix text circuits (.kmx).

Every qubit and classical bit is a packed bitset over a list of input rows (by default all legal invsqrt
x = 1..3,133,452, one bit per x). One pass over the op stream evaluates the circuit for every x at once
(about 2-6 s for a 46k-op invsqrt circuit). Semantics follow an earlier dataflow tracer,
which was validated shot-for-shot against kickmix:

  * an op runs only where every bit on the condition stack and its own `if bN` are 1; PUSH_CONDITION pushes
    the bit's current value and, like POP, ignores the stack; an inactive HMR does not write its bit;
  * HMR q b: outcome m; b := m; sign flips where q = m = 1; q := 0. R q: q := 0 (a 1 there is a dirty reset);
  * Z/CZ/CCZ/NEG flip the sign; Z_POW q t adds t/2 turns where q = 1.

Outcome models: 'zero' (every HMR returns 0: the deterministic branch, used for value analysis) or an integer
seed (independent random outcome per row and per HMR, used for equivalence checks).

Phase is kept as a sign bitset plus a per-row uint64 accumulator of Z_POW angles in units of 2^-64 turns
(each Z_POW rounded to the nearest unit). Two circuits with the same Z_POW multiset on equal wire values give
equal accumulators; a final verifier run remains the authority on phase.

Usage: python qhdl_bitsim.py CIRCUIT.kmx [zero|SEED]     (prints x restored / scratch clean / mean Toffoli)
Library: BitSim(ops, ...).run(stop, snaps, on_snap); final_state(path, outcome); contract(sim) checks the invsqrt
contract (x restored, scratch clean, |phase error| <= tol) on all legal x; equal_runs(a, b).
"""
from __future__ import annotations

import sys
from fractions import Fraction

import numpy as np

KINDS = ['X', 'CX', 'CCX', 'Z', 'CZ', 'CCZ', 'SWAP', 'CSWAP', 'R', 'HMR', 'NEG', 'Z_POW',
         'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1', 'PUSH_CONDITION', 'POP_CONDITION', 'INFO_TAG']
CODE = {k: i for i, k in enumerate(KINDS)}
(C_X, C_CX, C_CCX, C_Z, C_CZ, C_CCZ, C_SWAP, C_CSWAP, C_R, C_HMR, C_NEG, C_ZPOW,
 C_BINV, C_BST0, C_BST1, C_PUSH, C_POP, C_INFO) = range(18)
TOFFOLI_LIKE = (C_CCX, C_CCZ, C_CSWAP)
QUBIT_OPS = {C_X: 1, C_CX: 2, C_CCX: 3, C_Z: 1, C_CZ: 2, C_CCZ: 3, C_SWAP: 2, C_CSWAP: 3, C_R: 1, C_HMR: 1,
             C_ZPOW: 1}
MAX_X = 3_133_452
ALL1 = np.uint64(0xFFFFFFFFFFFFFFFF)


class Op:
    __slots__ = ('code', 'q', 'bit', 'cond', 'angle', 'text')

    def __init__(self, code, q, bit, cond, angle, text):
        self.code, self.q, self.bit, self.cond, self.angle, self.text = code, q, bit, cond, angle, text


def parse_kmx(path):
    """Return (ops, header_lines, x_qubits). Op i is the i-th non-header line."""
    ops, header, xq = [], [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            t = line.split()
            if not t:
                continue
            if t[0] in ('APPEND_TO_REGISTER', 'REGISTER'):
                header.append(line)
                if t[0] == 'APPEND_TO_REGISTER':
                    xq.append(int(t[1][1:]))
                continue
            ops.append(parse_line(line))
    return ops, header, xq


def parse_line(line):
    t = line.split()
    name = t[0]
    if name == 'INFO_TAG':
        return Op(C_INFO, (), -1, -1, 0, line)
    cond = -1
    if len(t) >= 2 and t[-2] == 'if':
        cond = int(t[-1][1:])
        t = t[:-2]
    code = CODE[name]
    args = t[1:]
    angle = 0
    bit = -1
    if code == C_ZPOW:
        fr = Fraction(args[1])  # half turns
        angle = int(round(fr * (1 << 63))) % (1 << 64)  # units of 2^-64 turns
        args = args[:1]
    if code == C_HMR:
        bit = int(args[1][1:])
        args = args[:1]
    if code in (C_BINV, C_BST0, C_BST1):
        bit = int(args[0][1:])
        args = []
    q = tuple(int(a[1:]) for a in args)
    return Op(code, q, bit, cond, angle, line)


def write_kmx(path, header, ops_or_lines):
    with open(path, 'w') as f:
        for h in header:
            f.write(h + '\n')
        for o in ops_or_lines:
            f.write((o.text if isinstance(o, Op) else o) + '\n')


def legal_rows():
    return np.arange(1, MAX_X + 1, dtype=np.int64)


def pack_bits(boolarr):
    """bool array (n,) -> packed uint64 words, row r at word r//64 bit r%64."""
    n = len(boolarr)
    W = (n + 63) // 64
    b = np.zeros(W * 64, np.uint8)
    b[:n] = boolarr
    return np.packbits(b.reshape(-1, 8)[:, ::-1], axis=1).reshape(-1).view('<u8').copy()


def unpack_bits(words, n):
    b = np.unpackbits(words.view(np.uint8).reshape(-1, 1), axis=1)[:, ::-1].reshape(-1)
    return b[:n].astype(bool)


class BitSim:
    def __init__(self, ops, rows=None, num_qubits=63, x_qubits=None, outcome='zero', count_toffoli=False):
        self.ops = ops
        self.rows = legal_rows() if rows is None else np.asarray(rows, dtype=np.int64)
        self.n = len(self.rows)
        self.W = (self.n + 63) // 64
        self.valid = pack_bits(np.ones(self.n, bool))
        self.nq = num_qubits
        self.q = np.zeros((num_qubits, self.W), np.uint64)
        self.xq = list(range(22)) if x_qubits is None else x_qubits
        for k, qq in enumerate(self.xq):
            self.q[qq] = pack_bits(((self.rows >> k) & 1).astype(bool))
        self.bits = {}
        self.stack = []
        self.act = None  # None = all rows active
        self.sign = np.zeros(self.W, np.uint64)
        self.rot = np.zeros(self.n, np.uint64)
        self.outcome = outcome
        self.rng = None if outcome == 'zero' else np.random.default_rng(int(outcome))
        self.dirty = 0
        self.pos = 0
        self.count_toffoli = count_toffoli
        self.toffoli_rows = 0  # sum over executed Toffoli-like ops of active rows

    def _bit(self, b):
        v = self.bits.get(b)
        if v is None:
            v = np.zeros(self.W, np.uint64)
            self.bits[b] = v
        return v

    def _recompute_act(self):
        if not self.stack:
            self.act = None
            return
        a = self.stack[0].copy()
        for s in self.stack[1:]:
            a &= s
        self.act = a

    def run(self, stop=None, snaps=None, on_snap=None):
        """Execute ops[pos:stop]. snaps: sorted op indices; before executing op i in snaps, call on_snap(i, self)."""
        ops = self.ops
        stop = len(ops) if stop is None else stop
        q = self.q
        snaps = list(snaps or [])
        si = 0
        while si < len(snaps) and snaps[si] < self.pos:
            si += 1
        for i in range(self.pos, stop):
            while si < len(snaps) and snaps[si] == i:
                on_snap(i, self)
                si += 1
            o = ops[i]
            c = o.code
            if c == C_PUSH:
                self.stack.append(self._bit(o.cond).copy())
                self._recompute_act()
                continue
            if c == C_POP:
                if self.stack:
                    self.stack.pop()
                    self._recompute_act()
                continue
            if c == C_INFO:
                continue
            act = self.act
            if o.cond >= 0:
                cb = self._bit(o.cond)
                act = cb.copy() if act is None else (act & cb)
            if act is not None and not act.any():
                continue
            a = o.q
            if c == C_CX:
                m = q[a[0]] if act is None else (q[a[0]] & act)
                q[a[1]] ^= m
            elif c == C_CZ:
                m = q[a[0]] & q[a[1]]
                self.sign ^= m if act is None else (m & act)
            elif c == C_Z:
                self.sign ^= q[a[0]] if act is None else (q[a[0]] & act)
            elif c == C_X:
                q[a[0]] ^= self.valid if act is None else act
            elif c == C_CCX:
                m = q[a[0]] & q[a[1]]
                if act is not None:
                    m &= act
                q[a[2]] ^= m
                if self.count_toffoli:
                    self.toffoli_rows += self._count(act)
            elif c == C_HMR:
                qq = a[0]
                m = self._outcome()
                b = self._bit(o.bit)
                if act is None:
                    b[:] = m
                    self.sign ^= q[qq] & m
                    q[qq] = 0
                else:
                    b[:] = (b & ~act) | (m & act)
                    self.sign ^= q[qq] & m & act
                    q[qq] &= ~act
            elif c == C_CSWAP:
                d = q[a[0]] & (q[a[1]] ^ q[a[2]])
                if act is not None:
                    d &= act
                q[a[1]] ^= d
                q[a[2]] ^= d
                if self.count_toffoli:
                    self.toffoli_rows += self._count(act)
            elif c == C_R:
                m = q[a[0]] if act is None else (q[a[0]] & act)
                if m.any():
                    self.dirty += int(np.bitwise_count(m).sum())
                q[a[0]] ^= m
            elif c == C_ZPOW:
                m = q[a[0]] if act is None else (q[a[0]] & act)
                sel = unpack_bits(m, self.n)
                self.rot[sel] += np.uint64(o.angle)
            elif c == C_CCZ:
                m = q[a[0]] & q[a[1]] & q[a[2]]
                self.sign ^= m if act is None else (m & act)
                if self.count_toffoli:
                    self.toffoli_rows += self._count(act)
            elif c == C_NEG:
                self.sign ^= self.valid if act is None else act
            elif c == C_SWAP:
                d = q[a[0]] ^ q[a[1]]
                if act is not None:
                    d &= act
                q[a[0]] ^= d
                q[a[1]] ^= d
            elif c == C_BINV:
                b = self._bit(o.bit)
                b ^= self.valid if act is None else act
            elif c == C_BST0:
                b = self._bit(o.bit)
                if act is None:
                    b[:] = 0
                else:
                    b &= ~act
            elif c == C_BST1:
                b = self._bit(o.bit)
                b |= self.valid if act is None else act
        while si < len(snaps) and snaps[si] == stop:
            on_snap(stop, self)
            si += 1
        self.pos = stop
        return self

    def _count(self, act):
        return self.n if act is None else int(np.bitwise_count(act).sum())

    def _outcome(self):
        if self.rng is None:
            return np.zeros(self.W, np.uint64)
        return self.rng.integers(0, 1 << 64, self.W, dtype=np.uint64, endpoint=False) & self.valid

    # ----- end-state helpers -----
    def phase_units(self):
        """Per-row total phase in 2^-64 turns (sign as half a turn)."""
        s = unpack_bits(self.sign, self.n).astype(np.uint64) << np.uint64(63)
        return self.rot + s

    def x_restored(self):
        for k, qq in enumerate(self.xq):
            ref = pack_bits(((self.rows >> k) & 1).astype(bool))
            if not np.array_equal(self.q[qq], ref):
                return False
        return True

    def scratch_clean(self):
        xs = set(self.xq)
        return all(not self.q[i].any() for i in range(self.nq) if i not in xs)


def load(path):
    ops, header, xq = parse_kmx(path)
    nq = 1 + max(max(o.q) for o in ops if o.q)
    return ops, header, xq, nq


def final_state(path, outcome='zero', rows=None):
    ops, header, xq, nq = load(path)
    s = BitSim(ops, rows=rows, num_qubits=nq, x_qubits=xq, outcome=outcome, count_toffoli=True).run()
    return s


if __name__ == '__main__':
    import time
    t0 = time.time()
    s = final_state(sys.argv[1], outcome=sys.argv[2] if len(sys.argv) > 2 else 'zero')
    print(dict(x_restored=s.x_restored(), scratch_clean=s.scratch_clean(), dirty=s.dirty,
               mean_toffoli=s.toffoli_rows / s.n, seconds=round(time.time() - t0, 1)))


# ----- invsqrt contract on all legal x (local analysis; the official verifier stays the authority) -----
_REF_CACHE = __import__('os').environ.get('QHDL_INVSQRT_REF_CACHE') or str(  # .npz cache of the reference table
    __import__('pathlib').Path(__import__('os').environ.get('QHDL_WORK') or __import__('pathlib').Path(__file__).resolve().parents[1] / 'work')
    / '_cache' / 'invsqrt_ref.npz')


def invsqrt_reference():
    import math
    import os
    if _REF_CACHE and os.path.exists(_REF_CACHE):
        z = np.load(_REF_CACHE)
        return z['target'], z['tol']
    num, shift, mask = 1 << 130, 25, (1 << 64) - 1
    rs = [math.isqrt(num // x) for x in range(1, MAX_X + 1)]
    target = np.array([((r + 1) >> 1) & mask for r in rs], dtype=np.uint64)
    tol = np.array([min(r >> shift, 1 << 63) for r in rs], dtype=np.uint64)
    if _REF_CACHE:
        os.makedirs(os.path.dirname(_REF_CACHE), exist_ok=True)
        np.savez(_REF_CACHE, target=target, tol=tol)
    return target, tol


def contract(sim):
    """x restored, scratch clean, per-row |phase error| <= tol; returns dict with worst ratio and min margin."""
    target, tol = invsqrt_reference()
    assert sim.n == MAX_X and sim.rows[0] == 1
    err = (sim.phase_units() - target).view(np.int64)
    used = np.abs(err).view(np.uint64)
    ratio = used / tol
    return dict(x_restored=sim.x_restored(), scratch_clean=sim.scratch_clean(), dirty=sim.dirty,
                phase_ok=bool((used <= tol).all()), worst_ratio=float(ratio.max()),
                worst_x=int(np.argmax(ratio)) + 1, min_margin=int((tol.astype(np.int64) - used.astype(np.int64)).min()),
                mean_toffoli=sim.toffoli_rows / sim.n)


def equal_runs(a, b):
    """Two finished BitSims (same rows, same outcome model): identical qubits, signs and rotation phases."""
    return bool(np.array_equal(a.q, b.q) and np.array_equal(a.phase_units(), b.phase_units()))
