#!/usr/bin/env python3
"""kmsim: an exact numpy simulator for the subset of the kickmix circuit language used in this competition.

It is an analysis tool, not the official verifier, and it does not need kickmix. It reads the kickmix text format
(``.kmx``; a gzip-compressed ``.kmx.gz`` is read transparently) and simulates a batch of classical basis inputs at
once, tracking qubit values, classical bits, a global phase per shot, Toffoli-like counts and rotation counts.

Semantics (from kickmix ``op_type.h``):
  * HMR = Hadamard + measure + reset: random outcome m, phase (-1)^(m*v), qubit -> 0, bit <- m.
  * R   = reset; resetting a qubit that holds 1 is an error (counted in ``bad_reset``).
  * Conditions: an op runs only if every pushed condition bit is 1 and its own ``if bN`` bit is 1.
  * The phase is kept in units of 2^-64 turns (uint64); ``Z_POW`` angles are in half-turns.
  * Toffoli-like ops: CCX, CCZ, CSWAP. ``rot`` counts every executed Z_POW, Clifford angles included.

Command line (invsqrt check over chosen inputs x)::

    python kmsim.py circuit.kmx                  # x = 1..4096 (fast)
    python kmsim.py circuit.kmx --xmax 3133452   # every x of the task (slow: minutes, about 1 GB RAM)

Check against an official receipt: on the public invsqrt circuit 9c708f7b9abc, every x in 1..3133452 gives
``worst_ratio = 0.9993237694056958``, identical to the official verifier's ``worst_error_over_tolerance``; the
mean Toffoli count is 1213.868 against the verifier's 1213.865 (HMR outcomes are random, so the mean is sampled).

Licence: Apache-2.0.
"""
import argparse
import gzip
from fractions import Fraction

import numpy as np

MASK = (1 << 64) - 1
TOFF = {"CCX", "CCZ", "CSWAP"}
INVSQRT_XMAX = 3133452  # largest input x of the invsqrt task, used by the examples


def half_turns_units(s):
    """Exact decimal half-turns -> phase units of 2^-64 turns."""
    f = Fraction(s)
    return int(round(f * (1 << 63))) & MASK  # half-turn = 2^63 units


def open_text(path):
    """Open a .kmx or .kmx.gz file as text."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def parse(path):
    """Parse a kmx file. Returns (ops, regs): ops = [(name, qubits, bits, cond_bit, angle_units)],
    regs = {register id: [qubit indices in order]}."""
    ops = []
    regs = {}
    with open_text(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "APPEND_TO_REGISTER":
                regs.setdefault(t[2], []).append(int(t[1][1:]))
                continue
            if t[0] == "REGISTER":
                continue
            cond = None
            if "if" in t:
                i = t.index("if")
                cond = int(t[i + 1][1:])
                t = t[:i]
            name = t[0]
            args = t[1:]
            qs = [int(a[1:]) for a in args if a.startswith("q")]
            bs = [int(a[1:]) for a in args if a.startswith("b")]
            ang = None
            if name == "Z_POW":
                ang = half_turns_units(args[1])
            ops.append((name, qs, bs, cond, ang))
    return ops, regs


def register_names(path):
    """{register name: register id} from the REGISTER lines, e.g. {"selection": "r0", "word_0": "r1"}."""
    m = {}
    with open_text(path) as fh:
        for line in fh:
            if line.startswith("REGISTER"):
                t = line.split()
                m[t[2].strip('"')] = t[1]
    return m


def sizes(ops, regs=None):
    """(number of qubits, number of classical bits) a circuit needs."""
    nq = 1 + max(max(o[1]) for o in ops if o[1])
    if regs:
        nq = max(nq, 1 + max(q for v in regs.values() for q in v))
    nb = 1 + max([max(o[2]) for o in ops if o[2]] + [o[3] for o in ops if o[3] is not None] + [0])
    return nq, nb


class Sim:
    """Batch simulator: q[nq, B] qubit values, b[nb, B] bits, ph[B] phase, toff[B]/rot[B] counters."""

    def __init__(self, ops, nq, nb, batch, seed=0):
        self.ops = ops
        self.B = batch
        self.q = np.zeros((nq, batch), dtype=bool)
        self.b = np.zeros((nb, batch), dtype=bool)
        self.ph = np.zeros(batch, dtype=np.uint64)
        self.rng = np.random.default_rng(seed)
        self.toff = np.zeros(batch, dtype=np.int64)
        self.rot = np.zeros(batch, dtype=np.int64)
        self.bad_reset = np.zeros(batch, dtype=bool)

    def run(self, start=0, stop=None, hook=None):
        """Run ops[start:stop]; hook(k, sim) is called after each op."""
        stack = getattr(self, "stack", [])
        q, b = self.q, self.b
        ones = np.ones(self.B, dtype=bool)
        for k in range(start, len(self.ops) if stop is None else stop):
            name, qs, bs, cond, ang = self.ops[k]
            if name == "PUSH_CONDITION":
                stack.append(cond)
                continue
            if name == "POP_CONDITION":
                stack.pop()
                continue
            act = ones
            if stack or cond is not None:
                act = ones.copy()
                for c in stack:
                    act &= b[c]
                if cond is not None:
                    act &= b[cond]
            if name == "CX":
                q[qs[1]] ^= q[qs[0]] & act
            elif name == "CZ":
                self._neg(q[qs[0]] & q[qs[1]] & act)
            elif name == "Z":
                self._neg(q[qs[0]] & act)
            elif name == "X":
                q[qs[0]] ^= act
            elif name == "CCX":
                q[qs[2]] ^= q[qs[0]] & q[qs[1]] & act
                self.toff += act
            elif name == "CCZ":
                self._neg(q[qs[0]] & q[qs[1]] & q[qs[2]] & act)
                self.toff += act
            elif name == "SWAP":
                a, c = q[qs[0]].copy(), q[qs[1]].copy()
                q[qs[0]] = np.where(act, c, a)
                q[qs[1]] = np.where(act, a, c)
            elif name == "CSWAP":
                m = act & q[qs[0]]
                a, c = q[qs[1]].copy(), q[qs[2]].copy()
                q[qs[1]] = np.where(m, c, a)
                q[qs[2]] = np.where(m, a, c)
                self.toff += act
            elif name == "NEG":
                self._neg(act)
            elif name == "HMR":
                m = self.rng.random(self.B) < 0.5
                self._neg(m & q[qs[0]] & act)
                q[qs[0]] &= ~act
                b[bs[0]] = np.where(act, m, b[bs[0]])
            elif name == "R":
                self.bad_reset |= q[qs[0]] & act
                q[qs[0]] &= ~act
            elif name == "Z_POW":
                m = q[qs[0]] & act
                self.ph = self.ph + np.where(m, np.uint64(ang), np.uint64(0))
                self.rot += act
            elif name == "BIT_STORE0":
                b[bs[0]] &= ~act
            elif name == "BIT_STORE1":
                b[bs[0]] |= act
            elif name == "BIT_INVERT":
                b[bs[0]] ^= act
            elif name in ("INFO_TAG", "DEBUG_PRINT"):
                pass
            else:
                raise ValueError(f"op {name} is not supported by kmsim")
            if hook is not None:
                hook(k, self)
        self.stack = stack

    def _neg(self, m):
        self.ph = self.ph + np.where(m, np.uint64(1 << 63), np.uint64(0))


def target_and_tol(xs, precision_bits=24):
    """invsqrt task: target phase round(2^64 / sqrt(x)) mod 2^64 and tolerance per input x."""
    import math
    rs = [math.isqrt((1 << 130) // int(x)) for x in xs]
    target = np.array([((r + 1) >> 1) & MASK for r in rs], dtype=np.uint64)
    tol = np.array([min(r >> (precision_bits + 1), 1 << 63) for r in rs], dtype=np.uint64)
    return target, tol


def check(path, xs, seed=0, ops_regs=None):
    """Simulate an invsqrt circuit on inputs xs (int64 array). Returns (report dict, Sim).

    report: nq, restored (input register unchanged), dirty (shots with a non-zero scratch qubit), bad_reset,
    worst_ratio (max |phase error| / tolerance; < 1 passes), toff_mean, rot_mean."""
    ops, regs = ops_regs or parse(path)
    nq = 1 + max(max(o[1]) for o in ops if o[1])
    nb = 1 + max((max(o[2]) for o in ops if o[2]), default=0)
    nb = max(nb, 1 + max((o[3] for o in ops if o[3] is not None), default=0))
    s = Sim(ops, nq, nb, len(xs), seed)
    xq = regs["r0"]
    for i, qi in enumerate(xq):
        s.q[qi] = (xs >> i) & 1 == 1
    s.run()
    xr = sum((s.q[qi].astype(np.uint64) << np.uint64(i)) for i, qi in enumerate(xq))
    restored = np.array_equal(xr, xs.astype(np.uint64))
    scratch = [i for i in range(nq) if i not in xq]
    dirty = s.q[scratch].any(axis=0)
    target, tol = target_and_tol(xs)
    err = (s.ph - target).view(np.int64)
    used = np.abs(err).astype(np.float64)
    return dict(nq=nq, restored=restored, dirty=int(dirty.sum()), bad_reset=int(s.bad_reset.sum()),
                worst_ratio=float((used / tol.astype(np.float64)).max()), toff_mean=float(s.toff.mean()),
                rot_mean=float(s.rot.mean())), s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate an invsqrt .kmx circuit and report its accuracy and cost.")
    ap.add_argument("kmx", help="invsqrt circuit in kmx text format (.kmx or .kmx.gz)")
    ap.add_argument("--xmin", type=int, default=1, help="first input x (default 1)")
    ap.add_argument("--xmax", type=int, default=4096, help=f"last input x (default 4096; the task uses up to {INVSQRT_XMAX})")
    ap.add_argument("--seed", type=int, default=0, help="seed for HMR outcomes")
    a = ap.parse_args(argv)
    xs = np.arange(a.xmin, a.xmax + 1, dtype=np.int64)
    rep, _ = check(a.kmx, xs, seed=a.seed)
    rep["pass_on_these_x"] = bool(rep["restored"] and rep["dirty"] == 0 and rep["bad_reset"] == 0 and rep["worst_ratio"] < 1)
    for k, v in rep.items():
        print(f"{k:12s} {v}")
    return rep


if __name__ == "__main__":
    main()
