# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""chained_and: bounds, synthesis, exhaustive checks and a CNF encoding for chained AND circuits on a (y, z) block.

Model
-----
A loader output block is a multi-output Boolean function f_w(y, z), w < m, of a small low address y (n bits, all
2^n values) and a row index z (here a 16-address block: the high address). Every function of y alone and every
function of z alone is free (the full low and high monomial caches: XOR of cached wires). An AND gate multiplies two
operands; an operand is

    a(y) XOR s(z) XOR (XOR of earlier gates selected by a chain mask),

so a gate with empty chain masks is a rank-2 product (A(y) + a(z)) (B(y) + b(z)) of the tensorqrom primitive, and a
gate with a nonempty mask is a chained (deep) product. An output is P_w(y) XOR q_w(z) XOR (XOR of gates selected by
e_w). Toffolis = number of gates (each computed once, erased by measurement).

Fixing a row z turns s(z), the operand's z-part, into a per-row bit, so a circuit is a y-circuit with 2k per-row
bits (sigma_j, tau_j) and m per-row output constants. A target row is met when, for some row bits, every output
equals the target on the row's care points up to one constant per output. Don't-cares (care mask bits 0) are free.

What this module provides
-------------------------
- `Target` (JSON in/out), `Circuit` (evaluate, depth), `verify` (exhaustive over every row and every care point;
  row bits given or found by exhaustive search over all 4^k settings).
- `rank2_lower_bound` (rank-2 products >= ceil(affine rank of the normalised full rows / 2)), and
  `general_lower_bound` (any chained circuit needs >= d ANDs, d the output dimension modulo free functions, and one
  more if no nonzero output combination has mixed rank <= 2).
- `construct_rank2`: a per-output rank-2 construction (sum over outputs of ceil(rank_w / 2) products).
- `linear_fiber_dim`: the affine dimension of the outputs that are linear in y (one nonlinear fibre) over all row
  settings of a circuit; `copy_gate_example` is a 2-AND chained circuit whose linear fibre has dimension 5, so
  chained products can beat the rank-2 bound (3) on linear-in-y targets.
- `encode_cnf` / `decode_model` / `check_assignment`: a complete CNF encoding of "a k-AND chained circuit of chain
  depth <= D meets these rows" for any external SAT solver (DIMACS), and a check that a known circuit satisfies it.

No SAT solver ships with this module. A timeout or a missing solver is never a proof; an UNSAT answer on any subset
of rows is a proof for the whole target (fewer rows can only make the problem easier).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__version__ = "0.3.0"


# ------------------------------------------------------------------------------------------------ GF(2) helpers

def rank(vectors) -> int:
    piv: dict[int, int] = {}
    for v in vectors:
        v = int(v)
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p]
            else:
                piv[p] = v
                break
    return len(piv)


def basis(vectors) -> list[int]:
    piv: dict[int, int] = {}
    out = []
    for v in vectors:
        w = int(v)
        while w:
            p = w.bit_length() - 1
            if p in piv:
                w ^= piv[p]
            else:
                piv[p] = w
                out.append(int(v))
                break
    return out


def coords(v: int, B: list[int]) -> int | None:
    """Mask x with XOR of B[i] over bits i of x equal to v, or None."""
    piv: dict[int, tuple[int, int]] = {}
    for i, b in enumerate(B):
        w, c = b, 1 << i
        while w:
            p = w.bit_length() - 1
            if p in piv:
                w ^= piv[p][0]
                c ^= piv[p][1]
            else:
                piv[p] = (w, c)
                break
    c = 0
    while v:
        p = v.bit_length() - 1
        if p not in piv:
            return None
        v ^= piv[p][0]
        c ^= piv[p][1]
    return c


# ------------------------------------------------------------------------------------------------ target

@dataclass
class Target:
    n: int                     # low (y) bits
    m: int                     # outputs
    rows: list                 # [(care mask over 2^n, [truth table per output])]
    names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return 1 << self.n

    @property
    def full(self) -> int:
        return (1 << self.N) - 1

    @classmethod
    def from_json(cls, path) -> "Target":
        d = json.loads(Path(path).read_text())
        n = int(d["ybits"])
        rows = [(int(r["care"]), [int(t) for t in r["tt"]]) for r in d["rows"]]
        meta = {k: v for k, v in d.items() if k not in ("rows",)}
        return cls(n=n, m=len(rows[0][1]), rows=rows, names=list(d.get("outputs", [])), meta=meta)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(dict(self.meta, ybits=self.n, outputs=self.names or
                                              [f"out{w}" for w in range(self.m)],
                                              rows=[dict(care=c, tt=t) for c, t in self.rows]), indent=1) + "\n")

    def norm(self, f: int) -> int:
        """f minus its value at y = 0 (free row constant)."""
        return f ^ self.full if f & 1 else f

    def full_rows(self) -> list[int]:
        """Normalised full-care rows as single integers (output w in bits w*2^n ..)."""
        out = []
        for care, tts in self.rows:
            if care == self.full:
                out.append(sum(self.norm(t) << (w * self.N) for w, t in enumerate(tts)))
        return out


# ------------------------------------------------------------------------------------------------ circuit

@dataclass
class Circuit:
    n: int
    m: int
    gates: list                 # [(a, chain_a, b, chain_b)]: a, b truth tables over y; chain masks over earlier gates
    e: list                     # per output: mask over gates
    P: list                     # per output: truth table over y

    @property
    def k(self) -> int:
        return len(self.gates)

    def depth(self) -> int:
        d = []
        for a, ca, b, cb in self.gates:
            parents = [d[i] for i in range(len(d)) if (ca | cb) >> i & 1]
            d.append(1 + max(parents, default=0))
        return max(d, default=0)

    def evaluate(self, sig: int, tau: int) -> list[int]:
        """Output truth tables over y for per-row operand bits sig (first operand) and tau (second), without the
        per-row output constants."""
        full = (1 << (1 << self.n)) - 1
        g = []
        for j, (a, ca, b, cb) in enumerate(self.gates):
            A, B = a, b
            for i in range(j):
                if ca >> i & 1:
                    A ^= g[i]
                if cb >> i & 1:
                    B ^= g[i]
            if sig >> j & 1:
                A ^= full
            if tau >> j & 1:
                B ^= full
            g.append(A & B)
        outs = []
        for w in range(self.m):
            o = self.P[w]
            for j in range(self.k):
                if self.e[w] >> j & 1:
                    o ^= g[j]
            outs.append(o)
        return outs

    def evaluate_all(self) -> np.ndarray:
        """Outputs for every one of the 4^k row settings: array (4^k, m) of truth tables (object ints avoided:
        uint64 for 2^n <= 64)."""
        assert (1 << self.n) <= 64
        k, full = self.k, np.uint64((1 << (1 << self.n)) - 1) if self.n < 6 else np.uint64(0xFFFFFFFFFFFFFFFF)
        S = np.arange(1 << (2 * k), dtype=np.int64)
        g = []
        for j, (a, ca, b, cb) in enumerate(self.gates):
            A = np.full(S.shape, np.uint64(a))
            B = np.full(S.shape, np.uint64(b))
            for i in range(j):
                if ca >> i & 1:
                    A ^= g[i]
                if cb >> i & 1:
                    B ^= g[i]
            A ^= np.where((S >> (2 * j)) & 1, full, np.uint64(0))
            B ^= np.where((S >> (2 * j + 1)) & 1, full, np.uint64(0))
            g.append(A & B)
        out = np.zeros((len(S), self.m), np.uint64)
        for w in range(self.m):
            o = np.full(S.shape, np.uint64(self.P[w]))
            for j in range(k):
                if self.e[w] >> j & 1:
                    o ^= g[j]
            out[:, w] = o
        return out

    def to_dict(self) -> dict:
        return dict(n=self.n, m=self.m, k=self.k, depth=self.depth(), gates=[list(g) for g in self.gates],
                    e=list(self.e), P=list(self.P))

    @classmethod
    def from_dict(cls, d) -> "Circuit":
        return cls(n=d["n"], m=d["m"], gates=[tuple(g) for g in d["gates"]], e=list(d["e"]), P=list(d["P"]))


def setting_bits(s: int, k: int) -> tuple[int, int]:
    """Row setting index of evaluate_all -> (sig, tau)."""
    sig = sum(((s >> (2 * j)) & 1) << j for j in range(k))
    tau = sum(((s >> (2 * j + 1)) & 1) << j for j in range(k))
    return sig, tau


def row_met(outs, care: int, tts, full: int) -> bool:
    for o, t in zip(outs, tts):
        d = (o ^ t) & care
        if d and d != care:
            return False
    return True


def verify(circ: Circuit, target: Target, rowbits=None) -> dict:
    """Exhaustive check over every row and every care point. With rowbits [(sig, tau), ...] the given settings are
    checked; otherwise every row searches all 4^k settings (exact: a row fails only if no setting meets it)."""
    assert circ.n == target.n and circ.m == target.m
    full = target.full
    ok, found = [], []
    table = None
    for r, (care, tts) in enumerate(target.rows):
        if rowbits is not None:
            sig, tau = rowbits[r]
            good = row_met(circ.evaluate(sig, tau), care, tts, full)
            ok.append(good)
            found.append((sig, tau) if good else None)
            continue
        if table is None:
            table = circ.evaluate_all()
        hit = np.ones(len(table), bool)
        for w in range(target.m):
            d = (table[:, w] ^ np.uint64(tts[w])) & np.uint64(care)
            hit &= (d == 0) | (d == np.uint64(care))
        idx = np.flatnonzero(hit)
        ok.append(bool(len(idx)))
        found.append(setting_bits(int(idx[0]), circ.k) if len(idx) else None)
    return dict(all_pass=all(ok), rows=len(ok), rows_met=int(sum(ok)), ands=circ.k, depth=circ.depth(),
                failed_rows=[i for i, x in enumerate(ok) if not x], rowbits=found,
                method="given row bits" if rowbits is not None else f"exhaustive over 4^{circ.k} row settings")


# ------------------------------------------------------------------------------------------------ bounds

def affine_rank(vectors) -> int:
    vectors = list(vectors)
    return rank([v ^ vectors[0] for v in vectors]) if vectors else 0


def rank2_lower_bound(target: Target) -> dict:
    """Rank-2 (chain depth 1) products >= ceil(r / 2), r the affine rank of the normalised full rows: a depth-1
    circuit's outputs are affine in its 2k row bits."""
    r = affine_rank(target.full_rows())
    return dict(affine_rank=r, bound=-(-r // 2))


def combo_rank(target: Target, c: int) -> int:
    rows = []
    for v in target.full_rows():
        x = 0
        for w in range(target.m):
            if c >> w & 1:
                x ^= (v >> (w * target.N)) & target.full
        rows.append(x)
    return affine_rank(rows)


def general_lower_bound(target: Target) -> dict:
    """Any chained circuit: outputs lie in free + span(gates), so k >= d = dim of the outputs modulo free functions
    (on the full rows). If k = d, the first gate (a rank-2 product, mixed rank <= 2) equals a nonzero output
    combination modulo free functions; so if every combination outside the free space has mixed rank > 2,
    k >= d + 1."""
    ranks = {c: combo_rank(target, c) for c in range(1, 1 << target.m)}
    free = [c for c, r in ranks.items() if r == 0]
    d = target.m - rank(free)
    nonfree = [r for r in ranks.values() if r > 0]
    plus = int(bool(nonfree) and min(nonfree) > 2)
    return dict(output_dim=d, min_combo_mixed_rank=min(nonfree) if nonfree else 0, bound=d + plus,
                combo_ranks={str(c): r for c, r in ranks.items()})


# ------------------------------------------------------------------------------------------------ constructions

def construct_rank2(target: Target):
    """Per output, a basis of the normalised full-row span relative to the first full row, paired into rank-2
    products (u1 + sigma)(u2 + tau). Returns (circuit, rowbits for every row or None where a partial row must be
    searched)."""
    N, full = target.N, target.full
    rows = target.full_rows()
    v0 = rows[0]
    gates, e, P, per_out = [], [], [], []
    for w in range(target.m):
        part = lambda v: (v >> (w * N)) & full  # noqa: E731
        B = basis([part(v) ^ part(v0) for v in rows])
        per_out.append(B)
        mask, Pw = 0, part(v0)
        for i in range(0, len(B), 2):
            u1 = B[i]
            u2 = B[i + 1] if i + 1 < len(B) else 0
            j = len(gates)
            gates.append((u2, 0, u1, 0))      # (u2 + sig)(u1 + tau) = u1 u2 + sig u1 + tau u2 + sig tau
            mask |= 1 << j
            Pw ^= u1 & u2
        e.append(mask)
        P.append(Pw)
    circ = Circuit(n=target.n, m=target.m, gates=gates, e=e, P=P)
    rowbits, j0 = [], []
    start = 0
    for B in per_out:
        j0.append(start)
        start += -(-len(B) // 2)
    for care, tts in target.rows:
        if care != full:
            rowbits.append(None)
            continue
        sig = tau = 0
        for w, B in enumerate(per_out):
            x = coords(target.norm(tts[w]) ^ ((v0 >> (w * N)) & full), B)
            assert x is not None
            for i in range(len(B)):
                if x >> i & 1:
                    j = j0[w] + i // 2
                    if i % 2 == 0:
                        sig |= 1 << j
                    else:
                        tau |= 1 << j
        rowbits.append((sig, tau))
    return circ, rowbits


def anf(f: int, n: int) -> list[int]:
    a = [(f >> y) & 1 for y in range(1 << n)]
    for i in range(n):
        for y in range(1 << n):
            if y >> i & 1:
                a[y] ^= a[y ^ (1 << i)]
    return a


def split_linear(f: int, n: int) -> tuple[int, int]:
    """(linear part over the n unit monomials, nonlinear ANF part) of a truth table, constant dropped."""
    a = anf(f, n)
    lin = sum(a[1 << i] << i for i in range(n))
    nl = sum(a[y] << y for y in range(1 << n) if bin(y).count("1") >= 2)
    return lin, nl


def linear_fiber_dim(circ: Circuit) -> dict:
    """Over all 4^k row settings: group outputs by their nonlinear ANF part; the largest affine rank of the linear
    parts within one group (one nonlinear fibre). A depth-1 circuit gives <= 2k."""
    groups: dict = {}
    table = circ.evaluate_all()
    for row in table:
        lin, nl = 0, []
        for w, f in enumerate(row):
            l_, q = split_linear(int(f), circ.n)
            lin |= l_ << (w * circ.n)
            nl.append(q)
        groups.setdefault(tuple(nl), set()).add(lin)
    best = max(groups.items(), key=lambda kv: affine_rank(sorted(kv[1])))
    return dict(dim=affine_rank(sorted(best[1])), fiber=list(best[0]), points=sorted(best[1]), fibers=len(groups))


def lin_tt(c: int, n: int) -> int:
    return sum((bin(c & y).count("1") & 1) << y for y in range(1 << n))


def copy_gate_example() -> Circuit:
    """n = 3, m = 2: g1 = (y1 + s1)(y2 + t1); g2 = (A + s2)(A + t2) with A = y1 y2 + y3 + g1 (a copy gate:
    g2 = (1 + s2 + t2) A + s2 t2, and A is linear in y for every row setting). Out0 = y1 y2 + g1, out1 = g2. Its
    linear fibre has affine dimension 5 with 2 ANDs; rank-2 products need 3."""
    n = 3
    y1, y2, y3 = lin_tt(1, n), lin_tt(2, n), lin_tt(4, n)
    a2 = (y1 & y2) ^ y3
    return Circuit(n=n, m=2, gates=[(y1, 0, y2, 0), (a2, 1, a2, 1)], e=[1, 2], P=[y1 & y2, 0])


def image_target(circ: Circuit, settings) -> Target:
    """The target whose rows are the circuit's outputs at the given (sig, tau) settings (full care)."""
    full = (1 << (1 << circ.n)) - 1
    return Target(n=circ.n, m=circ.m, rows=[(full, circ.evaluate(s, t)) for s, t in settings])


# ------------------------------------------------------------------------------------------------ CNF encoding

class CNF:
    """CNF builder. xor_native=True: every parity chain built by `xor_all` becomes one
    native XOR constraint (for CryptoMiniSat's Gauss–Jordan) instead of a Tseitin chain; `xors` holds
    (literals, rhs) meaning XOR(literals) == rhs. Two-input `XOR` (used by the lex/gauge clauses) stays clausal."""

    def __init__(self, xor_native: bool = False):
        self.nv = 0
        self.clauses: list[list[int]] = []
        self.names: dict[str, int] = {}
        self.xor_native = xor_native
        self.xors: list[tuple[list[int], bool]] = []

    def var(self, name: str | None = None) -> int:
        self.nv += 1
        if name is not None:
            self.names[name] = self.nv
        return self.nv

    def add(self, *cl):
        self.clauses.append(list(cl))

    def const(self, v: int, value: bool):
        self.add(v if value else -v)

    def AND(self, a: int, b: int) -> int:
        x = self.var()
        self.add(-x, a)
        self.add(-x, b)
        self.add(x, -a, -b)
        return x

    def XOR(self, a: int, b: int) -> int:
        x = self.var()
        self.add(-x, a, b)
        self.add(-x, -a, -b)
        self.add(x, -a, b)
        self.add(x, a, -b)
        return x

    def xor_all(self, lits: list[int], rhs_const: bool | None = None):
        """Returns a literal equal to the XOR of lits (a fresh false literal for an empty list)."""
        if not lits:
            z = self.var()
            self.add(-z)
            return z
        if self.xor_native and len(lits) > 1:
            x = self.var()
            self.xors.append((list(lits) + [x], False))  # x = XOR(lits)
            return x
        acc = lits[0]
        for l_ in lits[1:]:
            acc = self.XOR(acc, l_)
        return acc

    def dimacs(self) -> str:
        """DIMACS; XOR constraints use CryptoMiniSat's 'x' lines (an 'x' clause asserts the XOR is true)."""
        head = [f"p cnf {self.nv} {len(self.clauses) + len(self.xors)}"]
        body = [" ".join(map(str, c)) + " 0" for c in self.clauses]
        for lits, rhs in self.xors:
            lits = list(lits)
            if not rhs:
                lits[0] = -lits[0]
            body.append("x" + " ".join(map(str, lits)) + " 0")
        return "\n".join(head + body) + "\n"

    def xor_vars_rhs(self):
        """XOR constraints as (positive variables, rhs) for pycryptosat's add_xor_clause."""
        for lits, rhs in self.xors:
            par = rhs
            for l_ in lits:
                if l_ < 0:
                    par = not par
            yield [abs(l_) for l_ in lits], par


def _levels_ok(depth: int | None, k: int):
    """Allowed chain pairs (j, i), i < j: none for depth 1, all otherwise (other caps: pass `levels`)."""
    if depth not in (None, 1):
        raise ValueError("depth caps other than 1 need an explicit `levels` layering")
    return [(j, i) for j in range(k) for i in range(j)] if depth != 1 else []


def encode_cnf(target: Target, k: int, depth: int | None = None, rows=None, levels=None) -> CNF:
    """CNF: a k-AND chained circuit meets the selected rows (default all). depth=1 forbids chains; `levels` (a list
    of gate levels, nondecreasing) allows gate j to chain on gate i only if levels[i] < levels[j]; depth=None with
    levels=None allows every chain i < j (any depth up to k). WLOG a_j(0) = b_j(0) = P_w(0) = 0 (row bits and
    output constants absorb them)."""
    c = CNF()
    N = target.N
    rows = list(range(len(target.rows))) if rows is None else list(rows)
    if levels is not None:
        pairs = [(j, i) for j in range(k) for i in range(j) if levels[i] < levels[j]]
    else:
        pairs = _levels_ok(depth, k)
    a = [[c.var(f"a{j}_{y}") for y in range(N)] for j in range(k)]
    b = [[c.var(f"b{j}_{y}") for y in range(N)] for j in range(k)]
    for j in range(k):
        c.add(-a[j][0])
        c.add(-b[j][0])
    ca = {(j, i): c.var(f"ca{j}_{i}") for j, i in pairs}
    cb = {(j, i): c.var(f"cb{j}_{i}") for j, i in pairs}
    e = [[c.var(f"e{w}_{j}") for j in range(k)] for w in range(target.m)]
    P = [[c.var(f"P{w}_{y}") for y in range(N)] for w in range(target.m)]
    for w in range(target.m):
        c.add(-P[w][0])
    for r in rows:
        care, tts = target.rows[r]
        s = [c.var(f"s{r}_{j}") for j in range(k)]
        t = [c.var(f"t{r}_{j}") for j in range(k)]
        q = [c.var(f"q{r}_{w}") for w in range(target.m)]
        g = [[0] * N for _ in range(k)]
        for y in range(N):
            for j in range(k):
                la = [a[j][y], s[j]] + [c.AND(ca[(j, i)], g[i][y]) for (jj, i) in pairs if jj == j]
                lb = [b[j][y], t[j]] + [c.AND(cb[(j, i)], g[i][y]) for (jj, i) in pairs if jj == j]
                g[j][y] = c.AND(c.xor_all(la), c.xor_all(lb))
                c.names[f"g{r}_{j}_{y}"] = g[j][y]
            if not care >> y & 1:
                continue
            for w in range(target.m):
                lits = [P[w][y], q[w]] + [c.AND(e[w][j], g[j][y]) for j in range(k)]
                x = c.xor_all(lits)
                c.const(x, bool(tts[w] >> y & 1))
    c.names["_rows"] = rows   # type: ignore[assignment]
    return c


def decode_model(c: CNF, model: set[int], target: Target, k: int):
    """Circuit and row bits from a satisfying assignment (set of true variable numbers)."""
    N = target.N
    val = lambda name: c.names[name] in model  # noqa: E731
    gates = []
    for j in range(k):
        A = sum(val(f"a{j}_{y}") << y for y in range(N))
        B = sum(val(f"b{j}_{y}") << y for y in range(N))
        ma = sum((f"ca{j}_{i}" in c.names and val(f"ca{j}_{i}")) << i for i in range(j))
        mb = sum((f"cb{j}_{i}" in c.names and val(f"cb{j}_{i}")) << i for i in range(j))
        gates.append((A, ma, B, mb))
    e = [sum(val(f"e{w}_{j}") << j for j in range(k)) for w in range(target.m)]
    P = [sum(val(f"P{w}_{y}") << y for y in range(N)) for w in range(target.m)]
    circ = Circuit(n=target.n, m=target.m, gates=gates, e=e, P=P)
    rowbits = {r: (sum(val(f"s{r}_{j}") << j for j in range(k)), sum(val(f"t{r}_{j}") << j for j in range(k)))
               for r in c.names["_rows"]}   # type: ignore[index]
    return circ, rowbits


def assignment_from(c: CNF, circ: Circuit, target: Target, rowbits: dict) -> set[int] | None:
    """Propagate a known circuit (with a(0) = b(0) = P(0) = 0) through the encoding: returns the set of true
    variables, or None if some clause is violated (used to test the encoding)."""
    k, N = circ.k, target.N
    for a_, _, b_, _ in circ.gates:
        if a_ & 1 or b_ & 1:
            return None
    if any(p & 1 for p in circ.P):
        return None
    true: set[int] = set()
    known: dict[int, bool] = {}

    def setv(name, v):
        known[c.names[name]] = bool(v)

    for j, (A, ma, B, mb) in enumerate(circ.gates):
        for y in range(N):
            setv(f"a{j}_{y}", A >> y & 1)
            setv(f"b{j}_{y}", B >> y & 1)
        for i in range(j):
            if f"ca{j}_{i}" in c.names:
                setv(f"ca{j}_{i}", ma >> i & 1)
                setv(f"cb{j}_{i}", mb >> i & 1)
            elif (ma | mb) >> i & 1:
                return None
    for w in range(target.m):
        for j in range(k):
            setv(f"e{w}_{j}", circ.e[w] >> j & 1)
        for y in range(N):
            setv(f"P{w}_{y}", circ.P[w] >> y & 1)
    full = target.full
    for r in c.names["_rows"]:   # type: ignore[union-attr]
        sig, tau = rowbits[r]
        for j in range(k):
            setv(f"s{r}_{j}", sig >> j & 1)
            setv(f"t{r}_{j}", tau >> j & 1)
        outs = circ.evaluate(sig, tau)
        care, tts = target.rows[r]
        for w in range(target.m):
            d = (outs[w] ^ tts[w]) & care
            setv(f"q{r}_{w}", d == care and care != 0)
    # every remaining variable is a Tseitin definition: propagate in clause order until fixed point
    changed = True
    while changed:
        changed = False
        for cl in c.clauses:
            unk = [l_ for l_ in cl if abs(l_) not in known]
            if any(known.get(abs(l_)) == (l_ > 0) for l_ in cl if abs(l_) in known):
                continue
            if len(unk) == 1:
                known[abs(unk[0])] = unk[0] > 0
                changed = True
        for lits, rhs in c.xors:   # native XOR definitions (xor_native): fix the one unknown literal
            unk = [l_ for l_ in lits if abs(l_) not in known]
            if len(unk) != 1:
                continue
            par = rhs
            for l_ in lits:
                if abs(l_) in known:
                    par ^= known[abs(l_)] == (l_ > 0)
            known[abs(unk[0])] = par == (unk[0] > 0)
            changed = True
    # variables left unforced (one-directional symmetry-breaking auxiliaries) default to false
    true = {v for v, x in known.items() if x}
    if not check_assignment(c, true):
        return None
    return true


def check_assignment(c: CNF, true: set[int]) -> bool:
    if not all(any((l_ > 0) == (abs(l_) in true) for l_ in cl) for cl in c.clauses):
        return False
    for lits, rhs in c.xors:
        par = False
        for l_ in lits:
            par ^= (l_ > 0) == (abs(l_) in true)
        if par != rhs:
            return False
    return True


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", help="target JSON (ybits, outputs, rows[{care, tt}])")
    ap.add_argument("--out", help="write a JSON report here")
    ap.add_argument("--cnf-k", type=int, help="also write the CNF for k ANDs (DIMACS) to --cnf")
    ap.add_argument("--cnf", help="DIMACS path")
    ap.add_argument("--depth", type=int, help="chain depth cap for the CNF (1 = rank-2 products only)")
    a = ap.parse_args(argv)
    T = Target.from_json(a.target)
    circ, rb = construct_rank2(T)
    v = verify(circ, T, rowbits=None if any(x is None for x in rb) else rb)
    if not v["all_pass"]:
        v = verify(circ, T)
    rep = dict(version=__version__, target=a.target, rank2=rank2_lower_bound(T), general=general_lower_bound(T),
               rank2_construction=dict(ands=circ.k, verify={x: v[x] for x in v if x != "rowbits"},
                                       circuit=circ.to_dict()))
    if a.cnf_k:
        c = encode_cnf(T, a.cnf_k, depth=a.depth)
        Path(a.cnf).write_text(c.dimacs())
        rep["cnf"] = dict(k=a.cnf_k, depth=a.depth, vars=c.nv, clauses=len(c.clauses), path=a.cnf)
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(json.dumps({x: rep[x] for x in ("rank2",)}), rep["general"]["bound"], rep["rank2_construction"]["ands"],
          v["all_pass"])


if __name__ == "__main__":
    main()


# ================================================================================================ reductions (0.2.0)
# Exact search-space reductions R1-R7. Each preserves SAT/UNSAT of "a k-AND chained circuit meets the target"; the
# soundness arguments are summarised in the docstrings. Nothing here needs a solver except `cegar`, which calls an external DIMACS solver by command.

def _norm_care(f: int, care: int, full: int) -> int:
    """f on its care points, shifted so that the lowest care point reads 0 (per-output row constant removed)."""
    low = (care & -care).bit_length() - 1
    return (f ^ (full if f >> low & 1 else 0)) & care


def mixed_matrix(target: Target) -> list[int]:
    """R1. The doubly normalised target: one integer per full row z, bit (w * 2^n + y) = t_w(y, z) + t_w(0, z) +
    t_w(y, z0) + t_w(0, z0) with z0 the first full row. Its GF(2) rank is the mixed rank (= affine rank of the
    normalised full rows). Every free function (any function of y alone or of z alone) maps to zero."""
    rows = target.full_rows()
    return [v ^ rows[0] for v in rows] if rows else []


def mixed_columns(target: Target) -> dict:
    """R1. Column vectors of the doubly normalised target over the full rows: for every output w and y, the bit
    vector over full rows (row index = position in full_rows()). Returns dict (w, y) -> int and the rank."""
    M = mixed_matrix(target)
    N, cols = target.N, {}
    for w in range(target.m):
        for y in range(1, N):
            cols[(w, y)] = sum(((v >> (w * N + y)) & 1) << z for z, v in enumerate(M))
    return dict(columns=cols, rank=rank(cols.values()), rows=len(M))


def gate_mixed_parts(circ: Circuit, rowbits) -> list[list[int]]:
    """R1. For every gate j, its doubly normalised matrix over the given rows (list of (sig, tau) per row): list of
    row integers (bit y). Needs the full y-truth table of every gate at every row, computed from the row bits."""
    full = (1 << (1 << circ.n)) - 1
    per_gate = [[] for _ in range(circ.k)]
    for sig, tau in rowbits:
        g = []
        for j, (a, ca, b, cb) in enumerate(circ.gates):
            A, B = a, b
            for i in range(j):
                if ca >> i & 1:
                    A ^= g[i]
                if cb >> i & 1:
                    B ^= g[i]
            if sig >> j & 1:
                A ^= full
            if tau >> j & 1:
                B ^= full
            g.append(A & B)
        for j in range(circ.k):
            per_gate[j].append(g[j] ^ (full if g[j] & 1 else 0))     # row-normalised (y = 0 reads 0)
    return [[v ^ rows[0] for v in rows] for rows in per_gate]           # doubly normalised


def column_rank(rows: list[int]) -> int:
    """Rank of a 0/1 matrix given as row integers (= its column rank)."""
    return rank(rows)


# ---- R2: row and column collapse

def dominates(target: Target, r: int, rp: int) -> bool:
    """Row r dominates row rp: care(rp) subset of care(r) and the two agree on care(rp) up to one constant per
    output. Any circuit meeting r (with row bits sigma, tau) meets rp with the same row bits."""
    care, tts = target.rows[r]
    carep, ttsp = target.rows[rp]
    if carep & ~care:
        return False
    return all(((x ^ y) & carep) in (0, carep) for x, y in zip(tts, ttsp))


def row_classes(target: Target) -> list[list[int]]:
    """R2. Greedy domination classes: each class [rep, dropped...]; a dropped row is dominated by its rep. The
    reduced instance keeps the reps only. Soundness: dropping rows is a relaxation; a solution of the reps lifts by
    giving every dropped row its rep's row bits (the circuit's y-function at a row depends only on the row bits)."""
    classes: list[list[int]] = []
    for r in range(len(target.rows)):
        for cl in classes:
            if dominates(target, cl[0], r):
                cl.append(r)
                break
            if dominates(target, r, cl[0]):
                cl.insert(0, r)
                break
        else:
            classes.append([r])
    return classes


def reduce_rows(target: Target):
    """R2. (reduced target with representative rows only, classes)."""
    classes = row_classes(target)
    T = Target(n=target.n, m=target.m, rows=[target.rows[cl[0]] for cl in classes], names=target.names,
               meta=dict(target.meta, reduced_rows=[cl[0] for cl in classes]))
    return T, classes


def lift_rowbits(rowbits_reduced, classes, nrows: int):
    """R2. Row bits for every original row from the reduced instance's row bits (same bits as the class rep)."""
    out = [None] * nrows
    for bits, cl in zip(rowbits_reduced, classes):
        for r in cl:
            out[r] = bits
    return out


def column_merges(target: Target) -> list[tuple[int, int, list[int]]]:
    """R2. Pairs (y_keep, y_drop, constants c_w): column y_drop agrees with column y_keep on every row where both
    are care, up to one constant per output (the same constant on every row), and is care only where y_keep is.
    Soundness: a solution with y_drop as don't-care extends by a(y_drop) = a(y_keep), b likewise for every gate
    (so every gate value agrees on the two columns at every row) and P_w(y_drop) = P_w(y_keep) + c_w."""
    N, m, merges, dropped = target.N, target.m, [], set()
    for yd in range(N - 1, 0, -1):            # drop the higher column in favour of a lower one
        for yk in range(yd):
            if yk == yd or yk in dropped:
                continue
            ok, consts = True, [None] * m
            for care, tts in target.rows:
                if care >> yd & 1 and not care >> yk & 1:
                    ok = False
                    break
                if not (care >> yd & 1):
                    continue
                for w in range(m):
                    d = ((tts[w] >> yd) ^ (tts[w] >> yk)) & 1
                    if consts[w] is None:
                        consts[w] = d
                    elif consts[w] != d:
                        ok = False
            if ok:
                merges.append((yk, yd, [c or 0 for c in consts]))
                dropped.add(yd)
                break
    return merges


def reduce_columns(target: Target):
    """R2. Target with merged columns turned into don't-cares, and the merges."""
    merges = column_merges(target)
    drop = 0
    for _, yd, _ in merges:
        drop |= 1 << yd
    T = Target(n=target.n, m=target.m, rows=[(care & ~drop, tts) for care, tts in target.rows], names=target.names,
               meta=dict(target.meta, merged_columns=[list(x) for x in merges]))
    return T, merges


def lift_columns(circ: Circuit, merges) -> Circuit:
    """R2. Copy column y_keep into y_drop in every operand and output y-function (P_w gets the constant)."""
    gates, P = [list(g) for g in circ.gates], list(circ.P)
    for yk, yd, consts in merges:
        for g in gates:
            for idx in (0, 2):
                g[idx] = (g[idx] & ~(1 << yd)) | (((g[idx] >> yk) & 1) << yd)
        for w in range(circ.m):
            P[w] = (P[w] & ~(1 << yd)) | ((((P[w] >> yk) & 1) ^ consts[w]) << yd)
    return Circuit(n=circ.n, m=circ.m, gates=[tuple(g) for g in gates], e=list(circ.e), P=P)


# ---- R3: output basis

def output_basis(target: Target):
    """R3. An invertible m x m GF(2) matrix L (rows as masks over the old outputs) that puts the outputs' mixed
    parts into reduced echelon form (outputs that are free functions last). Only the span of the mixed parts
    matters: a circuit for the old outputs becomes one for the new by e' = L e, P' = L P, q' = L q."""
    M = mixed_matrix(target)
    N, m = target.N, target.m
    vecs = []
    for w in range(m):
        vecs.append(sum((((v >> (w * N)) & target.full) << (z * N)) for z, v in enumerate(M)))
    L = [1 << w for w in range(m)]
    rowsv = list(vecs)
    for i in range(m):                         # Gauss-Jordan on the mixed parts, tracking the combinations in L
        best = max(range(i, m), key=lambda r: rowsv[r].bit_length())
        if rowsv[best] == 0:
            break
        rowsv[i], rowsv[best] = rowsv[best], rowsv[i]
        L[i], L[best] = L[best], L[i]
        p = rowsv[i].bit_length() - 1
        for r in range(m):
            if r != i and rowsv[r] >> p & 1:
                rowsv[r] ^= rowsv[i]
                L[r] ^= L[i]
    return L


def apply_output_map(target: Target, L: list[int]) -> Target:
    rows = []
    for care, tts in target.rows:
        new = []
        for w in range(target.m):
            x = 0
            for v in range(target.m):
                if L[w] >> v & 1:
                    x ^= tts[v]
            new.append(x)
        rows.append((care, new))
    return Target(n=target.n, m=target.m, rows=rows, names=target.names, meta=dict(target.meta, output_map=list(L)))


def map_circuit_outputs(circ: Circuit, L: list[int]) -> Circuit:
    e, P = [], []
    for w in range(circ.m):
        ew = pw = 0
        for v in range(circ.m):
            if L[w] >> v & 1:
                ew ^= circ.e[v]
                pw ^= circ.P[v]
        e.append(ew)
        P.append(pw)
    return Circuit(n=circ.n, m=circ.m, gates=list(circ.gates), e=e, P=P)


def invert_output_map(L: list[int]) -> list[int]:
    m = len(L)
    rows = [(L[w], 1 << w) for w in range(m)]
    for i in range(m):
        best = next(r for r in range(i, m) if rows[r][0] >> (m - 1 - i) & 1) if any(
            rows[r][0] >> (m - 1 - i) & 1 for r in range(i, m)) else None
        if best is None:
            raise ValueError("singular output map")
        rows[i], rows[best] = rows[best], rows[i]
        for r in range(m):
            if r != i and rows[r][0] >> (m - 1 - i) & 1:
                rows[r] = (rows[r][0] ^ rows[i][0], rows[r][1] ^ rows[i][1])
    inv = [0] * m
    for i in range(m):
        inv[m - 1 - i] = rows[i][1]
    return inv


# ---- R4: operand gauge

def operand_key(mask: int, yfun: int, j: int, N: int) -> int:
    """Integer key of an operand: chain mask (most significant) then the y-part on y = 1..N-1."""
    return (mask << (N - 1)) | (yfun >> 1)


def gauge_gate(gate, j: int, N: int):
    """R4. Canonical representative of the GL(2,2) orbit of one gate's operand pair: of the three nonzero span
    elements u, v, u+v (keys over (chain mask, y-part)), the new first operand is the one with the largest key and
    the new second operand the one with the smallest. Products: u v = g, u (u+v) = g + u, v (u+v) = g + v, so the
    gate value changes by x in {0, u, v}, an element of span{free, earlier gates}. Returns (new gate, (hi, lo), x)
    with hi, lo indices into [u, v, u+v] and x = (mask, y-part) of the change (its z-part is a per-row bit)."""
    a, ca, b, cb = gate
    orbit = [(ca, a), (cb, b), (ca ^ cb, a ^ b)]
    keys = [operand_key(mk, f, j, N) for mk, f in orbit]
    order = sorted(range(3), key=lambda i: keys[i])
    lo, hi = order[0], order[-1]
    used = {hi, lo}
    x = (0, 0) if used == {0, 1} else orbit[0] if used == {0, 2} else orbit[1]
    u, v = orbit[hi], orbit[lo]
    return (u[1], u[0], v[1], v[0]), (hi, lo), x


def _gauge_steps(circ: Circuit):
    gates = [list(g) for g in circ.gates]
    e, P = list(circ.e), list(circ.P)
    N = 1 << circ.n
    choices = []
    for j in range(circ.k):
        new, choice, (xm, xf) = gauge_gate(tuple(gates[j]), j, N)
        gates[j] = list(new)
        choices.append(choice)
        if xm or xf:
            for jj in range(j + 1, circ.k):
                for mi, fi in ((1, 0), (3, 2)):
                    if gates[jj][mi] >> j & 1:
                        gates[jj][mi] ^= xm
                        gates[jj][fi] ^= xf
            for w in range(circ.m):
                if e[w] >> j & 1:
                    e[w] ^= xm
                    P[w] ^= xf
    return Circuit(n=circ.n, m=circ.m, gates=[tuple(g) for g in gates], e=e, P=P), choices


def gauge_canonical(circ: Circuit) -> Circuit:
    """R4. Canonicalise every gate in order j = 0..k-1. Gate j's value changes by x_j in span{free, g_<j}; x_j's
    (mask, y-part) is pushed into every later operand reading gate j and into every output using it (e_w ^= mask,
    P_w ^= y-part); its z-part is a per-row bit and is pushed into the per-row bits (see gauge_rowbits). The
    operand *functions* of every later gate are unchanged by the push, so the canonical circuit computes the same
    outputs with adjusted row bits."""
    return _gauge_steps(circ)[0]


def _operand_functions(circ: Circuit, sig: int, tau: int):
    """Per gate, (U, V, g) truth tables over y at one row with per-row bits sig, tau."""
    full = (1 << (1 << circ.n)) - 1
    out, g = [], []
    for j, (a, ca, b, cb) in enumerate(circ.gates):
        U, V = a ^ (full if sig >> j & 1 else 0), b ^ (full if tau >> j & 1 else 0)
        for i in range(j):
            if ca >> i & 1:
                U ^= g[i]
            if cb >> i & 1:
                V ^= g[i]
        g.append(U & V)
        out.append((U, V, U & V))
    return out


def gauge_rowbits(circ: Circuit, rowbits):
    """R4. Row bits for gauge_canonical(circ) from the original row bits, exactly: at every row the canonical
    gate j's operands are orbit[hi], orbit[lo] of the original (U_j, V_j, U_j + V_j) as functions of y, so the new
    per-row bit is the constant by which that function differs from a_new(y) + (chain of new gate values)."""
    can, choices = _gauge_steps(circ)
    full = (1 << (1 << circ.n)) - 1
    out = []
    for sig, tau in rowbits:
        orig = _operand_functions(circ, sig, tau)
        cs = ct = 0
        gnew = []
        for j, (a, ca, b, cb) in enumerate(can.gates):
            U, V, _ = orig[j]
            orbit = [U, V, U ^ V]
            hi, lo = choices[j]
            A, B = orbit[hi], orbit[lo]
            ZA, ZB = a, b
            for i in range(j):
                if ca >> i & 1:
                    ZA ^= gnew[i]
                if cb >> i & 1:
                    ZB ^= gnew[i]
            da, db = A ^ ZA, B ^ ZB
            assert da in (0, full) and db in (0, full), "gauge_rowbits: operand difference is not a row constant"
            cs |= (da == full) << j
            ct |= (db == full) << j
            gnew.append(A & B)
        out.append((cs, ct))
    return out


# ---- R5: gate order

def plain_gates(circ: Circuit) -> list[int]:
    return [j for j, g in enumerate(circ.gates) if not (g[1] | g[3])]


def reorder_gates(circ: Circuit, perm: list[int]) -> Circuit:
    """Circuit with gate perm[i] moved to position i (perm must respect chain dependencies)."""
    pos = {old: new for new, old in enumerate(perm)}
    remap = lambda mask: sum(1 << pos[i] for i in range(circ.k) if mask >> i & 1)  # noqa: E731
    gates = []
    for old in perm:
        a, ca, b, cb = circ.gates[old]
        assert all(pos[i] < pos[old] for i in range(circ.k) if (ca | cb) >> i & 1)
        gates.append((a, remap(ca), b, remap(cb)))
    return Circuit(n=circ.n, m=circ.m, gates=gates, e=[remap(x) for x in circ.e], P=list(circ.P))


def reorder_rowbits(rowbits, perm: list[int]):
    out = []
    for sig, tau in rowbits:
        out.append((sum(((sig >> old) & 1) << new for new, old in enumerate(perm)),
                    sum(((tau >> old) & 1) << new for new, old in enumerate(perm))))
    return out


def canonical_order(circ: Circuit) -> list[int]:
    """R5. Plain (depth-1) gates first, sorted by (a, b) key; then chained gates in their original (topological)
    order. Plain gates commute with everything, so this is a valid reorder of any circuit."""
    N = 1 << circ.n
    plain = sorted(plain_gates(circ), key=lambda j: (circ.gates[j][0] >> 1, circ.gates[j][2] >> 1))
    chained = [j for j in range(circ.k) if j not in set(plain)]
    return plain + chained


def canonicalise(circ: Circuit, rowbits=None):
    """R4 + R5 together: gauge-fix plain gates, sort plain gates first, gauge-fix chained gates in order. Returns
    (circuit, rowbits or None). The result satisfies the gauge and lex constraints of `encode_shape`."""
    c1 = gauge_canonical(circ)
    rb = gauge_rowbits(circ, rowbits) if rowbits is not None else None
    perm = canonical_order(c1)
    c2 = reorder_gates(c1, perm)
    rb = reorder_rowbits(rb, perm) if rb is not None else None
    c3 = gauge_canonical(c2)          # chained gates' keys depend on the mask numbering: fix again (plain unchanged)
    rb = gauge_rowbits(c2, rb) if rb is not None else None
    return c3, rb


# ---- R6: shapes

def shapes(k: int, mixed_rank: int) -> dict:
    """R6. Shape = number I of chained gates (gates with a nonzero chain mask). Every k-gate circuit has exactly one
    I in 0..k-1 (gate 0 cannot chain) and can be reordered to plain gates first, chained gates after in topological
    order (R5), so the shapes are complete. A plain gate's doubly normalised matrix has column rank <= 2 (its
    columns lie in span{s_j, t_j}), so I = 0 needs 2k >= mixed rank: shapes violating that are excluded by rank."""
    out = {}
    for I in range(k):
        plain = k - I
        feasible = True
        reason = ""
        if I == 0 and 2 * plain < mixed_rank:
            feasible, reason = False, f"depth-1 column rank <= 2 per gate: 2*{plain} < {mixed_rank}"
        out[I] = dict(chained=I, plain=plain, rank_feasible=feasible, reason=reason)
    return out


def _lex_ge(c: "CNF", X: list[int], Y: list[int]) -> None:
    """Clauses satisfiable iff X >= Y lexicographically (most significant first), one-directional aux encoding."""
    alpha = c.var()
    c.add(alpha)
    for x, y in zip(X, Y):
        nxt = c.var()
        c.add(-alpha, x, -y)                 # agree so far -> x >= y here
        c.add(-alpha, -x, -y, nxt)           # agree so far and equal here -> agree next
        c.add(-alpha, x, y, nxt)
        alpha = nxt


def encode_shape(target: Target, k: int, chained: int, rows=None, gauge: bool = True, lex: bool = True,
                 force_chain: bool = True, xor_native: bool = False) -> "CNF":
    """R1/R4/R5/R6 encoding: k gates, the first k - chained plain (direct 8-clause product encoding, no chain
    variables), the last `chained` gates with chain variables on every earlier gate (nonzero masks when
    force_chain). gauge: per gate, key(u) >= key(v), key(u) >= key(u+v), key(v) <= key(u+v) with key = (chain mask,
    y-part); lex: consecutive plain gates in non-decreasing (a, b) order. WLOG a(0) = b(0) = P(0) = 0."""
    c = CNF(xor_native=xor_native)
    N = target.N
    rows = list(range(len(target.rows))) if rows is None else list(rows)
    p = k - chained
    assert 0 <= chained < k or (k == 0 and chained == 0)
    a = [[c.var(f"a{j}_{y}") for y in range(N)] for j in range(k)]
    b = [[c.var(f"b{j}_{y}") for y in range(N)] for j in range(k)]
    for j in range(k):
        c.add(-a[j][0])
        c.add(-b[j][0])
    ca = {(j, i): c.var(f"ca{j}_{i}") for j in range(p, k) for i in range(j)}
    cb = {(j, i): c.var(f"cb{j}_{i}") for j in range(p, k) for i in range(j)}
    e = [[c.var(f"e{w}_{j}") for j in range(k)] for w in range(target.m)]
    P = [[c.var(f"P{w}_{y}") for y in range(N)] for w in range(target.m)]
    for w in range(target.m):
        c.add(-P[w][0])
    if force_chain:
        for j in range(p, k):
            c.add(*[ca[(j, i)] for i in range(j)], *[cb[(j, i)] for i in range(j)])
    if gauge:
        for j in range(k):
            ku = [ca[(j, i)] for i in reversed(range(j))] if j >= p else []
            kv = [cb[(j, i)] for i in reversed(range(j))] if j >= p else []
            ku += [a[j][y] for y in reversed(range(1, N))]
            kv += [b[j][y] for y in reversed(range(1, N))]
            kuv = [c.XOR(x, y) for x, y in zip(ku, kv)]
            _lex_ge(c, ku, kv)
            _lex_ge(c, ku, kuv)
            _lex_ge(c, kuv, kv)
    if lex:
        for j in range(p - 1):
            kj = [a[j][y] for y in reversed(range(1, N))] + [b[j][y] for y in reversed(range(1, N))]
            kn = [a[j + 1][y] for y in reversed(range(1, N))] + [b[j + 1][y] for y in reversed(range(1, N))]
            _lex_ge(c, kn, kj)
    for r in rows:
        care, tts = target.rows[r]
        s = [c.var(f"s{r}_{j}") for j in range(k)]
        t = [c.var(f"t{r}_{j}") for j in range(k)]
        q = [c.var(f"q{r}_{w}") for w in range(target.m)]
        g = [[0] * N for _ in range(k)]
        for y in range(N):
            for j in range(k):
                if j < p:
                    x = c.var()
                    if y == 0:
                        c.add(-x, s[j]); c.add(-x, t[j]); c.add(x, -s[j], -t[j])
                    else:
                        A, S, B, Tt = a[j][y], s[j], b[j][y], t[j]
                        # x -> (A xor S) and (B xor Tt)
                        c.add(-x, A, S); c.add(-x, -A, -S); c.add(-x, B, Tt); c.add(-x, -B, -Tt)
                        # not x -> (A == S) or (B == Tt)
                        c.add(x, A, -S, B, -Tt); c.add(x, A, -S, -B, Tt)
                        c.add(x, -A, S, B, -Tt); c.add(x, -A, S, -B, Tt)
                    g[j][y] = x
                else:
                    la = [a[j][y], s[j]] + [c.AND(ca[(j, i)], g[i][y]) for i in range(j)]
                    lb = [b[j][y], t[j]] + [c.AND(cb[(j, i)], g[i][y]) for i in range(j)]
                    g[j][y] = c.AND(c.xor_all(la), c.xor_all(lb))
                c.names[f"g{r}_{j}_{y}"] = g[j][y]
            if not care >> y & 1:
                continue
            for w in range(target.m):
                lits = [P[w][y], q[w]] + [c.AND(e[w][j], g[j][y]) for j in range(k)]
                c.const(c.xor_all(lits), bool(tts[w] >> y & 1))
    c.names["_rows"] = rows          # type: ignore[assignment]
    c.names["_shape"] = dict(k=k, chained=chained, plain=p, gauge=gauge, lex=lex, force_chain=force_chain)  # type: ignore[assignment]
    return c


def spanning_rows(target: Target) -> list[int]:
    """Row indices (in target order) whose doubly normalised rows form a basis of the mixed row space, plus every
    partial row (greedy, first rows first)."""
    full_idx = [r for r, (care, _) in enumerate(target.rows) if care == target.full]
    M = mixed_matrix(target)
    piv: dict[int, int] = {}
    chosen = [full_idx[0]] if full_idx else []
    for z, v in enumerate(M):
        w = v
        while w:
            pbit = w.bit_length() - 1
            if pbit in piv:
                w ^= piv[pbit]
            else:
                piv[pbit] = w
                chosen.append(full_idx[z])
                break
    chosen += [r for r, (care, _) in enumerate(target.rows) if care != target.full]
    return sorted(set(chosen))


# ---- R7: CEGAR with an external solver

def run_solver(cnf_path, solver: str = "kissat", timeout_s: float = 600, proof_path=None, extra=()) -> dict:
    """Run kissat or cadical on a DIMACS file. Returns status ('SAT', 'UNSAT', 'UNKNOWN'), the model (set of true
    variables) for SAT, the exit code, CPU seconds of the child and wall seconds. A timeout is 'UNKNOWN'."""
    import resource
    import subprocess
    import time as _time
    if solver == "kissat":
        cmd = ["kissat", f"--time={int(timeout_s)}", *extra, str(cnf_path)]
        if proof_path:
            cmd.append(str(proof_path))
    elif solver == "cadical":
        cmd = ["cadical", "-t", str(int(timeout_s)), *extra, str(cnf_path)]
        if proof_path:
            cmd.append(str(proof_path))
    elif solver == "cms":   # CryptoMiniSat (pycryptosat, Gauss–Jordan; XOR lines); no proof interface
        if proof_path:
            raise ValueError("cms: no proof interface")
        import os
        import sys
        py = os.environ.get("CMS_PYTHON") or sys.executable or "python3"   # any interpreter with pycryptosat
        cmd = [py, "-I", str(__import__("pathlib").Path(__file__).with_name("cms_solve.py")), str(cnf_path),
               str(int(timeout_s))]
    else:
        raise ValueError(solver)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = _time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True)
    wall = _time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    status, lits = "UNKNOWN", []
    for line in pr.stdout.splitlines():
        if line.startswith("s "):
            st = line[2:].strip()
            status = {"SATISFIABLE": "SAT", "UNSATISFIABLE": "UNSAT"}.get(st, "UNKNOWN")
        elif line.startswith("v "):
            lits += [int(x) for x in line[2:].split()]
    if pr.returncode == 10:
        status = "SAT"
    elif pr.returncode == 20:
        status = "UNSAT"
    return dict(cmd=cmd, status=status, exit=pr.returncode, cpu_s=round(cpu, 3), wall_s=round(wall, 3),
                model={x for x in lits if x > 0} if status == "SAT" else None,
                stdout_tail=pr.stdout[-2000:], stderr_tail=pr.stderr[-500:])


def cegar(target: Target, k: int, chained: int, workdir, solver: str = "kissat", timeout_s: float = 600,
          init_rows=None, add_rows: int = 1, max_iters: int = 60, budget_s: float | None = None,
          proof: bool = False, gauge: bool = True, lex: bool = True, force_chain: bool = True, log=print,
          xor_native: bool | None = None) -> dict:
    """R7. Solve on a row subset, verify any model exhaustively on the whole target, add failing rows, repeat.
    UNSAT on a subset is UNSAT for the whole target (subsets drop constraints only). Returns a dict with status
    'SAT' (circuit verified on every row), 'UNSAT' (rows used, proof path if written) or 'unresolved' (timeout or
    budget), and the per-iteration log (rows, vars, clauses, solver status, cpu, wall)."""
    from pathlib import Path as _P
    workdir = _P(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = list(init_rows) if init_rows is not None else spanning_rows(target)
    iters, cpu_total, wall_total = [], 0.0, 0.0
    result = dict(status="unresolved", k=k, chained=chained, solver=solver, iterations=iters)
    for it in range(max_iters):
        cnf = encode_shape(target, k, chained, rows=rows, gauge=gauge, lex=lex, force_chain=force_chain,
                           xor_native=(solver == "cms") if xor_native is None else xor_native)
        path = workdir / f"iter{it:02d}_rows{len(rows)}.cnf"
        path.write_text(cnf.dimacs())
        proof_path = (workdir / f"iter{it:02d}_rows{len(rows)}.proof") if proof else None
        remaining = timeout_s if budget_s is None else max(1.0, min(timeout_s, budget_s - cpu_total))
        res = run_solver(path, solver, remaining, proof_path)
        cpu_total += res["cpu_s"]
        wall_total += res["wall_s"]
        entry = dict(iter=it, rows=list(rows), n_rows=len(rows), vars=cnf.nv, clauses=len(cnf.clauses), xors=len(cnf.xors),
                     status=res["status"], exit=res["exit"], cpu_s=res["cpu_s"], wall_s=res["wall_s"],
                     timeout_s=remaining, cnf=str(path), proof=str(proof_path) if proof_path else None)
        if res["status"] == "UNSAT":
            iters.append(entry)
            log(f"  iter {it}: rows {len(rows)} vars {cnf.nv} clauses {len(cnf.clauses)} -> UNSAT in {res['cpu_s']}s")
            result.update(status="UNSAT", rows=rows, cpu_s=round(cpu_total, 2), wall_s=round(wall_total, 2),
                          unsat_cnf=str(path), proof=str(proof_path) if proof_path else None)
            return result
        if res["status"] != "SAT":
            entry["note"] = "timeout or unknown: unresolved, not a proof"
            iters.append(entry)
            log(f"  iter {it}: rows {len(rows)} vars {cnf.nv} clauses {len(cnf.clauses)} -> {res['status']} after {res['cpu_s']}s")
            result.update(status="unresolved", rows=rows, cpu_s=round(cpu_total, 2), wall_s=round(wall_total, 2))
            return result
        if proof_path and proof_path.exists():
            proof_path.unlink()          # a proof of a SAT instance is not evidence
            entry["proof"] = None
        circ, rb = decode_model(cnf, res["model"], target, k)
        assert check_assignment(cnf, res["model"])
        v_sub = verify(circ, target, rowbits=[rb.get(r, (0, 0)) for r in range(len(target.rows))])
        assert all(v_sub["rowbits"][r] is not None for r in rows), "decoded model fails its own rows"
        v = verify(circ, target)
        entry.update(decoded_depth=circ.depth(), rows_met_exhaustive=v["rows_met"], failed=v["failed_rows"][:10])
        iters.append(entry)
        log(f"  iter {it}: rows {len(rows)} vars {cnf.nv} clauses {len(cnf.clauses)} -> SAT in {res['cpu_s']}s; "
            f"exhaustive {v['rows_met']}/{v['rows']}")
        if v["all_pass"]:
            result.update(status="SAT", rows=rows, circuit=circ.to_dict(), rowbits=v["rowbits"],
                          verify={x: v[x] for x in v if x != "rowbits"}, cpu_s=round(cpu_total, 2),
                          wall_s=round(wall_total, 2))
            return result
        new = [r for r in v["failed_rows"] if r not in rows][:add_rows]
        assert new
        rows = sorted(rows + new)
        if budget_s is not None and cpu_total >= budget_s:
            result.update(status="unresolved", rows=rows, cpu_s=round(cpu_total, 2), wall_s=round(wall_total, 2),
                          note="CPU budget exhausted")
            return result
    result.update(status="unresolved", rows=rows, cpu_s=round(cpu_total, 2), wall_s=round(wall_total, 2),
                  note="iteration cap")
    return result


# ---- exact depth-1 search and a planted positive control

def pure_tensors(target: Target) -> dict:
    """Depth-1 algebra. R = row space of the doubly normalised target (vectors over (w, y)). For every output mask e
    the set {a : e (x) a in R} (a over y, a(0) = 0) by brute force. A k-product depth-1 circuit with 2k = rank R
    exists iff R has a basis of 2k pure tensors e_j (x) a_j, e_j (x) b_j paired by mask (each gate's mixed part is
    sigma_j b_j + tau_j a_j in the outputs e_j, so the rows of the target lie in their span, and equality of
    dimensions forces a basis)."""
    N, m = target.N, target.m
    M = mixed_matrix(target)
    B = basis(M)
    piv: dict[int, int] = {}
    for vec in B:
        w = vec
        while w:
            pb = w.bit_length() - 1
            if pb in piv:
                w ^= piv[pb]
            else:
                piv[pb] = w
                break

    def in_R(v):
        while v:
            pb = v.bit_length() - 1
            if pb in piv:
                v ^= piv[pb]
            else:
                return False
        return True
    out = {}
    for e in range(1, 1 << m):
        out[e] = [a for a in range(2, 1 << N, 2) if in_R(sum(a << (w * N) for w in range(m) if e >> w & 1))]
    return dict(rank=len(B), tensors=out)


def depth1_exact(target: Target, k: int | None = None, time_limit_s: float = 600.0) -> dict:
    """Exact search for a depth-1 circuit with k = rank/2 products (the tight case): DFS over gates (e, a, b) with
    pure tensors in R, keeping the chosen tensors independent; masks nondecreasing (gates commute). Returns status
    'SAT' with a verified circuit, 'UNSAT' (search exhausted: no pure-tensor basis), or 'unresolved' (time)."""
    import time as _time
    N, m, full = target.N, target.m, target.full
    pt = pure_tensors(target)
    r = pt["rank"]
    if k is None:
        k = r // 2
    if 2 * k < r:
        return dict(status="UNSAT", reason=f"2k = {2 * k} < rank {r}", rank=r, k=k)
    if 2 * k > r:
        return dict(status="unresolved", reason="depth1_exact handles the tight case 2k = rank only", rank=r, k=k)
    cands = {e: lst for e, lst in pt["tensors"].items() if len(lst) >= 2}
    es = sorted(cands)
    t0 = _time.time()
    found: list = []
    nodes = [0]
    timed_out = [False]

    def tensor(e, a):
        return sum(a << (w * N) for w in range(m) if e >> w & 1)

    def reduce(v, piv):
        while v:
            pb = v.bit_length() - 1
            if pb in piv:
                v ^= piv[pb]
            else:
                return v
        return 0

    def dfs(start, piv, chosen):
        nodes[0] += 1
        if len(chosen) == k:
            found.append(list(chosen))
            return True
        if _time.time() - t0 > time_limit_s:
            timed_out[0] = True
            return True
        for ei in range(start, len(es)):
            e = es[ei]
            lst = cands[e]
            for ia, a in enumerate(lst):
                wa = reduce(tensor(e, a), piv)
                if not wa:
                    continue
                piv2 = dict(piv)
                piv2[wa.bit_length() - 1] = wa
                for b in lst[ia + 1:]:
                    wb = reduce(tensor(e, b), piv2)
                    if not wb:
                        continue
                    piv3 = dict(piv2)
                    piv3[wb.bit_length() - 1] = wb
                    chosen.append((e, a, b))
                    if dfs(ei, piv3, chosen):
                        return True
                    chosen.pop()
        return False

    dfs(0, {}, [])
    res = dict(rank=r, k=k, nodes=nodes[0], seconds=round(_time.time() - t0, 3),
               tensors_per_mask={str(e): len(v) for e, v in pt["tensors"].items() if v})
    if timed_out[0]:
        return dict(res, status="unresolved", reason="time limit")
    if not found:
        return dict(res, status="UNSAT", reason="no pure-tensor basis of the row space")
    sol = found[0]
    v0 = target.full_rows()[0]
    gates = [(a, 0, b, 0) for e, a, b in sol]
    e_masks = [sum(1 << j for j, (e, a, b) in enumerate(sol) if e >> w & 1) for w in range(m)]
    P = []
    for w in range(m):
        pw = (v0 >> (w * N)) & full
        for e, a, b in sol:
            if e >> w & 1:
                pw ^= a & b
        P.append(pw)
    circ = Circuit(n=target.n, m=m, gates=gates, e=e_masks, P=P)
    ver = verify(circ, target)
    return dict(res, status="SAT" if ver["all_pass"] else "unresolved", circuit=circ.to_dict(),
                verify={x: ver[x] for x in ver if x != "rowbits"}, rowbits=ver["rowbits"],
                reason="" if ver["all_pass"] else "pure-tensor basis found but the partial row is not met")


LOADER4_MIRROR_PLAN = dict(
    # linear forms as masks over y1..y4; plain gates (a, b); copy gates (gates read, fresh form A); output masks
    plain=[(1, 2), (1, 2), (4, 8), (4, 8), (3, 4), (6, 9), (5, 10)],
    copies=[((0, 1), 8), ((2, 3), 15)],
    e=[0b000000101, 0b000001010, 0b000110000, 0b011000000, 0b101000000])


def plant_linear_chained(n: int, m: int, rows: int, rng, plan=LOADER4_MIRROR_PLAN, partial_care: int | None = None):
    """A target of the loader kind (every row exactly affine in y) with a known circuit of shape I = len(copies):
    plain depth-1 gates with linear operands from `plan`, then copy gates (A + G + s)(A + G + t) = (1 + s + t)(A + G)
    + s t reading a pair of plain gates with equal operands (their XOR G is linear in y at every row). P_w cancels
    the z-independent quadratic parts, so rows are affine. Random row bits. Returns (target, circuit, rowbits).
    The default plan mirrors loader 4: 7 plain + 2 copy gates, five outputs of mixed rank 4, total mixed rank 20."""
    N = 1 << n
    full = (1 << N) - 1
    gates = [(lin_tt(a, n), 0, lin_tt(b, n), 0) for a, b in plan["plain"]]
    for reads, A in plan["copies"]:
        mask = sum(1 << i for i in reads)
        gates.append((lin_tt(A, n), mask, lin_tt(A, n), mask))
    k = len(gates)
    e = list(plan["e"])
    assert len(e) == m
    P = []
    for w in range(m):
        pw = lin_tt(rng.getrandbits(n), n)
        for j, (a, ca, b, cb) in enumerate(gates):
            if e[w] >> j & 1 and not (ca | cb):
                pw ^= a & b
        P.append(pw)
    circ = Circuit(n=n, m=m, gates=gates, e=e, P=P)
    rowbits = [(rng.getrandbits(k), rng.getrandbits(k)) for _ in range(rows)]
    trows = []
    for r, (sig, tau) in enumerate(rowbits):
        outs = [o ^ (full if rng.random() < 0.5 else 0) for o in circ.evaluate(sig, tau)]
        care = full if (partial_care is None or r < rows - 1) else partial_care
        trows.append((care, outs))
    T = Target(n=n, m=m, rows=trows, names=[f"planted{w}" for w in range(m)],
               meta=dict(planted=dict(k=k, chained=len(plan["copies"]), plan=plan)))
    return T, circ, rowbits
