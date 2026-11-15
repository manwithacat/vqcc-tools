#!/usr/bin/env python3
"""qhdl plugin: `track` — symbolic value tracker for circuits of any size.

    qhdl track -c CIRCUIT [--region R | --ops A:B] [--warm] [--idle] [--show uncomputes|noops|zeros|cut|idle]
               [--emit-script edits.txt]

Generalises two earlier single-purpose DQI tracker prototypes (zero tracking and an affine span check, written by an
AI agent): hash-consed products, a GF(2)-affine span test
at peak-run boundaries. Changes: one streaming pass over the .kmb op streams in chunks (no op table), exact handling of
conditions (an op under PUSH_CONDITION / `if b` acts as ctx * effect instead of making its target opaque), symbol ids
recycled after a liveness sweep (so values stay short on 60M-op circuits), and classical bits dropped after their
last read.

Model. Every value (qubit or classical bit) is a GF(2) polynomial written as an XOR of symbols (a Python int bitset;
bit 0 is the constant 1). Symbols are: the input value of each wire at the start of the span (registers always;
scratch wires only if they may be dirty there), each measurement outcome, and one symbol per distinct product of two
polynomials (hash-consed on the exact operand pair, so recomputing the same product yields the same symbol and an
uncompute cancels). Everything the tracker states is a theorem about every shot:
  zero     the wire is |0> in every shot and for every input (sound: a proof).
  affine   the value is GF(2)-affine in the span-start input symbols (no products, no outcomes).
  cnot-recomputable  the value lies in the affine span of the OTHER wires' current values, so CNOTs (and an X)
           alone can uncompute it to |0> and rebuild it later (0 Toffolis).
  opaque   none of the above was PROVED. Opaque is not "needed": products are compared by exact operand identity,
           so two different expressions of the same function look different (e.g. a*(b+c) vs a*b + a*c).
Phases (Z, CZ, CCZ, NEG, Z_POW, the HMR kickback) are ignored: the tracker reasons about values only. An MBU rewrite
must still be checked (fdiff on invsqrt; the DQI draw gate / verifier).

Start state. With --ops/--region the default is a cold start at A: register wires get fresh symbols, a scratch wire
is |0> if its last touch before A was an unconditional reset (R/HMR at condition depth 0) or it was never touched,
otherwise fresh; classical bits are unknown until written. Relations created before A are therefore lost (sound,
weaker). --warm tracks from op 0 (exact relations, slower). Whole-circuit runs always start at op 0.

Reports (in the span):
  uncomputes  Toffolis (CCX) whose target is provably |0> right after them while it was not provably |0> before:
              measurement-based-uncomputation candidates. `mbu` names a replacement whose correctness follows from the
              proof: 'single' (t held a*b: HMR t -> m; CZ a b if m) or 'and3-dirty' (the dirty-borrow AND3 block
              CCX(a,b,d) CCX(d,c,t) CCX(a,b,d) CCX(d,c,t) with t zero after it, so t held a*b*c before it:
              HMR t -> m; CCZ a b c if m). --emit-script writes these as a `qhdl edit` script (one new classical bit,
              id = num_bits, shared by all sites).
  noops       Toffolis whose effect is provably zero (deletable) or provably linear (a CNOT/X would do).
  zeros       sites where a non-reset op leaves a wire provably |0> (an R could be inserted there; allocator food).
  cut         every wire's class at the end of the span.
  idle (--idle) for each wire idle across peak runs (from `qhdl peak`), whether it is cnot-recomputable at each
              run's start and end; a run is freeable when some idle wire is recomputable at both ends.
Library: track(path, A, B, ...) -> dict.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_kmb as K  # noqa: E402

CHUNK = 1 << 17
GC_EVERY = 1 << 17  # frequent sweeps keep symbol ids (and so int lengths) small; memo closure keeps nested products
MEMO_MAX = 1_500_000
SHOW = ('uncomputes', 'noops', 'zeros', 'cut', 'idle')


class Store:
    """Symbols, hash-consed products and garbage collection with id recycling."""

    def __init__(self):
        self.nsym = 1  # bit 0 = constant 1
        self.free = []
        self.memo = {}
        self.inputs = 0  # mask of span-start input symbols

    def fresh(self):
        if self.free:
            return 1 << self.free.pop()
        self.nsym += 1
        return 1 << (self.nsym - 1)

    def prod(self, x, y):
        if not x or not y:
            return 0
        if x == 1:
            return y
        if y == 1 or x == y:
            return x
        if x ^ y == 1:
            return 0
        cx, cy = x & 1, y & 1
        x0, y0 = x ^ cx, y ^ cy
        if x0 == y0:  # (x0+cx)(x0+cy) = x0 (1 + cx + cy) + cx cy
            r = x0 if not (cx ^ cy) else 0
            return r ^ (cx & cy)
        key = (x0, y0) if x0 < y0 else (y0, x0)
        m = self.memo.get(key)
        if m is None:
            m = self.memo[key] = self.fresh()
        if cy:
            m ^= x0
        if cx:
            m ^= y0
        if cx and cy:
            m ^= 1
        return m

    @staticmethod
    def _closure(core, memo):
        """Memo entries whose operands can still be rebuilt: keys made of live symbols, where a product symbol is
        live if a kept entry produces it (fixpoint: nested products such as (d + a*b)*c). Returns (kept, live)."""
        live = core
        keep = {}
        pending = memo
        while pending:
            rest = {}
            for key, m in pending.items():
                if (key[0] | key[1]) & ~live:
                    rest[key] = m
                else:
                    keep[key] = m
                    live |= m
            if len(rest) == len(pending):
                break
            pending = rest
        return keep, live

    def gc(self, roots):
        """Keep symbols reachable from roots (iterable of ints) and from surviving memo entries; recycle the rest."""
        core = 0
        for v in roots:
            core |= v
        core &= ~1
        keep, live = self._closure(core, self.memo)
        if len(keep) > MEMO_MAX:  # bound memory: drop the oldest entries (sound: only loses precision)
            keep, live = self._closure(core, dict(list(keep.items())[-MEMO_MAX // 2:]))
        self.memo = keep
        self.inputs &= live
        nb = self.nsym
        dead = ~live & ((1 << nb) - 1) & ~1
        if dead:
            raw = np.frombuffer(dead.to_bytes((nb + 7) // 8, 'little'), np.uint8)
            ids = np.flatnonzero(np.unpackbits(raw, bitorder='little'))
            self.free = ids[::-1].tolist()  # pop() returns the lowest id first
        else:
            self.free = []
        # shrink nsym while the top ids are free
        top = live.bit_length()
        if top < self.nsym:
            self.free = [f for f in self.free if f < top]
            self.nsym = max(top, 1)
        return len(self.memo), bin(live).count('1')


def dependent_wires(vals):
    """Bitmask of wires whose value lies in the affine span of the other wires' values (constant 1 is free)."""
    basis = {}
    deps = 0
    for w, v in enumerate(vals):
        x = v & ~1
        combo = 1 << w
        while x:
            h = x.bit_length() - 1
            b = basis.get(h)
            if b is None:
                basis[h] = (x, combo)
                break
            x ^= b[0]
            combo ^= b[1]
        if not x:
            deps |= combo
    return deps


def _cold_state(k, A, regs):
    """Per wire: True if provably |0> at cut A (last touch an unconditional depth-0 reset, or never touched)."""
    t = k.op_types
    nq = int(k.num_qubits)
    last = np.full(nq, -1, np.int64)
    last_reset = np.full(nq, -2, np.int64)
    pushes = np.flatnonzero(np.asarray(t[:A]) == K.PUSH_CONDITION)
    pops = np.flatnonzero(np.asarray(t[:A]) == K.POP_CONDITION)
    p3 = p2 = p1 = 0
    for c0 in range(0, A, 8 * CHUNK):
        c1 = min(A, c0 + 8 * CHUNK)
        seg = np.asarray(t[c0:c1])
        cls = seg & 0x30
        for cc, names in ((0x30, ('qqq0', 'qqq1', 'qqq2')), (0x20, ('qq0', 'qq1')), (0x10, ('q0',))):
            idx = np.flatnonzero(cls == cc)
            if cc == 0x30:
                off = p3
                p3 += len(idx)
            elif cc == 0x20:
                off = p2
                p2 += len(idx)
            else:
                off = p1
                p1 += len(idx)
            if not len(idx):
                continue
            ops = idx + c0
            for nm in names:
                q = np.asarray(k.stream(nm)[off:off + len(idx)]).astype(np.int64)
                np.maximum.at(last, q, ops)
            if cc == 0x10:
                q = np.asarray(k.stream('q0')[off:off + len(idx)]).astype(np.int64)
                rs = (seg[idx] == K.R) | (seg[idx] == K.HMR)
                if rs.any():
                    ro = ops[rs]
                    depth = np.searchsorted(pushes, ro) - np.searchsorted(pops, ro)
                    ok = depth == 0
                    np.maximum.at(last_reset, q[rs][ok], ro[ok])
    zero = (last < 0) | (last == last_reset)
    zero[list(regs)] = False
    return zero.tolist(), int(len(pushes) - len(pops))


def _last_bit_access(k):
    t = k.op_types
    nb = int(k.num_bits)
    last = np.full(max(nb, 1), -1, np.int64)
    pc = pb = 0
    for c0 in range(0, len(t), 8 * CHUNK):
        seg = np.asarray(t[c0:c0 + 8 * CHUNK])
        for flag, nm in ((0x80, 'bc'), (0x40, 'b0')):
            idx = np.flatnonzero(seg & flag)
            if not len(idx):
                continue
            off = pc if nm == 'bc' else pb
            b = np.asarray(k.stream(nm)[off:off + len(idx)]).astype(np.int64)
            np.maximum.at(last, b, idx + c0)
            if nm == 'bc':
                pc += len(idx)
            else:
                pb += len(idx)
    return last


def track(path, A=None, B=None, warm=False, checkpoints=(), max_events=200000, progress=None, macros=None,
          cold_state=None, last_bit=None, k=None, offsets=None):
    """Symbolic pass over ops [A, B) of a .kmb. Returns a dict (see the module docstring).

    macros: optional {start: (end, target, [(control wire, negated)])} — op blocks [start, end) whose exact net action
    is target ^= AND(literals) with every other wire restored (qhdl_census.find_chains proves this). Each is applied
    as one hash-consed multi-control product, so a compute chain and its uncompute chain cancel even when the chain
    borrows dirty wires. Reported in `macro_events` as (start, target, zero_before, zero_after).
    cold_state=(zero list, depth), last_bit, k and offsets (stream positions p3, p2, p1, pb, pc at A): precomputed by a caller that tracks many windows of one circuit
    (qhdl_census local proofs), so each window costs only its own ops."""
    t0 = time.time()
    k = K.read(str(path)) if k is None else k
    t = k.op_types
    N = len(t)
    A = 0 if A is None else A
    B = N if B is None else B
    if not 0 <= A < B <= N:
        raise ValueError(f'op range {A}:{B} outside the circuit (0..{N})')
    nq = int(k.num_qubits)
    regs = sorted({q for _, cs in k.registers for isq, q, _ in cs if isq and q < nq})
    S = Store()
    start = 0 if (warm or A == 0) else A
    bits_unknown = start > 0
    if start == 0:
        e = [0] * nq
        for w in regs:
            v = S.fresh()
            e[w] = v
            S.inputs |= v
        stack = []
    else:
        zero, depth = _cold_state(k, A, regs) if cold_state is None else cold_state
        e = [0] * nq
        for w in range(nq):
            if not zero[w]:
                v = S.fresh()
                e[w] = v
                S.inputs |= v
        stack = [S.fresh() for _ in range(depth)]
    ctxs = [1]
    for v in stack:
        ctxs.append(S.prod(ctxs[-1], v))
    bits = {}
    last_bit = _last_bit_access(k) if last_bit is None else last_bit
    # stream offsets at `start`
    if offsets is not None and start == A:
        p3, p2, p1, pb, pc = offsets
    else:
        pre = np.asarray(t[:start])
        pcls = pre & 0x30
        p3 = int(np.count_nonzero(pcls == 0x30))
        p2 = int(np.count_nonzero(pcls == 0x20))
        p1 = int(np.count_nonzero(pcls == 0x10))
        pb = int(np.count_nonzero(pre & 0x40))
        pc = int(np.count_nonzero(pre & 0x80))
        del pre, pcls
    st = {nm: k.stream(nm) for nm in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')}
    unc, noops, zsites = [], [], []
    nz_dropped = 0
    cps = sorted({c for c in checkpoints if A <= c <= B})
    cp_res = {}
    macros = {m: v for m, v in (macros or {}).items() if start <= m and v[0] <= B and
              not any(m < c < v[0] for c in cps)}
    inside = []
    for m, v in macros.items():
        inside.append((m, v[0]))
    inside.sort()
    ins_s = np.array([x for x, _ in inside], np.int64)
    ins_e = np.array([y for _, y in inside], np.int64)

    def in_macro(b):
        j = int(np.searchsorted(ins_s, b, 'right')) - 1
        return j >= 0 and ins_s[j] < b < ins_e[j]
    bounds = sorted(set([b for b in range(start, B, CHUNK) if not in_macro(b)] + [c for c in cps if start <= c < B]
                        + [A, B] + list(macros) + [v[0] for v in macros.values()]))
    bounds = [b for b in bounds if start <= b <= B]
    mev = []
    next_gc = start + GC_EVERY
    gcs = []
    ci = 0

    def bitval(b):
        v = bits.get(b)
        if v is None:
            v = S.fresh() if bits_unknown else 0
            bits[b] = v
        return v

    for seg_i in range(len(bounds) - 1):
        c0, c1 = bounds[seg_i], bounds[seg_i + 1]
        while ci < len(cps) and cps[ci] == c0:
            cp_res[c0] = dependent_wires(e)
            ci += 1
        if c0 >= next_gc:
            roots = list(e) + list(stack) + ctxs
            bits = {b: v for b, v in bits.items() if last_bit[b] >= c0}
            roots += list(bits.values())
            gcs.append((c0,) + S.gc(roots))
            next_gc = c0 + GC_EVERY
            if progress:
                progress(c0, N, time.time() - t0)
        seg = np.asarray(t[c0:c1])
        n = len(seg)
        cls = seg & 0x30
        QA = np.full(n, -1, np.int64)
        QB = np.full(n, -1, np.int64)
        QC = np.full(n, -1, np.int64)
        i3 = np.flatnonzero(cls == 0x30)
        i2 = np.flatnonzero(cls == 0x20)
        i1 = np.flatnonzero(cls == 0x10)
        QA[i3] = st['qqq0'][p3:p3 + len(i3)]
        QB[i3] = st['qqq1'][p3:p3 + len(i3)]
        QC[i3] = st['qqq2'][p3:p3 + len(i3)]
        QA[i2] = st['qq0'][p2:p2 + len(i2)]
        QB[i2] = st['qq1'][p2:p2 + len(i2)]
        QA[i1] = st['q0'][p1:p1 + len(i1)]
        p3 += len(i3)
        p2 += len(i2)
        p1 += len(i1)
        ib = np.flatnonzero(seg & 0x40)
        ic = np.flatnonzero(seg & 0x80)
        BB = np.full(n, -1, np.int64)
        BC = np.full(n, -1, np.int64)
        BB[ib] = st['b0'][pb:pb + len(ib)]
        BC[ic] = st['bc'][pc:pc + len(ic)]
        pb += len(ib)
        pc += len(ic)
        rec = c0 >= A
        T_, QA_, QB_, QC_, BB_, BC_ = seg.tolist(), QA.tolist(), QB.tolist(), QC.tolist(), BB.tolist(), BC.tolist()
        del QA, QB, QC, BB, BC, seg, cls
        ctx = ctxs[-1]
        prod = S.prod
        mac = macros.get(c0)
        if mac is not None and mac[0] == c1:
            _, tq, lits = mac
            vs = sorted((e[w] ^ (1 if neg else 0)) for w, neg in lits)
            p = ctx
            for v in vs:
                p = prod(p, v)
            old = e[tq]
            nv = old ^ p
            e[tq] = nv
            if rec:
                mev.append((c0, tq, not old, not nv))
            continue
        for j in range(n):
            op = T_[j]
            if op == 0x21 and ctx == 1:  # CX, the bulk of DQI
                s_ = QA_[j]
                v = e[s_]
                if v:
                    tq = QB_[j]
                    nv = e[tq] ^ v
                    e[tq] = nv
                    if not nv and rec:
                        zsites.append((c0 + j, tq, 'CX'))
                continue
            if op == 0x06:
                continue
            if op == 0x81:
                b = bitval(BC_[j])
                stack.append(b)
                ctxs.append(prod(ctxs[-1], b))
                ctx = ctxs[-1]
                continue
            if op == 0x01:
                if stack:
                    stack.pop()
                    ctxs.pop()
                ctx = ctxs[-1]
                continue
            c = ctx
            if op & 0x80:
                c = prod(ctx, bitval(BC_[j]))
            if not c:
                continue
            base = op & 0x7F
            if base == 0x21:
                tq = QB_[j]
                old = e[tq]
                nv = old ^ prod(c, e[QA_[j]])
                e[tq] = nv
                if rec and not nv and old:
                    zsites.append((c0 + j, tq, 'CX'))
            elif base == 0x31:
                a_, b_, tq = QA_[j], QB_[j], QC_[j]
                ea, eb = e[a_], e[b_]
                pab = prod(ea, eb)
                p = prod(c, pab)
                old = e[tq]
                nv = old ^ p
                e[tq] = nv
                if rec:
                    if not p:
                        if len(noops) < max_events:
                            noops.append((c0 + j, tq, 'no-op', op))
                    elif ea in (0, 1) or eb in (0, 1) or ea == eb or (ea ^ eb) == 1:
                        if len(noops) < max_events:
                            noops.append((c0 + j, tq, 'linear', op))
                    if not nv and old:
                        if len(unc) < max_events:
                            unc.append((c0 + j, tq, op, c == 1 and not stack, old == pab and c == 1))
                        zsites.append((c0 + j, tq, 'CCX'))
            elif base == 0x11:
                tq = QA_[j]
                e[tq] ^= c
                if rec and not e[tq]:
                    zsites.append((c0 + j, tq, 'X'))
            elif base == 0x12:  # R
                tq = QA_[j]
                e[tq] = 0 if c == 1 else e[tq] ^ prod(c, e[tq])
            elif base == 0x50:  # HMR: q := 0, bit := outcome
                tq = QA_[j]
                e[tq] = 0 if c == 1 else e[tq] ^ prod(c, e[tq])
                bits[BB_[j]] = S.fresh()
            elif base == 0x22:
                a_, b_ = QA_[j], QB_[j]
                if c == 1:
                    e[a_], e[b_] = e[b_], e[a_]
                else:
                    d = prod(c, e[a_] ^ e[b_])
                    e[a_] ^= d
                    e[b_] ^= d
                if rec:
                    for w in (a_, b_):
                        if not e[w] and e[a_ if w == b_ else b_]:
                            zsites.append((c0 + j, w, 'SWAP'))
            elif base == 0x32:
                cw, a_, b_ = QA_[j], QB_[j], QC_[j]
                d = prod(c, prod(e[cw], e[a_] ^ e[b_]))
                if d:
                    e[a_] ^= d
                    e[b_] ^= d
                    if rec:
                        for w in (a_, b_):
                            if not e[w]:
                                zsites.append((c0 + j, w, 'CSWAP'))
            elif base == 0x40:
                b = BB_[j]
                bits[b] = 0 if c == 1 else bitval(b) ^ prod(c, bitval(b))
            elif base == 0x41:
                b = BB_[j]
                bits[b] = 1 if c == 1 else bitval(b) ^ prod(c, bitval(b) ^ 1)
            elif base == 0x42:
                b = BB_[j]
                bits[b] = bitval(b) ^ c
            elif base in (0x10, 0x20, 0x30, 0x13, 0x00):
                pass  # phase only
            else:
                raise ValueError(f'op {c0 + j}: unhandled op type {hex(op)}')
        if len(zsites) > max_events:
            nz_dropped += len(zsites) - max_events
            zsites = zsites[:max_events]
    while ci < len(cps) and cps[ci] <= B:
        cp_res[cps[ci]] = dependent_wires(e)
        ci += 1
    deps = dependent_wires(e)
    regset = set(regs)
    cut = []
    for w in range(nq):
        v = e[w]
        kind = 'zero' if v == 0 else 'one' if v == 1 else (
            'affine' if not ((v & ~1) & ~S.inputs) else 'opaque')
        cut.append(dict(wire=f'q{w}', role='register' if w in regset else 'scratch', kind=kind,
                        cnot_recomputable=bool(deps >> w & 1), terms=bin(v).count('1')))
    return dict(path=str(path), num_ops=N, num_qubits=nq, A=A, B=B, start=start, cold=start > 0,
                seconds=round(time.time() - t0, 1), uncomputes=unc, macro_events=mev, noops=noops, zero_sites=zsites,
                zero_sites_dropped=nz_dropped, cut=cut, checkpoints=cp_res, gcs=gcs, registers=regs,
                num_bits=int(k.num_bits), symbols_at_end=S.nsym)


# ---------------------------------------------------------------- MBU pattern matching on the reported events
def classify_uncomputes(path, events):
    """For each (op, wire, code, uncond, single) event: the replacement pattern its proof supports, with operands."""
    import qhdl_edit as ED
    k = K.read(str(path))
    t = k.op_types
    want = sorted({i - d for i, *_ in events for d in range(4) if i - d >= 0})
    src = ED.Source(k)
    recs = {}
    for i in want:  # sorted: Source.at is incremental
        src.at(i)
        recs[i] = _record_at(src, i)
    out = []
    for i, w, code, uncond, single in events:
        r = dict(op=i, wire=w, conditional=not uncond, mbu='', edit='', ops=f'{i}..{i}')
        q = recs[i]['q']
        if uncond and all(i - d >= 0 and int(t[i - d]) == 0x31 for d in range(4)):
            o = [recs[i - 3 + d]['q'] for d in range(4)]
            a, b, d_ = o[0]
            d2, cc, tt = o[1]
            if d2 != d_:
                d2, cc = cc, d2
            ok = (o[2] == o[0] and o[3] == o[1] and d2 == d_ and tt == w and
                  len({a, b, cc, d_, tt}) == 5)
            if ok:
                r.update(mbu='and3-dirty', ops=f'{i - 3}..{i}', a=a, b=b, c=cc, dirty=d_,
                         edit=f'replace {i - 3}:{i + 1} => HMR q{tt} b{{NEW}}; CCZ q{a} q{b} q{cc} if b{{NEW}}')
        if not r['mbu'] and uncond and single and int(t[i]) == 0x31:
            a, b, tt = q
            r.update(mbu='single', a=a, b=b, edit=f'replace {i} => HMR q{tt} b{{NEW}}; CZ q{a} q{b} if b{{NEW}}')
        out.append(r)
    return out, int(k.num_bits)


def _record_at(src, i):
    c = src.cnt
    code = int(src.t[i])
    cls = code & 0x30
    q = ()
    if code != 0x81:
        if cls == 0x30:
            q = tuple(int(src.src[n][c['q3']]) for n in ('qqq0', 'qqq1', 'qqq2'))
        elif cls == 0x20:
            q = tuple(int(src.src[n][c['q2']]) for n in ('qq0', 'qq1'))
        elif cls == 0x10:
            q = (int(src.src['q0'][c['q1']]),)
    return dict(code=code, q=q)


# ---------------------------------------------------------------- CLI verb
def _span(api, a, k_path):
    import qhdl_peak as P
    if a.ops and a.region:
        raise api.QhdlError('pass --ops or --region, not both')
    k = K.read(str(k_path))
    n = len(k.op_types)
    if a.ops:
        try:
            A, B = (int(x) for x in a.ops.split(':'))
        except ValueError:
            raise api.QhdlError(f'bad --ops {a.ops!r} (use A:B)')
        if not 0 <= A < B <= n:
            raise api.QhdlError(f'--ops {a.ops} outside the circuit (0..{n})')
        return A, B, f'ops {A}..{B - 1}'
    if a.region:
        regs = P.regions(k)
        hit = [r for r in regs if r['name'] == a.region]
        if not hit:
            raise api.QhdlError(f'no top-level region {a.region!r}', 'regions: ' + ', '.join(r['name'] for r in regs))
        return hit[0]['start'], hit[0]['end'], f"region {a.region} (ops {hit[0]['start']}..{hit[0]['end'] - 1})"
    return 0, n, 'whole circuit'


def _peak_data(api, meta):
    import argparse
    ns = argparse.Namespace(circuit=meta['sha12'], level=None, show='runs', registers=True, all=False)
    api.v_peak(ns)
    return json.loads((api.ws_dir(meta['sha256']) / 'peak' / 'peak_peak.json').read_text())


def v_track(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    meta = api.ensure_open(a.circuit)
    sha, s12 = meta['sha256'], meta['sha12']
    src = api.circuit_path(meta)
    if src.suffix != '.kmb':
        raise api.QhdlError('track reads .kmb circuits')
    A, B, span = _span(api, a, src)
    d = api.ws_dir(sha) / 'track'
    tag = f"{A}_{B}{'_warm' if a.warm else ''}{'_idle' if a.idle else ''}"
    ev = d / f'track_{tag}.json'
    key = dict(tool_sha=api.tool_sha(), verb='track', A=A, B=B, warm=bool(a.warm), idle=bool(a.idle))
    res = api.cached(ev, key)
    if res is None:
        cps, runs, gaps = [], [], []
        if a.idle:
            pk = _peak_data(api, meta)
            runs = [(r['run'], r['start'], r['end']) for r in pk['run_rows'] if A <= r['start'] and r['end'] < B]
            cps = sorted({s for _, s, _ in runs} | {e_ + 1 for _, _, e_ in runs})
            rn = {r[0]: r for r in runs}
            for g in pk['idle_all']:
                if g['role'] != 'scratch' and not a.registers:
                    continue
                r0, r1 = (int(x[1:]) for x in g['runs'].split('..'))
                mine = [f'R{x}' for x in range(r0, r1 + 1) if f'R{x}' in rn]
                if mine:
                    gaps.append(dict(wire=g['wire'], role=g['role'], idle=g['idle'], runs=mine))
        r = track(src, A, B, warm=a.warm, checkpoints=cps)
        cls, nbits = classify_uncomputes(src, r['uncomputes'])
        idle_rows, freeable = [], set()
        if a.idle:
            rn = {x[0]: x for x in runs}
            for g in gaps:
                w = int(g['wire'][1:])
                ok = [x for x in g['runs'] if (r['checkpoints'][rn[x][1]] >> w & 1)
                      and (r['checkpoints'][rn[x][2] + 1] >> w & 1)]
                freeable.update(ok)
                idle_rows.append(dict(wire=g['wire'], role=g['role'], idle=g['idle'], runs=len(g['runs']),
                                      recomputable_runs=len(ok), first=g['runs'][0], last=g['runs'][-1]))
            idle_rows.sort(key=lambda x: (-x['recomputable_runs'], -x['runs'], int(x['wire'][1:])))
        nk = {}
        for c in r['cut']:
            nk[c['kind']] = nk.get(c['kind'], 0) + 1
        res = dict(cache_key=key, span=span, A=A, B=B, seconds=r['seconds'], cold=r['cold'],
                   uncomputes=cls, num_bits=nbits,
                   noops=[dict(op=i, wire=f'q{w}', kind=kd, conditional=bool(code & 0x80)) for i, w, kd, code in
                          r['noops']],
                   zero_sites=len(r['zero_sites']) + r['zero_sites_dropped'],
                   zero_by_op={}, cut_counts=nk, recomputable_at_cut=sum(c['cnot_recomputable'] for c in r['cut']),
                   idle=idle_rows, runs_in_span=len(runs), runs_freeable=len(freeable),
                   symbols_at_end=r['symbols_at_end'])
        for _, _, kd in r['zero_sites']:
            res['zero_by_op'][kd] = res['zero_by_op'].get(kd, 0) + 1
        d.mkdir(parents=True, exist_ok=True)
        regset = set(r['registers'])
        api.write_tsv(d / f'track_{tag}_zeros.tsv', ['op', 'wire', 'role', 'by'],
                      [[i, f'q{w}', 'register' if w in regset else 'scratch', kd] for i, w, kd in r['zero_sites']])
        api.write_tsv(d / f'track_{tag}_cut.tsv', list(r['cut'][0]), [list(c.values()) for c in r['cut']])
        res['tables'] = dict(zeros=api.rel(d / f'track_{tag}_zeros.tsv'), cut=api.rel(d / f'track_{tag}_cut.tsv'))
        res['cut'] = r['cut']
        api.write_json(ev, res)
    led.earn('static', 'sound symbolic propagation over the op stream (GF(2) polynomials, hash-consed products): '
                       "'zero' and 'affine' hold in every shot; 'opaque' means not proved, not 'needed'")
    unc = res['uncomputes']
    matched = [u for u in unc if u['mbu']]
    nn = sum(1 for x in res['noops'] if x['kind'] == 'no-op')
    nl = len(res['noops']) - nn
    cc = res['cut_counts']
    claim = (f"{s12} {res['span']} tracked in {res['seconds']} s ({'cold start at A' if res['cold'] else 'from op 0'}): "
             f"{len(unc)} Toffolis leave their target provably |0> (MBU candidates; {len(matched)} match a proved "
             f"replacement), {nn} Toffolis are provable no-ops, {nl} act linearly; {api.fmt_int(res['zero_sites'])} "
             f"zero sites (wire |0> after a non-reset op). At the end: {cc.get('zero', 0)} zero, "
             f"{cc.get('affine', 0)} affine, {cc.get('opaque', 0)} opaque wires, {res['recomputable_at_cut']} "
             'CNOT-recomputable')
    if a.idle:
        claim += (f"; {res['runs_freeable']} of {res['runs_in_span']} peak runs have an idle wire CNOT-recomputable "
                  'at both ends')
    show = a.show or ('idle' if a.idle else 'uncomputes')
    if show == 'uncomputes':
        rows = [{k: v for k, v in u.items() if k != 'edit'} for u in unc]
    elif show == 'noops':
        rows = res['noops']
    elif show == 'cut':
        rows = [c for c in res['cut'] if c['kind'] != 'zero' or c['role'] == 'register']
    elif show == 'zeros':
        import csv
        zp = Path(res['tables']['zeros'])
        with open(zp if zp.is_absolute() else api.ROOT / zp) as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
    else:
        if not a.idle:
            raise api.QhdlError('--show idle needs --idle')
        rows = res['idle']
    facts = dict(tables=res['tables'], symbols_at_end=res['symbols_at_end'], zero_by_op=res['zero_by_op'])
    if a.emit_script:
        lines = [f"# qhdl track MBU edits for {s12} {res['span']}: {len(matched)} proved sites; new bit "
                 f"b{res['num_bits']} (shared)"]
        lines += [u['edit'].replace('{NEW}', str(res['num_bits'])) for u in matched]
        Path(a.emit_script).write_text('\n'.join(lines) + '\n')
        facts['edit_script'] = str(a.emit_script)
    rr = api.cmd('track', '-c', s12, *(['--region', a.region] if a.region else []), *(['--ops', a.ops] if a.ops else []),
                 '--warm' if a.warm else None, '--idle' if a.idle else None,
                 *(['--show', a.show] if a.show else []))
    nxt = []
    if a.emit_script and matched:
        nxt.append(api.cmd('edit', '-c', s12, '--script', a.emit_script, '--out', '<new.kmb>'))
    if not a.idle:
        nxt.append(rr + ' --idle')
    return E.Envelope(led, claim, domain=f"{res['span']}: every shot, every input and outcome (symbolic; values "
                                         'only, phases not tracked)',
                      evidence=api.rel(ev), rerun=rr, tool_sha=api.tool_sha(), circuit_sha=sha, facts=facts,
                      rows=rows, table=res['tables']['zeros'] if show == 'zeros' else None, next_cmds=nxt)


def qhdl_register(subparsers, api):
    import argparse
    p = subparsers.add_parser(
        'track', help='symbolic value tracker (any size): provably-|0> Toffoli targets (MBU candidates), no-op '
                      'Toffolis, zero sites, CNOT-recomputable idle wires',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all', action='store_true', help='print every row')
    p.add_argument('-c', '--circuit', required=True, help='.kmb path or sha prefix')
    p.add_argument('--region', help='top-level INFO_TAG region (as in `peak` rows)')
    p.add_argument('--ops', help='A:B op span (B exclusive)')
    p.add_argument('--warm', action='store_true', help='track from op 0 even with --region/--ops (exact relations '
                                                       'across the span start; slower)')
    p.add_argument('--idle', action='store_true', help='test wires idle across peak runs for CNOT recompute '
                                                       '(runs `peak` first; 5-8 GB on DQI)')
    p.add_argument('--registers', action='store_true', help='--idle: include register wires')
    p.add_argument('--show', choices=SHOW, help='rows to print (default uncomputes, or idle with --idle)')
    p.add_argument('--emit-script', help='write the proved MBU replacements as a `qhdl edit` script')
    p.set_defaults(fn=lambda a: v_track(api, a))
