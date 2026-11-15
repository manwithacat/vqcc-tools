#!/usr/bin/env python3
"""qhdl plugin: the QHDL v1 rewrite registry.

    qhdl census  -c CIRCUIT --class all|K [--region R | --ops A:B] [--proof ...] [--sampled N] [--unfit]
    qhdl rewrite -c CIRCUIT --class K [--sites id,id,...] --out X.kmb [--replacement FILE] [--unproved]
                 [--certify auto|none|smoke|fdiff] [--force]

A rewrite class is a Python module `qhdl_rewrite_<name>.py` with a fixed interface (match / preconditions / price /
apply / certify), not a pattern language. `census --class` runs every class's `match` and prints sites; `rewrite`
applies ONE class to the sites whose obligations are discharged (or all of them with --unproved, said so in the
certificate), writes the candidate and its op map through `qhdl edit`, runs the cheapest certificate for the task,
and prints one envelope whose `next` is the full certificate. One class, one candidate, one certificate: applying two
classes is two rewrites.

Classes (v1, extended 2026-11-02):
  phase-fold          equal bounded ANF predicates -> exact modular phase combination, no new wires;
                       raw/general/T-angle rotations reported separately from Toffolis
  and3-dirty-mbu       the 4-Toffoli dirty-borrow AND3 uncompute CCX(a,b,d) CCX(d,c,t) CCX(a,b,d) CCX(d,c,t)
                       with t provably |0> after -> HMR t -> m; CCZ a b c if m   (4 -> 0.5 expected)
  flag-ladder-mbu      a 6-input dirty-borrow uncompute chain (16 Toffolis, or 14 with cached products)
                       with t |0> after and one clean helper wire -> HMR + conditional two-wire ladder with its own
                       measurement-based uncomputation (5 static, 2.25 expected)
  u3-mbu               busyaprime's partial MBU of a 3-input flag (3 Toffolis + HMR + CZ fix-up; correct because t
                       held AND(lits)) -> HMR t -> m; CCZ(lits) if m (0.5 expected)
  cc-chain             S12 / S6: a 12- or 6-input AND computed into a clean target by a dirty-borrow chain
                       -> the conditionally clean ancilla constructions (Khattar & Gidney 2024): 19/18 for n = 12 with
                       two clean ancillae, 8/7.5 for n = 6 with one; other n priced (2n - 3) but not built
  single-toffoli-mbu   a lone Toffoli whose target is provably |0> after it -> HMR t -> m; CZ a b if m (1 -> 0)
  and-merge            invsqrt: Toffoli-led windows whose live outputs' AND floor is below their Toffoli count
                       (`windows`); apply needs the agent's replacement window (--replacement), checked by fdiff

Matching is semantic. A block is a site when its exact net action over ALL values of every wire it touches (dirty
borrows included; algebraic normal form) is `t ^= AND(literals)` with every other wire restored and every in-block
measurement's phase cancelled by its fix-ups (wires measured inside the block may be assumed |0> at entry; that
assumption becomes an obligation). Zero-after and clean-wire obligations are discharged by `qhdl census` proofs
(every shot), the structural compute/uncompute rule, or the DQI lane simulation (sampled(N), evidence not proof); a
site whose shape does not fit a template is reported with the reason, never forced.

Addresses: sites carry an absolute anchor `op@<sha12>:<i>` and, on tagged circuits, a tag-relative address
`tag:<path>[k]+<offset>` (op offset inside the k-th instance of that INFO_TAG path) that survives edits outside the
region and transfers between circuits of one family. --sites accepts site ids (`and3-dirty-mbu#3`), `#3`, op indices,
op@ or tag: addresses.

Certificates state their scope {domain, extent, branches}. DQI: one kickmix-engine smoke draw (sweep-0) through
scripts/qhdl_surgery.py gate; a 48-draw sweep and the official verifier are `next`. invsqrt:
`fdiff --whole` (all legal x, zero + one random branch). Template constructions are checked exhaustively on 2^n
inputs before any splice. No certificate here is an official result.
Library: census_sites(api, meta, ...), rewrite(...); classes via load_classes().
"""
from __future__ import annotations

import bisect
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import qhdl_kmb as K  # noqa: E402

CCX, X_, CX, HMR, TAG, R_ = 0x31, 0x11, 0x21, 0x50, 0x06, 0x12
CZ_IF, CCZ_IF, CCZ, CZ = 0xA0, 0xB0, 0x30, 0x20
BLOCK_OPS = (CCX, X_, CX, HMR, TAG, CZ_IF, CCZ_IF)   # ops a semantic block may contain
MAX_BLOCK = 400
MAX_TERMS = 512
SHORT_RUN = 64        # runs this short try every Toffoli as a block start; longer runs only boundary starts
NEXT_TOUCH_SCAN = 1 << 20
KNOWN_TASKS = ('dqi', 'invsqrt', 'chemistry')
CLASSES = ('and3-dirty-mbu', 'flag-ladder-mbu', 'u3-mbu', 'cc-chain', 'single-toffoli-mbu', 'and-merge', 'phase-fold')
import vqcc_config as CFG  # noqa: E402
KMPY = CFG.KMPY
ENGINE_PY = CFG.ENGINE_PKG
SMOKE_SALT = 'sweep-0'
W_SHIFT = 36


class RewriteError(ValueError):
    def __init__(self, msg, hint=None):
        super().__init__(msg)
        self.hint = hint


# ---------------------------------------------------------------- GF(2) polynomials (sets of monomial bitmasks)
def pmul(p, q):
    out = set()
    for a in p:
        for b in q:
            out ^= {a | b}
    return out


def as_and(delta, var, inv):
    """delta == product of literals over var ids (degree >= 2)? -> [(wire, negated)] in var-id order, else None."""
    if not delta:
        return None
    top = max(delta, key=lambda m: bin(m).count('1'))
    deg = bin(top).count('1')
    if deg < 2 or sum(1 for m in delta if bin(m).count('1') == deg) != 1:
        return None
    lits, prod = [], {0}
    for i in range(top.bit_length()):
        if top >> i & 1:
            neg = (top ^ (1 << i)) in delta
            lits.append((inv[i], neg))
            prod = pmul(prod, {1 << i} ^ ({0} if neg else set()))
    return lits if prod == delta else None


def subst_zero(p, mask):
    """Drop monomials containing any variable in mask (those variables are 0)."""
    return {m for m in p if not (m & mask)} if mask else p


# ---------------------------------------------------------------- op records for sparse op indices (any size)
class Ops:
    """Random access to (code, qubits, bit) of ops by index without an op table: stream positions by searchsorted."""

    def __init__(self, k):
        self.k = k
        self.t = np.asarray(k.op_types)
        cls = self.t & 0x30
        self.i3 = np.flatnonzero(cls == 0x30)
        self.i2 = np.flatnonzero(cls == 0x20)
        self.i1 = np.flatnonzero(cls == 0x10)
        self.ib = np.flatnonzero((self.t & 0x40) > 0)
        self.ic = np.flatnonzero((self.t & 0x80) > 0)
        self.s = {n: k.stream(n) for n in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')}
        self.depth = K.condition_depth(self.t)

    def records(self, idxs):
        """{i: (code, qubits tuple, bit or None)} ; bit = b0 for ops with a bit operand, else the condition bit."""
        idxs = np.asarray(sorted(set(int(i) for i in idxs)), np.int64)
        out = {}
        if not len(idxs):
            return out
        c = self.t[idxs]
        m3, m2, m1 = (c & 0x30) == 0x30, (c & 0x30) == 0x20, (c & 0x30) == 0x10
        mb, mc = (c & 0x40) > 0, ((c & 0x80) > 0)
        q = {}
        for m, idx, names in ((m3, self.i3, ('qqq0', 'qqq1', 'qqq2')), (m2, self.i2, ('qq0', 'qq1')),
                              (m1, self.i1, ('q0',))):
            if m.any():
                p = np.searchsorted(idx, idxs[m])
                cols = [np.asarray(self.s[n][p]).tolist() for n in names]
                for j, i in enumerate(idxs[m].tolist()):
                    q[i] = tuple(int(col[j]) for col in cols)
        bits = {}
        if mb.any():
            p = np.searchsorted(self.ib, idxs[mb])
            for i, b in zip(idxs[mb].tolist(), np.asarray(self.s['b0'][p]).tolist()):
                bits[i] = int(b)
        if mc.any():
            p = np.searchsorted(self.ic, idxs[mc])
            for i, b in zip(idxs[mc].tolist(), np.asarray(self.s['bc'][p]).tolist()):
                bits.setdefault(i, int(b))
        for i, code in zip(idxs.tolist(), c.tolist()):
            out[i] = (int(code), q.get(i, ()), bits.get(i))
        return out

    def text(self, i, rec=None):
        code, q, b = rec if rec is not None else self.records([i])[i]
        base = code & 0x7F if code != 0x81 else code
        name = K.NAMES.get(base, hex(code))
        s = name + ''.join(f' q{x}' for x in q)
        if code & 0x40:
            s += f' b{b}'
        elif code & 0x80:
            s += (f' if b{b}' if code != 0x81 else f' b{b}')
        return s

    def touched(self, a, b):
        """Set of wires referenced by ops [a, b) (any operand)."""
        out = set()
        for code, q, _ in self.records(range(a, b)).values():
            out.update(q)
        return out

    def next_touch(self, w, a, limit=NEXT_TOUCH_SCAN):
        """First op >= a that references wire w (scanning up to `limit` ops), or None."""
        b = min(len(self.t), a + limit)
        best = None
        for idx, names in ((self.i3, ('qqq0', 'qqq1', 'qqq2')), (self.i2, ('qq0', 'qq1')), (self.i1, ('q0',))):
            lo, hi = np.searchsorted(idx, a), np.searchsorted(idx, b)
            if hi <= lo:
                continue
            hit = np.zeros(hi - lo, bool)
            for n in names:
                hit |= np.asarray(self.s[n][lo:hi]) == w
            j = np.flatnonzero(hit)
            if len(j):
                cand = int(idx[lo + j[0]])
                best = cand if best is None else min(best, cand)
        return best


# ---------------------------------------------------------------- the semantic block finder
def _block_from(ops, s, e_max, depth0=True):
    """Smallest block ops[s:e] (e <= e_max) whose exact net action is one target ^= AND(literals) (kind 'chain') or
    'target measured to |0>, correct iff target held AND(literals)' (kind 'mbu-partial'), with every other wire
    restored and every measurement's phase cancelled. Wires measured inside the block may be assumed |0> at entry
    (`assumed_zero`). Returns a dict or None."""
    var, state, written, order = {}, {}, set(), []
    obligations = []      # [bit, wire, poly at HMR, fix-up poly]
    open_ob = {}          # bit -> obligation (latest HMR writing that bit)
    hmr_wires = []
    ctrl_seen = []
    toff = 0

    def val(w):
        if w not in var:
            var[w] = len(var)
            state[w] = {1 << var[w]}
            order.append(w)
        return state[w]

    for e in range(s, e_max):
        code, q, b = ops[e]
        if code == TAG:
            continue
        if code == X_:
            state[q[0]] = val(q[0]) ^ {0}
            written.add(q[0])
        elif code == CX:
            pa, pb = val(q[0]), val(q[1])
            state[q[1]] = pb ^ pa
            written.add(q[1])
        elif code == CCX:
            pa, pb, pc = val(q[0]), val(q[1]), val(q[2])
            add = pmul(pa, pb)
            if len(add) > MAX_TERMS:
                return None
            state[q[2]] = pc ^ add
            written.add(q[2])
            toff += 1
            for x in (q[0], q[1]):
                if x not in ctrl_seen:
                    ctrl_seen.append(x)
        elif code == HMR:
            p = val(q[0])
            ob = [b, q[0], set(p), set()]
            obligations.append(ob)
            open_ob[b] = ob
            state[q[0]] = set()
            written.add(q[0])
            if q[0] not in hmr_wires:
                hmr_wires.append(q[0])
        elif code == CZ_IF:
            if b not in open_ob:
                return None
            open_ob[b][3] ^= pmul(val(q[0]), val(q[1]))
        elif code == CCZ_IF:
            if b not in open_ob:
                return None
            open_ob[b][3] ^= pmul(pmul(val(q[0]), val(q[1])), val(q[2]))
            toff += 1
        else:
            return None
        if any(len(p) > MAX_TERMS for p in state.values()):
            return None
        if toff < 2 and not (toff >= 1 and obligations):
            continue
        if code not in (CCX, CZ_IF, CCZ_IF, HMR, CX, X_):
            continue
        inv = {v: w for w, v in var.items()}
        # try without assumptions, then assuming every wire measured so far was |0> at entry
        for assume in ((), tuple(hmr_wires)):
            mask = 0
            for w in assume:
                mask |= 1 << var[w]
            moved = [w for w in order if subst_zero(state[w], mask) != subst_zero({1 << var[w]}, mask)]
            if len(moved) != 1:
                continue
            t = moved[0]
            st = subst_zero(state[t], mask)
            bad = [ob for ob in obligations if subst_zero(ob[2], mask) != subst_zero(ob[3], mask)]
            kind = None
            if not bad:
                delta = st ^ subst_zero({1 << var[t]}, mask)
                if not delta or any(m >> var[t] & 1 for m in delta):
                    continue
                lits = as_and(delta, var, inv)
                if lits is None or any(w == t for w, _ in lits):
                    continue
                kind = 'chain'
            elif len(bad) == 1 and bad[0][1] == t and st == set() and t not in assume:
                res = subst_zero(bad[0][2], mask) ^ subst_zero(bad[0][3], mask)
                if (1 << var[t]) not in res:
                    continue
                lits = as_and(res ^ {1 << var[t]}, var, inv)
                if lits is None or any(w == t for w, _ in lits) or any(m >> var[t] & 1 for m in res ^ {1 << var[t]}):
                    continue
                kind = 'mbu-partial'
            if kind is None:
                continue
            litw = {w for w, _ in lits}
            borrowed = sorted(w for w in written if w != t and w not in litw and w not in assume)
            if kind == 'chain' and len(lits) < 3 and not borrowed:
                continue   # a plain Toffoli (or two), even next to an unrelated measurement, is not a chain site
            inputs = [w for w in ctrl_seen if w in litw] + [w for w, _ in lits if w not in ctrl_seen]
            neg = dict(lits)
            return dict(start=s, end=e + 1, target=t, kind=kind, controls=[(w, neg[w]) for w in inputs],
                        borrowed=borrowed, toffolis=toff, hmr_wires=list(hmr_wires), assumed_zero=list(assume),
                        hmr_bits=sorted({ob[0] for ob in obligations}))
    return None


def find_blocks(ops_index, A=0, B=None, progress=None):
    """Semantic blocks in ops [A, B) at condition depth 0. Runs of BLOCK_OPS with >= 2 Toffolis (or a Toffoli and a
    measurement) are scanned: every Toffoli starts a candidate in runs <= SHORT_RUN ops; in longer runs only Toffolis
    at a boundary (first op of the run, or preceded by an X / INFO_TAG) do. Returns blocks sorted by start."""
    t = ops_index.t
    B = len(t) if B is None else B
    seg = t[A:B]
    ok = np.isin(seg, np.array(BLOCK_OPS, np.uint8)) & (ops_index.depth[A:B] == 0)
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    rs, re_ = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    ccx = np.r_[0, np.cumsum(seg == CCX)]
    hm = np.r_[0, np.cumsum(seg == HMR)]
    nccx, nh = ccx[re_] - ccx[rs], hm[re_] - hm[rs]
    keep = (nccx >= 2) | ((nccx >= 1) & (nh >= 1))
    rs, re_ = rs[keep] + A, re_[keep] + A
    out = []
    if not len(rs):
        return out
    t0 = time.time()
    for n_, (a, b) in enumerate(zip(rs.tolist(), re_.tolist())):
        rec = ops_index.records(range(a, b))
        short = (b - a) <= SHORT_RUN
        s = a
        prev = None
        while s < b:
            code = rec[s][0]
            if code == CCX and (short or prev in (None, X_, TAG)):
                r = _block_from(rec, s, min(b, s + MAX_BLOCK))
                if r is not None:
                    out.append(r)
                    s = r['end']
                    prev = CCX
                    continue
            prev = code if code != TAG else prev
            s += 1
        if progress and n_ % 200 == 0:
            progress(n_, len(rs), time.time() - t0)
    out.sort(key=lambda r: r['start'])
    for r in out:
        r['n'] = len(r['controls'])
    return out


def pair_blocks(blocks):
    """Attach to each block the previous block with the same target and literal controls (its compute)."""
    last = {}
    for c in blocks:
        key = (c['target'], tuple(sorted(c['controls'])))
        c['partner'] = last.get(key)
        last[key] = c
    return blocks


# ---------------------------------------------------------------- clean-wire model (static, sound)
class CleanModel:
    """Which wires are |0> at an op: last value-changing op is an unconditional depth-0 reset (R / HMR), or the wire
    was never written, or that op ends a site proved (or sampled) to leave the wire |0>. Registers are never clean."""

    def __init__(self, k, ops_index):
        import qhdl_census as C
        self.k = k
        self.ops = ops_index
        self.nq = int(k.num_qubits)
        self.regs = {q for _, cs in k.registers for isq, q, _ in cs if isq}
        self.keys, self.rst = C._write_keys(k, [])
        self.known = {}   # wire -> {last write op: status}  (zero-after sites)

    def add_zero_after(self, wire, last_write_op, status):
        self.known.setdefault(wire, {})[last_write_op] = status

    def last_write(self, w, i):
        """(op, is_reset) of the last value-changing op on w before op i, or None."""
        j = int(np.searchsorted(self.keys, (w << W_SHIFT) | i)) - 1
        if j < 0 or (int(self.keys[j]) >> W_SHIFT) != w:
            return None
        return int(self.keys[j] & ((1 << W_SHIFT) - 1)), bool(self.rst[j])

    def writes_between(self, w, a, b):
        return int(np.searchsorted(self.keys, (w << W_SHIFT) | b) - np.searchsorted(self.keys, (w << W_SHIFT) | a))

    def status(self, w, i):
        """'proved' | 'sampled(N)' | '' : is wire w |0> right before op i."""
        if w in self.regs or w >= self.nq:
            return ''
        lw = self.last_write(w, i)
        if lw is None:
            return 'proved'
        op, is_reset = lw
        if is_reset:
            return 'proved'
        st = self.known.get(w, {}).get(op, '')
        if st:
            return st
        # a reset (R / HMR) inside one PUSH_CONDITION block, with the wire |0> before the block: in the shots that run
        # the block it ends reset, in the others it keeps the |0> it had (the measurement-based ladders end this way)
        t, depth = self.ops.t, self.ops.depth
        code = int(t[op])
        if (code & 0x7F) in (R_, HMR) and code < 0x80 and int(depth[op]) == 1:
            start = op
            while start > 0 and depth[start] != 0 and op - start < 100_000:
                start -= 1
            end = op
            while end < len(t) and depth[end] != 0 and end - op < 100_000:
                end += 1
            if int(t[start]) == K.PUSH_CONDITION and end <= i and depth[start] == 0:
                return self.status(w, start)
        return ''

    def clean_scratch(self, i, exclude=(), span_end=None):
        """[(wire, status, last write op)] clean right before op i, untouched in [i, span_end), not in exclude;
        most recently erased first."""
        touched = self.ops.touched(i, span_end) if span_end and span_end > i else set()
        out = []
        for w in range(self.nq):
            if w in self.regs or w in exclude or w in touched:
                continue
            st = self.status(w, i)
            if st:
                lw = self.last_write(w, i)
                out.append((w, st, lw[0] if lw else -1))
        out.sort(key=lambda x: (-x[2], x[0]))
        return out


# ---------------------------------------------------------------- tag-relative addresses
class TagIndex:
    def __init__(self, k):
        import qhdl_peak as P
        self.inst = []     # (start, end, path, ordinal, depth)
        stack, counts = [], {}
        for o, kind, name in P._tag_events(k):
            if kind == '@enter':
                stack.append([name, o])
            elif kind == '@exit' and stack:
                nm, s = stack.pop()
                path = '/'.join(x[0] for x in stack) + ('/' if stack else '') + nm
                kk = counts.get(path, 0)
                counts[path] = kk + 1
                self.inst.append((s, o + 1, path, kk, len(stack)))
        self.inst.sort()
        self.starts = [x[0] for x in self.inst]
        self.by_path = {}
        for s, e, p, kk, d in self.inst:
            self.by_path.setdefault(p, {})[kk] = (s, e)

    @property
    def tagged(self):
        return bool(self.inst)

    def rel(self, i):
        """'tag:<path>[k]+off' of the innermost instance containing op i, or None."""
        j = bisect.bisect_right(self.starts, i) - 1
        while j >= 0:
            s, e, p, kk, _ = self.inst[j]
            if e > i:
                return f'tag:{p}[{kk}]+{i - s}'
            j -= 1
        return None

    def resolve(self, addr):
        m = re.fullmatch(r'tag:(.+?)\[(\d+)\]\+(\d+)', addr.strip())
        if not m:
            raise RewriteError(f'bad tag-relative address {addr!r} (use tag:<path>[k]+<offset>)')
        p, kk, off = m.group(1), int(m.group(2)), int(m.group(3))
        if p not in self.by_path or kk not in self.by_path[p]:
            raise RewriteError(f'no instance {kk} of tag path {p!r} in this circuit')
        s, e = self.by_path[p][kk]
        if s + off >= e:
            raise RewriteError(f'{addr}: offset {off} is beyond the instance (ops {s}..{e - 1})')
        return s + off


# ---------------------------------------------------------------- sites and obligations
def obligation(name, status, basis='', tier=None):
    """status: proved | sampled(N) | open | failed | n/a. tier defaults from the status."""
    if tier is None:
        tier = 'static' if status in ('proved', 'n/a') else (status if status.startswith('sampled') else None)
    return dict(name=name, status=status, tier=tier, basis=basis, claim_id=None)


def discharged(site):
    return all(o['status'] in ('proved', 'n/a') or o['status'].startswith('sampled') for o in site['obligations'])


def fully_proved(site):
    return all(o['status'] in ('proved', 'n/a') for o in site['obligations'])


def make_site(ctx, cls, core, block, n, now, price, wires, obligations, fits=True, reason='', **extra):
    a, b = core
    s = dict(id=None, cls=cls, circuit_sha=ctx.sha, anchor=f'op@{ctx.sha[:12]}:{a}', ops=[a, b],
             tag_rel=ctx.tags.rel(a) if ctx.tags.tagged else None,
             block=[block['start'], block['end']] if block else [a, b], n=n, now=now, price=price, wires=wires,
             obligations=obligations, fits=bool(fits), reason=reason, conditional=False)
    s.update(extra)
    return s


class Context:
    """Everything a class's match needs for one circuit span: blocks, census rows, the clean model, simulation."""

    def __init__(self, api, meta, A, B, proof='auto', sampled=1024, progress=None, census_rows=None, need=()):
        self.api, self.meta = api, meta
        self.sha, self.sha12 = meta['sha256'], meta['sha12']
        self.path = api.circuit_path(meta)
        self.k = K.read(str(self.path))
        self.N = len(self.k.op_types)
        self.A, self.B = A, B
        self.task = meta.get('task') or (meta.get('manifest') or {}).get('task')
        self.ops = Ops(self.k)
        self.tags = TagIndex(self.k)
        self.proof, self.sampled = proof, sampled
        self._blocks = None
        self._census = census_rows
        self._clean = None
        self._sim = {}
        self.progress = progress
        self.need = need
        self.notes = []
        names = {nm for nm, _ in self.k.registers}
        self.dqi = 'S0' in names and 'E0' in names

    @property
    def blocks(self):
        if self._blocks is None:
            self._blocks = pair_blocks(find_blocks(self.ops, self.A, self.B, progress=self.progress))
        return self._blocks

    @property
    def census(self):
        """qhdl census rows for the span (chains with zero proofs, lone Toffolis, prices, free wires); cached in the
        workspace exactly as the census verb caches them."""
        if self._census is None:
            self._census = census_result(self.api, self.meta, self.A, self.B, self.proof, self.sampled)['rows']
        return self._census

    @property
    def clean(self):
        if self._clean is None:
            self._clean = CleanModel(self.k, self.ops)
            for r in self.census:
                if r.get('zero'):
                    a, b = (int(x) for x in r['ops'].split('..'))
                    w = int(r['target'][1:])
                    lw = self._clean.last_write(w, b + 1)
                    if lw is not None:
                        self._clean.add_zero_after(w, lw[0], r['zero'])
        return self._clean

    def census_row(self, block):
        """The census chain/single row whose span contains this block's core and has the same target."""
        tgt = f"q{block['target']}"
        for r in self.census:
            a, b = (int(x) for x in r['ops'].split('..'))
            if a <= block['start'] and block['end'] - 1 <= b and r['target'] == tgt:
                return r
        return None

    def zero_after(self, block):
        """('proved' | 'sampled(N)' | '', basis) for the block's target right after the block."""
        r = self.census_row(block)
        if r is not None and r.get('zero'):
            how = 'qhdl census zero proof (every shot)' if r['zero'] == 'proved' else \
                f"qhdl census lane simulation: target |0> on all {r['zero'][8:-1]} sampled valid syndromes"
            return r['zero'], how
        p = block.get('partner')
        t = block['target']
        if p is not None:
            ws = [t] + [w for w, _ in block['controls']]
            if not any(self.clean.writes_between(w, p['end'], block['start']) for w in ws) \
                    and self.clean.status(t, p['start']) == 'proved' and not p.get('assumed_zero') \
                    and not block.get('assumed_zero'):
                return 'proved', ('structural rule: compute/uncompute pair with the target and controls untouched '
                                  'between and the target |0> before the compute')
        if self.dqi and self.sampled:
            st = self.sim_state(block['end'])
            if st is not None:
                if st[t] == 0:
                    return f'sampled({self.lanes})', f'lane simulation: target |0> on all {self.lanes} sampled valid syndromes'
                return '', f'lane simulation: target is 1 on {bin(st[t]).count("1")} of {self.lanes} lanes'
        return '', 'no proof found (no census proof, no structural pair, no lane simulation)'

    def clean_status(self, w, i):
        """('proved' | 'sampled(N)' | '', basis): is wire w |0> right before op i. Static proof first (reset, never
        written, proved zero-after site); else the DQI lane simulation (evidence, not a proof). Use only for wires
        whose |0> state the ORIGINAL circuit's correctness already relies on (a compute chain's target); never to pick
        ancillae, where a rare-event flag absent from the sampled lanes would be mistaken for a free wire."""
        st = self.clean.status(w, i)
        if st:
            return st, ('clean-wire model: never written, last value-changing op an unconditional reset, or a '
                        f'{st} zero-after site')
        if self.dqi and self.sampled:
            lanes = self.sim_state(i)
            if lanes is not None:
                if lanes[w] == 0:
                    return f'sampled({self.lanes})', f'lane simulation: |0> on all {self.lanes} sampled valid syndromes'
                return '', f'lane simulation: 1 on {bin(lanes[w]).count("1")} of {self.lanes} lanes'
        return '', 'not known to be |0> (no reset, no zero-after proof, no lane simulation)'

    def holds_and(self, block):
        """Sampled check that the target holds AND(literals) right before the block (partial-MBU precondition)."""
        if not (self.dqi and self.sampled):
            return '', 'no lane simulation available'
        st = self.sim_state(block['start'])
        if st is None:
            return '', 'no lane simulation available'
        ALL = (1 << self.lanes) - 1
        prod = ALL
        for w, neg in block['controls']:
            prod &= (st[w] ^ ALL) if neg else st[w]
        if st[block['target']] == prod:
            return f'sampled({self.lanes})', f'lane simulation: target == AND(literals) on all {self.lanes} sampled valid syndromes'
        return '', f"lane simulation: target != AND(literals) on {bin(st[block['target']] ^ prod).count('1')} lanes"

    # -- lane simulation (DQI), one pass for every probe the classes asked for
    def want_probe(self, op):
        self._sim.setdefault('probes', set()).add(int(op))

    def sim_state(self, op):
        probes = self._sim.setdefault('probes', set())
        probes.add(int(op))
        if 'res' not in self._sim or int(op) not in self._sim['res']['state_at']:
            if 'res' not in self._sim:   # first run: probe every block boundary at once (one pass, not one per site)
                for b in self.blocks:
                    probes.update((b['start'], b['end']))
                for r in self.census:
                    if r['pattern'] == 'single':
                        probes.update((r['op'], r['op'] + 1))
            self._run_sim(probes)
        return self._sim['res']['state_at'].get(int(op))

    @property
    def lanes(self):
        return self._sim['res']['lanes']

    def _run_sim(self, probes):
        import qhdl_simulate as SIM
        d = self.api.ws_dir(self.sha) / 'census'
        d.mkdir(parents=True, exist_ok=True)
        inp = d / f'census_inputs_{self.sampled}_1.npz'
        if not inp.exists():
            SIM.gen_inputs(self.path, inp, self.sampled, 1)
        t0 = time.time()
        res = SIM.simulate(self.path, inp, probes=sorted(probes))
        self._sim['res'] = res
        self._sim['probes'] = set(probes)
        self.notes.append(f'lane simulation {res["lanes"]} lanes, {len(probes)} probes, {round(time.time() - t0, 1)} s')


def census_result(api, meta, A, B, proof='auto', sampled=1024):
    """Run (or reuse) `qhdl census` on the span through the verb's own cache file and key."""
    import qhdl_census as C
    sha = meta['sha256']
    src = api.circuit_path(meta)
    d = api.ws_dir(sha) / 'census'
    tag = f'{A}_{B}_{proof}_s{sampled}'
    ev = d / f'census_{tag}.json'
    key = dict(tool_sha=api.tool_sha(), verb='census', A=A, B=B, warm=False, proof=proof, sampled=sampled)
    res = api.cached(ev, key)
    if res is None:
        res = C.census(src, A, B, warm=False, proof=proof, sampled=sampled, sim_dir=d)
        res['cache_key'] = key
        res['span'] = f'ops {A}..{B - 1}'
        d.mkdir(parents=True, exist_ok=True)
        cols = ['op', 'ops', 'path', 'pattern', 'target', 'n', 'now', 'zero', 'conditional', 'mbu', 'clean', 'cc',
                'best', 'saving', 'free_clean', 'saving_with_width', 'width', 'pairs_with', 'controls', 'borrowed']
        tsv = d / f'census_{tag}.tsv'
        api.write_tsv(tsv, cols, [[r_.get(c, '') for c in cols] for r_ in res['rows']])
        res['table'] = api.rel(tsv)
        api.write_json(ev, res)
    return res


# ---------------------------------------------------------------- registry
_CLASSES = None


def load_classes(names=None):
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = {}
        for nm in CLASSES:
            mod = __import__('qhdl_rewrite_' + nm.replace('-', '_'))
            _CLASSES[nm] = mod.CLASS
    if names is None:
        return dict(_CLASSES)
    out = {}
    for nm in names:
        if nm not in _CLASSES:
            raise RewriteError(f'unknown rewrite class {nm!r}', 'classes: ' + ', '.join(CLASSES))
        out[nm] = _CLASSES[nm]
    return out


class RewriteClass:
    """Interface every class module implements (its module exposes CLASS = <instance>)."""
    name = ''
    tasks = ()
    needs = ()
    bits = ()                 # names of the new classical bits the construction uses (shared by all its sites)
    certify = {}              # task -> certificate description
    summary = ''

    def match(self, ctx):          # -> [site]; every site carries price, obligations, fits, reason
        raise NotImplementedError

    def apply(self, site, ctx, bits):   # -> [edit lines] ; bits: name -> id
        raise NotImplementedError

    def template_check(self):      # -> str describing the exhaustive check of the construction (or '')
        return ''


def census_sites(api, meta, A, B, classes, proof='auto', sampled=1024, progress=None):
    ctx = Context(api, meta, A, B, proof=proof, sampled=sampled, progress=progress)
    sites = []
    for nm, cls in classes.items():
        if ctx.task in KNOWN_TASKS and cls.tasks and ctx.task not in cls.tasks:
            ctx.notes.append(f'{nm}: not for {ctx.task} circuits')
            continue
        found = cls.match(ctx)
        found.sort(key=lambda s: s['ops'][0])
        for j, s in enumerate(found):
            s['id'] = f'{nm}#{j}'
            s['cls'] = nm
            sites.append(s)
    sites.sort(key=lambda s: (s['ops'][0], s['cls']))
    return ctx, sites


# ---------------------------------------------------------------- edit emission and certificates
def write_candidate(api, meta, sites, cls, out, replacement=None, map_path=None, force=False, bits_from=None):
    """Compose the class's edit script for the chosen sites, apply it with qhdl edit, return (summary, script).
    New classical bits are numbered from bits_from (default: the circuit's num_bits), so several classes applied one
    after another can share one bit layout whatever their order."""
    import qhdl_edit as ED
    src = api.circuit_path(meta)
    k = K.read(str(src))
    nb = int(k.num_bits) if bits_from is None else int(bits_from)
    if nb < 0:
        raise RewriteError('--bits-from must be >= 0')
    bits = {name: nb + i for i, name in enumerate(cls.bits)}
    lines = [f"# qhdl rewrite --class {cls.name} on {meta['sha12']}: {len(sites)} site(s); new bits "
             + (', '.join(f'{n}=b{b}' for n, b in bits.items()) if bits else 'none')]
    ctx = SimpleCtx(api, meta, k, replacement)
    for s in sites:
        lines.append(f"# {s['id']} {s['anchor']}" + (f" {s['tag_rel']}" if s.get('tag_rel') else ''))
        lines.extend(cls.apply(s, ctx, bits))
    script = '\n'.join(lines) + '\n'
    out = Path(out)
    if out.exists() and not force:
        raise RewriteError(f'{out} exists; refusing to overwrite', 'pass --force or choose another --out')
    try:
        r = ED.edit_file(src, script, out, map_path)
    except ED.EditError as exc:
        raise RewriteError(f'edit failed: {exc}')
    sp = out.with_name(out.name + '.edit.txt')
    sp.write_text(script)
    r['script'] = str(sp)
    r['bits'] = bits
    return r, script


class SimpleCtx:
    def __init__(self, api, meta, k, replacement):
        self.api, self.meta, self.k = api, meta, k
        self.ops = Ops(k)
        self.replacement = replacement or {}


def parse_replacement(text):
    """Replacement file: sections headed `site <id>` or `ops A:B`, each followed by one op per line (kmx-style text
    or @i copies; '#' comments). -> {key: [op lines]} with keys 'and-merge#1' and 'A:B'."""
    out, cur = {}, None
    for raw in text.splitlines():
        line = re.sub(r'(^|\s)#.*$', '', raw).strip()   # '#' after whitespace starts a comment (site ids contain '#')
        if not line:
            continue
        m = re.fullmatch(r'(site|ops)\s+(\S+)', line)
        if m:
            cur = m.group(2)
            out[cur] = []
            continue
        if cur is None:
            raise RewriteError('replacement file: op lines before any `site <id>` / `ops A:B` heading')
        out[cur].append(line)
    return out


def gate_smoke(cand, out_json, threads=1):
    """One kickmix-engine draw (sweep-0) through scripts/qhdl_surgery.py gate in the kickmix venv."""
    import os
    env = dict(os.environ, PYTHONPATH=ENGINE_PY) if ENGINE_PY else dict(os.environ)
    args = [KMPY, str(HERE / 'qhdl_surgery.py'), 'gate', '--circuit', str(cand), '--salts', SMOKE_SALT,
            '--threads', str(threads), '--out', str(out_json)]
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=7200)
    if p.returncode != 0 or not Path(out_json).exists():
        tail = (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ['?'])[-1]
        raise RewriteError(f'gate smoke draw failed (exit {p.returncode}): {tail[:300]}',
                           'run `qhdl doctor`; the engine .so must export run_sequence')
    d = json.loads(Path(out_json).read_text())
    row = d['rows'][0]['candidate']
    return dict(status=row.get('status'), toffoli_mean=row.get('toffoli_count'), num_qubits=row.get('num_qubits'),
                value_errors=row.get('value_errors'), phase_errors=row.get('phase_errors'),
                max_toffoli=row.get('max_toffoli_count'), draw=d['rows'][0].get('draw'), seconds=round(time.time() - t0, 1),
                rerun=' '.join((['PYTHONPATH=' + ENGINE_PY] if ENGINE_PY else []) + args))


def fdiff_whole(api, parent_meta, cand_path, seed=7):
    """Whole-circuit exact comparison on all legal x (invsqrt), through qhdl_window.fdiff_whole."""
    import qhdl_window as W
    m2 = api.ensure_open(str(cand_path))
    L1, L2 = api.Loaded(parent_meta), api.Loaded(m2)
    t0 = time.time()
    r = W.fdiff_whole(L1, L2, seed)
    evidence = Path(cand_path).with_name(Path(cand_path).name + '.fdiff.json')
    api.write_json(evidence, dict(parent=parent_meta['sha256'], candidate=m2['sha256'],
                                  seed=seed, tool_sha=_tool_sha(api), **r))
    ok = all(d['phase_differs'] == 0 and not any(d['x_not_restored']) and not any(d['scratch_dirty'])
             for d in r['branches'].values())
    z = r['branches']['zero']['mean_toffoli']
    return dict(equivalent=ok, mean_toffoli=z, branches=r['branches'], contract=r.get('contract'),
                seconds=round(time.time() - t0, 1), sha=m2['sha256'], evidence=str(evidence))


# ---------------------------------------------------------------- CLI
def _span(api, a, src):
    import qhdl_track as T
    return T._span(api, a, src)


def _select(sites, spec, ctx):
    if not spec:
        return sites
    want = [x.strip() for x in spec.split(',') if x.strip()]
    chosen = []
    for w in want:
        hit = None
        if w.startswith('tag:'):
            op = ctx.tags.resolve(w)
            hit = [s for s in sites if s['ops'][0] == op or s['block'][0] == op]
        elif w.startswith('op@'):
            op = int(w.split(':')[1])
            hit = [s for s in sites if s['ops'][0] == op or s['block'][0] == op]
        elif w.isdigit():
            op = int(w)
            hit = [s for s in sites if s['ops'][0] <= op < s['ops'][1] or s['block'][0] <= op < s['block'][1]]
        elif w.startswith('#'):
            hit = [s for s in sites if s['id'].endswith(w)]
        else:
            hit = [s for s in sites if s['id'] == w]
        if not hit:
            raise RewriteError(f'--sites {w}: no site of this class there',
                               'ids and addresses are in the census --class output')
        chosen.extend(h for h in hit if h not in chosen)
    return chosen


def _site_row(s):
    ob = s['obligations']
    st = ('ok' if fully_proved(s) else 'sampled' if discharged(s) else 'open') if s['fits'] else 'unfit'
    pr = s['price']
    return dict(id=s['id'], ops=f"{s['ops'][0]}..{s['ops'][1] - 1}", tag=s.get('tag_rel') or '-', n=s['n'],
                now=s['now'], after=pr.get('expected'), saving=pr.get('saving'),
                rotations_delta=pr.get('rotations_delta'), general_rotations_delta=pr.get('general_rotations_delta'),
                t_gates_delta=pr.get('t_gates_delta'), status=st,
                proofs='; '.join(f"{o['name']}={o['status']}" for o in ob) if ob else '-',
                note=s['reason'] if s['reason'] else '')


def v_census_class(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    meta = api.ensure_open(a.circuit)
    src = api.circuit_path(meta)
    if src.suffix != '.kmb':
        raise api.QhdlError('census --class reads .kmb circuits')
    A, B, span = _span(api, a, src)
    names = list(CLASSES) if a.cls == 'all' else [x.strip() for x in a.cls.split(',')]
    classes = load_classes(names)
    d = api.ws_dir(meta['sha256']) / 'rewrite'
    tag = f"{A}_{B}_{'-'.join(names) if a.cls != 'all' else 'all'}_{a.proof}_s{a.sampled}"
    ev = d / f'sites_{tag}.json'
    key = dict(tool_sha=_tool_sha(api), verb='census-class', A=A, B=B, classes=names, proof=a.proof,
               sampled=a.sampled)
    res = api.cached(ev, key)
    if res is None:
        t0 = time.time()
        ctx, sites = census_sites(api, meta, A, B, classes, proof=a.proof, sampled=a.sampled)
        res = dict(cache_key=key, span=span, A=A, B=B, sites=sites, notes=ctx.notes, tagged=ctx.tags.tagged,
                   seconds=round(time.time() - t0, 1), blocks=len(ctx._blocks) if ctx._blocks else None)
        api.write_json(ev, res)
    sites = res['sites']
    led.earn('static', 'semantic block match (exact net action over all wire values, algebraic normal form), '
                       'zero / clean-wire proofs from qhdl census and the structural rule (every shot), prices a '
                       'static model')
    smp = sorted({o['status'] for s in sites for o in s['obligations'] if o['status'].startswith('sampled')})
    parts = [dict(part='matches, proofs, prices', tier='static')]
    for t_ in smp:
        led.earn(t_, f'obligations discharged by lane simulation on {t_[8:-1]} sampled valid syndromes (evidence, '
                     'not a proof)')
        parts.append(dict(part=f'obligations at {t_}', tier=t_))
    by = {}
    for s in sites:
        g = by.setdefault(s['cls'], dict(cls=s['cls'], sites=0, fit=0, proved=0, sampled=0, open=0, unfit=0,
                                         saving=0.0))
        g['sites'] += 1
        if not s['fits']:
            g['unfit'] += 1
            continue
        g['fit'] += 1
        if fully_proved(s):
            g['proved'] += 1
        elif discharged(s):
            g['sampled'] += 1
        else:
            g['open'] += 1
        g['saving'] += float(s['price'].get('saving') or 0)
    for g in by.values():
        g['saving'] = round(g['saving'], 2)
    fit = [s for s in sites if s['fits']]
    tot = sum(float(s['price'].get('saving') or 0) for s in fit if discharged(s))
    claim = (f"{meta['sha12']} {span}: {len(sites)} sites in {len(by)} class(es); {len(fit)} fit a template "
             f"({sum(1 for s in fit if fully_proved(s))} with every obligation proved, "
             f"{sum(1 for s in fit if discharged(s) and not fully_proved(s))} on sampled evidence only, "
             f"{sum(1 for s in fit if not discharged(s))} open), {len(sites) - len(fit)} reported unfit. Priced saving "
             f"of the discharged sites {tot:,.2f} executed Toffolis per shot (static model); build with `rewrite`, "
             'then certify')
    rotation_delta = sum(s['price'].get('rotations_delta', 0) for s in fit if discharged(s))
    general_delta = sum(s['price'].get('general_rotations_delta', 0) for s in fit if discharged(s))
    t_delta = sum(s['price'].get('t_gates_delta', 0) for s in fit if discharged(s))
    if any('rotations_delta' in s['price'] for s in fit):
        claim += (f'; Z_POW gates {rotation_delta:+d}, synthesized rotations {general_delta:+d}, '
                  f'T-angle gates {t_delta:+d}. Rotation charge is task-dependent')
    rows = [_site_row(s) for s in (sites if a.unfit else fit)]
    rr = api.cmd('census', '-c', meta['sha12'], '--class', a.cls, *(['--region', a.region] if a.region else []),
                 *(['--ops', a.ops] if a.ops else []), *(['--proof', a.proof] if a.proof != 'auto' else []),
                 *(['--sampled', a.sampled] if a.sampled != 1024 else []), '--unfit' if a.unfit else None)
    nxt = []
    for g in sorted(by.values(), key=lambda g: -g['saving']):
        if g['fit'] and g['cls'] != 'and-merge':
            nxt.append(api.cmd('rewrite', '-c', meta['sha12'], '--class', g['cls'], '--out', f"<cand-{g['cls']}.kmb>"))
            break
    facts = dict(by_class=sorted(by.values(), key=lambda g: g['cls']), evidence_sites=api.rel(ev),
                 addresses='tag-relative' if res.get('tagged') else 'absolute only (untagged circuit)',
                 notes=res.get('notes'), seconds=res.get('seconds'))
    return E.Envelope(led, claim, tier='static', domain=f'{span}: exact block functions; proofs every shot; prices '
                      'a static model', evidence=api.rel(ev), rerun=rr, tool_sha=_tool_sha(api),
                      circuit_sha=meta['sha256'], facts=facts, rows=rows, parts=parts, next_cmds=nxt)


def _tool_sha(api):
    import qhdl_envelope as E
    files = [HERE / f for f in getattr(api, 'TOOL_FILES', []) if (HERE / f).exists()]
    files += sorted(HERE.glob('qhdl_rewrite*.py'))
    return E.files_sha(files)


def v_rewrite(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    meta = api.ensure_open(a.circuit)
    src = api.circuit_path(meta)
    if src.suffix != '.kmb':
        raise api.QhdlError('rewrite reads .kmb circuits')
    out = Path(a.out)
    if out.suffix != '.kmb':
        raise api.QhdlError('--out must end in .kmb')
    if ',' in a.cls or a.cls == 'all':
        raise api.QhdlError('rewrite applies ONE class per candidate', 'run rewrite once per class; certify each')
    cls = load_classes([a.cls])[a.cls]
    A, B, span = _span(api, a, src)
    task = meta.get('task') or (meta.get('manifest') or {}).get('task')
    if task in KNOWN_TASKS and cls.tasks and task not in cls.tasks:
        raise api.QhdlError(f'{cls.name} is for {", ".join(cls.tasks)} circuits; {meta["sha12"]} is {task}')
    t0 = time.time()
    ctx, sites = census_sites(api, meta, A, B, {cls.name: cls}, proof=a.proof, sampled=a.sampled)
    chosen = _select(sites, a.sites, ctx)
    replacement = None
    if a.replacement:
        try:
            replacement = parse_replacement(Path(a.replacement).read_text())
        except OSError as exc:
            raise api.QhdlError(f'cannot read --replacement: {exc}')
        for s in chosen:
            key = s['id'] if s['id'] in replacement else f"{s['ops'][0]}:{s['ops'][1]}"
            if key in replacement:
                s['replacement'] = replacement[key]
                for o in s['obligations']:
                    if o['name'] == 'replacement':
                        o.update(status='proved', tier='static', basis='supplied by --replacement; checked by fdiff')
    unfit = [s for s in chosen if not s['fits']]
    if unfit and a.sites:
        raise api.QhdlError(f"site(s) {', '.join(s['id'] for s in unfit)} do not fit the template: "
                            + '; '.join(s['reason'] for s in unfit), 'a site is never forced')
    chosen = [s for s in chosen if s['fits']]
    undisch = [s for s in chosen if not discharged(s)]
    if undisch and not a.unproved:
        chosen = [s for s in chosen if discharged(s)]
    if not chosen:
        raise api.QhdlError(f'no {cls.name} site to apply' + (f' ({len(undisch)} with open obligations; --unproved applies '
                                                              'them and says so)' if undisch else ''),
                            api.cmd('census', '-c', meta['sha12'], '--class', cls.name, '--unfit'))
    r, script = write_candidate(api, meta, chosen, cls, out, replacement=replacement, map_path=a.map, force=a.force,
                                bits_from=a.bits_from)
    api.register_candidate(r['new_sha'], out, parent=meta['sha256'],
                           how=f"qhdl rewrite --class {cls.name} of {meta['sha12']} ({len(chosen)} sites)")
    sites_json = out.with_name(out.name + '.sites.json')
    api.write_json(sites_json, dict(parent=meta['sha256'], candidate=r['new_sha'], cls=cls.name, sites=chosen,
                                    bits=r['bits'], map=r['map'], script=r['script'], unproved=bool(a.unproved)))
    led.earn('static', 'sites matched semantically and spliced by `qhdl edit` (unedited ops copied byte for byte); '
                       'prices a static model' + (f"; template: {cls.template_check()}" if cls.template_check() else ''))
    parts = [dict(part='match, splice, price', tier='static')]
    smp = sorted({o['status'] for s in chosen for o in s['obligations'] if o['status'].startswith('sampled')})
    for t_ in smp:
        led.earn(t_, f'obligations discharged by lane simulation on {t_[8:-1]} sampled valid syndromes')
        parts.append(dict(part=f'obligations at {t_}', tier=t_))
    saving = sum(float(s['price'].get('saving') or 0) for s in chosen)
    now = sum(float(s['now']) for s in chosen)
    cert = dict(candidate=r['new_sha'], parent=meta['sha256'], map=r['map'], scope=None, tier=None,
                predicted=dict(toffolis_per_shot_delta=-round(saving, 2), static_toffoli_delta=None,
                               rotations_delta=sum(s['price'].get('rotations_delta', 0) for s in chosen),
                               general_rotations_delta=sum(s['price'].get('general_rotations_delta', 0) for s in chosen),
                               t_gates_delta=sum(s['price'].get('t_gates_delta', 0) for s in chosen)), observed=None,
                unproved_sites=[s['id'] for s in chosen if not discharged(s)])
    certify = a.certify
    if certify == 'auto':
        certify = 'fdiff' if task == 'invsqrt' else ('smoke' if task == 'dqi' or ctx.dqi else 'none')
    cert_rows = []
    nxt = []
    tier = 'static'
    if certify == 'smoke':
        g = gate_smoke(out, out.with_name(out.name + f'.smoke-{SMOKE_SALT}.gate.json'), threads=a.threads)
        led.earn('engine-parity', f"kickmix engine replay of ONE draw ({g['draw']}, 102,400 shots): status, value/"
                                  'phase errors and this draw\'s mean Toffolis; not the 48-draw expected cost, not '
                                  'the official verifier')
        parts.append(dict(part='smoke draw', tier='engine-parity'))
        cert.update(scope=dict(domain=f'draws:1@{SMOKE_SALT}', extent='whole', branches='engine-sampled'),
                    tier='engine-parity', observed=dict(status=g['status'], toffoli_mean=g['toffoli_mean'],
                                                        value_errors=g['value_errors'], phase_errors=g['phase_errors']),
                    evidence=str(out.with_name(out.name + f'.smoke-{SMOKE_SALT}.gate.json')), rerun=g['rerun'])
        cert_rows.append(dict(check='smoke draw', draw=g['draw'], status=g['status'], toffoli_mean=g['toffoli_mean'],
                              value_errors=g['value_errors'], phase_errors=g['phase_errors'], qubits=g['num_qubits'],
                              seconds=g['seconds'], tier='engine-parity'))
        tier = 'engine-parity'
        nxt.append('# a full 48-draw DQI sweep with the kickmix engine (expected cost)')
        nxt.append(api.cmd('verify', '-c', str(out), '--task', 'dqi'))
    elif certify == 'fdiff':
        f = fdiff_whole(api, meta, out)
        led.earn('exact-all-x', 'bitset simulation of parent and candidate on all legal x, zero + one random '
                                'outcome branch (seed 7): phase per x, x restored, scratch clean')
        parts.append(dict(part='fdiff --whole', tier='exact-all-x'))
        cert.update(scope=dict(domain='all-x', extent='whole', branches='zero+random(7)'), tier='exact-all-x',
                    observed=dict(equivalent=f['equivalent'], mean_toffoli=f['mean_toffoli']),
                    evidence=f['evidence'])
        for br, d_ in f['branches'].items():
            cert_rows.append(dict(check=f'fdiff --whole {br}', equivalent=(d_['phase_differs'] == 0 and not any(d_['x_not_restored'])
                                                                           and not any(d_['scratch_dirty'])),
                                  phase_differs_x=d_['phase_differs'], x_not_restored=d_['x_not_restored'],
                                  scratch_dirty=d_['scratch_dirty'], mean_toffoli=d_['mean_toffoli'], tier='exact-all-x'))
        tier = 'exact-all-x'
        nxt.append(api.cmd('verify', '-c', str(out)))
    else:
        cert.update(scope=dict(domain='none', extent='none', branches='none'), tier='static')
        nxt.append(api.cmd('rewrite', '-c', meta['sha12'], '--class', cls.name, '--out', str(out), '--force',
                           '--certify', 'auto'))
    cert_path = out.with_name(out.name + '.certificate.json')
    api.write_json(cert_path, cert)
    ok = cert.get('observed') or {}
    verdict = ''
    if certify == 'smoke':
        verdict = f"; smoke draw {ok.get('status')}, mean Toffolis {ok.get('toffoli_mean')}"
    elif certify == 'fdiff':
        verdict = f"; fdiff --whole {'EQUIVALENT' if ok.get('equivalent') else 'DIFFERS'}, executed Toffolis per x " \
                  f"{ok.get('mean_toffoli')}"
    claim = (f"{cls.name}: {len(chosen)} site(s) of {len([s for s in sites if s['fits']])} fitting applied to "
             f"{meta['sha12']} -> {r['new_sha'][:12]} ({r['new_ops'] - r['old_ops']:+d} ops, bits +{r['added_bits']}); "
             f"predicted -{saving:,.2f} executed Toffolis per shot (static model; {now:g} now)"
             + (f"; {len(cert['unproved_sites'])} site(s) applied UNPROVED" if cert['unproved_sites'] else '')
             + verdict + '. Not an official result')
    if any('rotations_delta' in s['price'] for s in chosen):
        claim += (f"; Z_POW gates {cert['predicted']['rotations_delta']:+d}, synthesized rotations "
                  f"{cert['predicted']['general_rotations_delta']:+d}, T-angle gates "
                  f"{cert['predicted']['t_gates_delta']:+d} (not a task-cost claim)")
    rows = [_site_row(s) for s in chosen] + cert_rows
    facts = dict(out=str(out), new_sha256=r['new_sha'], map=r['map'], script=r['script'], certificate=str(cert_path),
                 scope=cert['scope'], predicted=cert['predicted'], bits=r['bits'], seconds=round(time.time() - t0, 1))
    rr = api.cmd('rewrite', '-c', meta['sha12'], '--class', cls.name, *(['--sites', a.sites] if a.sites else []),
                 '--out', str(out), *(['--replacement', a.replacement] if a.replacement else []),
                 '--unproved' if a.unproved else None, *(['--certify', a.certify] if a.certify != 'auto' else []),
                 *(['--bits-from', a.bits_from] if a.bits_from is not None else []),
                 *(['--region', a.region] if a.region else []), *(['--ops', a.ops] if a.ops else []), '--force')
    env = E.Envelope(led, claim, tier=tier, domain=f"{span}: {cert['scope']}", evidence=str(cert_path), rerun=rr,
                     tool_sha=_tool_sha(api), circuit_sha=meta['sha256'], facts=facts, rows=rows, parts=parts,
                     next_cmds=nxt)
    if hasattr(env, 'stamp'):
        env.stamp(scope=cert['scope'], provenance='computed')
    return env


def qhdl_register(subparsers, api):
    import argparse
    # census --class: extend the census verb if qhdl_census registered it, else provide a census parser of our own
    p = subparsers.choices.get('census')
    if p is None:
        p = subparsers.add_parser('census', help='rewrite-class census (qhdl_census not loaded)',
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument('--all', action='store_true')
        p.add_argument('-c', '--circuit', required=True)
        p.add_argument('--region')
        p.add_argument('--ops')
        p.add_argument('--proof', default='auto')
        p.add_argument('--sampled', type=int, default=1024)
        old = None
    else:
        old = p.get_default('fn')
    p.add_argument('--class', dest='cls', help='rewrite registry: all or a class name (' + ', '.join(CLASSES) + ')')
    p.add_argument('--unfit', action='store_true', help='--class: also list sites that do not fit a template')

    def run_census(a):
        if getattr(a, 'cls', None):
            return v_census_class(api, a)
        if old is None:
            raise api.QhdlError('pass --class all|K (qhdl_census is not loaded)')
        return old(a)
    p.set_defaults(fn=run_census)
    q = subparsers.add_parser('rewrite', help='apply one rewrite class to matched sites -> candidate .kmb + map + '
                                              'certificate (smoke draw / fdiff --whole)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--all', action='store_true', help='print every row')
    q.add_argument('-c', '--circuit', required=True, help='.kmb path or sha prefix')
    q.add_argument('--class', dest='cls', required=True, help='one class: ' + ', '.join(CLASSES))
    q.add_argument('--sites', help='comma list: site ids, #k, op indices, op@sha12:i or tag:path[k]+off (default: all '
                                   'fitting sites with discharged obligations)')
    q.add_argument('--out', required=True, help='candidate .kmb path (map, edit script, sites and certificate JSON beside it)')
    q.add_argument('--map', help="op-index map path (default OUT + '.map.json')")
    q.add_argument('--replacement', help='and-merge: file of replacement windows (`site <id>` / `ops A:B` sections)')
    q.add_argument('--unproved', action='store_true', help='apply sites with open obligations too (recorded in the certificate)')
    q.add_argument('--certify', choices=['auto', 'none', 'smoke', 'fdiff'], default='auto',
                   help='auto: invsqrt fdiff --whole, DQI one engine smoke draw (sweep-0); none: splice only')
    q.add_argument('--threads', type=int, default=1, help='engine threads for the smoke draw (default 1)')
    q.add_argument('--region', help='restrict matching to a top-level INFO_TAG region')
    q.add_argument('--ops', help='restrict matching to ops A:B')
    q.add_argument('--proof', default='auto', help='census zero-proof mode (auto|track|local|structural)')
    q.add_argument('--sampled', type=int, default=1024, help='DQI lanes for sampled obligations (0 = off)')
    q.add_argument('--bits-from', type=int, help='first id of the new classical bits (default: the circuit\'s num_bits; '
                                                 'lets classes applied in any order share one bit layout)')
    q.add_argument('--force', action='store_true', help='overwrite --out')
    q.set_defaults(fn=lambda a: v_rewrite(api, a))
