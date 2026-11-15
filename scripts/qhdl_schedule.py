#!/usr/bin/env python3
"""qhdl plugin: `schedule` — static schedule queries for width-bound circuits.

    qhdl schedule plateaus   -c CIRCUIT [--ops A:B | --stage NAME] [--level Q]
    qhdl schedule hold       -c CIRCUIT --wire W[,W..] --from I --to J [--q Q] [--extra N]
    qhdl schedule attribute  -c CIRCUIT --range NAME=A:B [--range NAME=A:B ...] [--no-exec]
    qhdl schedule repairs    -c CIRCUIT [--ops A:B | --stage NAME] [--no-exec] [--no-holds] [--term-cap N]

Op ranges are half-open [A, B) in op indices of the opened workspace (`qhdl open`; ops.tsv).

plateaus   maximal runs of consecutive ops whose live width equals LEVEL (default: the circuit's peak) with op ranges.
hold       can wire(s) W be kept live over ops I..J (inclusive) without the live width exceeding Q (default: the
           declared qubit count)? A wire already live at an op costs nothing there; a dead wire costs +1. Answer yes/no
           plus the blocking ops (as runs). --extra N holds N further anonymous words over the whole span.
attribute  per op range: Toffoli-like gates (CCX/CCZ/CSWAP) split into unconditional (no enclosing condition),
           conditional (inside a PUSH_CONDITION_IF block or carrying its own condition bit) and repair-block work (a
           condition block opened within 3 ops after the HMR that produced its bit, so it fires on measurement
           outcome 1). Static counts always; mean executed Toffolis per shot when `qhdl exec` can be run (invsqrt
           circuits: exact over all legal x with a seeded outcome per HMR, tier sampled(n)). With two ranges the
           answer also gives the paired difference (second minus first).
repairs    every HMR-then-recompute repair block in the range: op range, HMR op and wire, Toffoli count by kind, mean
           executed Toffolis, live width at the block, and the algebraic degree of the block's phase as a polynomial
           over GF(2) in the values of the wires live when the block opens (wires dead at entry are zero; HMR
           outcomes inside the block are fresh variables; the block's own condition bit is factored out). Degree is
           reported only when computable (no Z_POW, no unsupported bit op, term count under --term-cap). With holds on,
           each recomputed CCX that repeats an earlier CCX on the same controls is also checked for the hold that would
           avoid the recompute (hold the earlier target wire from its first HMR to the block): feasible or blocked.

Tier: static (op stream, live-interval profile and symbolic phase algebra; no simulation). The per-shot means in
`attribute` and `repairs` use the existing `exec` result (sampled(n)) and are labelled as such. A hold answer is a
width-feasibility test only: it does not check that the held value is still correct for the later use.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOFFOLI_LIKE = {'CCX', 'CCZ', 'CSWAP'}
ZERO_OPS = {'R', 'HMR'}
TERM_CAP_DEFAULT = 20000


def base_kind(kind):
    return kind[:-3] if kind.endswith('_IF') else kind


# ---------------------------------------------------------------- table
class Table:
    """Op stream + live-interval profile. kinds[i] carries the `_IF` suffix when the op has its own condition bit."""

    def __init__(self, kinds, wires, bit, cond, prof, intervals):
        self.kinds, self.wires, self.bit, self.cond = kinds, wires, bit, cond
        self.prof = np.asarray(prof, dtype=np.int64)
        self.intervals = intervals                       # [(wire, start, end)] end inclusive
        self.n = len(kinds)
        self._by_wire = {}
        for w, s, e in intervals:
            self._by_wire.setdefault(w, []).append((s, e))
        for v in self._by_wire.values():
            v.sort()

    @classmethod
    def from_workspace(cls, d):
        d = Path(d)
        kinds, wires, bit, cond = [], [], [], []
        with open(d / 'ops.tsv', newline='') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                kinds.append(r['kind'])
                wires.append([int(x[1:]) for x in r['operands'].split(',')] if r['operands'] else [])
                bit.append(int(r['bit'][1:]) if r['bit'] else -1)
                cond.append(int(r['cond'][1:]) if r['cond'] else -1)
        iv = []
        with open(d / 'intervals.tsv', newline='') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                iv.append((int(r['wire']), int(r['start']), int(r['end'])))
        return cls(kinds, wires, bit, cond, np.load(d / 'profile.npy'), iv)

    @classmethod
    def from_ops(cls, ops, always_live=()):
        """Synthetic constructor. ops: [(kind, wires, bit=-1, cond=-1)]. A wire is live from its first touch after a
        zeroing (R or HMR) to the next zeroing, inclusive; `always_live` wires are live throughout."""
        kinds = [o[0] for o in ops]
        wires = [list(o[1]) for o in ops]
        bit = [o[2] if len(o) > 2 else -1 for o in ops]
        cond = [o[3] if len(o) > 3 else -1 for o in ops]
        n = len(ops)
        start, iv = {}, []
        for w in always_live:
            start[w] = 0
        for i, (k, ws) in enumerate(zip(kinds, wires)):
            for w in ws:
                start.setdefault(w, i)
            if base_kind(k) in ZERO_OPS:
                w = ws[0]
                iv.append((w, start.pop(w), i))
        iv += [(w, s, n - 1) for w, s in start.items()]
        prof = np.zeros(n + 1, np.int64)
        for _, s, e in iv:
            prof[s] += 1
            prof[e + 1] -= 1
        return cls(kinds, wires, bit, cond, np.cumsum(prof[:n]), iv)

    def live(self, w, k):
        return any(s <= k <= e for s, e in self._by_wire.get(w, ()))

    def block_end(self, i):
        """Index just past the POP_CONDITION matching the PUSH at i."""
        d, j = 1, i + 1
        while d and j < self.n:
            if self.kinds[j] == 'PUSH_CONDITION_IF':
                d += 1
            elif self.kinds[j] == 'POP_CONDITION':
                d -= 1
            j += 1
        return j


def parse_range(spec, table, stages=None):
    """'A:B' (half-open), or a stage name from stages.json."""
    if ':' in spec:
        a, b = spec.split(':')
        return (int(a) if a else 0), (int(b) if b else table.n)
    for s in stages or []:
        if s['name'] == spec:
            return s['start'], min(s['end'], table.n)
    raise ValueError(f'unknown range or stage: {spec!r}')


# ---------------------------------------------------------------- plateaus
def runs_of(mask, offset=0):
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    cut = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return [(int(s) + offset, int(e) + offset) for s, e in zip(starts, ends)]


def plateaus(t, a=0, b=None, level=None):
    b = t.n if b is None else b
    level = int(t.prof.max()) if level is None else int(level)
    runs = runs_of(t.prof[a:b] == level, a)
    rows = [dict(start=s, end=e, ops=e - s + 1) for s, e in runs]
    return dict(level=level, peak=int(t.prof.max()), range=[a, b], runs=len(rows),
                ops=sum(r['ops'] for r in rows), rows=rows)


# ---------------------------------------------------------------- hold
def hold(t, wires, i, j, q, extra=0):
    """Width test for keeping `wires` live over ops i..j inclusive."""
    add = np.full(j - i + 1, extra, np.int64)
    live_at_start = {}
    for w in wires:
        # the value held from op i is the one in the interval covering i; it costs nothing until that interval ends or the wire is next zeroed
        # (a later interval on the same wire is a different value, so the held copy needs its own wire again)
        e = next((e for s_, e in t._by_wire.get(w, ()) if s_ <= i <= e), None)
        live_at_start[w] = e is not None
        if e is not None:
            # ... and not past the next op that zeroes the wire (HMR/R): after it the wire holds a different value
            z = next((k for k in range(i, e + 1) if base_kind(t.kinds[k]) in ZERO_OPS and t.wires[k][:1] == [w]), None)
            e = e if z is None else z
        add += np.array([0 if (e is not None and k <= e) else 1 for k in range(i, j + 1)], np.int64)
    over = (t.prof[i:j + 1] + add) > q
    blocks = runs_of(over, i)
    need = int((t.prof[i:j + 1] + add).max())
    return dict(wires=list(wires), span=[i, j], q=q, wire_live_at_start={f'q{w}': v for w, v in live_at_start.items()}, extra_anonymous=extra, feasible=not blocks,
                max_width_needed=need, blocking_ops=int(over.sum()),
                blocking_runs=[dict(start=s, end=e, ops=e - s + 1) for s, e in blocks[:50]],
                blocking_runs_total=len(blocks))


# ---------------------------------------------------------------- conditions
def def_kind(t, bit, idx, window=4000):
    """Nearest earlier definition of a bit: ('HMR', j) or ('STORE', j) or ('?', None)."""
    for j in range(idx - 1, max(idx - window, -1), -1):
        if t.bit[j] == bit:
            if t.kinds[j] == 'HMR':
                return 'HMR', j
            if t.kinds[j] in ('BIT_STORE0', 'BIT_STORE1'):
                return 'STORE', j
    return '?', None


def is_repair_push(t, i):
    """PUSH at i opened on the outcome bit of an HMR within the 3 ops before it; returns the HMR op or None."""
    if t.kinds[i] != 'PUSH_CONDITION_IF':
        return None
    for h in range(max(i - 3, 0), i):
        if t.kinds[h] == 'HMR' and t.bit[h] == t.cond[i]:
            return h
    return None


def classify(t, a, b):
    """Per-op category for Toffoli-like ops in [a,b): unconditional | conditional | repair; nested flag."""
    stack, res = [], {}
    for i in range(a, b):
        k = t.kinds[i]
        if k == 'PUSH_CONDITION_IF':
            stack.append(i)
        elif k == 'POP_CONDITION':
            if stack:
                stack.pop()
        elif base_kind(k) in TOFFOLI_LIKE:
            if stack:
                cat = 'repair' if is_repair_push(t, stack[0]) is not None else 'conditional'
            elif t.cond[i] >= 0:
                cat = 'conditional'
            else:
                cat = 'unconditional'
            res[i] = (cat, len(stack) > 1)
    return res


def step_ops(t, a, b):
    """CCX ops inside a CX-CCX-CX 'step' (shared control c: c->a, c->b, CCX(a,b->t), c->t)."""
    st = set()
    for i in range(max(a, 2), min(b, t.n - 1)):
        if t.kinds[i] != 'CCX':
            continue
        p1, p2, n1 = i - 2, i - 1, i + 1
        if t.kinds[p1] == t.kinds[p2] == t.kinds[n1] == 'CX':
            c = t.wires[p1][0]
            x, y, tg = t.wires[i]
            if (t.wires[p2][0] == c and t.wires[n1][0] == c and {t.wires[p1][1], t.wires[p2][1]} == {x, y}
                    and t.wires[n1][1] == tg):
                st.add(i)
    return st


def attribute_range(t, a, b, w_sampled=None, w_zero=None):
    cls = classify(t, a, b)
    steps = step_ops(t, a, b)
    out = {}
    for cat in ('unconditional', 'conditional', 'repair'):
        ops = [i for i, (c, _) in cls.items() if c == cat]
        d = dict(static=len(ops))
        if w_sampled is not None:
            d['mean'] = round(sum(w_sampled.get(i, 0.0) for i in ops), 4)
            d['zero_branch'] = round(sum(w_zero.get(i, 0.0) for i in ops), 4)
        if cat == 'unconditional':
            d['in_cx_ccx_cx_step'] = sum(1 for i in ops if i in steps)
        if cat != 'unconditional':
            d['nested'] = sum(1 for i in ops if cls[i][1])
        out[cat] = d
    tot = dict(static=len(cls))
    if w_sampled is not None:
        tot['mean'] = round(sum(d['mean'] for d in out.values()), 4)
        tot['zero_branch'] = round(sum(d['zero_branch'] for d in out.values()), 4)
    return dict(range=[a, b], total=tot, **out)


def attribute(t, ranges, w_sampled=None, w_zero=None):
    res = {name: attribute_range(t, a, b, w_sampled, w_zero) for name, (a, b) in ranges.items()}
    out = dict(ranges=res)
    if len(res) == 2:
        (n1, r1), (n2, r2) = list(res.items())
        diff = {}
        for key in ('total', 'unconditional', 'conditional', 'repair'):
            diff[key] = {f: round(r2[key][f] - r1[key][f], 4) for f in r1[key] if f in r2[key]
                         and isinstance(r1[key][f], (int, float))}
        out['paired_difference'] = dict(second_minus_first=f'{n2} - {n1}', **diff)
    return out


# ---------------------------------------------------------------- phase polynomial of a block
class TooBig(Exception):
    pass


def _pxor(a, b):
    return a ^ b


def _pmul(a, b, cap):
    if not a or not b:
        return frozenset()
    r = set()
    for x in a:
        for y in b:
            r ^= {x | y}
    if len(r) > cap:
        raise TooBig
    return frozenset(r)


def _pmul_all(polys, cap):
    r = frozenset([0])
    for p in polys:
        r = _pmul(r, p, cap)
    return r


def phase_polynomial(t, i, j, cap=TERM_CAP_DEFAULT):
    """Phase accumulated by the condition block t[i:j] (PUSH at i, POP at j-1) as a GF(2) polynomial, with the block's
    own condition bit factored out. Monomials are int bitmasks over variables (wire w -> bit w; HMR outcomes and
    bits undefined in the block -> bits >= 1000). Wires dead at op i start as 0. Returns dict."""
    ONE = frozenset([0])
    state, bits = {}, {}
    fresh = [1000]
    unsupported = []

    def var(v):
        return frozenset([1 << v])

    def get_wire(w):
        if w not in state:
            state[w] = var(w) if t.live(w, i) else frozenset()
        return state[w]

    def get_bit(b):
        if b not in bits:
            fresh[0] += 1
            bits[b] = var(fresh[0])
        return bits[b]

    phase = frozenset()
    stack = []
    g = ONE
    try:
        for k in range(i + 1, j - 1):
            kind, ws = t.kinds[k], t.wires[k]
            if kind == 'PUSH_CONDITION_IF':
                stack.append(get_bit(t.cond[k]))
                g = _pmul_all(stack, cap)
                continue
            if kind == 'POP_CONDITION':
                stack.pop()
                g = _pmul_all(stack, cap)
                continue
            b = base_kind(kind)
            gk = _pmul(g, get_bit(t.cond[k]), cap) if t.cond[k] >= 0 else g
            if b == 'X':
                state[ws[0]] = _pxor(get_wire(ws[0]), gk)
            elif b == 'CX':
                state[ws[1]] = _pxor(get_wire(ws[1]), _pmul(gk, get_wire(ws[0]), cap))
            elif b == 'CCX':
                m = _pmul(gk, _pmul(get_wire(ws[0]), get_wire(ws[1]), cap), cap)
                state[ws[2]] = _pxor(get_wire(ws[2]), m)
            elif b in ('SWAP', 'CSWAP'):
                if b == 'SWAP':
                    p, q = ws
                    d = _pmul(gk, _pxor(get_wire(p), get_wire(q)), cap)
                else:
                    c, p, q = ws
                    d = _pmul(gk, _pmul(get_wire(c), _pxor(get_wire(p), get_wire(q)), cap), cap)
                state[p], state[q] = _pxor(get_wire(p), d), _pxor(get_wire(q), d)
            elif b == 'Z':
                phase = _pxor(phase, _pmul(gk, get_wire(ws[0]), cap))
            elif b == 'CZ':
                phase = _pxor(phase, _pmul(gk, _pmul(get_wire(ws[0]), get_wire(ws[1]), cap), cap))
            elif b == 'CCZ':
                m = _pmul(get_wire(ws[0]), _pmul(get_wire(ws[1]), get_wire(ws[2]), cap), cap)
                phase = _pxor(phase, _pmul(gk, m, cap))
            elif b == 'NEG':
                phase = _pxor(phase, gk)
            elif b == 'HMR' and gk == ONE:
                fresh[0] += 1
                phase = _pxor(phase, _pmul(var(fresh[0]), get_wire(ws[0]), cap))
                bits[t.bit[k]] = var(fresh[0])
                state[ws[0]] = frozenset()
            elif b == 'R' and gk == ONE:
                state[ws[0]] = frozenset()
            elif kind == 'BIT_STORE0' and gk == ONE:
                bits[t.bit[k]] = frozenset()
            elif kind == 'BIT_STORE1' and gk == ONE:
                bits[t.bit[k]] = ONE
            elif kind == 'BIT_INVERT' and gk == ONE:
                bits[t.bit[k]] = _pxor(get_bit(t.bit[k]), ONE)
            else:
                unsupported.append(kind)
            if len(phase) > cap:
                raise TooBig
    except TooBig:
        return dict(computable=False, reason=f'term count above cap {cap}', degree=None, terms=None, unsupported=unsupported)
    if unsupported:
        return dict(computable=False, reason='unsupported ops: ' + ','.join(sorted(set(unsupported))), degree=None,
                    terms=len(phase), unsupported=sorted(set(unsupported)))
    nonconst = [m for m in phase if m]
    deg = max((bin(m).count('1') for m in nonconst), default=None)
    return dict(computable=True, degree=deg, terms=len(phase), constant_term=0 in phase,
                phase_zero=not phase, unsupported=[])


# ---------------------------------------------------------------- repairs
def repairs(t, a, b, w_sampled=None, term_cap=TERM_CAP_DEFAULT, holds=True, q=None):
    q = int(t.prof.max()) if q is None else q
    rows, i = [], a
    while i < b:
        h = is_repair_push(t, i)
        if h is None:
            i += 1
            continue
        j = t.block_end(i)
        tof = [k for k in range(i, j) if base_kind(t.kinds[k]) in TOFFOLI_LIKE]
        if tof:
            kinds = {}
            for k in tof:
                kinds[base_kind(t.kinds[k])] = kinds.get(base_kind(t.kinds[k]), 0) + 1
            row = dict(start=i, end=j - 1, hmr=h, hmr_wire=t.wires[h][0], toffolis=len(tof), kinds=kinds,
                       width_at_block=int(t.prof[i]))
            if w_sampled is not None:
                row['mean_executed'] = round(sum(w_sampled.get(k, 0.0) for k in tof), 4)
            row['phase'] = phase_polynomial(t, i, j, term_cap)
            if holds:
                row['recompute_holds'] = _recompute_holds(t, a, i, j, q)
            rows.append(row)
        i = j
    return rows


def _recompute_holds(t, a, i, j, q):
    out = []
    for k in range(i, j):
        if t.kinds[k] != 'CCX':
            continue
        c = t.wires[k][:2]
        prev = next((m for m in range(k - 1, a - 1, -1)
                     if t.kinds[m] == 'CCX' and t.wires[m][:2] == c and not (i <= m < j)), None)
        if prev is None:
            out.append(dict(ccx=k, earlier=None))
            continue
        wire = t.wires[prev][2]
        hm = next((m for m in range(prev + 1, i) if t.kinds[m] == 'HMR' and t.wires[m][0] == wire), None)
        if hm is None:
            out.append(dict(ccx=k, earlier=prev, wire=wire, hmr=None))
            continue
        hd = hold(t, [wire], hm, i - 1, q)
        out.append(dict(ccx=k, earlier=prev, wire=wire, hmr=hm, hold_span=[hm, i - 1], feasible=hd['feasible'],
                        blocking_ops=hd['blocking_ops'], max_width_needed=hd['max_width_needed']))
    return out


# ---------------------------------------------------------------- qhdl verbs
def _weights(api, meta, seed, use_exec):
    """(w_sampled, w_zero, tier note) from the cached `exec` result, or (None, None, None)."""
    if not use_exec or meta.get('task') != 'invsqrt' or 'ops_table' in meta:
        return None, None, None
    res, _, ev = api.exec_small(meta, seed)
    if not res.get('enumerable'):
        return None, None, None
    n = res['rows']
    ws = {int(k): v / n for k, v in res['sampled'].items()}
    wz = {int(k): v / n for k, v in res['zero'].items()}
    return ws, wz, (n, api.rel(ev))


def _load(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    meta = api.ensure_open(a.circuit)
    d = api.ws_dir(meta['sha256'])
    if not (d / 'ops.tsv').exists():
        raise api.QhdlError(f"{meta['sha12']}: no ops table in the workspace",
                            'schedule needs ops.tsv, intervals.tsv and profile.npy (an opened enumerable circuit)')
    t = Table.from_workspace(d)
    stages = json.loads((d / 'stages.json').read_text())['stages'] if (d / 'stages.json').exists() else []
    return E, led, meta, d, t, stages


def _span(a, t, stages):
    if getattr(a, 'ops', None):
        return parse_range(a.ops, t, stages)
    if getattr(a, 'stage', None):
        return parse_range(a.stage, t, stages)
    return 0, t.n


def _finish(api, E, led, meta, d, name, claim, res, rows, rerun_args, facts=None):
    ev = d / 'schedule' / f'{name}.json'
    api.write_json(ev, res)
    return E.Envelope(led, claim, tier='static' if led.best() == 'static' else None, domain='op stream, static',
                      evidence=api.rel(ev), rerun=api.cmd('schedule', name, '-c', meta['sha12'], *rerun_args),
                      tool_sha=api.tool_sha(), circuit_sha=meta['sha256'], facts=facts or {}, rows=rows, parts=[],
                      next_cmds=[])


def v_plateaus(api, a):
    E, led, meta, d, t, stages = _load(api, a)
    lo, hi = _span(a, t, stages)
    res = plateaus(t, lo, hi, a.level)
    led.earn('static', 'maximal runs of ops at a given live width, from the live-interval profile (ops.tsv/profile.npy)')
    claim = (f"{meta['sha12']}: {res['runs']} plateau runs at live width {res['level']} (peak {res['peak']}) in ops "
             f"[{lo},{hi}), {res['ops']} ops in all")
    args = (['--ops', f'{lo}:{hi}'] if (lo, hi) != (0, t.n) else []) + (['--level', a.level] if a.level else [])
    return _finish(api, E, led, meta, d, 'plateaus', claim, res, res['rows'] if a.all else res['rows'][:40], args,
                   facts={k: v for k, v in res.items() if k != 'rows'})


def v_hold(api, a):
    E, led, meta, d, t, stages = _load(api, a)
    wires = [int(w.lstrip('q')) for w in a.wire.split(',')] if a.wire else []
    q = a.q if a.q is not None else meta['declared_qubits']
    res = hold(t, wires, a.start, a.end, q, a.extra)
    led.earn('static', 'width feasibility of holding wires over an op span: live-interval profile plus one per dead '
                       'wire; says nothing about the held value remaining usable')
    what = ', '.join(f'q{w}' for w in wires) + (f' + {a.extra} anonymous' if a.extra else '')
    claim = (f"{meta['sha12']}: holding {what} over ops {a.start}..{a.end} at Q={q} is "
             f"{'FEASIBLE' if res['feasible'] else 'NOT feasible'}; peak needed {res['max_width_needed']}, "
             f"{res['blocking_ops']} blocking ops in {res['blocking_runs_total']} runs")
    args = ['--wire', a.wire or '', '--from', a.start, '--to', a.end, '--q', q] + (['--extra', a.extra] if a.extra else [])
    return _finish(api, E, led, meta, d, 'hold', claim, res, res['blocking_runs'][:40], args,
                   facts={k: v for k, v in res.items() if k != 'blocking_runs'})


def v_attribute(api, a):
    E, led, meta, d, t, stages = _load(api, a)
    ranges = {}
    for spec in a.range:
        name, _, r = spec.partition('=') if '=' in spec else (spec, '', spec)
        ranges[name] = parse_range(r, t, stages)
    ws, wz, note = _weights(api, meta, a.seed, not a.no_exec)
    res = attribute(t, ranges, ws, wz)
    led.earn('static', 'Toffoli-like gate counts by condition context (op stream)')
    tier = 'static'
    if ws is not None:
        tier = f'sampled({note[0]})'
        led.earn(tier, f'mean executed Toffolis per shot from qhdl exec (seed {a.seed}; evidence {note[1]})')
        res['exec_evidence'] = note[1]
    parts = []
    for n, r in res['ranges'].items():
        tt = r['total']
        parts.append(f"{n} {tt['static']} static" + (f", mean {tt['mean']:g}" if 'mean' in tt else '')
                     + f" (unconditional {r['unconditional'].get('mean', r['unconditional']['static'])}, "
                       f"conditional {r['conditional'].get('mean', r['conditional']['static'])}, "
                       f"repair {r['repair'].get('mean', r['repair']['static'])})")
    claim = f"{meta['sha12']}: " + '; '.join(parts)
    args = [x for n, (lo, hi) in ranges.items() for x in ('--range', f'{n}={lo}:{hi}')] + (['--no-exec'] if a.no_exec else [])
    rows = [dict(range=n, **{k: v for k, v in r.items() if k != 'range'}) for n, r in res['ranges'].items()]
    return _finish(api, E, led, meta, d, 'attribute', claim, res, rows, args, facts=res.get('paired_difference', {}))


def v_repairs(api, a):
    E, led, meta, d, t, stages = _load(api, a)
    lo, hi = _span(a, t, stages)
    ws, _, note = _weights(api, meta, a.seed, not a.no_exec)
    rows = repairs(t, lo, hi, ws, a.term_cap, not a.no_holds, meta['declared_qubits'])
    led.earn('static', 'repair blocks from the op stream; phase polynomial by exact GF(2) symbolic execution of the block')
    if ws is not None:
        led.earn(f'sampled({note[0]})', f'mean executed Toffolis per shot from qhdl exec (seed {a.seed}; evidence {note[1]})')
    degs = [r['phase']['degree'] for r in rows if r['phase']['computable']]
    nh = sum(1 for r in rows for h in r.get('recompute_holds', []) if h.get('feasible'))
    nholds = sum(1 for r in rows for h in r.get('recompute_holds', []) if 'feasible' in h)
    res = dict(range=[lo, hi], blocks=len(rows), toffolis=sum(r['toffolis'] for r in rows),
               mean_executed=round(sum(r.get('mean_executed', 0) for r in rows), 4) if ws is not None else None,
               degrees_computed=len(degs), degree_counts={str(k): degs.count(k) for k in sorted(set(degs), key=str)},
               recompute_holds_checked=nholds, recompute_holds_feasible=nh, rows=rows)
    claim = (f"{meta['sha12']}: {res['blocks']} repair blocks in ops [{lo},{hi}) with {res['toffolis']} static Toffolis"
             + (f", mean {res['mean_executed']:g} per shot" if ws is not None else '')
             + f"; phase degrees {res['degree_counts']} ({res['degrees_computed']} computable)"
             + (f"; {nh} of {nholds} avoid-the-recompute holds feasible" if nholds else ''))
    args = (['--ops', f'{lo}:{hi}'] if (lo, hi) != (0, t.n) else []) + (['--no-exec'] if a.no_exec else []) \
        + (['--no-holds'] if a.no_holds else [])
    return _finish(api, E, led, meta, d, 'repairs', claim, res, rows if a.all else rows[:25], args,
                   facts={k: v for k, v in res.items() if k != 'rows'})


def _add_common(p):
    p.add_argument('-c', '--circuit', required=True, help='.kmb path or sha prefix')


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('schedule', help='static schedule queries: width plateaus, hold-span feasibility, '
                                               'Toffoli attribution by condition context, repair-block phase degree',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='schedule_verb', required=True)
    s = sub.add_parser('plateaus', help='maximal runs at the peak live width')
    _add_common(s)
    s.add_argument('--ops'), s.add_argument('--stage'), s.add_argument('--level', type=int)
    s.add_argument('--all', action='store_true')
    s.set_defaults(fn=lambda a: v_plateaus(api, a))
    s = sub.add_parser('hold', help='can wire(s) be held live over an op span within width Q')
    _add_common(s)
    s.add_argument('--wire', default='', help='wire(s), e.g. 41 or q41,q28')
    s.add_argument('--from', dest='start', type=int, required=True)
    s.add_argument('--to', dest='end', type=int, required=True)
    s.add_argument('--q', type=int, help='width bound (default: declared qubits)')
    s.add_argument('--extra', type=int, default=0, help='further anonymous words held over the whole span')
    s.add_argument('--all', action='store_true')
    s.set_defaults(fn=lambda a: v_hold(api, a))
    s = sub.add_parser('attribute', help='Toffolis by unconditional / conditional / repair for op ranges')
    _add_common(s)
    s.add_argument('--range', action='append', required=True, help='NAME=A:B (half-open) or a stage name; repeat')
    s.add_argument('--no-exec', action='store_true', help='static counts only (skip the exec means)')
    s.add_argument('--all', action='store_true')
    s.add_argument('--seed', type=int, default=getattr(api, 'EXEC_SEED', 1))
    s.set_defaults(fn=lambda a: v_attribute(api, a))
    s = sub.add_parser('repairs', help='HMR-then-recompute repair blocks with phase degree')
    _add_common(s)
    s.add_argument('--ops'), s.add_argument('--stage')
    s.add_argument('--no-exec', action='store_true'), s.add_argument('--no-holds', action='store_true')
    s.add_argument('--term-cap', type=int, default=TERM_CAP_DEFAULT)
    s.add_argument('--seed', type=int, default=getattr(api, 'EXEC_SEED', 1))
    s.add_argument('--all', action='store_true')
    s.set_defaults(fn=lambda a: v_repairs(api, a))


def main(argv=None):
    ap = argparse.ArgumentParser(description='standalone: run the schedule queries on an opened qhdl workspace dir')
    ap.add_argument('verb', choices=['plateaus', 'attribute', 'repairs'])
    ap.add_argument('workspace', type=Path)
    ap.add_argument('--ops', default=None, help='A:B')
    ap.add_argument('--range', action='append')
    a = ap.parse_args(argv)
    t = Table.from_workspace(a.workspace)
    lo, hi = parse_range(a.ops, t) if a.ops else (0, t.n)
    if a.verb == 'plateaus':
        out = plateaus(t, lo, hi)
    elif a.verb == 'repairs':
        out = repairs(t, lo, hi)
    else:
        out = attribute(t, {s.partition('=')[0]: parse_range(s.partition('=')[2], t) for s in a.range})
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
