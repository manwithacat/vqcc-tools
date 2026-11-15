#!/usr/bin/env python3
"""qhdl avenues, DQI (guide: docs/qhdl-avenues.md). Imported by qhdl_avenues.py.

    qhdl avenues --task dqi -c <sha prefix> [--top N] [--regions tree.json] [--out DIR] [--ops A:B]

DQI is a 58M-op stream with no op table, so everything here is a streaming, tag-independent pass over the .kmb (tags are
used only to NAME the region of a site). Tier: static model / projection; no executed means are measured.

  G3 repetition   Toffoli-like ops (CCZ/CCX/CSWAP and their _IF forms) are clustered when consecutive ones are at most
                  CLUSTER_GAP ops apart. A cluster's window runs from its first Toffoli to the last Z/CZ/conditioned op
                  or HMR of a Toffoli-touched wire in the tail (at most TAIL ops, never into the next cluster). Windows
                  are hashed up to wire AND classical-bit relabelling (first use; op codes kept, so conditioned ops are
                  part of the key) together with a 'fresh at entry' bit per first-touched wire (previous touch within
                  LOOKBACK ops is R or HMR, i.e. the wire is |0>; otherwise a free variable). Multiplicity = number of
                  windows of a class (disjoint by construction), sites carried with their condition depth.
  G2 floor        One representative per class (the CLASS_CAP classes with the largest static charged total whose window has
                  at most MAX_TOFFOLI Toffolis): exact arbitrary-entry symbolic ANF of the live value outputs and of
                  the phase polynomial f (Z, CZ, CCZ terms; HMR zero branch: wire := 0 and window-bit-conditioned ops
                  skipped). floor = max of
                    (a) qhdl_avenues.full_floor on the outputs that are provably live (next touch within LOOKAHEAD ops is
                        not R/HMR; requiring fewer outputs keeps it a lower bound): degree, nonlinear-output rank, Dickson,
                        cubic derivative, polar bounds;
                    (b) a phase degree floor: with k AND-type gates every wire polynomial has degree <= 2^k, so a
                        phase gate on two or three wires has degree <= 3 * 2^(k-1); a phase of degree d needs the
                        smallest k with 3 * 2^(k-1) >= d (written for this module; a sum of Z/CZ/CCZ terms over wire
                        polynomials is the only phase source it assumes). It is WEAK for high-degree phases: gap is an
                        upper bound on the possible saving, never a promise.
                  Windows with PUSH/POP, Z_POW or conditions on bits not written by an in-window HMR are 'unsupported'
                  (charged cost only, listed as ceilings).
  pricing         DQI cost = R * Q^2 * T_mean, so a saving of S static Toffolis changes GM by (1/3) ln(T/(T - S)) with
                  T = manifest zero-failure cost / (1.1913 * Q^2). S = gap x multiplicity (every static site once: an
                  upper bound). Sites inside a condition block run only when their mask bit is set; the scenario column
                  weights those at 1/64 (Atakan Aldemir's own ~27.6/shot for 1,764 sites; a scenario, not a measurement).
  negatives       rows of an optional negative-results ledger CSV (docs/negative-results/ledger.csv; none shipped,
                  then no rows) of task dqi/all matched by keyword, below a divider when they
                  close (name this component); keyword-only matches are 'related'.

Standard library plus NumPy. Written by an AI model (Claude).
"""
from __future__ import annotations

import array
import bisect
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_kmb as K  # noqa: E402
import qhdl_avenues as A  # noqa: E402

ROOT = HERE.parent
CLUSTER_GAP = 64          # ops between consecutive Toffoli-like ops that still belong to one component
TAIL = 160                # ops after the last Toffoli searched for the phase/measure tail
LOOKBACK = 2048           # ops searched backwards for the previous touch of an entry wire
LOOKAHEAD = 2048          # ops searched forwards for the next touch of an output wire
CLASS_CAP = 400           # classes floored (largest static charged total first)
MAX_TOFFOLI = 24          # floors only for windows of at most this many Toffolis
MAX_OPS_FLOOR = 3000
TERM_CAP = 6000
R_ZERO = 1.1913           # R(eps+) at zero failures (verify_submission_dqi.py)
COND_WEIGHT = 1.0 / 64    # scenario weight of a conditioned site (see module docstring)
BLK = 512
TOFF = (0x30, 0x31, 0x32, 0xB0, 0xB1, 0xB2)
R_, HMR_, ZPOW_ = 0x12, 0x50, 0x13
FRESH_KINDS = (R_, HMR_)


# ------------------------------------------------------------------ fast random-access decoding
class Fast:
    """Random-access op decoder with block prefix counts (no 58M-entry cumsums)."""

    def __init__(self, k):
        self.k = k
        self.t = np.asarray(k.op_types)
        self.n = len(self.t)
        self.s = {nm: k.stream(nm) for nm in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')}
        nb = self.n // BLK + 1
        pre = np.zeros((6, nb + 1), dtype=np.int64)    # c3, c2, c1, cb, cc, depth at the start of each block
        CH = 1 << 20
        run = np.zeros(6, dtype=np.int64)
        blocks_per = CH // BLK
        for i0 in range(0, nb, blocks_per):
            a, b = i0 * BLK, min(self.n, (i0 + blocks_per) * BLK)
            seg = self.t[a:b]
            m = len(seg)
            pad = (-m) % BLK
            if pad:
                seg = np.concatenate([seg, np.zeros(pad, dtype=seg.dtype)])
            sg = seg.reshape(-1, BLK)
            q = sg & 0x30
            cols = [(q == 0x30).sum(1), (q == 0x20).sum(1), (q == 0x10).sum(1), ((sg & 0x40) > 0).sum(1),
                    ((sg & 0x80) > 0).sum(1), (sg == 0x81).sum(1) - (sg == 0x01).sum(1)]
            for j, c in enumerate(cols):
                cs = np.cumsum(c)
                pre[j, i0 + 1:i0 + 1 + len(c)] = run[j] + cs
                run[j] += cs[-1] if len(cs) else 0
        self.pre = pre

    def counts_at(self, a):
        i = a // BLK
        base = self.pre[:, i].copy()
        seg = self.t[i * BLK:a]
        if len(seg):
            q = seg & 0x30
            base += np.array([(q == 0x30).sum(), (q == 0x20).sum(), (q == 0x10).sum(), ((seg & 0x40) > 0).sum(),
                              ((seg & 0x80) > 0).sum(), (seg == 0x81).sum() - (seg == 0x01).sum()], dtype=np.int64)
        return base

    def depth_before(self, a):
        return int(self.counts_at(a)[5])

    def decode(self, a, b):
        """ops [a, b) as a list of (t, wires tuple, b_bit or -1, c_bit or -1)."""
        a, b = max(0, int(a)), min(self.n, int(b))
        if b <= a:
            return []
        c3, c2, c1, cb, cc, _ = self.counts_at(a)
        t = self.t[a:b]
        q = t & 0x30
        i3, i2, i1 = int(c3), int(c2), int(c1)
        ib, ic = int(cb), int(cc)
        s = self.s
        n3, n2, n1 = int((q == 0x30).sum()), int((q == 0x20).sum()), int((q == 0x10).sum())
        nb_, nc_ = int(((t & 0x40) > 0).sum()), int(((t & 0x80) > 0).sum())
        q0, q1, q2 = s['qqq0'][i3:i3 + n3].tolist(), s['qqq1'][i3:i3 + n3].tolist(), s['qqq2'][i3:i3 + n3].tolist()
        p0, p1 = s['qq0'][i2:i2 + n2].tolist(), s['qq1'][i2:i2 + n2].tolist()
        r0 = s['q0'][i1:i1 + n1].tolist()
        bb = s['b0'][ib:ib + nb_].tolist()
        bc = s['bc'][ic:ic + nc_].tolist()
        out = []
        j3 = j2 = j1 = jb = jc = 0
        for tv in t.tolist():
            qc = tv & 0x30
            if qc == 0x30:
                w = (q0[j3], q1[j3], q2[j3])
                j3 += 1
            elif qc == 0x20:
                w = (p0[j2], p1[j2])
                j2 += 1
            elif qc == 0x10:
                w = (r0[j1],)
                j1 += 1
            else:
                w = ()
            bbit = cbit = -1
            if tv & 0x40:
                bbit = bb[jb]
                jb += 1
            if tv & 0x80:
                cbit = bc[jc]
                jc += 1
            out.append((tv, w, bbit, cbit))
        return out


# ------------------------------------------------------------------ G3
def toffoli_indices(t):
    return np.flatnonzero(np.isin(t, TOFF))


def clusters_of(tl, gap=CLUSTER_GAP):
    """-> list of (first_toffoli_op, last_toffoli_op, n_toffoli, next_cluster_first_op)."""
    if len(tl) == 0:
        return []
    brk = np.flatnonzero(np.diff(tl) > gap)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [len(tl) - 1]])
    out = []
    for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        nxt = int(tl[starts[i + 1]]) if i + 1 < len(starts) else 1 << 62
        out.append((int(tl[a]), int(tl[b]), b - a + 1, nxt))
    return out


def window_end(ops, off_last_t, touched):
    """Length of the window inside `ops` (decoded from the cluster start): through the last op of the tail that is an
    HMR of a Toffoli-touched wire, a Z/CZ, or a conditioned op."""
    end = off_last_t + 1
    for i in range(off_last_t + 1, len(ops)):
        tv, w, _, _ = ops[i]
        base = tv & 0x7F
        if tv == 0x81 or tv == 0x01 or tv == 0x06:
            continue
        if (base == 0x50 and w and w[0] in touched) or base in (0x10, 0x20) or (tv & 0x80):
            end = i + 1
    return end


def canon(ops):
    """-> (hash12, wire order, bit order). Wires and classical bits renamed by first use; op codes kept."""
    wren, bren, seq = {}, {}, array.array('q')
    for tv, w, b, c in ops:
        seq.append(tv)
        for x in w:
            r = wren.get(x)
            if r is None:
                r = wren[x] = len(wren)
            seq.append(r)
        if b >= 0:
            r = bren.get(('b', b))
            if r is None:
                r = bren[('b', b)] = len(bren)
            seq.append(1000000 + r)
        if c >= 0:
            r = bren.get(('b', c))
            if r is None:
                r = bren[('b', c)] = len(bren)
            seq.append(2000000 + r)
        seq.append(-1)
    return hashlib.blake2b(seq.tobytes(), digest_size=8).hexdigest(), list(wren)


def fresh_signature(fast, start, wires):
    """Per wire (in first-use order): 'f' fresh (previous touch is R/HMR), 'v' variable (touched earlier, any other op),
    'u' unknown (no touch within LOOKBACK). Doubling lookback windows."""
    need = set(wires)
    kind = {}
    hi = start
    span = 256
    while need and hi > 0 and (start - hi) < LOOKBACK:
        lo = max(0, hi - span, start - LOOKBACK)
        ops = fast.decode(lo, hi)
        for i in range(len(ops) - 1, -1, -1):
            tv, w, _, _ = ops[i]
            for x in w:
                if x in need:
                    kind[x] = tv & 0x7F
                    need.discard(x)
            if not need:
                break
        hi = lo
        span *= 2
        if lo == 0:
            break
    sig = ''.join('f' if kind.get(x) in FRESH_KINDS else ('v' if x in kind else 'u') for x in wires)
    return sig


class RegionIndex:
    """Deepest INFO_TAG region name of an op index (names the region; never used to find components)."""

    def __init__(self, tree):
        self.by_depth = {}
        for r in (tree or {}).get('tag_regions', []):
            self.by_depth.setdefault(r['depth'], []).append((r['start'], r['end'], r['path']))
        for d in self.by_depth:
            self.by_depth[d].sort()
        self.starts = {d: [x[0] for x in v] for d, v in self.by_depth.items()}

    @property
    def tagged(self):
        return bool(self.by_depth)

    def path(self, i):
        for d in sorted(self.by_depth, reverse=True):
            j = bisect.bisect_right(self.starts[d], i) - 1
            if j >= 0 and self.by_depth[d][j][1] > i:
                return self.by_depth[d][j][2]
        return 'untagged' if not self.by_depth else 'outside tags'


def build_classes(fast, regions, progress=None):
    """Streaming G3 pass. -> (classes dict, stats)."""
    t = fast.t
    tl = toffoli_indices(t)
    cl = clusters_of(tl)
    classes = {}
    t0 = time.time()
    for ci, (s, lastT, nT, nxt) in enumerate(cl):
        hi = min(lastT + TAIL + 1, nxt, fast.n)
        ops = fast.decode(s, hi)
        touched = set()
        for o in ops[:lastT - s + 1]:
            if o[0] & 0x7F in (0x30, 0x31, 0x32):
                touched.update(o[1])
        e = window_end(ops, lastT - s, touched)
        ops = ops[:e]
        if any((o[0] & 0x7F) == ZPOW_ for o in ops):
            key = ('zpow', ci)          # never merged: the angle word is not hashed
            sig, wires = key, []
        else:
            sig, wires = canon(ops)
        fresh = fresh_signature(fast, s, wires) if wires else ''
        key = (sig, fresh)
        c = classes.get(key)
        if c is None:
            c = classes[key] = dict(sig=sig, fresh=fresh, toffoli=sum(1 for o in ops if o[0] in TOFF), ops=len(ops), sites=[],
                                    cond_sites=0, regions=Counter())
        c['sites'].append((s, s + e))
        if fast.depth_before(s) > 0:
            c['cond_sites'] += 1
        if len(c['regions']) < 8 or regions.path(s) in c['regions']:
            c['regions'][regions.path(s)] += 1
        if progress and ci % 20000 == 0:
            progress(f'cluster {ci}/{len(cl)} {time.time() - t0:.0f}s classes {len(classes)}')
    return classes, dict(toffoli_like=int(len(tl)), clusters=len(cl), classes=len(classes), seconds=round(time.time() - t0, 1))


def merge_unknown_entry(classes):
    """A class whose entry freshness has 'u' (no touch within LOOKBACK; the wire may be untouched since allocation) joins
    the class with the same hash and 'f' at those positions when one exists (e.g. the first sites of a region). The joined
    sites are counted in `unknown_entry_sites` (their entry wires are unproved |0>). -> number of sites merged."""
    n = 0
    for key in [k for k in classes if 'u' in k[1] and not isinstance(k[0], tuple)]:
        tgt = (key[0], key[1].replace('u', 'f'))
        if tgt in classes and tgt != key:
            c, d = classes.pop(key), classes[tgt]
            d['sites'] += c['sites']
            d['cond_sites'] += c['cond_sites']
            d['regions'].update(c['regions'])
            d['unknown_entry_sites'] = d.get('unknown_entry_sites', 0) + len(c['sites'])
            n += len(c['sites'])
    for c in classes.values():
        c['sites'].sort()
    return n


# ------------------------------------------------------------------ G2: phase + value ANF of one window
class Unsupported(Exception):
    pass


class DqiSym:
    def __init__(self, fresh_wires):
        self.fresh = fresh_wires
        self.state, self.names, self.written = {}, [], []
        self.f = set()                 # phase monomials
        self.measured = set()
        self.hmr_bits = set()

    def get(self, w):
        p = self.state.get(w)
        if p is None:
            if w in self.fresh:
                p = set()
            else:
                if len(self.names) >= A.MAX_VARS:
                    raise A.TermCap
                p = {1 << len(self.names)}
                self.names.append(w)
            self.state[w] = p
        return p

    def write(self, w, p):
        if len(p) > TERM_CAP:
            raise A.TermCap
        self.state[w] = p
        if w not in self.written:
            self.written.append(w)

    def apply(self, op):
        tv, w, b, c = op
        base = tv & 0x7F
        if tv in (0x81, 0x01):
            raise Unsupported('push/pop condition')
        if tv & 0x80:
            if c in self.hmr_bits:
                return                    # zero branch of an in-window measurement: not applied
            raise Unsupported('condition on a bit written outside the window')
        if base == 0x11:
            self.write(w[0], self.get(w[0]) ^ {0})
        elif base == 0x21:
            self.write(w[1], self.get(w[1]) ^ self.get(w[0]))
        elif base == 0x31:
            a, bb, t = self.get(w[0]), self.get(w[1]), self.get(w[2])
            self.write(w[2], t ^ A.pmul(a, bb))
        elif base == 0x22:
            a, bb = self.get(w[0]), self.get(w[1])
            self.write(w[0], bb)
            self.write(w[1], a)
        elif base == 0x32:
            a, bb, cc = self.get(w[0]), self.get(w[1]), self.get(w[2])
            d = A.pmul(a, bb ^ cc)
            self.write(w[1], bb ^ d)
            self.write(w[2], cc ^ d)
        elif base == 0x30:
            self.f ^= A.pmul(A.pmul(self.get(w[0]), self.get(w[1])), self.get(w[2]))
        elif base == 0x20:
            self.f ^= A.pmul(self.get(w[0]), self.get(w[1]))
        elif base == 0x10:
            self.f ^= self.get(w[0])
        elif base == 0x00:
            self.f ^= {0}
        elif base == R_:
            self.state[w[0]] = set()
            self.measured.add(w[0])
        elif base == HMR_:
            self.get(w[0])
            self.state[w[0]] = set()
            self.measured.add(w[0])
            if b >= 0:
                self.hmr_bits.add(b)
        elif base == 0x06:
            return
        else:
            raise Unsupported(f'op {hex(tv)}')


def phase_degree_floor(d):
    """Smallest k with 3 * 2^(k-1) >= d for a phase of degree d >= 3 (see the module docstring); 0 for degree <= 2."""
    if d < 3:
        return 0
    k = 1
    while 3 * (1 << (k - 1)) < d:
        k += 1
    return k


def deg_of(monos):
    return max((m.bit_count() for m in monos), default=-1)


def next_touch_kinds(fast, end, wires):
    """kind of the first touch at/after `end` for each wire within LOOKAHEAD ops ({} for untouched)."""
    need, kind = set(wires), {}
    ops = fast.decode(end, end + LOOKAHEAD)
    for tv, w, _, _ in ops:
        for x in w:
            if x in need:
                kind[x] = tv & 0x7F
                need.discard(x)
        if not need:
            break
    return kind


def window_function(fast, a, b, fresh_override=None):
    """Exact arbitrary-entry function of ops [a, b) -> dict(status, phase_deg, value floor parts, ...)."""
    ops = fast.decode(a, b)
    wires = canon(ops)[1]
    sig = fresh_signature(fast, a, wires) if fresh_override is None else fresh_override
    fresh = {w for w, s in zip(wires, sig) if s == 'f'}
    st = DqiSym(fresh)
    try:
        for o in ops:
            st.apply(o)
    except Unsupported as e:
        return dict(status='unsupported', why=str(e))
    except A.TermCap:
        return dict(status='term_cap')
    pdeg = deg_of(st.f)
    nl = {w: p for w, p in st.state.items() if w in st.written and w not in st.measured and deg_of(p) >= 2}
    outs, lb_val, info = {}, 0, {}
    if nl:
        kinds = next_touch_kinds(fast, b, list(nl))
        for w, p in nl.items():
            if w in kinds and kinds[w] not in FRESH_KINDS:
                outs[f'q{w}'] = sorted(p)
        if outs:
            lb_val, info = A.full_floor(outs, len(st.names))
    lb_ph = phase_degree_floor(pdeg)
    return dict(status='exact_anf', floor=max(lb_val, lb_ph), floor_value=lb_val, floor_phase=lb_ph, phase_degree=pdeg,
                phase_terms=len(st.f), vars=len(st.names), live_nonlinear_outputs=len(outs), nonlinear_unproved_live=len(nl) - len(outs),
                floor_info=info, toffoli=sum(1 for o in ops if o[0] in TOFF), ops=len(ops))


# ------------------------------------------------------------------ pricing, negatives
def t_mean_from_manifest(row):
    if not row:
        return None, None
    try:
        q = float(row['qubits'])
        return float(row['dqi_zero_failure_cost']) / (R_ZERO * q * q), q
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None, None


def gm_pct(saving, t_mean):
    if not t_mean or saving <= 0:
        return 0.0
    if saving >= t_mean:
        return float('inf')
    return 100.0 * math.log(t_mean / (t_mean - saving)) / 3.0


NEG_WORDS = ('phase component', 'repeated', 'repeat', 'window', 'resynthesis', 'and count', 'and-count', 'rank floor', 'peephole',
             'component', 'mask projection')


def dqi_negatives(ledger, regions_seen):
    out = []
    regs = {r.split('/')[0] for r in regions_seen if r not in ('untagged', 'outside tags')}
    for r in ledger:
        text = ' '.join(r.get(k, '') for k in ('mechanism', 'family_target', 'key_numbers', 'why_it_failed', 'keywords'))
        low = text.lower()
        neg = r['outcome'] in ('kill', 'null', 'negative')
        named = [g for g in regs if g.lower() in low]
        if named and neg and any(k in low for k in NEG_WORDS):
            out.append(dict(id=r['id'], outcome=r['outcome'], covers=False, why=f'related: names region {named[0]} (keyword match; '
                                                                         'does not name this component)'))
    return out[:5]


def first_test(sha12, a, b, toffoli, floor):
    return dict(command=f'python3 scripts/qhdl.py avenues --task dqi -c {sha12} --ops {a}:{b}',
                stop=f'stop (close the lead) if the exact window function needs {toffoli} AND-type gates (floor {floor} is only a '
                     'lower bound) or no phase/value-equivalent block with fewer Toffoli-like gates is found within the budget',
                rung='verification ladder: window function vs candidate (arbitrary entry, all outcomes) -> splice at every site -> '
                     'tagged DQI region check -> 48/96-draw expected cost -> local official verifier')


# ------------------------------------------------------------------ the verb core
def run(path, sha12, manifest_row=None, regions_tree=None, ledger=None, top=10, class_cap=CLASS_CAP, progress=None):
    t0 = time.time()
    k = K.read(str(path))
    fast = Fast(k)
    regions = RegionIndex(regions_tree)
    classes, stats = build_classes(fast, regions, progress)
    merged_unknown = merge_unknown_entry(classes)
    stats['sites_merged_unknown_entry'] = merged_unknown
    tmean, q = t_mean_from_manifest(manifest_row)
    cls = sorted(classes.values(), key=lambda c: -c['toffoli'] * len(c['sites']))
    floored = 0
    rows = []
    for c in cls:
        c['mult'] = len(c['sites'])
        c['charged_total'] = c['toffoli'] * c['mult']
        c['floor'] = None
        c['status'] = 'not floored (beyond class cap)'
        if floored < class_cap and c['toffoli'] <= MAX_TOFFOLI and c['ops'] <= MAX_OPS_FLOOR and not isinstance(c['sig'], tuple):
            a, b = c['sites'][0]
            res = window_function(fast, a, b, fresh_override=c['fresh'])
            floored += 1
            c['status'] = res['status']
            c['detail'] = res
            if res['status'] == 'exact_anf' and res['floor'] == 0 and res['nonlinear_unproved_live'] > 0:
                c['status'] = 'floor unproved (nonlinear outputs not provably live within the lookahead)'
            elif res['status'] == 'exact_anf':
                c['floor'] = res['floor']
        elif c['toffoli'] > MAX_TOFFOLI or c['ops'] > MAX_OPS_FLOOR:
            c['status'] = f'window too large to floor (> {MAX_TOFFOLI} Toffolis or {MAX_OPS_FLOOR} ops)'
    ledger = ledger if ledger is not None else A.load_ledger(task='dqi')
    all_regions = Counter()
    for c in cls:
        all_regions.update(c['regions'])
    for c in cls:
        gap = max(0, c['toffoli'] - c['floor']) if c['floor'] is not None else None
        n_uncond = c['mult'] - c['cond_sites']
        S = (gap or 0) * c['mult']
        S_scen = (gap or 0) * (n_uncond + COND_WEIGHT * c['cond_sites'])
        reg = c['regions'].most_common(1)[0][0] if c['regions'] else 'untagged'
        a, b = c['sites'][0]
        negs = dqi_negatives(ledger, [reg]) if gap else []
        rows.append(dict(component=f"ops {a}:{b} sig {c['sig'] if not isinstance(c['sig'], tuple) else 'zpow'}/{c['fresh']}",
                         region=reg, regions=dict(c['regions'].most_common(4)), charged_each=c['toffoli'], ops_each=c['ops'],
                         floor=c['floor'], gap=gap, multiplicity=c['mult'], cond_sites=c['cond_sites'],
                         charged_total=c['charged_total'], saving_static=S, gm_effect_pct=round(gm_pct(S, tmean), 6),
                         gm_effect_pct_cond_scenario=round(gm_pct(S_scen, tmean), 6),
                         gm_if_free_pct=round(gm_pct(c['charged_total'], tmean), 5), status=c['status'],
                         floor_detail={kk: v for kk, v in (c.get('detail') or {}).items() if kk != 'status'},
                         unknown_entry_sites=c.get('unknown_entry_sites', 0),
                         sites=[list(s) for s in c['sites'][:(5000 if gap else 12)]], negatives=negs,
                         closed=any(n['covers'] for n in negs), first_test=first_test(sha12, a, b, c['toffoli'], c['floor'])))
    fam = Counter()
    for r in rows:
        fam[r['component'].split(' sig ')[1].split('/')[0]] += r['multiplicity']
    for r in rows:      # classes that differ only in entry freshness are the same function: report the family total too
        r['sig_family_multiplicity'] = fam[r['component'].split(' sig ')[1].split('/')[0]]
    open_rows = [r for r in rows if r['gap'] and not r['closed']]
    open_rows.sort(key=lambda r: (-r['gm_effect_pct'], r['ops_each']))
    closed_rows = [r for r in rows if r['gap'] and r['closed']]
    zero = [r for r in rows if r['gap'] == 0]
    unfloored = [r for r in rows if r['gap'] is None]
    unfloored.sort(key=lambda r: -r['charged_total'])
    return dict(stats=stats, regions_tagged=regions.tagged, t_mean=tmean, q=q, open=open_rows, closed=closed_rows, gap0=len(zero),
                unfloored=unfloored, classes=len(rows), floored=floored, seconds=round(time.time() - t0, 1),
                total_static_toffoli=stats['toffoli_like'])


def explain_window(path, a, b):
    k = K.read(str(path))
    fast = Fast(k)
    return window_function(fast, a, b)


# ------------------------------------------------------------------ CLI hook
def v_dqi(api, a):
    import qhdl_envelope as E
    if not a.circuit:
        raise api.QhdlError('--task dqi needs -c <circuit sha prefix or path>', 'e.g. -c bb4287e77eb6')
    P = api.resolve(a.circuit)
    sha12 = P['sha'][:12]
    mrow = P.get('manifest')
    if a.ops:
        x, y = [int(v) for v in a.ops.split(':')]
        res = explain_window(P['path'], x, y)
        led = E.Ledger()
        led.earn('static', 'exact arbitrary-entry ANF of one window (value outputs and phase polynomial)')
        rows = [{'k': kk, 'v': str(vv)[:90]} for kk, vv in res.items() if kk != 'floor_info']
        return E.Envelope(led, f"dqi {sha12} window {x}:{y}: {res.get('status')} floor {res.get('floor')} "
                                f"(phase degree {res.get('phase_degree')}, {res.get('toffoli')} Toffoli-like)", domain='dqi window',
                          evidence=api.rel(Path(P['path'])), rerun=api.cmd('avenues', '--task', 'dqi', '-c', sha12, '--ops', a.ops),
                          tool_sha=api.tool_sha(), circuit_sha=P['sha'], facts=res, rows=rows, table=None, next_cmds=[])
    tree = None
    if a.regions == 'none':
        tree = None                       # skip region naming entirely (untagged file or no tree cached)
    elif a.regions:
        tree = json.loads(Path(a.regions).read_text())
    else:
        try:
            import qhdl_diff as D
            tree, _ = D.regions_for(api, P['path'], sha12)
        except Exception as exc:  # tags are only names; an untagged or unreadable tree must not stop the pass
            tree = None
            print(f'regions unavailable ({exc}); sites are named "untagged"', file=sys.stderr)
    res = run(P['path'], sha12, mrow, tree, top=a.top, progress=lambda m: print(m, file=sys.stderr))
    out = (Path(a.out) / 'avenues_dqi.json') if a.out else Path(api.WORK) / sha12 / 'avenues' / 'avenues_dqi.json'
    full = dict(schema='vqcc-qhdl-avenues-dqi/1', circuit_sha256=P['sha'], tier='static model / projection',
                params=dict(cluster_gap=CLUSTER_GAP, tail=TAIL, lookback=LOOKBACK, class_cap=CLASS_CAP, max_toffoli=MAX_TOFFOLI,
                            cond_weight=COND_WEIGHT, r_zero=R_ZERO), **res)
    api.write_json(out, full)
    led = E.Ledger()
    led.earn('static', 'static Toffoli-like counts per repeated window class x multiplicity; exact arbitrary-entry ANF floors; '
                       'prices are arithmetic through R Q^2 T_mean (no executed means measured)')
    rows = []
    shown = res['open'][:a.top]
    for i, r in enumerate(shown):
        rows.append({'#': i + 1, 'ops': r['component'].split(' sig ')[0].replace('ops ', ''), 'region': r['region'][:18],
                     'cost': r['charged_each'], 'floor': r['floor'], 'gap': r['gap'], 'mult': r['multiplicity'],
                     'gm_pct': r['gm_effect_pct'], 'gm_cond64': r['gm_effect_pct_cond_scenario'],
                     'neg': ','.join(n['id'] for n in r['negatives'][:2]) or 0, 'tier': 'static'})
    below = res['closed'][:max(0, a.top - len(shown))]
    if below:
        rows.append({'#': '---', 'ops': 'closed by negatives (below the divider)', 'tier': 'static'})
        for i, r in enumerate(below):
            rows.append({'#': len(shown) + i + 1, 'ops': r['component'].split(' sig ')[0].replace('ops ', ''), 'region': r['region'][:18],
                         'cost': r['charged_each'], 'floor': r['floor'], 'gap': r['gap'], 'mult': r['multiplicity'],
                         'gm_pct': r['gm_effect_pct'], 'tier': 'static'})
    if res['unfloored']:
        rows.append({'#': '---', 'ops': 'charged-only ceilings (no floor: too large, term cap or unsupported); gm_pct = ceiling if free', 'tier': 'static'})
        for r in res['unfloored'][:3]:
            rows.append({'#': 'ceil', 'ops': r['component'].split(' sig ')[0].replace('ops ', ''), 'region': r['region'][:18],
                         'cost': r['charged_each'], 'floor': 'n/a', 'gap': 'n/a', 'mult': r['multiplicity'],
                         'gm_pct': r['gm_if_free_pct'], 'tier': 'static'})
    claim = (f"{len(res['open'])} open DQI avenues (+{len(res['closed'])} closed) from {res['classes']} repeated-window classes over "
             f"{res['stats']['clusters']} clusters of {res['stats']['toffoli_like']} Toffoli-like ops "
             f"({'tagged' if res['regions_tagged'] else 'untagged file: region names unavailable'}); {res['floored']} classes floored, "
             f"{res['gap0']} at their floor; gaps are upper bounds, static weights")
    facts = dict(stats=res['stats'], t_mean=res['t_mean'], q=res['q'], tagged=res['regions_tagged'], full_json=api.rel(out),
                 pricing='GM = (1/3) ln(T/(T-S)), T = zero-failure cost/(1.1913 Q^2); S = gap x multiplicity (static, every site)')
    return E.Envelope(led, claim, domain='DQI: tag-independent Toffoli clusters hashed up to wire relabelling; exact ANF floors',
                      evidence=api.rel(out), rerun=api.cmd('avenues', '--task', 'dqi', '-c', sha12, '--top', a.top),
                      tool_sha=api.tool_sha(), circuit_sha=P['sha'], facts=facts, rows=rows, table=api.rel(out),
                      next_cmds=[r['first_test']['command'] for r in shown[:2]])
