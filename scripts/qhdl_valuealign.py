#!/usr/bin/env python3
"""qhdl plugin: `valuealign` — source-bound parent/child value alignment (companion to `valuegraph`).

    qhdl valuealign -c PARENT -c2 CHILD [--out DIR] [--all]

For two circuits, align the values of their qhdl-value-graph/v1 graphs and write DIR/alignment.jsonl + summary.json.
Statuses (static tier; op streams only, no simulation):
  parent value  matched | erased_in_child | unknown      child value  matched | new_in_child
The word "equivalent" is never produced. A match is a *static correspondence with evidence*, not a proof that the two
values hold the same quantum state: wire names, op order and gate scheduling differ between a parent and its child.

How a match is made
  1. Structural normal form (nf). Every value gets an interned expression id: X, CX and SWAP are exact (XOR-linear
     over atoms); a CCX target adds one AND atom keyed by the (sorted) nf ids of its two controls; a CSWAP is the two
     matching XOR-with-AND updates; HMR outcomes (numbered per nf id) and guarded writes are opaque atoms. Wire names
     and op order do not enter, so a renamed or reordered copy of a computation gets the same id. This is a syntactic
     hash: equal nf is *evidence*, never a semantic claim (a different gate order of the same function can differ;
     the same nf can arise from two distinct values, which is reported as ambiguity).
  2. Candidates are the parent and child values with an equal non-constant nf id. A 1:1 group is a match; larger
     groups are resolved by the op-text anchor (difflib on op text, `qhdl_edit.auto_hunks`), then by live-interval
     overlap in anchor-interpolated coordinates; a pair with neither is left unknown (ambiguous).
  3. Producer-structure propagation: an unmatched value whose producer opcode and every producer input are matched to
     a child value with the same producer shape gets that value (zeros after a matched reset or measurement, guarded
     writes, values whose nf differs only through a reordered reduction).
Evidence per match: `op_text_anchor` (the producer ops are the same text at aligned positions), `producer_inputs`
(fraction of the parent's producer inputs whose match is among the child's inputs), `consumer_roles` (Jaccard of the
(opcode, role) multisets of the consumers), `interval_iou` (live interval overlap after mapping the parent's cuts into
the child's through the anchors), `nf_group` (sizes of the candidate group). Confidence class: high (nf 1:1 or
anchored, inputs all aligned, and consumers or interval agree), medium (nf or structure with partial agreement), low
(ambiguous group, structure-only, or weak agreement). Classes are a fixed rule, not a probability.

erased_in_child: the matched child value is killed by an unconditional measurement or reset while the parent's value is
not (it is transformed in place or kept). `recovered_candidates` lists child values that stand in the child for the
erased one at the parent's later consumers (positional input substitution; votes, birth cut and wire). Whether the
child's recovery is correct is not checked here.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_bitsim as B  # noqa: E402
import qhdl_valuegraph as G  # noqa: E402

SCHEMA = 'qhdl-value-alignment/v1'
SET_CAP = 400
HARD_KILLS = ('measured', 'reset')
ALIGN_TOL = 2               # cuts of slack when comparing live intervals (point-like values)


def _base(opc):
    return opc[:-3] if opc.endswith('_IF') else opc


# ---------------------------------------------------------------- structural normal form
class Interner:
    """Shared by both circuits so equal expressions get equal ids."""

    def __init__(self):
        self.atoms, self.exprs, self.info = {}, {}, []

    def atom(self, key):
        a = self.atoms.get(key)
        if a is None:
            a = self.atoms[key] = len(self.atoms)
        return a

    def expr(self, c, fs):
        k = (c, fs)
        e = self.exprs.get(k)
        if e is None:
            e = self.exprs[k] = len(self.info)
            self.info.append(k)
        return e

    def one(self, key):
        return self.expr(0, frozenset((self.atom(key),)))

    def zero(self):
        return self.expr(0, frozenset())

    def const1(self):
        return self.expr(1, frozenset())

    def xor(self, a, b):
        ca, fa = self.info[a]
        cb, fb = self.info[b]
        return self.expr(ca ^ cb, fa ^ fb)

    def flip(self, a):
        c, f = self.info[a]
        return self.expr(c ^ 1, f)

    def and_(self, a, b):
        x, y = (a, b) if a <= b else (b, a)
        return self.one(('and', x, y))

    def trivial(self, e):
        return not self.info[e][1]


def normal_forms(values, ops, it):
    """-> dict value id -> expression id (insertion order of `values` is creation order, producers first)."""
    E = {}
    meas = Counter()
    for vid, v in values.items():
        p = v['producer']
        opc = _base(p['opcode'])
        ins = p['inputs']
        if opc == 'REGISTER_INPUT':
            E[vid] = it.one(('in', v['wire']))
            continue
        if opc == 'INIT_ZERO':
            E[vid] = it.zero()
            continue
        if opc == 'INIT_BIT':
            E[vid] = it.one(('ib', v['bit']))
            continue
        ie = [E[i] for i in ins]
        if p.get('conditional'):
            pos = 0
            if opc in ('SWAP', 'CSWAP') and p['op'] is not None:
                pos = list(ops[p['op']].q).index(v['wire'])
            E[vid] = it.one(('g', opc, pos, ie[0] if ie else -1, tuple(sorted(ie[1:]))))
            continue
        if opc in ('R', 'HMR') and v.get('post_reset'):
            E[vid] = it.zero()
        elif opc == 'HMR':
            e = E[p['measured_value']] if 'measured_value' in p else ie[0]
            n = meas[e]
            meas[e] += 1
            E[vid] = it.one(('m', e, n))
        elif opc == 'X':
            E[vid] = it.flip(ie[0])
        elif opc == 'CX':
            E[vid] = it.xor(ie[0], ie[1])
        elif opc == 'CCX':
            E[vid] = it.xor(ie[0], it.and_(ie[1], ie[2]))
        elif opc == 'SWAP':
            pos = list(ops[p['op']].q).index(v['wire'])
            E[vid] = ie[1 - pos]
        elif opc == 'CSWAP':
            pos = list(ops[p['op']].q).index(v['wire'])      # q = (control, a, b); a is position 1
            c, a, b = ie[0], ie[1], ie[2]
            t = it.and_(c, it.xor(a, b))
            E[vid] = it.xor(a if pos == 1 else b, t)
        elif opc == 'BIT_STORE0':
            E[vid] = it.zero()
        elif opc == 'BIT_STORE1':
            E[vid] = it.const1()
        elif opc == 'BIT_INVERT':
            E[vid] = it.flip(ie[0])
        else:
            E[vid] = it.one(('?', opc, tuple(ie)))
        if len(it.info[E[vid]][1]) > SET_CAP:
            E[vid] = it.one(('cap', opc, tuple(ie)))
    return E


# ---------------------------------------------------------------- side data
def signatures(ops, xq, nq, values, n_rows=1024, seed=7):
    """Sampled truth-table hash of every non-constant value, under the 'zero' outcome model (every HMR returns 0, so
    repair blocks never fire). Tier sampled(n_rows): the same random legal x rows are used for both circuits."""
    import numpy as np
    rows = B.legal_rows()
    rows = np.random.default_rng(seed).choice(rows, size=n_rows, replace=False)
    sim = B.BitSim(ops, rows=rows, num_qubits=nq, x_qubits=list(xq), outcome='zero')
    by_op = defaultdict(list)
    for vid, v in values.items():
        by_op[v['producer']['op']].append(vid)
    sig, zero = {}, set()

    def grab(vids):
        for vid in vids:
            v = values[vid]
            arr = sim.q[v['wire']] if 'wire' in v else sim._bit(v['bit'])
            b = arr.tobytes()
            if not any(b):
                zero.add(vid)
            sig[vid] = None if (not any(b) or b == sim.valid.tobytes()) else int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'big')
    grab(by_op.get(None, []))
    for i in range(len(ops)):
        sim.run(stop=i + 1)
        if i in by_op:
            grab(by_op[i])
    return sig, zero


class Side:
    def __init__(self, ops, xq, nq, it, n_rows=0):
        self.ops = ops
        self.values, self.uses, _ = G.build(ops, xq, nq, None)
        self.nf = normal_forms(self.values, ops, it)
        self.sig, self.zero = signatures(ops, xq, nq, self.values, n_rows) if n_rows else (None, set())
        self.n = len(ops)
        self.n_rows = n_rows
        self.cons = {}
        for vid, v in self.values.items():
            self.cons[vid] = Counter((B.KINDS[ops[c['op']].code], c['role']) for c in v['consumers'])

    def shape(self, v, pos_of=None):
        p = v['producer']
        return _base(p['opcode'])


def _lis_pairs(pairs):
    """pairs sorted by x; longest strictly increasing subsequence in y -> list of (x, y)."""
    tails, idx, prev = [], [], [-1] * len(pairs)
    for i, (_, y) in enumerate(pairs):
        j = bisect.bisect_left(tails, y)
        if j == len(tails):
            tails.append(y)
            idx.append(i)
        else:
            tails[j] = y
            idx[j] = i
        prev[i] = idx[j - 1] if j else -1
    out, k = [], idx[-1] if idx else -1
    while k >= 0:
        out.append(pairs[k])
        k = prev[k]
    return out[::-1]


class CutMap:
    """Monotone piecewise-linear map from parent cuts to child cuts through anchor pairs."""

    def __init__(self, pairs, n_parent, n_child):
        pts = sorted(set([(0, 0), (n_parent, n_child)] + list(pairs)))
        pts = _lis_pairs([p for p in pts if p[0] not in (0, n_parent)])
        self.xs = [0] + [p[0] for p in pts] + [n_parent]
        self.ys = [0] + [p[1] for p in pts] + [n_child]
        self.anchors = len(pts)

    def __call__(self, x):
        j = bisect.bisect_right(self.xs, x)
        if j >= len(self.xs):
            return self.ys[-1]
        if j == 0:
            return self.ys[0]
        x0, x1, y0, y1 = self.xs[j - 1], self.xs[j], self.ys[j - 1], self.ys[j]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _interval(v):
    b = v['birth_cut']
    e = v['last_use_cut'] if v['last_use_cut'] is not None else b
    return b - ALIGN_TOL, max(e, b) + 1 + ALIGN_TOL


def _iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    uni = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return 1.0 if uni <= 0 and a[0] == b[0] else (inter / uni if uni > 0 else 0.0)


def _jacc(a, b):
    ka = set(a) | set(b)
    if not ka:
        return 1.0
    return sum(min(a[k], b[k]) for k in ka) / sum(max(a[k], b[k]) for k in ka)


# ---------------------------------------------------------------- alignment
def align(P, C, it, hunks=None, use_sig=False):
    """-> result dict. P, C: Side. hunks: op-text hunks parent->child (computed when None)."""
    import qhdl_edit as ED
    if hunks is None:
        hunks = ED.auto_hunks([o.text.strip() for o in P.ops], [o.text.strip() for o in C.ops])
    opmap = {}
    for tag, i1, i2, j1, j2 in hunks:
        if tag == 'equal':
            for k in range(i2 - i1):
                opmap[i1 + k] = j1 + k
    pv, cv = P.values, C.values
    groups_p, groups_c = defaultdict(list), defaultdict(list)
    kp = {v: (None if it.trivial(e) else e) for v, e in P.nf.items()} if not use_sig else P.sig
    kc = {v: (None if it.trivial(e) else e) for v, e in C.nf.items()} if not use_sig else C.sig
    pref = 'sig' if use_sig else 'nf'
    for vid, e in kp.items():
        if e is not None:
            groups_p[e].append(vid)
    for vid, e in kc.items():
        if e is not None:
            groups_c[e].append(vid)
    match, how = {}, {}            # parent id -> child id ; parent id -> (method, group sizes)
    taken = set()

    def text_anchor(p, c):
        po, co = pv[p]['producer']['op'], cv[c]['producer']['op']
        return po is not None and co is not None and opmap.get(po) == co

    # stage A1: unique 1:1 groups
    ambiguous = []
    for e, ps in groups_p.items():
        cs = groups_c.get(e)
        if not cs:
            continue
        if len(ps) == 1 and len(cs) == 1:
            match[ps[0]] = cs[0]
            how[ps[0]] = (pref + '_unique', 1, 1)
            taken.add(cs[0])
        else:
            ambiguous.append(e)
    # anchors for the cut map: unique matches with real producer ops, plus equal-hunk starts
    pairs = []
    for p, c in match.items():
        po, co = pv[p]['producer']['op'], cv[c]['producer']['op']
        if po is not None and co is not None:
            pairs.append((po + 1, co + 1))
    for tag, i1, i2, j1, j2 in hunks:
        if tag == 'equal' and i2 > i1:
            pairs.append((i1, j1))
    cmap = CutMap(pairs, P.n, C.n)
    # stage A2: ambiguous groups by text anchor, then interval overlap
    for e in ambiguous:
        ps, cs = groups_p[e], [c for c in groups_c[e] if c not in taken]
        if not cs:
            continue
        cb = sorted(cs, key=lambda c: cv[c]['birth_cut'])
        births = [cv[c]['birth_cut'] for c in cb]
        cand = []
        for p in ps:
            pi = _interval(pv[p])
            mp = (cmap(pi[0]), cmap(pi[1]))
            j = bisect.bisect_left(births, mp[0])
            for c in cb[max(0, j - 6):j + 6]:
                sc = (3.0 if text_anchor(p, c) else 0.0) + _iou(mp, _interval(cv[c]))
                cand.append((sc, p, c, text_anchor(p, c), _iou(mp, _interval(cv[c]))))
        cand.sort(key=lambda t: -t[0])
        for sc, p, c, ta, iou in cand:
            if p in match or c in taken or not (ta or iou >= 0.2):
                continue
            match[p] = c
            how[p] = (pref + ('_text_anchor' if ta else '_interval'), len(ps), len(groups_c[e]))
            taken.add(c)
    # stage B: producer-structure propagation
    def key_of(v, M):
        p = v['producer']
        if p['op'] is None:
            return None
        ins = []
        for i in p['inputs']:
            m = M(i)
            if m is None:
                return None
            ins.append(m)
        pos = 0
        if _base(p['opcode']) in ('SWAP', 'CSWAP'):
            pos = 1  # position matters; handled below by wire role
        if not ins:
            return None
        return (_base(p['opcode']), bool(v.get('post_reset')), v['kind'], ins[0], tuple(sorted(ins[1:])))

    for _ in range(8):
        idx = defaultdict(list)
        for c, v in cv.items():
            if c in taken:
                continue
            k = key_of(v, lambda i: i)
            if k:
                idx[k].append(c)
        added = 0
        for p, v in pv.items():
            if p in match:
                continue
            k = key_of(v, lambda i: match.get(i))
            if k is None:
                continue
            cs = idx.get(k)
            if not cs:
                continue
            if len(cs) == 1 and cs[0] not in taken:
                match[p] = cs[0]
                how[p] = ('producer_structure', 1, 1)
                taken.add(cs[0])
                added += 1
            elif len(cs) > 1:
                best = [c for c in cs if c not in taken]
                pi = _interval(v)
                mp = (cmap(pi[0]), cmap(pi[1]))
                best.sort(key=lambda c: -_iou(mp, _interval(cv[c])))
                if best and (text_anchor(p, best[0]) or _iou(mp, _interval(cv[best[0]])) >= 0.2) and \
                        (len(best) == 1 or _iou(mp, _interval(cv[best[0]])) > _iou(mp, _interval(cv[best[1]])) + 1e-9):
                    match[p] = best[0]
                    how[p] = ('producer_structure_ambiguous', len(cs), len(cs))
                    taken.add(best[0])
                    added += 1
        if not added:
            break
    # stage C: constants and unkeyed values, by producer-op text anchor on the same wire/bit
    by_prod = {}
    for c, v in cv.items():
        if c not in taken:
            by_prod[(v['producer']['op'], v.get('wire'), v.get('bit'), v['kind'])] = c
    for p, v in pv.items():
        if p in match:
            continue
        po = v['producer']['op']
        if po is None:
            c = by_prod.get((None, v.get('wire'), v.get('bit'), v['kind']))
        else:
            c = by_prod.get((opmap.get(po), v.get('wire'), v.get('bit'), v['kind'])) if po in opmap else None
        if c is not None and c not in taken and (po is None or kp.get(p) is None):
            match[p] = c
            how[p] = ('anchor_only', 1, 1)
            taken.add(c)
    return dict(pref=pref, kp=kp, kc=kc, match=match, how=how, cmap=cmap, opmap=opmap, hunks=hunks, groups_p=groups_p, groups_c=groups_c)


def evidence(P, C, res, p, c):
    pv, cv = P.values[p], C.values[c]
    method, gp, gc = res['how'][p]
    po, co = pv['producer']['op'], cv['producer']['op']
    ta = po is not None and co is not None and res['opmap'].get(po) == co
    pin, cin = pv['producer']['inputs'], cv['producer']['inputs']
    cset = set(cin)
    ins_aligned = [res['match'].get(i) in cset for i in pin]
    frac = (sum(ins_aligned) / len(pin)) if pin else 1.0
    cm = res['cmap']
    pi = _interval(pv)
    iou = _iou((cm(pi[0]), cm(pi[1])), _interval(cv))
    cj = _jacc(P.cons[p], C.cons[c])
    ev = {
        'method': method,
        'key_group': [gp, gc],
        'nf_equal': P.nf[p] == C.nf[c],
        'sig_equal': (P.sig[p] == C.sig[c] and P.sig[p] is not None) if P.sig else None,
        'op_text_anchor': ('same_op' if ta else ('parent_op_anchored_elsewhere' if po in res['opmap'] else 'no_anchor')),
        'producer_opcode': [pv['producer']['opcode'], cv['producer']['opcode']],
        'producer_inputs': {'aligned_fraction': round(frac, 3), 'n': len(pin)},
        'consumer_roles': {'jaccard': round(cj, 3), 'parent': sum(P.cons[p].values()), 'child': sum(C.cons[c].values())},
        'interval_iou': round(iou, 3),
        'intervals': {'parent': [pv['birth_cut'], pv['last_use_cut']], 'child': [cv['birth_cut'], cv['last_use_cut']],
                      'parent_mapped': [round(cm(pi[0])), round(cm(pi[1]))]},
    }
    unique = method.endswith('_unique') or ta
    agree = (cj >= 0.5) or (iou >= 0.3)
    strong = cj >= 0.5 and iou >= 0.5
    if method.endswith('_unique') or method.endswith('_text_anchor'):
        conf = 'high' if (frac == 1.0 and agree and unique) or strong else ('medium' if agree or frac == 1.0 else 'low')
    elif method.endswith('_interval'):
        conf = 'medium' if (frac >= 0.5 and agree) or strong else 'low'
    elif method == 'producer_structure':
        conf = 'medium' if agree else 'low'
    elif method == 'anchor_only':
        conf = 'medium' if ta and frac == 1.0 else 'low'
    else:
        conf = 'low'
    return ev, conf


def _chain(S, vid):
    """Supersession chain after `vid` (successor links), as value ids."""
    out = []
    k = (S.values[vid].get('kill') or {}).get('successor')
    while k is not None and len(out) < 5000:
        out.append(k)
        k = (S.values[k].get('kill') or {}).get('successor')
    return out


def _first_zero(S, vid):
    """First later version on the chain with an all-zero sampled signature (needs --sig), as a value id or None."""
    for k in _chain(S, vid):
        if k in S.zero:
            return k
    return None


EARLY_SLACK = 50


def erase_info(P, C, res, p, c):
    """Child erases the matched value earlier than the parent does. Static evidence: unconditional measurement or reset
    of the child value where the parent's is not. Sampled evidence (--sig): a later version of the child's wire is
    all-zero on every sampled row (erased by uncomputation), while the parent's chain is not zero until later."""
    pk, ck = P.values[p].get('kill') or {}, C.values[c].get('kill') or {}
    cm = res['cmap']
    lu = P.values[p]['last_use_cut']
    out = None
    p_hard = pk.get('kind') in HARD_KILLS + ('conditional_measure', 'conditional_reset')
    earlier = (ck.get('kind') in HARD_KILLS and p_hard and pk.get('kind') in HARD_KILLS
               and cm(P.values[p]['kill_cut']) - C.values[c]['kill_cut'] > EARLY_SLACK)
    if ck.get('kind') in HARD_KILLS and (not p_hard or earlier):
        out = dict(mechanism='hard_kill_earlier_in_child' if earlier else 'hard_kill_in_child', tier='static',
                   child_kill=dict(op=ck['op'], kind=ck['kind'], cut=C.values[c]['kill_cut']),
                   parent_kill=dict(op=pk.get('op'), kind=pk.get('kind'), cut=P.values[p]['kill_cut']),
                   child_erased_cut=C.values[c]['kill_cut'])
    elif C.sig is not None:
        cz = _first_zero(C, c)
        if cz is not None:
            ce = C.values[cz]['birth_cut']
            pz = _first_zero(P, p)
            pzc = P.values[pz]['birth_cut'] if pz else None
            if pzc is None or cm(pzc) - ce > EARLY_SLACK:
                out = dict(mechanism='zeroed_in_child', tier=f'sampled({C.n_rows})',
                           child_zero_value=cz, child_zero_cut=ce, child_zero_producer=C.values[cz]['producer']['opcode'],
                           parent_zero_value=pz, parent_zero_cut=pzc,
                           child_erased_cut=ce)
    if out is None:
        return None
    out['parent_last_use_mapped'] = round(cm(lu)) if lu is not None else None
    out['parent_holds_past_child_erase'] = (lu is not None and cm(lu) > out['child_erased_cut'])
    return out


def recovered_candidates(P, C, res, p, c, erase):
    """Child values that stand in for the erased value after the child erased it. Two kinds of evidence, listed apart:
    `substitution` (static): at the parent's consumers of p, the matched child op reads this other value in the same
    input position; `same_signature` (sampled, needs --sig): an unmatched child value born after the erase with the
    same sampled truth table as p."""
    ce = erase['child_erased_cut']
    inv = {v: k for k, v in res['match'].items()}
    votes = Counter()
    for q, v in P.values.items():
        pr = v['producer']
        if pr['op'] is None or p not in pr['inputs']:
            continue
        m = res['match'].get(q)
        if m is None:
            continue
        cin = C.values[m]['producer']['inputs']
        if _base(C.values[m]['producer']['opcode']) != _base(pr['opcode']):
            continue
        k = pr['inputs'].index(p)
        if k >= len(cin):
            continue
        r = cin[k]
        if r == c or C.values[r]['birth_cut'] < ce or r in inv:
            continue
        votes[r] += 1

    def row(r, kind, n=None):
        d = dict(value=r, evidence=kind, birth_cut=C.values[r]['birth_cut'], wire=C.values[r].get('wire'),
                 producer=C.values[r]['producer']['opcode'], producer_op=C.values[r]['producer']['op'])
        if n is not None:
            d['votes'] = n
        return d
    out = [row(r, 'substitution', n) for r, n in votes.most_common(3)]
    if P.sig is not None and res['kp'].get(p) is not None:
        same = [r for r in res['groups_c'].get(res['kp'][p], []) if r not in inv and C.values[r]['birth_cut'] >= ce]
        same.sort(key=lambda r: C.values[r]['birth_cut'])
        out += [row(r, 'same_signature') for r in same[:3]]
    return out


def build_rows(P, C, res):
    rows, summ = [], Counter()
    inv = {c: p for p, c in res['match'].items()}
    for p, v in P.values.items():
        base = dict(side='parent', id=p, kind=v['kind'], wire=v.get('wire', v.get('bit')), birth_cut=v['birth_cut'],
                    last_use_cut=v['last_use_cut'])
        c = res['match'].get(p)
        if c is not None:
            ev, conf = evidence(P, C, res, p, c)
            er = erase_info(P, C, res, p, c)
            solid = er is not None and conf != 'low'
            r = dict(base, status='erased_in_child' if solid else 'matched', matches=[dict(value=c, confidence=conf, evidence=ev)])
            if er and not solid:
                r['erase_hint'] = dict(er, note='match confidence is low, so this is not reported as erased_in_child')
            if solid:
                r['erase'] = er
                r['recovered_candidates'] = recovered_candidates(P, C, res, p, c, er)
            r['confidence'] = conf
            summ[r['status'] + ':' + conf] += 1
        else:
            e = res['kp'][p]
            if e is None:
                why = 'constant value (not matched)'
            elif not res['groups_c'].get(e):
                why = 'no child value with an equal ' + ('signature' if res['pref'] == 'sig' else 'normal form')
            else:
                why = 'equal normal form in the child but ambiguous or already taken'
            r = dict(base, status='unknown', matches=[], reason=why, confidence='none')
            summ['unknown'] += 1
        rows.append(r)
    for c, v in C.values.items():
        if c in inv:
            continue
        rows.append(dict(side='child', id=c, kind=v['kind'], wire=v.get('wire', v.get('bit')), birth_cut=v['birth_cut'],
                         last_use_cut=v['last_use_cut'], status='new_in_child', matches=[],
                         producer=v['producer']['opcode'], producer_op=v['producer']['op']))
        summ['new_in_child'] += 1
    return rows, summ


def summarize(P, C, rows, summ, res, psha, csha, sig_rows=0):
    st = Counter(r['status'] for r in rows if r['side'] == 'parent')
    return dict(schema=SCHEMA, evidence_tier=('static' if not sig_rows else f'static + sampled({sig_rows}) signature evidence'), parent_sha256=psha, child_sha256=csha,
                parent_values=len(P.values), child_values=len(C.values), parent_ops=P.n, child_ops=C.n,
                parent_status=dict(st), child_new_in_child=sum(1 for r in rows if r['status'] == 'new_in_child'),
                matched_total=len(res['match']), by_status_confidence=dict(summ),
                method_counts=dict(Counter(h[0] for h in res['how'].values())),
                op_text_equal_fraction_of_parent=round(sum(1 for _ in res['opmap']) / max(1, P.n), 4),
                cut_map_anchors=res['cmap'].anchors,
                key=res['pref'], sig_rows=sig_rows, caveat='static correspondences with evidence; never a statement of semantic equivalence. Structural normal '
                       'form equality is a syntactic hash of the op stream; confidence classes are a fixed rule, not probabilities.')


def write_out(out_dir, rows, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / 'alignment.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, separators=(',', ':')) + '\n')
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


# ---------------------------------------------------------------- qhdl verb
def _load(api, ref):
    meta = api.ensure_open(ref)
    d = api.ws_dir(meta['sha256'])
    ops, xq, nq = G.load_ops(d / 'circuit.kmx')
    return meta, d, ops, xq, max(nq, meta['declared_qubits'])


def v_valuealign(api, a):
    import qhdl_envelope as E
    mp, dp, opsp, xqp, nqp = _load(api, a.circuit)
    mc, dc, opsc, xqc, nqc = _load(api, a.circuit2)
    it = Interner()
    P, C = Side(opsp, xqp, nqp, it, a.sig), Side(opsc, xqc, nqc, it, a.sig)
    res = align(P, C, it, use_sig=bool(a.sig))
    rows, summ = build_rows(P, C, res)
    summary = summarize(P, C, rows, summ, res, mp['sha256'], mc['sha256'], a.sig)
    if a.out:
        out = Path(a.out)
    else:
        out = dp / f"valuealign-{mc['sha12']}"
        k = 1
        while out.exists():
            k += 1
            out = dp / f"valuealign-{mc['sha12']}-{k}"
    write_out(out, rows, summary)
    led = E.Ledger()
    led.earn('static', 'value-graph alignment from the two op streams (no simulation)')
    if a.sig:
        led.earn(f'sampled({a.sig})', f'sampled truth-table signatures of values on {a.sig} random legal x (zero-outcome model)')
    er = [r for r in rows if r['status'] == 'erased_in_child']
    slim = [dict(parent=r['id'], child=r['matches'][0]['value'], confidence=r['confidence'],
                 mechanism=r['erase']['mechanism'], child_erased_cut=r['erase']['child_erased_cut'],
                 recovered=[x['value'] for x in r['recovered_candidates']]) for r in er]
    claim = (f"{mp['sha12']} -> {mc['sha12']}: {len(res['match'])} of {len(P.values)} parent values matched "
             f"({len(er)} erased_in_child), {summary['child_new_in_child']} child values new_in_child "
             f"({'static' if not a.sig else 'static op-stream evidence + sampled(%d) signature keys' % a.sig}; correspondence, not equivalence)")
    tier = 'static' if not a.sig else f'sampled({a.sig})'
    parts = [] if not a.sig else [
        dict(part='op-stream evidence (text anchors, producer inputs, consumer roles, intervals, hard kills)', tier='static',
             basis='value graphs of the two op streams'),
        dict(part='match keys and zeroed-in-child erasures', tier=f'sampled({a.sig})',
             basis=f'truth-table hashes on {a.sig} random legal x, zero-outcome model')]
    return E.Envelope(led, claim, tier=tier, domain='two op streams, static correspondence' + ('' if not a.sig else ' + sampled signatures'), evidence=api.rel(out / 'alignment.jsonl'),
                      rerun=api.cmd('valuealign', '-c', mp['sha12'], '-c2', mc['sha12'], *(['--sig', a.sig] if a.sig else []),
                                    *(['--out', a.out] if a.out else [])),
                      tool_sha=api.tool_sha(), circuit_sha=mp['sha256'],
                      facts={k: v for k, v in summary.items() if k not in ('caveat',)}, rows=slim, parts=parts, next_cmds=[])


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('valuealign', help='align the values of a parent and child circuit (static, evidence per match)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--circuit', required=True, help='parent circuit (.kmb path or sha prefix)')
    p.add_argument('-c2', '--circuit2', required=True, help='child circuit')
    p.add_argument('--out')
    p.add_argument('--sig', type=int, default=0, help='N>0: add sampled truth-table signatures on N random legal x as match keys (tier sampled(N), recorded per match)')
    p.add_argument('--all', action='store_true')
    p.set_defaults(fn=lambda a: v_valuealign(api, a))


def main(argv=None):
    ap = argparse.ArgumentParser(description='standalone: align two opened workspace dirs (circuit.kmx)')
    ap.add_argument('parent', type=Path)
    ap.add_argument('child', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--sig', type=int, default=0)
    a = ap.parse_args(argv)
    it = Interner()
    sides = []
    for d in (a.parent, a.child):
        ops, xq, nq = G.load_ops(d / 'circuit.kmx')
        sides.append(Side(ops, xq, nq, it, a.sig))
    P, C = sides
    res = align(P, C, it, use_sig=bool(a.sig))
    rows, summ = build_rows(P, C, res)
    summary = summarize(P, C, rows, summ, res, str(a.parent), str(a.child), a.sig)
    write_out(a.out, rows, summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
