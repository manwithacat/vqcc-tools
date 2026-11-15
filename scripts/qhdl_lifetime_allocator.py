#!/usr/bin/env python3
"""Certified lifetime allocator for kickmix circuits.

Recolours qubit ids by lifetime without changing the gate sequence. Every qubit operand is a reference
(op index, logical id). A logical id's references split into *intervals*:

  * an interval ENDS at an unconditional reset touch (R or HMR, not an _IF form, condition-stack depth 0):
    after it the wire is |0> in every shot;
  * an interval also ends at the id's LAST touch overall: the verifiers require every scratch qubit to be |0>
    at the end and nothing touches the wire afterwards, so on every input the original passes, it is already
    |0> right after that touch (register qubits excepted);
  * the next touch after an end STARTS a new interval, which needs only a wire holding |0>.

Register qubits (inputs/outputs read by the verifier) are pinned: their first interval runs from op 0 and their
last to the end of the circuit, both on the register's own id. With --register-gaps the intervals strictly between
(after a reset of the register wire and before its final recomputation) are coloured like scratch; their pinned
colour is then free in the gap. Without it the register id is reserved for the whole circuit.

Colouring is greedy left-edge (lowest free id) over intervals sorted by start, which is optimal for an interval
graph when no register gaps are used: the number of colours equals the maximum number of simultaneously live
intervals (the clique number, reported as `lower_bound`).

Output: a .kmb identical to the input except the qubit streams, register qubit ids and header num_qubits, plus a
JSON report. `check_equivalence` independently re-derives, from the input and output alone, that the output is the
input under a time-varying qubit renaming that only moves values onto wires holding |0>.

Usage: python qhdl_lifetime_allocator.py IN.kmb OUT.kmb [--register-gaps] [--report OUT.json] [--backend rust]

--backend rust computes the intervals, profile and recoloured qubit streams with kickmix_analysis (crate
kmx-analysis in the engine repo, streaming; byte-identical on a parity set of circuits); colouring and the independent
equivalence check stay in Python. Default python (the reference).
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402

QSTREAMS = ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def reset_ops(k):
    """Boolean per op: unconditional R/HMR (op type exactly R or HMR, condition-stack depth 0)."""
    t = k.op_types
    depth = K.condition_depth(t)
    return ((t == K.R) | (t == K.HMR)) & (depth == 0), depth


def references(k):
    ops, qs, segs = K.op_qubits(k)
    key = (qs.astype(np.uint64) << np.uint64(32)) | ops.astype(np.uint64)  # unique: an op never repeats a qubit
    order = np.argsort(key)  # by logical id, then op index
    del key
    return ops, qs, order, segs


def register_qubits(k):
    out = {}
    for name, contents in k.registers:
        for is_q, qid, _ in contents:
            if is_q:
                out[qid] = name
    return out


def block_of(t):
    """Function op -> innermost enclosing PUSH_CONDITION op index (or -1), from the PUSH/POP event list."""
    ev = np.flatnonzero((t == K.PUSH_CONDITION) | (t == K.POP_CONDITION))
    blk = np.empty(len(ev), np.int64)
    stack = []
    for i, o in enumerate(ev.tolist()):
        if t[o] == K.PUSH_CONDITION:
            stack.append(o)
        else:
            stack.pop()
        blk[i] = stack[-1] if stack else -1

    def f(o):
        i = int(np.searchsorted(ev, o, 'left')) - 1  # last event strictly before o
        return int(blk[i]) if i >= 0 else -1
    return f


def zero_after_refs(k, so, sq, first, depth, regs, cond_rule=True):
    """Per sorted reference: is the wire |0> in every shot right after this touch (by reset)?

    Unconditional R/HMR always. With cond_rule, also a conditional reset (R/HMR inside a PUSH_CONDITION block, or
    R_IF/HMR_IF) when (1) the wire was |0> just before the innermost condition began (its previous touch before
    that point left it |0>, or it was never touched and is not a register) and (2) every touch of the wire between
    that point and this reset lies in the same innermost condition block. Then either the block runs, and the reset
    leaves |0>, or it does not run and none of those touches happened, so the wire is still |0>. (kickmix evaluates a
    block's condition once, at PUSH_CONDITION: sim.h `current_base_condition &= bits[...]`.) An R_IF/HMR_IF outside a
    block qualifies only when the wire was already |0> before it.
    """
    t = k.op_types
    rr = (t == K.R) | (t == K.HMR)
    za = (rr & (depth == 0))[so]
    if not cond_rule:
        return za
    is_cond = (rr & (depth > 0)) | (t == 0x92) | (t == 0xD0)
    cand = np.flatnonzero(is_cond[so])
    if len(cand) == 0:
        return za
    block = block_of(t)
    grp = np.maximum.accumulate(np.where(first, np.arange(len(first)), 0))
    regset = set(regs)
    for j in cand[np.argsort(so[cand], kind='stable')].tolist():
        o = int(so[j])
        if depth[o] > 0 and t[o] not in (K.R, K.HMR):
            continue  # an _IF reset inside a block may skip while earlier block touches ran
        P = block(o) if depth[o] > 0 else -1
        if depth[o] == 0:  # an _IF op outside any block: its own condition only
            P = o
        g0 = int(grp[j])
        k0 = g0 + int(np.searchsorted(so[g0:j], P, 'left'))
        if any(block(int(so[x])) != P for x in range(k0, j)) if P != o else False:
            continue
        prev = k0 - 1
        if prev < g0:
            ok = int(sq[j]) not in regset
        else:
            ok = bool(za[prev])
        if ok:
            za[j] = True
    return za


def build_intervals(k, ops, qs, order, is_reset, depth=None, regs=(), cond_rule=False):
    sq = qs[order]
    so = ops[order]
    n = len(sq)
    first = np.ones(n, bool)
    first[1:] = sq[1:] != sq[:-1]
    last = np.ones(n, bool)
    last[:-1] = sq[1:] != sq[:-1]
    if cond_rule:
        reset_ref = zero_after_refs(k, so, sq, first, depth, regs, True)
    else:
        reset_ref = is_reset[so]
    start = first.copy()
    start[1:] |= reset_ref[:-1] & ~first[1:]
    end = reset_ref | last
    interval_id = np.cumsum(start) - 1
    s_idx = np.flatnonzero(start)
    e_idx = np.flatnonzero(end)
    assert len(s_idx) == len(e_idx) and np.all(s_idx <= e_idx)
    return dict(sq=sq, so=so, first=first, last=last, reset_ref=reset_ref, start=start, end=end,
                interval_id=interval_id, logical=sq[s_idx].astype(np.int64), s=so[s_idx].astype(np.int64),
                e=so[e_idx].astype(np.int64), is_first=first[s_idx], is_last=last[e_idx])


def live_profile(iv, num_ops, regs, register_gaps=False):
    """Number of live wires just after each op boundary; intervals occupy [s, e] inclusive."""
    s, e = iv['s'].copy(), iv['e'].copy()
    reg = np.isin(iv['logical'], np.array(sorted(regs), dtype=np.int64))
    s[reg & iv['is_first']] = 0
    e[reg & iv['is_last']] = num_ops - 1
    if not register_gaps:
        # Registers live the whole circuit: drop their intervals and add a constant.
        keep = ~reg
        s, e = s[keep], e[keep]
        base = len(regs)
    else:
        base = 0
    delta = np.zeros(num_ops + 1, np.int64)
    np.add.at(delta, s, 1)
    np.add.at(delta, e + 1, -1)
    return np.cumsum(delta)[:num_ops] + base


def colour(iv, num_ops, regs, register_gaps=False):
    logical, s, e = iv['logical'], iv['s'].copy(), iv['e'].copy()
    is_first, is_last = iv['is_first'], iv['is_last']
    nI = len(logical)
    col = np.full(nI, -1, np.int64)
    reg_ids = sorted(regs)
    reg_set = set(reg_ids)
    isreg = np.array([int(q) in reg_set for q in logical.tolist()]) if nI else np.zeros(0, bool)
    # Pinned intervals.
    pinned = isreg & (is_first | is_last | (not register_gaps))
    s[isreg & is_first] = 0
    e[isreg & is_last] = num_ops - 1
    col[pinned] = logical[pinned]
    # Windows when a register colour is free (register_gaps): from end of its first interval to start of its last.
    windows = {}
    if register_gaps:
        for q in reg_ids:
            lo, hi = np.searchsorted(logical, q, 'left'), np.searchsorted(logical, q, 'right')
            if hi == lo:
                continue
            fi, la = lo, hi - 1
            if fi != la:
                windows[q] = (int(e[fi]) + 1, int(s[la]) - 1)  # usable for intervals within [lo, hi]
    free_ids = []  # heap of ids not reserved by registers
    next_new = 0
    reserved = set(reg_ids)
    busy = []  # heap (end, colour) for non-register colours
    win_busy_until = {q: -1 for q in windows}
    todo = np.flatnonzero(~pinned)
    todo = todo[np.lexsort((logical[todo], s[todo]))]
    for i in todo.tolist():
        si, ei = int(s[i]), int(e[i])
        while busy and busy[0][0] < si:
            heapq.heappush(free_ids, heapq.heappop(busy)[1])
        c = None
        # Prefer a register gap that fully contains the interval (lowest id first).
        for q, (lo, hi) in windows.items():
            if lo <= si and ei <= hi and win_busy_until[q] < si:
                c = q
                win_busy_until[q] = ei
                break
        if c is None:
            if free_ids:
                c = heapq.heappop(free_ids)
            else:
                while next_new in reserved:
                    next_new += 1
                c = next_new
                next_new += 1
            heapq.heappush(busy, (ei, c))
        col[i] = c
    assert (col >= 0).all()
    return col


def recolour(k, ops, qs, order, iv, col):
    new_sorted = col[iv['interval_id']].astype(np.uint32)
    new_qs = np.empty_like(qs)
    new_qs[order] = new_sorted
    lens = [k.counts['qqq']] * 3 + [k.counts['qq']] * 2 + [k.counts['q']]
    out, pos = {}, 0
    for name, n in zip(QSTREAMS, lens):
        out[name] = new_qs[pos:pos + n]
        pos += n
    return out, new_qs


def _ksort(major, minor):
    """argsort by (major, minor); both < 2**32."""
    return np.argsort((major.astype(np.uint64) << np.uint64(32)) | minor.astype(np.uint64), kind='stable')


def check_equivalence(in_path, out_path):
    """Independent check that OUT = IN under a lifetime-respecting qubit renaming. Returns a dict; raises on failure."""
    a, b = K.read(in_path), K.read(out_path)
    # 1. Every non-qubit byte identical.
    for name in ['op_types', 'b0', 'bc', 'angles', 'info_tags']:
        oa, na, da = a.offsets[name]
        ob, nb, db = b.offsets[name]
        assert oa == ob and na == nb
    qa = [a.offsets[n] for n in QSTREAMS]
    mask = np.ones(len(a.raw), bool)
    mask[20:24] = False  # header num_qubits
    for off, n, _ in qa:
        mask[off:off + 4 * n] = False
    for _, contents in a.registers:
        for _, _, pos in contents:
            mask[pos:pos + 4] = False
    assert len(a.raw) == len(b.raw)
    assert np.array_equal(np.asarray(a.raw)[mask], np.asarray(b.raw)[mask]), 'non-qubit bytes differ'
    # Register structure: same names, kinds; map logical -> physical register id.
    regmap = {}
    for (na_, ca), (nb_, cb) in zip(a.registers, b.registers):
        assert na_ == nb_ and len(ca) == len(cb)
        for (qa_, ia, _), (qb_, ib, _) in zip(ca, cb):
            assert qa_ == qb_
            if qa_:
                regmap[ia] = ib
            else:
                assert ia == ib
    ops, ql, _ = K.op_qubits(a)
    _, qp, _ = K.op_qubits(b)
    is_reset, _ = reset_ops(a)
    nops = len(a.op_types)
    # Per-op injectivity on physical ids (kickmix also enforces this when loading).
    o2 = _ksort(ops, qp)
    dup = (ops[o2][1:] == ops[o2][:-1]) & (qp[o2][1:] == qp[o2][:-1])
    assert not dup.any(), 'two operands of one op share a physical wire'
    # 2. Logical view: a logical keeps its wire between consecutive touches unless a reset intervened.
    ol = _ksort(ql, ops)
    L, P, O = ql[ol], qp[ol], ops[ol]
    same = L[1:] == L[:-1]
    prev_reset = is_reset[O[:-1]]
    must_stay = same & ~prev_reset
    assert np.array_equal(P[1:][must_stay], P[:-1][must_stay]), 'logical moved wire while live'
    firstL = np.ones(len(L), bool)
    firstL[1:] = ~same
    lastL = np.ones(len(L), bool)
    lastL[:-1] = ~same
    regarr = np.zeros(int(max(L.max(), max(regmap) if regmap else 0)) + 1, bool)
    for q in regmap:
        regarr[q] = True
    fresh = (firstL & ~regarr[L]) | np.concatenate([[False], same & prev_reset])  # value |0> at this touch
    dead_after = is_reset[O] | (lastL & ~regarr[L])  # wire |0> after this touch
    # Register endpoints sit on the register's declared physical wire.
    for q, p in regmap.items():
        lo, hi = np.searchsorted(L, q, 'left'), np.searchsorted(L, q, 'right')
        if hi > lo:
            assert P[lo] == p and P[hi - 1] == p, ('register endpoint moved', q)
    # 3. Physical view: ownership changes only from a dead value to a fresh one.
    fresh_r = np.empty_like(fresh)
    fresh_r[ol] = fresh
    dead_r = np.empty_like(dead_after)
    dead_r[ol] = dead_after
    op_ = _ksort(qp, ops)
    PP, LL = qp[op_], ql[op_]
    sameP = PP[1:] == PP[:-1]
    switch = sameP & (LL[1:] != LL[:-1])
    assert dead_r[op_][:-1][switch].all(), 'wire reused before previous value was reset/dead'
    assert fresh_r[op_][1:][switch].all(), 'value moved onto a wire without being fresh'
    # First touch of each physical wire: either a fresh value, or a register at its own declared wire.
    firstP = np.ones(len(PP), bool)
    firstP[1:] = ~sameP
    fp_ok = fresh_r[op_][firstP] | np.array([regmap.get(int(l), -1) == int(p) for l, p in zip(LL[firstP], PP[firstP])])
    assert fp_ok.all(), 'first use of a wire is neither fresh nor its register'
    # Register wires: nobody else touches a register's physical wire before its first or after its last touch.
    for q, p in regmap.items():
        lo, hi = np.searchsorted(PP, p, 'left'), np.searchsorted(PP, p, 'right')
        if hi > lo:
            assert LL[lo] == q and LL[hi - 1] == q, ('register wire shared at an endpoint', q)
    return dict(status='EQUIVALENT', num_qubits_in=a.num_qubits, num_qubits_out=b.num_qubits, num_ops=int(nops),
                num_refs=int(len(ops)), wire_switches=int(switch.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--register-gaps', action='store_true')
    ap.add_argument('--report')
    ap.add_argument('--profile-out', help='write per-op live width (npy, int16)')
    ap.add_argument('--backend', choices=['python', 'rust'], default='python')
    a = ap.parse_args()
    t0 = time.time()
    k = K.read(a.input)
    regs = register_qubits(k)
    nops = len(k.op_types)
    if a.backend == 'rust':
        import kickmix_analysis as KA  # analysis venv only
        lv = KA.liveness(a.input, register_gaps=a.register_gaps)
        iv, prof = lv['intervals'], lv['profile']
        touched = np.flatnonzero(lv['wire_last'] >= 0)
        n_used, max_id1, n_refs = len(touched), int(touched.max()) + 1, lv['num_refs']
        n_uncond, n_cond = lv['unconditional_resets'], lv['conditional_resets']
    else:
        is_reset, depth = reset_ops(k)
        ops, qs, order, _ = references(k)
        iv = build_intervals(k, ops, qs, order, is_reset)
        prof = live_profile(iv, nops, regs, a.register_gaps)
        used = np.unique(qs)
        n_used, max_id1, n_refs = len(used), int(qs.max()) + 1, len(ops)
        n_uncond = is_reset.sum()
        n_cond = ((((k.op_types == K.R) | (k.op_types == K.HMR)) & (depth > 0)).sum()
                  + ((k.op_types == 0x92) | (k.op_types == 0xD0)).sum())
    lb = int(prof.max())
    reg_ids = np.array(sorted(regs), dtype=np.int64)
    rep = dict(input=a.input, input_sha256=sha256(a.input), num_qubits_in=int(k.num_qubits),
               used_ids=int(n_used), max_id_plus_1=int(max_id1),
               registers={n: len([c for c in cs if c[0]]) for n, cs in k.registers},
               register_id_range=[int(reg_ids.min()), int(reg_ids.max())] if len(reg_ids) else None,
               num_ops=int(nops), num_refs=int(n_refs), num_intervals=int(len(iv['s'])),
               unconditional_resets=int(n_uncond),
               conditional_resets=int(n_cond),
               register_qubits_with_resets=int(len(set(iv['logical'][~iv['is_first']].tolist()) & set(regs))),
               register_gaps=a.register_gaps, lower_bound=lb,
               peak_ops_first=[int(x) for x in np.flatnonzero(prof == lb)[:20]],
               peak_op_count=int((prof == lb).sum()))
    if a.profile_out:
        np.save(a.profile_out, prof.astype(np.int16))
    if a.output:
        col = colour(iv, nops, regs, a.register_gaps)
        if a.backend == 'rust':
            streams = KA.recolour(a.input, iv['logical'], col)
            nz = [int(v.max()) for v in streams.values() if len(v)]
            newQ = max(nz) + 1 if nz else 0
        else:
            streams, new_qs = recolour(k, ops, qs, order, iv, col)
            newQ = int(new_qs.max()) + 1 if len(new_qs) else 0
        regmax = int(reg_ids.max()) + 1 if len(reg_ids) else 0
        newQ = max(newQ, regmax)
        K.patch_qubits(k, a.output, streams, {}, newQ)
        rep.update(output=a.output, output_sha256=sha256(a.output), num_qubits_out=newQ,
                   optimal=bool(newQ == lb))
        rep['equivalence'] = check_equivalence(a.input, a.output)
    rep['seconds'] = round(time.time() - t0, 1)
    text = json.dumps(rep, indent=2)
    if a.report:
        Path(a.report).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
