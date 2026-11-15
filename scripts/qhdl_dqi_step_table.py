#!/usr/bin/env python3
"""DQI step table: one row per INFO_TAG region instance.

Usage: python qhdl_dqi_step_table.py CIRCUIT.kmb OUT_PREFIX [--shots 1024]
Writes OUT_PREFIX.regions.csv (every region instance) and OUT_PREFIX.summary.csv (instances merged by tag path).

Live width uses the lifetime intervals of qhdl_lifetime_allocator with the conditional-reset rule and register gaps
(a syndrome qubit counts as live only inside its own intervals). Static Toffolis count CCX/CCZ/CSWAP (and _IF forms)
in the op range. Mean executed Toffolis are measured by segmented kickmix simulation on `--shots` inputs drawn with
the official verifier's sampler (seed fixed), cutting at every INFO_TAG op at condition depth 0.
"""
import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402
import qhdl_lifetime_allocator as A  # noqa: E402


def regions(k):
    t = k.op_types
    tag_ops = np.flatnonzero(t == K.INFO_TAG)
    tags = k.info_tags()
    stack, out = [], []
    counters = {}
    for o, txt in zip(tag_ops.tolist(), tags):
        kind, _, name = txt.partition(':')
        if kind == '@enter':
            parent = stack[-1]['path'] if stack else ''
            path = f'{parent}/{name}' if parent else name
            counters[path] = counters.get(path, 0) + 1
            stack.append(dict(path=path, name=name, enter=o, depth=len(stack), instance=counters[path]))
        elif kind == '@exit':
            r = stack.pop()
            assert r['name'] == name, (o, txt, r)
            r['exit'] = o
            out.append(r)
    assert not stack
    out.sort(key=lambda r: (r['enter'], r['depth']))
    return out, tag_ops


def executed_toffolis(path, k, cut_ops, shots, seed=2026):
    """Mean executed Toffoli-like ops per shot for each segment between consecutive cuts."""
    import kickmix as km
    from qhdl_dqi_probe import load_verifier, draw_inputs
    V = load_verifier()
    circ = km.Circuit.from_file(path)
    spec, (syn, mbuc, _) = draw_inputs(V, shots, seed)
    regs = circ.register_data
    sq = km.array([q for n in [f'S{i}' for i in range(spec.n)] for q in regs[n]])
    eb = km.array([b for n in [f'E{j}' for j in range(spec.m)] for b in regs[n]])
    sim = km.Simulator(batch_size=shots, debug_print=False, counting_operations=True, seed=11)
    sim.use_same_registers_as(circ)
    sim.write_across_shots(sq, V.to_bits(syn, spec.degree))
    sim.write_across_shots(eb, V.to_bits(mbuc, spec.degree))
    sl = K.Slicer(k)
    bounds = [0] + list(cut_ops) + [len(k.op_types)]
    seg = np.zeros(len(bounds) - 1)
    tmp = tempfile.mkdtemp(prefix='steptable')
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        p = os.path.join(tmp, 'seg.kmb')
        sl.write(p, a, b)
        sim.clear_operation_statistics()
        sim.do(km.Circuit.from_file(p))
        seg[i] = sim.operation_statistics().num_toffoli_like / shots
    os.remove(os.path.join(tmp, 'seg.kmb'))
    return np.array(bounds), seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('out_prefix')
    ap.add_argument('--shots', type=int, default=1024)
    a = ap.parse_args()
    t0 = time.time()
    k = K.read(a.circuit)
    t = k.op_types
    is_reset, depth = A.reset_ops(k)
    ops, qs, order, _ = A.references(k)
    regs = A.register_qubits(k)
    iv = A.build_intervals(k, ops, qs, order, is_reset, depth, regs, cond_rule=True)
    n = len(t)
    prof = A.live_profile(iv, n, regs, register_gaps=True)
    isreg = np.isin(iv['logical'], np.array(sorted(regs), np.int64))
    s, e = iv['s'].copy(), iv['e'].copy()
    s[isreg & iv['is_first']] = 0
    e[isreg & iv['is_last']] = n - 1
    rs, re_, rq = s[isreg], e[isreg], iv['logical'][isreg]
    qname = {}
    for name, contents in k.registers:
        for is_q, qid, _ in contents:
            if is_q:
                qname[qid] = name
    toff = np.isin(t, np.array(K.TOFFOLI_LIKE, np.uint8))
    ctoff = np.concatenate([[0], np.cumsum(toff)])
    regs_list, tag_ops = regions(k)
    print('regions', len(regs_list), 'profile ready', round(time.time() - t0, 1), flush=True)
    # executed Toffolis per segment, cutting at tag ops (depth 0)
    cuts = [int(o) for o in tag_ops.tolist() if depth[o] == 0 and (o == 0 or t[o - 1] != K.PUSH_CONDITION)]
    bounds, seg = executed_toffolis(a.circuit, k, cuts, a.shots)
    cseg = np.concatenate([[0], np.cumsum(seg)])
    print('executed counts ready', round(time.time() - t0, 1), flush=True)

    pos = {int(b): i for i, b in enumerate(bounds.tolist())}

    def executed(lo, hi):  # ops [lo, hi]; the exit tag itself carries no Toffolis
        if lo in pos and hi in pos:
            return float(cseg[pos[hi]] - cseg[pos[lo]])
        return float('nan')

    rows = []
    for r in regs_list:
        lo, hi = r['enter'], r['exit']
        seg_p = prof[lo:hi + 1]
        pk = lo + int(np.argmax(seg_p))
        live_reg = (rs <= pk) & (re_ >= pk)
        live_names = {}
        for q in rq[live_reg].tolist():
            live_names[qname[q]] = live_names.get(qname[q], 0) + 1
        dead = [nm for nm in sorted(set(qname.values()), key=lambda x: int(x[1:])) if nm not in live_names]
        partial = [f'{nm}:{c}' for nm, c in sorted(live_names.items(), key=lambda kv: int(kv[0][1:])) if c < 12]
        rows.append(dict(path=r['path'], name=r['name'], depth=r['depth'], instance=r['instance'], enter_op=lo,
                         exit_op=hi, ops=hi - lo + 1, static_toffolis=int(ctoff[hi + 1] - ctoff[lo]),
                         executed_toffolis_mean=round(executed(lo, hi), 3), live_min=int(seg_p.min()),
                         live_max=int(seg_p.max()), live_entry=int(prof[lo]), live_exit=int(prof[hi]), peak_op=pk,
                         syndrome_qubits_live_at_peak=int(live_reg.sum()), ancillas_live_at_peak=int(prof[pk] - live_reg.sum()),
                         registers_dead_at_peak=' '.join(dead), registers_partial_at_peak=' '.join(partial)))
    with open(a.out_prefix + '.regions.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for r in rows:
        d = agg.setdefault(r['path'], dict(path=r['path'], depth=r['depth'], instances=0, ops=0, static_toffolis=0,
                                           executed_toffolis_mean=0.0, live_min=10 ** 9, live_max=0,
                                           first_enter=r['enter_op'], last_exit=0, max_ancillas=0))
        d['instances'] += 1
        d['ops'] += r['ops']
        d['static_toffolis'] += r['static_toffolis']
        d['executed_toffolis_mean'] += r['executed_toffolis_mean']
        d['live_min'] = min(d['live_min'], r['live_min'])
        d['live_max'] = max(d['live_max'], r['live_max'])
        d['last_exit'] = max(d['last_exit'], r['exit_op'])
        d['max_ancillas'] = max(d['max_ancillas'], r['ancillas_live_at_peak'])
    summ = sorted(agg.values(), key=lambda d: d['first_enter'])
    for d in summ:
        d['executed_toffolis_mean'] = round(d['executed_toffolis_mean'], 2)
    with open(a.out_prefix + '.summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    tot = dict(circuit=a.circuit, sha256=A.sha256(a.circuit), num_qubits=k.num_qubits, peak=int(prof.max()),
               static_toffolis=int(toff.sum()), executed_toffolis_mean=float(seg.sum()), shots=a.shots,
               regions=len(rows), seconds=round(time.time() - t0, 1))
    Path(a.out_prefix + '.totals.json').write_text(__import__('json').dumps(tot, indent=1) + '\n')
    print(tot)


if __name__ == '__main__':
    main()
