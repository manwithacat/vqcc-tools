#!/usr/bin/env python3
"""Live-width profile per INFO_TAG region and anatomy of the peak.

Usage: python qhdl_dqi_peak_anatomy.py CIRCUIT.kmb OUT.json
Uses the same interval definition as qhdl_lifetime_allocator.py (registers live for the whole circuit).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402
import qhdl_lifetime_allocator as A  # noqa: E402


def main():
    src, out = sys.argv[1], sys.argv[2]
    k = K.read(src)
    t = k.op_types
    nops = len(t)
    is_reset, _ = A.reset_ops(k)
    ops, qs, order, _ = A.references(k)
    iv = A.build_intervals(k, ops, qs, order, is_reset)
    regs = A.register_qubits(k)
    prof = A.live_profile(iv, nops, regs)
    peak = int(prof.max())

    # Region of each op: innermost tag path, with occurrence counters for the outer phases.
    tag_ops = np.flatnonzero(t == K.INFO_TAG)
    tags = k.info_tags()
    assert len(tag_ops) == len(tags)
    stack, bounds = [], []  # bounds: (op index where a path begins, path)
    outer_count = collections.Counter()
    for o, txt in zip(tag_ops.tolist(), tags):
        kind, _, name = txt.partition(':')
        if kind == '@enter':
            if len(stack) == 0:
                outer_count[name] += 1
                stack.append(f'{name}#{outer_count[name]}')
            elif len(stack) == 1:
                outer_count[(stack[0], name)] += 1
                stack.append(f'{name}#{outer_count[(stack[0], name)]}')
            else:
                stack.append(name)
        elif kind == '@exit':
            assert stack and stack[-1].split('#')[0] == name, (o, txt, stack)
            stack.pop()
        bounds.append((o, '/'.join(stack) or '(untagged)'))
    starts = np.array([0] + [b[0] for b in bounds], np.int64)
    paths = ['(untagged)'] + [b[1] for b in bounds]

    def path_of(op):
        return paths[int(np.searchsorted(starts, op, 'right')) - 1]

    # Per leaf-region-name statistics (instances merged; occurrence numbers dropped below the outer level).
    seg_max = np.maximum.reduceat(prof, starts) if len(starts) else np.array([])
    seg_len = np.diff(np.append(starts, nops))
    by_name = collections.defaultdict(lambda: dict(max_live=0, ops=0, instances=0, ops_at_peak=0))
    at_peak = prof == peak
    peak_counts = np.add.reduceat(at_peak.astype(np.int64), starts)
    for p, m, n, pk in zip(paths, seg_max.tolist(), seg_len.tolist(), peak_counts.tolist()):
        if n == 0:
            continue
        key = '/'.join(x.split('#')[0] if i else x.split('#')[0] for i, x in enumerate(p.split('/')))
        d = by_name[key]
        d['max_live'] = max(d['max_live'], int(m))
        d['ops'] += int(n)
        d['instances'] += 1
        d['ops_at_peak'] += int(pk)
    # Outer-phase profile.
    outer = collections.defaultdict(lambda: dict(max_live=0, first_op=None, last_op=None))
    for p, st, n, m in zip(paths, starts.tolist(), seg_len.tolist(), seg_max.tolist()):
        if n == 0:
            continue
        o = p.split('/')[0]
        d = outer[o]
        d['max_live'] = max(d['max_live'], int(m))
        d['first_op'] = st if d['first_op'] is None else d['first_op']
        d['last_op'] = st + n - 1

    # Peak anatomy: intervals live at the first peak op.
    peak_ops = np.flatnonzero(at_peak)
    p0 = int(peak_ops[0])
    runs = np.split(peak_ops, np.flatnonzero(np.diff(peak_ops) > 1) + 1)
    reg_arr = np.isin(iv['logical'], np.array(sorted(regs), np.int64))
    live = (~reg_arr) & (iv['s'] <= p0) & (iv['e'] >= p0)
    live_idx = np.flatnonzero(live)
    anc = []
    for i in live_idx.tolist():
        anc.append(dict(logical=int(iv['logical'][i]), start=int(iv['s'][i]), end=int(iv['e'][i]),
                        length=int(iv['e'][i] - iv['s'][i] + 1), start_region=path_of(int(iv['s'][i])),
                        end_region=path_of(int(iv['e'][i])), ends_by='reset' if is_reset[int(iv['e'][i])] else 'last-touch'))
    anc_by_start = collections.Counter(a['start_region'].split('/')[0] + ' / ' + a['start_region'].split('/')[-1].split('#')[0]
                                       for a in anc)
    # Long-lived ancillas (alive across most of the circuit) versus local temporaries.
    long_lived = [a for a in anc if a['length'] > nops // 4]
    rep = dict(circuit=src, input_sha256=A.sha256(src), num_qubits=int(k.num_qubits), lifetime_peak=peak,
               registers=len(regs), ancilla_live_at_peak=len(anc), num_ops=int(nops), num_tags=len(tags),
               tag_kinds=collections.Counter(':'.join(x.split(':')[:2]) for x in tags).most_common(40),
               peak_ops_total=int(len(peak_ops)), peak_runs=len(runs),
               peak_runs_first=[[int(r[0]), int(r[-1]), path_of(int(r[0]))] for r in runs[:40]],
               peak_runs_by_region=collections.Counter(path_of(int(r[0])).split('/')[0].split('#')[0] + ' / ' +
                                                       path_of(int(r[0])).split('/')[-1].split('#')[0] for r in runs).most_common(),
               first_peak_op=p0, first_peak_region=path_of(p0),
               ancilla_at_first_peak_by_start_region=anc_by_start.most_common(),
               long_lived_ancillas=len(long_lived),
               long_lived_examples=long_lived[:10],
               ancillas_at_first_peak=sorted(anc, key=lambda a: a['start'])[:120],
               outer_phases=dict(outer), regions=dict(sorted(by_name.items(), key=lambda kv: -kv[1]['max_live'])[:60]))
    Path(out).write_text(json.dumps(rep, indent=1, default=str) + '\n')
    print(json.dumps({x: rep[x] for x in ['num_qubits', 'lifetime_peak', 'ancilla_live_at_peak', 'peak_ops_total',
                                           'peak_runs', 'first_peak_region', 'long_lived_ancillas',
                                           'ancilla_at_first_peak_by_start_region', 'peak_runs_by_region']}, indent=1,
                     default=str))
    print(json.dumps(rep['outer_phases'], indent=1))


if __name__ == '__main__':
    main()
