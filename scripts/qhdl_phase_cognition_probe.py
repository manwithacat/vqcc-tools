"""Independent scalar check of the fixed QHDL phase-cognition pilot exports.

This consumes TSV gate operands and JSON ANFs, not the symbolic scanner.
No candidate is constructed and no competition verifier is run.
"""
import argparse
import collections
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time


def rank(rows):
    basis = {}
    for values in rows:
        row = list(map(Fraction, values))
        for pivot, old in sorted(basis.items()):
            if row[pivot]:
                scale = row[pivot]
                row = [a - scale*b for a, b in zip(row, old)]
        pivot = next((i for i, v in enumerate(row) if v), None)
        if pivot is not None:
            scale = row[pivot]
            basis[pivot] = [v / scale for v in row]
    return len(basis)


def kind(angle):
    angle = int(angle, 16)
    return 'clifford' if angle % (1 << 126) == 0 else 't' if angle % (1 << 125) == 0 else 'general'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    start = time.monotonic()
    full_path, = a.workspace.glob('phase/profile_0_47258_*.json')
    small_path, = a.workspace.glob('phase/profile_15244_15254_*.json')
    full, small = [json.loads(p.read_text()) for p in (full_path, small_path)]
    with (a.workspace / 'ops.tsv').open() as f:
        ops = list(csv.DictReader(f, delimiter='\t'))
    groups = collections.defaultdict(list)
    for p in full['predicates']:
        groups[p['region']].append(p)
    region_counts = [dict(region=r, **dict(collections.Counter(kind(p['angle128']) for p in ps)))
                     for r, ps in groups.items()]
    charged = sorted(op for p in full['predicates'] if kind(p['angle128']) == 'general' for op in p['ops'])
    # Actual non-unitary boundaries, independently of ANF capacity cuts.
    gaps = []
    for left, right in zip(charged, charged[1:]):
        measured = [int(o['index']) for o in ops[left+1:right] if o['kind'] == 'HMR']
        gaps.append(dict(left=left, right=right, hmr_count=len(measured), first_hmr=measured[0] if measured else None))
    selected = ops[15244:15254]
    wires = sorted({int(w[1:]) for o in selected for w in o['operands'].split(',') if w})
    assert len(wires) <= 16
    predicates = {p['ops'][0]: [int(m, 16) for m in p['predicate']] for p in small['predicates']}
    truth_rows = set()
    mismatch_count = 0
    mux_mismatches = 0
    # All entry wires arbitrary, including scratch. No reachability assumptions.
    for assignment in range(1 << len(wires)):
        state = {w: (assignment >> i) & 1 for i, w in enumerate(wires)}
        entry = dict(state)
        entry_mask = sum(v << w for w, v in state.items())
        values = []
        for op in selected:
            q = [int(w[1:]) for w in op['operands'].split(',')]
            code = op['kind']
            if code == 'Z_POW':
                value = state[q[0]]
                anf = sum((entry_mask & m) == m for m in predicates[int(op['index'])]) % 2
                mismatch_count += value != anf
                values.append(value)
            elif code == 'CX':
                state[q[1]] ^= state[q[0]]
            elif code == 'CSWAP':
                if state[q[0]]:
                    state[q[1]], state[q[2]] = state[q[2]], state[q[1]]
            else:
                raise ValueError(code)
        u = entry[28] ^ (entry[1] if entry[21] ^ entry[23] else entry[50])
        v = entry[2] if entry[20] else u
        mux = entry[0] if entry[27] else v
        mux_mismatches += mux != values[-1]
        truth_rows.add(tuple([1] + values))
    affine_rank = rank(sorted(truth_rows))
    assert mismatch_count == 0 and mux_mismatches == 0
    assert affine_rank == 1 + len(predicates)
    # Check that the sole other multi-predicate group consists of distinct entry bits.
    other_multi = []
    for region, ps in groups.items():
        if len(ps) > 1 and region != 634:
            masks = [[int(m, 16) for m in p['predicate']] for p in ps]
            independent_bits = all(len(ms) == 1 and ms[0] > 0 and ms[0] & (ms[0]-1) == 0 for ms in masks)
            independent_bits &= len({tuple(ms) for ms in masks}) == len(masks)
            other_multi.append(dict(region=region, independent_entry_bits=independent_bits))
    result = dict(circuit_sha=full['circuit_sha'], scope='static model / projection; exact scalar local semantics, no verifier',
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (full_path, small_path, a.workspace/'ops.tsv')},
                  phase_kinds=dict(collections.Counter(kind(p['angle128']) for p in full['predicates'])),
                  regions=region_counts, consecutive_general_rotation_gaps=gaps,
                  all_general_pairs_separated_by_hmr=all(g['hmr_count'] > 0 for g in gaps),
                  selected_ops=[15244,15254], selected_wires=wires, exhaustive_entry_states=1 << len(wires),
                  predicate_mismatches=mismatch_count, mux_mismatches=mux_mismatches,
                  affine_rank_including_constant=affine_rank, truth_patterns=[list(r) for r in sorted(truth_rows)],
                  other_multi_predicate_regions=other_multi,
                  limitations=['conditional phases excluded by profiler', 'cross-measurement identities not tested',
                               'no approximate synthesis or arbitrary replacement predicate search',
                               'TSV operands share QHDL byte decoder; scalar semantics independent'],
                  seconds=time.monotonic()-start)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('phase_kinds','all_general_pairs_separated_by_hmr','selected_wires','exhaustive_entry_states','predicate_mismatches','mux_mismatches','affine_rank_including_constant','seconds')}))


if __name__ == '__main__':
    main()
