#!/usr/bin/env python3
"""Fingerprint a kickmix .kmb circuit.

For one circuit, writes a JSON fingerprint with:
  * header: sha256, num_qubits, num_ops, op-type histogram, static Toffoli-like count;
  * phases: the top-level @enter/@exit regions (op range, static T, CSWAP count, CSWAP bursts as runs of consecutive
    CSWAP ops with their width histogram, R/CCX ladders of >= 8 alternating pairs, max live width if --profile);
  * tag multiset: count and static T (inclusive) per tag path (nested @enter/@exit names joined by '/');
  * optional source table (--srcmap from qhdl_source_provenance_build.py): per (phase, source key) op count, static T,
    CSWAP count, max live width and the op range, so every netlist feature can be pinned to a source line.

Usage: python qhdl_kmb_fingerprint.py CIRCUIT.kmb OUT.json [--profile WIDTH.npy] [--srcmap OUT.srcmap.npz]
           [--srcmap-json OUT.srcmap.json] [--phases unextract,extract]
Compare two fingerprints with --diff A.json B.json (prints tag paths whose count or T differ, and phase deltas).
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402

CSWAP_T = (0x32, 0xB2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def runs(idx, gap=1):
    if len(idx) == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > gap)
    st = np.r_[0, br + 1]
    en = np.r_[br, len(idx) - 1]
    return [(int(idx[s]), int(e - s + 1)) for s, e in zip(st, en)]


def walk_tags(k):
    t = k.op_types
    tag_ops = np.flatnonzero(t == K.INFO_TAG)
    tags = k.info_tags()
    stack, regions = [], []
    for o, s in zip(tag_ops.tolist(), tags):
        kind, _, name = s.partition(':')
        if kind == '@enter':
            stack.append((name, o))
        elif kind == '@exit' and stack and stack[-1][0] == name:
            path = '/'.join(n for n, _ in stack)
            regions.append((path, stack[-1][1], o, len(stack) - 1))
            stack.pop()
    return regions


def fingerprint(path, profile=None, srcmap=None, srckeys=None, phases=None):
    k = K.read(path)
    t = k.op_types
    is_t = np.isin(t, K.TOFFOLI_LIKE)
    is_cs = np.isin(t, CSWAP_T)
    cT = np.r_[0, np.cumsum(is_t)]
    cC = np.r_[0, np.cumsum(is_cs)]
    vals, cnt = np.unique(t, return_counts=True)
    fp = dict(path=str(path), sha256=sha256(path), num_qubits=int(k.num_qubits), num_ops=int(len(t)),
              static_toffoli=int(is_t.sum()), cswap=int(is_cs.sum()),
              op_hist={hex(int(v)): int(c) for v, c in zip(vals, cnt)})
    regions = walk_tags(k)
    multiset = collections.defaultdict(lambda: [0, 0, 0])
    for p, a, b, d in regions:
        m = multiset[p]
        m[0] += 1
        m[1] += int(cT[b + 1] - cT[a])
        m[2] += int(cC[b + 1] - cC[a])
    fp['tags'] = {p: dict(count=c, static_T=tt, cswap=cs) for p, (c, tt, cs) in sorted(multiset.items())}
    pair = np.flatnonzero((t[:-1] == K.R) & (t[1:] == 0x31))
    ph = []
    for p, a, b, d in regions:
        if d != 0:
            continue
        cs_idx = np.flatnonzero(is_cs[a:b + 1]) + a
        bursts = runs(cs_idx)
        lad = [(o, n) for o, n in runs(pair[(pair >= a) & (pair <= b)], gap=2) if n >= 8]
        row = dict(name=p, ops=[int(a), int(b)], static_T=int(cT[b + 1] - cT[a]), cswap=int(len(cs_idx)),
                   cswap_bursts=len(bursts),
                   cswap_burst_widths=dict(collections.Counter(str(n) for _, n in bursts)),
                   ladders=len(lad), ladder_lengths=dict(collections.Counter(str(n) for _, n in lad)))
        if profile is not None:
            seg = profile[a:b + 1]
            row['max_width'] = int(seg.max())
            row['ops_at_max'] = int((seg == seg.max()).sum())
        ph.append(row)
    fp['phases'] = ph
    if srcmap is not None:
        want = set(phases) if phases else None
        table = []
        for p, a, b, d in regions:
            if d != 0 or (want and p not in want):
                continue
            keys = srcmap[a:b + 1]
            order = np.argsort(keys, kind='stable')
            ks = keys[order]
            bounds = np.flatnonzero(np.diff(ks)) + 1
            for grp in np.split(order, bounds):
                if len(grp) == 0:
                    continue
                key = int(keys[grp[0]])
                opsg = grp + a
                row = dict(phase=p, key=srckeys[key] if key >= 0 else '-', ops=int(len(opsg)),
                           first=int(opsg.min()), last=int(opsg.max()), static_T=int(is_t[opsg].sum()),
                           cswap=int(is_cs[opsg].sum()))
                if profile is not None:
                    row['max_width'] = int(profile[opsg].max())
                    row['ops_at_max_overall'] = int((profile[opsg] == profile.max()).sum())
                table.append(row)
        table.sort(key=lambda r: (r['phase'], r['first']))
        fp['source_table'] = table
    return fp


def diff(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    print(f"A {A['sha256'][:12]} Q{A['num_qubits']} T{A['static_toffoli']} ops{A['num_ops']}  |  "
          f"B {B['sha256'][:12]} Q{B['num_qubits']} T{B['static_toffoli']} ops{B['num_ops']}")
    for p in sorted(set(A['tags']) | set(B['tags'])):
        x, y = A['tags'].get(p), B['tags'].get(p)
        if x != y:
            print(f'  tag {p}: A {x}  B {y}')
    for pa, pb in zip(A['phases'], B['phases']):
        d = {k: (pa.get(k), pb.get(k)) for k in pa if k != 'ops' and pa.get(k) != pb.get(k)}
        if d:
            print(f"  phase {pa['name']}: {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit', nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--profile')
    ap.add_argument('--srcmap')
    ap.add_argument('--srcmap-json')
    ap.add_argument('--phases')
    ap.add_argument('--diff', nargs=2)
    a = ap.parse_args()
    if a.diff:
        diff(*a.diff)
        return
    prof = np.load(a.profile, mmap_mode='r') if a.profile else None
    sm = np.load(a.srcmap)['plain_key'] if a.srcmap else None
    keys = json.load(open(a.srcmap_json))['keys'] if a.srcmap_json else None
    fp = fingerprint(a.circuit, None if prof is None else np.asarray(prof), sm, keys,
                     a.phases.split(',') if a.phases else None)
    Path(a.out).write_text(json.dumps(fp, indent=1) + '\n')
    for r in fp['phases']:
        print(f"{r['name']:20s} T {r['static_T']:7d} cswap {r['cswap']:5d} bursts {r['cswap_bursts']:4d} "
              f"{r['cswap_burst_widths']} ladders {r['ladders']} max_w {r.get('max_width')}")


if __name__ == '__main__':
    main()
