#!/usr/bin/env python3
"""qhdl diff: what changed between a base circuit and a new public file of the same task (avenues generator G6).

    qhdl diff -c <base> -c2 <new> [--task dqi|invsqrt|chemistry] [--top N] [--all] [--out DIR] [--deep]

One bounded JSON envelope per pair: a ranked list of changed components with multiplicity, per-instance delta and a
technique hint from a fixed vocabulary (never free text). The full rows go to the evidence file (--out, default
work/_diff/<base>__<new>/diff.json). Nothing is emitted, nothing is scored: every number is `static` (netlist counts,
not per-shot means) and the verb says so. Cost attribution: the per-component deltas are summed and compared with the
whole-file delta; the remainder is reported as `unmatched`, never hidden.

  dqi        58M ops, no op table: the region tree from `workbench summarize` (cached in work/_diff/regions) gives static
             Toffolis per top-level tag path. Components are the top-level paths (convert, extract, unextract, ...);
             nested paths (e.g. `extract/inverse24 scalar`, 23 instances) are drill-down rows and are NOT added to the sum.
             Each changed component is then aligned op by op (numpy hashes of type, operands, bits and angles) with a
             local-resync diff; the aligned gaps are its call sites: multiplicity = number of gaps, and the dominant gap
             signature (Toffolis in the old gap -> in the new gap) is the per-site change.
  invsqrt    the whole circuit is aligned the same way; changed op ranges are given in both files with static
             Toffoli-like gates and Z_POW counts. A changed range is named a splice when one other public invsqrt file
             contains >= 90% of its K-grams (credit from the manifest).
  chemistry  -c / -c2 are family sha prefixes of the stage's parts/families.csv: per-slot loader Toffolis per pass,
             table key, per-slot word hashes, NPZ (factorization) SHA, lambda_eff, Q and modelled non-angle Toffolis.

Technique vocabulary: component-resynthesis, repeated-component, splice, weights-refit, table-change, width-change,
unknown. Tier: static for every number.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_envelope as E  # noqa: E402

TECHNIQUES = ('component-resynthesis', 'repeated-component', 'splice', 'weights-refit', 'table-change',
              'width-change', 'unknown')
K_GRAM = 16            # ops in a resync anchor
MERGE_RUN = 64         # matched runs shorter than this between two gaps join them (one call site)
MAX_GAPS = 400_000
import vqcc_config as CFG  # noqa: E402
WORKBENCH = CFG.WORKBENCH
SPLICE_COVER = 0.90
REPEAT_SITES = 8
REPEAT_SHARE = 0.80


# ---------------------------------------------------------------- op hashing and alignment (numpy)
def _np():
    import numpy as np
    return np


def op_hashes(k, a, b, bits=False):
    """uint64 hash per op in [a, b): op type, qubit/bit operands, condition bit and the Z_POW angle word."""
    np = _np()
    t_all = k.op_types
    a, b = int(a), int(b)
    pre = np.asarray(t_all[:a])
    t = np.asarray(t_all[a:b])
    U = np.uint64
    n = len(t)
    h = t.astype(U) * U(0x9E3779B97F4A7C15)
    consts = [U(0xC2B2AE3D27D4EB4F), U(0x165667B19E3779F9), U(0xD6E8FEB86659FD93), U(0xFF51AFD7ED558CCD),
              U(0xC4CEB9FE1A85EC53), U(0x94D049BB133111EB), U(0xBF58476D1CE4E5B9), U(0x2545F4914F6CDD1D)]

    def add(mask_pre, mask_t, names):
        nb = int(np.count_nonzero(mask_pre))
        sel = np.flatnonzero(mask_t)
        for nm, c in names:
            s = k.stream(nm)[nb:nb + len(sel)].astype(U)
            h[sel] += (s + U(1)) * c
        return sel

    with np.errstate(over='ignore'):
        add((pre & 0x30) == 0x30, (t & 0x30) == 0x30, list(zip(['qqq0', 'qqq1', 'qqq2'], consts[0:3])))
        add((pre & 0x30) == 0x20, (t & 0x30) == 0x20, list(zip(['qq0', 'qq1'], consts[3:5])))
        add((pre & 0x30) == 0x10, (t & 0x30) == 0x10, [('q0', consts[5])])
        if bits:   # classical bit ids are allocation labels (renumbered when circuits are spliced): off by default
            add((pre & 0x40) > 0, (t & 0x40) > 0, [('b0', consts[6])])
            add((pre & 0x80) > 0, (t & 0x80) > 0, [('bc', consts[7])])
        # Z_POW angle word (two u64 per 16-byte angle record), indexed by the count of Z_POW ops before a
        z_pre = int(np.count_nonzero((pre & 0x7F) == 0x13))
        zs = np.flatnonzero((t & 0x7F) == 0x13)
        if len(zs):
            off, cnt, _ = k.offsets['angles']
            words = np.frombuffer(k.raw, dtype='<u8', count=2 * cnt, offset=off).reshape(cnt, 2)
            ang = words[z_pre:z_pre + len(zs)]
            h[zs] += (ang[:, 0] + U(1)) * U(0xA24BAED4963EE407) + (ang[:, 1] + U(1)) * U(0x9FB21C651E98DF25)
        h ^= h >> U(29)
        h *= U(0xBF58476D1CE4E5B9)
        h ^= h >> U(32)
    return h


def _kgrams(h, K):
    np = _np()
    n = len(h) - K + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    g = np.zeros(n, dtype=np.uint64)
    P = np.uint64(0x100000001B3)
    with np.errstate(over='ignore'):
        for t in range(K):
            g = g * P + h[t:t + n]
    return g


def align(ha, hb, K=K_GRAM, merge=MERGE_RUN, max_gaps=MAX_GAPS):
    """Local-resync alignment of two op-hash arrays. Returns (gaps, matched) where each gap is (i0, i1, j0, j1) with
    ha[i0:i1] replacing hb[j0:j1]; ops outside the gaps are identical in order. A tail that cannot be resynchronised is
    one final gap flagged by the caller via `unaligned` (returned as a third value)."""
    np = _np()
    na, nb = len(ha), len(hb)
    i = j = 0
    gaps = []
    unaligned = False
    chunk = 1 << 12
    while i < na or j < nb:
        # common run
        L = chunk
        while True:
            m = min(L, na - i, nb - j)
            if m <= 0:
                break
            eq = ha[i:i + m] == hb[j:j + m]
            if eq.all():
                i += m
                j += m
                if m < L:
                    break
                L *= 2
                continue
            r = int(np.argmin(eq))
            i += r
            j += r
            break
        if i >= na and j >= nb:
            break
        if i >= na or j >= nb:
            gaps.append((i, na, j, nb))
            break
        # mismatch at (i, j): find the nearest K-gram anchor in both
        W = 2048
        found = None
        while True:
            wa, wb = min(na - i, W + K), min(nb - j, W + K)
            ga, gb = _kgrams(ha[i:i + wa], K), _kgrams(hb[j:j + wb], K)
            if len(ga) and len(gb):
                ua, ia = np.unique(ga, return_index=True)
                ub, ib = np.unique(gb, return_index=True)
                common, xa, xb = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
                if len(common):
                    cost = ia[xa].astype(np.int64) + ib[xb].astype(np.int64)
                    c = int(np.argmin(cost))
                    found = (int(ia[xa][c]), int(ib[xb][c]))
                    break
            if W >= max(na - i, nb - j):
                break
            W *= 4
        if found is None:
            gaps.append((i, na, j, nb))
            unaligned = True
            break
        di, dj = found
        if di == 0 and dj == 0:   # K-gram matches here although ops[i] != ops[j] cannot happen; guard against loops
            di = dj = 1
        gaps.append((i, i + di, j, j + dj))
        i += di
        j += dj
        if len(gaps) > max_gaps:
            gaps.append((i, na, j, nb))
            unaligned = True
            break
    # merge gaps separated by a short matched run (one call site)
    out = []
    for g in gaps:
        if out and g[0] - out[-1][1] < merge and g[0] >= out[-1][1]:
            out[-1] = (out[-1][0], g[1], out[-1][2], g[3])
        else:
            out.append(g)
    return out, unaligned


def span_stats(k, a, b):
    """Static counts over ops [a, b): Toffoli-like, of those inside a condition block, Z_POW, non-Clifford Z_POW."""
    np = _np()
    a, b = int(a), int(b)
    if b <= a:
        return dict(ops=0, toffoli=0, conditional=0, zpow=0, zpow_nonclifford=0)
    t = np.asarray(k.op_types[a:b])
    tl = np.isin(t, (0x30, 0x31, 0x32, 0xB0, 0xB1, 0xB2))
    delta = np.zeros(len(t), dtype=np.int32)
    delta[t == 0x81] = 1
    delta[t == 0x01] = -1
    depth = np.cumsum(delta) - delta
    cond = tl & ((depth > 0) | (t >= 0x80))
    zs = (t & 0x7F) == 0x13
    nz = 0
    if zs.any():
        pre = int(np.count_nonzero((np.asarray(k.op_types[:a]) & 0x7F) == 0x13))
        off, cnt, _ = k.offsets['angles']
        words = np.frombuffer(k.raw, dtype='<u8', count=2 * cnt, offset=off).reshape(cnt, 2)[pre:pre + int(zs.sum())]
        # angle units of 2^-64 turns are the first u64 word pair; Clifford = multiple of 2^62 (see qhdl_exec.zpow_class)
        lo = words[:, 0]
        nz = int(np.count_nonzero((lo % np.uint64(1 << 62)) != 0)) if words.shape[0] else 0
    return dict(ops=b - a, toffoli=int(tl.sum()), conditional=int(cond.sum()), zpow=int(zs.sum()), zpow_nonclifford=nz)


def gap_signatures(ka, kb, gaps, offa=0, offb=0):
    """Per gap static counts in both files and the signature (Toffolis, Z_POW, ops) old -> new."""
    rows = []
    for (i0, i1, j0, j1) in gaps:
        sa = span_stats(ka, offa + i0, offa + i1)
        sb = span_stats(kb, offb + j0, offb + j1)
        rows.append(dict(a=[offa + i0, offa + i1], b=[offb + j0, offb + j1], old=sa, new=sb))
    return rows


def hint_for_gaps(rows, unaligned=False):
    """Technique hint from call-site structure. Sites whose Toffoli count does not change are glue and ignored: many
    equal-signature sites with a changed Toffoli count -> repeated-component; else component-resynthesis."""
    moved = [r for r in rows if r['new']['toffoli'] != r['old']['toffoli']]
    if not moved:
        return 'component-resynthesis', None
    sig = Counter((r['old']['toffoli'], r['new']['toffoli']) for r in moved)
    (top, cnt), = sig.most_common(1)
    share = cnt / len(moved)
    dom = dict(old_toffoli=top[0], new_toffoli=top[1], sites=cnt, share=round(share, 4), moved_sites=len(moved))
    if cnt >= REPEAT_SITES and share >= REPEAT_SHARE and not unaligned:
        return 'repeated-component', dom
    return 'component-resynthesis', dom


# ---------------------------------------------------------------- manifest helpers
def manifest(stage):
    if not (Path(stage) / 'manifest.csv').is_file():  # no dataset: file arguments still work, prefixes do not
        return []
    with open(Path(stage) / 'manifest.csv', newline='') as f:
        return list(csv.DictReader(f))


def mrow(rows, ref, task=None):
    hits = [r for r in rows if r['sha256'].startswith(ref) or r['sha12'].startswith(ref)]
    if task:
        hits = [r for r in hits if r['task'] == task]
    shas = {r['sha256'] for r in hits}
    if len(shas) != 1:
        return None
    return hits[0]


def credit(row):
    if not row:
        return None
    return dict(author=row.get('author'), url=row.get('source_notebook_url'), version=row.get('source_notebook_version'),
                public_since=row.get('public_since'), sha12=row.get('sha12'))


# ---------------------------------------------------------------- DQI
def workbench_command():
    """How to run the fork's workbench: its launcher if it can run (it needs the fork's .venv or WORKBENCH_PYTHON),
    else workbench.py beside it (or QHDL_WORKBENCH pointing at workbench.py) with this interpreter; None if absent."""
    wb = Path(WORKBENCH)
    if wb.suffix == '.py' and wb.is_file():
        return [sys.executable, str(wb)]
    launcher_ok = wb.is_file() and os.access(wb, os.X_OK) and (
        os.environ.get('WORKBENCH_PYTHON') or (wb.parent / '../../.venv/bin/python').exists())
    if launcher_ok:
        return [str(wb)]
    script = wb.with_name('workbench.py')
    if script.is_file():
        return [sys.executable, str(script)]
    return None


def regions_for(api, path, sha12, force=False):
    """Region tree JSON from `workbench summarize` (static Toffolis per INFO_TAG region); cached under work/_diff/."""
    d = Path(api.WORK) / '_diff' / 'regions'
    d.mkdir(parents=True, exist_ok=True)
    out = d / f'{sha12}.json'
    if out.exists() and not force:
        return json.loads(out.read_text()), out
    cmd = workbench_command()
    if cmd is None:
        raise api.QhdlError(f'workbench not found for {sha12}: no launcher at {WORKBENCH} and no workbench.py beside it',
                            'set QHDL_VIEWER_DIR to a kickmix-macos tools/circuit-viewer checkout (or QHDL_WORKBENCH to '
                            'its launcher or workbench.py)')
    r = subprocess.run(cmd + ['summarize', str(path), '--out', str(out)], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not out.exists():
        detail = (r.stderr.strip() or r.stdout.strip())[-300:]      # the fork's launcher reports on stdout
        raise api.QhdlError(f'workbench summarize failed for {sha12}: {detail}',
                            f'run {" ".join(cmd)} summarize {path} --out FILE')
    return json.loads(out.read_text()), out


def region_groups(tree):
    """path -> dict(depth, instances, toffoli, conditional, ops, hmr, spans[(start,end)])."""
    g = {}
    for r in tree['tag_regions']:
        x = g.setdefault(r['path'], dict(depth=r['depth'], instances=0, toffoli=0, conditional=0, ops=0, hmr=0, spans=[]))
        m = r['metrics']
        x['instances'] += 1
        x['toffoli'] += m['toffoli']
        x['conditional'] += m['conditional_toffoli']
        x['ops'] += m['ops']
        x['hmr'] += m['hmr']
        x['spans'].append((r['start'], r['end']))
    return g


def diff_dqi(api, P, X, deep=False):
    import qhdl_kmb as KM
    ta, ra = regions_for(api, P['path'], P['sha'][:12])
    tb, rb = regions_for(api, X['path'], X['sha'][:12])
    ga, gb = region_groups(ta), region_groups(tb)
    tot_a, tot_b = ta['metrics']['toffoli'], tb['metrics']['toffoli']
    facts_extra = dict(hmr_delta=tb['metrics']['hmr'] - ta['metrics']['hmr'],
                       conditional_delta=tb['metrics']['conditional_toffoli'] - ta['metrics']['conditional_toffoli'],
                       ops_delta=tb['metrics']['ops'] - ta['metrics']['ops'],
                       qubits=(ta['circuit']['qubits'], tb['circuit']['qubits']))
    # A file with no INFO_TAG regions cannot be attributed by tag path. Missing tags are not removals, so
    # refuse region attribution and report the whole-file delta as one row (use a tag-independent diff instead).
    tagless = [s for s, g in ((P['sha'][:12], ga), (X['sha'][:12], gb)) if not g]
    if tagless:
        whole = tot_b - tot_a
        row = dict(component='<whole file: tagless>', depth=0, instances_old=1, instances_new=1, toffoli_old=tot_a,
                   toffoli_new=tot_b, delta=whole, conditional_delta=facts_extra['conditional_delta'],
                   ops_delta=facts_extra['ops_delta'], hmr_delta=facts_extra['hmr_delta'], nested=False,
                   technique='unknown', sites=None, tagless=tagless)
        facts_extra['tagless'] = tagless
        return dict(task='dqi', old_total=tot_a, new_total=tot_b, whole_delta=whole, component_sum=whole,
                    unmatched=0, rows=[row], extra=facts_extra, region_files=[api.rel(ra), api.rel(rb)])
    paths = sorted(set(ga) | set(gb))
    ka = kb = None
    rows = []
    for p in paths:
        a = ga.get(p) or dict(depth=gb[p]['depth'], instances=0, toffoli=0, conditional=0, ops=0, hmr=0, spans=[])
        b = gb.get(p) or dict(depth=ga[p]['depth'], instances=0, toffoli=0, conditional=0, ops=0, hmr=0, spans=[])
        changed = (a['toffoli'], a['conditional'], a['ops'], a['hmr'], a['instances']) != \
                  (b['toffoli'], b['conditional'], b['ops'], b['hmr'], b['instances'])
        if not changed and not (deep and a['depth'] == 0):
            continue
        row = dict(component=p, depth=a['depth'], instances_old=a['instances'], instances_new=b['instances'],
                   toffoli_old=a['toffoli'], toffoli_new=b['toffoli'], delta=b['toffoli'] - a['toffoli'],
                   conditional_delta=b['conditional'] - a['conditional'], ops_delta=b['ops'] - a['ops'],
                   hmr_delta=b['hmr'] - a['hmr'], nested=a['depth'] > 0)
        if a['instances'] and a['instances'] == b['instances']:
            row['per_instance_delta'] = round(row['delta'] / a['instances'], 4)
        row['sites'] = None
        rows.append(row)
    # op-level alignment of the changed top-level components
    top = [r for r in rows if not r['nested']]
    for r in top:
        if r['instances_old'] != 1 or r['instances_new'] != 1:
            r['technique'] = 'component-resynthesis' if r['delta'] else 'unknown'
            continue
        if ka is None:
            ka, kb = KM.read(str(P['path'])), KM.read(str(X['path']))
        sa, ea = ga[r['component']]['spans'][0]
        sb, eb = gb[r['component']]['spans'][0]
        ha, hb = op_hashes(ka, sa, ea), op_hashes(kb, sb, eb)
        gaps, unal = align(ha, hb)
        if not gaps:
            if not deep:
                r['technique'] = 'unknown'
            else:
                r['technique'] = None
            r['sites'] = 0
            continue
        sigs = gap_signatures(ka, kb, gaps, sa, sb)
        tech, dom = hint_for_gaps(sigs, unal)
        r['technique'] = tech
        r['sites'] = len(gaps)
        r['unaligned_tail'] = bool(unal)
        r['gap_ops_old'] = sum(g[1] - g[0] for g in gaps)
        r['gap_ops_new'] = sum(g[3] - g[2] for g in gaps)
        r['gap_toffoli_delta'] = sum(s['new']['toffoli'] - s['old']['toffoli'] for s in sigs)
        hist = Counter((s['old']['toffoli'], s['new']['toffoli']) for s in sigs if s['new']['toffoli'] != s['old']['toffoli'])
        r['site_signatures'] = [dict(old_toffoli=o, new_toffoli=n, sites=c) for (o, n), c in hist.most_common(5)]
        r['sites_toffoli_moved'] = sum(hist.values())
        wsum = Counter()
        for s_ in sigs:
            if s_['new']['toffoli'] != s_['old']['toffoli']:
                wsum[(s_['old']['toffoli'], s_['new']['toffoli'], s_['old']['ops'], s_['new']['ops'])] += s_['new']['toffoli'] - s_['old']['toffoli']
        r['delta_by_signature'] = [dict(old_toffoli=o, new_toffoli=n, old_ops=oo, new_ops=no, sites=sum(1 for q in sigs if (q['old']['toffoli'], q['new']['toffoli'], q['old']['ops'], q['new']['ops']) == (o, n, oo, no)), delta=d)
                                   for (o, n, oo, no), d in sorted(wsum.items(), key=lambda kv: kv[1])[:8]]
        byd = sorted(sigs, key=lambda s: s['new']['toffoli'] - s['old']['toffoli'])
        r['top_gaps'] = [dict(old=s['a'], new=s['b'], toffoli=[s['old']['toffoli'], s['new']['toffoli']],
                              ops=[s['old']['ops'], s['new']['ops']]) for s in byd[:6]]
        tot = r['gap_toffoli_delta']
        acc, n90 = 0, 0
        for s in byd:
            acc += s['new']['toffoli'] - s['old']['toffoli']
            n90 += 1
            if tot and acc / tot >= 0.9:
                break
        r['sites_for_90pct_of_delta'] = n90
        r['dominant_site'] = dom
        r['gaps_first'] = [dict(old=s['a'], new=s['b']) for s in sigs[:3]]
    rows.sort(key=lambda r: (r['nested'], -abs(r['delta']), r['component']))
    top = [r for r in rows if not r['nested']]
    summed = sum(r['delta'] for r in top)
    whole = tot_b - tot_a
    return dict(task='dqi', old_total=tot_a, new_total=tot_b, whole_delta=whole, component_sum=summed,
                unmatched=whole - summed, rows=rows, extra=facts_extra, region_files=[api.rel(ra), api.rel(rb)])


# ---------------------------------------------------------------- invsqrt
def _runs(mask, minlen):
    np = _np()
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [[int(a), int(b)] for a, b in zip(st, en) if b - a >= minlen]


def other_invsqrt(api, stage, exclude):
    rows = [r for r in manifest(stage) if r['task'] == 'invsqrt' and r['sha256'] not in exclude]
    return rows


def diff_invsqrt(api, P, X, stage, donors=True):
    import qhdl_kmb as KM
    np = _np()
    ka, kb = KM.read(str(P['path'])), KM.read(str(X['path']))
    na, nb = len(ka.op_types), len(kb.op_types)
    ha, hb = op_hashes(ka, 0, na), op_hashes(kb, 0, nb)
    gaps, unal = align(ha, hb)
    sigs = gap_signatures(ka, kb, gaps)
    whole_a, whole_b = span_stats(ka, 0, na), span_stats(kb, 0, nb)
    rows = []
    for idx, s in enumerate(sigs):
        g = gaps[idx]
        rows.append(dict(old_ops=s['a'], new_ops=s['b'], old=s['old'], new=s['new'],
                         toffoli_delta=s['new']['toffoli'] - s['old']['toffoli'],
                         zpow_delta=s['new']['zpow'] - s['old']['zpow'],
                         nonclifford_delta=s['new']['zpow_nonclifford'] - s['old']['zpow_nonclifford'],
                         conditional_delta=s['new']['conditional'] - s['old']['conditional']))
    # per-gap donor identification (best other public file by K-gram coverage of that gap) and the provenance map of the
    # whole new file (greedy cover of its ops by public invsqrt files; ties go to the earliest public_since)
    prov = []
    if donors:
        gx = _kgrams(hb, K_GRAM)
        cache = {}
        for r in other_invsqrt(api, stage, {X['sha']}):
            path = Path(stage) / 'circuits' / 'invsqrt' / f"{r['sha12']}.kmb"
            if path.exists():
                kd = KM.read(str(path))
                cache[r['sha12']] = (np.unique(_kgrams(op_hashes(kd, 0, len(kd.op_types)), K_GRAM)), r)
        for row, g in zip(rows, gaps):
            if g[3] - g[2] < K_GRAM:
                row['donors'] = []
                row['technique'] = 'component-resynthesis'
                continue
            seg = gx[g[2]:g[3] - K_GRAM + 1]
            cands = []
            for sha12, (gd, mr) in cache.items():
                if sha12 == P['sha'][:12]:
                    continue
                c = float(np.isin(seg, gd).mean()) if len(seg) else 0.0
                if c >= 0.5:
                    cands.append((c, mr.get('public_since') or '', sha12, mr))
            cands.sort(key=lambda t: (-round(t[0], 3), t[1], t[2]))
            row['donors'] = [dict(sha12=s12, coverage=round(c, 4), credit=credit(mr)) for c, _, s12, mr in cands[:3]]
            row['splice'] = bool(cands and cands[0][0] >= SPLICE_COVER and g[3] - g[2] >= 64)
            row['technique'] = 'splice' if row['splice'] else 'component-resynthesis'
        covered = np.zeros(nb, dtype=bool)
        masks = {}
        for sha12, (gd, mr) in cache.items():
            hit = np.isin(gx, gd)
            d = np.zeros(nb + 1, dtype=np.int32)
            idx = np.flatnonzero(hit)
            np.add.at(d, idx, 1)
            np.add.at(d, np.minimum(idx + K_GRAM, nb), -1)
            masks[sha12] = (np.cumsum(d[:nb]) > 0, mr)
        for _ in range(4):
            gains = {k_: int(np.count_nonzero(m_ & ~covered)) for k_, (m_, _) in masks.items()}
            top_gain = max(gains.values()) if gains else 0
            tol = max(1, nb // 1000)
            tied = [k_ for k_, g_ in gains.items() if g_ >= top_gain - tol]
            pick = min(tied, key=lambda k_: (masks[k_][1].get('public_since') or '', k_)) if tied else None
            best = (gains[pick], pick, masks[pick][1], '') if pick else None
            if not best or best[0] < max(64, nb // 100):
                break
            m = masks[best[1]][0]
            new = m & ~covered
            runs = _runs(new, 32)
            prov.append(dict(sha12=best[1], ops_new=best[0], share=round(best[0] / nb, 4), ranges=runs[:6], n_ranges=len(runs),
                             credit=credit(best[2])))
            covered |= m
        left = int(nb - np.count_nonzero(covered))
        prov_left = left
    else:
        prov_left = None
    for row in rows:
        row.setdefault('technique', 'component-resynthesis')
    rows.sort(key=lambda r: -(abs(r['toffoli_delta']) + abs(r['new_ops'][1] - r['new_ops'][0]) * 1e-9))
    return dict(task='invsqrt', old_total=whole_a['toffoli'], new_total=whole_b['toffoli'],
                whole_delta=whole_b['toffoli'] - whole_a['toffoli'],
                component_sum=sum(r['toffoli_delta'] for r in rows),
                unmatched=(whole_b['toffoli'] - whole_a['toffoli']) - sum(r['toffoli_delta'] for r in rows),
                rows=rows, unaligned_tail=bool(unal), provenance=prov, provenance_uncovered_ops=prov_left,
                extra=dict(zpow_old=whole_a['zpow'], zpow_new=whole_b['zpow'],
                           nonclifford_old=whole_a['zpow_nonclifford'], nonclifford_new=whole_b['zpow_nonclifford'],
                           ops_old=na, ops_new=nb, matched_old=na - sum(g[1] - g[0] for g in gaps),
                           qubits=(ka.num_qubits, kb.num_qubits)))


# ---------------------------------------------------------------- chemistry
def _csv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def load_chem(stage, ref):
    stage = Path(stage)
    fams = [r for r in _csv(stage / 'parts/families.csv') if r['family'].startswith(ref)]
    if len(fams) != 1:
        raise ValueError(f'chemistry family prefix {ref!r}: {len(fams)} matches in {stage}/parts/families.csv')
    fam = fams[0]
    ang = {r['part_id']: r for r in _csv(stage / 'parts/angle_circuits.csv')}
    slots = [ang[p] for p in fam['parts'].split(';')]
    man = next((r for r in manifest(stage) if r['task'] == 'chemistry' and r['sha12'] == fam['family']), None)
    return fam, slots, man


def diff_chem(api, P, X, stage):
    fa, sa, ma = load_chem(stage, P)
    fb, sb, mb = load_chem(stage, X)
    rows = []
    for k, (a, b) in enumerate(zip(sa, sb)):
        same_file = a['file_sha256'] == b['file_sha256']
        rows.append(dict(slot=k, static_old=int(a['static_toffoli']), static_new=int(b['static_toffoli']),
                         delta_per_pass=int(b['static_toffoli']) - int(a['static_toffoli']),
                         delta_per_shot=float(b['toffoli_per_shot']) - float(a['toffoli_per_shot']),
                         words_same=a['words_sha256'] == b['words_sha256'], prev_words_same=a['prev_words_sha256'] == b['prev_words_sha256'],
                         loader_same=same_file, qubits=(int(a['num_qubits']), int(b['num_qubits']))))
    rows = [r for r in rows if not (r['loader_same'] and r['words_same'] and r['delta_per_pass'] == 0)]
    ta, tb = float(fa['T_official']), float(fb['T_official'])
    la, lb = float(fa['lambda_official']), float(fb['lambda_official'])
    qa, qb = int(fa['Q_official']), int(fb['Q_official'])
    ang_a = sum(float(s['toffoli_per_shot']) for s in sa)
    ang_b = sum(float(s['toffoli_per_shot']) for s in sb)
    non_a, non_b = ta - ang_a, tb - ang_b
    changed = [r for r in rows if not r['loader_same']]
    flags = dict(table_changed=fa['table_key'] != fb['table_key'], words_changed=any(not r['words_same'] for r in rows),
                 npz_changed=fa['npz_sha256'] != fb['npz_sha256'], lambda_changed=la != lb, q_changed=qa != qb,
                 loaders_changed=len(changed))
    hints = []
    if flags['q_changed']:
        hints.append('width-change')
    if flags['table_changed'] or flags['words_changed']:
        hints.append('table-change')
    if changed and not flags['table_changed'] and not flags['words_changed']:
        hints.append('component-resynthesis')
    if flags['npz_changed'] and not flags['table_changed'] and not flags['words_changed']:
        hints.append('weights-refit')
    if not hints and rows:
        hints.append('unknown')
    rows.sort(key=lambda r: (r['delta_per_pass'], r['slot']))
    cost_a, cost_b = float(fa['cost_official']), float(fb['cost_official'])
    return dict(task='chemistry', old_total=ta, new_total=tb, whole_delta=tb - ta, component_sum=ang_b - ang_a,
                unmatched=(tb - ta) - (ang_b - ang_a), rows=rows, hints=hints, flags=flags,
                extra=dict(cost_old=cost_a, cost_new=cost_b, cost_ratio=cost_b / cost_a, lambda_old=la, lambda_new=lb,
                           q=(qa, qb), non_angle_old=non_a, non_angle_new=non_b,
                           table_old=fa['table_key'], table_new=fb['table_key'],
                           npz_old=fa['npz_sha256'], npz_new=fb['npz_sha256'],
                           loaders_per_pass=(ang_a / 2, ang_b / 2),
                           credit_new=credit(mb), credit_old=credit(ma)))


# ---------------------------------------------------------------- envelope
def top_technique(res):
    if res['task'] == 'chemistry':
        return res['hints']
    rs = [r for r in res['rows'] if not r.get('nested') and r.get('technique')]
    if res['task'] == 'dqi':
        base = {r['technique'] for r in rs}
        if not rs:
            return ['unknown']
        return sorted(base)
    out = {r['technique'] for r in rs}
    prov = res.get('provenance') or []
    if len(prov) >= 2 and sum(p_['share'] for p_ in prov) >= 0.9:
        out.add('splice')
    return sorted(out) or ['unknown']


def public_rows(res):
    out = []
    if res['task'] == 'dqi':
        for r in res['rows']:
            row = {'component': ('  ' if r['nested'] else '') + r['component'], 'inst': f"{r['instances_old']}>{r['instances_new']}",
                   'd': r['delta'], 'cond': r['conditional_delta'], 'per_inst': r.get('per_instance_delta', 'n/a'),
                   'sites': r.get('sites_toffoli_moved', r['sites']) if r['sites'] is not None else '-',
                   'hint': r.get('technique') or ('nested' if r['nested'] else '-')}
            ds = (r.get('delta_by_signature') or [None])[0]
            if ds:
                row['site'] = f"{ds['old_toffoli']}T>{ds['new_toffoli']}T x{ds['sites']}"
            out.append(row)
    elif res['task'] == 'invsqrt':
        for r in res['rows']:
            d = (r.get('donors') or [None])[0]
            out.append({'old': f"{r['old_ops'][0]}:{r['old_ops'][1]}", 'new': f"{r['new_ops'][0]}:{r['new_ops'][1]}",
                        'T': f"{r['old']['toffoli']}>{r['new']['toffoli']}", 'd': r['toffoli_delta'], 'zp': r['zpow_delta'],
                        'hint': r['technique'], 'src': (f"{d['sha12'][:8]} {int(100 * d['coverage'])}%" if d else '-')})
    else:
        for r in res['rows']:
            out.append({'slot': r['slot'], 'old': r['static_old'], 'new': r['static_new'],
                        'd_pass': r['delta_per_pass'], 'words_same': r['words_same'], 'loader_same': r['loader_same']})
    return out


def run(api, a):
    t0 = time.time()
    stage = Path(a.stage) if a.stage else Path(api.DATASET)
    rows_m = manifest(stage)
    if a.task == 'chemistry' or (a.task is None and (mrow(rows_m, a.circuit, 'chemistry') or mrow(rows_m, a.circuit2, 'chemistry'))):
        task = 'chemistry'
        P, X = a.circuit, a.circuit2
        mp, mx = mrow(rows_m, P, 'chemistry'), mrow(rows_m, X, 'chemistry')
        pid = mp['sha12'] if mp else P
        xid = mx['sha12'] if mx else X
        res = diff_chem(api, pid, xid, stage)
        psha = xsha = None
        mp_row, mx_row = mp, mx
    else:
        P, X = api.resolve(a.circuit), api.resolve(a.circuit2)
        mp_row, mx_row = P.get('manifest'), X.get('manifest')
        task = a.task or (mp_row or mx_row or {}).get('task') or 'unknown'
        if mp_row and mx_row and mp_row['task'] != mx_row['task']:
            raise api.QhdlError(f"different tasks: {mp_row['task']} vs {mx_row['task']}", 'diff compares two files of one task')
        if task not in ('dqi', 'invsqrt') and P['sha'] != X['sha']:
            raise api.QhdlError('cannot tell the task; pass --task dqi|invsqrt|chemistry')
        pid, xid = P['sha'][:12], X['sha'][:12]
        psha, xsha = P['sha'], X['sha']
        if P['sha'] == X['sha']:
            res = dict(task=task, old_total=None, new_total=None, whole_delta=0, component_sum=0, unmatched=0, rows=[], extra={},
                       identical=True)
        elif task == 'dqi':
            res = diff_dqi(api, P, X, deep=a.deep)
        else:
            res = diff_invsqrt(api, P, X, stage)
    led = E.Ledger()
    led.earn('static', 'netlist counts and recorded manifest fields; no per-shot means')
    rows = public_rows(res)
    techs = top_technique(res) if rows else ['none']
    out_dir = Path(a.out) if a.out else Path(api.WORK) / '_diff' / f'{pid}__{xid}'
    full = dict(schema='vqcc-qhdl-diff/1', task=res['task'], old=pid, new=xid, old_sha256=psha, new_sha256=xsha,
                credit_new=credit(mx_row), credit_old=credit(mp_row), techniques=techs,
                whole_delta=res['whole_delta'], component_sum=res['component_sum'], unmatched=res['unmatched'],
                result=res, tier='static', seconds=round(time.time() - t0, 1))
    ev = out_dir / 'diff.json'
    api.write_json(ev, full)
    whole = res['whole_delta']
    unm = res['unmatched']
    frac = (abs(unm) / abs(whole)) if whole else 0.0
    unit = 'Toffolis/shot' if res['task'] == 'chemistry' else 'static Toffolis'
    if (res.get('extra') or {}).get('tagless'):
        claim = (f"dqi {pid}>{xid}: {whole:+g} {unit} whole-file only; region attribution refused because "
                 f"{', '.join(res['extra']['tagless'])} has no INFO_TAG regions (use a tag-independent diff)")
    elif res.get('identical') or not rows:
        claim = f'{pid} and {xid} are the same circuit: 0 changed components' if res.get('identical') else \
            f'{pid} vs {xid}: no changed components found ({unit} delta {whole:+g})'
    else:
        nchg = len([r for r in rows if not str(r.get('component', '')).startswith(' ')]) if res['task'] == 'dqi' else len(rows)
        claim = (f"{res['task']} {pid}>{xid}: {whole:+g} {unit}; {nchg} components sum {res['component_sum']:+g}, "
                 f"unmatched {unm:+g} ({100 * frac:.1f}%); {', '.join(techs)}")
    facts = dict(whole_delta=whole, component_sum=res['component_sum'], unmatched=unm,
                 unmatched_pct=round(100 * frac, 3), techniques=techs, new_author=(credit(mx_row) or {}).get('author'))
    ex = res.get('extra') or {}
    if res['task'] == 'dqi':
        facts.update(cond_delta=ex.get("conditional_delta"), hmr_delta=ex.get("hmr_delta"), ops_delta=ex.get("ops_delta"), tagless=ex.get("tagless"))
    elif res['task'] == 'invsqrt':
        facts.update(zpow=[ex.get('zpow_old'), ex.get('zpow_new')],
                     provenance=[f"{p_['sha12'][:8]} {int(100 * p_['share'])}%"
                                 for p_ in (res.get('provenance') or [])])
    elif res['task'] == 'chemistry':
        fl = res['flags']
        facts.update(cost_ratio=round(ex['cost_ratio'], 6), per_pass=list(ex['loaders_per_pass']),
                     table_chg=fl['table_changed'], words_chg=fl['words_changed'], npz_chg=fl['npz_changed'],
                     loaders_chg=fl['loaders_changed'])
    return E.Envelope(led, claim, domain=f"{res['task']} pair, static counts",
                      evidence=api.rel(ev), rerun=api.cmd('diff', '-c', pid, '-c2', xid, '--top', a.top), tool_sha=api.tool_sha(),
                      circuit_sha=xsha, facts=facts, rows=rows, table=api.rel(ev), next_cmds=[])


def qhdl_register(subparsers, api):
    import argparse
    p = subparsers.add_parser('diff', help='what changed between a base circuit and a new public file: ranked components, '
                                           'multiplicity, per-instance delta, technique hint (avenues G6)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--circuit', help='the base circuit: sha prefix or .kmb path (chemistry: family sha prefix)')
    p.add_argument('-c2', '--circuit2', dest='circuit2', help='the new file')
    p.add_argument('--task', choices=['dqi', 'invsqrt', 'chemistry'])
    p.add_argument('--stage', help='circuits dataset stage dir (default: the workspace stage)')
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--deep', action='store_true', help='dqi: also hash-compare top-level regions whose counts did not change')
    p.add_argument('--out', help='directory for the full diff.json (default work/_diff/<base>__<new>/)')
    p.add_argument('--json', action='store_true', help='(envelope is always JSON)')
    p.add_argument('--all', action='store_true')

    def fn(a):
        if not a.circuit or not a.circuit2:
            raise api.QhdlError('diff needs -c <base> and -c2 <new file>')
        return run(api, a)
    p.set_defaults(fn=fn)
