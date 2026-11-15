"""Live-width peak of a .kmb circuit of any size, per INFO_TAG region (backs `qhdl peak`).

Same interval model as qhdl_lifetime_allocator.py and qhdl_dqi_peak_anatomy.py (registers live for the whole circuit,
a scratch interval ends at an unconditional R/HMR or at its last touch), so the peak here is the width the
allocator can reach by renaming alone. Works on the op-type and qubit streams with NumPy: a 60M-op DQI circuit takes
about 20 s and 5-8 GB of memory, so `qhdl peak` caches the result in the workspace.

Two liveness models (`liveness=`):
  allocator  (default) registers are live for the whole circuit; scratch intervals end at an unconditional R/HMR or
             last touch. This is the width a renaming allocator can reach without reusing register wires.
  reset      as `qhdl runs`/`idle` (qhdl_idle_wires): a register wire that is reset mid-circuit is free until its next
             touch, like scratch (qhdl_lifetime_allocator live_profile(register_gaps=True)). Same peak on the
             public circuits seen so far, but fewer ops at the peak and a different run count and numbering
             (bce2948ff24e: 92 runs / 16,129 ops allocator, 99 runs / 15,742 ops reset).
analyse() reports the run count under the other model in `other_liveness`.

Backends (`backend=`): 'python' (default, the reference) or 'rust' (kickmix_analysis, crate kmx-analysis in the
engine repo; streaming, byte-identical output on a parity set of circuits).

Library: analyse(path, level=None, liveness='allocator', backend='python') -> dict (json-able); regions(k) -> top-level INFO_TAG regions;
region_kinds(k, regions) -> static per-kind histogram per region (also used by `qhdl exec` on large circuits).
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402
import qhdl_lifetime_allocator as A  # noqa: E402

TOFF = np.zeros(256, bool)
TOFF[list(K.TOFFOLI_LIKE)] = True


def op_name(code: int) -> str:
    code = int(code)
    if code == K.PUSH_CONDITION:
        return 'PUSH_CONDITION'
    base = K.NAMES.get(code & 0x7F, hex(code & 0x7F))
    return base + ('_IF' if code & 0x80 else '')


def level_of(spec, peak):
    if spec in (None, 'peak'):
        return peak
    s = str(spec)
    if s.startswith('peak-'):
        return peak - int(s[5:])
    return int(s)


def _tag_events(k):
    """[(op index, kind, name)] for every INFO_TAG."""
    t = k.op_types
    pos = np.flatnonzero(t == K.INFO_TAG).tolist()
    tags = k.info_tags()
    assert len(pos) == len(tags)
    out = []
    for o, txt in zip(pos, tags):
        kind, _, name = txt.strip().strip('"').partition(':')
        out.append((o, kind, name))
    return out


def regions(k, events=None):
    """Top-level INFO_TAG regions [dict(name, start, end)] covering every op ('untagged' gaps; repeats get .2, .3),
    or ten equal slices seg0..seg9 when the circuit has no tags (same convention as qhdl `profile`)."""
    n = len(k.op_types)
    events = _tag_events(k) if events is None else events
    raw, depth = [], 0
    for o, kind, name in events:
        if kind == '@enter':
            if depth == 0:
                raw.append([name, o, None])
            depth += 1
        elif kind == '@exit' and depth:
            depth -= 1
            if depth == 0:
                raw[-1][2] = o + 1
    raw = [(nm, s, e if e is not None else n) for nm, s, e in raw]
    if not raw:
        return [dict(name=f'seg{j}', start=n * j // 10, end=n * (j + 1) // 10) for j in range(10)]
    out, pos, seen = [], 0, {}
    for name, s, e in raw:
        if s > pos:
            out.append(['untagged', pos, s])
        out.append([name, s, e])
        pos = e
    if pos < n:
        out.append(['untagged', pos, n])
    res = []
    for name, s, e in out:
        seen[name] = seen.get(name, 0) + 1
        res.append(dict(name=name if seen[name] == 1 else f'{name}.{seen[name]}', start=s, end=e))
    return res


def leaf_segments(k, events=None):
    """Segments of constant tag path: (starts array, merged path per segment; instance numbers dropped)."""
    events = _tag_events(k) if events is None else events
    stack, starts, paths = [], [0], ['(untagged)']
    for o, kind, name in events:
        if kind == '@enter':
            stack.append(name)
        elif kind == '@exit' and stack:
            stack.pop()
        starts.append(o)
        paths.append('/'.join(stack) or '(untagged)')
    return np.array(starts, np.int64), paths


def region_kinds(k, regs):
    """Per region: {kind name: static count}, plus Toffoli-like split into unconditional (condition depth 0, not an
    _IF form: runs in every shot) and conditional (may be skipped)."""
    t = k.op_types
    depth = K.condition_depth(t)
    out = []
    for r in regs:
        s, e = r['start'], r['end']
        seg = np.asarray(t[s:e])
        bc = np.bincount(seg, minlength=256)
        kinds = {op_name(c): int(bc[c]) for c in np.flatnonzero(bc) if c != K.INFO_TAG}
        tm = TOFF[seg]
        cond = tm & ((depth[s:e] > 0) | (seg >= 0x80))
        out.append(dict(kinds=dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
                        static_toffolis=int(tm.sum()), unconditional_toffolis=int((tm & ~cond).sum())))
    return out


LIVENESS = ('allocator', 'reset')
BACKENDS = ('python', 'rust')


def _kickmix_analysis():
    try:
        import kickmix_analysis
    except ImportError as e:  # not in the project environment on purpose
        raise ImportError('backend="rust" needs the kickmix_analysis module (engine repo crate kmx-analysis); run '
                          'with $QHDL_ANALYSIS_PYTHON (an interpreter that can import kickmix_analysis)') from e
    return kickmix_analysis


def run_bounds(prof, lvl):
    idx = np.flatnonzero(prof >= lvl)
    if not len(idx):
        return idx, np.zeros(0, np.int64), np.zeros(0, np.int64)
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return idx, idx[np.r_[0, cuts]], idx[np.r_[cuts - 1, len(idx) - 1]]


def analyse(path, level=None, liveness='allocator', backend='python'):
    if liveness not in LIVENESS:
        raise ValueError(f'liveness must be one of {LIVENESS}')
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}')
    reset_model = liveness == 'reset'
    k = K.read(str(path))
    t = k.op_types
    nops = len(t)
    events = _tag_events(k)
    regmap = A.register_qubits(k)
    if backend == 'rust':
        KA = _kickmix_analysis()
        lv = KA.liveness(str(path), register_gaps=reset_model)
        iv = lv['intervals']
        prof = lv['profile']  # int32; same values as the int64 Python profile
        alt = KA.live_profile(iv, nops, regmap, register_gaps=not reset_model)
    else:
        is_reset, _ = A.reset_ops(k)
        ops, qs, order, _ = A.references(k)
        iv = A.build_intervals(k, ops, qs, order, is_reset)
        del ops, qs, order
        prof = A.live_profile(iv, nops, regmap, register_gaps=reset_model)
        alt = A.live_profile(iv, nops, regmap, register_gaps=not reset_model)
    peak = int(prof.max())
    lvl = level_of(level, peak)
    at = prof >= lvl
    idx, run_s, run_e = run_bounds(prof, lvl)
    alt_peak = int(alt.max())
    alt_lvl = level_of(level, alt_peak)
    alt_idx, alt_s, _ = run_bounds(alt, alt_lvl)
    other = dict(liveness=LIVENESS[1 - LIVENESS.index(liveness)], peak=alt_peak, level=int(alt_lvl),
                 runs=int(len(alt_s)), ops_at_level=int(len(alt_idx)))
    del alt
    run_len = run_e - run_s + 1
    run_cum = np.r_[0, np.cumsum(run_len)]

    regs = regions(k, events)
    rstarts = np.array([r['start'] for r in regs], np.int64)
    starts, paths = leaf_segments(k, events)

    def region_of(op):
        return regs[int(np.searchsorted(rstarts, op, 'right')) - 1]['name']

    def path_of(op):
        return paths[int(np.searchsorted(starts, op, 'right')) - 1]

    # Top-level region rows.
    kinds = region_kinds(k, regs)
    run_region = [region_of(int(s)) for s in run_s.tolist()]
    rc = collections.Counter(run_region)
    region_rows = []
    for r, kd in zip(regs, kinds):
        seg = prof[r['start']:r['end']]
        if len(seg) == 0:
            continue
        region_rows.append(dict(region=r['name'], ops=f"{r['start']}..{r['end'] - 1}", max_live=int(seg.max()),
                                ops_at_level=int(at[r['start']:r['end']].sum()), runs=rc.get(r['name'], 0),
                                static_toffolis=kd['static_toffolis']))

    # Leaf-path rows (instances merged).
    seg_len = np.diff(np.append(starts, nops))
    nz = seg_len > 0
    st_nz = starts[nz]
    smax = np.maximum.reduceat(prof, st_nz)
    sat = np.add.reduceat(at.astype(np.int64), st_nz)
    stf = np.add.reduceat(TOFF[np.asarray(t)].astype(np.int64), st_nz)
    by = {}
    for p, m, n_, a_, tf in zip([paths[i] for i in np.flatnonzero(nz)], smax.tolist(), seg_len[nz].tolist(),
                                sat.tolist(), stf.tolist()):
        d = by.setdefault(p, dict(path=p, max_live=0, ops=0, segments=0, ops_at_level=0, static_toffolis=0))
        d['max_live'] = max(d['max_live'], int(m))
        d['ops'] += int(n_)
        d['segments'] += 1
        d['ops_at_level'] += int(a_)
        d['static_toffolis'] += int(tf)
    path_rows = sorted(by.values(), key=lambda d: (-d['ops_at_level'], -d['max_live'], d['path']))
    run_rows = [dict(run=f'R{i}', start=int(s), end=int(e), ops=int(e - s + 1), path=path_of(int(s)))
                for i, (s, e) in enumerate(zip(run_s.tolist(), run_e.tolist()))]
    runs_by_path = collections.Counter(r['path'] for r in run_rows).most_common()

    # Anatomy at the first op at the level: scratch intervals live there, by the region where they started.
    reg_ids = np.array(sorted(regmap), np.int64)
    is_reg_iv = np.isin(iv['logical'], reg_ids)
    anat = {}
    if len(idx):
        p0 = int(idx[0])
        live = (~is_reg_iv) & (iv['s'] <= p0) & (iv['e'] >= p0)
        anat = dict(first_op=p0, first_path=path_of(p0), scratch_live=int(live.sum()),
                    scratch_by_start=collections.Counter(path_of(int(s)) for s in iv['s'][live].tolist()).most_common(8),
                    long_lived_scratch=int((live & ((iv['e'] - iv['s'] + 1) > nops // 4)).sum()))

    # Idle gaps: consecutive touches of one wire with no touch between, the wire live throughout (scratch: inside
    # one interval; register: always live), that strictly contain at least one whole run at the level.
    if backend == 'rust':
        # Same gaps from a streaming pass (Python prefilter P[b] - P[a+1] > 0 is implied by containing a whole run).
        ga0, gb0, gw0 = KA.idle_gaps(str(path), run_s, run_e, register_gaps=reset_model)
        wf, wl = lv['wire_first'], lv['wire_last']
        maxq = int(max(len(wl) - 1, int(reg_ids.max()) if len(reg_ids) else 0, k.num_qubits - 1))
        regmask = np.zeros(maxq + 1, bool)
        regmask[reg_ids] = True
        gaps_a, gaps_b, gaps_w = [ga0], [gb0], [gw0]
        for q in reg_ids.tolist():
            if q < len(wl) and wl[q] >= 0:
                gaps_a.append(np.array([-1, int(wl[q])], np.int64))
                gaps_b.append(np.array([int(wf[q]), nops], np.int64))
                gaps_w.append(np.array([q, q], np.int64))
            else:
                gaps_a.append(np.array([-1], np.int64))
                gaps_b.append(np.array([nops], np.int64))
                gaps_w.append(np.array([q], np.int64))
    else:
        sq, so, first, rr = iv['sq'], iv['so'], iv['first'], iv['reset_ref']
        maxq = int(max(int(sq.max()) if len(sq) else 0, int(reg_ids.max()) if len(reg_ids) else 0, k.num_qubits - 1))
        regmask = np.zeros(maxq + 1, bool)
        regmask[reg_ids] = True
        P = np.r_[0, np.cumsum(at, dtype=np.int64)]  # P[i] = ops at level before op i
        gaps_a, gaps_b, gaps_w = [], [], []
        CH = 20_000_000
        for c0 in range(0, max(len(sq) - 1, 0), CH):
            c1 = min(c0 + CH, len(sq) - 1)
            a = so[c0:c1]
            b = so[c0 + 1:c1 + 1]
            w = sq[c0:c1]
            cont = (~first[c0 + 1:c1 + 1]) & ((~rr[c0:c1]) | (regmask[w] & (not reset_model)))
            m = cont & (b - a > 1)
            m[m] = (P[b[m]] - P[a[m] + 1]) > 0
            gaps_a.append(a[m].astype(np.int64))
            gaps_b.append(b[m].astype(np.int64))
            gaps_w.append(w[m].astype(np.int64))
        # Registers before their first and after their last touch (or never touched).
        touched = np.unique(sq) if len(sq) else np.zeros(0, np.int64)
        fi = np.flatnonzero(first)
        la = np.r_[fi[1:] - 1, len(sq) - 1] if len(fi) else fi
        for q in reg_ids.tolist():
            j = int(np.searchsorted(touched, q))
            if j < len(touched) and touched[j] == q:
                jj = int(np.searchsorted(sq[fi], q))
                gaps_a.append(np.array([-1, int(so[la[jj]])], np.int64))
                gaps_b.append(np.array([int(so[fi[jj]]), nops], np.int64))
                gaps_w.append(np.array([q, q], np.int64))
            else:
                gaps_a.append(np.array([-1], np.int64))
                gaps_b.append(np.array([nops], np.int64))
                gaps_w.append(np.array([q], np.int64))
    ga = np.concatenate(gaps_a) if gaps_a else np.zeros(0, np.int64)
    gb = np.concatenate(gaps_b) if gaps_b else np.zeros(0, np.int64)
    gw = np.concatenate(gaps_w) if gaps_w else np.zeros(0, np.int64)
    i1 = np.searchsorted(run_s, ga, 'right')  # first run starting after a
    i2 = np.searchsorted(run_e, gb, 'left')   # runs ending before b
    closed = np.maximum(i2 - i1, 0)
    keep = closed > 0
    ga, gb, gw, i1, i2, closed = ga[keep], gb[keep], gw[keep], i1[keep], i2[keep], closed[keep]
    lops = run_cum[i2] - run_cum[i1]
    o = np.lexsort((ga, gw, -lops, -closed))
    names = {}
    for nm, contents in k.registers:
        for is_q, qid, _ in contents:
            if is_q:
                names[qid] = nm
    idle_rows = []
    for j in o.tolist():
        w = int(gw[j])
        idle_rows.append(dict(wire=f'q{w}', role=f'register {names.get(w, "?")}' if regmask[w] else 'scratch',
                              idle=f'{int(ga[j]) + 1}..{int(gb[j]) - 1}', runs_closed=int(closed[j]),
                              runs=f'R{int(i1[j])}..R{int(i2[j]) - 1}', ops_closed=int(lops[j]),
                              first_path=run_rows[int(i1[j])]['path']))
    runs_with_idle_scratch = np.zeros(len(run_s) + 1, np.int64)
    sc = ~regmask[gw]
    np.add.at(runs_with_idle_scratch, i1[sc], 1)
    np.add.at(runs_with_idle_scratch, i2[sc], -1)
    covered_scratch = int((np.cumsum(runs_with_idle_scratch)[:len(run_s)] > 0).sum())
    return dict(input_sha256=A.sha256(path), num_ops=int(nops), num_qubits=int(k.num_qubits), registers=len(regmap),
                liveness=liveness, other_liveness=other, peak=peak, level=int(lvl), ops_at_level=int(at.sum()), runs=len(run_s), regions=region_rows,
                paths=path_rows, run_rows=run_rows, runs_by_path=runs_by_path, anatomy=anat, idle=idle_rows,
                runs_with_idle_scratch=covered_scratch, region_kinds=[dict(region=r['name'], **kd)
                                                                      for r, kd in zip(regs, kinds)])
