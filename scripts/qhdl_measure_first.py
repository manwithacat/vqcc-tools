#!/usr/bin/env python3
"""qhdl plugin: `measure-first` — write-only wire runs that end in an HMR.

    qhdl measure-first -c CIRCUIT            (or: python3 scripts/qhdl_measure_first.py CIRCUIT.kmb [--json OUT])

Identity (exact for every input and outcome; credited to Kirill Tymchishin's written proof for the CCX case and to
Md. Hamid Hosen's measure-first rewrite of write-only runs): if after op s a wire t is only *written* (CX/CCX target,
X) and then measured by HMR at op h, its value reaches the circuit only through the HMR phase (-1)^(m (t_s + f)), f the
XOR of the writes. Moving the HMR to s turns each CCX(a,b->t) into CZ(a,b) if m (one Toffoli saved), each CX(c->t)
into Z(c) if m, and each X(t) into a sign if m; the outcome distribution is unchanged.

Reads are conservative: any control role, CZ/CCZ/Z/Z_POW on t, SWAP/CSWAP (any role), R, any op carrying a condition
bit, and any op inside a PUSH/POP condition block end a run (a run must also end in an unconditioned HMR).

Width: a wire's live interval runs from its first touch after a zeroing (start, R, HMR) to the next zeroing. A run's
wire becomes free from its first write s to the HMR h, so the report also gives the live-width peak before and after
freeing every run (a static model: Q is the circuit's wire count, so a lower live peak needs a re-packing to count).

Tier: static (exact pattern match on the op stream; the rewrite itself is an identity, not a search).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402

READ, W_CX, W_CCX, W_X, MEAS = 0, 1, 2, 3, 4
OPC = {v: k for k, v in K.NAMES.items()}


def scan(path: str | Path, width: bool = True, conditioned: bool = False) -> dict:
    """conditioned=True also lets conditioned writes (condition bit or PUSH/POP block) continue a run: a conditioned
    CCX becomes a CZ conditioned on its condition and on m (the identity holds per branch); such runs are tallied
    separately (ccx_conditioned) because their Toffolis only count in the shots where they fire."""
    t0 = time.time()
    k = K.read(str(path))
    t = k.op_types
    n_ops = len(t)
    base = t & np.uint8(0x7F)
    flagged = ((t & np.uint8(0x80)) != 0) & (t != np.uint8(K.PUSH_CONDITION))
    cond = flagged | (K.condition_depth(t) > 0)
    ops, qs, (i3, i2, i1) = K.op_qubits(k)
    n3, n2, n1 = len(i3), len(i2), len(i1)
    # role per operand, in op_qubits order: qqq0, qqq1, qqq2, qq0, qq1, q0
    kind = np.full(len(ops), READ, np.int8)
    b3, b2, b1 = base[i3], base[i2], base[i1]
    o = 2 * n3
    kind[o:o + n3][b3 == OPC['CCX']] = W_CCX                       # CCX target slot
    o = 3 * n3 + n2
    kind[o:o + n2][b2 == OPC['CX']] = W_CX                         # CX target slot
    o = 3 * n3 + 2 * n2
    kind[o:o + n1][b1 == OPC['X']] = W_X
    kind[o:o + n1][b1 == OPC['HMR']] = MEAS
    is_cond = cond[ops]
    if conditioned:
        kind[is_cond & (kind == MEAS)] = READ                      # the HMR itself must be unconditioned
    else:
        kind[is_cond] = READ                                       # anything conditioned blocks
    zeroing = np.zeros(len(ops), bool)
    zeroing[o:o + n1] = (b1 == OPC['HMR']) | (b1 == OPC['R'])
    order = np.lexsort((ops, qs))
    q_s, op_s, k_s, z_s, c_s = qs[order], ops[order], kind[order], zeroing[order], is_cond[order]
    # segments: break after a READ or MEAS, and at a new wire; each segment = writes... + terminator
    start = np.ones(len(q_s), bool)
    start[1:] = (q_s[1:] != q_s[:-1]) | (k_s[:-1] == READ) | (k_s[:-1] == MEAS)
    seg = np.cumsum(start) - 1
    nseg = int(seg[-1]) + 1 if len(seg) else 0
    last = np.r_[np.flatnonzero(start)[1:] - 1, len(q_s) - 1] if nseg else np.array([], int)
    first = np.flatnonzero(start)
    ends_meas = k_s[last] == MEAS
    cnt = {nm: np.bincount(seg, weights=(k_s == code), minlength=nseg).astype(np.int64)
           for nm, code in (('ccx', W_CCX), ('cx', W_CX), ('x', W_X))}
    writes = cnt['ccx'] + cnt['cx'] + cnt['x']
    ccx_c = np.bincount(seg, weights=(k_s == W_CCX) & c_s, minlength=nseg).astype(np.int64)
    site = np.flatnonzero(ends_meas & (writes > 0))
    rows = [dict(wire=int(q_s[first[s]]), first_write=int(op_s[first[s]]), hmr=int(op_s[last[s]]),
                 ccx=int(cnt['ccx'][s]), ccx_conditioned=int(ccx_c[s]), cx=int(cnt['cx'][s]), x=int(cnt['x'][s]))
            for s in site]
    res = dict(circuit=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), num_qubits=k.num_qubits,
               ops=n_ops, conditioned_writes_allowed=conditioned, sites=len(rows),
               ccx_sites=sum(1 for r in rows if r['ccx']),
               toffoli_saving_static=sum(r['ccx'] - r['ccx_conditioned'] for r in rows),
               toffoli_conditioned=sum(r['ccx_conditioned'] for r in rows), cx_to_z=sum(r['cx'] for r in rows),
               rows=rows)
    if width:
        res.update(_width(n_ops, q_s, op_s, z_s, rows))
    res['seconds'] = round(time.time() - t0, 1)
    return res


def _width(n_ops, q_s, op_s, z_s, rows):
    """Live-interval width profile before/after freeing every run's wire on [first_write, hmr)."""
    diff = np.zeros(n_ops + 1, np.int32)
    # live interval: first touch after a zeroing (or wire start) .. the next zeroing op (inclusive)
    newwire = np.ones(len(q_s), bool)
    newwire[1:] = q_s[1:] != q_s[:-1]
    after_zero = np.zeros(len(q_s), bool)
    after_zero[1:] = z_s[:-1] & ~newwire[1:]
    begin = newwire | after_zero
    bidx = np.flatnonzero(begin)
    # each interval ends at the last element before the next begin
    eidx = np.r_[bidx[1:] - 1, len(q_s) - 1]
    np.add.at(diff, op_s[bidx], 1)
    np.add.at(diff, op_s[eidx] + 1, -1)
    prof = np.cumsum(diff[:-1])
    peak = int(prof.max())
    peak_ops = int((prof == peak).sum())
    d2 = np.zeros(n_ops + 1, np.int32)
    over = 0
    for r in rows:
        a, b = r['first_write'], r['hmr']
        d2[a] -= 1
        d2[b] += 1
        if prof[a:b].max(initial=0) == peak:
            over += 1
    prof2 = prof + np.cumsum(d2[:-1])
    peak2 = int(prof2.max())
    return dict(live_peak=peak, live_peak_ops=peak_ops, runs_overlapping_peak=over, live_peak_after=peak2,
                live_peak_ops_after=int((prof2 == peak2).sum()))


def plan(path, rows, include_cx_only=False):
    """Edit script for the measure-first rewrite of each run (exact for every input and outcome).

    For a run on wire w from its first write s to its HMR at h (bit b): insert `HMR qw bb` before s; replace each write
    in place (CCX qa qx qw -> CZ qa qx if bb; CX qc qw -> Z qc if bb; X qw -> NEG if bb), so every fix-up sees its
    controls at the time of the original write; delete the HMR at h. A run is refused if any op strictly between s
    and h reads or writes bit b (moving the measurement would change what that op sees). Conditioned runs are not
    planned. Runs without a Toffoli save nothing and are skipped unless include_cx_only."""
    k = K.read(str(path))
    dec = K.Decoder(k)
    t = k.op_types.astype(np.uint8)
    cond = (((t & np.uint8(0x80)) != 0) & (t != np.uint8(K.PUSH_CONDITION))) | (K.condition_depth(k.op_types) > 0)
    hb, hc = (t & 0x40) > 0, (t & 0x80) > 0
    bit_w = np.full(len(t), -1, np.int64); bit_w[hb] = k.stream('b0')
    bit_c = np.full(len(t), -1, np.int64); bit_c[hc] = k.stream('bc')
    ops, qs, _ = K.op_qubits(k)
    lines, done, refused = [], [], []
    for r in rows:
        if r.get('ccx_conditioned') or (not r['ccx'] and not include_cx_only):
            continue
        w, s, h = r['wire'], r['first_write'], r['hmr']
        hmr = dec.text(h).split()
        if hmr[0] != 'HMR' or hmr[1] != f'q{w}' or cond[h]:
            refused.append(dict(r, reason=f'op {h} is not an unconditioned HMR on q{w}: {dec.text(h)}')); continue
        b = int(hmr[2][1:])
        inside = slice(s + 1, h)
        if (bit_w[inside] == b).any() or (bit_c[inside] == b).any():
            refused.append(dict(r, reason=f'bit b{b} is used between op {s} and op {h}')); continue
        js = np.unique(ops[(qs == w) & (ops >= s) & (ops < h)])
        edits = [f'insert {s} => HMR q{w} b{b}']
        ok = True
        for j in js:
            j = int(j); op = dec.text(j).split()
            if cond[j]:
                ok = False; break
            if op[0] == 'CCX' and op[3] == f'q{w}':
                edits.append(f'replace {j} => CZ {op[1]} {op[2]} if b{b}')
            elif op[0] == 'CX' and op[2] == f'q{w}':
                edits.append(f'replace {j} => Z {op[1]} if b{b}')
            elif op[0] == 'X' and op[1] == f'q{w}':
                edits.append(f'replace {j} => NEG if b{b}')
            else:
                ok = False; break
        if not ok:
            refused.append(dict(r, reason=f'op on q{w} in the run is not a plain write')); continue
        edits.append(f'delete {h}')
        lines += [f'# measure-first: q{w}, ops {s}..{h}, bit b{b}: {r["ccx"]} CCX -> CZ, {r["cx"]} CX -> Z, {r["x"]} X -> sign']
        lines += edits
        done.append(dict(r, bit=b))
    return '\n'.join(lines) + '\n', done, refused


def apply_rewrite(path, out, rows, include_cx_only=False):
    """Plan and write the rewritten circuit through `qhdl edit`; returns a summary."""
    import qhdl_edit as ED
    script, done, refused = plan(path, rows, include_cx_only)
    if not done:
        return dict(applied=0, refused=refused, out=None)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + '.edits.txt').write_text(script)
    summary = ED.edit_file(path, script, out)
    return dict(applied=len(done), toffolis_removed=sum(r['ccx'] for r in done), refused=refused, out=str(out),
                edit_script=str(out) + '.edits.txt', **{k_: v for k_, v in summary.items() if k_ in ('old_sha256', 'new_sha256', 'ops_before', 'ops_after')})


def v_measure_first(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    meta = api.ensure_open(a.circuit)
    sha, s12 = meta['sha256'], meta['sha12']
    src = api.circuit_path(meta)
    d = api.ws_dir(sha) / 'measure_first'
    ev = d / 'measure_first.json'
    key = dict(tool_sha=api.tool_sha(), verb='measure-first')
    res = api.cached(ev, key)
    if res is None:
        res = scan(src)
        res['cache_key'] = key
        d.mkdir(parents=True, exist_ok=True)
        api.write_json(ev, res)
    led.earn('static', 'exact pattern match on the op stream (write-only wire runs ending in an unconditioned HMR); '
                       'the rewrite is an identity for every input and outcome; width is a static interval model')
    claim = (f"{s12}: {res['sites']} write-only runs ending in an HMR, {res['ccx_sites']} with Toffolis; "
             f"measure-first saves {res['toffoli_saving_static']} static Toffolis ({res['cx_to_z']} CX become "
             f"conditioned Z). Live-width peak {res.get('live_peak')} -> {res.get('live_peak_after')} if every run's "
             f"wire is freed at its first write ({res.get('runs_overlapping_peak')} runs touch a peak op). Static model")
    if getattr(a, 'apply', False):
        if not a.out:
            raise api.QhdlError('measure-first --apply needs --out X.kmb', 'add --out')
        app = apply_rewrite(src, a.out, res['rows'], a.include_cx_only)
        led.earn('static', 'rewrite emitted through qhdl edit; not yet certified (run the task certificate)')
        claim = (f"{s12}: measure-first applied to {app['applied']} run(s), {app.get('toffolis_removed', 0)} static "
                 f"Toffolis removed, {len(app['refused'])} refused; wrote {app['out']} (uncertified)")
        nxt = [api.cmd('fdiff', '-c', s12, '-c2', app['out'], '--whole')] if app['out'] else []
        return E.Envelope(led, claim, tier='static', domain='edit plan from the exact pattern match',
                          evidence=app.get('edit_script') or api.rel(ev), rerun=api.cmd('measure-first', '-c', s12,
                          '--apply', '--out', a.out), tool_sha=api.tool_sha(), circuit_sha=sha, facts=app,
                          rows=app['refused'][:20], parts=[], next_cmds=nxt)
    rows = sorted(res['rows'], key=lambda r: -r['ccx'])[: (None if a.all else 20)]
    return E.Envelope(led, claim, tier='static', domain='whole circuit, static', evidence=api.rel(ev),
                      rerun=api.cmd('measure-first', '-c', s12), tool_sha=api.tool_sha(), circuit_sha=sha,
                      facts={k_: v for k_, v in res.items() if k_ != 'rows'}, rows=rows, parts=[], next_cmds=[])


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('measure-first', help='write-only wire runs before an HMR: Toffolis that become '
                                                    'measurement-conditioned CZs (and width freed)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--circuit', required=True, help='.kmb path or sha prefix')
    p.add_argument('--all', action='store_true', help='print every row')
    p.add_argument('--apply', action='store_true', help='write the rewritten circuit (runs with a Toffoli) to --out')
    p.add_argument('--out', help='output .kmb for --apply')
    p.add_argument('--include-cx-only', action='store_true', help='with --apply, also rewrite runs without a Toffoli')
    p.set_defaults(fn=lambda a: v_measure_first(api, a))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('circuits', nargs='+')
    ap.add_argument('--json', type=Path, help='write all results (rows included) to this new file')
    ap.add_argument('--no-width', action='store_true')
    ap.add_argument('--conditioned', action='store_true', help='let conditioned writes continue a run (tallied apart)')
    a = ap.parse_args(argv)
    out = []
    for c in a.circuits:
        r = scan(c, width=not a.no_width, conditioned=a.conditioned)
        out.append(r)
        print(json.dumps({k_: v for k_, v in r.items() if k_ != 'rows'}), flush=True)
    if a.json:
        if a.json.exists():
            raise SystemExit(f'{a.json} exists; outputs are never overwritten')
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
