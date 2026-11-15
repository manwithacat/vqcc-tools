#!/usr/bin/env python3
"""qhdl measure-out: rewrite class that removes a product by measurement instead of a cancelling Toffoli.

    qhdl measure-out -c <circuit> --out X.kmb [--region NAME | --range A:B] [--sites i,j,..] [--dry-run]
    python3 scripts/qhdl_measure_out.py -c <circuit> --out X.kmb ...        (same, standalone)

Sites come from `qhdl retain` (scripts/qhdl_retain.py): its `measure-out` class only; `retain` rows are skipped (they
need a product kept alive across HMRs and are over-flagged). A site is a pair of unconditional Toffolis with equal
control forms, `on` = CCX(c1, x1 -> t1) that puts the product p = f(c1)·f(x1) into t1, and `off` = CCX(c2, x2 -> t2)
that cancels it (equal control forms, so p is the same function). Two exact rewrites, no new qubits:

  mixed target (needs a spare wire s: |0> and untouched from `on` through `off`, found by `qhdl retain`)
      on : CCX c1 x1 t1          ->  CCX c1 x1 s ; CX s t1             s now holds p, t1 gets p as before
      off: CCX c2 x2 t2          ->  CX s t2 ; HMR s m ; CZ c2 x2 if m   t2 loses p; s measured in the X basis (and
                                                                         reset to |0>); the (-1)^(m·p) phase is
                                                                         repaired by CZ on the wires whose forms
                                                                         multiply to p
  pure target (t2 holds exactly p; no spare needed)
      off: CCX c2 x2 t2          ->  HMR t2 m ; CZ c2 x2 if m

One Toffoli less per site, one HMR and one classically conditioned CZ more (Clifford, free), Q unchanged. Spare wires
are booked greedily so overlapping sites never share one (`qhdl_retain.pack`). Skipped and counted, never guessed: sites
whose `on` op is shared by several `off` ops, CSWAP products, ops under a condition. The tier of everything printed is
*static*; a multi-draw DQI check (e.g. scripts/dqi_fast_rank.py) and the unchanged local official verifier are the
next steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_retain as RT  # noqa: E402

CCX = 0x31


def _ops_index(k):
    import qhdl_rewrite as RW
    return RW.Ops(k)


def select(k, sites, ops=None):
    """Measure-out candidates among `sites` (output of qhdl_retain.analyse for one region), packed so that no spare
    wire is booked twice. Returns (chosen site dicts with .plan filled in, skipped Counter)."""
    skipped = Counter()
    cand = []
    by_on = Counter(s['since'] for s in sites if s['kind'] == 'measure-out')
    for s in sites:
        if s['kind'] != 'measure-out':
            continue
        if not s['free']:
            skipped['no_free_spare'] += 1
            continue
        if s.get('cswap'):
            skipped['cswap'] += 1
            continue
        if by_on[s['since']] > 1:
            skipped['shared_on_op'] += 1
            continue
        cand.append(s)
    if not cand:
        return [], skipped
    ops = ops or _ops_index(k)
    need = sorted({s['op'] for s in cand} | {s['since'] for s in cand})
    rec = ops.records(need)
    ok = []
    for s in cand:
        on, off = rec[s['since']], rec[s['op']]
        if on[0] != CCX or off[0] != CCX:        # conditional or not a plain CCX
            skipped['not_plain_ccx'] += 1
            continue
        s['_on'], s['_off'] = on[1], off[1]
        ok.append(s)
    # greedy interval booking of spare wires; pure targets need none. Sort by end op (interval scheduling).
    taken = []                                    # (end, wire, start)
    chosen = []
    for s in sorted(ok, key=lambda s: (s['op'], s['since'])):
        if s.get('pure'):
            s['_spare'] = None
            chosen.append(s)
            continue
        busy = {w for (e, w, st) in taken if e >= s['since'] and st <= s['op']}
        w = next((x for x in s['_w'] if x not in busy), None)
        if w is None:
            skipped['spare_booked'] += 1
            continue
        taken.append((s['op'], w, s['since']))
        s['_spare'] = w
        chosen.append(s)
    return chosen, skipped


def edit_lines(sites, first_bit):
    """Edit-script lines (qhdl_edit syntax) for the chosen sites; bits numbered from first_bit. -> (lines, n_bits)"""
    lines = []
    b = first_bit
    for s in sites:
        c2, x2, t2 = s['_off']
        if s['_spare'] is None:
            lines.append(f"# site off={s['op']} pure target q{t2}")
            lines.append(f"replace {s['op']} => HMR q{t2} b{b}; CZ q{c2} q{x2} if b{b}")
        else:
            c1, x1, t1 = s['_on']
            sp = s['_spare']
            lines.append(f"# site on={s['since']} off={s['op']} spare q{sp}")
            lines.append(f"replace {s['since']} => CCX q{c1} q{x1} q{sp}; CX q{sp} q{t1}")
            lines.append(f"replace {s['op']} => CX q{sp} q{t2}; HMR q{sp} b{b}; CZ q{c2} q{x2} if b{b}")
        b += 1
    return lines, b - first_bit


def counts_of(k):
    import numpy as np
    t = np.asarray(k.op_types)
    base = t & 0x7F
    plain = (t & 0x80) == 0
    return dict(ops=int(len(t)), toffoli_plain=int(np.count_nonzero(plain & ((base == 0x31) | (base == 0x32)))),
                toffoli_conditional=int(np.count_nonzero((~plain) & ((base == 0x31) | (base == 0x32)))),
                hmr=int(np.count_nonzero(base == 0x50)), ccz=int(np.count_nonzero(base == 0x30)),
                cz=int(np.count_nonzero(base == 0x20)), num_qubits=int(k.num_qubits), num_bits=int(k.num_bits))


def jobs_for(api, P, k, a):
    if a.range:
        lo, hi = (int(x) for x in a.range.split(':'))
        return [(f'ops {lo}:{hi}', lo, hi)]
    row = P.get('manifest') or {}
    task = a.task or row.get('task') or 'unknown'
    if task == 'dqi':
        regs = RT.dqi_regions(api, P)
        want = [a.region] if a.region else [r for r in regs if r not in ('mask_projection', 'initial', 'uninitial', 'restore_masks')]
        for r in want:
            if r not in regs:
                raise api.QhdlError(f'region {r!r} not found; regions: {", ".join(sorted(regs))}')
        return [(r, regs[r][0], regs[r][1]) for r in want]
    return [('whole', 0, int(k.op_types.shape[0]))]


def run(api, a):
    import qhdl_edit as ED
    import qhdl_kmb as KM
    t0 = time.time()
    P = api.resolve(a.circuit)
    k = KM.read(str(P['path']))
    ops = None
    chosen_all, skipped_all, per = [], Counter(), []
    for label, lo, hi in jobs_for(api, P, k, a):
        r = RT.analyse(k, lo, hi)
        if ops is None:
            ops = _ops_index(k)
        chosen, sk = select(k, r['sites'], ops)
        for s in chosen:
            s['region'] = label
        chosen_all.extend(chosen)
        skipped_all.update(sk)
        per.append(dict(region=label, ops=[lo, hi], measure_out_sites=sum(1 for s in r['sites'] if s['kind'] == 'measure-out'),
                        chosen=len(chosen), skipped=dict(sk)))
    if a.sites:
        keep = {int(x) for x in a.sites.split(',')}
        chosen_all = [s for s in chosen_all if s['op'] in keep or s['since'] in keep]
    lines, nbits = edit_lines(chosen_all, int(k.num_bits))
    script = '# qhdl measure-out on %s: %d site(s); new bits b%d..b%d\n' % (P['sha'][:12], len(chosen_all), int(k.num_bits),
                                                                         int(k.num_bits) + nbits - 1) + '\n'.join(lines) + '\n'
    res = dict(schema='vqcc-qhdl-measure-out/1', tier='static', circuit=P['sha'][:12], circuit_sha256=P['sha'],
               sites_applied=len(chosen_all), pure=sum(1 for s in chosen_all if s['_spare'] is None),
               spare=sum(1 for s in chosen_all if s['_spare'] is not None), skipped=dict(skipped_all), per_region=per,
               before=counts_of(k), new_bits=nbits)
    if a.dry_run or not chosen_all:
        res['seconds'] = round(time.time() - t0, 1)
        return res, script
    out = Path(a.out)
    if out.exists():
        raise api.QhdlError(f'{out} exists; refusing to overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    r = ED.edit_file(P['path'], script, out, None)
    sp = out.with_name(out.name + '.edit.txt')
    sp.write_text(script)
    k2 = KM.read(str(out))
    res.update(out=str(out), out_sha256=r['new_sha'], map=r['map'], script=str(sp), after=counts_of(k2),
               added_qubits=r['added_qubits'])
    b, c = res['before'], res['after']
    res['delta'] = dict(toffoli_plain=c['toffoli_plain'] - b['toffoli_plain'], hmr=c['hmr'] - b['hmr'],
                        num_qubits=c['num_qubits'] - b['num_qubits'], ops=c['ops'] - b['ops'])
    res['sites'] = [dict(region=s['region'], on=s['since'], off=s['op'], spare=s['_spare'], pure=s['_spare'] is None)
                    for s in chosen_all]
    res['seconds'] = round(time.time() - t0, 1)
    (out.with_name(out.name + '.sites.json')).write_text(json.dumps(res, indent=1) + '\n')
    return res, script


def qhdl_register(subparsers, api):
    import qhdl_envelope as E
    p = subparsers.add_parser('measure-out', help='rewrite: a product removed by a cancelling CCX is removed by HMR plus a '
                                                  'conditioned CZ instead (sites from `qhdl retain`)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)

    def fn(a):
        if not a.circuit or not (a.out or a.dry_run):
            raise api.QhdlError('measure-out needs -c <circuit> and --out X.kmb (or --dry-run)')
        res, _ = run(api, a)
        led = E.Ledger()
        led.earn('static', 'netlist rewrite; no verifier run')
        d = res.get('delta') or {}
        claim = (f"{res['circuit']}: {res['sites_applied']} measure-out site(s) "
                 + (f"applied, Toffolis {d.get('toffoli_plain'):+d}, HMR {d.get('hmr'):+d}, qubits {d.get('num_qubits'):+d}"
                    if d else 'planned (dry run); nothing written') + '; unverified')
        return E.Envelope(led, claim, domain='netlist rewrite, static', evidence=res.get('map') or '', rerun=api.cmd('measure-out', '-c', res['circuit']),
                          tool_sha=api.tool_sha(), circuit_sha=res['circuit_sha256'], facts=res, rows=[], table='', next_cmds=[])
    p.set_defaults(fn=fn)


def add_args(p):
    p.add_argument('-c', '--circuit', help='sha prefix or .kmb path')
    p.add_argument('--task', choices=['dqi', 'invsqrt'])
    p.add_argument('--region', help='dqi: one top-level tag region (default: all but the I/O stubs)')
    p.add_argument('--range', help='op window A:B')
    p.add_argument('--sites', help='comma list of op indices (on or off) to keep')
    p.add_argument('--out', help='candidate .kmb path (edit script, map and sites JSON beside it)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--json', action='store_true', help='(envelope is always JSON)')
    p.add_argument('--all', action='store_true')


def _main(argv=None):
    """Standalone entry: qhdl.py re-executes itself isolated (-I), so hand the arguments to `qhdl.py measure-out`."""
    import os
    os.execv(sys.executable, [sys.executable, str(HERE / 'qhdl.py'), 'measure-out'] + list(sys.argv[1:] if argv is None else argv))


if __name__ == '__main__':
    _main()
