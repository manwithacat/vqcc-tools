#!/usr/bin/env python3
"""Iterate census -> certified park rewrite while each qubit removed costs less than it is worth.

Each round: run qhdl_width_census on the current .kmx (exact information test on), turn its parks into edits,
drop parks whose recompute reads a wire that another park holds erased at that moment, apply them with
qhdl_park_rewrite (bitset contract on all legal x in three measurement branches, then kickmix .kmb + allocator).
Stops when a peak run is blocked, the round's added Toffolis exceed T_eq/(Q-1) (the break-even for one qubit at
the current cost), or the allocator does not reach the lower width.

Usage: python qhdl_park_iterate.py CIRCUIT.kmx OUTDIR --teq T_EQ_OF_INPUT [--rounds 4]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_width_census as W  # noqa: E402


def edits_from_parks(parks):
    edits = []
    held = []  # (wire, a, r)
    merged = {}
    for p in parks:  # one park per (wire, gap): keep the latest recompute point (covers the earlier runs too)
        k = (p['w'], p['a'])
        if k not in merged or p['r'] > merged[k]['r']:
            merged[k] = p
    parks = list(merged.values())
    for p in sorted(parks, key=lambda p: p['a']):
        if p['kind'] == 'dead-tail':
            e = dict(op='advance_hmr', b=p['b'], a=p['a'])
            win = (p['w'], p['a'], p['b'])
            reads = set()
        else:
            terms = [tuple(t) for t in (p['terms'] or [])]
            e = dict(op='park', w=p['w'], a=p['a'], r=p['r'], terms=terms)
            win = (p['w'], p['a'], p['r'])
            reads = {v for t in terms for v in t if v != '1'}
        clash = False
        for (hw, ha, hr) in held:
            if hw == win[0] and not (win[2] <= ha or win[1] >= hr):
                clash = True  # same wire, overlapping windows
            if f'q{hw}' in reads and ha < win[2] <= hr:
                clash = True  # recompute reads an erased wire
        if clash:
            continue
        # a later park's erased window may contain this recompute: checked when that park is considered
        later_reads_ok = True
        if later_reads_ok:
            edits.append(e)
            held.append(win)
    # second pass: drop recomputes reading wires erased by parks that start before them
    final = []
    for e in edits:
        if e['op'] != 'park':
            final.append(e)
            continue
        reads = {v for t in e['terms'] for v in t if v != '1'}
        bad = any(f"q{o['w']}" in reads and o['a'] < e['r'] <= o['r'] for o in edits if o is not e and o['op'] == 'park')
        if not bad:
            final.append(e)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmx')
    ap.add_argument('outdir')
    ap.add_argument('--teq', type=float, required=True, help='Toffoli-equivalent of the input (from a verifier run)')
    ap.add_argument('--rounds', type=int, default=4)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cur = a.kmx
    teq = a.teq
    log = []
    for rnd in range(1, a.rounds + 1):
        rep, per_run, chosen, runs = W.run_census(cur, exact_info=True)
        (out / f'round{rnd}_census.json').write_text(json.dumps(dict(report=rep, parks=chosen, per_run=per_run),
                                                                   indent=1, default=str))
        q = rep['peak']
        breakeven = teq / (q - 1)
        entry = dict(round=rnd, input=cur, peak=q, census=rep, breakeven_T=breakeven)
        log.append(entry)
        if rep['runs_blocked']:
            entry['stop'] = f"{rep['runs_blocked']} peak runs blocked"
            break
        edits = edits_from_parks(chosen)
        (out / f'round{rnd}_edits.json').write_text(json.dumps(edits, indent=1))
        name = f'round{rnd}'
        r = subprocess.run([sys.executable, str(HERE / 'qhdl_park_rewrite.py'), cur, str(out / f'{name}_edits.json'),
                            str(out / name), '--name', name, '--greedy'], capture_output=True, text=True)
        rp = out / name / f'{name}_report.json'
        if not rp.exists():
            entry['stop'] = 'rewrite failed: ' + r.stderr[-500:]
            break
        rr = json.loads(rp.read_text())
        entry['rewrite'] = dict(check_pass=rr['check']['pass'], kmb=rr.get('kmb'), kmb_sha256=rr.get('kmb_sha256'),
                                qubits=rr.get('alloc', {}).get('num_qubits_out'),
                                added_toffolis=sum(e.get('toffolis', 0) for e in rr['edits']),
                                mean_toffoli=rr['check'].get('101', {}).get('mean_toffoli'))
        if not rr['check']['pass']:
            entry['stop'] = 'bitset contract failed'
            break
        added = entry['rewrite']['added_toffolis']
        if added > breakeven:
            entry['stop'] = f'added {added} T > break-even {breakeven:.1f}'
            break
        if entry['rewrite']['qubits'] is None or entry['rewrite']['qubits'] >= q:
            entry['stop'] = 'allocator did not reach peak-1'
            break
        teq = teq + added
        cur = rr['kmx']
        (out / 'iterate_log.json').write_text(json.dumps(log, indent=1, default=str))
    (out / 'iterate_log.json').write_text(json.dumps(log, indent=1, default=str))
    for e in log:
        print(e['round'], e['peak'], e['census']['census_total_T_priced_runs'], e['census']['runs_blocked'],
              e.get('rewrite'), e.get('stop'))


if __name__ == '__main__':
    main()
