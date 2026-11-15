#!/usr/bin/env python3
"""qhdl retain: retained-product detector (avenues G3 repetition, priced).

    qhdl retain -c <circuit> [--region NAME | --range A:B] [--top N] [--out DIR] [--no-gen]

A *retained product* is a nonlinear value p = f(a)·g(b) (an AND/CCX product, or the c·(a xor b) term of a CSWAP) that is
removed from the state (HMR/reset of the wire holding it, or a second CCX that cancels it) and then computed again with
identical control values later, while the circuit never needed the qubit that held it in between. Keeping it alive
across the gap removes the recompute (and the cancelling CCX, if the removal was a Toffoli).

Method (static, exact for the netlist, no sampling): every wire carries a GF(2)-affine form over symbolic atoms (one atom
per wire at the start of the scanned window, one per distinct product key). CX / SWAP / X update forms exactly; a CCX
whose two control forms are equal (as an unordered pair) to an earlier one has the same product atom, so the matching is
robust to the CX frames that surround each product in these circuits. Classically conditioned data-changing ops give the
touched wires a fresh unknown atom (conservative). A CCX is classified as
  first     the key was not seen before;
  off       the target already holds the product: this CCX cancels it (an uncompute by Toffoli);
  copy      the product is held by another wire right now (a duplicate; a CX could replace the Toffoli);
  recompute the key was seen, and no wire holds it now: the retention candidate.
For every recompute the gap starts where the product was removed (HMR/R of its wire, or an `off` CCX, else the previous
event). The site is `free` when that wire is not touched at all between removal and recompute (so retention needs no
extra qubit), else `reused` (a width check is needed; not counted as free). Price: 1 Toffoli for the recompute plus 1
when the removal was a CCX; conditional sites are priced separately because executed cost is a per-shot mean.

Tier: static (netlist counts). Nothing is emitted and nothing is verified: a flagged site is a candidate, and keeping a
product alive also moves phase corrections (the HMR fix-ups), which only a verifier run can check.
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
import qhdl_envelope as E  # noqa: E402

CX, CCX, CSWAP, SWAP, X, R, HMR, CZ = 0x21, 0x31, 0x32, 0x22, 0x11, 0x12, 0x50, 0x20
PUSH, POP = 0x81, 0x01
MAX_SITES_KEPT = 200_000


def _np():
    import numpy as np
    return np


def op_arrays(k, a, b):
    """Per-op lists for ops [a, b): type, and the operand tuples (3Q / 2Q / 1Q) in stream order."""
    np = _np()
    t_all = k.op_types
    pre = np.asarray(t_all[:a]) & 0x30
    t = np.asarray(t_all[a:b])
    c3, c2, c1 = int(np.count_nonzero(pre == 0x30)), int(np.count_nonzero(pre == 0x20)), int(np.count_nonzero(pre == 0x10))
    q = t & 0x30
    n3, n2, n1 = int(np.count_nonzero(q == 0x30)), int(np.count_nonzero(q == 0x20)), int(np.count_nonzero(q == 0x10))
    s = {n: np.asarray(k.stream(n)) for n in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0')}
    return (t.tolist(), list(zip(s['qqq0'][c3:c3 + n3].tolist(), s['qqq1'][c3:c3 + n3].tolist(), s['qqq2'][c3:c3 + n3].tolist())),
            list(zip(s['qq0'][c2:c2 + n2].tolist(), s['qq1'][c2:c2 + n2].tolist())), s['q0'][c1:c1 + n1].tolist())


CHUNK = 1_500_000


def ops_stream(k, a, b):
    """Yield (op index, type, operands) for ops [a, b) in chunks (bounded memory): a 3/2-tuple of wires, a wire, or None."""
    for ca in range(a, b, CHUNK):
        cb = min(b, ca + CHUNK)
        types, q3, q2, q1 = op_arrays(k, ca, cb)
        i3 = i2 = i1 = 0
        for j, t in enumerate(types):
            c = t & 0x30
            if c == 0x30:
                q = q3[i3]; i3 += 1
            elif c == 0x20:
                q = q2[i2]; i2 += 1
            elif c == 0x10:
                q = q1[i1]; i1 += 1
            else:
                q = None
            yield ca + j, t, q


def scan(k, a, b, watch=None, seed=20261112, trace=None):
    """Scan ops [a, b) of one circuit; returns dict(sites, counts). Pure function of the netlist.

    Identity of a form is a 128-bit Zobrist hash (XOR of one random value per atom: exactly linear, so CX/SWAP/X update
    it exactly; collisions have probability about 2^-128 per comparison). Membership of a *product* atom in a wire's
    form (needed for off/copy/recompute) is tracked by a second GF(2) bit-vector over product atoms only.

    `watch` (second pass): {'start': {op: [site ids]}, 'end': {op: [site ids]}}. At the state just after op `start`
    every wire that is known zero is recorded with its touch counter; at op `end` (before it runs) the wires still
    untouched are the spare qubits that were free across the whole interval: result['spare'][site id] = sorted wires."""
    import random
    rng = random.Random(seed)
    nq = int(k.num_qubits)
    n_ops = 0
    rnd = lambda: rng.getrandbits(128) | 1
    CONST = rnd()
    hs = [rnd() for _ in range(nq)]                      # form hash per wire: starts as the wire's own atom
    pb = [0] * nq                                         # product-atom membership per wire (bit i = product atom i)
    keys = {}                                             # key (ha, hb) -> product index
    zob = []                                              # product index -> zobrist value
    hist = []                                             # product index -> dict
    gen = [0] * nq
    depth = 0
    sites = []
    cnt = Counter()
    hmr_deaths = 0
    ws = (watch or {}).get('start', {})
    we = (watch or {}).get('end', {})
    snaps = {}
    spare = {}

    def holders(bit):
        return [w for w in range(nq) if pb[w] & bit]

    def add(**kw):
        if len(sites) < MAX_SITES_KEPT:
            sites.append(kw)

    for op, t, qv in ops_stream(k, a, b):
        n_ops += 1
        if ws and (op - 1) in ws:                       # state just after op-1
            z = {w: gen[w] for w in range(nq) if hs[w] == 0}
            for sid in ws[op - 1]:
                snaps[sid] = z
        if we and op in we:
            for sid in we[op]:
                z = snaps.get(sid, {})
                spare[sid] = [w for w, g in z.items() if gen[w] == g]
        base_t = t & 0x7F
        cls = t & 0x30
        cond = bool(t & 0x80) or depth > 0
        if t == PUSH:
            depth += 1
            continue
        if t == POP:
            depth -= 1
            continue
        if cls == 0x30:
            c, x, y = qv
            gen[c] += 1; gen[x] += 1; gen[y] += 1
            if base_t == 0x30:                          # CCZ: phase only
                continue
            if cond:
                cnt['conditional_toffoli'] += 1
                if base_t == 0x31:
                    hs[y] ^= rnd()
                else:
                    u = rnd(); hs[x] ^= u; hs[y] ^= u
                continue
            if base_t == 0x31:                          # CCX(c, x -> y)
                fa, fb, tgt = hs[c], hs[x], (y,)
            else:                                       # CSWAP(c; x, y): product c*(x ^ y) into both
                fa, fb, tgt = hs[c], hs[x] ^ hs[y], (x, y)
            kk = (fa, fb) if fa <= fb else (fb, fa)
            cnt['toffoli'] += 1
            idx = keys.get(kk)
            if idx is None:
                idx = len(zob)
                keys[kk] = idx
                z = rnd()
                zob.append(z)
                hist.append(dict(on_op=op, last_op=op, last_kind='first', death=None, n=1))
                bit = 1 << idx
                for w in tgt:
                    hs[w] ^= z
                    pb[w] ^= bit
                cnt['first'] += 1
                if trace is not None:
                    trace.append((op, idx, 'first'))
                continue
            bit, z, h = 1 << idx, zob[idx], hist[idx]
            h['n'] += 1
            if pb[tgt[0]] & bit:                        # off: cancels the product held by the target
                pure = len(tgt) == 1 and hs[tgt[0]] == z
                for w in tgt:
                    hs[w] ^= z
                    pb[w] ^= bit
                if trace is not None:
                    trace.append((op, idx, 'off'))
                cnt['off'] += 1
                add(kind='measure-out', op=op, key=idx, since=h['on_op'], gap_ops=op - h['on_op'], pure=pure, price=1,
                    wire=tgt[0], target_form='pure' if pure else 'mixed', cswap=len(tgt) == 2)
                h.update(last_op=op, last_kind='off', death=(op, tgt[0], gen[tgt[0]]))
                continue
            held = holders(bit)
            for w in tgt:
                hs[w] ^= z
                pb[w] ^= bit
            if trace is not None:
                trace.append((op, idx, 'copy' if held else 'recompute'))
            if held:                                    # copy: some other wire holds it right now
                cnt['copy'] += 1
                add(kind='copy', op=op, key=idx, since=h['last_op'], gap_ops=op - h['last_op'], pure=hs[held[0]] == z,
                    price=1, holder=held[0])
                h['last_op'] = op
                continue
            d = h['death']
            removal = 'ccx' if (d and h['last_kind'] == 'off') else ('hmr' if d else 'other')
            since = d[0] if d else h['last_op']
            cnt['recompute'] += 1
            if d is not None:
                add(kind='retain', op=op, key=idx, since=since, gap_ops=op - since, removal=removal,
                    price=1 + (1 if removal == 'ccx' else 0), prior_uses=h['n'] - 1,
                    holder_wire=d[1])
            else:
                cnt['recompute_no_removal_seen'] += 1
            h.update(last_op=op, last_kind='on', death=None, on_op=op)
            continue
        if cls == 0x20:
            c, x = qv
            gen[c] += 1; gen[x] += 1
            if base_t == 0x20:
                continue
            if base_t == CX:
                if cond:
                    hs[x] ^= rnd()
                else:
                    hs[x] ^= hs[c]
                    pb[x] ^= pb[c]
            elif base_t == SWAP:
                if cond:
                    hs[c] ^= rnd(); hs[x] ^= rnd()
                else:
                    hs[c], hs[x] = hs[x], hs[c]
                    pb[c], pb[x] = pb[x], pb[c]
            continue
        if cls == 0x10:
            w = qv
            gen[w] += 1
            if base_t == HMR or base_t == R:
                if cond:
                    hs[w] ^= rnd()
                    continue
                q = pb[w]
                while q:                                # record a death for every product atom this wire held
                    low = q & -q
                    h = hist[low.bit_length() - 1]
                    if h['death'] is None or h['last_kind'] != 'off':
                        h['death'] = (op, w, gen[w])
                        h['last_kind'] = 'hmr'
                    hmr_deaths += 1
                    q ^= low
                hs[w] = 0                               # a reset wire is |0>: the empty form
                pb[w] = 0
            elif base_t == X:
                hs[w] ^= rnd() if cond else CONST
            continue
    return dict(sites=sites, counts=dict(cnt), hmr_deaths=hmr_deaths, ops=n_ops, products=len(zob), spare=spare)


def analyse(k, a, b, seed=20261112):
    """Two passes: find candidate sites, then find which have a spare (zero, untouched) qubit across the whole interval.
    A `measure-out` site on a pure target needs no spare qubit. Returns dict(sites, counts, ...)."""
    r1 = scan(k, a, b, seed=seed)
    start, end = {}, {}
    for sid, s in enumerate(r1['sites']):
        s['id'] = sid
        if s['kind'] == 'copy' or (s['kind'] == 'measure-out' and s['pure']):
            continue
        st = s['since']
        if st is None:
            continue
        start.setdefault(st, []).append(sid)
        end.setdefault(s['op'], []).append(sid)
    r2 = scan(k, a, b, watch=dict(start=start, end=end), seed=seed)
    sp = r2['spare']
    for s in r1['sites']:
        w = sp.get(s['id'])
        if s['kind'] == 'copy' or (s['kind'] == 'measure-out' and s['pure']):
            s['spare'] = None
            s['free'] = bool(s['pure']) if s['kind'] == 'measure-out' else bool(s.get('pure'))
        else:
            s['spare'] = len(w) if w is not None else None
            s['free'] = bool(w)
            s['_w'] = w
    return r1


def pack(sites):
    """Greedy interval packing: each chosen site takes one spare wire not already taken by an overlapping chosen site.
    Pure measure-outs and pure copies need no wire. Returns (chosen ids, total price)."""
    need = sorted((s for s in sites if s['free'] and s.get('_w')), key=lambda s: (s['op'], s['since']))
    taken = []                                          # (end_op, wire)
    chosen = []
    for s in need:
        busy = {w for (e, w, st) in taken if e >= s['since'] and st <= s['op']}
        w = next((x for x in s['_w'] if x not in busy), None)
        if w is None:
            continue
        taken.append((s['op'], w, s['since']))
        chosen.append(s['id'])
    chosen += [s['id'] for s in sites if s['free'] and not s.get('_w')]
    price = sum(s['price'] for s in sites if s['id'] in set(chosen))
    return chosen, price


# ---------------------------------------------------------------- verb
TOP_REGIONS_MAX = 40


def dqi_regions(api, P):
    """Top-level tag regions {path: (start, end)} from a cached `workbench summarize` tree (work/_diff/regions), else computed once via qhdl_diff.regions_for."""
    sha12 = P['sha'][:12]
    for c in (Path(api.WORK) / '_diff' / 'regions' / f'{sha12}.json',):
        if c.exists():
            tree = json.loads(c.read_text())
            break
    else:
        import qhdl_diff as DF
        tree, _ = DF.regions_for(api, P['path'], sha12)
    return {r['path']: (r['start'], r['end']) for r in tree['tag_regions'] if r['depth'] == 0}


def summarise(sites, a, b, tag):
    ch, packed_price = pack(sites)
    cls = Counter()
    for s in sites:
        cls[(s['kind'], 'free' if s['free'] else 'blocked')] += 1
    price = lambda kind: sum(s['price'] for s in sites if s['kind'] == kind and s['free'])
    return dict(region=tag, ops=[a, b], sites={f'{k}/{f}': v for (k, f), v in sorted(cls.items())},
                free_price=dict(measure_out=price('measure-out'), retain=price('retain'), copy=price('copy')),
                free_total=price('measure-out') + price('retain') + price('copy'), packed_total=packed_price)


def public_site(s):
    d = {k: v for k, v in s.items() if not k.startswith('_')}
    return d


def run(api, a):
    t0 = time.time()
    import qhdl_kmb as KM
    P = api.resolve(a.circuit)
    row = P.get('manifest') or {}
    task = a.task or row.get('task') or 'unknown'
    k = KM.read(str(P['path']))
    jobs = []                                           # (label, kmb, lo, hi, multiplier note)
    if a.range:
        lo, hi = (int(x) for x in a.range.split(':'))
        jobs.append((f'ops {lo}:{hi}', k, lo, hi))
    elif task == 'dqi':
        regs = dqi_regions(api, P)
        want = [a.region] if a.region else [r for r in regs if r not in ('mask_projection', 'initial', 'uninitial', 'restore_masks')]
        for r in want:
            if r not in regs:
                raise api.QhdlError(f'region {r!r} not found; regions: {", ".join(sorted(regs))}')
            jobs.append((r, k, regs[r][0], regs[r][1]))
    elif task == 'chemistry':
        raise api.QhdlError('chemistry: pass --task chemistry and a family sha prefix', 'qhdl retain --task chemistry -c <family>')
    else:
        jobs.append(('whole', k, 0, int(k.op_types.shape[0])))
    per = []
    allsites = []
    for label, kk, lo, hi in jobs:
        r = analyse(kk, lo, hi)
        sm = summarise(r['sites'], lo, hi, label)
        sm['counts'] = r['counts']
        per.append(sm)
        for s in r['sites']:
            s = public_site(s)
            s['region'] = label
            allsites.append(s)
    return finish(api, a, t0, P, row, task, per, allsites)


def run_chem(api, a):
    t0 = time.time()
    import csv
    import qhdl_diff as DF
    import qhdl_kmb as KM
    stage = Path(a.stage) if a.stage else Path(api.DATASET)
    fam, slots, man = DF.load_chem(stage, a.circuit)
    per, allsites = [], []
    for s in slots:
        path = stage / s['dataset_path']
        kk = KM.read(str(path))
        n = int(kk.op_types.shape[0])
        r = analyse(kk, 0, n)
        sm = summarise(r['sites'], 0, n, f"slot{s['slot_k']}")
        sm['counts'] = r['counts']
        per.append(sm)
        for x in r['sites']:
            x = public_site(x)
            x['region'] = f"slot{s['slot_k']}"
            allsites.append(x)
    P = dict(sha=(man or {}).get('sha256') or fam['family'], path=stage)
    return finish(api, a, t0, P, man or {}, 'chemistry', per, allsites)


def finish(api, a, t0, P, row, task, per, allsites):
    led = E.Ledger()
    led.earn('static', 'netlist scan; no per-shot means, no verifier run')
    out_dir = Path(a.out) if a.out else Path(api.WORK) / '_retain' / P['sha'][:12]
    free = [s for s in allsites if s['free']]
    free.sort(key=lambda s: (-s['price'], s['gap_ops']))
    tot = sum(p['free_total'] for p in per)
    packed = sum(p['packed_total'] for p in per)
    by_kind = Counter()
    for s in free:
        by_kind[s['kind']] += s['price']
    full = dict(schema='vqcc-qhdl-retain/1', task=task, circuit=P['sha'][:12], circuit_sha256=P['sha'], tier='static',
                regions=per, free_total=tot, packed_total=packed, free_by_kind=dict(by_kind),
                sites=allsites, seconds=round(time.time() - t0, 1))
    ev = out_dir / 'retain.json'
    api.write_json(ev, full)
    unit = 'Toffolis per pass' if task == 'chemistry' else 'static Toffolis'
    claim = (f"{task} {P['sha'][:12]}: {tot} {unit} flagged on free-width sites ({packed} after spare-wire packing); "
             f"measure-out {by_kind['measure-out']}, retain {by_kind['retain']}, copy {by_kind['copy']}; candidates, unverified")
    rows = [dict(region=p['region'], free_total=p['free_total'], packed=p['packed_total'],
                 measure_out=p['free_price']['measure_out'], retain=p['free_price']['retain'], copy=p['free_price']['copy'],
                 blocked=sum(v for kx, v in p['sites'].items() if kx.endswith('/blocked'))) for p in per]
    rows.sort(key=lambda r: -r['free_total'])
    top_sites = [dict(region=s['region'], kind=s['kind'], op=s['op'], since=s['since'], gap=s['gap_ops'], price=s['price'],
                      spare=s.get('spare'), form=s.get('target_form', s.get('removal', '-'))) for s in free[:a.top]]
    facts = dict(free_total=tot, packed_total=packed, free_by_kind=dict(by_kind), regions=len(per),
                 flagged_sites=len(free), top_sites=top_sites[:5])
    return E.Envelope(led, claim, domain=f'{task} netlist, static counts', evidence=api.rel(ev),
                      rerun=api.cmd('retain', '-c', P['sha'][:12], *(['--region', a.region] if a.region else [])),
                      tool_sha=api.tool_sha(), circuit_sha=P['sha'], facts=facts, rows=rows, table=api.rel(ev), next_cmds=[])


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('retain', help='retained-product detector: products measured out or recomputed while a '
                                             'qubit is free (avenues G3, priced)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--circuit', help='sha prefix or .kmb path (chemistry: family sha prefix)')
    p.add_argument('--task', choices=['dqi', 'invsqrt', 'chemistry'])
    p.add_argument('--region', help='dqi: one top-level tag region (default: all but the I/O stubs)')
    p.add_argument('--range', help='op window A:B (state starts unknown at A)')
    p.add_argument('--stage', help='circuits dataset stage dir (chemistry)')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--out', help='directory for retain.json (default work/_retain/<sha12>/)')
    p.add_argument('--json', action='store_true', help='(envelope is always JSON)')
    p.add_argument('--all', action='store_true')

    def fn(a):
        if not a.circuit:
            raise api.QhdlError('retain needs -c <circuit>')
        if a.task == 'chemistry':
            return run_chem(api, a)
        return run(api, a)
    p.set_defaults(fn=fn)


# ---------------------------------------------------------------- positive control (parent -> child pair)
def control(api, parent, child, region, out):
    """Does the detector, run on the parent, account for the realised parent->child saving, and is the same saving no
    longer flagged on the child? Uses qhdl_diff's aligned call sites of `region` as the realised gain."""
    import qhdl_diff as DF
    import qhdl_kmb as KM
    P, X = api.resolve(parent), api.resolve(child)
    rp, rx = dqi_regions(api, P)[region], dqi_regions(api, X)[region]
    kp, kx = KM.read(str(P['path'])), KM.read(str(X['path']))
    gaps, unal = DF.align(DF.op_hashes(kp, *rp), DF.op_hashes(kx, *rx))
    sig = DF.gap_signatures(kp, kx, gaps, rp[0], rx[0])
    ap, ax = analyse(kp, *rp), analyse(kx, *rx)
    realised = accounted = 0
    rows = []
    for g in sig:
        d = g['old']['toffoli'] - g['new']['toffoli']
        if d <= 0:
            continue
        lo, hi = g['a']
        lo2, hi2 = g['b']
        pf = [s for s in ap['sites'] if s['kind'] in ('measure-out', 'retain') and s['free'] and lo <= s['op'] < hi]
        cf = [s for s in ax['sites'] if s['kind'] in ('measure-out', 'retain') and s['free'] and lo2 <= s['op'] < hi2]
        got = min(d, sum(s['price'] for s in pf))
        realised += d
        accounted += got
        rows.append(dict(old=g['a'], new=g['b'], realised=d, parent_flagged=sum(s['price'] for s in pf),
                         parent_kinds=dict(Counter(s['kind'] for s in pf)), child_flagged=sum(s['price'] for s in cf)))
    allp = [s for s in ap['sites'] if s['free'] and s['kind'] in ('measure-out', 'retain')]
    allc = [s for s in ax['sites'] if s['free'] and s['kind'] in ('measure-out', 'retain')]
    kinds = lambda L: dict(Counter({k: sum(s['price'] for s in L if s['kind'] == k) for k in ('measure-out', 'retain')}))
    res = dict(schema='vqcc-qhdl-retain-control/1', tier='static', parent=P['sha'], child=X['sha'], region=region,
               realised=realised, accounted=accounted, accounted_pct=round(100 * accounted / realised, 2) if realised else None,
               whole_delta=ap['counts']['toffoli'] - ax['counts']['toffoli'],
               parent_flagged_free=kinds(allp), child_flagged_free=kinds(allc),
               changed_sites=len(rows),
               child_flagged_in_changed_gaps=sum(r['child_flagged'] for r in rows),
               parent_flagged_in_changed_gaps=sum(r['parent_flagged'] for r in rows),
               unaligned=bool(unal), rows=rows)
    api.write_json(Path(out) / 'control.json', res)
    return res


def _main(argv=None):
    ap = argparse.ArgumentParser(description='retained-product detector, positive control')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('control')
    c.add_argument('-c', '--parent', required=True)
    c.add_argument('-c2', '--child', required=True)
    c.add_argument('--region', default='unextract')
    c.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    import qhdl as api
    res = control(api, a.parent, a.child, a.region, a.out)
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    _main()
