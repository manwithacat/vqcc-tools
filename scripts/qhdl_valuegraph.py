#!/usr/bin/env python3
"""qhdl plugin: `valuegraph` — static value graph of a KMB.

    qhdl valuegraph export -c CIRCUIT [--out DIR]          manifest.json + values.jsonl + uses.jsonl
    qhdl valuegraph show   -c CIRCUIT (--value ID | --op I | --wire W | --bit B)
    qhdl valuegraph live   -c CIRCUIT --cut K [--through K2] [--min-span N]

Schema `qhdl-value-graph/v1`, evidence tier static (op stream only: no simulation, no input enumeration).

Cut convention: cut k is the point before op k (equivalently after op k-1); cuts run 0..num_ops. A value produced by
op i has birth_cut i+1 (initial values 0). A value is available at cut c when birth_cut <= c and (kill_cut is null
or c < kill_cut). last_use_cut is the index of the last consuming op (the value must exist at that cut).

Values are wire-level SSA versions: every op that changes a wire (X, CX, CCX targets, SWAP/CSWAP pair, R, HMR) makes a
new version of that wire and *supersedes* the old one (kill.kind `superseded_in_place`: the old value is not erased
in the physical sense, it was transformed). Real erasure is kill.kind `reset` (R) or `measured` (HMR). Diagonal ops
(Z, CZ, CCZ, Z_POW) read their operands with role `phase` and create no value. Classical bits are versioned the same
way (HMR outcome, BIT_STORE0/1, BIT_INVERT). Ops under PUSH_CONDITION_IF or carrying their own `if bN` list the
guard bit values in `guards`; a guarded write keeps the old value as an input (it may be unchanged). A guarded
R/HMR is not treated as a certain kill (kill_cut null, `conditional_kill` set).

Semantic lifetime (birth_cut, last_use_cut, kill_cut) is separate from the carrier: `carrier` names the allocator
interval (the `q42#3` addresses of intervals.tsv, wire live from a touch after an unconditional R/HMR to the next
one) that physically holds the value. A reusable wire is a different carrier interval for each value generation.

HMR is an explicit kill of the measured value, a new classical outcome value (`bN:vK`), a new zero quantum value on
the wire and a `phase_obligation` on the kill: status is always `unknown` (the repair is not proved here); the
syntactic fact `guarded_blocks` lists the condition blocks opened on the outcome bit. Nothing about a value's
expression, inverse cone or recomputability is claimed: `expression` is `unknown` throughout.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_bitsim as B  # noqa: E402

SCHEMA = 'qhdl-value-graph/v1'
OPNAME = B.KINDS
PHASE_OPS = (B.C_Z, B.C_CZ, B.C_CCZ, B.C_ZPOW)


def _vid(prefix, n, k):
    return f'{prefix}{n}:v{k}'


def build(ops, xq, nq, intervals=None):
    """-> (values dict id->record, uses list, info). ops: list of qhdl_bitsim.Op. intervals: [(address, wire, start, end)]
    (inclusive ends) or None."""
    n = len(ops)
    xset = set(xq)
    values, uses = {}, []
    wcur, wver, bcur, bver = {}, {}, {}, {}
    touches = {}
    stack = []                       # [(guard value id, push op)]
    open_blocks = []                 # indices into blocks
    blocks = []                      # {push, pop, guard}

    def new_value(kind, prefix, num, birth_cut, producer, **extra):
        ver = (wver if prefix == 'q' else bver)
        k = ver.get(num, 0)
        ver[num] = k + 1
        vid = _vid(prefix, num, k)
        rec = dict(id=vid, kind=kind, birth_cut=birth_cut, last_use_cut=None, kill_cut=None, producer=producer,
                   consumers=[], expression='unknown')
        rec['wire' if prefix == 'q' else 'bit'] = num
        rec.update(extra)
        values[vid] = rec
        (wcur if prefix == 'q' else bcur)[num] = vid
        return vid

    def cur_wire(w):
        if w not in wcur:
            if w in xset:
                new_value('input', 'q', w, 0, dict(op=None, opcode='REGISTER_INPUT', condition_depth=0, inputs=[],
                                                   guards=[]), initial=True)
            else:
                new_value('quantum', 'q', w, 0, dict(op=None, opcode='INIT_ZERO', condition_depth=0, inputs=[],
                                                     guards=[]), initial=True, known_value=0)
        return wcur[w]

    def cur_bit(b):
        if b not in bcur:
            new_value('classical', 'b', b, 0, dict(op=None, opcode='INIT_BIT', condition_depth=0, inputs=[], guards=[]),
                      initial=True, known_value='unknown')
        return bcur[b]

    def use(i, vid, role, guards, op):
        v = values[vid]
        v['consumers'].append(dict(op=i, role=role))
        if v['last_use_cut'] is None or i > v['last_use_cut']:
            v['last_use_cut'] = i
        uses.append(dict(op=i, opcode=OPNAME[op.code] + ('_IF' if op.cond >= 0 else ''), value=vid, role=role,
                         guards=guards, condition_depth=len(stack)))

    def supersede(old, i, kind, succ):
        v = values[old]
        v['kill_cut'] = i + 1
        v['kill'] = dict(op=i, kind=kind, successor=succ)

    for i, op in enumerate(ops):
        c = op.code
        if c == B.C_INFO:
            continue
        if c == B.C_POP:
            if open_blocks:
                blocks[open_blocks.pop()]['pop'] = i
            if stack:
                stack.pop()
            continue
        guards = [g for g, _ in stack]
        own = None
        if op.cond >= 0:
            own = cur_bit(op.cond)
            guards = guards + [own]
        depth = len(stack)
        if c == B.C_PUSH:
            g = cur_bit(op.cond)
            use(i, g, 'guard_block', [x for x, _ in stack], op)
            blocks.append(dict(push=i, pop=None, guard=g))
            open_blocks.append(len(blocks) - 1)
            values[g].setdefault('guarded_blocks', []).append(len(blocks) - 1)
            stack.append((g, i))
            continue
        if own is not None:
            use(i, own, 'guard', [x for x, _ in stack], op)
        conditional = bool(guards)
        pmeta = dict(op=i, opcode=OPNAME[c] + ('_IF' if op.cond >= 0 else ''), condition_depth=depth, guards=guards,
                     conditional=conditional)
        if c == B.C_NEG:
            uses.append(dict(op=i, opcode=pmeta['opcode'], value=None, role='global_phase', guards=guards,
                             condition_depth=depth))
            continue
        if c in (B.C_BINV, B.C_BST0, B.C_BST1):
            old = cur_bit(op.bit)
            ins = [old] if c == B.C_BINV or conditional else []
            if c == B.C_BINV:
                use(i, old, 'bit_read', guards, op)
            elif conditional:
                use(i, old, 'bit_keep', guards, op)
            nv = new_value('classical', 'b', op.bit, i + 1, dict(pmeta, inputs=ins + guards),
                           **({'known_value': 0 if c == B.C_BST0 else 1} if c != B.C_BINV and not conditional else {}))
            supersede(old, i, 'superseded_in_place', nv)
            continue
        qs = list(op.q)
        for w in qs:
            touches.setdefault(w, []).append(i)
        if c in PHASE_OPS:
            for w in qs:
                use(i, cur_wire(w), 'phase', guards, op)
            if c == B.C_ZPOW:
                uses[-1]['angle_units_2^-64_turns'] = int(op.angle)
            continue
        if c == B.C_HMR:
            w = qs[0]
            old = cur_wire(w)
            use(i, old, 'measure', guards, op)
            if conditional:
                values[old]['conditional_kill'] = dict(op=i, guards=guards)
            else:
                values[old]['kill_cut'] = i + 1
            ob = cur_bit(op.bit)
            outcome = new_value('classical', 'b', op.bit, i + 1, dict(pmeta, inputs=[old] + guards,
                                                                       measured_value=old), outcome_of=old)
            supersede(ob, i, 'superseded_in_place', outcome)
            zero = new_value('quantum', 'q', w, i + 1, dict(pmeta, inputs=[old] + guards), post_reset=True,
                             **({} if conditional else {'known_value': 0}))
            if not conditional:
                values[old]['kill'] = dict(op=i, kind='measured', outcome_value=outcome, successor=zero)
                values[old]['kill']['phase_obligation'] = dict(
                    status='unknown', outcome_value=outcome, measured_value=old,
                    meaning='sign (-1)^(outcome * measured value) must be repaired on outcome 1; repair not proved here',
                    guarded_blocks='see outcome value guarded_blocks')
            else:
                values[old]['kill'] = dict(op=i, kind='conditional_measure', outcome_value=outcome, successor=zero,
                                           phase_obligation=dict(status='unknown', outcome_value=outcome,
                                                                 measured_value=old))
            continue
        if c == B.C_R:
            w = qs[0]
            old = cur_wire(w)
            use(i, old, 'reset', guards, op)
            zero = new_value('quantum', 'q', w, i + 1, dict(pmeta, inputs=[old] + guards), post_reset=True,
                             **({} if conditional else {'known_value': 0}))
            if conditional:
                values[old]['conditional_kill'] = dict(op=i, guards=guards)
                values[old]['kill'] = dict(op=i, kind='conditional_reset', successor=zero)
            else:
                values[old]['kill_cut'] = i + 1
                values[old]['kill'] = dict(op=i, kind='reset', successor=zero)
            continue
        # reversible updates: X CX CCX SWAP CSWAP
        roles = {B.C_X: 'w', B.C_CX: 'rw', B.C_CCX: 'rrw', B.C_SWAP: 'ss', B.C_CSWAP: 'rss'}[c]
        olds = [cur_wire(w) for w in qs]
        for w, o, r in zip(qs, olds, roles):
            use(i, o, {'w': 'target', 'r': 'control', 's': 'swap'}[r], guards, op)
        ctrl = [o for o, r in zip(olds, roles) if r == 'r']
        for w, o, r in zip(qs, olds, roles):
            if r == 'r':
                continue
            ins = ([o] + ctrl + guards) if r == 'w' else (olds + guards)
            nv = new_value('quantum', 'q', w, i + 1, dict(pmeta, inputs=ins))
            supersede(o, i, 'superseded_in_place', nv)

    for vid, v in values.items():
        if v['kill_cut'] is None and 'conditional_kill' not in v:
            v['kill'] = dict(kind='live_at_end')
    for bi, b in enumerate(blocks):
        if b['pop'] is None:
            b['pop'] = n
    for v in values.values():
        if 'guarded_blocks' in v:
            v['guarded_blocks'] = [dict(push_op=blocks[x]['push'], pop_op=blocks[x]['pop']) for x in v['guarded_blocks']]
    # block guards keep the bit alive to the end of the block
    for b in blocks:
        g = values[b['guard']]
        end = b['pop']
        if g['last_use_cut'] is None or end > g['last_use_cut']:
            g['last_use_cut'] = end
    # outcome obligations: list the blocks guarded by the outcome bit
    for v in values.values():
        k = v.get('kill')
        if k and k.get('outcome_value') and 'phase_obligation' in k:
            o = values[k['outcome_value']]
            k['phase_obligation']['guarded_blocks'] = o.get('guarded_blocks', [])
            k['phase_obligation']['outcome_consumers'] = list(o['consumers'])
    # carriers
    ivs = {}
    for addr, w, s, e in (intervals or []):
        ivs.setdefault(w, []).append((s, e, addr))
    for w in ivs:
        ivs[w].sort()
    starts = {w: [x[0] for x in l] for w, l in ivs.items()}
    for v in values.values():
        if 'wire' not in v:
            continue
        w = v['wire']
        p = v['producer']['op']
        if p is None:
            probe = (touches.get(w) or [0])[0]
        elif v.get('post_reset') and not v['producer']['conditional']:
            t = touches.get(w, [])
            j = bisect.bisect_right(t, p)
            probe = t[j] if j < len(t) else None
        else:
            probe = p
        car = None
        if probe is not None and w in ivs:
            j = bisect.bisect_right(starts[w], probe) - 1
            if j >= 0 and ivs[w][j][0] <= probe <= ivs[w][j][1]:
                s, e, addr = ivs[w][j]
                car = dict(interval=addr, start_op=s, end_op=e)
        v['carrier'] = car if car else 'unknown'
    # successor chain
    for v in values.values():
        k = v.get('kill')
        if k and k.get('successor'):
            values[k['successor']]['prev'] = v['id']
    return values, uses, dict(num_ops=n, blocks=blocks)


def load_ops(kmx):
    ops, header, xq = B.parse_kmx(str(kmx))
    nq = 1 + max(max(o.q) for o in ops if o.q)
    return ops, xq, nq


def read_intervals(ws):
    out = []
    with open(Path(ws) / 'intervals.tsv', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            out.append((r['address'], int(r['wire']), int(r['start']), int(r['end'])))
    return out


def export(ops, xq, nq, intervals, sha, out_dir, declared_qubits=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    values, uses, info = build(ops, xq, nq, intervals)
    hashes = {}
    for name, rows in (('values.jsonl', values.values()), ('uses.jsonl', uses)):
        p = out_dir / name
        with open(p, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, separators=(',', ':')) + '\n')
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    kinds = {}
    for v in values.values():
        kinds[v['kind']] = kinds.get(v['kind'], 0) + 1
    man = dict(schema=SCHEMA, source_sha256=sha, num_qubits=declared_qubits or nq, num_ops=info['num_ops'],
               cut_convention='cut k = before op k = after op k-1; birth_cut = producing op + 1; initial values 0; '
                              'last_use_cut = last consuming op; kill_cut = killing op + 1',
               values='values.jsonl', uses='uses.jsonl', evidence_tier='static', value_count=len(values),
               use_count=len(uses), values_by_kind=kinds, condition_blocks=len(info['blocks']),
               file_sha256=hashes,
               unknown_policy='value expression, inverse cone, recomputability and HMR phase repair are `unknown`; '
                              'only structural facts of the op stream are recorded')
    (out_dir / 'manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    return man, values, uses


def live_at(values, cut, through=None, min_span=0):
    """Values available at `cut` (and still unkilled and still used at or after `through`)."""
    through = cut if through is None else through
    res = []
    for v in values.values():
        if v['birth_cut'] > cut:
            continue
        kc = v['kill_cut']
        if kc is not None and kc <= through:
            continue
        lu = v['last_use_cut']
        if lu is None or lu < through:
            continue
        if lu - v['birth_cut'] < min_span:
            continue
        res.append(v)
    return res


# ---------------------------------------------------------------- qhdl verbs
def _graph(api, a):
    import qhdl_envelope as E
    meta = api.ensure_open(a.circuit)
    d = api.ws_dir(meta['sha256'])
    if not (d / 'circuit.kmx').exists():
        raise api.QhdlError(f"{meta['sha12']}: no workspace kmx", 'valuegraph needs an opened enumerable circuit')
    ops, xq, nq = load_ops(d / 'circuit.kmx')
    return E, meta, d, ops, xq, max(nq, meta['declared_qubits'])


def _env(api, E, led, meta, claim, ev, rerun, facts, rows):
    return E.Envelope(led, claim, tier='static', domain='op stream, static', evidence=api.rel(ev),
                      rerun=api.cmd('valuegraph', *rerun), tool_sha=api.tool_sha(), circuit_sha=meta['sha256'],
                      facts=facts, rows=rows, parts=[], next_cmds=[])


def v_export(api, a):
    E, meta, d, ops, xq, nq = _graph(api, a)
    out = Path(a.out) if a.out else d / 'valuegraph'
    if out.exists() and not a.out:
        k = 1
        while (d / f'valuegraph-{k}').exists():
            k += 1
        out = d / f'valuegraph-{k}'
    led = E.Ledger()
    man, values, uses = export(ops, xq, nq, read_intervals(d), meta['sha256'], out, meta['declared_qubits'])
    led.earn('static', 'value graph from the op stream and the allocator intervals (no simulation)')
    claim = (f"{meta['sha12']}: {man['value_count']} values, {man['use_count']} uses, {man['condition_blocks']} "
             f"condition blocks exported as {SCHEMA}")
    return _env(api, E, led, meta, claim, out / 'manifest.json', ['export', '-c', meta['sha12']]
                + (['--out', a.out] if a.out else []), {k: v for k, v in man.items() if k != 'file_sha256'}, [])


def _graph_values(api, a):
    E, meta, d, ops, xq, nq = _graph(api, a)
    values, uses, info = build(ops, xq, nq, read_intervals(d))
    return E, meta, d, values, uses, info


def v_show(api, a):
    E, meta, d, values, uses, info = _graph_values(api, a)
    led = E.Ledger()
    led.earn('static', 'value graph from the op stream (no simulation)')
    if a.value:
        sel = [values[a.value]] if a.value in values else []
    elif a.op is not None:
        sel = [v for v in values.values() if v['producer']['op'] == a.op or (v.get('kill') or {}).get('op') == a.op]
    elif a.wire is not None:
        sel = [v for v in values.values() if v.get('wire') == a.wire]
    else:
        sel = [v for v in values.values() if v.get('bit') == a.bit]
    claim = f"{meta['sha12']}: {len(sel)} value records selected"
    return _env(api, E, led, meta, claim, d / 'meta.json',
                ['show', '-c', meta['sha12']] + [x for x in (['--value', a.value] if a.value else [])
                                                 + (['--op', a.op] if a.op is not None else [])
                                                 + (['--wire', a.wire] if a.wire is not None else [])
                                                 + (['--bit', a.bit] if a.bit is not None else [])],
                {}, sel[:20])


def v_live(api, a):
    E, meta, d, values, uses, info = _graph_values(api, a)
    led = E.Ledger()
    led.earn('static', 'value graph from the op stream (no simulation)')
    rows = live_at(values, a.cut, a.through, a.min_span)
    rows.sort(key=lambda v: v['birth_cut'])
    slim = [dict(id=v['id'], kind=v['kind'], birth_cut=v['birth_cut'], last_use_cut=v['last_use_cut'],
                 kill_cut=v['kill_cut'], carrier=v['carrier'] if isinstance(v['carrier'], str) else v['carrier']['interval'])
            for v in rows]
    ev = d / 'valuegraph' / f'live_{a.cut}_{a.through if a.through is not None else a.cut}.json'
    api.write_json(ev, slim)
    claim = f"{meta['sha12']}: {len(slim)} values available at cut {a.cut} and used at or after cut " \
            f"{a.through if a.through is not None else a.cut}"
    return _env(api, E, led, meta, claim, ev, ['live', '-c', meta['sha12'], '--cut', a.cut]
                + (['--through', a.through] if a.through is not None else []) + (['--min-span', a.min_span] if a.min_span else []),
                {'count': len(slim)}, slim[:20])


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('valuegraph', help='static value graph (qhdl-value-graph/v1): export, show, live',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='valuegraph_verb', required=True)
    s = sub.add_parser('export', help='write manifest.json, values.jsonl, uses.jsonl')
    s.add_argument('-c', '--circuit', required=True), s.add_argument('--out')
    s.add_argument('--all', action='store_true')
    s.set_defaults(fn=lambda a: v_export(api, a))
    s = sub.add_parser('show', help='value records by id, op, wire or bit')
    s.add_argument('-c', '--circuit', required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--value'), g.add_argument('--op', type=int), g.add_argument('--wire', type=int)
    g.add_argument('--bit', type=int)
    s.add_argument('--all', action='store_true')
    s.set_defaults(fn=lambda a: v_show(api, a))
    s = sub.add_parser('live', help='values available at a cut and still used at or after --through')
    s.add_argument('-c', '--circuit', required=True)
    s.add_argument('--cut', type=int, required=True), s.add_argument('--through', type=int)
    s.add_argument('--min-span', type=int, default=0, help='only values with last_use_cut - birth_cut >= N')
    s.add_argument('--all', action='store_true')
    s.set_defaults(fn=lambda a: v_live(api, a))


def main(argv=None):
    ap = argparse.ArgumentParser(description='standalone: export the value graph of an opened workspace dir')
    ap.add_argument('workspace', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--sha', required=True)
    a = ap.parse_args(argv)
    ops, xq, nq = load_ops(a.workspace / 'circuit.kmx')
    man, _, _ = export(ops, xq, nq, read_intervals(a.workspace), a.sha, a.out)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
