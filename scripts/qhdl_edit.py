#!/usr/bin/env python3
"""qhdl plugin: `edit` — splice ops into a .kmb by original op index and write a new .kmb plus an op-index map
(one shared decode/edit/encode path instead of ad-hoc per-experiment helpers).

    qhdl edit -c CIRCUIT --script edits.txt --out new.kmb [--map new.map.json]

Edit script, one edit per line ('#' starts a comment). Indices are 0-based op indices of the ORIGINAL circuit
(INFO_TAG ops count), so edits never shift each other (equivalent to applying them back to front):

    delete A              delete op A
    delete A:B            delete ops A..B-1
    insert A => OP; OP    insert before original op A (A = number of ops appends)
    replace A => OP; OP   replace op A
    replace A:B => OP; OP replace ops A..B-1

OP is kmx-style text: `CCX q1 q2 q3`, `HMR q5 b7`, `CCZ q1 q2 q3 if b7` (or `CCZ_IF q1 q2 q3 b7`), `CX q1 q2`,
`R q4`, `X q4`, `SWAP q1 q2`, `CSWAP q0 q1 q2`, `Z/CZ/NEG`, `BIT_STORE0 b3`, `BIT_STORE1 b3`, `BIT_INVERT b3`,
`PUSH_CONDITION if b3` (or `PUSH_CONDITION b3`), `POP_CONDITION`, `INFO_TAG "text"`, `@i` (a copy of original op i,
with its condition, angle or tag) and `Z_POW qN @i` (rotation angle copied from original Z_POW op i; new angles are
not supported because the 128-bit fixed-point angle encoding is not reproduced here). A classical bit id >= the
circuit's num_bits extends num_bits (new bits); a qubit id >= num_qubits extends num_qubits and is reported.

Two edits may not touch the same original op; several inserts at one index keep their script order and come before
a replace/delete starting there. Everything not edited is copied byte-for-byte (registers, tags, angles, header).

The map (default OUT + '.map.json') lists hunks like difflib opcodes [tag, i1, i2, j1, j2] (old ops i1..i2-1 ->
new ops j1..j2-1; tag equal/replace/delete/insert). `qhdl fdiff --map` uses it to translate op ranges: an old cut
A maps to the new index of the first op emitted at or after A (insertions at A fall inside a window starting at A
and outside one ending at A); a cut strictly inside an edited range is refused.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_kmb as K  # noqa: E402

BASE = {'NEG': 0x00, 'POP_CONDITION': 0x01, 'INFO_TAG': 0x06, 'Z': 0x10, 'X': 0x11, 'R': 0x12, 'Z_POW': 0x13,
        'CZ': 0x20, 'CX': 0x21, 'SWAP': 0x22, 'CCZ': 0x30, 'CCX': 0x31, 'CSWAP': 0x32, 'BIT_STORE0': 0x40,
        'BIT_STORE1': 0x41, 'BIT_INVERT': 0x42, 'HMR': 0x50, 'PUSH_CONDITION': 0x81}
U32 = ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')


class EditError(ValueError):
    pass


# ---------------------------------------------------------------- parsing
def parse_range(s, n_ops, allow_end=False):
    m = re.fullmatch(r'(\d+)(?::(\d+))?', s.strip())
    if not m:
        raise EditError(f'bad op index or range {s!r} (use A or A:B)')
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) is not None else a + 1
    hi = n_ops if allow_end else n_ops - 1
    if not 0 <= a <= hi or (m.group(2) is not None and not a < b <= n_ops) or (m.group(2) is None and a > hi):
        raise EditError(f'op range {s!r} outside the circuit (0..{n_ops - 1}, {n_ops} ops)')
    return a, b


def parse_op(text):
    """Op text -> edit record; Z_POW accepts @source or angle128:HEX (full-turn units / 2^128)."""
    text = text.strip()
    if not text:
        raise EditError('empty op')
    if text.startswith('@'):
        if not re.fullmatch(r'@\d+', text):
            raise EditError(f'bad op copy {text!r} (use @i)')
        return dict(copy=int(text[1:]))
    if text.startswith('INFO_TAG'):
        rest = text[len('INFO_TAG'):].strip()
        if rest.startswith('"') and rest.endswith('"') and len(rest) >= 2:
            rest = rest[1:-1]
        return dict(code=0x06, q=(), b0=None, bc=None, tag=rest.encode())
    toks = text.split()
    name = toks[0]
    args = toks[1:]
    cond = None
    if len(args) >= 2 and args[-2] == 'if':
        cond = args[-1]
        args = args[:-2]
    if name.endswith('_IF') and name != 'PUSH_CONDITION':
        name = name[:-3]
        if not args:
            raise EditError(f'{text!r}: an _IF form needs its condition bit last')
        cond = args[-1]
        args = args[:-1]
    if name not in BASE:
        raise EditError(f'unknown op {name!r} in {text!r}')
    base = BASE[name]
    angle_from = None
    angle_literal = None
    if name == 'Z_POW':
        if len(args) != 2:
            raise EditError('Z_POW takes qN @i or qN angle128:<32 hex digits>')
        if re.fullmatch(r'@\d+', args[1]):
            angle_from = int(args[1][1:])
        elif re.fullmatch(r'angle128:[0-9a-fA-F]{32}', args[1]):
            angle_literal = int(args[1].split(':')[1], 16).to_bytes(16, 'little')
        else:
            raise EditError('Z_POW angle must be @i or angle128:<32 hex digits> (full-turn units / 2^128)')
        args = args[:1]
    if name == 'PUSH_CONDITION':
        if cond is None:
            if len(args) != 1:
                raise EditError(f'{text!r}: PUSH_CONDITION takes one bit')
            cond = args[0]
            args = []
        elif args:
            raise EditError(f'{text!r}: PUSH_CONDITION takes one bit')
    qs = [a for a in args if a.startswith('q')]
    bs = [a for a in args if a.startswith('b')]
    if len(qs) + len(bs) != len(args):
        raise EditError(f'{text!r}: operands must be qN or bN')
    try:
        q = tuple(int(a[1:]) for a in qs)
        b0 = int(bs[0][1:]) if bs else None
        bc = int(cond[1:]) if cond is not None else None
    except ValueError:
        raise EditError(f'{text!r}: bad operand number')
    if cond is not None and not cond.startswith('b'):
        raise EditError(f'{text!r}: condition must be a bit bN')
    nq = {0x30: 3, 0x20: 2, 0x10: 1}.get(base & 0x30, 0) if base != 0x81 else 0
    if len(q) != nq:
        raise EditError(f'{text!r}: {name} takes {nq} qubit(s)')
    if bool(base & 0x40) != (b0 is not None) or len(bs) > 1:
        raise EditError(f'{text!r}: {name} takes {"one bit" if base & 0x40 else "no bit"} operand')
    if len(set(q)) != len(q):
        raise EditError(f'{text!r}: repeated qubit')
    code = base
    if bc is not None and base != 0x81:
        if base in (0x01, 0x06):
            raise EditError(f'{text!r}: {name} cannot be conditioned')
        code = base | 0x80
    return dict(code=code, q=q, b0=b0, bc=bc, angle_from=angle_from, angle_literal=angle_literal)


def parse_script(text, n_ops):
    """-> list of edits dict(kind, a, b, ops, line). Sorted by position; overlaps refused."""
    edits = []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip() if 'INFO_TAG' in raw else raw.split('#', 1)[0].strip()  # a tag text may contain '#'
        if not line:
            continue
        head, sep, body = line.partition('=>')
        parts = head.split()
        if len(parts) != 2 or parts[0] not in ('delete', 'insert', 'replace'):
            raise EditError(f'line {ln}: {raw.strip()!r}: expected `delete A[:B]`, `insert A => ...` or '
                            '`replace A[:B] => ...`')
        kind = parts[0]
        if kind == 'delete' and sep:
            raise EditError(f'line {ln}: delete takes no ops')
        if kind != 'delete' and not sep:
            raise EditError(f'line {ln}: {kind} needs `=> OP; OP`')
        if kind == 'insert':
            if ':' in parts[1]:
                raise EditError(f'line {ln}: insert takes one index')
            a, _ = parse_range(parts[1], n_ops, allow_end=True)
            b = a
        else:
            a, b = parse_range(parts[1], n_ops)
        ops = []
        if sep:
            try:
                ops = [parse_op(s) for s in body.split(';') if s.strip()]
            except EditError as exc:
                raise EditError(f'line {ln}: {exc}')
            if not ops:
                raise EditError(f'line {ln}: no ops after =>')
        edits.append(dict(kind=kind, a=a, b=b, ops=ops, line=ln))
    order = sorted(range(len(edits)), key=lambda j: (edits[j]['a'], edits[j]['b'] > edits[j]['a'], j))
    edits = [edits[j] for j in order]
    last_end = -1
    for e in edits:
        if e['b'] > e['a']:
            if e['a'] < last_end:
                raise EditError(f"line {e['line']}: ops {e['a']}..{e['b'] - 1} overlap an earlier edit")
            last_end = e['b']
        elif e['a'] < last_end:
            raise EditError(f"line {e['line']}: insert at {e['a']} falls inside an edited range")
    return edits


# ---------------------------------------------------------------- applying
def raw_tags(k):
    off, n, _ = k.offsets['info_tags']
    ends = np.frombuffer(k.raw, dtype='<u8', count=n, offset=off).tolist()
    base = off + 8 * n
    out, s = [], 0
    for e in ends:
        out.append(bytes(k.raw[base + s: base + e]))
        s = e
    return out


class Source:
    """Per-stream positions of original ops, computed lazily for sorted boundaries (no full prefix arrays: a DQI
    circuit has 60M ops)."""

    def __init__(self, k):
        self.k = k
        self.t = k.op_types
        self.pos = 0
        self.cnt = dict(q3=0, q2=0, q1=0, b0=0, bc=0, ang=0, tag=0)
        self.src = {n: k.stream(n) for n in U32}
        off, n, _ = k.offsets['angles']
        self.angles = np.frombuffer(k.raw, dtype=np.uint8, count=16 * n, offset=off).reshape(n, 16) if n else \
            np.zeros((0, 16), np.uint8)
        self.tags = raw_tags(k)

    @staticmethod
    def counts(seg):
        seg = np.asarray(seg)
        cls = seg & 0x30
        return dict(q3=int(np.count_nonzero(cls == 0x30)), q2=int(np.count_nonzero(cls == 0x20)),
                    q1=int(np.count_nonzero(cls == 0x10)), b0=int(np.count_nonzero(seg & 0x40)),
                    bc=int(np.count_nonzero(seg & 0x80)), ang=int(np.count_nonzero((seg & 0x7F) == 0x13)),
                    tag=int(np.count_nonzero(seg == 0x06)))

    def at(self, i):
        """Stream counters before op i (i must be >= the last call)."""
        if i < self.pos:
            raise EditError('internal: boundaries out of order')
        c = self.counts(self.t[self.pos:i])
        for kk in self.cnt:
            self.cnt[kk] += c[kk]
        self.pos = i
        return dict(self.cnt)

    def op(self, i):
        """Full record of original op i (random access; for @i copies)."""
        t = int(self.t[i])
        c = self.counts(self.t[:i])
        rec = dict(code=t, q=(), b0=None, bc=None, angle=None, tag=None)
        cls = t & 0x30
        if t != 0x81:
            if cls == 0x30:
                rec['q'] = tuple(int(self.src[n][c['q3']]) for n in ('qqq0', 'qqq1', 'qqq2'))
            elif cls == 0x20:
                rec['q'] = tuple(int(self.src[n][c['q2']]) for n in ('qq0', 'qq1'))
            elif cls == 0x10:
                rec['q'] = (int(self.src['q0'][c['q1']]),)
        if t & 0x40:
            rec['b0'] = int(self.src['b0'][c['b0']])
        if t & 0x80:
            rec['bc'] = int(self.src['bc'][c['bc']])
        if (t & 0x7F) == 0x13:
            rec['angle'] = bytes(self.angles[c['ang']])
        if t == 0x06:
            rec['tag'] = self.tags[c['tag']]
        return rec


def apply(k, edits):
    """-> (op_types, streams, angles bytes, tags, hunks, info). hunks: [tag, i1, i2, j1, j2] (difflib style)."""
    S = Source(k)
    n = len(k.op_types)
    parts = {nm: [] for nm in U32}
    ot_parts, ang_parts, tags_out = [], [], []
    hunks = []
    new_pos = 0
    prev = 0
    maxq, maxb = -1, -1
    copies = {}

    def take(a, b):
        nonlocal new_pos
        if b <= a:
            return
        c0 = S.at(a)
        c1 = S.at(b)
        ot_parts.append(np.asarray(S.t[a:b], np.uint8))
        for nm, kk in (('qqq0', 'q3'), ('qqq1', 'q3'), ('qqq2', 'q3'), ('qq0', 'q2'), ('qq1', 'q2'), ('q0', 'q1'),
                       ('b0', 'b0'), ('bc', 'bc')):
            parts[nm].append(np.asarray(S.src[nm][c0[kk]:c1[kk]]))
        ang_parts.append(S.angles[c0['ang']:c1['ang']].tobytes())
        tags_out.extend(S.tags[c0['tag']:c1['tag']])
        if hunks and hunks[-1][0] == 'equal' and hunks[-1][2] == a:
            hunks[-1][2] = b
            hunks[-1][4] += b - a
        else:
            hunks.append(['equal', a, b, new_pos, new_pos + b - a])
        new_pos += b - a

    def emit(op):
        nonlocal maxq, maxb
        if 'copy' in op:
            j = op['copy']
            if not 0 <= j < n:
                raise EditError(f'@{j}: no such op')
            if j not in copies:
                copies[j] = S.op(j)
            r = copies[j]
            code, q, b0, bc, ang, tag = r['code'], r['q'], r['b0'], r['bc'], r['angle'], r['tag']
        else:
            code, q, b0, bc, tag = op['code'], op['q'], op['b0'], op['bc'], op.get('tag')
            ang = op.get('angle_literal')
            if op.get('angle_from') is not None:
                j = op['angle_from']
                if not 0 <= j < n or (int(S.t[j]) & 0x7F) != 0x13:
                    raise EditError(f'Z_POW ... @{j}: original op {j} is not a Z_POW')
                if j not in copies:
                    copies[j] = S.op(j)
                ang = copies[j]['angle']
        ot_parts.append(np.array([code], np.uint8))
        cls = code & 0x30
        if code != 0x81:
            names = {0x30: ('qqq0', 'qqq1', 'qqq2'), 0x20: ('qq0', 'qq1'), 0x10: ('q0',)}.get(cls, ())
            for nm, x in zip(names, q):
                parts[nm].append(np.array([x], np.uint32))
                maxq = max(maxq, x)
        if code & 0x40:
            parts['b0'].append(np.array([b0], np.uint32))
            maxb = max(maxb, b0)
        if code & 0x80:
            parts['bc'].append(np.array([bc], np.uint32))
            maxb = max(maxb, bc)
        if (code & 0x7F) == 0x13:
            ang_parts.append(ang)
        if code == 0x06:
            tags_out.append(tag)

    for e in edits:
        take(prev, e['a'])
        j1 = new_pos
        for op in e['ops']:
            emit(op)
            new_pos += 1
        prev = max(prev, e['b'])
        tag = 'insert' if e['b'] == e['a'] else ('delete' if not e['ops'] else 'replace')
        hunks.append([tag, e['a'], e['b'], j1, new_pos])
    take(prev, n)
    ot = np.concatenate(ot_parts) if ot_parts else np.zeros(0, np.uint8)
    streams = {nm: (np.concatenate(v).astype(np.uint32) if v else np.zeros(0, np.uint32)) for nm, v in parts.items()}
    info = dict(max_new_qubit=maxq, max_new_bit=maxb)
    return ot, streams, b''.join(ang_parts), tags_out, hunks, info


def edit_file(src, script_text, out, map_path=None):
    """Apply an edit script to src (.kmb) and write out (.kmb) and the map JSON. Returns a summary dict."""
    import hashlib
    k = K.read(str(src))
    n = len(k.op_types)
    edits = parse_script(script_text, n)
    if not edits:
        raise EditError('the edit script has no edits')
    ot, streams, ang, tags, hunks, info = apply(k, edits)
    nq = max(int(k.num_qubits), info['max_new_qubit'] + 1)
    nb = max(int(k.num_bits), info['max_new_bit'] + 1)
    regs = [(nm, [(isq, i) for isq, i, _ in cs]) for nm, cs in k.registers]
    out = Path(out)
    K.to_file(out, ot, streams, ang, tags, regs, num_qubits=nq, num_bits=nb)
    old_sha = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    new_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    m = dict(old_sha256=old_sha, new_sha256=new_sha, old_ops=n, new_ops=int(len(ot)), hunks=hunks,
             edits=[dict(kind=e['kind'], a=e['a'], b=e['b'], ops=len(e['ops']), line=e['line']) for e in edits])
    mp = Path(map_path) if map_path else out.with_name(out.name + '.map.json')
    mp.write_text(json.dumps(m, indent=0) + '\n')
    added_q = nq - int(k.num_qubits)
    return dict(old_sha=old_sha, new_sha=new_sha, out=str(out), map=str(mp), old_ops=n, new_ops=int(len(ot)),
                edits=len(edits), num_qubits=nq, num_bits=nb, added_qubits=added_q,
                added_bits=nb - int(k.num_bits), hunks=hunks)


# ---------------------------------------------------------------- op-index map
class OpMap:
    """old cut -> new cut from difflib-style hunks [tag, i1, i2, j1, j2]."""

    def __init__(self, hunks, old_ops=None, new_ops=None):
        self.h = [list(x) for x in hunks]
        self.old_ops = old_ops if old_ops is not None else (self.h[-1][2] if self.h else 0)
        self.new_ops = new_ops if new_ops is not None else (self.h[-1][4] if self.h else 0)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d['hunks'], d.get('old_ops'), d.get('new_ops')), d

    def cut(self, a):
        """New index of old cut a (before old op a). Raises EditError if a is strictly inside an edited range."""
        if a == self.old_ops:
            return self.new_ops
        for tag, i1, i2, j1, j2 in self.h:
            if a == i1:
                return j1
            if i1 < a < i2:
                if tag == 'equal':
                    return j1 + (a - i1)
                raise EditError(f'cut @{a} falls inside edited ops {i1}..{i2 - 1}; choose a cut outside the edit')
        raise EditError(f'cut @{a} outside the mapped circuit ({self.old_ops} ops)')

    def edits(self):
        return [h for h in self.h if h[0] != 'equal']


def auto_hunks(a_lines, b_lines):
    """Hunks aligning two op-text sequences (common prefix/suffix stripped, then difflib)."""
    import difflib
    n, m = len(a_lines), len(b_lines)
    p = 0
    while p < min(n, m) and a_lines[p] == b_lines[p]:
        p += 1
    s = 0
    while s < min(n, m) - p and a_lines[n - 1 - s] == b_lines[m - 1 - s]:
        s += 1
    hunks = []
    if p:
        hunks.append(['equal', 0, p, 0, p])
    sm = difflib.SequenceMatcher(None, a_lines[p:n - s], b_lines[p:m - s], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        hunks.append([tag, i1 + p, i2 + p, j1 + p, j2 + p])
    if s:
        hunks.append(['equal', n - s, n, m - s, m])
    return hunks


# ---------------------------------------------------------------- CLI verb
def v_edit(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    c = api.resolve(a.circuit)
    src = Path(c['path'])
    if src.suffix != '.kmb':
        raise api.QhdlError('edit reads and writes .kmb circuits', 'convert a .kmx with the kickmix venv first')
    out = Path(a.out)
    if out.suffix != '.kmb':
        raise api.QhdlError('--out must end in .kmb (edit writes KMB bytes)')
    if out.exists() and not a.force:
        raise api.QhdlError(f'{out} exists; refusing to overwrite', 'pass --force or choose another --out')
    try:
        script = Path(a.script).read_text()
    except OSError as exc:
        raise api.QhdlError(f'cannot read --script: {exc}')
    try:
        r = edit_file(src, script, out, a.map)
    except EditError as exc:
        raise api.QhdlError(str(exc), 'see `qhdl edit --help` for the script format')
    api.register_candidate(r['new_sha'], out, parent=c['sha'], how=f"qhdl edit of {c['sha'][:12]} "
                                                                   f"({r['edits']} edits, {Path(a.script).name})")
    led.earn('static', 'op streams spliced by original index; unedited ops, registers, tags and angles copied '
                       'byte-for-byte (no semantic check: run fdiff / the verifier)')
    eh = [h for h in r['hunks'] if h[0] != 'equal']
    rows = [dict(edit=h[0], old=f'{h[1]}..{h[2] - 1}' if h[2] > h[1] else f'@{h[1]}',
                 new=f'{h[3]}..{h[4] - 1}' if h[4] > h[3] else f'@{h[3]}', old_ops=h[2] - h[1], new_ops=h[4] - h[3])
            for h in eh]
    claim = (f"wrote {r['new_sha'][:12]} ({api.fmt_int(r['new_ops'])} ops, {r['new_ops'] - r['old_ops']:+d}) from "
             f"{c['sha'][:12]} with {r['edits']} edits; qubits {r['num_qubits']}"
             + (f" (+{r['added_qubits']} new)" if r['added_qubits'] else '') + f", bits {r['num_bits']}"
             + (f" (+{r['added_bits']} new)" if r['added_bits'] else '')
             + '. Not checked: compare with fdiff --map, then verify')
    facts = dict(out=str(out), new_sha256=r['new_sha'], map=r['map'])
    nxt = []
    task = (c['manifest'] or {}).get('task')
    if task == 'invsqrt' or (task is None and r['old_ops'] <= api.MAX_OPS_TABLE):
        nxt.append(api.cmd('fdiff', '-c', c['sha'][:12], '-c2', str(out), '--map', r['map']))
    nxt.append(api.cmd('verify', '-c', str(out)))
    return E.Envelope(led, claim, domain='op streams (static splice)', evidence=r['map'],
                      rerun=api.cmd('edit', '-c', c['sha'][:12], '--script', a.script, '--out', str(out),
                                    *(['--map', a.map] if a.map else []), '--force'),
                      tool_sha=api.tool_sha(), circuit_sha=c['sha'], facts=facts, rows=rows, next_cmds=nxt)


def qhdl_register(subparsers, api):
    import argparse
    p = subparsers.add_parser(
        'edit', help='splice ops by original index (insert/delete/replace script) -> new .kmb + op-index map',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all', action='store_true', help='print every row')
    p.add_argument('-c', '--circuit', required=True, help='input circuit (.kmb path or sha prefix)')
    p.add_argument('--script', required=True, help='edit script (see above)')
    p.add_argument('--out', required=True, help='output .kmb path')
    p.add_argument('--map', help="op-index map path (default OUT + '.map.json')")
    p.add_argument('--force', action='store_true', help='overwrite --out')
    p.set_defaults(fn=lambda a: v_edit(api, a))


if __name__ == '__main__':  # library smoke: python3 qhdl_edit.py IN.kmb SCRIPT OUT.kmb
    print(json.dumps(edit_file(sys.argv[1], Path(sys.argv[2]).read_text(), sys.argv[3]), default=str)[:2000])
