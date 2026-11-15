#!/usr/bin/env python3
"""Certified park/recompute rewrite for kickmix text circuits.

Edits (JSON list), each one of:
  {"op": "park", "w": 23, "a": 225, "r": 17541, "terms": [["1"], ["q5"], ["q3","q7"]]}
      after op a:   HMR qw b<new>
      before op r:  X qw / CX qv qw / CCX qu qv qw per term, then  Z qw if b<new>
  {"op": "advance_hmr", "b": 17541, "a": 9000}
      move the unconditional HMR at op b to right after op a (refused if its bit is read in between)

Checks, all on every legal x (local analysis tier; the official verifier is the authority):
  1. the edited circuit meets the invsqrt contract (x restored, scratch clean, phase within tolerance) in the
     zero measurement branch and in two random branches;
  2. its per-row phase equals the original's in the zero branch (both circuits are outcome independent);
  3. static checks: inserted ops at condition depth 0, fresh classical bits.
Then the .kmx is converted to .kmb with kickmix and re-coloured by the lifetime allocator (fewest qubits).

Usage: python qhdl_park_rewrite.py IN.kmx EDITS.json OUTDIR [--kickmix-python PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_bitsim as B  # noqa: E402
import qhdl_idle_wires as I  # noqa: E402

import vqcc_config as CFG  # noqa: E402
KMPY = CFG.KMPY


def dickson(terms):
    """Split a GF(2) polynomial of degree <= 2 into (constant, linear vars, [(L1, L2)]) with
    sum of quadratic terms = XOR_k L1_k * L2_k; len(products) = rank/2 (Dickson's normal form).
    L1 = (pivot, [other vars]) meaning pivot XOR others."""
    const = 0
    lin = set()
    quad = set()
    for t in terms:
        t = tuple(t)
        if t == ('1',):
            const ^= 1
        elif len(t) == 1:
            lin ^= {t[0]}
        else:
            quad ^= {tuple(sorted(t))}
    prods = []
    while quad:
        i, j = sorted(quad)[0]
        A = set()  # partners of i other than j
        Cs = set()  # partners of j other than i
        rest = set()
        for (u, v) in quad:
            if (u, v) == (i, j):
                continue
            if i in (u, v):
                A ^= {v if u == i else u}
            elif j in (u, v):
                Cs ^= {v if u == j else u}
            else:
                rest ^= {(u, v)}
        # x_i x_j + x_i A + x_j C = (x_i + C)(x_j + A) + A*C
        for a in A:
            for c in Cs:
                if a == c:
                    lin ^= {a}  # a*a = a
                else:
                    rest ^= {tuple(sorted((a, c)))}
        prods.append(((i, sorted(Cs)), (j, sorted(A))))
        quad = rest
    return const, sorted(lin), prods


def emit_recompute(w, terms):
    const, lin, prods = dickson(terms)
    out = []
    for (p1, o1), (p2, o2) in prods:
        out += [f'CX {v} {p1}' for v in o1] + [f'CX {v} {p2}' for v in o2]
        out.append(f'CCX {p1} {p2} q{w}')
        out += [f'CX {v} {p2}' for v in reversed(o2)] + [f'CX {v} {p1}' for v in reversed(o1)]
    out += [f'CX {v} q{w}' for v in lin]
    if const:
        out.append(f'X q{w}')
    return out, len(prods)


def max_bit(ops):
    m = -1
    for o in ops:
        if o.bit >= 0:
            m = max(m, o.bit)
        if o.cond >= 0:
            m = max(m, o.cond)
    return m


def apply_edits(ops, edits, depth):
    """Return new op-text list. Inserts keyed by position: before[i] / after[i]; removals set."""
    before, after, remove = {}, {}, set()
    nb = max_bit(ops) + 1
    log = []
    for e in edits:
        if e['op'] == 'park':
            w, a, r = e['w'], e['a'], e['r']
            assert depth[a + 1 if a + 1 < len(ops) else a] == 0 or ops[a + 1].code == B.C_POP, ('erase in block', a)
            assert depth[r] == 0, ('recompute in a conditional block', r)
            for i in range(a + 1, r):
                assert w not in ops[i].q, ('wire touched inside park window', w, i)
            bit = nb
            nb += 1
            after.setdefault(a, []).append(f'HMR q{w} b{bit}')
            lines, nand = emit_recompute(w, e['terms'])
            lines.append(f'Z q{w} if b{bit}')
            before.setdefault(r, []).extend(lines)
            log.append(dict(edit={k: v for k, v in e.items() if k != 'terms'}, bit=bit, toffolis=nand))
        elif e['op'] == 'advance_hmr':
            b, a = e['b'], e['a']
            o = ops[b]
            assert o.code == B.C_HMR and o.cond < 0 and depth[b] == 0
            w = o.q[0]
            for i in range(a + 1, b):
                assert w not in ops[i].q, ('wire touched between', i)
                assert ops[i].cond != o.bit and ops[i].bit != o.bit, ('bit used between', i)
            remove.add(b)
            after.setdefault(a, []).append(o.text)
            log.append(dict(edit=e))
    out = []
    for i, o in enumerate(ops):
        out.extend(before.get(i, []))
        if i not in remove:
            out.append(o.text)
        out.extend(after.get(i, []))
    return out, log


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(path, ref_path=None):
    res = {}
    ref = None
    if ref_path:
        ref = B.final_state(ref_path, 'zero')
    for oc in ('zero', 101, 202):
        s = B.final_state(path, oc)
        c = B.contract(s)
        if oc == 'zero' and ref is not None:
            c['phase_equal_to_original_zero_branch'] = bool(np.array_equal(s.phase_units(), ref.phase_units()))
        res[str(oc)] = c
    res['pass'] = all(v['x_restored'] and v['scratch_clean'] and v['phase_ok'] and v['dirty'] == 0
                      for k, v in res.items() if isinstance(v, dict))
    return res


def to_kmb(kmx, kmb):
    code = ('import kickmix as km, sys; c = km.Circuit.from_file(sys.argv[1]); c.to_file(sys.argv[2], format="kmb"); '
            'print(c.num_qubits)')
    out = subprocess.run([KMPY, '-c', code, kmx, kmb], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmx')
    ap.add_argument('edits')
    ap.add_argument('outdir')
    ap.add_argument('--name', default='candidate')
    ap.add_argument('--greedy', action='store_true',
                    help='add edits one at a time, keeping each only if the zero-branch contract and phase equality hold')
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ops, header, xq, nq = B.load(a.kmx)
    depth, *_ = I.structure(ops, xq, nq)
    edits = json.loads(Path(a.edits).read_text())
    rejected = []
    if a.greedy:
        ref = B.final_state(a.kmx, 'zero')
        keep = []
        tmp = out / f'{a.name}_greedy_tmp.kmx'
        for e in edits:
            lines, _ = apply_edits(ops, keep + [e], depth)
            B.write_kmx(tmp, header, lines)
            s_ = B.final_state(str(tmp), 'zero')
            c_ = B.contract(s_)
            ok = (c_['x_restored'] and c_['scratch_clean'] and c_['phase_ok'] and s_.dirty == 0
                  and np.array_equal(s_.phase_units(), ref.phase_units()))
            if ok:
                keep.append(e)
            else:
                rejected.append({k: v for k, v in e.items() if k != 'terms'})
        tmp.unlink()
        edits = keep
    lines, log = apply_edits(ops, edits, depth)
    kmx = out / f'{a.name}.kmx'
    B.write_kmx(kmx, header, lines)
    new_ops, _, _, _ = B.load(str(kmx))
    d2, _, _, prof = I.structure(new_ops, xq, nq)
    rep = dict(input=a.kmx, input_sha256=sha(a.kmx), edits=log, rejected_edits=rejected, kmx=str(kmx), kmx_sha256=sha(kmx),
               peak_width_after=int(prof.max()), ops_at_peak=int((prof >= prof.max()).sum()))
    rep['check'] = check(str(kmx), a.kmx)
    if rep['check']['pass']:
        raw = out / f'{a.name}_raw.kmb'
        rep['raw_kmb_qubits'] = to_kmb(str(kmx), str(raw))
        alloc = out / f'{a.name}.kmb'
        subprocess.run([sys.executable, str(HERE / 'qhdl_lifetime_allocator.py'), str(raw), str(alloc),
                        '--register-gaps', '--report', str(out / f'{a.name}_alloc.json')], check=True)
        rep['kmb'] = str(alloc)
        rep['kmb_sha256'] = sha(alloc)
        rep['alloc'] = json.loads((out / f'{a.name}_alloc.json').read_text())
    rep['seconds'] = round(time.time() - t0, 1)
    (out / f'{a.name}_report.json').write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('alloc',)}, indent=1, default=str))


if __name__ == '__main__':
    main()
