#!/usr/bin/env python3
"""qhdl plugin: `simulate` — bit-parallel classical value simulation of a DQI circuit on valid syndromes.

    qhdl simulate -c <dqi> [--syndromes N] [--seed S] [--weights official|uniform] [--at OP[,OP...]]
                  [--wires q7,q885] [--stop B]

Origin: generalised from an earlier bit-parallel DQI simulator prototype, which
found the pole-parity token and confirmed the source structure of yoyobar's 6d58bd6d0232. Changes here: one verb,
inputs generated in the kickmix venv by the official verifier's own sampler (default: the official error-weight
distribution; `--weights uniform` reproduces the prototype's uniform 0..35), seeded outcomes, per-probe summaries and an end
check.

Model (values only). Each lane is one shot: the 840 syndrome wires hold a valid syndrome from
verify_submission_dqi.sample_shots, the E_j classical registers hold its classical record (mbuc). Ops act on
Python-int bitsets of N lanes: X/CX/CCX/SWAP/CSWAP exactly, R zeroes, HMR zeroes the wire and writes a fair random
outcome per lane (seed), PUSH_CONDITION / *_IF act per lane, BIT_STORE0/1/BIT_XOR on classical bits. Phase ops (Z,
CZ, CCZ, NEG, Z_POW, HMR kickback) are ignored. So a reported value is the computational-basis value in that shot
(every circuit here is a basis permutation plus phases), on N sampled valid inputs: tier sampled(N), never a proof
(rare-event flags with probability ~2^-12 per shot are usually absent from 4,096 lanes).

Reports: at each --at op (state BEFORE that op) the number of wires constant 0 / constant 1 / varying over the lanes,
and per requested wire (default: up to 20 varying scratch wires) the fraction of lanes holding 1. Always: the end
check (every register wire back to its input, every scratch wire |0>, lanes violating). The full probe states go to
an .npz in the workspace (probes, wires x lanes packed bits) for your own analysis.
A 60M-op DQI circuit takes about 20 s at 4,096 lanes (pure Python big-int bit-parallelism; 6d58bd6d0232).
Library: gen_inputs(kmb, out, n, seed, weights) (kickmix venv), simulate(kmb, inputs_npz, probes, stop, seed,
backend='auto').
Backends: 'auto' (default: rust if it loads), 'python'
(the reference loop) or 'rust' (kickmix_analysis.value_sim, crate kmx-analysis in the engine repo; same Mersenne Twister stream, identical
values). rust_module() / rust_state(k, z) are importable by other scripts that run this loop.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import qhdl_kmb as K  # noqa: E402

import vqcc_config as CFG  # noqa: E402
VERIFIER_DIR = CFG.VERIFIERS / 'dqi'
CH = 1 << 20
BACKENDS = ('python', 'rust', 'auto')  # 'auto' (default): rust if kickmix_analysis loads
_KA = None
_AUTO = None


def rust_module():
    """kickmix_analysis with value_sim. Found on sys.path (the analysis venv has a .pth) or in
    KICKMIX_ANALYSIS_PATH (a directory holding the kickmix_analysis package; read even under python -I)."""
    global _KA
    if _KA is None:
        extra = os.environ.get('KICKMIX_ANALYSIS_PATH')
        if extra and extra not in sys.path:
            sys.path.append(extra)
        try:
            import kickmix_analysis as ka
        except ImportError as e:
            raise ImportError("backend='rust' needs kickmix_analysis (crate kmx-analysis): run with "
                              "$QHDL_ANALYSIS_PYTHON (an interpreter with kickmix_analysis) or set "
                              "KICKMIX_ANALYSIS_PATH to .../crates/kmx-analysis/python") from e
        if not hasattr(ka, 'value_sim'):
            raise ImportError(f'kickmix_analysis at {ka.__file__} has no value_sim (needs a build with value_sim)')
        _KA = ka
    return _KA


def check_backend(backend):
    """Resolve a backend choice to 'python' or 'rust'; 'auto' picks rust only if kickmix_analysis loads (the two are
    byte-identical)."""
    global _AUTO
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}')
    if backend == 'auto':
        if _AUTO is None:
            try:
                rust_module()
                _AUTO = 'rust'
            except ImportError:
                _AUTO = 'python'
        return _AUTO
    return backend


def rust_state(k, z):
    """(wires uint64[nq, W], cbits uint64[num_bits + 1, W], L): the Python loop's initial w and cb, packed."""
    ka = rust_module()
    bits, qids, ebits, mbuc = z['bits'], z['qids'].tolist(), z['ebits'].tolist(), z['mbuc']
    deg = int(z['degree']) if 'degree' in z.files else 12
    L = bits.shape[1]
    W = (L + 63) // 64
    nq, nb = int(k.num_qubits), int(k.num_bits)
    wires = np.zeros((nq, W), np.uint64)
    packed = ka.pack_lanes(bits)
    for row, q in enumerate(qids):
        wires[q] = packed[row]
    cb = np.zeros((nb + 1, W), np.uint64)
    for j in range(mbuc.shape[1]):
        col = mbuc[:, j]
        planes = ka.pack_lanes(np.stack([(col >> bi) & 1 for bi in range(deg)]))
        for bi in range(deg):
            cb[ebits[deg * j + bi]] = planes[bi]
    return wires, cb, L


def row_int(row):
    return int.from_bytes(np.ascontiguousarray(row, '<u8').tobytes(), 'little')

_GEN = r'''
import sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import numpy as np, kickmix as km
import verify_submission_dqi as V, qhdl_kmb as K
kmb, out, n, seed, weights = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
spec = V.ProblemSpecDQI.benchmark_instance()
tables = V.FieldTables(km.GF2Field(spec.degree))
rng = np.random.default_rng(seed)
wp = (np.ones(spec.max_errors + 1) / (spec.max_errors + 1)) if weights == 'uniform' else None
syn, mbuc, ph = V.sample_shots(spec, tables, rng, n, wp)
bits = V.to_bits(syn, spec.degree)
k = K.read(kmb)
regs = {nm: cs for nm, cs in k.registers}
qids = []
for i in range(spec.n):
    cs = regs[f'S{i}']
    assert all(isq for isq, _, _ in cs) and len(cs) == spec.degree, f'register S{i}'
    qids.extend(q for _, q, _ in cs)
ebits = []
for j in range(spec.m):
    ebits.extend(b for _, b, _ in regs[f'E{j}'])
np.savez(out, bits=bits, syndromes=syn, qids=np.array(qids), ebits=np.array(ebits), mbuc=mbuc,
         degree=spec.degree)
print('ok', n)
'''


def gen_inputs(kmb, out, n, seed, weights='official', python=None):
    """Valid DQI syndromes from the official sampler (runs in the kickmix venv). Writes an .npz."""
    py = python or CFG.KMPY
    r = subprocess.run([py, '-I', '-c', _GEN, str(VERIFIER_DIR), str(HERE), str(kmb), str(out), str(n), str(seed),
                        weights], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not r.stdout.strip().endswith(str(n)):
        tail = (r.stderr.strip().splitlines() or ['?'])[-1]
        raise ValueError(f'input generation failed in the kickmix venv: {tail[:200]}')
    return out


def _pack(arr_bool):
    return int.from_bytes(np.packbits(arr_bool.astype(np.uint8), bitorder='little').tobytes(), 'little')


def simulate(kmb, inputs, probes=(), stop=None, seed=1, progress=None, backend='auto'):
    """Bit-parallel value simulation. Returns dict(state_at={op: [int per wire]}, end=[int per wire], lanes, ...)."""
    if check_backend(backend) == 'rust':
        return _simulate_rust(kmb, inputs, probes, stop, seed)
    k = K.read(str(kmb))
    t = np.asarray(k.op_types).astype(np.uint8)
    N = len(t)
    B = N if stop is None else min(N, int(stop))
    z = np.load(inputs)
    bits, qids, ebits, mbuc = z['bits'], z['qids'].tolist(), z['ebits'].tolist(), z['mbuc']
    deg = int(z['degree']) if 'degree' in z.files else 12
    L = bits.shape[1]
    ALL = (1 << L) - 1
    rnd = random.Random(seed)
    nq, nb = int(k.num_qubits), int(k.num_bits)
    w = [0] * nq
    for row, q in enumerate(qids):
        w[q] = _pack(bits[row])
    w_in = list(w)
    cb = [0] * (nb + 1)
    for j in range(mbuc.shape[1]):
        col = mbuc[:, j]
        for bi in range(deg):
            cb[ebits[deg * j + bi]] = _pack((col >> bi) & 1)
    probes = sorted({int(p) for p in probes if 0 <= int(p) <= B})
    snaps = {}
    cls = t & 0x30
    # stream offsets per chunk (prefix counts at chunk starts only: no 60M-entry prefix arrays)
    S = {n: k.stream(n) for n in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')}
    o3 = o2 = o1 = ob = oc = 0
    stack = []
    mask = ALL
    t0 = time.time()
    pi = 0
    grb = rnd.getrandbits
    for s in range(0, B, CH):
        e = min(B, s + CH)
        seg = t[s:e]
        cs = cls[s:e]
        n3, n2, n1 = int((cs == 0x30).sum()), int((cs == 0x20).sum()), int((cs == 0x10).sum())
        nbb, ncc = int(((seg & 0x40) > 0).sum()), int(((seg & 0x80) > 0).sum())
        types = seg.tolist()
        A0, A1, A2 = (np.asarray(S[x][o3:o3 + n3]).tolist() for x in ('qqq0', 'qqq1', 'qqq2'))
        P0, P1 = (np.asarray(S[x][o2:o2 + n2]).tolist() for x in ('qq0', 'qq1'))
        Q0 = np.asarray(S['q0'][o1:o1 + n1]).tolist()
        B0 = np.asarray(S['b0'][ob:ob + nbb]).tolist()
        BC = np.asarray(S['bc'][oc:oc + ncc]).tolist()
        o3, o2, o1, ob, oc = o3 + n3, o2 + n2, o1 + n1, ob + nbb, oc + ncc
        i3 = i2 = i1 = ib = ic = 0
        nxt = probes[pi] if pi < len(probes) else -1
        for j, op in enumerate(types):
            i = s + j
            if i == nxt:
                snaps[i] = list(w)
                pi += 1
                nxt = probes[pi] if pi < len(probes) else -1
            if op == 0x06:
                continue
            if op == 0x81:
                stack.append(mask)
                mask &= cb[BC[ic]]
                ic += 1
                continue
            if op == 0x01:
                mask = stack.pop() if stack else ALL
                continue
            mk = mask
            if op & 0x80:
                mk &= cb[BC[ic]]
                ic += 1
            base = op & 0x7F
            c = base & 0x30
            if c == 0x30:
                a, b, d = A0[i3], A1[i3], A2[i3]
                i3 += 1
                if base == 0x31:
                    w[d] ^= w[a] & w[b] & mk
                elif base == 0x32:
                    x = w[a] & (w[b] ^ w[d]) & mk
                    w[b] ^= x
                    w[d] ^= x
            elif c == 0x20:
                a, b = P0[i2], P1[i2]
                i2 += 1
                if base == 0x21:
                    w[b] ^= w[a] & mk
                elif base == 0x22:
                    x = (w[a] ^ w[b]) & mk
                    w[a] ^= x
                    w[b] ^= x
            elif c == 0x10:
                a = Q0[i1]
                i1 += 1
                if base == 0x11:
                    w[a] ^= mk
                elif base == 0x12:
                    w[a] &= ~mk
                elif base == 0x50:
                    w[a] &= ~mk
                    bb = B0[ib]
                    ib += 1
                    cb[bb] = (cb[bb] & ~mk) | (grb(L) & mk)
                elif base & 0x40:
                    ib += 1
            elif base & 0x40:
                bb = B0[ib]
                ib += 1
                if base == 0x40:
                    cb[bb] &= ~mk
                elif base == 0x41:
                    cb[bb] |= mk
                elif base == 0x42:
                    cb[bb] ^= mk
        if progress:
            progress(e, B, time.time() - t0)
    if B in probes and B not in snaps:
        snaps[B] = list(w)
    regs = sorted({q for _, cs in k.registers for isq, q, _ in cs if isq and q < nq})
    return dict(state_at=snaps, end=w, inputs=w_in, lanes=L, ops=B, num_ops=N, registers=regs,
                seconds=round(time.time() - t0, 1), complete=B == N)


def _simulate_rust(kmb, inputs, probes, stop, seed):
    ka = rust_module()
    k = K.read(str(kmb))
    N = len(k.op_types)
    B = N if stop is None else min(N, int(stop))
    z = np.load(inputs)
    wires, cb, L = rust_state(k, z)
    nq = int(k.num_qubits)
    w_in = [row_int(x) for x in wires]
    probes = sorted({int(p) for p in probes if 0 <= int(p) <= B})
    allw = np.arange(nq)
    t0 = time.time()
    r = ka.value_sim(str(kmb), wires, cb, L, seed=seed, points=(probes, [allw] * len(probes)), stop=B)
    pts = r['points']
    snaps = {p: [row_int(x) for x in pts[i * nq:(i + 1) * nq]] for i, p in enumerate(probes)}
    end = [row_int(x) for x in r['end_wires']]
    regs = sorted({q for _, cs in k.registers for isq, q, _ in cs if isq and q < nq})
    return dict(state_at=snaps, end=end, inputs=w_in, lanes=L, ops=B, num_ops=N, registers=regs,
                seconds=round(time.time() - t0, 1), complete=B == N)


def popcount(v):
    return bin(v).count('1')


def summarise(state, lanes, regs, wires=None, limit=20):
    """(counts dict, rows) for one probe state."""
    ALL = (1 << lanes) - 1
    regset = set(regs)
    c0 = sum(1 for v in state if v == 0)
    c1 = sum(1 for v in state if v == ALL)
    counts = dict(const0=c0, const1=c1, varying=len(state) - c0 - c1)
    if wires is None:
        wires = [q for q, v in enumerate(state) if q not in regset and v not in (0, ALL)][:limit]
    rows = []
    for q in wires:
        v = state[q]
        rows.append(dict(wire=f'q{q}', role='register' if q in regset else 'scratch',
                         ones=round(popcount(v) / lanes, 4), value='0' if v == 0 else '1' if v == ALL else 'varies'))
    return counts, rows


def _parse_wires(s):
    if not s:
        return None
    out = []
    for x in s.split(','):
        x = x.strip()
        if x:
            out.append(int(x[1:] if x.startswith('q') else x))
    return out


def v_simulate(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    meta = api.ensure_open(a.circuit)
    sha, s12 = meta['sha256'], meta['sha12']
    src = api.circuit_path(meta)
    if src.suffix != '.kmb':
        raise api.QhdlError('simulate reads .kmb circuits')
    k = K.read(str(src))
    names = {nm for nm, _ in k.registers}
    if 'S0' not in names or 'E0' not in names:
        raise api.QhdlError('simulate needs a DQI circuit (registers S0..S69 and E0..)',
                            'for invsqrt use window/fdiff/exec (exact over all x)')
    N = len(k.op_types)
    at = [int(x) for x in a.at.split(',')] if a.at else []
    for x in at:
        if not 0 <= x <= N:
            raise api.QhdlError(f'--at {x} outside the circuit (0..{N})')
    wires = _parse_wires(a.wires)
    d = api.ws_dir(sha) / 'simulate'
    d.mkdir(parents=True, exist_ok=True)
    inp = d / f'inputs_{a.syndromes}_{a.seed}_{a.weights}.npz'
    if not inp.exists():
        gen_inputs(src, inp, a.syndromes, a.seed, a.weights, python=getattr(api, 'KMPY', None))
    stop = a.stop if a.stop is not None else None
    r = simulate(src, inp, probes=at, stop=stop, seed=a.seed, backend=a.backend)
    L = r['lanes']
    ALL = (1 << L) - 1
    tag = f"{a.syndromes}_{a.seed}_{a.weights}_{'-'.join(map(str, at)) or 'end'}"
    ev = d / f'simulate_{tag}.json'
    npz = d / f'simulate_{tag}.npz'
    keys = sorted(r['state_at'])
    nbytes = (L + 7) // 8
    arr = np.zeros((len(keys) + 1, len(r['end']), nbytes), np.uint8)
    for j, key in enumerate(keys + ['end']):
        st = r['state_at'][key] if key != 'end' else r['end']
        for q, v in enumerate(st):
            arr[j, q] = np.frombuffer(v.to_bytes(nbytes, 'little'), np.uint8)
    np.savez_compressed(npz, probes=np.array(keys + [r['ops']]), state=arr, lanes=L)
    regset = set(r['registers'])
    bad_reg = 0
    bad_lanes = 0
    for q, v in enumerate(r['end']):
        diff = v ^ r['inputs'][q] if q in regset else v
        if diff:
            bad_reg += 1
            bad_lanes |= diff
    rows, probe_facts = [], {}
    for key in keys:
        cnt, rr = summarise(r['state_at'][key], L, r['registers'], wires)
        probe_facts[str(key)] = cnt
        rows += [dict(at=key, **x) for x in rr]
    if not keys:
        cnt, rr = summarise(r['end'], L, r['registers'], wires)
        rows = [dict(at=r['ops'], **x) for x in rr]
    led.earn(f'sampled({L})', f"bit-parallel value simulation on {L} valid syndromes from the official sampler "
                              f"(weights {a.weights}, seed {a.seed}); HMR outcomes fair random per lane; phases ignored")
    endtxt = (f"end check: all registers restored and all scratch |0> on all {L} lanes" if r['complete'] and not bad_reg
              else f"end check: {bad_reg} wires wrong on {popcount(bad_lanes)} lanes" if r['complete']
              else f"stopped at op {r['ops']} (no end check)")
    claim = (f"{s12}: simulated {api.fmt_int(r['ops'])} ops on {L} valid syndromes in {r['seconds']} s; "
             + (f"{len(keys)} probe(s): " + '; '.join(f"@{kk}: {v['varying']} varying, {v['const0']} always 0, "
                                                        f"{v['const1']} always 1" for kk, v in
                                                        list(probe_facts.items())[:3]) + '. ' if keys else '')
             + endtxt + '. Sampled: absence on these lanes is not a proof')
    write = dict(probes=probe_facts, end_bad_wires=bad_reg, end_bad_lanes=popcount(bad_lanes), lanes=L,
                 seconds=r['seconds'], inputs=api.rel(inp), states=api.rel(npz), rows=rows)
    api.write_json(ev, write)
    rr_ = api.cmd('simulate', '-c', s12, '--syndromes', a.syndromes, *(['--seed', a.seed] if a.seed != 1 else []),
                  *(['--weights', a.weights] if a.weights != 'official' else []), *(['--at', a.at] if a.at else []),
                  *(['--wires', a.wires] if a.wires else []), *(['--stop', a.stop] if a.stop is not None else []))
    return E.Envelope(led, claim, domain=f"{L} valid DQI syndromes (official sampler, {a.weights} weights, seed "
                                         f"{a.seed}); values only", evidence=api.rel(ev), rerun=rr_,
                      tool_sha=api.tool_sha(), circuit_sha=sha,
                      facts=dict(states=api.rel(npz), inputs=api.rel(inp), end_bad_wires=bad_reg,
                                 end_bad_lanes=popcount(bad_lanes)),
                      rows=rows, next_cmds=[api.cmd('census', '-c', s12), api.cmd('peak', '-c', s12)])


def qhdl_register(subparsers, api):
    import argparse
    p = subparsers.add_parser('simulate', help='DQI: value of any wire at any op over N valid syndromes (sampled, '
                                               'bit-parallel)',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all', action='store_true', help='print every row')
    p.add_argument('-c', '--circuit', required=True, help='DQI .kmb path or sha prefix')
    p.add_argument('--syndromes', type=int, default=4096, help='lanes (valid syndromes; default 4096)')
    p.add_argument('--seed', type=int, default=1, help='sampler and outcome seed')
    p.add_argument('--weights', choices=['official', 'uniform'], default='official',
                   help='error-weight distribution: the verifier\'s (default) or uniform on 0..max_errors')
    p.add_argument('--at', help='probe op indices (state before each), comma list')
    p.add_argument('--wires', help='wires to report at each probe, e.g. q885,q890 (default: varying scratch)')
    p.add_argument('--stop', type=int, help='stop after this many ops (no end check)')
    p.add_argument('--backend', choices=BACKENDS, default='auto',
                   help='value loop: python (reference), rust (kickmix_analysis) or auto (default)')
    p.set_defaults(fn=lambda a: v_simulate(api, a))
