#!/usr/bin/env python3
"""Circuit surgery: modules, interfaces, compatibility and verified splices.

Formalises hengck23's "circuit surgery" (public discussion 753607): cut modules out of public circuits, paste them
into another, keep a splice only if it passes gates. Definitions:

  module     contiguous op range between two cuts at condition depth 0 (default cuts: top-level INFO_TAG regions);
  interface  state at a cut = per-wire value columns over a fixed sampled input set + the phase column, plus the
             structural live-wire set and crossing classical bits;
  compatible donor B[c1':c2') into host A[c1:c2) iff one relabelling pi makes the entry states equal and the exit
             states equal (multiset of nontrivial per-wire (entry column, exit column) pairs equal) and the phase
             deltas agree, with no non-input classical bit crossing the donor's cuts.

Verbs (one entrypoint; every verb prints/writes a JSON envelope with an evidence tier, never overwrites):

  library  CIRCUIT.kmb... --out DIR      per circuit: modules (op range, static T, canonical structural hash up to
                                         qubit/bit relabelling), live wires and crossing bits at every cut
                                         [tier structural]; engine state dumps at every cut on fixed inputs
                                         (3 seeds x B batches x 256 shots), cumulative mean T at the cuts, phase
                                         column [tier sampled-engine]. Writes DIR/<sha12>.json and DIR/<sha12>.cuts.npz.
  matrix   DIR --out OUT.json            pairwise compatibility per stage and per stage range [structural/sampled].
  splice   --host H.kmb --donor D.kmb --stages a,b --lib DIR --out OUT.kmb
                                         builds host with donor's stages, wires relabelled by the sampled interface
                                         match, donor scratch on fresh ids, then the certified lifetime allocator.
  adaptor  --host H --donor D --lib DIR --cut STAGE[:entry|exit]
                                         classifies the state gap at a cut: equal up to pi / GF(2)-affine (CNOT
                                         adaptor) / nonlinear.

Run with the kickmix venv python and the engine package on PYTHONPATH. Public circuits are data: they are parsed by
qhdl_kmb (our reader) and simulated; nothing from them is executed as code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_kmb as K  # noqa: E402

ROOT = HERE.parent
import vqcc_config as CFG  # noqa: E402
VERIFIER = CFG.VERIFIERS / 'dqi/verify_submission_dqi.py'
ALLOCATOR = HERE / 'qhdl_lifetime_allocator.py'
CCX_LIKE = np.array(K.TOFFOLI_LIKE, dtype=np.uint8)
GROUPS = [('qqq0', 0x30), ('qqq1', 0x30), ('qqq2', 0x30), ('qq0', 0x20), ('qq1', 0x20), ('q0', 0x10)]
MIRROR = {'initial': 'uninitial', 'extract': 'unextract', 'synthesis': 'unsynthesis', 'convert': 'unconvert',
          'mask_projection': 'restore_masks'}


def tool_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def envelope(claim, tier, domain, evidence, rerun, circuit_sha=None, **extra):
    return dict(claim=claim, tier=tier, domain=domain, evidence=str(evidence), rerun=rerun, tool_sha=tool_sha(),
                circuit_sha=circuit_sha, **extra)


def stage_name(tag):
    return tag[:-len('_lookahead')] if tag.endswith('_lookahead') else tag


# ----------------------------------------------------------------------------------------------------------------
# Structure


def top_modules(k):
    """Top-level @enter/@exit regions [a, b) plus untagged gaps, covering [0, num_ops)."""
    t = k.op_types
    tag_ops = np.flatnonzero(t == K.INFO_TAG)
    stack, regions = [], []
    for o, s in zip(tag_ops.tolist(), k.info_tags()):
        kind, _, name = s.partition(':')
        if kind == '@enter':
            stack.append((name, o))
        elif kind == '@exit' and stack and stack[-1][0] == name:
            n, a = stack.pop()
            if not stack:
                regions.append((n, a, o + 1))
    out, pos = [], 0
    for n, a, b in regions:
        if a > pos:
            out.append(('gap', pos, a))
        out.append((n, a, b))
        pos = b
    if pos < len(t):
        out.append(('gap', pos, len(t)))
    # disambiguate repeated stage names
    seen = Counter()
    res = []
    for n, a, b in out:
        st = stage_name(n)
        seen[st] += 1
        res.append(dict(tag=n, stage=st if seen[st] == 1 else f'{st}#{seen[st]}', a=int(a), b=int(b)))
    return res


class Streams:
    """Prefix counters to slice every operand stream by op range."""

    def __init__(self, k):
        self.k = k
        t = k.op_types
        q = t & 0x30
        z = lambda m: np.concatenate([[0], np.cumsum(m, dtype=np.int64)])
        self.c = {0x30: z(q == 0x30), 0x20: z(q == 0x20), 0x10: z(q == 0x10)}
        self.cb = z((t & 0x40) > 0)
        self.cc = z(((t & 0x80) > 0))
        self.ct = z(t == K.INFO_TAG)
        self.s = {n: k.stream(n) for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc']}
        self._tags = None

    def q(self, name, cls, a, b):
        c = self.c[cls]
        return self.s[name][c[a]:c[b]]

    def b0(self, a, b):
        return self.s['b0'][self.cb[a]:self.cb[b]]

    def bc(self, a, b):
        return self.s['bc'][self.cc[a]:self.cc[b]]

    def tags(self, a, b):
        if self._tags is None:
            self._tags = self.k.info_tags()
        return self._tags[self.ct[a]:self.ct[b]]

    def op_index(self, cls, a, b):
        """Op index of each operand row of class cls in [a, b)."""
        t = self.k.op_types[a:b]
        return np.flatnonzero((t & 0x30) == cls) + a


def module_structure(k, S, a, b):
    """Canonical hash up to qubit/bit relabelling, static T, touched qubits with first/last touch, bits touched."""
    t = k.op_types
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(t[a:b]).tobytes())
    qs, os_ = [], []
    for name, cls in GROUPS:
        qs.append(np.asarray(S.q(name, cls, a, b), dtype=np.int64))
        os_.append(S.op_index(cls, a, b))
    slot_len = [len(x) for x in qs]
    Q = np.concatenate(qs)
    O = np.concatenate(os_).astype(np.int64)
    order = np.argsort((Q.astype(np.uint64) << np.uint64(32)) | O.astype(np.uint64), kind='stable')
    sq, so = Q[order], O[order]
    first = np.ones(len(sq), bool)
    first[1:] = sq[1:] != sq[:-1]
    back = np.empty(len(sq), np.int64)
    back[1:] = so[1:] - so[:-1]
    back[first] = -1
    back_orig = np.empty_like(back)
    back_orig[order] = back
    pos = 0
    for n in slot_len:  # per stream group: relabel-invariant back-reference distances
        h.update(back_orig[pos:pos + n].tobytes())
        pos += n
    # bits: same back-reference encoding over the combined b0/bc reference list
    b0 = np.asarray(S.b0(a, b), dtype=np.int64)
    bc = np.asarray(S.bc(a, b), dtype=np.int64)
    ob0 = np.flatnonzero((t[a:b] & 0x40) > 0) + a
    obc = np.flatnonzero((t[a:b] & 0x80) > 0) + a
    B = np.concatenate([b0, bc])
    OB = np.concatenate([ob0, obc]).astype(np.int64)
    ordb = np.argsort((B.astype(np.uint64) << np.uint64(32)) | OB.astype(np.uint64), kind='stable')
    sb, sob = B[ordb], OB[ordb]
    fb = np.ones(len(sb), bool)
    fb[1:] = sb[1:] != sb[:-1]
    bb = np.empty(len(sb), np.int64)
    bb[1:] = sob[1:] - sob[:-1]
    bb[fb] = -1
    bbo = np.empty_like(bb)
    bbo[ordb] = bb
    h.update(bbo[:len(b0)].tobytes())
    h.update(bbo[len(b0):].tobytes())
    # touched qubits: first / last touch op and op types
    last = np.ones(len(sq), bool)
    last[:-1] = sq[1:] != sq[:-1]
    touched = sq[first]
    return dict(canon=h.hexdigest()[:16], static_T=int(np.isin(t[a:b], CCX_LIKE).sum()),
                cswap=int(np.isin(t[a:b], np.array([0x32, 0xB2], np.uint8)).sum()), num_ops=int(b - a),
                touched=touched, first_op=so[first], last_op=so[last],
                bits=np.unique(B))


def module_structure_chunked(k, S, a, b, chunk=12_000_000):
    """module_structure for modules longer than `chunk` ops (memory): canon = hash of per-chunk canons (each chunk
    exact up to its own relabelling; weaker than a whole-module canon), touched/first/last/bits merged."""
    if b - a <= chunk:
        return module_structure(k, S, a, b)
    parts = [module_structure(k, S, x, min(x + chunk, b)) for x in range(a, b, chunk)]
    h = hashlib.sha256(b'chunked')
    for p in parts:
        h.update(p['canon'].encode())
    first, last = {}, {}
    for p in parts:
        for q, f, l in zip(p['touched'].tolist(), p['first_op'].tolist(), p['last_op'].tolist()):
            first.setdefault(q, f)
            last[q] = l
    tq = np.array(sorted(first), np.int64)
    return dict(canon='c' + h.hexdigest()[:15], static_T=sum(p['static_T'] for p in parts),
                cswap=sum(p['cswap'] for p in parts), num_ops=int(b - a), touched=tq,
                first_op=np.array([first[q] for q in tq.tolist()], np.int64),
                last_op=np.array([last[q] for q in tq.tolist()], np.int64),
                bits=np.unique(np.concatenate([p['bits'] for p in parts])))


def live_at_cuts(k, mods, structs, regs_q, e_bits):
    """Structural live-wire set and crossing classical bits at each module start (cut)."""
    t = k.op_types
    depth = K.condition_depth(t)
    is_reset = ((t == K.R) | (t == K.HMR)) & (depth == 0)
    nq = int(k.num_qubits)
    last_op = np.full(nq, -1, np.int64)  # last touch before the cut
    res = []
    bits_before = set()
    bits_after_by_mod = [set(s['bits'].tolist()) for s in structs]
    regset = np.zeros(nq, bool)
    regset[list(regs_q)] = True
    # next touch at/after each cut: process modules in reverse to get "first touch at or after module m"
    next_op = np.full(nq, -1, np.int64)
    next_by_mod = [None] * len(mods)
    for m in range(len(mods) - 1, -1, -1):
        s = structs[m]
        next_op = next_op.copy()
        next_op[s['touched']] = s['first_op']
        next_by_mod[m] = next_op
    after_bits = [set() for _ in mods] + [set()]
    for m in range(len(mods) - 1, -1, -1):
        after_bits[m] = after_bits[m + 1] | bits_after_by_mod[m]
    for m, md in enumerate(mods):
        nx = next_by_mod[m]
        held = (last_op >= 0) & ~is_reset[np.maximum(last_op, 0)]
        nxt_ok = (nx >= 0) & ~((t[np.maximum(nx, 0)] == K.R) & (depth[np.maximum(nx, 0)] == 0))
        live = (held & nxt_ok) | regset
        cross = (bits_before & after_bits[m]) - e_bits
        res.append(dict(stage=md['stage'], cut=md['a'], live=int(live.sum()),
                        live_nonreg=int((live & ~regset).sum()), crossing_bits=len(cross)))
        s = structs[m]
        last_op[s['touched']] = s['last_op']
        bits_before |= bits_after_by_mod[m]
    return res


def bit_triples(k, S, mods):
    """Classical-bit coupling between modules: unique (bit, writer module, reader module) with writer != reader.
    writer = module of the last write before the read (-1: never written, i.e. an input or initial 0). A conditional
    write or BIT_INVERT counts as a read and a write; reads precede writes within one op."""
    t = k.op_types
    ob0 = np.flatnonzero((t & 0x40) > 0)
    obc = np.flatnonzero((t & 0x80) > 0)
    b0 = np.asarray(S.s['b0'], dtype=np.int64)
    bc = np.asarray(S.s['bc'], dtype=np.int64)
    tb = t[ob0]
    rw = ((tb & 0x7F) == 0x42) | ((tb & 0x80) > 0)
    bits = np.concatenate([bc, b0[rw], b0])
    ops = np.concatenate([obc, ob0[rw], ob0]).astype(np.int64)
    isw = np.concatenate([np.zeros(len(bc) + int(rw.sum()), bool), np.ones(len(b0), bool)])
    key = (bits.astype(np.uint64) << np.uint64(34)) | (ops.astype(np.uint64) << np.uint64(1)) | isw.astype(np.uint64)
    o = np.argsort(key, kind='stable')
    bits, ops, isw = bits[o], ops[o], isw[o]
    starts = np.array([m['a'] for m in mods], np.int64)
    mod = np.searchsorted(starts, ops, 'right') - 1
    n = len(bits)
    idx = np.where(isw, np.arange(n), -1)
    grp_start = np.ones(n, bool)
    grp_start[1:] = bits[1:] != bits[:-1]
    # last write index before each event within the same bit (reset at group start)
    gid = np.cumsum(grp_start) - 1
    lastw = np.maximum.accumulate(np.where(isw, np.arange(n), -1))
    first_of_grp = np.flatnonzero(grp_start)
    prev = np.empty(n, np.int64)
    prev[0] = -1
    prev[1:] = lastw[:-1]
    prev[prev < first_of_grp[gid]] = -1
    rd = ~isw
    wm = np.where(prev >= 0, mod[np.maximum(prev, 0)], -1)
    trip = np.stack([bits[rd], wm[rd], mod[rd]], axis=1)
    trip = trip[trip[:, 1] != trip[:, 2]]
    return np.unique(trip, axis=0) if len(trip) else np.zeros((0, 3), np.int64)


# ----------------------------------------------------------------------------------------------------------------
# Engine sampling


def engine():
    import kickmix_engine as ke
    from kickmix_engine.dqi import DQITask
    return ke, DQITask


def sample_inputs(seeds, batches, uniform_batches=0):
    """Circuit-independent inputs: the verifier's sample_shots draw (engine re-implementation) for fixed seeds.
    Per seed: `batches` batches at the natural error-weight distribution, then `uniform_batches` batches with the
    error weight uniform on 0..max_errors, so rare low-weight paths (termination flags) appear in the sample."""
    ke, DQITask = engine()
    task = DQITask(str(VERIFIER), verifier_path=str(VERIFIER))  # path unused for sampling
    nat = task.sampler_args()
    kmax = task.spec.max_errors
    uni = dict(nat, cdf=[float(min(i + 1, kmax + 1)) / (kmax + 1) for i in range(len(nat['cdf']))])
    qin, bin_, ph, tasks = [], [], [], []
    for sd in seeds:
        tl = task.tasks('', seed=sd)
        for args, sel in ((nat, tl[:batches]), (uni, tl[batches:batches + uniform_batches])):
            if not sel:
                continue
            task._sargs = args
            s = task.engine_sample(sel, threads=4, e_rows=np.arange(task.spec.m * task.spec.degree))
            qin.append(s['qin'])
            bin_.append(s['bin'])
            ph.append(s['phase'])
            tasks += sel
    task._sargs = nat
    sims = np.stack([ke.seed_words(x[2]) for x in tasks])
    alt = np.stack([ke.seed_words(x[2] ^ 0x5DEECE66D) for x in tasks])
    return dict(qin=np.concatenate(qin), bin=np.concatenate(bin_), phase=np.concatenate(ph), sims=sims,
                alt_sims=alt, spec=task.spec, tasks=[[int(a), int(b), int(c)] for a, b, c in tasks])


def registers(k):
    regs = {n: c for n, c in k.registers}
    return regs


def syn_and_e(k, spec):
    regs = registers(k)
    syn = [i for j in range(spec.n) for (_, i, _) in regs[f'S{j}']]
    e = [i for j in range(spec.m) for (_, i, _) in regs[f'E{j}']]
    return syn, e


def run_prefix(k, S, c, tmpdir, inp, syn, e_ids, threads):
    """Run ops [0, c) on the sampled inputs, with the fixed simulator seeds and with alternative seeds.
    Returns (dump columns [nq, words], phase words, total Toffolis, value-error batches) for each seed set."""
    ke, _ = engine()
    path = Path(tmpdir) / f'prefix_{os.getpid()}.kmb'
    write_ops(path, k, S, [(0, c, None, None)], registers=(), tags=False, num_qubits=int(k.num_qubits))
    prog = ke.compile_kmb(str(path))
    path.unlink()
    comp = np.asarray(prog.bit_compact, dtype=np.int64)
    e = np.asarray(e_ids, dtype=np.int64)
    ec = np.where(e < len(comp), comp[np.clip(e, 0, max(len(comp) - 1, 0))], -1) if len(comp) else np.full(len(e), -1)
    keep = np.flatnonzero(ec >= 0)
    nq = int(k.num_qubits)
    assert prog.num_qubits == nq, (prog.num_qubits, nq)
    outs = []
    for sims in (inp['sims'], inp['alt_sims']):
        r = ke.run_batches(prog, sims, list(syn), inp['qin'], ec[keep].tolist(), inp['bin'][:, keep, :],
                           threads=threads, dump=True)
        dq = np.ascontiguousarray(np.transpose(r['dump_q'], (1, 0, 2))).reshape(nq, -1)
        bh = np.zeros(int(k.num_bits), np.uint64)
        used = np.flatnonzero(comp[:int(k.num_bits)] >= 0) if len(comp) else np.zeros(0, np.int64)
        if len(used):
            db = np.transpose(r['dump_bits'], (1, 0, 2)).reshape(r['dump_bits'].shape[1], -1)
            bh[used] = col_hash(db[comp[used]])
        outs.append((dq, r['phase'].reshape(-1), np.asarray(r['toffoli_count'], np.int64), int(r['val_err'].any(axis=1).sum()),
                     bh))
    return outs


_W = None


def col_hash(cols):
    """64-bit fingerprint per row of uint64 words (0 for an all-zero row)."""
    global _W
    n = cols.shape[1]
    if _W is None or len(_W) < n:
        _W = np.random.default_rng(12345).integers(1, 1 << 63, size=max(n, 64), dtype=np.uint64) | np.uint64(1)
    with np.errstate(over='ignore'):
        x = cols.astype(np.uint64)
        h = (x * _W[:n]).sum(axis=1, dtype=np.uint64)
        h ^= ((x ^ (x >> np.uint64(29))) * _W[:n][::-1]).sum(axis=1, dtype=np.uint64)
    h[(cols == 0).all(axis=1)] = 0
    return h


def write_ops(path, k, S, pieces, registers=(), tags=True, extra_bits=None, num_qubits=None, num_bits=None):
    """Write a .kmb from pieces [(a, b, qmap, bmap)] of circuit k (or (kmb, Streams, a, b, qmap, bmap) for another
    circuit). qmap/bmap: None or int arrays mapping old id -> new id."""
    ots, st, tg = [], defaultdict(list), []
    synth, pos = [], 0
    for p in pieces:
        if p[0] == 'resets':  # synthetic unconditional R on wires known (sampled) to hold |0>
            w = np.asarray(p[1], dtype=np.uint32)
            synth.append(np.arange(pos, pos + len(w)))
            pos += len(w)
            ots.append(np.full(len(w), K.R, np.uint8))
            for name, _ in GROUPS:
                st[name].append(w if name == 'q0' else np.zeros(0, np.uint32))
            for name in ('b0', 'bc'):
                st[name].append(np.zeros(0, np.uint32))
            continue
        if len(p) == 4:
            kk, SS, (a, b, qmap, bmap) = k, S, p
        else:
            kk, SS, a, b, qmap, bmap = p
        ots.append(np.asarray(kk.op_types[a:b]))
        pos += int(b - a)
        for name, cls in GROUPS:
            x = np.asarray(SS.q(name, cls, a, b))
            st[name].append(qmap[x] if qmap is not None else x)
        for name, f in (('b0', SS.b0), ('bc', SS.bc)):
            x = np.asarray(f(a, b))
            st[name].append(bmap[x] if bmap is not None else x)
        if tags:
            tg += [s.encode() for s in SS.tags(a, b)]
    ot = np.concatenate(ots)
    streams = {n: np.concatenate(v).astype(np.uint32) for n, v in st.items()}
    if not tags:
        keep = ot != K.INFO_TAG
        ot = ot[keep]
    K.write_kmb(str(path), ot, streams, b'', tg, registers, num_qubits=num_qubits, num_bits=num_bits)
    return np.concatenate(synth) if synth else np.zeros(0, np.int64)


def strip_ops(src, dst, positions):
    """Copy src .kmb without the 1-qubit ops at `positions` (synthetic R ops); registers and tags kept."""
    k = K.read(src)
    t = np.asarray(k.op_types)
    assert np.all(t[positions] == K.R)
    keep = np.ones(len(t), bool)
    keep[positions] = False
    q1 = np.flatnonzero((t & 0x30) == 0x10)
    keep1 = keep[q1]
    streams = {n: np.asarray(k.stream(n)) for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'b0', 'bc']}
    streams['q0'] = np.asarray(k.stream('q0'))[keep1]
    regs = [(n, [(isq, i) for isq, i, _ in cs]) for n, cs in k.registers]
    K.write_kmb(str(dst), t[keep], streams, b'', [x.encode() for x in k.info_tags()], regs,
                num_qubits=int(k.num_qubits))


def cmd_library(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(',')]
    inp = sample_inputs(seeds, a.batches, a.uniform_batches)
    spec = inp['spec']
    for path in a.circuits:
        t0 = time.time()
        sha = sha256(path)
        js, nz = out / f'{sha[:12]}.json', out / f'{sha[:12]}.cuts.npz'
        if js.exists() and nz.exists():
            print(f'skip {sha[:12]} (exists)')
            continue
        k = K.read(path)
        assert k.counts['angle'] == 0
        S = Streams(k)
        mods = top_modules(k)
        depth = K.condition_depth(k.op_types)
        structs = [module_structure_chunked(k, S, m['a'], m['b']) for m in mods]
        syn, e_ids = syn_and_e(k, spec)
        live = live_at_cuts(k, mods, structs, set(syn), set(e_ids))
        cuts = [m['a'] for m in mods] + [len(k.op_types)]
        cols, phases, tofs, verrs, alt_phase_eq, alt_col_eq = [], [], [], [], [], []
        phases_alt, bhs, bhs_alt = [], [], []
        for c in cuts:
            if c == 0:
                dq = np.zeros((int(k.num_qubits), inp['qin'].shape[0] * 4), np.uint64)
                for i, q in enumerate(syn):
                    dq[q] = inp['qin'][:, i, :].reshape(-1)
                ph, tf, ve = np.zeros(inp['qin'].shape[0] * 4, np.uint64), np.zeros(inp['qin'].shape[0], np.int64), 0
                ph2 = ph
                bh = np.zeros(int(k.num_bits), np.uint64)
                ecols = np.transpose(inp['bin'], (1, 0, 2)).reshape(len(e_ids), -1)
                bh[np.asarray(e_ids)] = col_hash(ecols)
                bh2 = bh
                alt_ok = alt_c = True
            else:
                (dq, ph, tf, ve, bh), (dq2, ph2, _, _, bh2) = run_prefix(k, S, c, tmp, inp, syn, e_ids, a.threads)
                alt_ok = bool(np.array_equal(ph, ph2))
                alt_c = bool(np.array_equal(dq, dq2))
            cols.append(dq)
            phases.append(ph)
            phases_alt.append(ph2)
            bhs.append(bh)
            bhs_alt.append(bh2)
            tofs.append(tf)
            verrs.append(ve)
            alt_phase_eq.append(alt_ok)
            alt_col_eq.append(alt_c)
        nshots = inp['qin'].shape[0] * 256
        natmask = np.array(([True] * a.batches + [False] * a.uniform_batches) * len(seeds))
        nnat = int(natmask.sum()) * 256
        tofn = [int(x[natmask].sum()) for x in tofs]
        tofs = [int(x.sum()) for x in tofs]
        modrows = []
        for i, (m, s) in enumerate(zip(mods, structs)):
            modrows.append(dict(stage=m['stage'], tag=m['tag'], ops=[m['a'], m['b']], depth_at_cut=int(depth[m['a']]),
                                num_ops=s['num_ops'], static_T=s['static_T'], cswap=s['cswap'], canon=s['canon'],
                                touched_qubits=int(len(s['touched'])), bits_touched=int(len(s['bits'])),
                                mean_T=(tofs[i + 1] - tofs[i]) / nshots,
                                mean_T_natural=(tofn[i + 1] - tofn[i]) / nnat,
                                entry=live[i], nonzero_wires_entry=int((cols[i] != 0).any(axis=1).sum()),
                                nonzero_wires_exit=int((cols[i + 1] != 0).any(axis=1).sum()),
                                entry_state_seed_independent=alt_col_eq[i] and alt_phase_eq[i],
                                exit_state_seed_independent=alt_col_eq[i + 1] and alt_phase_eq[i + 1]))
        rec = envelope(f'module library of {sha[:12]}', 'structural + sampled-engine', dict(
            seeds=seeds, batches_per_seed=a.batches, uniform_weight_batches_per_seed=a.uniform_batches, shots=nshots, sampler='engine sample_dqi (exact sample_shots)'),
            nz, 'qhdl_surgery.py library ' + str(path), sha,
            path=str(path), num_qubits=int(k.num_qubits), num_bits=int(k.num_bits), num_ops=int(len(k.op_types)),
            static_T=int(np.isin(k.op_types, CCX_LIKE).sum()), mean_T_sampled=tofs[-1] / nshots,
            mean_T_natural=tofn[-1] / nnat,
            final_value_error_batches=verrs[-1], modules=modrows, seconds=round(time.time() - t0, 1))
        trip = bit_triples(k, S, mods)
        np.savez_compressed(nz, trip=trip, cols=np.stack(cols), phase=np.stack(phases), phase_alt=np.stack(phases_alt),
                            bith=np.stack(bhs), bith_alt=np.stack(bhs_alt), toffoli=np.array(tofs),
                            cuts=np.array(cuts), tasks=np.array(inp['tasks'], dtype=object))
        js.write_text(json.dumps(rec, indent=1) + '\n')
        print(f"{sha[:12]} Q{k.num_qubits} meanT(nat) {tofn[-1] / nnat:.1f} mods {len(mods)} {rec['seconds']}s",
              flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Interfaces and compatibility


def load_lib(d, sha12):
    js = json.loads((Path(d) / f'{sha12}.json').read_text())
    nz = np.load(Path(d) / f'{sha12}.cuts.npz', allow_pickle=True)
    return js, nz


def stage_index(js):
    return {m['stage']: i for i, m in enumerate(js['modules'])}


def pair_keys(cols_in, cols_out):
    """Per wire: hashable key of (entry column, exit column); trivial wires (0, 0) -> None."""
    keys = []
    nzi = (cols_in != 0).any(axis=1)
    nzo = (cols_out != 0).any(axis=1)
    for w in range(cols_in.shape[0]):
        if not nzi[w] and not nzo[w]:
            keys.append(None)
        else:
            keys.append(cols_in[w].tobytes() + b'|' + cols_out[w].tobytes())
    return keys


def col_multiset(cols):
    nz = (cols != 0).any(axis=1)
    return Counter(cols[w].tobytes() for w in np.flatnonzero(nz))


def runs_of(js, stages):
    """Module index runs [(i0, i1)] (i1 exclusive) covering the stage set, in order."""
    si = stage_index(js)
    idx = sorted(si[s] for s in stages)
    out = []
    for i in idx:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [tuple(x) for x in out]


def bit_closure(nz, js, runs):
    """Classical-bit coupling of a module set. Returns dict of bit sets keyed by role (see compat)."""
    trip = nz['trip']
    inset = np.zeros(len(js['modules']) + 1, bool)
    for i0, i1 in runs:
        inset[i0:i1] = True
    w, r = trip[:, 1], trip[:, 2]
    win = np.where(w >= 0, inset[np.maximum(w, 0)], False)
    rin = inset[r]
    run_of = np.full(len(js['modules']), -1)
    for j, (i0, i1) in enumerate(runs):
        run_of[i0:i1] = j
    bith, bitha = nz['bith'], nz['bith_alt']
    imp = trip[rin & ~win]  # read inside, written outside (or never written)
    exp = trip[win & ~rin]  # written inside, read outside
    internal = trip[win & rin]

    def det_at(bitsel, cut_idx):
        return bith[cut_idx, bitsel] == bitha[cut_idx, bitsel]
    imp_cut = np.array([runs[run_of[x]][0] for x in imp[:, 2]], np.int64) if len(imp) else np.zeros(0, np.int64)
    exp_cut = np.array([runs[run_of[x]][1] for x in exp[:, 1]], np.int64) if len(exp) else np.zeros(0, np.int64)
    imp_det = det_at(imp[:, 0], imp_cut) if len(imp) else np.zeros(0, bool)
    exp_det = det_at(exp[:, 0], exp_cut) if len(exp) else np.zeros(0, bool)
    return dict(imports=imp, import_cut=imp_cut, import_det=imp_det, exports=exp, export_cut=exp_cut,
                export_det=exp_det, internal=internal)


_EXP = {}


def expected_phase(js):
    """Expected phase words of the library's sampled inputs (re-drawn deterministically from the recorded seeds)."""
    d = js['domain']
    key = (tuple(d['seeds']), d['batches_per_seed'], d.get('uniform_weight_batches_per_seed', 0))
    if key not in _EXP:
        _EXP[key] = sample_inputs(list(key[0]), key[1], key[2])['phase'].reshape(-1)
    return _EXP[key]


def shot_ok(js, nz):
    """uint64 words: 1 where the circuit is correct on that sampled shot (inputs restored, ancillas clean, phase
    as expected). The reachable domain for interface comparisons is the set of shots both circuits get right."""
    c = nz['cols']
    bad = np.bitwise_or.reduce(c[-1] ^ c[0], axis=0)
    bad |= nz['phase'][-1] ^ expected_phase(js)
    return ~bad


def masked(nz, mask):
    out = dict(nz)
    out['cols'] = nz['cols'] & mask
    out['phase'] = nz['phase'] & mask
    out['phase_alt'] = nz['phase_alt'] & mask
    return out


def compat(hjs, hnz, djs, dnz, stages, detail=False):
    """Compatibility of donor's stage set into host's (the compatibility definition in the module docstring), per run plus bit closure, on the
    sampled shots both circuits get right end to end."""
    hr, dr = runs_of(hjs, stages), runs_of(djs, stages)
    if [b - a for a, b in hr] != [b - a for a, b in dr]:
        return dict(stages=list(stages), compatible=False, reason='run shapes differ')
    mask = shot_ok(hjs, hnz) & shot_ok(djs, dnz)
    domain_shots = int(sum(bin(int(x)).count('1') for x in mask))
    hnz, dnz = masked(hnz, mask), masked(dnz, mask)
    hc, dc = hnz['cols'], dnz['cols']
    run_rows, ok = [], True
    for (h0, h1), (d0, d1) in zip(hr, dr):
        e_eq = col_multiset(hc[h0]) == col_multiset(dc[d0])
        x_eq = col_multiset(hc[h1]) == col_multiset(dc[d1])
        hk = Counter(x for x in pair_keys(hc[h0], hc[h1]) if x is not None)
        dk = Counter(x for x in pair_keys(dc[d0], dc[d1]) if x is not None)
        p_eq = hk == dk
        run_rows.append(dict(host_cuts=[h0, h1], donor_cuts=[d0, d1], entry_state_equal=e_eq, exit_state_equal=x_eq,
                             wire_pairs_equal=p_eq, unmatched_pairs=sum((dk - hk).values())))
        ok &= p_eq
    # phase: XOR of run deltas, under both simulator-seed sets
    def delta(nz, runs, key):
        d = np.zeros_like(nz[key][0])
        for a, b in runs:
            d ^= nz[key][a] ^ nz[key][b]
        return d
    hd, hda = delta(hnz, hr, 'phase'), delta(hnz, hr, 'phase_alt')
    dd, dda = delta(dnz, dr, 'phase'), delta(dnz, dr, 'phase_alt')
    host_det, donor_det = bool(np.array_equal(hd, hda)), bool(np.array_equal(dd, dda))
    ph_eq = host_det and donor_det and bool(np.array_equal(hd, dd))
    ok &= ph_eq
    hb, db = bit_closure(hnz, hjs, hr), bit_closure(dnz, djs, dr)
    # donor imports: random -> coupled to a donor module outside the set; deterministic nonzero -> need host bit
    d_imp_random = int((~db['import_det']).sum())
    need = set()
    for (bit, _, _), cut, det in zip(db['imports'].tolist(), db['import_cut'].tolist(), db['import_det'].tolist()):
        h = int(dnz['bith'][cut, bit])
        if det and h:
            need.add((h, cut))
    # host bits available at the matching host cut (deterministic, same hash)
    cutmap = {d[0]: h[0] for h, d in zip(hr, dr)}
    missing_imports = 0
    hav = {}
    for h, cut in need:
        hc_ = cutmap[cut]
        if hc_ not in hav:
            hh, hha = hnz['bith'][hc_], hnz['bith_alt'][hc_]
            hav[hc_] = set(hh[(hh == hha) & (hh != 0)].tolist())
        if h not in hav[hc_]:
            missing_imports += 1
    d_exp_random = int((~db['export_det']).sum())
    h_imp_random = int((~hb['import_det']).sum())
    h_exp_random = int((~hb['export_det']).sum())
    # host deterministic exports must be reproduced: a donor bit with the same hash at the donor exit cut, or the
    # host bit already holding that value at the host entry
    exitmap = {h[1]: d[1] for h, d in zip(hr, dr)}
    entrymap = {h[1]: h[0] for h in hr}
    missing_exports = 0
    dav = {}
    for (bit, _, _), cut, det in zip(hb['exports'].tolist(), hb['export_cut'].tolist(), hb['export_det'].tolist()):
        if not det:
            continue
        h = int(hnz['bith'][cut, bit])
        if h == 0 or int(hnz['bith'][entrymap[cut], bit]) == h:
            continue
        dc_ = exitmap[cut]
        if dc_ not in dav:
            dh, dha = dnz['bith'][dc_], dnz['bith_alt'][dc_]
            dav[dc_] = set(dh[(dh == dha) & (dh != 0)].tolist())
        if h not in dav[dc_]:
            missing_exports += 1
    bits_ok = d_imp_random == 0 and d_exp_random == 0 and h_imp_random == 0 and h_exp_random == 0 \
        and missing_imports == 0 and missing_exports == 0
    ok &= bits_ok
    hm = [hjs['modules'][i] for a, b in hr for i in range(a, b)]
    dm = [djs['modules'][i] for a, b in dr for i in range(a, b)]
    same_canon = [x['canon'] == y['canon'] for x, y in zip(hm, dm)]
    dT = sum(m['mean_T_natural'] for m in dm) - sum(m['mean_T_natural'] for m in hm)
    reasons = []
    if not all(r['wire_pairs_equal'] for r in run_rows):
        reasons.append('qubit interface')
    if not ph_eq:
        reasons.append('phase delta' + ('' if host_det and donor_det else ' (seed-dependent)'))
    if not bits_ok:
        reasons.append('bit coupling')
    res = dict(stages=list(stages), domain_shots=domain_shots, runs=run_rows, structural_identical=all(same_canon),
               canon_equal=same_canon, phase_delta_equal=ph_eq, phase_delta_det=[host_det, donor_det],
               donor_imports_random=d_imp_random, donor_exports_random=d_exp_random,
               host_imports_random=h_imp_random, host_exports_random=h_exp_random,
               missing_det_imports=missing_imports, missing_det_exports=missing_exports,
               compatible=bool(ok), reasons=reasons, dT_sampled=dT,
               host_live_entry=hm[0]['entry']['live'], donor_live_entry=dm[0]['entry']['live'])
    return res


STAGE_SETS = [['initial'], ['extract'], ['synthesis'], ['convert'], ['scan'], ['unconvert'], ['unsynthesis'],
              ['unextract'], ['uninitial'],
              ['convert', 'unconvert'], ['synthesis', 'unsynthesis'], ['extract', 'unextract'],
              ['convert', 'scan', 'unconvert'], ['synthesis', 'convert', 'unconvert', 'unsynthesis'],
              ['synthesis', 'convert', 'scan', 'unconvert', 'unsynthesis'],
              ['extract', 'synthesis', 'unsynthesis', 'unextract'],
              ['extract', 'synthesis', 'convert', 'unconvert', 'unsynthesis', 'unextract'],
              ['extract', 'synthesis', 'convert', 'scan', 'unconvert', 'unsynthesis', 'unextract']]


def cmd_matrix(a):
    d = Path(a.lib)
    libs = {}
    for js in sorted(d.glob('*.json')):
        j = json.loads(js.read_text())
        if 'modules' in j:
            libs[js.stem] = j
    nzs = {s: dict(np.load(d / f'{s}.cuts.npz', allow_pickle=True)) for s in libs}
    rows = []
    for rg in STAGE_SETS:
        for h in libs:
            for dn in libs:
                if h == dn:
                    continue
                if not all(s in stage_index(libs[h]) and s in stage_index(libs[dn]) for s in rg):
                    continue
                c = compat(libs[h], nzs[h], libs[dn], nzs[dn], rg)
                c.update(host=h, donor=dn)
                rows.append(c)
        print('+'.join(rg), sum(r['compatible'] for r in rows if r['stages'] == rg), flush=True)
    classes = {}
    for rg in STAGE_SETS:
        key = '+'.join(rg)
        parent = {s: s for s in libs}

        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for r in rows:
            if r['stages'] == rg and r['compatible']:
                parent[f(r['host'])] = f(r['donor'])
        grp = defaultdict(list)
        for s in libs:
            if all(x in stage_index(libs[s]) for x in rg):
                grp[f(s)].append(s)
        classes[key] = sorted((sorted(g) for g in grp.values()), key=lambda g: (-len(g), g))
    if Path(a.out).exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    res = envelope('pairwise module compatibility (stage sets)', 'sampled-engine',
                   dict(lib=str(d), circuits=len(libs)), a.out, f'qhdl_surgery.py matrix {d} --out {a.out}',
                   pairs=rows, classes=classes)
    Path(a.out).write_text(json.dumps(res, indent=1, default=int) + '\n')
    for key, g in classes.items():
        print(key, [len(x) for x in g])


# ----------------------------------------------------------------------------------------------------------------
# Splice


def match_wires(hc0, hc1, dc0, dc1, dtouched, hregpos, dregpos):
    """pi for donor-touched wires: donor wire -> host wire with the same (entry, exit) column pair.
    Prefers the same register position. Raises if the interfaces differ."""
    hk = pair_keys(hc0, hc1)
    dk = pair_keys(dc0, dc1)
    by = defaultdict(list)
    for w, x in enumerate(hk):
        if x is not None:
            by[x].append(w)
    need = Counter(x for x in dk if x is not None)
    have = Counter({x: len(v) for x, v in by.items()})
    if need != have:
        raise ValueError(f'interface mismatch: {sum((need - have).values())} donor pairs unmatched, '
                         f'{sum((have - need).values())} host pairs unmatched')
    pi, used = {}, set()
    for w in dtouched.tolist():
        x = dk[w]
        rp = dregpos.get(w)
        if x is not None and rp is not None and rp in hregpos:
            hw = hregpos[rp]
            if hk[hw] == x and hw not in used:
                pi[w] = hw
                used.add(hw)
    # donor register wires that hold |0> at both cuts (register gaps used as scratch): keep them on the host register
    # wire of the same position when that wire is also |0> at both host cuts (registers are pinned, so this costs
    # no width; a fresh id would)
    for w in dtouched.tolist():
        rp = dregpos.get(w)
        if dk[w] is None and rp is not None and rp in hregpos:
            hw = hregpos[rp]
            if hk[hw] is None and hw not in used:
                pi[w] = hw
                used.add(hw)
    for w in dtouched.tolist():
        x = dk[w]
        if x is None or w in pi:
            continue
        for hw in by[x]:
            if hw not in used:
                pi[w] = hw
                used.add(hw)
                break
        else:
            raise ValueError('ran out of host wires in a class')
    return pi


def cmd_splice(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    stages = a.stages.split(',')
    hk_, dk_ = K.read(a.host), K.read(a.donor)
    hsha, dsha = sha256(a.host), sha256(a.donor)
    hjs, hnz = load_lib(a.lib, hsha[:12])
    djs, dnz = load_lib(a.lib, dsha[:12])
    hnz, dnz = dict(hnz), dict(dnz)
    HS, DS = Streams(hk_), Streams(dk_)
    c = compat(hjs, hnz, djs, dnz, stages)
    mask = shot_ok(hjs, hnz) & shot_ok(djs, dnz)
    hnz, dnz = masked(hnz, mask), masked(dnz, mask)
    if not c['compatible'] and not a.force:
        raise SystemExit(json.dumps(dict(error='incompatible', compat=c), default=int))
    hr, dr = runs_of(hjs, stages), runs_of(djs, stages)
    hreg = {(n, j): i for n, cs in hk_.registers for j, (isq, i, _) in enumerate(cs) if isq}
    dregpos = {i: (n, j) for n, cs in dk_.registers for j, (isq, i, _) in enumerate(cs) if isq}
    hE = {(n, j): i for n, cs in hk_.registers for j, (isq, i, _) in enumerate(cs) if not isq}
    dE = {i: (n, j) for n, cs in dk_.registers for j, (isq, i, _) in enumerate(cs) if not isq}
    nq_next = int(hk_.num_qubits) + 64
    nb_next = int(hk_.num_bits)
    bmap = np.full(int(dk_.num_bits) + 1, -1, np.int64)
    pieces, pos, info = [], 0, []
    db_imp = bit_closure(dnz, djs, dr)
    hb_cl = bit_closure(hnz, hjs, hr)
    ddep = K.condition_depth(dk_.op_types)
    d0pos = np.concatenate([np.flatnonzero(ddep == 0), [len(dk_.op_types)]])
    del ddep
    for (h0, h1), (d0, d1) in zip(hr, dr):
        ha, hb = hjs['modules'][h0]['ops'][0], hjs['modules'][h1 - 1]['ops'][1]
        da, db = djs['modules'][d0]['ops'][0], djs['modules'][d1 - 1]['ops'][1]
        dtouched = np.unique(np.concatenate([np.asarray(DS.q(n, cl, da, db)) for n, cl in GROUPS]))
        pi = match_wires(hnz['cols'][h0], hnz['cols'][h1], dnz['cols'][d0], dnz['cols'][d1], dtouched, hreg,
                         dregpos)
        qmap = np.full(int(dk_.num_qubits), -1, np.int64)
        fresh = 0
        for w in dtouched.tolist():
            if w in pi:
                qmap[w] = pi[w]
            else:
                qmap[w] = nq_next
                nq_next += 1
                fresh += 1
        # bits read before written in this donor run, from outside the set: E inputs by register position,
        # deterministic values by hash from a host bit at the host entry cut, constant 0 -> fresh
        dbits = np.unique(np.concatenate([np.asarray(DS.b0(da, db)), np.asarray(DS.bc(da, db))]))
        imp = db_imp['imports']
        imp_here = set(imp[(db_imp['import_cut'] == d0)][:, 0].tolist()) if len(imp) else set()
        hh, hha = hnz['bith'][h0], hnz['bith_alt'][h0]
        hash2host = {}
        for b_, (x, y) in enumerate(zip(hh.tolist(), hha.tolist())):
            if x == y and x != 0:
                hash2host.setdefault(x, b_)
        n_e = n_det = n_zero = 0
        for x in dbits.tolist():
            if bmap[x] >= 0:
                continue  # written in an earlier run of the set
            if x in imp_here:
                if x in dE:
                    bmap[x] = hE[dE[x]]
                    n_e += 1
                    continue
                hval = int(dnz['bith'][d0, x])
                if hval != 0:
                    bmap[x] = hash2host[hval]
                    n_det += 1
                    continue
                n_zero += 1
            bmap[x] = nb_next
            nb_next += 1
        info.append(dict(host_ops=[ha, hb], donor_ops=[da, db], donor_touched=int(len(dtouched)), mapped=len(pi),
                         fresh_qubits=fresh, bits_E=n_e, bits_det_from_host=n_det, bits_zero_import=n_zero))
        pieces.append((hk_, HS, pos, ha, None, None))
        regq = np.zeros(int(hk_.num_qubits), bool)
        regq[list(hreg.values())] = True
        z_in = np.flatnonzero(~(hnz['cols'][h0] != 0).any(axis=1) & ~regq)
        z_out = np.flatnonzero(~(hnz['cols'][h1] != 0).any(axis=1) & ~regq)
        if not a.no_resets:
            pieces.append(('resets', z_in))
        # mapped donor wires that hold |0> at the exit: reset right after their last touch in the run, so the
        # allocator can end the interval where the donor's own interval ended
        dz_out = ~(dnz['cols'][d1] != 0).any(axis=1)
        last_touch = {}
        for n, cl in GROUPS:
            qq = np.asarray(DS.q(n, cl, da, db))
            oo = DS.op_index(cl, da, db)
            if len(qq):
                o_ = np.argsort(qq, kind='stable')
                qs_, os_ = qq[o_], oo[o_]
                lastm = np.ones(len(qs_), bool)
                lastm[:-1] = qs_[1:] != qs_[:-1]
                for q_, op_ in zip(qs_[lastm].tolist(), os_[lastm].tolist()):
                    if op_ > last_touch.get(q_, -1):
                        last_touch[q_] = op_
        z0 = d0pos[(d0pos > da) & (d0pos <= db)]  # insertion points: before an op at condition depth 0
        early = sorted((int(z0[np.searchsorted(z0, op_ + 1)]) - 1, int(pi[q_])) for q_, op_ in last_touch.items()
                       if q_ in pi and dz_out[q_] and not a.no_resets)
        cur = da
        i = 0
        while i < len(early):
            op_ = early[i][0]
            ws = []
            while i < len(early) and early[i][0] == op_:
                ws.append(early[i][1])
                i += 1
            pieces.append(('donor', cur, op_ + 1, qmap.copy()))
            pieces.append(('resets', ws))
            cur = op_ + 1
        pieces.append(('donor', cur, db, qmap.copy()))
        if not a.no_resets:
            pieces.append(('resets', z_out))
        info[-1].update(resets_entry=int(len(z_in)), resets_exit=int(len(z_out)), resets_after_last_touch=len(early))
        pos = hb
    # host deterministic exports: the donor bit carrying the same value at the donor exit takes the host bit's id
    remap = {}
    exitmap = {h[1]: d[1] for h, d in zip(hr, dr)}
    entrymap = {h[1]: h[0] for h in hr}
    for (bit, _, _), cut, det in zip(hb_cl['exports'].tolist(), hb_cl['export_cut'].tolist(),
                                     hb_cl['export_det'].tolist()):
        hval = int(hnz['bith'][cut, bit])
        if not det or hval == 0 or int(hnz['bith'][entrymap[cut], bit]) == hval:
            continue
        dc_ = exitmap[cut]
        cand = np.flatnonzero((dnz['bith'][dc_] == hval) & (dnz['bith_alt'][dc_] == hval) & (bmap[:len(dnz['bith'][dc_])] >= int(hk_.num_bits)))
        if not len(cand):
            raise SystemExit(f'host export bit {bit} has no donor counterpart')
        remap[int(bmap[cand[0]])] = bit
    if remap:
        for x in np.flatnonzero(bmap >= 0).tolist():
            bmap[x] = remap.get(int(bmap[x]), int(bmap[x]))
    final = []
    for p in pieces:
        if p[0] == 'donor':
            _, da, db, qmap = p
            final.append((dk_, DS, da, db, qmap, bmap))
        else:
            final.append(p)
    final.append((hk_, HS, pos, len(hk_.op_types), None, None))
    regs = [(n, [(isq, i) for isq, i, _ in cs]) for n, cs in hk_.registers]
    raw = Path(a.tmp) / (out.stem + '.raw.kmb')
    synth = write_ops(raw, hk_, HS, final, registers=regs)
    r = subprocess.run([sys.executable, str(ALLOCATOR), str(raw), str(out), '--report', str(out) + '.alloc.json'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('allocator failed: ' + r.stderr[-2000:])
    al = json.loads(Path(str(out) + '.alloc.json').read_text())
    raw.unlink()
    osha = sha256(out)
    stripped = None
    if a.strip_resets and len(synth):
        sp = Path(str(out).replace('.kmb', '.stripped.kmb'))
        if sp.exists():
            raise SystemExit(f'refusing to overwrite {sp}')
        strip_ops(out, sp, synth)
        stripped = dict(path=str(sp), sha256=sha256(sp), removed_resets=int(len(synth)),
                        note='synthetic R ops removed after colouring: wire reuse at those points rests on the '
                             'sampled |0> claim (tier sampled), not on the allocator certificate')
    rep = dict(host=a.host, host_sha256=hsha, donor=a.donor, donor_sha256=dsha, stages=stages, compat=c, runs=info,
               host_exports_remapped=len(remap), synthetic_resets=int(len(synth)), stripped=stripped,
               allocator=dict(lower_bound=al['lower_bound'], num_qubits_out=al['num_qubits_out'],
                              equivalence=al['equivalence']['status']), out_sha256=osha)
    env = envelope(f'splice {hsha[:12]} <- {dsha[:12]}[{",".join(stages)}]',
                   'sampled-engine (interface) + certified allocator (not yet verified)', 'library sample',
                   str(out) + '.splice.json', ' '.join(['qhdl_surgery.py'] + sys.argv[1:]), osha, **rep)
    Path(str(out) + '.splice.json').write_text(json.dumps(env, indent=1, default=int) + '\n')
    print(json.dumps(dict(out_sha256=osha, stripped=stripped, allocator=rep['allocator'], runs=info, compatible=c['compatible'],
                          dT_sampled=c['dT_sampled']), default=int))


# ----------------------------------------------------------------------------------------------------------------
# Adaptor classification


def gf2_rank_rows(rows):
    """Rank over GF(2) of python-int row vectors."""
    basis = {}
    r = 0
    for v in rows:
        while v:
            p = v.bit_length() - 1
            if p in basis:
                v ^= basis[p]
            else:
                basis[p] = v
                r += 1
                break
    return r, basis


def reduce(v, basis):
    while v:
        p = v.bit_length() - 1
        if p in basis:
            v ^= basis[p]
        else:
            return v
    return 0


def cmd_adaptor(a):
    hsha, dsha = a.host_sha12, a.donor_sha12
    hjs, hnz = load_lib(a.lib, hsha)
    djs, dnz = load_lib(a.lib, dsha)
    stage, _, side = a.cut.partition(':')
    side = side or 'entry'
    hi, di = stage_index(hjs)[stage], stage_index(djs)[stage]
    if side == 'exit':
        hi, di = hi + 1, di + 1
    mask = shot_ok(hjs, hnz) & shot_ok(djs, dnz)
    H, D = hnz['cols'][hi] & mask, dnz['cols'][di] & mask
    toint = lambda col: int.from_bytes(col.tobytes(), 'little')
    nshots = H.shape[1] * 64
    ones = (1 << nshots) - 1
    hrows = [toint(H[w]) for w in np.flatnonzero((H != 0).any(axis=1))]
    drows = [toint(D[w]) for w in np.flatnonzero((D != 0).any(axis=1))]
    hset, dset = Counter(hrows), Counter(drows)
    ones = int.from_bytes(mask.tobytes(), 'little')
    rh, bh = gf2_rank_rows(hrows + [ones])
    rd, bd = gf2_rank_rows(drows + [ones])
    d_in_h = sum(1 for v in drows if reduce(v, bh) == 0)
    h_in_d = sum(1 for v in hrows if reduce(v, bd) == 0)
    res = envelope(f'adaptor class at {stage}:{side} between host {hsha} and donor {dsha}', 'sampled-engine',
                   dict(shots=nshots, jointly_correct_shots=bin(ones).count('1')), f'{a.lib}', ' '.join(['qhdl_surgery.py'] + sys.argv[1:]),
                   host_nonzero=len(hrows), donor_nonzero=len(drows),
                   exact_column_matches=sum((hset & dset).values()),
                   host_affine_rank=rh, donor_affine_rank=rd,
                   donor_cols_in_host_affine_span=d_in_h, host_cols_in_donor_affine_span=h_in_d,
                   classification=('equal up to pi' if hset == dset else
                                   'GF(2)-affine (CNOT adaptor)' if d_in_h == len(drows) and h_in_d == len(hrows)
                                   else 'nonlinear'))
    print(json.dumps(res, indent=1))
    if a.out:
        if Path(a.out).exists():
            raise SystemExit('refusing to overwrite')
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')


def cmd_compat(a):
    """Compatibility of one donor into one host for every standard stage set (plus --stages), with per-module diff."""
    hjs, hnz = load_lib(a.lib, a.host_sha12)
    djs, dnz = load_lib(a.lib, a.donor_sha12)
    hnz, dnz = dict(hnz), dict(dnz)
    mods = [dict(stage=x['stage'], canon_equal=x['canon'] == y['canon'], host_mean_T=x['mean_T_natural'],
                 donor_mean_T=y['mean_T_natural'], host_live_entry=x['entry']['live'],
                 donor_live_entry=y['entry']['live']) for x, y in zip(hjs['modules'], djs['modules'])]
    sets = list(STAGE_SETS) + ([a.stages.split(',')] if a.stages else [])
    rows = [compat(hjs, hnz, djs, dnz, st) for st in sets
            if all(x in stage_index(hjs) and x in stage_index(djs) for x in st)]
    env = envelope(f'compatibility of {a.donor_sha12} modules into {a.host_sha12}', 'sampled-engine',
                   hjs['domain'], a.out, ' '.join(['qhdl_surgery.py'] + sys.argv[1:]), a.host_sha12,
                   modules=mods, sets=rows)
    if a.out:
        if Path(a.out).exists():
            raise SystemExit(f'refusing to overwrite {a.out}')
        Path(a.out).write_text(json.dumps(env, indent=1, default=int) + '\n')
    for r in rows:
        print(f"{'+'.join(r['stages']):60s} compatible={r['compatible']} dT={r['dT_sampled']:9.1f} "
              f"unmatched={[x['unmatched_pairs'] for x in r['runs']]} {r['reasons']}")


def cmd_gate(a):
    """Engine verifies (official verify() fields) of a candidate and optional control at fixed draws, then an optional
    official cross-check at the default salt. Tier: engine (parity-tested with kickmix) / official-local."""
    from kickmix_engine.dqi import DQITask, load_verifier
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    v = load_verifier(str(VERIFIER))
    csha = sha256(a.circuit)
    draws = []
    # --salt-anchor: derive the salted seeds from a fixed hash instead of the candidate's, so different circuits see
    # the same inputs (fixed draw set; e.g. 48 sweep seeds anchored on one circuit hash)
    anchor = (a.salt_anchor if len(a.salt_anchor or '') == 64 else sha256(a.salt_anchor)) if a.salt_anchor else csha
    for salt in a.salts.split(','):
        salt = '' if salt == 'default' else salt
        draws.append(('salt:' + (salt or 'default'), int(v._initial_seed(anchor, salt))))
    if a.seed_of:
        rsha = sha256(a.seed_of)
        draws.append(('default draw of ' + rsha[:12], int(v._initial_seed(rsha, ''))))
    rows = []
    for label, seed in draws:
        row = dict(draw=label, seed=str(seed))
        for role, path in (('candidate', a.circuit), ('control', a.control)):
            if not path:
                continue
            t0 = time.time()
            r = DQITask(path, verifier_path=str(VERIFIER)).verify(seed=seed, threads=a.threads)
            row[role] = {k: r[k] for k in ('status', 'score', 'toffoli_count', 'num_qubits', 'value_errors',
                                           'phase_errors', 'max_toffoli_count', 'num_ops')}
            row[role]['seconds'] = round(time.time() - t0, 1)
        if a.control:
            c, h = row['candidate'], row['control']
            row['same_errors'] = (c['value_errors'], c['phase_errors']) == (h['value_errors'], h['phase_errors'])
            row['dT'] = c['toffoli_count'] - h['toffoli_count']
            row['dcost_pct'] = 100 * (c['score'] / h['score'] - 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    cross = None
    if a.cross_check:
        cmd = [sys.executable, '-m', 'kickmix_engine', 'verify', 'dqi', a.circuit, '--verifier', str(VERIFIER),
               '--cross-check', '--official-python', sys.executable, '--threads', str(a.threads),
               '--official-workers', str(a.official_workers)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            cross = json.loads(r.stdout)
        except ValueError:
            cross = dict(error=r.stdout[-2000:] + r.stderr[-2000:], returncode=r.returncode)
        cross['seconds'] = round(time.time() - t0, 1)
        cross['returncode'] = r.returncode
        print(json.dumps({k: cross.get(k) for k in ('match', 'returncode', 'seconds')}))
    tier = 'official verifier PASS (local, default salt; cross-check)' if cross and cross.get('match') and \
        cross.get('official', {}).get('status') == 'PASS' else 'engine verify (sampled draws)'
    env = envelope(f'gate for {csha[:12]}', tier, dict(draws=[d[0] for d in draws], shots=102400), out,
                   ' '.join(['qhdl_surgery.py'] + sys.argv[1:]), csha, circuit=a.circuit, control=a.control,
                   rows=rows, cross_check=cross)
    out.write_text(json.dumps(env, indent=1, default=str) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='verb', required=True)
    p = sub.add_parser('library')
    p.add_argument('circuits', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--tmp', required=True)
    p.add_argument('--seeds', default='101,102,103')
    p.add_argument('--batches', type=int, default=2)
    p.add_argument('--uniform-batches', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p = sub.add_parser('matrix')
    p.add_argument('lib')
    p.add_argument('--out', required=True)
    p = sub.add_parser('splice')
    p.add_argument('--host', required=True)
    p.add_argument('--donor', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--lib', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tmp', required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--strip-resets', action='store_true',
                   help='also write OUT.stripped.kmb: the coloured circuit without the synthetic R ops')
    p.add_argument('--no-resets', action='store_true',
                   help='do not insert R on host wires that hold |0> (sampled) at the run cuts')
    p = sub.add_parser('adaptor')
    p.add_argument('--host-sha12', required=True)
    p.add_argument('--donor-sha12', required=True)
    p.add_argument('--lib', required=True)
    p.add_argument('--cut', required=True)
    p.add_argument('--out')
    p = sub.add_parser('compat')
    p.add_argument('--host-sha12', required=True)
    p.add_argument('--donor-sha12', required=True)
    p.add_argument('--lib', required=True)
    p.add_argument('--stages')
    p.add_argument('--out')
    p = sub.add_parser('gate')
    p.add_argument('--circuit', required=True)
    p.add_argument('--control')
    p.add_argument('--salts', default='default,salt-check-1,salt-check-2')
    p.add_argument('--seed-of', help='also verify at the default draw of this reference circuit')
    p.add_argument('--salt-anchor', help='derive salted seeds from this SHA-256 (64 hex) or file instead of the candidate')
    p.add_argument('--cross-check', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--official-workers', type=int, default=3)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(compat=cmd_compat, gate=cmd_gate, library=cmd_library, matrix=cmd_matrix, splice=cmd_splice, adaptor=cmd_adaptor)[a.verb](a)


if __name__ == '__main__':
    main()
