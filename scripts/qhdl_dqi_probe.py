#!/usr/bin/env python3
"""Segmented kickmix simulation of a DQI circuit with qubit snapshots at chosen op indices.

probe(kmb, cuts, num_shots, seed) -> dict cut -> bool array [num_qubits, num_shots] of qubit values just BEFORE op
`cut`. Inputs are drawn with the official verifier's own `sample_shots` (its module is imported read-only).
Cuts must sit at condition-stack depth 0.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import vqcc_config as CFG  # noqa: E402
VERIFIER = CFG.VERIFIERS / 'dqi/verify_submission_dqi.py'


def load_verifier():
    spec = importlib.util.spec_from_file_location('verify_submission_dqi_ro', VERIFIER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def draw_inputs(V, num_shots, seed):
    import kickmix as km
    spec = V.ProblemSpecDQI.benchmark_instance()
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(seed)
    return spec, V.sample_shots(spec, tables, rng, num_shots)


def probe(kmb_path, cuts, num_shots=1024, seed=12345, workdir=None, sim_seed=777, extra_segments=None):
    import kickmix as km
    V = load_verifier()
    k = K.read(kmb_path)
    depth = K.condition_depth(k.op_types)
    cuts = sorted(set(int(c) for c in cuts))
    for c in cuts:
        assert depth[c] == 0 and (c == 0 or k.op_types[c - 1] != K.PUSH_CONDITION), ('cut inside a block', c)
    sl = K.Slicer(k)
    circ = km.Circuit.from_file(kmb_path)
    spec, (syn, mbuc, phases) = draw_inputs(V, num_shots, seed)
    regs = circ.register_data
    s_names = [f'S{i}' for i in range(spec.n)]
    e_names = [f'E{j}' for j in range(spec.m)]
    sq = km.array([q for n in s_names for q in regs[n]])
    eb = km.array([b for n in e_names for b in regs[n]])
    sim = km.Simulator(batch_size=num_shots, debug_print=False, counting_operations=False, seed=sim_seed)
    sim.use_same_registers_as(circ)
    sim.write_across_shots(sq, V.to_bits(syn, spec.degree))
    sim.write_across_shots(eb, V.to_bits(mbuc, spec.degree))
    allq = km.array([km.q(i) for i in range(k.num_qubits)])
    out = {}
    bounds = [0] + cuts + [len(k.op_types)]
    tmp = workdir or tempfile.mkdtemp(prefix='dqiprobe')
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if a in cuts:
            out[a] = np.array(sim.read_across_shots(allq), dtype=bool)
        if b > a:
            path = os.path.join(tmp, f'seg_{a}_{b}.kmb')
            sl.write(path, a, b)
            sim.do(km.Circuit.from_file(path))
            os.remove(path)
    final = dict(phase_flipped=np.array(sim.read_phase_flipped_across_shots()), syndromes_out=np.array(sim.read_across_shots(sq)),
                 ancilla=np.array(sim.read_across_shots(allq)), syn_in=V.to_bits(syn, spec.degree), phases=phases)
    return out, final


if __name__ == '__main__':
    import json
    path = sys.argv[1]
    cuts = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else []
    snaps, fin = probe(path, cuts, 256)
    val_err = (fin['syndromes_out'] ^ fin['syn_in']).any(axis=0) | fin['ancilla'][840:].any(axis=0)
    print(json.dumps(dict(shots=256, value_errors=int(val_err.sum()),
                          phase_errors=int(((fin['phase_flipped'] ^ fin['phases']) & ~val_err).sum()))))
