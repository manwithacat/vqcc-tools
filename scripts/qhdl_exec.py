"""Executed (charged) Toffolis per region for enumerable circuits (backs `qhdl exec`).

The verifiers charge the per-shot MEAN of Toffoli-like gates that actually run; a gate inside a PUSH_CONDITION block
or with its own `if bN` runs only where its condition bits are 1. Those bits come from HMR outcomes and BIT_* ops,
never from qubit values, so the executed count of a shot depends on its measurement outcomes only.

Two passes of qhdl_bitsim over all legal x:
  zero    every HMR returns 0 (the deterministic branch). Exact over all legal x for that branch.
  seeded  an independent fair outcome per HMR per x (numpy seed). The mean over the 3,133,452 rows is a
          sampled(3133452) estimate of the verifier's mean over random outcomes; its standard error is reported.
Per op, the executed count is the number of rows where the op ran; per region and per gate kind they are summed.

Rotations (invsqrt verifier, data/official_full/verifiers/invsqrt/verify_submission_invsqrt.py): every executed
Z_POW is classed by its angle: Clifford (multiple of 1/2 half turn) is free, T (multiple of 1/4) costs 1 T, any
other angle is a rotation costing 0.56 log2(1/eps) + 5.3 T, with eps = pi * min_margin / 2^64 / n_rot set by the
smallest phase margin the verifier sees. Charge = Toffolis + T / 4. Executed Z_POW counts per op come from the same
passes (an op runs where its condition bits are 1, like a Toffoli). eps itself needs the verifier's min margin, so
the Toffoli-equivalent of the rotations is priced with a recorded report's rotation_eps (qhdl exec says which).

Library: executed(ops, xq, nq, seed) -> dict with per-op arrays and per-row statistics; zpow_class(angle_units);
rotation_t_count(eps); rotation_toffoli_eq(n_rot, n_t, eps).
"""
from __future__ import annotations

import sys
from pathlib import Path

import math

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_bitsim as B  # noqa: E402


class CountSim(B.BitSim):
    """BitSim that also keeps each row's own Toffoli count (for the standard error of the mean)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, count_toffoli=True, **kw)
        self.row_t = np.zeros(self.n, np.uint32)
        self.uncond = 0

    def _count(self, act):
        if act is None:
            self.uncond += 1
            return self.n
        self.row_t += B.unpack_bits(act, self.n)
        return int(np.bitwise_count(act).sum())


T_PER_TOFFOLI = 4


def zpow_class(angle_units):
    """Verifier class of a Z_POW angle given in units of 2^-64 turns: 'clifford' | 't' | 'rotation'."""
    a = int(angle_units) % (1 << 64)
    if a % (1 << 62) == 0:
        return 'clifford'
    if a % (1 << 61) == 0:
        return 't'
    return 'rotation'


def rotation_t_count(eps):
    """T cost of one arbitrary-angle rotation at synthesis error eps (verifier formula, arXiv:2203.10064)."""
    return 0.56 * math.log2(1 / eps) + 5.3


def rotation_toffoli_eq(n_rot, n_t, eps):
    """Toffoli-equivalents of n_rot rotations and n_t T gates per shot (charge = Toffolis + T / 4); None when there
    are rotations but no eps to price them."""
    if n_rot and not eps:
        return None
    return ((n_rot * rotation_t_count(eps) if n_rot else 0.0) + n_t) / T_PER_TOFFOLI


def static_kinds(ops):
    """Per op: kind name (with _IF for an own condition) and whether it sits in a PUSH_CONDITION block."""
    names, in_block, depth = [], [], 0
    for o in ops:
        names.append(B.KINDS[o.code] + ('_IF' if o.cond >= 0 and o.code != B.C_PUSH else ''))
        in_block.append(depth > 0)
        if o.code == B.C_PUSH:
            depth += 1
        elif o.code == B.C_POP and depth:
            depth -= 1
    return names, in_block


def executed(ops, xq, nq, outcome):
    """outcome 'zero' or an int seed. -> dict(per_op {index: rows executed}, mean, rows, sd, uncond)."""
    tidx = [i for i, o in enumerate(ops) if o.code in B.TOFFOLI_LIKE]
    zidx = {i for i, o in enumerate(ops) if o.code == B.C_ZPOW}
    snaps = sorted(set(tidx) | {i + 1 for i in tidx} | zidx)
    seen, zrun = {}, {}

    def cb(i, sim):
        seen[i] = sim.toffoli_rows
        if i in zidx:  # rows where Z_POW i runs (its condition bits; qubit values do not matter)
            o = ops[i]
            act = sim.act
            if o.cond >= 0:
                c = sim._bit(o.cond)
                act = c if act is None else (act & c)
            zrun[i] = sim.n if act is None else int(np.bitwise_count(act & sim.valid).sum())
    sim = CountSim(ops, num_qubits=nq, x_qubits=xq, outcome=outcome)
    sim.run(snaps=snaps, on_snap=cb)
    per_op = {i: int(seen[i + 1] - seen[i]) for i in tidx}
    per_row = sim.row_t.astype(np.float64) + sim.uncond
    return dict(per_op=per_op, zpow_per_op=zrun, rows=sim.n, mean=float(per_row.mean()), sd=float(per_row.std()),
                total_rows=int(sim.toffoli_rows), uncond=sim.uncond, clean=sim.scratch_clean() and sim.x_restored())
