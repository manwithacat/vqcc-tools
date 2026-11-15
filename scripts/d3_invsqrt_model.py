#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Parametric integer model of the 46-qubit inverse-square-root algorithm.

The arithmetic follows Syed Asad Ali's exact model `whole_model.py` (Apache-2.0; Copyright 2026 Syed Asad Ali, credits
"Opus and Astra"), re-implemented here so that the tables are arguments. Nothing from the author's tree is executed;
the published model JSON is read as data.

    python3 scripts/d3_invsqrt_model.py MODEL.json [OUT.json]

Reports the worst phase error over all 3,133,452 inputs as a fraction of the verifier's relative tolerance (|target| x 2^-24), and
the verifier's charge for given Toffolis and rotation count: charge = T + n_rot x (0.56 log2(1/eps) + 5.3)/4 with
eps = pi x margin / n_rot (margin in turns left under the tolerance).
"""
import json, math, sys
import numpy as np

MAX_X = 3133452
M = 37
TOL = 2.0 ** -24


def phase_units(x, C_TABLE, U0_TABLE, rows=range(9, 36)):
    """Phase in 2^-37 turns (mod one turn) for inputs x, given the per-root tables C and U0 (indexed by v)."""
    x = np.asarray(x, dtype=np.int64)
    leading = np.floor(np.log2(x.astype(np.float64))).astype(np.int64)
    leading = np.where((np.int64(1) << leading) > x, leading - 1, leading)
    leading = np.where((np.int64(1) << (leading + 1)) <= x, leading + 1, leading)
    e = (21 - leading) // 2
    norm = x << (2 * e)
    Q = np.sqrt((norm << 4).astype(np.float64)).astype(np.int64)
    Q = np.where(Q * Q > (norm << 4), Q - 1, Q)
    Q = np.where((Q + 1) * (Q + 1) <= (norm << 4), Q + 1, Q)
    v = Q - 4096; d = 2 * Q + 1; h = ((norm << 6) - d * d) // 64
    C = C_TABLE[v]; U0 = U0_TABLE[v]; q = h + 512
    F = np.zeros(len(x), dtype=np.int64)
    for i in range(10):
        shift = 8 - i
        term = (C + (1 << (shift - 1))) >> shift if shift > 0 else C if shift == 0 else 2 * C + 1
        F += (2 * ((q >> i) & 1) - 1) * term
    U = U0 - F // 2
    product = U & 32767; r = 4096 - d
    phase = np.full(len(x), -((1 << 37) - (1 << 29)), dtype=np.int64)
    for j in rows:
        sign = (r < 0).astype(np.int64); phase -= sign << (37 - j)
        if j <= 23:
            bit = (product >> (23 - j)) & 1
            if j == 9: bit = -bit
            raw = 2 * r + (2 * sign - 1) * d + bit
        elif j == 24:
            raw = 2 * r + (2 * sign - 1) * d
        else:
            k = j - 26 if j >= 28 else 0; V = d // 2
            A = ((((V >> (k + 1)) + ((V >> k) & 1)) << (k + 1)) - (V & 1)) if k else V
            u = (r - 1) // 2; raw = 2 * (2 * u + sign + (2 * sign - 1) * A) + 1
        nextu = raw // 2
        nextu = ((nextu + 8192) & 16383) - 8192; r = 2 * nextu + (raw & 1)
    phase -= (r < 0).astype(np.int64) << 1; p = (phase - 1) & ((1 << 37) - 1)
    word = ((p << e) + (p >> (27 - e))) & ((1 << 37) - 1)
    return word, e, v


def error_ratio(word, x, beta=1.0):
    """(|model phase - beta/sqrt(x)| / tol(x), tol(x) - |error| in turns) per input. The verifier's tolerance is
    RELATIVE: tol(x) = 2^-25 x 2 x |beta|/sqrt(x) = |target| x 2^-24, capped at half a turn (reference_values)."""
    target = beta / np.sqrt(x.astype(np.longdouble))
    got = word.astype(np.longdouble) / np.longdouble(2 ** M)
    diff = got - target
    diff -= np.round(diff)
    tol = np.minimum(target * np.longdouble(2.0 ** -24), 0.5)
    return (np.abs(diff) / tol).astype(np.float64), (tol - np.abs(diff)).astype(np.float64)


def charge(toffolis, n_rot, min_margin_turns, t_base=1.0):
    """The verifier's Toffoli-equivalent charge; min_margin_turns = min over x of tol(x) - |error(x)|."""
    eps = math.pi * min_margin_turns / n_rot
    return toffolis + (t_base + n_rot * (0.56 * math.log2(1 / eps) + 5.3)) / 4


def main():
    d = json.load(open(sys.argv[1]))
    C = np.array(d["C"], dtype=np.int64); U0 = np.array(d["U0"], dtype=np.int64)
    x = np.arange(1, MAX_X + 1, dtype=np.int64)
    word, e, v = phase_units(x, C, U0)
    ratio, margin = error_ratio(word, x)
    w = int(np.argmax(ratio))
    eps_pick = 6.003064377662461e-13          # verifier's rotation_eps for two public 46-qubit circuits (33 rotations)
    m_pick = eps_pick * 33 / math.pi
    rep = dict(model=d.get("name"), rows=d.get("row_count"), worst_ratio=float(ratio[w]), worst_x=int(x[w]),
               published_worst_ratio=d.get("worst_ratio"), min_margin_turns=float(margin.min()),
               note="integer model with exact dyadic weights; the built circuit refits its rotation angles",
               # dcde70780434: the public 46-qubit circuit (Atakan Aldemir); measure_first_variant: a public measure-first variant of it
               charge_checks=dict(
                   dcde70780434=dict(model=charge(1480.7496921576028, 33, m_pick), verifier=1712.2937635111161),
                   measure_first_variant=dict(model=charge(1478.750911494735, 33, m_pick), verifier=1710.2949828482483)))
    print(json.dumps(rep, indent=1))
    if len(sys.argv) > 2: open(sys.argv[2], "w").write(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
