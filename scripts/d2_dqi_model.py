#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Design calculator for the public DQI decoder (Syed Asad Ali's v12 architecture): Toffolis and Q² x T as a
function of the NODE DEGREE d of the continued-fraction extraction (d = 2 is the built design: 35 quadratic nodes
over GF(2^24)). Static model / projection; no circuits. Usage: python3 scripts/d2_dqi_model.py [OUT.json]

Model (from a per-stage Toffoli census of the built decoder and the algorithm anatomy):
  nodes m = 70/d (70 GF(2^12) words of syndrome); a node's field is GF(2^(12d)) with product rank rho(d).
  extraction queries, both ways:  m(m-1) x rho(d)                       (slot i applies i query products each way)
  extraction divisions, both ways: (m-3) x 834 x (d/2)^alpha            (alpha: how division cost grows with d)
  synthesis forward:  sum over nodes (r + d) x 42, r = words in N       (one long x short polynomial product per node)
  synthesis return:   sum r x 42 + m x 360 x (d/2)^alpha
  everything else (middle, finish, scan, charts, tail) is held at its measured value.
  width: Q = 841 + 24d (carry 12d + helpers 12d), or 865 + 12d if only the helpers grow.
rho(d): 99 (built); 153 = 17 points over GF(16) x 9; 225 = 15 points over GF(64) x 15 (216 with an elliptic curve
over GF(16)); 285, 345, 405 by GF(64) evaluation. These are upper bounds from plain interpolation, not built here.
"""
import json, sys
from pathlib import Path

MEASURED = dict(total=384003, extraction=73122 + 78611, synthesis=27965 + 37230, Q=889)
RHO = {2: 99, 3: 153, 4: 225, 5: 285, 6: 345, 7: 405}


def model(d, alpha, rho=None, wide_carry=True):
    rho = rho or RHO[d]
    m = 70 / d
    queries = m * (m - 1) * rho
    divisions = max(m - 3, 0) * 834 * (d / 2) ** alpha
    nodes = int(round(m)) - 1
    r_at = [min(34.0, (k + 1) * d / 2) for k in range(nodes)]
    synth_f = sum((r + d) * 42 for r in r_at)
    synth_r = sum(r * 42 for r in r_at) + nodes * 360 * (d / 2) ** alpha
    return dict(queries=queries, divisions=divisions, synth_f=synth_f, synth_r=synth_r,
                Q=(841 + 24 * d) if wide_carry else (865 + 12 * d))


def main():
    base = model(2, 2.0)
    fixed_ext = MEASURED["extraction"] - base["queries"] - base["divisions"]     # charts, tail, seed: held fixed
    fixed_syn = MEASURED["synthesis"] - base["synth_f"] - base["synth_r"]
    other = MEASURED["total"] - MEASURED["extraction"] - MEASURED["synthesis"]
    rows = []
    for d in (2, 3, 4, 5, 6):
        for alpha in (1.6, 2.0):
            for wide in (True, False):
                for rho in ({4: (225, 216)}.get(d, (RHO[d],))):
                    c = model(d, alpha, rho, wide)
                    T = c["queries"] + c["divisions"] + fixed_ext + c["synth_f"] + c["synth_r"] + fixed_syn + other
                    cost = (c["Q"] / MEASURED["Q"]) ** 2 * T / MEASURED["total"]
                    rows.append(dict(d=d, alpha=alpha, rho=rho, wide_carry=wide, Q=c["Q"], T=round(T),
                                     dT_pct=round(100 * (T / MEASURED["total"] - 1), 1),
                                     cost_pct=round(100 * (cost - 1), 1), gm_pct=round(-100 * (cost - 1) / 3, 1)))
    ctrl = [r for r in rows if r["d"] == 2 and r["wide_carry"]][0]
    assert abs(ctrl["T"] - MEASURED["total"]) <= 1 and ctrl["Q"] == 889, ctrl
    print(f"control d=2: T {ctrl['T']} Q {ctrl['Q']}; fixed parts: extraction {fixed_ext:.0f}, synthesis {fixed_syn:.0f}, other {other}")
    print(" d alpha rho  carry    Q       T    dT%   cost%  GM%")
    for r in rows:
        if r["d"] > 2:
            print(f"{r['d']:2} {r['alpha']:4} {r['rho']:4}  {'wide' if r['wide_carry'] else 'narrow':6} {r['Q']:4} {r['T']:7} {r['dT_pct']:6} {r['cost_pct']:6} {r['gm_pct']:5}")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(dict(measured=MEASURED, rows=rows, tier="static model / projection"), indent=1))


if __name__ == "__main__":
    main()
