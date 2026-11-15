#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Chemistry design calculator: T_total x Q_total of the official score as a function of the factorization shape.

Uses the official verifier's own `heuristic_costs` (state preparation and Givens model) plus the loader term, so it
needs no circuits. Run with a Python that can import kickmix and the official verifiers:
    $QHDL_KICKMIX_PYTHON -I scripts/d1_chem_shape_model.py [OUT.json]
Shape (R groups, B vectors per group, C rank, brot word bits); loaders: `angles_T` Toffolis for both passes, scaled
with the number of lookup addresses R*B + N when the shape changes (a rough assumption, labelled in the output).
The base shape PICK is a verified public chemistry family (R 8, B 143, C 44, 8-bit words, 4,033 loader Toffolis per pass).
Static model / projection; lambda_eff and the accuracy bounds are NOT modelled here.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402
CFG.add_verifiers_path()  # the official verifiers (not shipped): QHDL_OFFICIAL_ROOT
from dfthcsos import verify as V  # noqa: E402

N, J, SEL, CAP = 76, 5, 11, 128
PICK = dict(R=8, B=143, C=44, brot=8, angles_T=2 * 4033)


def model(R, B, C, brot=8, angles_T=None, J=J):
    c = V.heuristic_costs(N, R, B, C, brot)
    if angles_T is None:
        angles_T = PICK["angles_T"] * (R * B + N) / (PICK["R"] * PICK["B"] + N) * brot / PICK["brot"]
    pers = c["persistent_qubits"] + J * brot
    temp = max(c["temporary_qubits"], CAP - SEL - J * brot)
    T = c["toffoli"] + angles_T
    return dict(R=R, B=B, C=C, brot=brot, vectors=R * B, T=T, Q=pers + temp, TQ=T * (pers + temp),
                angles_T=angles_T, outer_T=c["outer_state_prep_toffoli"], inner_T=c["inner_state_prep_toffoli"],
                givens_T=c["givens_rotations_toffoli"], persistent=pers, temporary=temp)


def main():
    base = model(**{k: PICK[k] for k in ("R", "B", "C", "brot")}, angles_T=PICK["angles_T"])
    assert (base["T"], base["Q"]) == (14728, 427), base   # control: the base family's verified T_total and Q_total
    rel = lambda m: round(100 * (m["TQ"] / base["TQ"] - 1), 2)
    out = dict(pick=base, tier="static model / projection; loader Toffolis scaled with addresses and word bits (assumption)")
    out["one_step"] = {
        "one fewer group (R 8->7)": rel(model(7, 143, 44)),
        "one rank less (C 44->43)": rel(model(8, 143, 43)),
        "one word bit less (brot 8->7)": rel(model(8, 143, 44, 7)),
        "1% fewer loader Toffolis": rel(model(8, 143, 44, angles_T=PICK["angles_T"] * 0.99)),
        "B at the band edge (129)": rel(model(8, 129, 44)),
    }
    bands, prev = [], None
    for B in range(8, 330):
        t = model(8, B, 44)["temporary"]
        if prev is None or abs(t - prev[2]) > 6:
            if prev: bands.append(dict(B_from=prev[0], B_to=B - 1, temporary=prev[2]))
            prev = [B, B, t]
    bands.append(dict(B_from=prev[0], B_to=329, temporary=prev[2]))
    out["temporary_qubit_bands_by_B"] = bands
    out["same_vector_count"] = [dict(B=B, R=round(1144 / B), rel_TQ_pct=rel(model(round(1144 / B), B, 44)),
                                     Q=model(round(1144 / B), B, 44)["Q"]) for B in (60, 96, 128, 129, 143, 163, 191, 229, 286)]
    out["shares_of_T"] = {k: round(100 * base[k] / base["T"], 1) for k in ("angles_T", "inner_T", "givens_T", "outer_T")}
    out["shares_of_Q"] = dict(inner_state_prep_temporary=round(100 * 303 / 427, 1), persistent=round(100 * 124 / 427, 1))
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
