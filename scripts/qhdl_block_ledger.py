#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic).
"""Block ledger: account for the Toffolis of one circuit region against another, block type by block type.

Generic: any .kmb, any two op ranges (for
example a forward stage and its reverse), optionally labelled from a reference circuit whose ops are mapped to source
lines (the srcmap written by scripts/qhdl_source_provenance_build.py).

A *block* is a maximal run of Toffoli-like ops (CCX, CCZ, CSWAP and their conditional forms) with no gap longer than
--gap ops between consecutive ones. Its *signature* is (CCX, CSWAP, CCZ, distinct targets, distinct controls). Equal
signatures in the two ranges are taken as the same construction; the ledger prints, per signature, the Toffolis and
block counts in each range and the difference, so a surplus can be attributed to "the same block run k times more
often" or "a block only one side has". Labels: for each signature, the most common source key over the reference
circuit's blocks of that signature (a heuristic: it names the construction, it does not prove identity).

    python3 -I scripts/qhdl_block_ledger.py CIRCUIT.kmb --a extract=2643973:6537394 --b unextract=52142222:57937019 \
        [--ref REF.kmb --ref-srcmap MAP.json --ref-a A0:A1 --ref-b B0:B1] [--gap 64] [--top 25] [--json OUT.json]

Tier: static model / projection (static op counts; conditional ops counted as if executed). Read only.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K  # noqa: E402

KIND = {0x30: "CCZ", 0x31: "CCX", 0x32: "CSWAP"}


def parse_range(s: str):
    name, _, rng = s.rpartition("=")
    a, b = rng.split(":")
    return (name or "range"), int(a), int(b)


def block_rows(k: K.Kmb, a: int, b: int, gap: int, keymap=None):
    """[(signature, toffolis, label_key or None, first_op)] for the blocks in ops [a, b)."""
    t = np.asarray(k.op_types)
    tof = np.isin(t[a:b], K.TOFFOLI_LIKE)
    idx = np.flatnonzero(tof) + a
    if len(idx) == 0:
        return []
    q3 = np.flatnonzero((t & 0x30) == 0x30)
    pos = np.searchsorted(q3, idx)
    s0, s1, s2 = k.stream("qqq0"), k.stream("qqq1"), k.stream("qqq2")
    base = t[idx] & 0x7F
    rows = []
    for part in np.split(np.arange(len(idx)), np.flatnonzero(np.diff(idx) > gap) + 1):
        kinds = collections.Counter(KIND[int(base[i])] for i in part)
        j = pos[part]
        sig = (kinds.get("CCX", 0), kinds.get("CSWAP", 0), kinds.get("CCZ", 0), len(set(s2[j].tolist())),
               len(set(s0[j].tolist()) | set(s1[j].tolist())))
        lab = None
        if keymap is not None:
            lab = collections.Counter(int(keymap[i]) for i in idx[part]).most_common(1)[0][0]
        rows.append((sig, len(part), lab, int(idx[part[0]])))
    return rows


def tally(rows):
    tof, n, first = collections.Counter(), collections.Counter(), {}
    for sig, c, _, op in rows:
        tof[sig] += c
        n[sig] += 1
        first.setdefault(sig, op)
    return tof, n, first


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("circuit")
    ap.add_argument("--a", required=True, help="NAME=START:END (ops, end exclusive)")
    ap.add_argument("--b", required=True, help="NAME=START:END")
    ap.add_argument("--ref", help="reference .kmb whose ops have a source map")
    ap.add_argument("--ref-srcmap", help="srcmap JSON (keys) beside an .npz with plain_key per op")
    ap.add_argument("--ref-a", help="START:END in the reference")
    ap.add_argument("--ref-b", help="START:END in the reference")
    ap.add_argument("--gap", type=int, default=64)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    labels = collections.defaultdict(collections.Counter)
    if a.ref:
        keys = json.loads(Path(a.ref_srcmap).read_text())["keys"]
        km = np.load(Path(a.ref_srcmap).with_suffix(".npz"))["plain_key"]
        rk = K.read(a.ref)
        for rng in (a.ref_a, a.ref_b):
            if rng:
                r0, r1 = (int(x) for x in rng.split(":"))
                for sig, c, lab, _ in block_rows(rk, r0, r1, a.gap, km):
                    labels[sig][keys[lab]] += c

    k = K.read(a.circuit)
    na, a0, a1 = parse_range(a.a)
    nb, b0, b1 = parse_range(a.b)
    TA, NA, FA = tally(block_rows(k, a0, a1, a.gap))
    TB, NB, FB = tally(block_rows(k, b0, b1, a.gap))
    tot_a, tot_b = sum(TA.values()), sum(TB.values())
    print(f"{Path(a.circuit).name}: {na} {tot_a} Toffolis in {sum(NA.values())} blocks; "
          f"{nb} {tot_b} in {sum(NB.values())}; difference {tot_b - tot_a:+d} (static, gap {a.gap})")
    sigs = sorted(set(TA) | set(TB), key=lambda s: -abs(TB.get(s, 0) - TA.get(s, 0)))
    out = []
    for s in sigs:
        lab = labels[s].most_common(1)[0][0] if labels.get(s) else None
        row = dict(signature=dict(zip(("ccx", "cswap", "ccz", "targets", "controls"), s)),
                   **{f"{na}_toffolis": TA.get(s, 0), f"{na}_blocks": NA.get(s, 0),
                      f"{nb}_toffolis": TB.get(s, 0), f"{nb}_blocks": NB.get(s, 0)},
                   diff=TB.get(s, 0) - TA.get(s, 0), label=lab, first_op=FB.get(s, FA.get(s)))
        out.append(row)
    for r in out[:a.top]:
        s = r["signature"]
        print(f"  ccx{s['ccx']:4d} cswap{s['cswap']:3d} ccz{s['ccz']:3d} tg{s['targets']:3d} ct{s['controls']:3d} | "
              f"{na} {r[f'{na}_toffolis']:6d} x{r[f'{na}_blocks']:<5d} {nb} {r[f'{nb}_toffolis']:6d} "
              f"x{r[f'{nb}_blocks']:<5d} diff {r['diff']:+6d}  {(r['label'] or '(no reference match)')[:80]}")
    covered = sum(r["diff"] for r in out[:a.top])
    if tot_b != tot_a:
        print(f"top {a.top} signatures account for {covered:+d} of {tot_b - tot_a:+d} "
              f"({100 * covered / (tot_b - tot_a):.1f}%)")
    if a.json:
        p = Path(a.json)
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        p.write_text(json.dumps(dict(circuit=a.circuit, a=[na, a0, a1], b=[nb, b0, b1], gap=a.gap,
                                     reference=a.ref, tier="static model / projection", rows=out), indent=1) + "\n")


if __name__ == "__main__":
    main()
