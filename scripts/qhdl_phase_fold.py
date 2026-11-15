#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Phase-fold detector (identities ID-011/ID-012): a diagonal gate on t conjugated by the same CCX(a,b,t) pair.

CCX(a,b,t) ... CCX(a,b,t) where, in between, a and b are not written and t is touched only by Z(t) and CZ(c,t)
(any other use of t, or any write to a or b, closes the window). Replacing the pair:
  each Z(t)      -> Z(t) + CZ(a,b)            (free)
  each CZ(c,t)   -> CZ(c,t) + CCZ(a,b,c)      (one Toffoli-like; CZ(a,b) if c is a or b)
so the window saves 2 - (number of CZ(c,t) with c not in {a,b}) Toffoli-like gates. The fold is exact for every input
(t may start dirty). Kirill Tymchishin found and applied the CZ case (seven folds in DQI); the Z case and this
streaming detector are ours. Conditional gates in the window are allowed only if both CCX are unconditional and at
condition depth 0 (an added gate inherits its partner's condition).

Streams the .kmx once; memory is the number of open windows. Usage: python3 scripts/qhdl_phase_fold.py CIRCUIT.kmx
[--max-span N] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kmx")
    ap.add_argument("--max-span", type=int, default=5000)
    ap.add_argument("--json")
    a = ap.parse_args()
    open_by_t = {}        # t -> window dict
    watch = {}            # wire -> set of t whose window it can invalidate (a, b)
    hits, n, depth = [], 0, 0

    def close(t):
        w = open_by_t.pop(t, None)
        if w:
            for x in (w["a"], w["b"]):
                watch.get(x, set()).discard(t)

    with open(a.kmx) as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0] in ("REGISTER", "APPEND_TO_REGISTER"):
                continue
            i = n
            n += 1
            op = tok[0]
            cond = None
            if len(tok) >= 3 and tok[-2] == "if":
                cond, tok = tok[-1], tok[:-2]
            if op == "PUSH_CONDITION":
                depth += 1
                continue
            if op == "POP_CONDITION":
                depth -= 1
                continue
            qs = [x for x in tok[1:] if x.startswith("q")]
            # 1. writes to a or b close the windows that depend on them
            written = set()
            if op in ("X", "R", "HMR"):
                written = {qs[0]}
            elif op in ("CX", "CCX"):
                written = {qs[-1]}
            elif op in ("SWAP",):
                written = set(qs)
            elif op == "CSWAP":
                written = set(qs[1:])
            for wq in written:
                for t in list(watch.get(wq, ())):
                    close(t)
            # 2. ops touching an open window's target
            for q in set(qs):
                w = open_by_t.get(q)
                if not w:
                    continue
                t = q
                if op == "CCX" and qs[2] == t and {qs[0], qs[1]} == {w["a"], w["b"]} and cond is None and depth == 0:
                    cz = [c for c in w["cz"] if c not in (w["a"], w["b"])]
                    saves = 2 - len(cz)
                    if saves > 0:
                        hits.append({"t": t, "a": w["a"], "b": w["b"], "first": w["i"], "second": i,
                                     "z": w["z"], "cz": w["cz"], "saves": saves})
                    close(t)
                    w = None
                elif op == "Z" and qs == [t]:
                    w["z"] += 1
                elif op == "CZ" and t in qs:
                    w["cz"].append(qs[0] if qs[1] == t else qs[1])
                else:
                    close(t)
                if w and i - w["i"] > a.max_span:
                    close(t)
            # 3. a new unconditional CCX opens a window on its target (unless it just closed one)
            if op == "CCX" and cond is None and depth == 0 and qs[2] not in open_by_t and \
                    not (hits and hits[-1]["second"] == i):
                t = qs[2]
                open_by_t[t] = {"i": i, "a": qs[0], "b": qs[1], "z": 0, "cz": []}
                watch.setdefault(qs[0], set()).add(t)
                watch.setdefault(qs[1], set()).add(t)
    out = {"circuit": a.kmx, "ops": n, "sites": len(hits), "toffolis_saved_static": sum(h["saves"] for h in hits),
           "tier": "static exact identity (ID-011/012); emitted circuits still need the official verifier", "hits": hits}
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "hits"}))
    for h in hits[:20]:
        print(f"  {h['t']} CCX({h['a']},{h['b']}) ops {h['first']}..{h['second']}: Z x{h['z']}, CZ with "
              f"{h['cz']} -> saves {h['saves']}")


if __name__ == "__main__":
    sys.exit(main())
