#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Circuit identity registry: list, show, validate fixtures, and scan a circuit with every tool-backed detector.

The registry is docs/identities/identities.json (schema in docs/identities/README.md). Each identity has a minimal
fixture: a left-hand side and a right-hand side in kickmix text over a few wires. `validate` checks every fixture whose
validity is "exact" by exhaustive simulation: all basis inputs and all measurement-outcome assignments (HMR outcomes
are uniform and independent, so equal results for every assignment by bit name means an equal operator and equal
outcome distribution). It compares final wire values, measurement-bit values and the sign. "reachable" identities hold
only on reachable states and are not fixture-checkable in isolation; their detectors must certify whole circuits.

Usage:
  python3 scripts/qhdl_identities.py list [--task invsqrt|dqi|chemistry]
  python3 scripts/qhdl_identities.py show ID
  python3 scripts/qhdl_identities.py validate
  python3 scripts/qhdl_identities.py scan -c CIRCUIT --task TASK [--run]   (prints, or with --run executes, detectors)
"""
from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "docs/identities/identities.json"
FIELDS = ["id", "name", "summary", "validity", "lhs", "rhs", "toffoli_delta", "qubit_delta", "preconditions",
          "tasks", "credit", "evidence", "detector", "emitter", "status"]
VALIDITY = {"exact", "reachable"}
STATUS = {"tool-backed", "catalogued"}


def load():
    return json.loads(REG.read_text())


def parse(line):
    t = line.split()
    cond = None
    if "if" in t:
        i = t.index("if")
        cond = t[i + 1]
        t = t[:i]
    return t[0], t[1:], cond


def wires_bits(lines):
    qs, bs = set(), set()
    for ln in lines:
        op, args, cond = parse(ln)
        for a in args:
            (qs if a.startswith("q") else bs).add(a)
        if cond:
            bs.add(cond)
    return qs, bs


def run(lines, state, outcomes):
    """Classical-basis simulation with a sign. state: dict wire->0/1; outcomes: dict bit->0/1 (HMR result by bit)."""
    v, bits, sign = dict(state), {}, 0
    for ln in lines:
        op, a, cond = parse(ln)
        if cond is not None and not bits.get(cond, 0):
            continue
        g = lambda k: v[a[k]]
        if op == "X":
            v[a[0]] ^= 1
        elif op == "CX":
            v[a[1]] ^= g(0)
        elif op == "CCX":
            v[a[2]] ^= g(0) & g(1)
        elif op == "Z":
            sign ^= g(0)
        elif op == "CZ":
            sign ^= g(0) & g(1)
        elif op == "CCZ":
            sign ^= g(0) & g(1) & g(2)
        elif op == "SWAP":
            v[a[0]], v[a[1]] = v[a[1]], v[a[0]]
        elif op == "CSWAP":
            if g(0):
                v[a[1]], v[a[2]] = v[a[2]], v[a[1]]
        elif op == "HMR":
            m = outcomes[a[1]]
            sign ^= m & v[a[0]]
            v[a[0]] = 0
            bits[a[1]] = m
        elif op == "R":
            if v[a[0]]:
                return None  # dirty reset: not a valid execution
        else:
            raise ValueError(f"fixture op not supported: {ln}")
    return v, bits, sign


def check_fixture(lhs, rhs, zero=()):
    """zero: wires the identity assumes clean on entry (e.g. a free ancilla); they are fixed to 0."""
    qs1, bs1 = wires_bits(lhs)
    qs2, bs2 = wires_bits(rhs)
    qs, bs = sorted((qs1 | qs2) - set(zero)), sorted(bs1 | bs2)
    worst = None
    for vals in itertools.product((0, 1), repeat=len(qs)):
        st = dict(zip(qs, vals)) | {z: 0 for z in zero}
        for outs in itertools.product((0, 1), repeat=len(bs)):
            oc = dict(zip(bs, outs))
            r1, r2 = run(lhs, st, oc), run(rhs, st, oc)
            if r1 is not None and r2 is not None:
                # a bit written only by the rhs is a fresh record (precondition: unused elsewhere); compare the rest
                keep = set(r1[1])
                r2 = (r2[0], {k: x for k, x in r2[1].items() if k in keep}, r2[2])
            if r1 != r2:
                worst = {"input": st, "outcomes": oc, "lhs": r1, "rhs": r2}
                return False, worst
    return True, {"inputs": 2 ** len(qs), "outcome_assignments": 2 ** len(bs)}


def toffolis(lines):
    return sum(parse(ln)[0] in ("CCX", "CCZ", "CSWAP") for ln in lines)


def cmd_validate(reg):
    bad = 0
    for e in reg["identities"]:
        miss = [f for f in FIELDS if f not in e]
        probs = [f"missing {miss}"] if miss else []
        if e.get("validity") not in VALIDITY:
            probs.append(f"validity {e.get('validity')!r}")
        if e.get("status") not in STATUS:
            probs.append(f"status {e.get('status')!r}")
        if e.get("status") == "tool-backed" and not e.get("detector"):
            probs.append("tool-backed without a detector")
        d = toffolis(e["rhs"]) - toffolis(e["lhs"]) if e.get("lhs") and e.get("rhs") else None
        if d is not None and d != e.get("toffoli_delta"):
            probs.append(f"fixture Toffoli delta {d} != declared {e.get('toffoli_delta')}")
        if e.get("validity") == "exact" and e.get("lhs"):
            ok, info = check_fixture(e["lhs"], e["rhs"], e.get("fixture_zero", []))
            if not ok:
                probs.append(f"fixture NOT equal: {info}")
            verdict = f"fixture exact over {info.get('inputs')} inputs x {info.get('outcome_assignments')} outcomes" \
                if ok else "fixture FAILS"
        else:
            verdict = "reachable: certify whole circuits" if e.get("validity") == "reachable" else "no fixture"
        bad += bool(probs)
        print(f"{e.get('id', '?'):7} {'OK ' if not probs else 'BAD'} {e.get('name', '')[:52]:52} {verdict}")
        for p in probs:
            print(f"        {p}")
    print(f"{len(reg['identities'])} identities, {bad} with problems")
    return 1 if bad else 0


PRIVATE_MARKS = ("research/", "experiments/", "X-0", "card ", "ssr_", "R T14", "NR-0", "outputs/")
PUBLIC_TOOLS = ("scripts/qhdl.py", "scripts/c20_and_merge_scan.py", "scripts/qhdl_phase_fold.py")


def cmd_export(reg, path):
    """Public copy: drop evidence lines and detectors that point at non-public material; keep everything else."""
    out = {"schema": "README.md (identities section)", "updated": reg["updated"], "identities": []}
    for e in reg["identities"]:
        e = dict(e)
        e["evidence"] = [x for x in e["evidence"] if not any(m in x for m in PRIVATE_MARKS)]
        for k in ("detector", "emitter"):
            if e[k] and not any(tool in e[k] for tool in PUBLIC_TOOLS):
                e[k] = ""
        if not e["detector"]:
            e["status"] = "catalogued"
        e["credit"] = (e["credit"].replace("the rule and emitter are R's (ssr_quadratic_rank)", "an exact rule exists in our research code")
                       .replace("; R's ssr_public_clifford_splice", ""))
        out["identities"].append(e)
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    leaks = [m for m in PRIVATE_MARKS if m in text]
    if leaks:
        print(f"refusing: internal references remain: {leaks}")
        return 1
    Path(path).write_text(text)
    print(f"wrote {path}: {len(out['identities'])} identities, "
          f"{sum(e['status'] == 'tool-backed' for e in out['identities'])} tool-backed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("--task")
    s = sub.add_parser("show")
    s.add_argument("id")
    sub.add_parser("validate")
    s = sub.add_parser("export")
    s.add_argument("--public", required=True, type=Path, help="write a public copy (internal references removed)")
    s = sub.add_parser("scan")
    s.add_argument("-c", "--circuit", required=True)
    s.add_argument("--task", required=True, choices=["invsqrt", "dqi", "chemistry"])
    s.add_argument("--run", action="store_true")
    s.add_argument("--out", default=None, help="directory for detector outputs (default: $QHDL_WORK/identities)")
    a = ap.parse_args()
    reg = load()
    if a.cmd == "list":
        for e in reg["identities"]:
            if a.task and a.task not in e["tasks"]:
                continue
            print(f"{e['id']:7} {e['validity']:9} {e['toffoli_delta']:+d} T {e['status']:11} {e['name']}")
        return 0
    if a.cmd == "show":
        e = next((x for x in reg["identities"] if x["id"] == a.id), None)
        print(json.dumps(e, indent=1, ensure_ascii=False) if e else f"no identity {a.id}")
        return 0 if e else 1
    if a.cmd == "validate":
        return cmd_validate(reg)
    if a.cmd == "export":
        return cmd_export(reg, a.public)
    import os
    out = Path(a.out or Path(os.environ.get("QHDL_WORK", ROOT / "work")) / "identities").resolve()
    circ = Path(a.circuit).resolve()
    kmx = circ if circ.suffix == ".kmx" else out / (circ.stem + ".kmx")
    if a.run:
        out.mkdir(parents=True, exist_ok=True)
        if not kmx.exists():   # text form for detectors that read .kmx
            subprocess.run([os.environ.get("QHDL_KICKMIX_PYTHON", sys.executable), "-c",
                            "import kickmix,sys;kickmix.Circuit.from_file(sys.argv[1]).to_file(sys.argv[2],format='kmx')",
                            str(circ), str(kmx)], check=True)
    rc = 0
    for e in reg["identities"]:
        if e["status"] != "tool-backed" or a.task not in e["tasks"]:
            continue
        cmd = e["detector"].format(circuit=shlex.quote(str(circ)), kmx=shlex.quote(str(kmx)), out=shlex.quote(str(out)))
        print(f"== {e['id']} {e['name']} ({e['validity']})\n   {cmd}")
        if a.run:
            cp = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=3600)
            print("   " + "\n   ".join((cp.stdout.strip() or cp.stderr.strip()).splitlines()[-6:]))
            rc |= cp.returncode != 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
