#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""C20: semantic AND-merge scanner for invsqrt circuits (reachable-value version of the quadratic-rank rule).

Signature (after Jeff Emerson's public AND-merge cut on Syed Asad Ali's 44-qubit invsqrt, re-implemented here from
the idea): a target wire t is written several times (X, CX into t, CCX into t), with at least two CCX, and never read
in between (no control use, phase, measurement, reset or swap). The net toggle f that this segment applies to t is a
function of reachable values. If f = L1*L2 + L3 with L1, L2, L3 affine in the other wires' values at some point k of
the segment, every write in the segment can be replaced by ONE CCX at k on in-place parity-combined controls, plus
free CX/X into t: (number of CCX in the segment) - 1 Toffolis fewer. If f is affine, all of them go.

Fitting uses wire values over sampled legal x in the all-zero and a seeded random measurement branch (BitSim
snapshots). A sample fit is only a proposal: every emitted circuit must pass `qhdl fdiff --whole` (all legal x, two
branches) and the official verifier.

Usage: python3 scripts/c20_and_merge_scan.py CIRCUIT.kmx OUT_DIR [--rows N] [--support S] [--emit]
Writes OUT_DIR/scan.json (one row per segment) and, with --emit, OUT_DIR/merge_<i>.kmx per accepted site plus
OUT_DIR/merge_all.kmx with every non-overlapping site applied.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_bitsim as B  # noqa: E402

WRITE = (B.C_X, B.C_CX, B.C_CCX)


def target_of(o):
    return o.q[-1] if o.code in WRITE else None


def segments(ops):
    """Per wire t: maximal runs of ops touching t in which t is only written (X/CX/CCX target). Returns list of
    (t, [op indices of writes]) with at least two CCX writes."""
    touch: dict[int, list[int]] = {}
    for i, o in enumerate(ops):
        for qq in set(o.q or ()):
            touch.setdefault(qq, []).append(i)
    out = []
    for t, idx in touch.items():
        run = []
        for i in idx:
            o = ops[i]
            if o.code in WRITE and o.q[-1] == t and t not in o.q[:-1]:
                run.append(i)
            else:
                if sum(ops[j].code == B.C_CCX for j in run) >= 2:
                    out.append((t, run))
                run = []
        if sum(ops[j].code == B.C_CCX for j in run) >= 2:
            out.append((t, run))
    return sorted(out, key=lambda s: s[1][0])


def depth_at(ops):
    d, out = 0, []
    for o in ops:
        out.append(d)
        if o.code == B.C_PUSH:
            d += 1
        elif o.code == B.C_POP:
            d -= 1
    return out


def to_int(words):
    return int.from_bytes(np.ascontiguousarray(words).view(np.uint8).tobytes(), 'little')


def affine_fit(target, cols, mask):
    """Find (S, c): XOR of cols[S] ^ c*mask == target on mask. cols: dict name->int. Returns None if impossible."""
    basis = []  # (vec, combo set, const)
    items = [(v & mask, {k}, 0) for k, v in cols.items()] + [(mask, set(), 1)]
    piv = {}
    for v, combo, c in items:
        combo = set(combo)
        while v:
            p = v.bit_length() - 1
            if p in piv:
                pv, pc, pk = piv[p]
                v ^= pv
                combo ^= pc
                c ^= pk
            else:
                piv[p] = (v, combo, c)
                break
    r, combo, c = target & mask, set(), 0
    while r:
        p = r.bit_length() - 1
        if p not in piv:
            return None
        pv, pc, pk = piv[p]
        r ^= pv
        combo ^= pc
        c ^= pk
    return combo, c


def fit_product(f, cols, mask, support_wires, max_support):
    """f = L1*L2 + L3 on mask. Returns (L1, (L2set, c2), (L3set, c3)) or None. L1 has no constant."""
    a = affine_fit(f, cols, mask)
    if a is not None:
        return ("affine", None, a)
    for s in range(1, max_support + 1):
        for L1 in itertools.combinations(support_wires, s):
            v1 = 0
            for w in L1:
                v1 ^= cols[w]
            m1, m0 = mask & v1, mask & ~v1
            if not m1 or not m0:
                continue
            l3 = affine_fit(f, cols, m0)
            if l3 is None:
                continue
            m = affine_fit(f, cols, m1)
            if m is None:
                continue
            L2 = (l3[0] ^ m[0], l3[1] ^ m[1])
            return (set(L1), L2, l3)
    return None


def emission(t, L1, L2, L3):
    """Text ops placing L1*L2 + L3 into t with in-place pivots, or None if no valid pivots."""
    L1, (s2, c2), (s3, c3) = set(L1), L2, L3
    s2 = set(s2)
    for flip in (False, True):
        a, b, bc, l3s, l3c = L1, s2, c2, set(s3), c3
        if flip:  # L1*L2 = L1*(L2+L1) + L1
            b, l3s = b ^ a, l3s ^ a
        p1s = [w for w in sorted(a) if w not in b]
        p2s = [w for w in sorted(b) if w not in a]
        if not p1s or not p2s or not b:
            continue
        p1, p2 = p1s[0], p2s[0]
        pre = [f"CX q{w} q{p1}" for w in sorted(a - {p1})] + [f"CX q{w} q{p2}" for w in sorted(b - {p2})]
        if bc:
            pre.append(f"X q{p2}")
        body = pre + [f"CCX q{p1} q{p2} q{t}"] + pre[::-1]
        body += [f"CX q{w} q{t}" for w in sorted(l3s)] + ([f"X q{t}"] if l3c else [])
        return body
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--support", type=int, default=3)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--allow-conditional", action="store_true",
                    help="also fit segments containing conditional writes (their predicates are then only sampled)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    ops, header, xq, nq = B.load(a.circuit)
    dep = depth_at(ops)
    segs = segments(ops)
    rng = np.random.default_rng(20261115)
    # random rows plus the edges of the domain (smallest and largest x), where rare carries and branches live;
    # a 4,096-random-row sample let a false affine fit through at q8 (213 inputs differ, x = 1 among them)
    edge = np.r_[np.arange(1, 2049), np.arange(B.MAX_X - 2047, B.MAX_X + 1)]
    pool = np.setdiff1d(np.arange(1, B.MAX_X + 1), edge)
    rows = np.sort(np.r_[edge, rng.choice(pool, size=a.rows, replace=False)])
    # snapshot points: every write position and the op after the last write, in both branches
    need = sorted({i for _, run in segs for i in range(run[0], run[-1] + 2)})
    snaps = {}
    for branch in ("zero", 7):
        store = {}

        def on_snap(i, sim, store=store):
            store[i] = [to_int(sim.q[w]) for w in range(nq)]
        sim = B.BitSim(ops, rows=rows, num_qubits=nq, x_qubits=xq, outcome=branch)
        sim.run(snaps=[i for i in need if i < len(ops)], on_snap=on_snap)
        if len(ops) in need:
            store[len(ops)] = [to_int(sim.q[w]) for w in range(nq)]
        snaps[branch] = store
    n = len(rows)
    full = (1 << (2 * n)) - 1

    def state(i):  # both branches concatenated: branch zero in the low n bits
        z, r = snaps["zero"][i], snaps[7][i]
        return [z[w] | (r[w] << n) for w in range(nq)]
    results = []
    for si, (t, run) in enumerate(segs):
        nccx = sum(ops[j].code == B.C_CCX for j in run)
        # Review rule: a removed write that is conditional (own `if bN` or inside a condition block) is a
        # predicate the sampled fit only sees statistically; reject such segments unless --allow-conditional
        conditional = [j for j in run if ops[j].cond not in (None, -1) or dep[j]]
        if conditional and not a.allow_conditional:
            results.append({"segment": si, "target": t, "writes": run, "ccx": nccx, "found": None,
                            "rejected": f"{len(conditional)} conditional write(s), first at op {conditional[0]}"})
            continue
        before, after = state(run[0]), state(run[-1] + 1)
        f = before[t] ^ after[t]
        touched = sorted({w for j in range(run[0], run[-1] + 1) for w in (ops[j].q or ()) if w != t})
        row = {"segment": si, "target": t, "writes": run, "ccx": nccx, "found": None}
        for k in range(run[0], run[-1] + 2):   # any position in the segment (the merged CCX may sit between them)
            if dep[k] if k < len(ops) else 0:
                continue
            st = state(k)
            cols = {w: st[w] for w in range(nq) if w != t}
            sup = touched if len(touched) <= 18 else touched[:18]
            fit = fit_product(f, cols, full, sup, a.support)
            if fit is None:
                continue
            if fit[0] == "affine":
                s3, c3 = fit[2]
                body = [f"CX q{w} q{t}" for w in sorted(s3)] + ([f"X q{t}"] if c3 else [])
                row["found"] = {"k": k, "kind": "affine", "saves": nccx, "body": body}
                break
            body = emission(t, *fit)
            if body:
                row["found"] = {"k": k, "kind": "product", "saves": nccx - 1, "body": body,
                                "L1": sorted(fit[0]), "L2": [sorted(fit[1][0]), fit[1][1]],
                                "L3": [sorted(fit[2][0]), fit[2][1]]}
                break
        results.append(row)
    hits = [r for r in results if r["found"] and r["found"]["saves"] > 0]
    summary = {"circuit": str(a.circuit), "ops": len(ops), "segments": len(segs), "hits": len(hits),
               "toffolis_saved_static": sum(r["found"]["saves"] for r in hits), "rows_per_branch": n,
               "branches": ["zero", "seed 7"], "support_max": a.support, "seconds": round(time.time() - t0, 1),
               "tier": "local prescreen (sampled fit); emitted circuits need fdiff --whole and the official verifier"}
    (a.out / "scan.json").write_text(json.dumps({"summary": summary, "segments": results}, indent=1))
    print(json.dumps(summary))
    for r in hits:
        print(f"  q{r['target']} writes {r['writes'][0]}..{r['writes'][-1]} ccx {r['ccx']} -> "
              f"{r['found']['kind']} at {r['found']['k']} saves {r['found']['saves']}")
    if a.emit and hits:
        lines = [o.text for o in ops]

        def build(chosen):
            drop = {j for r in chosen for j in r["writes"]}
            ins = {r["found"]["k"]: r["found"]["body"] for r in chosen}
            out = []
            for i in range(len(lines) + 1):
                out += ins.get(i, [])
                if i < len(lines) and i not in drop:
                    out.append(lines[i])
            return header + out
        for r in hits:
            (a.out / f"merge_{r['segment']}.kmx").write_text("\n".join(build([r])) + "\n")
        chosen, used = [], set()
        for r in hits:
            span = set(range(r["writes"][0], r["writes"][-1] + 2))
            if not span & used:
                chosen.append(r)
                used |= span
        (a.out / "merge_all.kmx").write_text("\n".join(build(chosen)) + "\n")
        print(f"emitted {len(hits)} single-site circuits and merge_all.kmx ({len(chosen)} sites)")


if __name__ == "__main__":
    main()
