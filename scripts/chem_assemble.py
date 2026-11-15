#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Per-slot chemistry assembler over the chemistry parts ledger.

For every word table in the parts ledger (scripts/chem_parts_ledger.py) and every public register layout used with
it, takes the cheapest loader per slot among the parts that perform exactly that slot's transition (block k-1 ->
block k of the table) with that layout, then pairs the loader set with every npz that passes the accuracy gates on
the table (compatibility.csv). Predicted cost (static model from parts, not a verifier result):

    cost = lambda_eff(npz, table) * (sum of slot Toffolis per shot + heuristic Toffolis) * Q

with the slot Toffolis measured in each part's first family and Q from the verifier's resource model. The best --top
distinct candidates are written as `<out>/cand_NN/dfthcsos/{submission.npz, angle_k_*.kmb}` (files copied and
re-hashed) with a parts.json credit list; --verify runs each through the unmodified official verifier
(scripts/local_verify_reference.py --tasks dfthcsos, FCIDUMP cache on). Every part's author is credited.

With --best-json PATH (the dataset's parts/assembly_best.json) the top candidate is compared with the stored one
(same npz and the same file in every slot = unchanged, nothing run). If it differs it is built and verified, and PATH
is rewritten only if it PASSES and is cheaper than the stored verified cost (or the stored assembly can no longer be
built from the dataset). Equal cost keeps the stored file, so the result is stable.

Usage:
  $QHDL_KICKMIX_PYTHON -I scripts/chem_assemble.py --out $QHDL_OUTPUTS/<new dir> \
      [--parts-dir $QHDL_DATASET/parts] [--probe-dir PROBE_CACHE] [--top 4] [--verify]
      [--best-json $QHDL_DATASET/parts/assembly_best.json]   (implies --top 1 unless given, verifies when changed)
      [--ledger <evidence dir>]    (an evidence dir with probe/ and parts/ instead)
The defaults are the dataset's parts/ and the probe cache of scripts/chem_parts_ledger.py.
Writes <out>/candidates.csv (all ranked combinations), <out>/verified.csv and <out>/summary.json. Refuses to write
into an existing candidate directory (evidence is never overwritten).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chem_parts_ledger as L  # noqa: E402

import vqcc_config as CFG  # noqa: E402
KICKMIX_PY = CFG.KMPY


def slot_options(probes):
    """All observed (part, transition, layout, measured T) records from PASS probes."""
    opts = {}
    for f, r in probes.items():
        if r.get("status") != "PASS":
            continue
        for p in r["parts"]:
            key = (p["sha256"], p["k"], p["prev_words_sha256"], p["words_sha256"], p["layout_sha256"])
            e = opts.setdefault(key, {"sha256": p["sha256"], "k": p["k"], "prev": p["prev_words_sha256"],
                                      "words": p["words_sha256"], "layout": p["layout_sha256"],
                                      "num_qubits": p["num_qubits"], "static_toffoli": p["static_toffoli"],
                                      "selection_bits": p["selection_bits"], "T": [], "families": [],
                                      "file": p["file"]})
            e["T"].append(p["toffoli_per_shot"])
            e["families"].append(f)
    return list(opts.values())


def cheapest_loaders(table_blocks, options, first_rank):
    """{layout: [option per slot]} choosing the minimum first-family T per slot; layouts missing a slot dropped.

    table_blocks: ordered words sha per slot; options: slot_options(); first_rank: {family: release order}.
    Each option's T is that of its first family (by release order).
    """
    chain = L.chain(table_blocks)
    by_layout = {}
    for o in options:
        k = o["k"]
        if k >= len(chain) or (o["prev"], o["words"]) != chain[k]:
            continue
        fam0 = min(o["families"], key=lambda f: first_rank.get(f, 1 << 30))
        t0 = o["T"][o["families"].index(fam0)]
        cand = {**o, "T0": t0, "first_family": fam0}
        slots = by_layout.setdefault(o["layout"], [None] * len(chain))
        cur = slots[k]
        if cur is None or (t0, first_rank.get(fam0, 1 << 30)) < (cur["T0"], first_rank.get(cur["first_family"],
                                                                                        1 << 30)):
            slots[k] = cand
    return {lay: s for lay, s in by_layout.items() if all(x is not None for x in s)}


def rank_candidates(tables, options, compat_rows, factorizations, heuristic, first_rank):
    """All (table, layout, npz) combinations with predicted costs, cheapest first.

    heuristic(R, B, C, brot) -> dict like verify.heuristic_costs.
    """
    fz = {r["npz_sha256"]: r for r in factorizations}
    cands = []
    for t in tables:
        blocks = t["block_words_sha256"].split(";")
        loaders = cheapest_loaders(blocks, options, first_rank)
        for c in compat_rows:
            if c["table_key"] != t["table_key"] or str(c["pass"]) != "True":
                continue
            f = fz[c["npz_sha256"]]
            heur = heuristic(int(f["R"]), int(f["B"]), int(f["C"]), int(t["brot"]))
            for lay, slots in loaders.items():
                T = sum(s["T0"] for s in slots)
                pred = L.predict_cost(float(c["lambda_eff"]), T, heur, int(t["J"]), int(t["brot"]),
                                      slots[0]["selection_bits"], max(s["num_qubits"] for s in slots))
                cands.append({"table_key": t["table_key"], "table_label": t["table_label"], "layout_sha256": lay, "npz_sha256": c["npz_sha256"],
                              "npz_family": f["first_family"], "lambda_eff": float(c["lambda_eff"]),
                              "angle_toffoli": T, **{f"pred_{k}": v for k, v in pred.items()}, "slots": slots})
    cands.sort(key=lambda c: c["pred_cost"])
    return cands


def build_candidate(cand, probes, manifest, dest):
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; outputs/ is never overwritten")
    d = dest / "dfthcsos"
    d.mkdir(parents=True)
    npz_src = Path(probes[cand["npz_family"]]["dir"]) / "submission.npz"
    shutil.copyfile(npz_src, d / "submission.npz")
    if L.sha256_file(d / "submission.npz") != cand["npz_sha256"]:
        raise RuntimeError("npz hash changed in copy")
    credits = []
    for s in cand["slots"]:
        src = Path(probes[s["first_family"]]["dir"]) / s["file"]
        out = d / f"angle_k_{s['k']}.kmb"
        shutil.copyfile(src, out)
        if L.sha256_file(out) != s["sha256"]:
            raise RuntimeError(f"slot {s['k']} hash changed in copy")
        cr = L.credit(manifest, s["first_family"])
        credits.append({"slot": s["k"], "part_id": f"k{s['k']:02d}-{s['sha256'][:12]}",
                        "dataset_path": L.angle_path(s["first_family"], s["k"]),
                        "file_sha256": s["sha256"], "source_family": s["first_family"],
                        "author": cr["author"], "notebook": cr["notebook"], "notebook_version": cr["notebook_version"],
                        "release": cr["release"], "toffoli_per_shot_source": s["T0"],
                        "also_in": sorted(set(s["families"]) - {s["first_family"]})})
    cr = L.credit(manifest, cand["npz_family"])
    record = {"table_key": cand["table_key"], "table_label": cand["table_label"],
              "layout_sha256": cand["layout_sha256"],
              "npz": {"sha256": cand["npz_sha256"], "dataset_path": f"factorizations/{cand['npz_family']}.npz",
                      "source_family": cand["npz_family"], **cr},
              "slots": credits, "prediction": {k: cand[k] for k in cand if k.startswith("pred_") or
                                               k in ("lambda_eff", "angle_toffoli")},
              "files_sha256": {p.name: L.sha256_file(p) for p in sorted(d.iterdir())}}
    (dest / "parts.json").write_text(json.dumps(record, indent=1))
    return record


def verify_candidate(dest):
    report = Path(dest) / "report.json"
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    t = time.time()
    p = subprocess.run([KICKMIX_PY, str(L.REPO / "scripts" / "local_verify_reference.py"), "--root",
                        str(CFG.OFFICIAL_ROOT), "--submission", str(dest), "--out", str(report),
                        "--tasks", "dfthcsos"], capture_output=True, text=True, cwd=str(L.REPO), env=env)
    (Path(dest) / "verify.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    res = {"returncode": p.returncode, "seconds": round(time.time() - t, 1)}
    r = json.loads(report.read_text()).get("results", {}).get("dfthcsos") if report.exists() else None
    if r is None:  # the official verify() raised: the log holds the reason
        tail = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
        res.update(status="FAIL", error=tail[-1][:400] if tail else "no report")
    else:
        s = r.get("summary") or {}
        res.update(status=s.get("status", "FAIL"), cost=s.get("score"), lambda_eff=s.get("lambda_eff"),
                   toffoli=s.get("toffoli_count"), qubits=s.get("num_qubits"),
                   submission_sha256=s.get("submission_sha256"), error=r.get("error"),
                   report_sha256=L.sha256_file(report))
    return res


def assembly_signature(npz_sha256, slot_shas):
    return (npz_sha256, tuple(slot_shas))


def stored_signature(best):
    if not best:
        return None
    return assembly_signature(best["npz"]["sha256"], [s["file_sha256"] for s in sorted(best["slots"],
                                                                                    key=lambda s: s["slot"])])


def best_decision(stored, cand_sig, verified, stored_buildable):
    """What to do with assembly_best.json. stored: the current JSON or None; cand_sig: signature of the top candidate;
    verified: None (not run) or {'status', 'cost'}; stored_buildable: every stored part is still in the ledger.
    Returns 'unchanged' | 'verify' | 'write' | 'keep'."""
    if stored is not None and stored_signature(stored) == cand_sig and stored_buildable:
        return "unchanged"
    if verified is None:
        return "verify"
    if verified.get("status") != "PASS":
        return "keep"
    if stored is None or not stored_buildable or float(verified["cost"]) < float(stored["verified_cost"]):
        return "write"
    return "keep"


def best_record(rec, cand, v, tables, manifest, best_public):
    """assembly_best.json content from a verified candidate's parts.json record."""
    t = next(x for x in tables if x["table_key"] == cand["table_key"])
    wc = L.credit(manifest, t["first_family"])
    return {
        "schema": 1, "task": "chemistry", "tier": "local official verifier, default salt (not a leaderboard score)",
        "table_key": cand["table_key"], "table_label": cand["table_label"],
        "word_table_origin": {"family": t["first_family"], "author": wc["author"], "notebook": wc["notebook"],
                              "release": wc["release"]},
        "npz": rec["npz"], "slots": rec["slots"],
        "lambda_eff": v["lambda_eff"], "toffoli": v["toffoli"], "qubits": v["qubits"],
        "predicted_cost": cand["pred_cost"], "verified_cost": v["cost"], "verified_status": v["status"],
        "verify_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "verifier_submission_sha256": v.get("submission_sha256"), "verifier_report_sha256": v.get("report_sha256"),
        "best_public_family": best_public["family"], "best_public_family_cost": best_public["score"],
        "authors": sorted({s["author"] for s in rec["slots"]} | {rec["npz"]["author"], wc["author"]}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-slot chemistry assembler over the parts ledger.")
    ap.add_argument("--ledger", help="evidence dir with probe/ and parts/ (instead of --probe-dir/--parts-dir)")
    ap.add_argument("--probe-dir", default=str(L.DEFAULT_PROBES))
    ap.add_argument("--parts-dir", default=str(L.STAGE_PARTS))
    ap.add_argument("--out", required=True, help="a new directory for candidates and reports")
    ap.add_argument("--manifest", default=str(L.DEFAULT_MANIFEST))
    ap.add_argument("--verifiers", default=str(L.DEFAULT_VERIFIERS))
    ap.add_argument("--top", type=int)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--best-json", help="parts/assembly_best.json to compare with and update (only after a PASS)")
    ap.add_argument("--control-cost", type=float, default=340781040.24)
    args = ap.parse_args(argv)
    if args.ledger:
        args.probe_dir, args.parts_dir = L.resolve(args.ledger) / "probe", L.resolve(args.ledger) / "parts"
    top = args.top or (1 if args.best_json else 4)
    parts_dir = L.resolve(args.parts_dir)
    out = L.resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dv = L.import_verifier(L.resolve(args.verifiers))
    manifest = L.load_manifest(L.resolve(args.manifest))
    tables = L.read_csv(parts_dir / "word_tables.csv")
    if tables and "table_key" not in tables[0]:  # the 2026-10-29 evidence layout
        for t in tables:
            t["table_key"], t["table_label"] = L.table_key(t["table_sha256"]), t["table_id"]
    lab2key = {t["table_label"]: t["table_key"] for t in tables}
    compat = L.read_csv(parts_dir / "compatibility.csv")
    for c in compat:
        c.setdefault("table_key", lab2key.get(c.get("table_id")))
    fz = L.read_csv(parts_dir / "factorizations.csv")
    present = {r["family"] for r in L.read_csv(parts_dir / "families.csv")}
    probes = L.load_probes(L.resolve(args.probe_dir), only=present)
    order = sorted(probes, key=lambda f: L.release_key(manifest.get(f, {}).get("release_date"), f))
    first_rank = {f: i for i, f in enumerate(order)}
    options = slot_options(probes)
    cands = rank_candidates(tables, options, compat, fz,
                            lambda R, B, C, brot: dv.heuristic_costs(L.N_ORBITALS, R, B, C, brot), first_rank)
    rows = [{**{k: v for k, v in c.items() if k != "slots"},
             "parts": ";".join(f"k{s['k']:02d}-{s['sha256'][:12]}@{s['first_family']}" for s in c["slots"])}
            for c in cands]
    L.write_csv(out / "candidates.csv", rows, list(rows[0]))
    best_public = min((r for r in probes.values() if r["status"] == "PASS"), key=lambda r: r["score"])
    summary = {"combinations": len(cands), "best_public_family": best_public["family"],
               "best_public_cost": best_public["score"], "control_manual_merge_cost": args.control_cost, "top": []}
    picked, seen = [], set()
    for c in cands:  # distinct file sets only
        sig = assembly_signature(c["npz_sha256"], [s["sha256"] for s in c["slots"]])
        if sig not in seen:
            seen.add(sig)
            picked.append(c)
        if len(picked) >= top:
            break

    stored = None
    if args.best_json and Path(args.best_json).is_file():
        stored = json.loads(Path(args.best_json).read_text())
    have = {(o["sha256"], o["k"]) for o in options}
    stored_buildable = bool(stored) and all((s["file_sha256"], s["slot"]) in have for s in stored["slots"]) and \
        any(r["npz_sha256"] == stored["npz"]["sha256"] for r in fz)

    ver_rows = []
    for i, c in enumerate(picked):
        sig = assembly_signature(c["npz_sha256"], [s["sha256"] for s in c["slots"]])
        decision = best_decision(stored, sig, None, stored_buildable) if (args.best_json and i == 0) else None
        row = {"candidate": f"cand_{i:02d}", "table_key": c["table_key"], "table_label": c["table_label"],
               "npz_family": c["npz_family"], "pred_lambda": c["lambda_eff"], "pred_toffoli": c["pred_toffoli"],
               "pred_qubits": c["pred_qubits"], "pred_cost": c["pred_cost"]}
        if decision == "unchanged":
            row.update(best_json="unchanged (same files as the stored assembly; not re-verified)")
            ver_rows.append(row)
            summary["top"].append(row)
            print(f"cand_{i:02d}: same as {args.best_json}; nothing to verify", flush=True)
            continue
        dest = out / f"cand_{i:02d}"
        rec = build_candidate(c, probes, manifest, dest)
        row.update(part_authors=";".join(sorted({s["author"] for s in rec["slots"]})), npz_author=rec["npz"]["author"])
        if args.verify or decision == "verify":
            v = verify_candidate(dest)
            row.update({f"verified_{k}": v.get(k) for k in ("status", "cost", "lambda_eff", "toffoli", "qubits",
                                                           "submission_sha256", "report_sha256", "seconds",
                                                           "error")})
            if v.get("toffoli") is not None:
                row["toffoli_dev"] = v["toffoli"] - c["pred_toffoli"]
                row["lambda_dev"] = v["lambda_eff"] - c["lambda_eff"]
            print(dest.name, row.get("verified_status"), row.get("verified_cost"), "pred", c["pred_cost"],
                  flush=True)
            if decision is not None:
                d2 = best_decision(stored, sig, v, stored_buildable)
                row["best_json"] = d2
                if d2 == "write":
                    best = best_record(rec, c, v, tables, manifest, best_public)
                    Path(args.best_json).write_text(json.dumps(best, indent=1, ensure_ascii=False) + "\n")
                    (out / "assembly_best.json").write_text(json.dumps(best, indent=1, ensure_ascii=False) + "\n")
                    print(f"wrote {args.best_json}: verified {v['cost']}", flush=True)
        ver_rows.append(row)
        summary["top"].append(row)
    cols = list(dict.fromkeys(k for r in ver_rows for k in r))
    L.write_csv(out / "verified.csv", ver_rows, cols)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
