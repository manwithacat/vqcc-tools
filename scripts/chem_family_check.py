#!/usr/bin/env python3
"""Chemistry family consistency check against the parts ledger (scripts/chem_parts_ledger.py).

Answers, for one chemistry family (submission.npz + angle_k_*.kmb), "what did this release change, and does the
change keep the system consistent?" even when every file on its own is valid. It compares the family with a
reference set of earlier families from scripts/chem_parts_ledger.py:

  * --all: every family in the ledger against the families released strictly before it (manifest release_date;
    a date without a time counts as 00:00, so same-minute releases do not see each other);
  * FAMILY_DIR: one directory against the whole ledger (minus itself). If the directory is a ledger family its
    probe is reused; otherwise its files are hashed and read statically (kickmix, no simulation), and with
    --simulate the official evaluate_circuits is run to recover its words.

One verdict line per family: class (identical copy / loader-only / weights-only / loader+weights / new word table /
FAILS VERIFIER), the table it builds on, lambda_eff and Toffolis against the best earlier family on that table,
status OK / NOTE / BREAK, and flags. A detail block follows for every family with flags. Flags (explained in plain
words in FLAG_HELP below): VERIFY_FAIL, CHAIN_BREAK, UNKNOWN_PART, NEW_WORD_TABLE, MIXED_TABLES,
LAYOUT_MISMATCH, TREE_MISMATCH, WEIGHTS_FAIL_ON_TABLE, NPZ_WORDS_DIFFER.

Usage:
  python3 scripts/chem_family_check.py --all [--families SHA12 ...] [--since 2026-10-29] [--md R.md] [--json R.json]
      [--probe-dir PROBE_CACHE] [--parts-dir $QHDL_DATASET/parts]   (the defaults of scripts/chem_parts_ledger.py)
      [--ledger <evidence dir>]   (an evidence dir with probe/ and parts/ instead)
  $QHDL_KICKMIX_PYTHON -I scripts/chem_family_check.py FAMILY_DIR [--simulate]
--all needs only python3 with numpy; FAMILY_DIR outside the ledger needs a Python that can import kickmix.
The checking logic (check_family, reference_from_probes, classify) is pure and unit-tested in tests/test_chem_parts.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chem_parts_ledger as L  # noqa: E402

FLAG_HELP = {
    "VERIFY_FAIL": ("BREAK", "the official verifier rejects the family as shipped"),
    "CHAIN_BREAK": ("BREAK", "a loader was built to start from a different word block than the previous slot "
                             "leaves; it would load garbage or fail the clean-up checks"),
    "UNKNOWN_PART": ("NOTE", "an angle file is not in the ledger and was not simulated, so its words are unknown"),
    "NEW_WORD_TABLE": ("NOTE", "at least one word block differs from every earlier table: the angles themselves "
                               "changed, not just a loader or the weights"),
    "MIXED_TABLES": ("NOTE", "word blocks come from two or more earlier tables and no single table matches"),
    "LAYOUT_MISMATCH": ("BREAK|NOTE", "BREAK: slots use different selection/word qubit numbering (the verifier "
                                      "binds all to circuit 0); NOTE: a numbering not seen before with this table, "
                                      "so its loaders cannot be mixed with the table's earlier loaders"),
    "TREE_MISMATCH": ("NOTE", "the npz's Givens tree differs from the tree(s) earlier families used with this "
                              "word table; the same integers then mean different rotations"),
    "WEIGHTS_FAIL_ON_TABLE": ("BREAK|NOTE", "the npz fails the accuracy gates on the table it builds on: BREAK if "
                                            "that is the family's own table, NOTE if the weights only fit new words"),
    "NPZ_WORDS_DIFFER": ("NOTE", "the npz's stored angle_words are not the words the circuits load (the verifier "
                                 "uses the loaded words; the npz copy is stale or from another table)"),
}


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def reference_from_probes(probes, table_id, metrics=True):
    """Reference state from a set of probe records (dicts as written by chem_parts_ledger.py probe).

    table_id: {table_sha256: 'Txx'}. Only PASS families contribute parts and tables.
    Returns {parts: {(sha, k): {...}}, tables: {table_sha: {...}}, npz: {sha: [families]}}.
    """
    ref = {"parts": {}, "tables": {}, "npz": {}}
    for f, r in probes.items():
        ref["npz"].setdefault(r["npz_sha256"], []).append(f)
        if r.get("status") != "PASS":
            continue
        t = ref["tables"].setdefault(r["table_sha256"], {
            "table_id": table_id.get(r["table_sha256"], "T??"), "blocks": r["block_words_sha256"],
            "J": r["J"], "rows": r["rows"], "trees": set(), "layouts": set(), "families": [],
            "best_lambda": None, "best_T": None, "best_cost": None, "best_family": None})
        t["trees"].add(r["tree_edges_sha256"])
        t["layouts"].update(p["layout_sha256"] for p in r["parts"])
        t["families"].append(f)
        if metrics:
            lam, T, cost = r["metrics"]["lambda_eff"], r["resources"]["toffoli"], r["score"]
            t["best_lambda"] = lam if t["best_lambda"] is None else min(t["best_lambda"], lam)
            t["best_T"] = T if t["best_T"] is None else min(t["best_T"], T)
            if t["best_cost"] is None or cost < t["best_cost"]:
                t["best_cost"], t["best_family"] = cost, f
        for p in r["parts"]:
            e = ref["parts"].setdefault((p["sha256"], p["k"]), {"chains": set(), "families": []})
            e["chains"].add((p["prev_words_sha256"], p["words_sha256"]))
            e["families"].append(f)
    return ref


def classify(new_table, loader_changed, weights_changed, failed):
    if failed:
        return "FAILS VERIFIER"
    if new_table:
        return "new word table"
    return {(False, False): "identical copy", (True, False): "loader-only",
            (False, True): "weights-only", (True, True): "loader+weights"}[(loader_changed, weights_changed)]


def check_family(fam, ref, compat):
    """Flags and class of one family against a reference state.

    fam: {family, status, error, npz_sha256, tree_edges_sha256, parts: [{k, sha256, layout_sha256,
          [prev_words_sha256, words_sha256]}], block_words_sha256 (list or None), npz_words_equal (bool or None),
          J, rows, metrics/resources/score (optional)}.
    compat: {(npz_sha256, table_id): {'pass': bool, 'lambda_eff': float, 'reason': str}}.
    Returns a dict with class, status, flags [(name, severity, detail)], claimed table, per-slot reuse.
    """
    flags = []

    def flag(name, severity, detail):
        flags.append((name, severity, detail))

    failed = fam.get("status") == "FAIL"
    if failed:
        flag("VERIFY_FAIL", "BREAK", fam.get("error") or "rejected")
    parts = sorted(fam["parts"], key=lambda p: p["k"])
    slots = len(parts)

    # what each slot loads: from the family's own probe, else from the ledger part (static mode)
    blocks = list(fam.get("block_words_sha256") or [None] * slots)
    known_chain = {}
    for p in parts:
        e = ref["parts"].get((p["sha256"], p["k"]))
        if e is not None:
            known_chain[p["k"]] = e
            if blocks[p["k"]] is None and len(e["chains"]) == 1:
                blocks[p["k"]] = next(iter(e["chains"]))[1]
    unknown = [p["k"] for p in parts if blocks[p["k"]] is None]
    if unknown:
        flag("UNKNOWN_PART", "NOTE", f"slots {unknown}: angle file not in the ledger and not simulated")

    # chain: a known loader must start from the block the previous slot leaves
    for k, e in sorted(known_chain.items()):
        prev = L.ZERO_BLOCK if k == 0 else blocks[k - 1]
        if prev is None:
            continue
        if not any(c[0] == prev for c in e["chains"]):
            starts = sorted({c[0][:12] for c in e["chains"]})
            flag("CHAIN_BREAK", "BREAK", f"slot {k}: angle_k_{k} ({parts[k]['sha256'][:12]}, from "
                 f"{e['families'][0]}) starts from block {starts} but slot {k - 1} leaves {prev[:12]}")

    # table match
    tables = ref["tables"]
    same = [t for t in tables.values() if len(t["blocks"]) == slots and t["rows"] == fam.get("rows", t["rows"])]
    per_slot = {k: [t["table_id"] for t in same if blocks[k] is not None and t["blocks"][k] == blocks[k]]
                for k in range(slots)}
    exact = [t for t in same if all(blocks[k] is not None and t["blocks"][k] == blocks[k] for k in range(slots))]
    claimed = exact[0] if exact else (max(same, key=lambda t: (sum(t["table_id"] in per_slot[k] for k in per_slot),
                                                               -int(t["table_id"][1:])))
                                      if same else None)
    if claimed is not None and not any(claimed["table_id"] in v for v in per_slot.values()):
        claimed = None
    new_table = not exact and not unknown
    if new_table:
        changed = [k for k in range(slots) if claimed is None or claimed["table_id"] not in per_slot[k]]
        flag("NEW_WORD_TABLE", "NOTE", f"slots {changed} load blocks not in "
             f"{claimed['table_id'] if claimed else 'any earlier table'}"
             + (f" (closest table {claimed['table_id']}, {slots - len(changed)}/{slots} slots shared)"
                if claimed else ""))
        hit = {tid for v in per_slot.values() for tid in v}
        if len(hit) >= 2:
            cover = ", ".join(f"k{k}:{'/'.join(v) if v else 'new'}" for k, v in per_slot.items())
            flag("MIXED_TABLES", "NOTE", f"blocks from {sorted(hit)}: {cover}")

    # layout
    lays = {p["layout_sha256"] for p in parts}
    if len(lays) > 1:
        flag("LAYOUT_MISMATCH", "BREAK", f"{len(lays)} register layouts across slots")
    elif claimed is not None and not lays <= claimed["layouts"]:
        flag("LAYOUT_MISMATCH", "NOTE", f"register numbering {next(iter(lays))[:12]} not seen with "
             f"{claimed['table_id']} (seen: {sorted(x[:12] for x in claimed['layouts'])})")

    # tree, weights
    if claimed is not None and fam["tree_edges_sha256"] not in claimed["trees"]:
        flag("TREE_MISMATCH", "NOTE", f"tree {fam['tree_edges_sha256'][:12]} vs {claimed['table_id']} trees "
             f"{sorted(x[:12] for x in claimed['trees'])}")
    lam_on_claimed = None
    if claimed is not None:
        c = compat.get((fam["npz_sha256"], claimed["table_id"]))
        if c is not None:
            lam_on_claimed = c.get("lambda_eff")
            if not c["pass"]:
                flag("WEIGHTS_FAIL_ON_TABLE", "NOTE" if new_table else "BREAK",
                     f"npz on {claimed['table_id']}: {c.get('reason') or 'fails'}")
    if fam.get("npz_words_equal") is False:
        flag("NPZ_WORDS_DIFFER", "NOTE", "stored angle_words differ from the loaded words")

    loader_new = [p["k"] for p in parts if (p["sha256"], p["k"]) not in ref["parts"]]
    weights_changed = fam["npz_sha256"] not in ref["npz"]
    cls = classify(new_table, bool(loader_new), weights_changed, failed)
    reused_from = {}
    for p in parts:
        e = ref["parts"].get((p["sha256"], p["k"]))
        reused_from[p["k"]] = e["families"][0] if e else None
    status = "BREAK" if any(s == "BREAK" for _, s, _ in flags) else ("NOTE" if flags else "OK")
    return {"family": fam["family"], "class": cls, "status": status, "flags": flags,
            "table": claimed["table_id"] if claimed else None, "new_loader_slots": loader_new,
            "weights_changed": weights_changed, "reused_from": reused_from,
            "lambda_on_table": lam_on_claimed,
            "ref_best_lambda": claimed["best_lambda"] if claimed else None,
            "ref_best_T": claimed["best_T"] if claimed else None,
            "ref_best_cost": claimed["best_cost"] if claimed else None,
            "ref_best_family": claimed["best_family"] if claimed else None}


def verdict_line(res, fam, credit):
    lam = fam.get("metrics", {}).get("lambda_eff")
    T = fam.get("resources", {}).get("toffoli")
    cost = fam.get("score")
    on = f"on {res['table']}" if res["table"] else "on -"
    parts = [f"{fam['family']}", f"{credit.get('author', '')[:18]:<18}", f"{credit.get('release', ''):<16}",
             f"{res['class']:<16} {on:<6}"]
    if lam is not None:
        dl = f" ({lam - res['ref_best_lambda']:+.6f})" if res["ref_best_lambda"] is not None else ""
        dT = f" ({T - res['ref_best_T']:+.0f})" if res["ref_best_T"] is not None else ""
        dc = (f" ({100 * (cost / res['ref_best_cost'] - 1):+.3f}% vs {res['ref_best_family']})"
              if res["ref_best_cost"] else "")
        parts.append(f"lambda {lam:.6f}{dl}  T {T:.0f}{dT}  cost {cost:.6e}{dc}")
    names = [n for n, _, _ in res["flags"]]
    shown = [n if names.count(n) == 1 else f"{n} x{names.count(n)}" for n in dict.fromkeys(names)]
    parts.append(f"{res['status']:<5} flags: {', '.join(shown) or '-'}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def all_jobs(probes, manifest, only=None, since=None):
    """(family record, earlier families in release order, rule, same-time same-table families) per family, release
    order. Each family is compared with the families released strictly before it."""
    key = {f: L.release_key(manifest.get(f, {}).get("release_date"), "") for f in probes}
    order = sorted(probes, key=lambda g: (key[g], g))
    jobs = []
    for f in order:
        if (only is not None and f not in only) or (since and key[f][0] < since):
            continue
        earlier = {g: probes[g] for g in order if key[g] < key[f]}
        ties = sorted(g for g in probes if g != f and key[g] == key[f]
                      and probes[g].get("table_sha256") == probes[f].get("table_sha256"))
        jobs.append((probes[f], earlier, "families released before it", ties))
    return jobs


def check_all(probes, table_label, compat, manifest):
    """{family: check_family result} for every family against the ones released before it (used by the ledger build).
    table_label: {table_sha256: 'Txx'}; compat keyed by (npz_sha256, 'Txx')."""
    return {fam["family"]: check_family(fam, reference_from_probes(earlier, table_label), compat)
            for fam, earlier, _, _ in all_jobs(probes, manifest)}


def load_ledger(probe_dir, parts_dir):
    """Probes plus {table_sha256: label} and compat keyed by (npz_sha256, label). Reads both the current layout
    (table_label) and the older evidence layout (table_id)."""
    probes = L.load_probes(probe_dir)
    tables = L.read_csv(Path(parts_dir) / "word_tables.csv")
    lab = "table_label" if tables and "table_label" in tables[0] else "table_id"
    table_label = {r["table_sha256"]: r[lab] for r in tables}
    compat = {}
    for r in L.read_csv(Path(parts_dir) / "compatibility.csv"):
        compat[(r["npz_sha256"], r[lab])] = {
            "pass": r["pass"] == "True", "lambda_eff": float(r["lambda_eff"]) if r["lambda_eff"] else None,
            "reason": r["reason"]}
    return probes, table_label, compat


def static_family(path, simulate, verifiers):
    """Probe-like record for a directory not in the ledger (kickmix needed)."""
    dv = L.import_verifier(verifiers)
    if simulate:
        rec, _ = L.probe_family(dv, path)
        return rec
    import numpy as np
    path = Path(path)
    npz = path / "submission.npz"
    rec = {"family": path.name, "dir": str(path), "npz_sha256": L.sha256_file(npz), "status": None,
           "parts": L.static_parts(path), "block_words_sha256": None}
    with np.load(npz, allow_pickle=False) as z:
        tree = np.asarray(z["tree_edges"], dtype=np.int64)
        rec["tree_edges_sha256"] = L.sha256_bytes(tree.tobytes() + str(tree.shape).encode())
        rec["rows"] = L.N_ORBITALS + int(np.shape(z["W"])[0]) * int(np.shape(z["W"])[1])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chemistry family consistency check against the parts ledger.")
    ap.add_argument("family_dir", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--families", nargs="*", help="--all: report only these families (sha12)")
    ap.add_argument("--ledger", help="an evidence dir holding probe/ and parts/ (e.g. the --out of an earlier ledger run)")
    ap.add_argument("--probe-dir", default=str(L.DEFAULT_PROBES))
    ap.add_argument("--parts-dir", default=str(L.STAGE_PARTS))
    ap.add_argument("--manifest", default=str(L.DEFAULT_MANIFEST))
    ap.add_argument("--verifiers", default=str(L.DEFAULT_VERIFIERS))
    ap.add_argument("--since", help="--all: report only families released on/after this date (all are checked)")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--md")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    if not args.all and not args.family_dir:
        ap.error("give FAMILY_DIR or --all")
    if args.ledger:
        args.probe_dir, args.parts_dir = L.resolve(args.ledger) / "probe", L.resolve(args.ledger) / "parts"
    probes, table_label, compat = load_ledger(L.resolve(args.probe_dir), L.resolve(args.parts_dir))
    manifest = L.load_manifest(L.resolve(args.manifest))
    key = {f: L.release_key(manifest.get(f, {}).get("release_date"), "") for f in probes}

    if args.all:
        jobs = all_jobs(probes, manifest, set(args.families) if args.families is not None else None, args.since)
    else:
        path = L.resolve(args.family_dir)
        fam = probes.get(path.name) if path.name in probes and not args.simulate else None
        if fam is None:
            fam = static_family(path, args.simulate, L.resolve(args.verifiers))
        jobs = [(fam, {g: probes[g] for g in sorted(probes, key=lambda g: (key[g], g)) if g != path.name},
                 "the whole ledger", [])]

    lines, details, out = [], [], []
    for fam, earlier, rule, ties in jobs:
        ref = reference_from_probes(earlier, table_label)
        res = check_family(fam, ref, compat)
        cr = L.credit(manifest, fam["family"])
        line = verdict_line(res, fam, cr)
        lines.append(line)
        out.append({**res, "author": cr["author"], "release": cr["release"], "reference": rule, "verdict": line,
                    "lambda_eff": fam.get("metrics", {}).get("lambda_eff"),
                    "toffoli": fam.get("resources", {}).get("toffoli"), "cost": fam.get("score")})
        if res["flags"] or res["class"] not in ("identical copy",):
            d = [f"### {fam['family']} ({cr['author']}, {cr['release']}): {res['class']} on "
                 f"{res['table'] or '-'}, {res['status']}"]
            if ties:
                d.append(f"- same release time and same word table (not compared): {', '.join(ties)}")
            reused = {}
            for k, src in res["reused_from"].items():
                reused.setdefault(src, []).append(k)
            d.append("- slots: " + "; ".join(f"{'new' if s is None else 'reused from ' + s} {v}"
                                              for s, v in reused.items()))
            d.append(f"- weights: {'new npz' if res['weights_changed'] else 'npz seen before'}"
                     + (f"; lambda_eff of this npz on {res['table']}: {res['lambda_on_table']:.6f}"
                        if res["lambda_on_table"] is not None else ""))
            for n, s, msg in res["flags"]:
                d.append(f"- **{n}** ({s}): {msg}")
            details.append("\n".join(d))

    md = ["# Chemistry family check", "",
          f"Probes: `{L.resolve(args.probe_dir)}`; parts: `{L.resolve(args.parts_dir)}`; reference for each family: "
          f"{jobs[0][2] if jobs else '-'}. Status: OK = no flags; NOTE = a change that limits how parts can be "
          "mixed; BREAK = the family (or a part in it) cannot work as shipped.", "",
          "```", *lines, "```", "", "## Details", "", *("\n\n".join(details).split("\n")), "",
          "## Flags", "", *(f"- **{n}** ({s}): {h}" for n, (s, h) in FLAG_HELP.items())]
    text = "\n".join(md) + "\n"
    print(text)
    if args.md:
        Path(args.md).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
