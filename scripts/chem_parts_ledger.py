#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Chemistry parts ledger.

A public chemistry family is `submission.npz` (the factorization: weights + Givens tree) plus fifteen
`angle_k_*.kmb` loaders. The official verifier simulates the loaders in order: circuit k starts with the word
registers holding block k-1 of the words (zero for k = 0) and must leave block k (columns k*J .. k*J+J-1). So a
loader is a *transition* block k-1 -> block k; it can be swapped into another family only if both blocks (and the
public register layout, which every circuit must share with circuit 0) are the same there. This script splits every
family into parts and writes them into a parts directory (columns and their order: scripts/chem_parts_schema.py):

  angle_circuits.csv   one row per (angle file sha256, slot k): prev_words_sha256 -> words_sha256, layout, static and
                       measured Toffolis, credits, dataset_path of a copy in the dataset;
  word_tables.csv      one row per distinct ordered list of word blocks; table_key = sha12 (stable), table_label T00..
                       ranked by the cheapest family (changes between versions); angles_sha256 hashes the whole matrix;
  factorizations.csv   one row per distinct npz: R, B, C, rows, brot, tree hash, credits, dataset_path;
  compatibility.csv    npz x word table (same row count): official `evaluate_words` metrics and pass;
  families.csv         per family: table, npz, parts, official vs parts-only lambda/T/Q (gate G1), family-check verdict;
  events.csv           first appearance of every table, loader and npz in release order (history for charts);
  word_table_blocks.csv per word table and angle loader k: the block's data origin and its circuit history;
  words/<table_key>.npy the word table as int16 (values < 2**brot; hashes are defined on the int64 values);
  summary.json         counts and the G1 check.

Probes (per-family simulation results, local only, never published) live in an append-only cache directory, one
<sha12>.json + <sha12>.words.npy per family. All subcommands except relabel need a Python that can import kickmix (they import the official verifier):

  probe   simulate each family in --families that has no probe yet, with the official `evaluate_circuits`
          (hash-derived seed, 64 shots/address) and the simulator wrapped so the Toffoli counter is read after every
          circuit. --force re-probes (writes a new file; the old one is renamed *.superseded-<time>).
  build   read the probes of the families present in --families plus the manifest, write the parts directory. The
          previous compatibility.csv in the parts directory is reused by (npz_sha256, table_key), so only new pairs
          are evaluated (--full re-evaluates all).
  update  probe then build: the incremental mode, seconds per new family.
  relabel add the stable W-labels and the lineage columns (scripts/chem_word_tables.py) to an existing parts directory
          (--from-parts) and write the result to a NEW --parts-dir; no kickmix needed. build/update do the same step.

Every build appends any new word table to the label registry scripts/chem_word_table_registry.json (W01, W02, ... by
first public appearance, never renumbered). Files are written atomically (temporary file, then rename), so a hard-linked
copy of the dataset directory keeps the bytes it was made with.

Usage:
  KMX=$QHDL_KICKMIX_PYTHON
  $KMX -I scripts/chem_parts_ledger.py update --stage [--report RUN.json]   # parts into the dataset dir
  $KMX -I scripts/chem_parts_ledger.py build --parts-dir $QHDL_OUTPUTS/<new dir>/parts [--probe-dir DIR] [--full]
  $KMX -I scripts/chem_parts_ledger.py probe [--only SHA12 ...] [--force]
  python3 scripts/chem_parts_ledger.py relabel --from-parts $QHDL_DATASET/parts --parts-dir <new dir>/parts
Defaults: --families = the dataset's circuits/chemistry, --manifest = the dataset's manifest.csv, --probe-dir =
the probe cache under the repository's outputs directory. --stage sets --parts-dir to the dataset's parts/ (the dataset
directory is $QHDL_DATASET, see scripts/vqcc_config.py). Relative paths resolve against the
repository root. The FCIDUMP cache of scripts/local_verify_reference.py is used unless --no-fcidump-cache. The pure
helpers (hashing, ordering, cost model, events, compatibility reuse) are unit-tested in tests/test_chem_parts.py.
"""
import argparse
import csv
import datetime
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chem_parts_schema import PARTS_FILES, columns  # noqa: E402
import chem_word_tables as W  # noqa: E402
import vqcc_config as CFG  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VERIFIERS = CFG.VERIFIERS
STAGE = CFG.DATASET          # dataset layout: manifest.csv, circuits/<task>/, parts/
DEFAULT_FAMILIES = STAGE / "circuits" / "chemistry"
DEFAULT_MANIFEST = STAGE / "manifest.csv"
STAGE_PARTS = STAGE / "parts"
DEFAULT_PROBES = REPO / "outputs" / "chemistry-parts-probes"
DEFAULT_OUT = REPO / "outputs" / "chemistry-parts"   # an evidence run (probe/ + parts/)
REGISTRY = Path(__file__).resolve().parent / "chem_word_table_registry.json"  # stable W-labels, committed with the code
ZERO_BLOCK = "ZERO"          # prev_words_sha256 of slot 0: the word registers start at zero
QUBIT_CAP = 128              # per angle circuit (dfthcsos.verify.QUBIT_CAP)
N_ORBITALS = 76

# ---------------------------------------------------------------------------
# Pure helpers (no kickmix, no verifier)
# ---------------------------------------------------------------------------

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def block_sha(block):
    """sha256 of an int64 word block plus its shape string (the probe's words_sha, full 64 hex)."""
    blk = np.ascontiguousarray(np.asarray(block).astype(np.int64))
    return sha256_bytes(blk.tobytes() + str(blk.shape).encode())


def split_blocks(words, J, slots):
    """Word blocks per slot: columns k*J .. k*J+J-1, zero-padded to slots*J columns (the verifier drops the pad)."""
    words = np.asarray(words, dtype=np.int64)
    pad = slots * J - words.shape[1]
    if pad < 0:
        raise ValueError(f"{words.shape[1]} columns do not fit {slots} slots of J={J}")
    full = np.concatenate([words, np.zeros((words.shape[0], pad), dtype=np.int64)], axis=1) if pad else words
    return [np.ascontiguousarray(full[:, k * J:(k + 1) * J]) for k in range(slots)]


def chain(block_shas):
    """(prev, out) per slot: slot 0 starts from ZERO_BLOCK, slot k from block k-1."""
    return [(ZERO_BLOCK if k == 0 else block_shas[k - 1], s) for k, s in enumerate(block_shas)]


def table_sha(block_shas):
    return sha256_bytes(";".join(block_shas).encode())


def layout_sha(selection_ids, word_ids):
    """Signature of the public registers the verifier binds to circuit 0 (selection + word_j qubit ids)."""
    return sha256_bytes(json.dumps({"selection": list(map(int, selection_ids)),
                                    "words": [list(map(int, w)) for w in word_ids]}).encode())


def release_key(release_date, name=""):
    """Sort key for manifest release_date ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'); a date alone sorts as 00:00."""
    s = (release_date or "").strip()
    if not s:
        return ("9999-99-99 99:99", name)
    if len(s) == 10:
        s += " 00:00"
    return (s, name)


def predict_cost(lambda_eff, angle_toffoli, heuristic, J, brot, selection_bits, max_part_qubits):
    """Cost model of dfthcsos.verify.resource_costs / score_from_metrics from parts.

    heuristic: dict with 'toffoli', 'persistent_qubits', 'temporary_qubits' (verify.heuristic_costs).
    Returns dict toffoli, qubits, angles_qubits, cost (inf if an angle circuit exceeds the 128-qubit cap).
    """
    persistent = int(heuristic["persistent_qubits"]) + J * brot
    angle_temp = int(max_part_qubits) - int(selection_bits) - J * brot
    qubits = persistent + max(int(heuristic["temporary_qubits"]), angle_temp)
    toffoli = float(angle_toffoli) + float(heuristic["toffoli"])
    cost = float("inf") if max_part_qubits > QUBIT_CAP else float(lambda_eff) * toffoli * qubits
    return {"toffoli": toffoli, "qubits": qubits, "angles_qubits": int(max_part_qubits), "cost": cost}


def write_csv(path, rows, columns):
    # Atomic: write a temporary file and rename it over the target, so a hard link to the old file (e.g. a
    # hard-linked copy prepared for upload) keeps its bytes instead of being truncated in place
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for r in rows:
            extra = set(r) - set(columns)
            if extra:  # a field missing from the schema would vanish silently: refuse
                tmp.unlink()
                raise ValueError(f"{path.name}: fields not in the column list: {sorted(extra)}")
            w.writerow({c: r.get(c, "") for c in columns})
    tmp.replace(path)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_manifest(path):
    """Chemistry rows of the dataset manifest, keyed by sha12."""
    return {r["sha12"]: r for r in read_csv(path) if r["task"] in ("chemistry", "dfthcsos")}


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def angle_files(family_dir):
    return sorted(Path(family_dir).glob("angle_k_*.kmb"), key=lambda p: int(p.stem.split("_")[-1]))


# ---------------------------------------------------------------------------
# Verifier-backed helpers (import kickmix lazily)
# ---------------------------------------------------------------------------

def import_verifier(verifiers=DEFAULT_VERIFIERS, fcidump_cache=True):
    """Import dfthcsos.verify (and install the FCIDUMP cache of local_verify_reference.py)."""
    if str(verifiers) not in sys.path:
        sys.path.insert(0, str(verifiers))
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from dfthcsos import verify as dv
    if fcidump_cache and not getattr(dv.load_reference, "official", None):
        import local_verify_reference as lvr
        lvr.install_fcidump_cache(lvr.DEFAULT_FCIDUMP_CACHE, [])
    return dv


def circuit_layout(circuit):
    """(selection ids, [word_j ids], layout_sha) of a kickmix circuit, or (None, None, None) if registers missing."""
    from dfthcsos.scoreinfra import circuits as C
    regs = circuit.register_data
    if C.SELECTION_REGISTER not in regs:
        return None, None, None
    sel = [q.id for q in regs[C.SELECTION_REGISTER]]
    idx = sorted(int(m.group(1)) for n in regs if (m := C._WORD_REGISTER_PATTERN.fullmatch(n)))
    words = [[q.id for q in regs[C.word_register(j)]] for j in idx]
    return sel, words, layout_sha(sel, words)


def static_parts(family_dir):
    """Per angle file: sha256, static Toffoli-like count, num_qubits, layout. Needs kickmix (no simulation)."""
    from dfthcsos.scoreinfra import circuits as C
    out = []
    for k, p in enumerate(angle_files(family_dir)):
        c = C.read_circuit(p)
        sel, words, lsha = circuit_layout(c)
        out.append({"k": k, "file": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size,
                    "static_toffoli": int(c.operation_statistics().num_toffoli_like),
                    "num_qubits": int(c.num_qubits), "selection_bits": len(sel) if sel else None,
                    "J": len(words) if words else None, "brot": len(words[0]) if words else None,
                    "layout_sha256": lsha})
    return out


def _recording_km(km, marks):
    """A stand-in for the kickmix module whose Simulator records the cumulative Toffoli count after each do()."""
    class RecordingSimulator:
        def __init__(self, *a, **kw):
            self._sim = km.Simulator(*a, **kw)

        def do(self, circuit):
            self._sim.do(circuit)
            marks.append(int(self._sim.operation_statistics().num_toffoli_like))

        def __getattr__(self, name):
            return getattr(self._sim, name)

    ns = types.SimpleNamespace(**{n: getattr(km, n) for n in dir(km) if not n.startswith("__")})
    ns.Simulator = RecordingSimulator
    return ns


def probe_family(dv, family_dir):
    """Official evaluate_circuits on one family plus per-slot Toffoli attribution. Returns (record, words or None)."""
    from dfthcsos.scoreinfra import circuits as C
    family_dir = Path(family_dir)
    npz = family_dir / "submission.npz"
    rec = {"family": family_dir.name, "dir": str(family_dir), "npz_sha256": sha256_file(npz),
           "parts": static_parts(family_dir)}
    with np.load(npz, allow_pickle=False) as z:
        tree = np.asarray(z["tree_edges"], dtype=np.int64)
        rec["tree_edges_sha256"] = sha256_bytes(tree.tobytes() + str(tree.shape).encode())
        rec["npz_brot"] = int(z["brot"]) if "brot" in z.files else None
        npz_words = np.asarray(z["angle_words"], dtype=np.int64) if "angle_words" in z.files else None
        rec["npz_RBC"] = list(np.shape(z["W"]))
    t = time.time()
    marks = []
    real_km = C.km
    try:
        inputs = dv.load_inputs(str(npz))
        edges = dv.load_tree_edges(None, family_dir)
        C.km = _recording_km(real_km, marks)
        try:
            r = dv.evaluate_circuits(inputs, str(family_dir), edges)
        finally:
            C.km = real_km
    except Exception as ex:  # noqa: BLE001  (a failing family is evidence too)
        rec.update(status="FAIL", error=f"{type(ex).__name__}: {ex}"[:600], seconds=round(time.time() - t, 1))
        return rec, None
    words = np.asarray(r["angle_words"], dtype=np.int64)
    J, slots = int(r["batch_size"]), len(rec["parts"])
    shots = inputs.num_vectors * 64
    deltas = np.diff([0] + marks)
    if len(deltas) != 2 * slots or marks[-1] / shots != r["resources"]["angles_toffoli"]:
        raise RuntimeError(f"{family_dir.name}: Toffoli attribution does not add up "
                           f"({len(deltas)} marks, {marks[-1] / shots} vs {r['resources']['angles_toffoli']})")
    fwd, rev = deltas[:slots], deltas[slots:][::-1]
    shas = [block_sha(b) for b in split_blocks(words, J, slots)]
    for k, p in enumerate(rec["parts"]):
        p.update(prev_words_sha256=chain(shas)[k][0], words_sha256=shas[k],
                 toffoli_forward=float(fwd[k] / shots), toffoli_reverse=float(rev[k] / shots),
                 toffoli_per_shot=float((fwd[k] + rev[k]) / shots))
    reason = dv.rejection_reason(r["metrics"], r["resources"])
    rec.update(status="PASS" if reason is None else "FAIL", error=reason, J=J, brot=int(r["brot"]),
               rows=int(words.shape[0]), slots=slots, words_shape=list(words.shape), block_words_sha256=shas,
               table_sha256=table_sha(shas), npz_words_equal=bool(npz_words is not None and npz_words.shape ==
                                                                   words.shape and np.array_equal(npz_words, words)),
               npz_words_sha256=block_sha(npz_words) if npz_words is not None else None,
               metrics={k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v))
                        for k, v in r["metrics"].items()},
               resources={k: (v if isinstance(v, (bool, str)) else float(v)) for k, v in r["resources"].items()},
               score=float(r["score"]), submission_sha256=r["submission_sha256"],
               seconds=round(time.time() - t, 1))
    return rec, words



def run_probe(args):
    """Probe every family in --families without a probe (or all with --force). Returns the list probed."""
    dv = import_verifier(resolve(args.verifiers), not args.no_fcidump_cache)
    out = resolve(args.probe_dir)
    out.mkdir(parents=True, exist_ok=True)
    fams = sorted(p for p in resolve(args.families).iterdir() if (p / "submission.npz").is_file())
    if args.only:
        fams = [p for p in fams if p.name in set(args.only)]
    done = []
    for p in fams:
        jf = out / f"{p.name}.json"
        if jf.exists():
            if not args.force:
                continue
            stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            for old in (jf, out / f"{p.name}.words.npy"):  # keep the earlier evidence
                if old.exists():
                    old.rename(old.with_name(old.name + f".superseded-{stamp}"))
        rec, words = probe_family(dv, p)
        if words is not None:
            np.save(out / f"{p.name}.words.npy", words)
        jf.write_text(json.dumps(rec, indent=1))
        done.append(p.name)
        print(p.name, rec["status"], rec.get("metrics", {}).get("lambda_eff"),
              rec.get("resources", {}).get("toffoli"), rec["seconds"], "s", flush=True)
    return done


# ---------------------------------------------------------------------------
# Build the parts tables (pure parts first)
# ---------------------------------------------------------------------------

def load_probes(probe_dir, only=None):
    """{family: probe record} from <probe_dir>/*.json (or <probe_dir>/probe/*.json for an old evidence dir)."""
    probe_dir = Path(probe_dir)
    if not list(probe_dir.glob("*.json")) and (probe_dir / "probe").is_dir():
        probe_dir = probe_dir / "probe"
    probes = {}
    for jf in sorted(probe_dir.glob("*.json")):
        if only is None or jf.stem in only:
            probes[jf.stem] = json.loads(jf.read_text())
    return probes


def credit(manifest, fam):
    m = manifest.get(fam, {})
    return {"author": m.get("author", ""), "notebook": m.get("source_notebook_url", ""),
            "notebook_version": m.get("source_notebook_version", ""), "release": m.get("release_date", ""),
            "first_seen": m.get("first_seen", "")}


def table_key(table_sha256):
    return table_sha256[:12]


def angle_path(family, k):
    """dataset_path of a family's slot-k loader."""
    return f"circuits/chemistry/{family}/angle_k_{k}.kmb"


def load_prev_compat(parts_dir):
    """Earlier compatibility rows keyed by (npz_sha256, table_key); accepts the 2026-10-29 evidence layout (table_id)."""
    parts_dir = Path(parts_dir)
    f = parts_dir / "compatibility.csv"
    if not f.is_file():
        return {}
    rows = read_csv(f)
    label_to_key = {}
    if rows and "table_key" not in rows[0] and (parts_dir / "word_tables.csv").is_file():
        label_to_key = {r["table_id"]: table_key(r["table_sha256"]) for r in read_csv(parts_dir / "word_tables.csv")}
    out = {}
    for r in rows:
        key = r.get("table_key") or label_to_key.get(r.get("table_id"))
        if key:
            out[(r["npz_sha256"], key)] = r
    return out


def compat_plan(npz_rows, tables, prev):
    """Which (npz, table) pairs need evaluate_words: same row count and not in `prev`. Returns (reuse, todo) lists of
    (npz_sha256, table_key)."""
    reuse, todo = [], []
    for fz in npz_rows:
        for t in tables:
            if int(t["rows"]) != int(fz["rows"]):
                continue
            key = (fz["npz_sha256"], t["table_key"])
            (reuse if key in prev else todo).append(key)
    return reuse, todo


def derive_events(order, probes, tkey, manifest, part_T):
    """First appearance of each table, loader and npz, walking families in release order (PASS families only)."""
    seen_t, seen_p, seen_n, events = set(), set(), set(), []
    for f in order:
        r = probes[f]
        if r.get("status") != "PASS":
            continue
        c = credit(manifest, f)
        base = {"release": c["release"], "family": f, "author": c["author"], "table_key": tkey[r["table_sha256"]],
                "lambda_eff": r["metrics"]["lambda_eff"], "cost_on_table": r["score"]}
        if r["table_sha256"] not in seen_t:
            seen_t.add(r["table_sha256"])
            events.append({**base, "kind": "table", "slot": "", "id": tkey[r["table_sha256"]],
                           "toffoli": r["resources"]["toffoli"]})
        for p in r["parts"]:
            key = (p["sha256"], p["k"])
            if key not in seen_p:
                seen_p.add(key)
                events.append({**base, "kind": "loader", "slot": p["k"], "id": f"k{p['k']:02d}-{p['sha256'][:12]}",
                               "toffoli": part_T[key]})
        if r["npz_sha256"] not in seen_n:
            seen_n.add(r["npz_sha256"])
            events.append({**base, "kind": "weights", "slot": "", "id": r["npz_sha256"][:12], "toffoli": ""})
    return events


def build_ledger(probes, manifest, dv, prev_compat=None, log=print):
    """All parts tables from probe records. dv = dfthcsos.verify (evaluate_words, heuristic_costs). Returns
    ({file name: rows}, {table_key: int64 words}, summary)."""
    import chem_family_check as F
    prev_compat = prev_compat or {}
    ok = {f: r for f, r in probes.items() if r["status"] == "PASS"}
    order = sorted(probes, key=lambda f: release_key(manifest.get(f, {}).get("release_date"), f))
    rank = {f: i for i, f in enumerate(order)}

    by_table = {}
    for f, r in ok.items():
        by_table.setdefault(r["table_sha256"], []).append(f)
    tables = sorted(by_table, key=lambda t: (min(ok[f]["score"] for f in by_table[t]), t))
    tkey = {t: table_key(t) for t in tables}
    label = {t: f"T{i:02d}" for i, t in enumerate(tables)}
    words_by_key, table_rows = {}, []
    for i, t in enumerate(tables):
        fams = sorted(by_table[t], key=rank.get)
        r0 = ok[fams[0]]
        words = np.load(Path(r0["_probe_dir"]) / f"{fams[0]}.words.npy")
        words_by_key[tkey[t]] = words
        c = credit(manifest, fams[0])
        best = min(fams, key=lambda f: (ok[f]["score"], rank[f]))
        table_rows.append({
            "table_key": tkey[t], "table_label": label[t], "rank": i, "table_sha256": t,
            "angles_sha256": block_sha(words), "J": r0["J"], "brot": r0["brot"], "rows": r0["rows"],
            "slots": r0["slots"], "block_words_sha256": ";".join(r0["block_words_sha256"]),
            "words_npy": f"parts/words/{tkey[t]}.npy", "n_families": len(fams), "families": ";".join(fams),
            "tree_edges_sha256": ";".join(sorted({ok[f]["tree_edges_sha256"] for f in fams})),
            "layout_sha256": ";".join(sorted({p["layout_sha256"] for f in fams for p in ok[f]["parts"]})),
            "first_family": fams[0], "first_author": c["author"], "first_release": c["release"],
            "best_family": best, "best_cost": ok[best]["score"]})

    parts = {}
    for f in sorted(ok, key=rank.get):
        for p in ok[f]["parts"]:
            e = parts.setdefault((p["sha256"], p["k"]), {"p": p, "fams": [], "T": [], "chains": set(), "tables": set()})
            e["fams"].append(f)
            e["T"].append(p["toffoli_per_shot"])
            e["chains"].add((p["prev_words_sha256"], p["words_sha256"]))
            e["tables"].add(tkey[ok[f]["table_sha256"]])
    part_rows = []
    for (fsha, k), e in sorted(parts.items(), key=lambda kv: (kv[0][1], rank[kv[1]["fams"][0]], kv[0][0])):
        p, c = e["p"], credit(manifest, e["fams"][0])
        part_rows.append({
            "part_id": f"k{k:02d}-{fsha[:12]}", "file_sha256": fsha, "slot_k": k,
            "dataset_path": angle_path(e["fams"][0], k), "bytes": p["bytes"],
            "prev_words_sha256": p["prev_words_sha256"], "words_sha256": p["words_sha256"],
            "chain_variants": len(e["chains"]), "word_tables": ";".join(sorted(e["tables"])), "J": p["J"],
            "brot": p["brot"], "selection_bits": p["selection_bits"], "layout_sha256": p["layout_sha256"],
            "num_qubits": p["num_qubits"], "static_toffoli": p["static_toffoli"],
            "toffoli_per_shot": e["T"][0], "toffoli_per_shot_min": min(e["T"]), "toffoli_per_shot_max": max(e["T"]),
            "n_families": len(e["fams"]), "families": ";".join(e["fams"]), "first_family": e["fams"][0],
            "first_author": c["author"], "first_notebook": c["notebook"],
            "first_notebook_version": c["notebook_version"], "first_release": c["release"]})
    part_T = {(r["file_sha256"], int(r["slot_k"])): float(r["toffoli_per_shot"]) for r in part_rows}

    npzs = {}
    for f in sorted(probes, key=rank.get):
        npzs.setdefault(probes[f]["npz_sha256"], []).append(f)
    fz_rows = []
    for nsha, fams in sorted(npzs.items(), key=lambda kv: rank[kv[1][0]]):
        r0, c = probes[fams[0]], credit(manifest, fams[0])
        R, B, C = r0["npz_RBC"]
        stored = next((tkey[ok[f]["table_sha256"]] for f in fams if f in ok and probes[f].get("npz_words_equal")),
                      "none")
        fz_rows.append({"npz_sha256": nsha, "dataset_path": f"factorizations/{fams[0]}.npz", "R": R, "B": B, "C": C,
                        "rows": N_ORBITALS + R * B, "brot": r0["npz_brot"], "keep_bits": dv.KEEP_BITS,
                        "tree_edges_sha256": r0["tree_edges_sha256"], "stored_words_table": stored,
                        "n_families": len(fams), "families": ";".join(fams), "first_family": fams[0],
                        "first_author": c["author"], "first_notebook": c["notebook"], "first_release": c["release"]})

    reuse, todo = compat_plan(fz_rows, table_rows, prev_compat)
    fz_by = {r["npz_sha256"]: r for r in fz_rows}
    t_by = {r["table_key"]: r for r in table_rows}
    computed, t0, inputs_cache = {}, time.time(), {}
    for nsha, key in todo:
        fz, tr = fz_by[nsha], t_by[key]
        npz = Path(probes[fz["first_family"]]["dir"]) / "submission.npz"
        if nsha not in inputs_cache:
            inputs_cache = {nsha: (dv.load_inputs(str(npz)), dv.load_tree_edges(str(npz), None))}
        inputs, edges = inputs_cache[nsha]
        try:
            m = dv.evaluate_words(inputs, words_by_key[key], edges, int(tr["brot"]))
            passed, reason = dv.passes_accuracy(m), "; ".join(dv._floor_rejection_reasons(m))
            if not passed and not reason:
                reason = (f"error_1 {1e3 * m['error_1']:.4f} mHa, error_2 {1e3 * m['error_2']:.4f} mHa, "
                          f"gap_in_domain {bool(m['gap_in_domain'])}")
        except Exception as ex:  # noqa: BLE001
            m, passed, reason = {}, False, f"{type(ex).__name__}: {ex}"[:300]
        computed[(nsha, key)] = {"error_1_mHa": 1e3 * m["error_1"] if m else "",
                                 "error_2_mHa": 1e3 * m["error_2"] if m else "", "Lambda": m.get("Lambda", ""),
                                 "E_gap": m.get("E_gap", ""), "lambda_eff": m.get("lambda_eff", ""),
                                 "pass": passed, "reason": reason}
    comp_rows = []
    for nsha, key in sorted(reuse + todo, key=lambda x: (rank[fz_by[x[0]]["first_family"]], t_by[x[1]]["rank"])):
        src = computed.get((nsha, key)) or prev_compat[(nsha, key)]
        comp_rows.append({"npz_sha256": nsha, "npz_first_family": fz_by[nsha]["first_family"], "table_key": key,
                          "table_label": t_by[key]["table_label"],
                          "tree_match": fz_by[nsha]["tree_edges_sha256"] in t_by[key]["tree_edges_sha256"].split(";"),
                          **{c: src[c] for c in ("error_1_mHa", "error_2_mHa", "Lambda", "E_gap", "lambda_eff",
                                                 "pass", "reason")}})
    log(f"compatibility: {len(reuse)} pairs reused, {len(todo)} evaluated in {time.time() - t0:.1f} s")
    lam = {(r["npz_sha256"], r["table_key"]): r for r in comp_rows}

    # family check, each family against the families released before it (chem_family_check.py --all)
    compat_lbl = {(r["npz_sha256"], r["table_label"]): {"pass": str(r["pass"]) == "True",
                                                        "lambda_eff": float(r["lambda_eff"]) if r["lambda_eff"] != ""
                                                        else None, "reason": r["reason"]} for r in comp_rows}
    checks = F.check_all(probes, label, compat_lbl, manifest)

    fam_rows = []
    for f in order:
        r, c = probes[f], credit(manifest, f)
        chk = checks[f]
        row = {"family": f, "author": c["author"], "release": c["release"], "first_seen": c["first_seen"],
               "status": r["status"], "error": r.get("error") or "", "npz_sha256": r["npz_sha256"],
               "check_class": chk["class"], "check_status": chk["status"],
               "check_flags": ";".join(dict.fromkeys(n for n, _, _ in chk["flags"]))}
        if f in ok:
            key = tkey[r["table_sha256"]]
            heur = dv.heuristic_costs(N_ORBITALS, *r["npz_RBC"], r["brot"])
            T_parts = sum(part_T[(p["sha256"], p["k"])] for p in r["parts"])
            cmp_row = lam[(r["npz_sha256"], key)]
            pred = predict_cost(float(cmp_row["lambda_eff"]), T_parts, heur, r["J"], r["brot"],
                                r["parts"][0]["selection_bits"], max(p["num_qubits"] for p in r["parts"]))
            row.update(table_key=key, table_label=label[r["table_sha256"]], J=r["J"], brot=r["brot"],
                       parts=";".join(f"k{p['k']:02d}-{p['sha256'][:12]}" for p in r["parts"]),
                       lambda_official=r["metrics"]["lambda_eff"], lambda_parts=float(cmp_row["lambda_eff"]),
                       lambda_abs_dev=abs(float(cmp_row["lambda_eff"]) - r["metrics"]["lambda_eff"]),
                       T_official=r["resources"]["toffoli"], T_parts=pred["toffoli"],
                       T_dev=pred["toffoli"] - r["resources"]["toffoli"],
                       Q_official=int(r["resources"]["qubits"]), Q_parts=pred["qubits"],
                       cost_official=r["score"], cost_parts=pred["cost"], npz_words_equal=r["npz_words_equal"])
        fam_rows.append(row)
    events = derive_events(order, probes, tkey, manifest, part_T)
    g1 = [r for r in fam_rows if r["status"] == "PASS"]
    summary = {
        "families": len(probes), "families_pass": len(ok), "families_fail": sorted(set(probes) - set(ok)),
        "word_tables": len(table_rows), "angle_parts": len(part_rows), "factorizations": len(fz_rows),
        "compatibility_pairs": len(comp_rows), "compatibility_pass": sum(str(r["pass"]) == "True" for r in comp_rows),
        "compatibility_reused": len(reuse), "compatibility_evaluated": len(todo), "events": len(events),
        "G1_max_lambda_abs_dev": max((r["lambda_abs_dev"] for r in g1), default=None),
        "G1_max_abs_T_dev": max((abs(r["T_dev"]) for r in g1), default=None),
        "G1_Q_mismatch": [r["family"] for r in g1 if r["Q_parts"] != r["Q_official"]],
        "tables": [{k: tr[k] for k in ("table_key", "table_label", "J", "brot", "rows", "slots", "n_families",
                                       "best_family", "best_cost", "first_author", "first_release")}
                   for tr in table_rows],
    }
    files = {"parts/angle_circuits.csv": part_rows, "parts/word_tables.csv": table_rows,
             "parts/factorizations.csv": fz_rows, "parts/compatibility.csv": comp_rows,
             "parts/families.csv": fam_rows, "parts/events.csv": events}
    return files, words_by_key, summary


def add_forensics(files, manifest, registry, today=""):
    """Stable W-labels and lineage (scripts/chem_word_tables.py) added to the parts tables. files: {name: rows} as
    build_ledger returns or read back from CSV; manifest: chemistry rows by sha12 (R, B, C of each table's first family).
    Returns (files with the new columns and parts/word_table_blocks.csv, the registry with any new table appended)."""
    tables = files["parts/word_tables.csv"]
    shapes = {t["table_sha256"]: {c: manifest.get(t["first_family"], {}).get(c, "") for c in ("R", "B", "C")}
              for t in tables}
    registry = W.wt_assign(registry, tables, shapes, today)
    by_sha, by_key = W.wt_labels(registry)
    entry = {e["table_sha256"]: e for e in registry["tables"]}
    lin = W.wt_lineage(tables, by_sha)
    for t in tables:
        e, ln = entry[t["table_sha256"]], lin[t["table_sha256"]]
        t.update(stable_label=e["label"], label_description=e["description"], R=e["R"], B=e["B"], C=e["C"],
                 parent_label=ln["parent_label"], parent_shared_blocks=ln["parent_shared_blocks"],
                 changed_loaders=";".join(map(str, ln["changed_loaders"])), lineage_note=ln["lineage_note"])
    for name in ("parts/families.csv", "parts/compatibility.csv", "parts/events.csv"):
        for r in files[name]:
            r["stable_label"] = by_key.get(r.get("table_key") or "", "")
    for r in files["parts/angle_circuits.csv"]:
        r["word_table_labels"] = ";".join(by_key.get(k, "") for k in (r.get("word_tables") or "").split(";") if k)
    files["parts/word_table_blocks.csv"] = W.wt_block_rows(tables, by_sha, files["parts/families.csv"],
                                                           files["parts/angle_circuits.csv"])
    return files, registry


def save_registry(registry, path=REGISTRY):
    """Write the registry if it changed (atomic). Refuses to drop or renumber an entry already on disk."""
    path = Path(path)
    old = W.wt_load(path)
    new_by_sha = {e["table_sha256"]: e["label"] for e in registry["tables"]}
    for e in old.get("tables", []):
        if new_by_sha.get(e["table_sha256"]) != e["label"]:
            raise ValueError(f"registry: {e['label']} ({e['table_key']}) would be dropped or renumbered")
    if old == registry:
        return False
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(registry, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return True


def write_ledger(parts_dir, files, words_by_key, summary):
    """Write the tables (schema column order), words as int16 and summary.json into parts_dir."""
    parts_dir = Path(parts_dir)
    (parts_dir / "words").mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        write_csv(parts_dir / Path(name).name, rows, columns(name))
    for key, words in words_by_key.items():
        w = np.asarray(words, dtype=np.int64)
        if w.min() < 0 or w.max() >= 1 << 15:
            raise ValueError(f"table {key}: words do not fit int16")
        tmp = parts_dir / "words" / f".{key}.tmp.npy"
        np.save(tmp, w.astype(np.int16))
        tmp.replace(parts_dir / "words" / f"{key}.npy")
    tmp = parts_dir / ".summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1) + "\n")
    tmp.replace(parts_dir / "summary.json")


def run_build(args):
    dv = import_verifier(resolve(args.verifiers), not args.no_fcidump_cache)
    parts_dir = resolve(args.parts_dir)
    present = {p.name for p in resolve(args.families).iterdir() if (p / "submission.npz").is_file()}
    probe_dir = resolve(args.probe_dir)
    probes = load_probes(probe_dir, only=present)
    for r in probes.values():
        r["_probe_dir"] = str(probe_dir if list(probe_dir.glob("*.json")) else probe_dir / "probe")
    missing = sorted(present - set(probes))
    if missing:
        print(f"WARNING: {len(missing)} families have no probe (run `probe` or `update`): {missing}")
    manifest = load_manifest(resolve(args.manifest))
    prev = {} if args.full else (load_prev_compat(parts_dir) or
                                 (load_prev_compat(resolve(args.seed_compat)) if args.seed_compat else {}))
    files, words, summary = build_ledger(probes, manifest, dv, prev)
    files, registry = add_forensics(files, manifest, W.wt_load(REGISTRY), datetime.date.today().isoformat())
    if save_registry(registry):
        print(f"registry: {len(registry['tables'])} word tables in {REGISTRY}")
    summary["stable_labels"] = {t["table_key"]: t["stable_label"] for t in files["parts/word_tables.csv"]}
    write_ledger(parts_dir, files, words, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "tables"}, indent=1))
    return summary, missing


def run_relabel(args):
    """Read an existing parts directory, add the stable labels and lineage, write a NEW parts directory (CSV files
    rewritten, words/*.npy and the JSON files hard-linked from the source)."""
    import os
    src, dst = resolve(args.from_parts), resolve(args.parts_dir)
    if src.resolve() == dst.resolve():
        raise SystemExit("relabel writes a new parts directory: --parts-dir must differ from --from-parts")
    files = {}
    for name in PARTS_FILES:
        f = src / Path(name).name
        if name != "parts/word_table_blocks.csv":  # recomputed below; label columns already present are overwritten
            files[name] = read_csv(f)
    manifest = load_manifest(resolve(args.manifest))
    files, registry = add_forensics(files, manifest, W.wt_load(REGISTRY), datetime.date.today().isoformat())
    if save_registry(registry):
        print(f"registry: {len(registry['tables'])} word tables in {REGISTRY}")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "words").mkdir(exist_ok=True)
    for name, rows in files.items():
        write_csv(dst / Path(name).name, rows, columns(name))
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_file() and p.suffix != ".csv" and not (dst / rel).exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(p, dst / rel)
    print(json.dumps({t["table_key"]: [t["stable_label"], t["lineage_note"]] for t in files["parts/word_tables.csv"]},
                     indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chemistry parts ledger: probe families, build the parts tables.")
    ap.add_argument("command", choices=["probe", "build", "update", "relabel"])
    ap.add_argument("--from-parts", help="relabel: the existing parts directory to read")
    ap.add_argument("--families", default=str(DEFAULT_FAMILIES))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--probe-dir", default=str(DEFAULT_PROBES))
    ap.add_argument("--parts-dir", help="where to write the tables (build/update need it, or --stage)")
    ap.add_argument("--stage", dest="to_stage", action="store_true", help=f"same as --parts-dir {STAGE_PARTS}")
    ap.add_argument("--verifiers", default=str(DEFAULT_VERIFIERS))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true", help="probe: re-probe (earlier files kept as *.superseded-*)")
    ap.add_argument("--full", action="store_true", help="build: re-evaluate every compatibility pair")
    ap.add_argument("--seed-compat", help="build: parts dir to reuse compatibility rows from when --parts-dir has none")
    ap.add_argument("--report", help="JSON with this run's probed families, timings and summary")
    ap.add_argument("--no-fcidump-cache", action="store_true")
    args = ap.parse_args(argv)
    if args.to_stage:
        args.parts_dir = str(STAGE_PARTS)
    if args.command != "probe" and not args.parts_dir:
        ap.error("give --parts-dir or --stage")
    t0 = time.time()
    report = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "probed": []}
    if args.command in ("probe", "update"):
        report["probed"] = run_probe(args)
    if args.command == "relabel":
        if not args.from_parts:
            ap.error("relabel needs --from-parts")
        run_relabel(args)
        return 0
    if args.command in ("build", "update"):
        summary, missing = run_build(args)
        report.update(summary=summary, missing_probes=missing, parts_dir=str(resolve(args.parts_dir)))
    report["seconds"] = round(time.time() - t0, 1)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
