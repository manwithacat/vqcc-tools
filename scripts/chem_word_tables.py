# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
# Chemistry word tables: stable labels, lineage and frontier forensics.
#
# A word table is the 8-bit angle data the angle loaders k (ANGLE_k QROMs) write; it splits into one block of words per
# angle loader. The parts ledger's T00, T01, ... labels are ranks (cheapest family first) and change between versions;
# the W-labels here are stable: W01, W02, ... in order of first public appearance (first_release, ties by table_sha256),
# assigned once in scripts/chem_word_table_registry.json, never reused or renumbered.
#
# Pure functions of parts rows (dicts of strings, as csv.DictReader gives) and the registry; standard library only.
# scripts/chem_parts_ledger.py imports this module to write the label and lineage columns, and
# its text can be embedded verbatim in a notebook (so no __future__ import, no top-level side effects, and every helper
# name starts with wt_ to stay clear of the notebook's own names).
import json as _wt_json

WT_FORMAT = "W{:02d}"
WT_SCHEME = ("W-labels: one per distinct word table (table_sha256), numbered W01, W02, ... in order of first public "
             "appearance (first_release; a date alone counts as 00:00; ties by table_sha256). Assigned once, never "
             "reused or renumbered; new tables are appended.")


def wt_when(s):
    # Sort key of a release string: 'YYYY-MM-DD' counts as 00:00 (as release_key in chem_parts_ledger.py)
    s = (s or "").strip()
    if not s:
        return "9999-99-99 99:99"
    return s + " 00:00" if len(s) == 10 else s[:16]


def wt_blocks(t):
    # Block hashes of a word_tables.csv row, one per angle loader k
    return [b for b in (t.get("block_words_sha256") or "").split(";") if b]


def wt_shape(t):
    return tuple(str(t.get(c, "")) for c in ("rows", "J", "brot", "slots"))


def wt_num(label):
    return int(str(label)[1:])


def wt_describe(e):
    rbc = ", ".join(str(e.get(c, "?")) for c in ("R", "B", "C"))
    return (f"{e.get('first_author') or '?'}, first public {e.get('first_release') or '?'}; (R, B, C) = ({rbc}); "
            f"J = {e.get('J')}, brot = {e.get('brot')}, {e.get('rows')} rows, {e.get('slots')} angle loaders")


def wt_assign(registry, tables, shapes=None, today=""):
    # New registry: the old entries unchanged, then every table not yet in it, numbered on in order of first public
    # appearance. tables: word_tables.csv rows; shapes: {table_sha256: {'R','B','C'}} of each table's first family.
    reg = {"scheme": WT_SCHEME, "tables": [dict(e) for e in (registry or {}).get("tables", [])]}
    known = {e["table_sha256"] for e in reg["tables"]}
    n = max((wt_num(e["label"]) for e in reg["tables"]), default=0)
    new = {t["table_sha256"]: t for t in tables if t["table_sha256"] not in known}
    for sha in sorted(new, key=lambda s: (wt_when(new[s].get("first_release")), s)):
        t, n = new[sha], n + 1
        e = {"label": WT_FORMAT.format(n), "table_sha256": sha, "table_key": sha[:12],
             "first_family": t.get("first_family", ""), "first_author": t.get("first_author", ""),
             "first_release": t.get("first_release", ""),
             **{c: str((shapes or {}).get(sha, {}).get(c, "")) for c in ("R", "B", "C")},
             **{c: str(t.get(c, "")) for c in ("J", "brot", "rows", "slots")}, "assigned": today}
        e["description"] = wt_describe(e)
        reg["tables"].append(e)
    labels = [e["label"] for e in reg["tables"]]
    shas = [e["table_sha256"] for e in reg["tables"]]
    if len(set(labels)) != len(labels) or len(set(shas)) != len(shas):
        raise ValueError("word-table registry: a label or a table is listed twice")
    return reg


def wt_load(path):
    try:
        with open(path) as f:
            return _wt_json.load(f)
    except FileNotFoundError:
        return {"scheme": WT_SCHEME, "tables": []}


def wt_labels(registry):
    # {table_sha256: label} and {table_key: label}
    by_sha = {e["table_sha256"]: e["label"] for e in registry.get("tables", [])}
    return by_sha, {s[:12]: lab for s, lab in by_sha.items()}


def wt_lineage(tables, label_of):
    # {table_sha256: lineage} for word_tables.csv rows. Parent = the earlier table (lower W-number) sharing the most
    # block hashes at the same angle loader k; ties go to the most recent. changed = the angle loaders k whose block
    # differs from the parent's. A table sharing no block: 'new shape, no parent' when no earlier table has its shape.
    order = sorted(tables, key=lambda t: wt_num(label_of[t["table_sha256"]]))
    out = {}
    for i, t in enumerate(order):
        b, best = wt_blocks(t), None
        for p in order[:i]:
            shared = sum(1 for x, y in zip(b, wt_blocks(p)) if x == y)
            if shared and (best is None or shared >= best[0]):
                best = (shared, p)
        if best:
            pb = wt_blocks(best[1])
            changed = [k for k in range(len(b)) if k >= len(pb) or b[k] != pb[k]]
            plab = label_of[best[1]["table_sha256"]]
            note = (f"same blocks as {plab}" if not changed else
                    f"inherits {best[0]} of {len(b)} blocks from {plab}; new data for angle loaders "
                    + ", ".join(map(str, changed)))
            out[t["table_sha256"]] = {"parent_label": plab, "parent_shared_blocks": best[0], "changed_loaders": changed,
                                      "lineage_note": note}
        else:
            same = [p for p in order[:i] if wt_shape(p) == wt_shape(t)]
            note = ("first word table" if i == 0 else
                    f"same shape as {label_of[same[-1]['table_sha256']]}, no shared blocks" if same
                    else "new shape, no parent")
            out[t["table_sha256"]] = {"parent_label": "", "parent_shared_blocks": 0,
                                      "changed_loaders": list(range(len(b))), "lineage_note": note}
    return out


def wt_loader_history(families, circuits, key):
    # {k: [(release, part_id, toffoli, author, family)]} for word table `key`: each angle-loader circuit at the first time a
    # family on this table used it, in public order (families.csv and angle_circuits.csv rows). Circuits inherited from
    # an earlier table count from their first use on this one.
    T = {c["part_id"]: (float(c["toffoli_per_shot"]), c.get("first_author", "")) for c in circuits}
    fams = sorted((f for f in families if f.get("table_key") == key and f.get("status", "PASS") == "PASS"),
                  key=lambda f: (wt_when(f.get("release")), f["family"]))
    seen, hist = set(), {}
    for f in fams:
        for pid in (f.get("parts") or "").split(";"):
            if not pid or pid in seen or pid not in T:
                continue
            seen.add(pid)
            hist.setdefault(int(pid[1:3]), []).append((wt_when(f.get("release")), pid, T[pid][0], f.get("author", ""),
                                                       f["family"]))
    return hist


def wt_history_summary(h):
    # (circuits, cheaper compiles, first Toffolis, cheapest Toffolis, cheapest part_id) of one angle loader's history
    if not h:
        return 0, 0, None, None, ""
    best, cheaper, best_id = h[0][2], 0, h[0][1]
    for _, pid, t, _, _ in h[1:]:
        if t < best - 1e-9:
            best, best_id, cheaper = t, pid, cheaper + 1
    return len(h), cheaper, h[0][2], best, best_id


def wt_block_rows(tables, label_of, families, circuits):
    # parts/word_table_blocks.csv rows: one per (word table, angle loader k), with the block's data origin (the
    # W-label of the first table where this block appeared at this k) and the circuit history on this table
    order = sorted(tables, key=lambda t: wt_num(label_of[t["table_sha256"]]))
    lin = wt_lineage(tables, label_of)
    origin, rows = {}, []
    for t in order:
        lab, key = label_of[t["table_sha256"]], t["table_key"]
        hist = wt_loader_history(families, circuits, key)
        par = lin[t["table_sha256"]]
        for k, b in enumerate(wt_blocks(t)):
            src = origin.setdefault((k, b), lab)
            n, cheaper, first, best, best_id = wt_history_summary(hist.get(k, []))
            rows.append({"table_key": key, "stable_label": lab, "angle_loader_k": k, "block_words_sha256": b,
                         "data_origin": src, "new_data": src == lab,
                         "changed_vs_parent": bool(par["parent_label"]) and k in par["changed_loaders"],
                         "parent_label": par["parent_label"], "n_circuits": n, "n_cheaper_compiles": cheaper,
                         "first_toffoli": first if first is not None else "",
                         "cheapest_toffoli": best if best is not None else "", "cheapest_part_id": best_id})
    return rows


def wt_frontier(steps, fam_by_sha, table_by_key, label_by_key, released, cost):
    # One dict per chemistry frontier record (steps from frontier_of(), public order), classified against the record
    # before it: 'new word table' (new shape, or same shape with some blocks changed), 'cheaper angle-loader circuit'
    # (same table, other loaders), 'angle-loader circuits and weights', 'weights only' (same table and loaders), or
    # 'same files'. Families missing from the parts ledger get table '' and kind 'not in the parts ledger'.
    out, prev = [], None
    for r in steps:
        f = fam_by_sha.get(r["sha12"]) or {}
        key = f.get("table_key") or ""
        rec = {"when": released(r), "circuit": r["sha12"], "author": r.get("author", ""), "cost": cost(r),
               "table_key": key, "label": label_by_key.get(key, ""), "RBC": tuple(r.get(c, "") for c in "RBC"),
               "kind": "first record", "detail": "", "changed_loaders": [], "from_label": "", "cost_before": None}
        if prev is not None:
            pf = fam_by_sha.get(prev["circuit"]) or {}
            rec["from_label"], rec["cost_before"] = prev["label"], prev["cost"]
            if not key or not pf.get("table_key"):
                rec["kind"] = "not in the parts ledger"
            elif key != pf["table_key"]:
                a, b = table_by_key.get(pf["table_key"], {}), table_by_key.get(key, {})
                if wt_shape(a) != wt_shape(b):
                    rec["kind"] = "new word table, new shape"
                    rec["detail"] = (f"(R, B, C) {', '.join(prev['RBC'])} → {', '.join(rec['RBC'])}; "
                                     f"{a.get('rows')} → {b.get('rows')} rows")
                else:
                    ch = [k for k, (x, y) in enumerate(zip(wt_blocks(a), wt_blocks(b))) if x != y]
                    rec["kind"] = "new word table, same shape"
                    rec["changed_loaders"] = ch
                    rec["detail"] = (f"{len(ch)} of {len(wt_blocks(b))} blocks changed (angle loaders "
                                     f"{', '.join(map(str, ch))})")
            else:
                pa, pb = (pf.get("parts") or "").split(";"), (f.get("parts") or "").split(";")
                ch = [k for k, (x, y) in enumerate(zip(pa, pb)) if x != y]
                w = pf.get("npz_sha256") != f.get("npz_sha256")
                rec["changed_loaders"] = ch
                rec["kind"] = ("angle-loader circuits and weights" if ch and w else "cheaper angle-loader circuit"
                               if ch else "weights only" if w else "same files")
                if ch:
                    try:
                        dt = float(f["T_official"]) - float(pf["T_official"])
                        rec["detail"] = (f"angle loaders {', '.join(map(str, ch))} recompiled, {dt:+,.0f} Toffolis")
                    except (KeyError, ValueError):
                        rec["detail"] = f"angle loaders {', '.join(map(str, ch))} recompiled"
        out.append(rec)
        prev = rec
    return out


def wt_switches(recs):
    # The records where the frontier moved to another word table
    return [r for r in recs[1:] if r["label"] != r["from_label"]]


def wt_held(recs):
    # Spells on the frontier, in order: [{'label', 'table_key', 'from', 'to' (None = still holds), 'records',
    # 'best_cost', 'first_holder', 'last_holder'}]. A table that regains the frontier gets a second spell.
    spells = []
    for r in recs:
        if not spells or spells[-1]["label"] != r["label"]:
            if spells:
                spells[-1]["to"] = r["when"]
            spells.append({"label": r["label"], "table_key": r["table_key"], "from": r["when"], "to": None,
                           "records": 0, "best_cost": r["cost"], "first_holder": r["author"], "last_holder": r["author"]})
        s = spells[-1]
        s["records"] += 1
        s["best_cost"] = min(s["best_cost"], r["cost"])
        s["last_holder"] = r["author"]
    return spells
