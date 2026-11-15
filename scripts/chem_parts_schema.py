#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Column order and descriptions of the chemistry parts tables.

Standard library only. scripts/chem_parts_ledger.py writes every parts/*.csv with exactly these columns in this order,
and the same order is used for the dataset's column descriptions. Kaggle keeps column descriptions by POSITION, so
never reorder: append new columns at the end (and rewrite the page's descriptions).

Usage: from chem_parts_schema import PARTS_FILES, columns
"""
from __future__ import annotations

S, I, N, B = "string", "integer", "number", "boolean"

PARTS_FILES: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "parts/angle_circuits.csv": (
        "One row per chemistry angle circuit (loader) and slot: which word block it turns into which, its register "
        "layout, its measured Toffolis per shot and who first published it. A loader fits another family only if both "
        "word blocks and the layout match there.",
        [
            ("part_id", "k<slot>-<first 12 hex of file_sha256>: the loader in this slot", S),
            ("file_sha256", "SHA-256 of the angle_k_<slot>.kmb file", S),
            ("slot_k", "Slot k: the file is angle_k_<k>.kmb and runs as the k-th circuit of the family", I),
            ("dataset_path", "A copy of this exact file in this dataset (in the family that first published it)", S),
            ("bytes", "File size in bytes", I),
            ("prev_words_sha256", "Word block the circuit starts from: block k-1 of its table (ZERO for slot 0). Hash = "
                                  "SHA-256 of the int64 block bytes plus its shape string", S),
            ("words_sha256", "Word block the circuit leaves in the word registers: block k (columns k*J..k*J+J-1)", S),
            ("chain_variants", "Number of distinct (prev, words) transitions seen for this file in this slot (1 = consistent)", I),
            ("word_tables", "table_key(s) of the word tables this loader belongs to (semicolon separated)", S),
            ("J", "Words loaded per circuit (number of word_j registers)", I),
            ("brot", "Bits per angle word (width of each word_j register)", I),
            ("selection_bits", "Width of the selection (address) register", I),
            ("layout_sha256", "Hash of the selection and word_j qubit numbering; every circuit of a family must share it", S),
            ("num_qubits", "Qubits of this circuit, workspace included (cap 128)", I),
            ("static_toffoli", "Toffoli-like gates in the file (CCX, CCZ, CSWAP), one pass", I),
            ("toffoli_per_shot", "Measured mean Toffolis per shot, forward plus reverse pass, in its first family (official "
                                 "simulator, hash-derived seed)", N),
            ("toffoli_per_shot_min", "Minimum of that measurement over every family containing the file", N),
            ("toffoli_per_shot_max", "Maximum of that measurement over every family containing the file", N),
            ("n_families", "Number of dataset chemistry families containing this file in this slot", I),
            ("families", "Those families (sha12 of the family directory, semicolon separated)", S),
            ("first_family", "Earliest-released family containing it (release_date; a date alone counts as 00:00)", S),
            ("first_author", "Author of that family's source notebook: the credit for this loader", S),
            ("first_notebook", "Source notebook URL of the first family", S),
            ("first_notebook_version", "Version of that notebook as read", S),
            ("first_release", "release_date of the first family", S),
            ("word_table_labels", "Stable W-labels of the word tables in word_tables, same order (semicolon separated)", S),
        ]),
    "parts/word_tables.csv": (
        "One row per distinct word table: the ordered list of word blocks the 15 (or 19, 25) loaders produce. Families on "
        "the same table load identical angles and differ only in loaders or weights.",
        [
            ("table_key", "Stable id: first 12 hex of table_sha256", S),
            ("table_label", "T00, T01, ... ranked by the cheapest family on the table; may change between versions "
                            "(kept for compatibility; use stable_label to refer to a table)", S),
            ("rank", "0 = table of the cheapest family", I),
            ("table_sha256", "SHA-256 of the semicolon-joined block hashes in slot order", S),
            ("angles_sha256", "SHA-256 of the whole int64 word matrix plus its shape; equal for one angle table batched "
                              "at different J", S),
            ("J", "Words per circuit", I),
            ("brot", "Bits per word", I),
            ("rows", "Rows of the table: N + R*B vectors (N = 76 orbitals)", I),
            ("slots", "Number of angle circuits", I),
            ("block_words_sha256", "Word block hash per slot, semicolon separated, slot order", S),
            ("words_npy", "The word table as .npy, stored as int16 (values < 2**brot); hashes are defined on the int64 "
                          "values, so cast to int64 before hashing", S),
            ("n_families", "Number of families on this table", I),
            ("families", "Those families (semicolon separated)", S),
            ("tree_edges_sha256", "Givens trees used with this table (hash of the npz tree_edges, semicolon separated)", S),
            ("layout_sha256", "Register layouts used with this table (semicolon separated)", S),
            ("first_family", "Earliest-released family on this table: the origin of these angle words", S),
            ("first_author", "Its author", S),
            ("first_release", "Its release_date", S),
            ("best_family", "Cheapest family on the table", S),
            ("best_cost", "Its local official-verifier cost", N),
            ("stable_label", "Stable label W01, W02, ... in order of first public appearance (first_release, ties by "
                             "table_sha256); assigned once and never reused or renumbered (registry: "
                             "scripts/chem_word_table_registry.json in the project repository)", S),
            ("label_description", "First author, first public date and shape: (R, B, C) of the first family, J, brot, "
                                  "rows and the number of angle loaders", S),
            ("R", "Rank of the first family's factorization", I),
            ("B", "Bases of the first family's factorization", I),
            ("C", "Ensemble copies of the first family's factorization", I),
            ("parent_label", "W-label of the earlier table sharing the most word blocks (ties: the most recent); blank "
                             "if it shares none", S),
            ("parent_shared_blocks", "Number of word blocks shared with the parent (same angle loader k, same hash)", I),
            ("changed_loaders", "Angle loaders k whose block differs from the parent's: the QROMs that had to be "
                                "rebuilt (semicolon separated; every k if there is no parent)", S),
            ("lineage_note", "The lineage in words: 'inherits n of 15 blocks from Wx', 'new shape, no parent', ...", S),
        ]),
    "parts/factorizations.csv": (
        "One row per distinct factorization (submission.npz): shape, Givens tree, credits.",
        [
            ("npz_sha256", "SHA-256 of the npz file", S),
            ("dataset_path", "factorizations/<sha12 of the first family>.npz in this dataset", S),
            ("R", "Rank (first dimension of W)", I),
            ("B", "Bases (second dimension of W)", I),
            ("C", "Ensemble copies (third dimension of W)", I),
            ("rows", "N + R*B: rows of a word table this npz can be evaluated on", I),
            ("brot", "brot stored in the npz", I),
            ("keep_bits", "KEEP_BITS of the verifier's state-preparation model (fixed)", I),
            ("tree_edges_sha256", "Hash of the npz tree_edges (int64 bytes plus shape)", S),
            ("stored_words_table", "table_key of the table equal to the npz's stored angle_words, or none (the verifier "
                                   "uses the words the circuits load, not the stored copy)", S),
            ("n_families", "Families shipping this npz", I),
            ("families", "Those families (semicolon separated)", S),
            ("first_family", "Earliest-released family with this npz", S),
            ("first_author", "Its author: the credit for these weights", S),
            ("first_notebook", "Its source notebook URL", S),
            ("first_release", "Its release_date", S),
        ]),
    "parts/compatibility.csv": (
        "Every factorization evaluated on every word table with the same row count, with the official evaluate_words "
        "(no circuit simulation): does this npz pass the accuracy gates with these words, and at what lambda_eff.",
        [
            ("npz_sha256", "The factorization", S),
            ("npz_first_family", "Family it first appeared in", S),
            ("table_key", "The word table", S),
            ("table_label", "Its rank label in this version", S),
            ("tree_match", "True if the npz's Givens tree is one already used with this table", B),
            ("error_1_mHa", "One-body residual in milliHartree (limit 20)", N),
            ("error_2_mHa", "Two-body residual in milliHartree (limit 20)", N),
            ("Lambda", "LCU one-norm", N),
            ("E_gap", "SOS gap estimate", N),
            ("lambda_eff", "Effective lambda: the first factor of the chemistry cost", N),
            ("pass", "True if every accuracy, gap and floor gate passes", B),
            ("reason", "Why it fails, if it does", S),
            ("stable_label", "Stable W-label of the word table", S),
        ]),
    "parts/families.csv": (
        "One row per chemistry family: its word table, factorization and loader per slot, the official lambda_eff, "
        "Toffolis and qubits next to the same numbers rebuilt from parts, and the consistency check's verdict.",
        [
            ("family", "sha12 of the family directory (circuits/chemistry/<family>/)", S),
            ("author", "Source notebook author", S),
            ("release", "release_date", S),
            ("first_seen", "Date collected", S),
            ("status", "PASS or FAIL under the official verifier code", S),
            ("error", "Rejection reason, if any", S),
            ("table_key", "Its word table", S),
            ("table_label", "Rank label of that table", S),
            ("J", "Words per circuit", I),
            ("brot", "Bits per word", I),
            ("npz_sha256", "Its factorization", S),
            ("parts", "part_id per slot, semicolon separated", S),
            ("lambda_official", "lambda_eff from the official simulation of the family", N),
            ("lambda_parts", "lambda_eff of (npz, table) from compatibility.csv", N),
            ("lambda_abs_dev", "|lambda_parts - lambda_official|", N),
            ("T_official", "Total Toffolis from the official simulation", N),
            ("T_parts", "Sum of the parts' toffoli_per_shot plus the verifier's heuristic Toffolis", N),
            ("T_dev", "T_parts - T_official", N),
            ("Q_official", "Total qubits from the verifier", I),
            ("Q_parts", "Total qubits rebuilt from parts", I),
            ("cost_official", "Local official-verifier cost", N),
            ("cost_parts", "Cost rebuilt from parts", N),
            ("npz_words_equal", "True if the npz's stored angle_words equal the loaded words", B),
            ("check_class", "Against families released before it: identical copy, loader-only, weights-only, "
                            "loader+weights, new word table or FAILS VERIFIER", S),
            ("check_status", "OK, NOTE (limits how parts can be mixed) or BREAK (cannot work as shipped)", S),
            ("check_flags", "Flags raised (see the About text), semicolon separated", S),
            ("stable_label", "Stable W-label of its word table", S),
        ]),
    "parts/events.csv": (
        "History in long form: one row each time a new word table, loader or factorization first appears, for charting "
        "trends per slot and per table.",
        [
            ("release", "release_date of the family that introduced it", S),
            ("family", "That family", S),
            ("author", "Its author", S),
            ("kind", "table, loader or weights", S),
            ("table_key", "Word table it belongs to (weights: the table of the introducing family)", S),
            ("slot", "Slot k for a loader, blank otherwise", I),
            ("id", "part_id (loader), table_key (table) or first 12 hex of npz_sha256 (weights)", S),
            ("toffoli", "Loader: toffoli_per_shot; table: the family's total Toffolis", N),
            ("lambda_eff", "lambda_eff of the introducing family", N),
            ("cost_on_table", "Local cost of the introducing family", N),
            ("stable_label", "Stable W-label of the word table", S),
        ]),
    "parts/word_table_blocks.csv": (
        "One row per word table and angle loader k: where that block of angle words first appeared (its data origin) "
        "and how the circuits for it improved on this table. 'New data' means the QROM had to be rebuilt; a cheaper "
        "compile loads the same data with fewer Toffolis.",
        [
            ("table_key", "The word table (first 12 hex of table_sha256)", S),
            ("stable_label", "Its stable W-label", S),
            ("angle_loader_k", "Angle loader k (official name ANGLE_k; file angle_k_<k>.kmb)", I),
            ("block_words_sha256", "Hash of the block of words this angle loader writes", S),
            ("data_origin", "W-label of the first table (lowest W-number) with this block at this k", S),
            ("new_data", "True if this table is the block's origin: the data first appeared here", B),
            ("changed_vs_parent", "True if the block differs from the parent table's block at this k", B),
            ("parent_label", "The table's parent (see word_tables.csv)", S),
            ("n_circuits", "Distinct circuits for this angle loader used by families on this table", I),
            ("n_cheaper_compiles", "How many of those, in public order, were cheaper than every earlier one", I),
            ("first_toffoli", "Toffolis per shot of the first circuit used on this table", N),
            ("cheapest_toffoli", "Toffolis per shot of the cheapest", N),
            ("cheapest_part_id", "part_id of the cheapest (angle_circuits.csv)", S),
        ]),
}

OTHER_FILES = {
    "parts/assembly_best.json": "Cheapest verified chemistry assembly from public parts: per slot the loader's "
                                "dataset_path, SHA-256 and author, the npz used, lambda_eff, Toffolis, qubits, predicted "
                                "and verified cost (local official verifier, default salt) and the verify date.",
    "parts/summary.json": "Counts and the parts-only reproduction check (gate G1) for this version of the ledger.",
}


def columns(name: str) -> list[str]:
    return [c for c, _, _ in PARTS_FILES[name][1]]
