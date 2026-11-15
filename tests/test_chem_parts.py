"""Pure-logic tests for the chemistry parts tools (scripts/chem_parts_ledger.py and friends).

No kickmix and no official verifier: hashing, ordering, the cost model, the family check and the per-slot
assembler are tested on synthetic records. Needs NumPy (system python3); skipped where it is missing.
"""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the locked uv environment has no NumPy
    raise unittest.SkipTest("numpy not available")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import chem_assemble as A  # noqa: E402
import chem_family_check as F  # noqa: E402
import chem_parts_ledger as L  # noqa: E402


def probe(fam, blocks, files, npz="n1", tree="t1", layout="L1", status="PASS", T=10.0, lam=48.0, cost=1.0,
          npz_words_equal=True):
    """Synthetic probe record: blocks = words sha per slot, files = angle file sha per slot."""
    ch = L.chain(blocks)
    return {"family": fam, "status": status, "error": None if status == "PASS" else "boom",
            "npz_sha256": npz, "tree_edges_sha256": tree, "J": 2, "rows": 4, "brot": 8,
            "block_words_sha256": list(blocks), "table_sha256": L.table_sha(blocks),
            "npz_words_equal": npz_words_equal,
            "metrics": {"lambda_eff": lam}, "resources": {"toffoli": T * len(blocks)}, "score": cost,
            "parts": [{"k": k, "sha256": f, "layout_sha256": layout, "prev_words_sha256": ch[k][0],
                       "words_sha256": ch[k][1], "toffoli_per_shot": T, "num_qubits": 100, "static_toffoli": int(T),
                       "selection_bits": 2, "file": f"angle_k_{k}.kmb"} for k, f in enumerate(files)]}


class HelperTests(unittest.TestCase):
    def test_block_sha_depends_on_values_shape_not_dtype(self):
        a = np.arange(6).reshape(2, 3)
        self.assertEqual(L.block_sha(a), L.block_sha(a.astype(np.int32)))
        self.assertNotEqual(L.block_sha(a), L.block_sha(a.reshape(3, 2)))
        self.assertNotEqual(L.block_sha(a), L.block_sha(a + 1))

    def test_block_sha_matches_probe_convention(self):
        blk = np.array([[1, 2], [3, 4]], dtype=np.int64)
        import hashlib
        self.assertEqual(L.block_sha(blk), hashlib.sha256(blk.tobytes() + b"(2, 2)").hexdigest())

    def test_split_blocks_pads_last_slot(self):
        w = np.arange(2 * 5).reshape(2, 5)
        b = L.split_blocks(w, 2, 3)
        self.assertEqual([x.shape for x in b], [(2, 2)] * 3)
        np.testing.assert_array_equal(b[2], [[4, 0], [9, 0]])
        with self.assertRaises(ValueError):
            L.split_blocks(w, 2, 2)

    def test_chain_starts_from_zero(self):
        self.assertEqual(L.chain(["a", "b", "c"]), [(L.ZERO_BLOCK, "a"), ("a", "b"), ("b", "c")])

    def test_release_key_orders_date_only_first_and_blank_last(self):
        keys = sorted(["2026-10-28 13:30", "2026-10-28", "", "2026-10-27"], key=L.release_key)
        self.assertEqual(keys, ["2026-10-27", "2026-10-28", "2026-10-28 13:30", ""])

    def test_layout_sha_is_order_sensitive(self):
        self.assertNotEqual(L.layout_sha([0, 1], [[2, 3]]), L.layout_sha([1, 0], [[2, 3]]))

    def test_predict_cost_reproduces_official_0418(self):
        # 0418007f2646 official: lambda 48.63242802356732, T 16306 (8376 angles), Q 430, cost 340990159.68148416;
        # heuristic part from its probe resources (R8 B143 C60 brot 8).
        heur = {"toffoli": 16306 - 8376, "persistent_qubits": 430 - 40 - 340, "temporary_qubits": 340}
        p = L.predict_cost(48.63242802356732, 8376, heur, 5, 8, 11, 126)
        self.assertEqual(p["qubits"], 430)
        self.assertAlmostEqual(p["cost"], 48.63242802356732 * 16306 * 430)
        self.assertEqual(L.predict_cost(1, 1, heur, 5, 8, 11, 129)["cost"], float("inf"))

    def test_predict_cost_angle_temporaries_can_dominate(self):
        heur = {"toffoli": 0, "persistent_qubits": 10, "temporary_qubits": 5}
        self.assertEqual(L.predict_cost(1, 1, heur, 1, 4, 2, 20)["qubits"], 10 + 4 + (20 - 2 - 4))


class FamilyCheckTests(unittest.TestCase):
    def setUp(self):
        self.base = probe("A", ["b0", "b1", "b2"], ["f0", "f1", "f2"])
        self.other = probe("O", ["c0", "c1", "c2"], ["g0", "g1", "g2"], npz="n9")
        self.tid = {self.base["table_sha256"]: "T00", self.other["table_sha256"]: "T01"}
        self.ref = F.reference_from_probes({"A": self.base, "O": self.other}, self.tid)
        self.compat = {("n1", "T00"): {"pass": True, "lambda_eff": 48.0, "reason": ""},
                       ("n2", "T00"): {"pass": True, "lambda_eff": 47.9, "reason": ""},
                       ("n3", "T00"): {"pass": False, "lambda_eff": None, "reason": "error_2 21 mHa"}}

    def check(self, fam):
        return F.check_family(fam, self.ref, self.compat)

    def names(self, res):
        return [n for n, _, _ in res["flags"]]

    def test_identical_copy(self):
        r = self.check(probe("B", ["b0", "b1", "b2"], ["f0", "f1", "f2"]))
        self.assertEqual((r["class"], r["status"], r["table"]), ("identical copy", "OK", "T00"))

    def test_loader_only(self):
        r = self.check(probe("B", ["b0", "b1", "b2"], ["f0", "h1", "f2"]))
        self.assertEqual((r["class"], r["new_loader_slots"], r["status"]), ("loader-only", [1], "OK"))
        self.assertEqual(r["reused_from"], {0: "A", 1: None, 2: "A"})

    def test_weights_only_and_loader_plus_weights(self):
        self.assertEqual(self.check(probe("B", ["b0", "b1", "b2"], ["f0", "f1", "f2"], npz="n2"))["class"],
                         "weights-only")
        self.assertEqual(self.check(probe("B", ["b0", "b1", "b2"], ["h0", "f1", "f2"], npz="n2"))["class"],
                         "loader+weights")

    def test_new_word_table_on_closest_table(self):
        r = self.check(probe("B", ["b0", "x1", "x2"], ["f0", "h1", "h2"], npz="n2"))
        self.assertEqual((r["class"], r["table"]), ("new word table", "T00"))
        self.assertIn("NEW_WORD_TABLE", self.names(r))
        self.assertNotIn("MIXED_TABLES", self.names(r))

    def test_mixed_tables(self):
        r = self.check(probe("B", ["b0", "b1", "c2"], ["f0", "f1", "h2"], npz="n2"))
        self.assertIn("MIXED_TABLES", self.names(r))

    def test_chain_break_static_mix_of_valid_parts(self):
        # Each file is valid in its own family, but O's slot-2 loader starts from c1, not b1.
        fam = {"family": "S", "status": None, "npz_sha256": "n1", "tree_edges_sha256": "t1", "rows": 4,
               "block_words_sha256": None, "npz_words_equal": None,
               "parts": [{"k": 0, "sha256": "f0", "layout_sha256": "L1"},
                         {"k": 1, "sha256": "f1", "layout_sha256": "L1"},
                         {"k": 2, "sha256": "g2", "layout_sha256": "L1"}]}
        r = self.check(fam)
        self.assertEqual(r["status"], "BREAK")
        self.assertIn("CHAIN_BREAK", self.names(r))

    def test_unknown_part_static(self):
        fam = {"family": "S", "status": None, "npz_sha256": "n1", "tree_edges_sha256": "t1", "rows": 4,
               "block_words_sha256": None, "npz_words_equal": None,
               "parts": [{"k": 0, "sha256": "f0", "layout_sha256": "L1"},
                         {"k": 1, "sha256": "zz", "layout_sha256": "L1"},
                         {"k": 2, "sha256": "f2", "layout_sha256": "L1"}]}
        r = self.check(fam)
        self.assertIn("UNKNOWN_PART", self.names(r))
        self.assertNotIn("NEW_WORD_TABLE", self.names(r))

    def test_layout_tree_weights_npz_flags(self):
        r = self.check(probe("B", ["b0", "b1", "b2"], ["h0", "h1", "h2"], layout="L2", tree="t2", npz="n3",
                             npz_words_equal=False))
        got = {n: s for n, s, _ in r["flags"]}
        self.assertEqual(got, {"LAYOUT_MISMATCH": "NOTE", "TREE_MISMATCH": "NOTE",
                               "WEIGHTS_FAIL_ON_TABLE": "BREAK", "NPZ_WORDS_DIFFER": "NOTE"})

    def test_layout_mismatch_inside_family_breaks(self):
        fam = probe("B", ["b0", "b1", "b2"], ["h0", "h1", "h2"])
        fam["parts"][1]["layout_sha256"] = "L2"
        r = self.check(fam)
        self.assertEqual([(n, s) for n, s, _ in r["flags"]], [("LAYOUT_MISMATCH", "BREAK")])

    def test_verify_fail(self):
        r = self.check(probe("B", ["b0", "b1", "b2"], ["f0", "f1", "f2"], status="FAIL"))
        self.assertEqual((r["class"], r["status"]), ("FAILS VERIFIER", "BREAK"))

    def test_every_flag_has_help(self):
        for name in ("VERIFY_FAIL", "CHAIN_BREAK", "UNKNOWN_PART", "NEW_WORD_TABLE", "MIXED_TABLES",
                     "LAYOUT_MISMATCH", "TREE_MISMATCH", "WEIGHTS_FAIL_ON_TABLE", "NPZ_WORDS_DIFFER"):
            self.assertIn(name, F.FLAG_HELP)


class AssemblerTests(unittest.TestCase):
    def test_cheapest_loader_per_slot_respects_chain_and_layout(self):
        blocks = ["b0", "b1", "b2"]
        a = probe("A", blocks, ["f0", "f1", "f2"], T=10)
        b = probe("B", blocks, ["h0", "h1", "h2"], T=10)
        b["parts"][1]["toffoli_per_shot"] = 7          # cheaper slot 1
        c = probe("C", ["b0", "x1", "b2"], ["f0", "y1", "y2"], T=1)  # cheap, but other words
        d = probe("D", blocks, ["q0", "q1", "q2"], T=1, layout="L2")  # cheap, other layout
        d["parts"] = d["parts"][:2]                     # layout L2 lacks slot 2
        rank = {"A": 0, "B": 1, "C": 2, "D": 3}
        lo = A.cheapest_loaders(blocks, A.slot_options({"A": a, "B": b, "C": c, "D": d}), rank)
        self.assertEqual(set(lo), {"L1"})
        self.assertEqual([s["sha256"] for s in lo["L1"]], ["f0", "h1", "f2"])

    def test_rank_candidates_uses_passing_npz_only(self):
        blocks = ["b0", "b1"]
        a = probe("A", blocks, ["f0", "f1"], T=10)
        tables = [{"table_key": "k0", "table_label": "T00", "block_words_sha256": "b0;b1", "J": "2", "brot": "8"}]
        compat = [{"table_key": "k0", "npz_sha256": "n1", "pass": "True", "lambda_eff": "48"},
                  {"table_key": "k0", "npz_sha256": "n2", "pass": True, "lambda_eff": "47"},
                  {"table_key": "k0", "npz_sha256": "n3", "pass": "False", "lambda_eff": "1"}]
        fz = [{"npz_sha256": n, "R": "1", "B": "1", "C": "1", "first_family": "A"} for n in ("n1", "n2", "n3")]
        heur = lambda R, B, C, brot: {"toffoli": 100, "persistent_qubits": 50, "temporary_qubits": 200}  # noqa
        c = A.rank_candidates(tables, A.slot_options({"A": a}), compat, fz, heur, {"A": 0})
        self.assertEqual([x["npz_sha256"] for x in c], ["n2", "n1"])
        self.assertAlmostEqual(c[0]["pred_cost"], 47 * 120 * (50 + 16 + 200))


class LedgerBuildHelperTests(unittest.TestCase):
    def test_compat_plan_reuses_known_pairs_and_skips_other_sizes(self):
        npz = [{"npz_sha256": "n1", "rows": "4"}, {"npz_sha256": "n2", "rows": "4"}, {"npz_sha256": "n3", "rows": "9"}]
        tables = [{"table_key": "a", "rows": 4}, {"table_key": "b", "rows": 9}]
        reuse, todo = L.compat_plan(npz, tables, {("n1", "a"): {}})
        self.assertEqual(reuse, [("n1", "a")])
        self.assertEqual(todo, [("n2", "a"), ("n3", "b")])

    def test_load_prev_compat_maps_old_table_ids(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            L.write_csv(Path(d) / "word_tables.csv", [{"table_id": "T00", "table_sha256": "abcdef0123456789"}],
                        ["table_id", "table_sha256"])
            L.write_csv(Path(d) / "compatibility.csv", [{"npz_sha256": "n1", "table_id": "T00", "pass": "True"}],
                        ["npz_sha256", "table_id", "pass"])
            self.assertEqual(list(L.load_prev_compat(d)), [("n1", "abcdef012345")])
            self.assertEqual(L.load_prev_compat(Path(d) / "missing"), {})

    def test_derive_events_first_appearances_in_order(self):
        a = probe("A", ["b0", "b1"], ["f0", "f1"], npz="n1", T=10)
        b = probe("B", ["b0", "b1"], ["f0", "g1"], npz="n2", T=10)
        c = probe("C", ["b0", "b1"], ["f0", "g1"], npz="n2", status="FAIL")
        tk = {a["table_sha256"]: "k0"}
        part_T = {("f0", 0): 10.0, ("f1", 1): 10.0, ("g1", 1): 7.0}
        ev = L.derive_events(["A", "B", "C"], {"A": a, "B": b, "C": c}, tk, {}, part_T)
        self.assertEqual([(e["family"], e["kind"], e["id"]) for e in ev],
                         [("A", "table", "k0"), ("A", "loader", "k00-f0"), ("A", "loader", "k01-f1"),
                          ("A", "weights", "n1"), ("B", "loader", "k01-g1"), ("B", "weights", "n2")])
        self.assertEqual(ev[4]["toffoli"], 7.0)

    def test_write_ledger_uses_schema_columns_and_int16_words(self):
        import tempfile
        from chem_parts_schema import PARTS_FILES, columns
        files = {name: [{c: "x" for c in columns(name)}] for name in PARTS_FILES}
        words = np.array([[1, 255], [7, 0]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as d:
            L.write_ledger(d, files, {"k0": words}, {"n": 1})
            for name in PARTS_FILES:
                header = (Path(d) / Path(name).name).read_text().splitlines()[0].split(",")
                self.assertEqual(header, columns(name))
            back = np.load(Path(d) / "words" / "k0.npy")
            self.assertEqual(back.dtype, np.int16)
            self.assertEqual(L.block_sha(back), L.block_sha(words))  # hashes are defined on int64 values
            with self.assertRaises(ValueError):
                L.write_ledger(d, files, {"k1": words + 40000}, {})
            bad = {**files, "parts/events.csv": [{"unknown": 1}]}
            with self.assertRaises(ValueError):
                L.write_ledger(d, bad, {}, {})


class BestJsonTests(unittest.TestCase):
    stored = {"npz": {"sha256": "n1"}, "slots": [{"slot": 1, "file_sha256": "b"}, {"slot": 0, "file_sha256": "a"}],
              "verified_cost": 100.0}

    def test_same_files_is_unchanged(self):
        sig = A.assembly_signature("n1", ["a", "b"])
        self.assertEqual(A.stored_signature(self.stored), sig)
        self.assertEqual(A.best_decision(self.stored, sig, None, True), "unchanged")
        self.assertEqual(A.best_decision(self.stored, sig, None, False), "verify")  # parts gone: rebuild

    def test_write_only_after_a_cheaper_pass(self):
        sig = A.assembly_signature("n2", ["a", "b"])
        self.assertEqual(A.best_decision(self.stored, sig, None, True), "verify")
        self.assertEqual(A.best_decision(self.stored, sig, {"status": "FAIL", "cost": 1}, True), "keep")
        self.assertEqual(A.best_decision(self.stored, sig, {"status": "PASS", "cost": 100.0}, True), "keep")
        self.assertEqual(A.best_decision(self.stored, sig, {"status": "PASS", "cost": 99.0}, True), "write")
        self.assertEqual(A.best_decision(None, sig, {"status": "PASS", "cost": 1e9}, True), "write")
        self.assertEqual(A.best_decision(self.stored, sig, {"status": "PASS", "cost": 1e9}, False), "write")


class AllJobsTests(unittest.TestCase):
    def test_reference_is_strictly_earlier_and_ties_listed(self):
        a = probe("A", ["b0"], ["f0"])
        b = probe("B", ["b0"], ["f1"])
        c = probe("C", ["b0"], ["f2"])
        man = {"A": {"release_date": "2026-10-28"}, "B": {"release_date": "2026-10-28 10:00"},
               "C": {"release_date": "2026-10-28 10:00"}}
        jobs = {fam["family"]: (sorted(earlier), ties) for fam, earlier, _, ties in
                F.all_jobs({"C": c, "B": b, "A": a}, man)}
        self.assertEqual(jobs, {"A": ([], []), "B": (["A"], ["C"]), "C": (["A"], ["B"])})
        self.assertEqual([f["family"] for f, _, _, _ in F.all_jobs({"A": a, "B": b}, man, only={"B"})], ["B"])


if __name__ == "__main__":
    unittest.main()
