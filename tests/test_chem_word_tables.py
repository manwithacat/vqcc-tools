"""Stable word-table labels, lineage and frontier forensics (scripts/chem_word_tables.py, 2026-10-31)."""
import csv
import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import chem_word_tables as W  # noqa: E402

import vqcc_config as CFG  # noqa: E402
STAGE = CFG.DATASET
REGISTRY = ROOT / "scripts/chem_word_table_registry.json"


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def table(sha, release, blocks, rows=1220, author="a", J=5):
    return {"table_sha256": sha * 8, "table_key": (sha * 8)[:12], "first_release": release, "first_author": author,
            "first_family": sha, "J": str(J), "brot": "8", "rows": str(rows), "slots": str(len(blocks)),
            "block_words_sha256": ";".join(blocks)}


def blocks(prefix, n=15, keep=None, base="b"):
    # n block hashes; positions in `keep` take the base table's hash
    return [f"{base}{k}" if keep and k in keep else f"{prefix}{k}" for k in range(n)]


class LabelTest(unittest.TestCase):
    def test_numbered_by_first_appearance_and_stable(self):
        t1 = table("a1", "2026-10-28 13:30", blocks("b"))
        t2 = table("a2", "2026-10-28", blocks("c"))            # date alone = 00:00: earlier than t1
        t3 = table("a3", "2026-10-28 13:30", blocks("d"))      # same time as t1: ties by table_sha256
        reg = W.wt_assign({}, [t1, t2, t3], today="2026-10-31")
        by_sha, _ = W.wt_labels(reg)
        self.assertEqual([by_sha[t["table_sha256"]] for t in (t2, t1, t3)], ["W01", "W02", "W03"])
        # Rebuild with a new table that sorts FIRST by release: it is appended, nothing is renumbered
        t0 = table("a0", "2026-10-20", blocks("e"))
        reg2 = W.wt_assign(reg, [t3, t0, t1, t2], today="2026-11-01")
        by2, _ = W.wt_labels(reg2)
        for t in (t1, t2, t3):
            self.assertEqual(by2[t["table_sha256"]], by_sha[t["table_sha256"]])
        self.assertEqual(by2[t0["table_sha256"]], "W04")
        self.assertEqual(reg2["tables"][:3], reg["tables"])
        # A table that disappeared from a build keeps its label; assigning again changes nothing
        self.assertEqual(W.wt_assign(reg2, [t1], today="x"), reg2)

    def test_registry_refuses_renumbering(self):
        import chem_parts_ledger as L
        reg = W.wt_assign({}, [table("a1", "2026-10-28", blocks("b"))])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reg.json"
            self.assertTrue(L.save_registry(reg, p))
            self.assertFalse(L.save_registry(reg, p))      # unchanged: not rewritten
            bad = json.loads(p.read_text())
            bad["tables"][0]["label"] = "W07"
            with self.assertRaises(ValueError):
                L.save_registry(bad, p)

    def test_committed_registry(self):
        reg = json.loads(REGISTRY.read_text())
        labels = [e["label"] for e in reg["tables"]]
        self.assertEqual(labels, [W.WT_FORMAT.format(i) for i in range(1, len(labels) + 1)])
        _, by_key = W.wt_labels(reg)
        self.assertEqual(by_key["27c22740f98d"], "W15")  # Agniv Banerjee
        self.assertEqual(by_key["4e3c8f079137"], "W16")  # Atakan Aldemir (rank label T00 before 31 Oct 11:39)
        self.assertEqual(by_key["a97dd6931a6b"], "W19")  # Lera, C43
        self.assertEqual(by_key["6f2b1b5635b6"], "W20")  # Lera, C42


class LineageTest(unittest.TestCase):
    def test_parent_and_changed_loaders(self):
        agniv = table("ag", "2026-10-28 13:30", blocks("b"))
        other = table("ot", "2026-10-28 14:00", blocks("x", keep={0, 1}))   # shares 2 blocks with agniv
        atakan = table("at", "2026-10-29 13:35", blocks("n", keep={0, 1, 2, 3, 4, 5, 7, 8, 12}))
        lera = table("le", "2026-10-31 11:39", blocks("l"), rows=1363)
        lera2 = table("lf", "2026-10-31 14:28", blocks("m"), rows=1363)
        tabs = [agniv, other, atakan, lera, lera2]
        by_sha, _ = W.wt_labels(W.wt_assign({}, tabs))
        lin = W.wt_lineage(tabs, by_sha)
        a = lin[atakan["table_sha256"]]
        self.assertEqual((a["parent_label"], a["parent_shared_blocks"]), ("W01", 9))
        self.assertEqual(a["changed_loaders"], [6, 9, 10, 11, 13, 14])
        self.assertIn("inherits 9 of 15 blocks from W01", a["lineage_note"])
        self.assertEqual(lin[lera["table_sha256"]]["lineage_note"], "new shape, no parent")
        self.assertEqual(lin[lera2["table_sha256"]]["lineage_note"], "same shape as W04, no shared blocks")
        self.assertEqual(lin[agniv["table_sha256"]]["lineage_note"], "first word table")
        # ties go to the most recent earlier table
        tie = table("ti", "2026-10-30", blocks("q", keep={0, 1}))
        lin2 = W.wt_lineage([agniv, other, tie], W.wt_labels(W.wt_assign({}, [agniv, other, tie]))[0])
        self.assertEqual(lin2[tie["table_sha256"]]["parent_label"], "W02")

    def test_block_origin_and_history(self):
        agniv = table("ag", "2026-10-28 13:30", blocks("b", n=3))
        atakan = table("at", "2026-10-29 13:35", ["b0", "n1", "b2"])
        tabs = [agniv, atakan]
        by_sha, _ = W.wt_labels(W.wt_assign({}, tabs))
        fams = [{"family": "f1", "release": "2026-10-28 13:30", "table_key": agniv["table_key"], "status": "PASS",
                 "parts": "k00-p0;k01-p1;k02-p2", "author": "Agniv"},
                {"family": "f2", "release": "2026-10-28 15:00", "table_key": agniv["table_key"], "status": "PASS",
                 "parts": "k00-q0;k01-p1;k02-p2", "author": "H"},
                {"family": "f3", "release": "2026-10-29 13:35", "table_key": atakan["table_key"], "status": "PASS",
                 "parts": "k00-q0;k01-r1;k02-p2", "author": "Atakan"}]
        circ = [{"part_id": p, "toffoli_per_shot": t} for p, t in
                (("k00-p0", "610"), ("k00-q0", "602"), ("k01-p1", "600"), ("k01-r1", "590"), ("k02-p2", "500"))]
        rows = {(r["stable_label"], r["angle_loader_k"]): r for r in W.wt_block_rows(tabs, by_sha, fams, circ)}
        self.assertEqual(rows[("W01", 0)]["n_cheaper_compiles"], 1)
        self.assertEqual((rows[("W01", 0)]["first_toffoli"], rows[("W01", 0)]["cheapest_toffoli"]), (610.0, 602.0))
        self.assertEqual(rows[("W02", 0)]["data_origin"], "W01")      # inherited block, and its circuit counts from
        self.assertEqual(rows[("W02", 0)]["n_circuits"], 1)          # its first use on W02
        self.assertTrue(rows[("W02", 1)]["new_data"])
        self.assertTrue(rows[("W02", 1)]["changed_vs_parent"])

    @unittest.skipUnless((STAGE / "parts/word_table_blocks.csv").is_file(), "public circuits dataset not present (QHDL_DATASET)")
    def test_stage_atakan_agniv(self):
        tabs = {r["table_key"]: r for r in _read(STAGE / "parts/word_tables.csv")}
        at = tabs["4e3c8f079137"]
        self.assertEqual((at["stable_label"], at["parent_label"], at["parent_shared_blocks"]), ("W16", "W15", "9"))
        self.assertEqual(at["changed_loaders"], "6;9;10;11;13;14")
        self.assertEqual(tabs["a97dd6931a6b"]["lineage_note"], "new shape, no parent")


def rec_row(sha, when, cost, R="8", C="60"):
    return {"task": "chemistry", "sha12": sha, "public_since": when, "release_date": when, "local_task_score": str(cost),
            "author": f"author-{sha}", "R": R, "B": "143", "C": C}


if __name__ == "__main__":
    unittest.main()
