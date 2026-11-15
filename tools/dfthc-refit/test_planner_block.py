# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for planner_block.py (generic-loader block planner).

Pure tests (NumPy only): the phase-1 ranking, and end-to-end plans on a synthetic generic table built from a known
paired plan on a known partition (exact, real rows unchanged, no worse than the known plan). The integration test
plans generic loader 0 of family 5619c37ef6bd (Syed Asad Ali's table; its words as .npy in DFTHC_REFIT_5619_WORDS,
about one CPU-minute without exchange moves) on his recovered partition and checks <= 218 products, exact.

    "$QHDL_KICKMIX_PYTHON" -I -B test_planner_block.py -v
"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfthc_refit as dr  # noqa: E402
import planner_block as pb  # noqa: E402
import planner_shared as ps  # noqa: E402
import recover as rc  # noqa: E402
from test_planner_shared import synthetic  # noqa: E402

from testpaths import env_path  # noqa: E402

W5619 = env_path("DFTHC_REFIT_5619_WORDS")
SYED_GENERIC = ([1, 5, 13], [2, 6, 14], [3, 7, 15], [4, 8, 11], [9, 10, 12])


class PureTests(unittest.TestCase):
    def test_cuts_score(self):
        table, _ = synthetic(5)
        A = len(table) - 4
        blocks = ps.random_triples(15, random.Random(5))
        P = ps.SharedPlanner(table, A, (), coded_block=False)
        P.setup(blocks)
        ok, sp = P.feasible([])
        self.assertTrue(ok)
        b = blocks[0]
        # a relation on one coordinate (low monomial b[0] + 1, output 0) shrinks exactly the 4 forms containing it
        r = 1 << (b[0] * P.m)
        self.assertEqual(pb.cuts_score(P, r, sp), 4)
        self.assertEqual(pb.cuts_score(P, 0, sp), 0)

    def test_plan_generic_synthetic(self):
        # admission from nothing on a tiny synthetic table (rank 23 rows, kernels of ~95 per block) stops far above
        # the known 16-pair plan (that is planner_shared.structural's case); only validity is asserted here
        table, n_true = synthetic(5)
        A = len(table) - 4
        blocks = ps.random_triples(15, random.Random(5))      # the partition synthetic(5) drew
        counts = []
        for ex in (None, dict(cpu_cap=5)):
            res = pb.plan_generic(table, A, blocks, seed=0, phase1_kw=dict(check_from=2, cpu_cap=10),
                                  admit_kw=dict(cpu_cap=5), exchange_kw=ex)
            self.assertTrue(ps.exact_rows(res["delta"], res["terms"]))
            self.assertTrue(np.array_equal(res["delta"][:A], table[:A]))
            self.assertLess(res["products"], (15 * 35 + 1) // 2)
            self.assertEqual(sum(res["per_block"]), res["relations"])
            self.assertEqual(res["products"], res["pairs"] + res["singles"])
            counts.append(res["phases"])
        self.assertLessEqual(counts[1]["exchange"]["best"][0], counts[1]["phase2"]["products"])


class Family5619Test(unittest.TestCase):
    def test_generic_loader_0(self):
        if not W5619.is_file():
            self.skipTest("family 5619c37ef6bd words missing (set DFTHC_REFIT_5619_WORDS)")
        words = np.load(W5619).astype(np.int64)
        delta, cw = dr.loader_delta(words, 0)
        self.assertEqual(cw, [])
        blocks = [[s - 1 for s in b] for b in SYED_GENERIC]
        res = pb.plan_generic(delta, len(words), blocks, seed=0, admit_kw=dict(cpu_cap=30))
        self.assertTrue(ps.exact_rows(res["delta"], res["terms"]))
        self.assertTrue(np.array_equal(res["delta"][:len(words)], dr.full_block_delta(delta)[:len(words)]))
        self.assertLessEqual(res["products"], 218)


if __name__ == "__main__":
    unittest.main()
