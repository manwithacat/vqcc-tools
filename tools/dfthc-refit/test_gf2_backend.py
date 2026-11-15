# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for the GF(2) backend switch.

backend="python" is the reference; backend="rust" uses kickmix_analysis (crate kmx-analysis in the engine
repository). These tests compare the two directly and skip when kickmix_analysis is not importable. Run with the
interpreter that has the engine analysis extension ($QHDL_ANALYSIS_PYTHON), or with any interpreter and
KICKMIX_ANALYSIS_PATH set to .../crates/kmx-analysis/python:

    "$QHDL_ANALYSIS_PYTHON" -I -B test_gf2_backend.py -v
"""
import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner_block as pb  # noqa: E402
import planner_shared as ps  # noqa: E402
import price_words as pw  # noqa: E402
import recover as rc  # noqa: E402
from test_planner_shared import synthetic  # noqa: E402

# deterministic (no binding CPU cap) but light admission, so each plan pair takes seconds
LIGHT = dict(cpu_cap=1e9, max_fails=1, patience=4, trials=1)

try:
    rc.rust_module()
    RUST = None
except ImportError as e:
    RUST = str(e)


class SwitchTests(unittest.TestCase):
    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            rc.make_basis(8, "fortran")

    def test_default_python(self):
        self.assertIsInstance(rc.make_basis(8, "python"), rc.Basis)


@unittest.skipIf(RUST, RUST or "")
class ParityTests(unittest.TestCase):
    def test_basis_ops(self):
        rng = random.Random(7)
        for width in (35, 525):
            P, Q = rc.make_basis(width, "python"), rc.make_basis(width, "rust")
            for _ in range(3000):
                if len(P) and rng.random() < 0.5:
                    v = 0
                    for x in P.vectors:
                        if rng.getrandbits(1):
                            v ^= x
                else:
                    v = rng.getrandbits(width)
                self.assertEqual(P.reduce(v), Q.reduce(v))
                self.assertEqual(P.solve(v), Q.solve(v))
                self.assertEqual(ps.decompose(P, v), Q.decompose(v))
                self.assertEqual(P.add(v), Q.add(v))
            self.assertEqual(P.vectors, Q.vectors)

    def test_matrix_ops(self):
        rng = random.Random(8)
        for _ in range(200):
            m = rng.randrange(1, 70)
            rows = [rng.getrandbits(m) for _ in range(rng.randrange(0, m + 3))]
            self.assertEqual(rc.nullspace(rows, m, "python"), rc.nullspace(rows, m, "rust"))
            self.assertEqual(rc.rank(rows, "python"), rc.rank(rows, "rust"))
            self.assertEqual(pw.gf2_rank(rows, "python"), pw.gf2_rank(rows, "rust"))
            V1, V2 = rc.nullspace(rows[:m // 2], m), rc.nullspace(rows[m // 2:], m)
            self.assertEqual(ps.intersect(V1, V2, m, "python"), ps.intersect(V1, V2, m, "rust"))

    def test_plan_identical_synthetic(self):
        table, _ = synthetic(3)
        A = len(table) - 4
        out = [ps.plan(table, A, (), seed=1, partitions=2, keep=1, admit_kw=LIGHT, backend=b)
               for b in ("python", "rust")]
        self.assertEqual(out[0]["terms"], out[1]["terms"])
        self.assertEqual(out[0]["products"], out[1]["products"])

    def test_plan_identical_synthetic_coded(self):
        table, _ = synthetic(7, coded=True)
        A = len(table) - 4
        blocks = ps.random_triples(15, random.Random(7))     # the same draw as synthetic(7)
        out = [ps.plan(table, A, (0,), seed=0, blocks=blocks, admit_kw=LIGHT, backend=b)
               for b in ("python", "rust")]
        self.assertEqual(out[0]["terms"], out[1]["terms"])

    def test_plan_generic_identical_synthetic(self):
        table, _ = synthetic(5)
        A = len(table) - 4
        blocks = ps.random_triples(15, random.Random(5))
        p1 = dict(check_from=2, cpu_cap=1e9, max_fails=1, patience=4)
        out = [pb.plan_generic(table, A, blocks, seed=0, phase1_kw=p1, admit_kw=dict(LIGHT, protect=4), backend=b)
               for b in ("python", "rust")]
        self.assertEqual(out[0]["terms"], out[1]["terms"])


if __name__ == "__main__":
    unittest.main()
