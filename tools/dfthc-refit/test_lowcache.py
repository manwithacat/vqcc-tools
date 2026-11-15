# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Tests for lowcache.py and the `lowcache` subcommand (dfthc_refit 1.8.0-dev).

Pure tests need NumPy only. The integration test needs kickmix and Heliosli's family 1f0eb655dd13 in the public
circuits dataset (QHDL_DATASET, or DFTHC_REFIT_1F0E); it checks the recorded result: floor proved on every slot,
shipped gap 0.

    python -I -m unittest test_lowcache -v      (from tools/dfthc-refit)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gauge  # noqa: E402
import lowcache as lc  # noqa: E402

from testpaths import VERIFIERS, dataset_family  # noqa: E402

FAM = dataset_family("DFTHC_REFIT_1F0E", "1f0eb655dd13")
try:
    import kickmix  # noqa: F401
    HAVE_KM = True
except ImportError:
    HAVE_KM = False


def monomial(m):
    return np.array([int((a & m) == m) for a in range(16)], np.uint8)


def signed_table(rng, rows, cols, brot=8):
    w = rng.integers(0, 1 << (brot - 1), size=(rows, cols))
    return (w | ((w >> (brot - 2)) & 1) << (brot - 1)).astype(np.int64)


class PureTests(unittest.TestCase):
    def test_anf_terms(self):
        self.assertEqual(lc.anf_terms(monomial(6)), [6])
        self.assertEqual(lc.anf_terms(monomial(6) ^ monomial(1)), [1, 6])
        self.assertEqual(lc.anf_terms(np.zeros(16, np.uint8)), [])

    def test_rank_full_and_deficient(self):
        rng = np.random.default_rng(1)
        full = rng.integers(0, 128, size=(16 * 40, 5)).astype(np.int64)
        self.assertEqual(lc.low_rank(full), 15)
        # a table that ignores low address bit 3: every monomial containing x3 has a zero row
        t = full.reshape(40, 16, 5).copy()
        t[:, 8:16] = t[:, 0:8]
        self.assertEqual(lc.low_rank(t.reshape(-1, 5)), 7)

    def test_slot_floor_on_a_monomial_gauge(self):
        rng = np.random.default_rng(2)
        rows = 200
        w = signed_table(rng, rows, 10)
        q = np.stack([monomial(m) for m in (3, 5, 6, 9, 10)], axis=1)
        q2 = q.copy()
        q2[:, 4] = 0                                         # slot 1: word 4's gauge changes
        pat = np.concatenate([q[np.arange(rows) % 16], q2[np.arange(rows) % 16]], axis=1)
        words = gauge.to_family_gauge(w, pat)
        r1 = lc.slot_lowcache(words, 1)
        self.assertEqual(r1["free_monomials"], {0: 3, 1: 5, 2: 6, 3: 9})
        self.assertEqual(r1["floor"], 7)
        self.assertEqual(r1["changing_words"], [4])
        r0 = lc.slot_lowcache(words, 0)                     # slot 0 loads from zero: every word changes
        self.assertEqual(r0["floor"], 11)
        rep = lc.family_lowcache(words, shipped_low={0: 11, 1: 8})
        self.assertEqual(rep["gap_total"], 1)
        self.assertEqual(rep["columns_with_changing_gauge"], [0, 1, 2, 3, 4, 9])


@unittest.skipUnless(HAVE_KM and (FAM / "submission.npz").is_file() and (VERIFIERS / "dfthcsos").is_dir(),
                     "needs kickmix, the 1f0eb655dd13 family (QHDL_DATASET) and the official verifiers "
                     "(QHDL_OFFICIAL_ROOT)")
class Family1f0eTest(unittest.TestCase):
    def test_floor_met_on_every_slot(self):
        import dfthc_refit as dr
        rep = dr.lowcache_family(FAM, verifier_dir=VERIFIERS, verbose=False)
        self.assertTrue(all(r["floor_proved"] for r in rep["slots"]))
        self.assertEqual([r["floor"] for r in rep["slots"]], [11] + [6] * 9 + [7, 7, 8, 10, 10])
        self.assertEqual(rep["gap_total"], 0)
        self.assertEqual(rep["changing_columns_untouchable"], [51, 60, 68, 69])


if __name__ == "__main__":
    unittest.main()
