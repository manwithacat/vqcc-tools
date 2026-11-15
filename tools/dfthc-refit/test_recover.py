# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for recover.py (loader plan recovery).

Pure tests (GF(2) helpers, ANF rows, structure census) always run. The round-trip test needs a kickmix build (run
with the kickmix interpreter) and Lera's tensorqrom.py; the integration test needs the 6566cd395128 family and its
words. Each is located through an environment variable (see testpaths.py); tests skip when missing:

    DFTHC_REFIT_LERA_SRC     directory with tensorqrom.py
    DFTHC_REFIT_6566         the 6566cd395128 dfthcsos directory (default $QHDL_DATASET/circuits/chemistry/6566cd395128)
    DFTHC_REFIT_6566_WORDS   the words its loaders load, as .npy (e.g. `recover --out DIR` writes DIR/words.npy)

    "$QHDL_KICKMIX_PYTHON" -I -B test_recover.py -v
"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfthc_refit as dr  # noqa: E402
import recover as rc  # noqa: E402

from testpaths import LERA, dataset_family, env_path  # noqa: E402

F6566 = dataset_family("DFTHC_REFIT_6566", "6566cd395128")
W6566 = env_path("DFTHC_REFIT_6566_WORDS")


def signed_table(rng, A, J=5, brot=8):
    t = rng.integers(0, 1 << (brot - 1), size=(A, J)).astype(np.int64)
    return t | (((t >> (brot - 2)) & 1) << (brot - 1))


def single_terms(full, brot=8, lowbits=4):
    """A trivially exact decomposition: one single product (low monomial l, high monomial h) per nonzero cell."""
    rows, m, L = rc.anf_rows(full, brot, lowbits)
    terms = []
    for h, r in enumerate(rows):
        for l in range(L):
            s = (r >> (l * m)) & ((1 << m) - 1)
            if s:
                terms.append((1 << l, 1 << h, 0, 0, s))
    return terms


class PureTests(unittest.TestCase):
    def test_nullspace_and_basis(self):
        rng = random.Random(1)
        rows = [rng.getrandbits(20) for _ in range(7)]
        ns = rc.nullspace(rows, 20)
        self.assertEqual(len(ns), 20 - rc.rank(rows))
        for x in ns:
            for r in rows:
                self.assertEqual((x & r).bit_count() & 1, 0)
        B = rc.Basis()
        for r in rows:
            B.add(r)
        c = B.solve(rows[0] ^ rows[3])
        v = 0
        for i in rc.set_bits(c):
            v ^= B.vectors[i]
        self.assertEqual(v, rows[0] ^ rows[3])

    def test_anf_rows_match_lera_tensor(self):
        try:
            tq, _ = dr._load_tensorqrom(LERA)
        except Exception as e:  # kickmix or Lera's source missing
            self.skipTest(f"Lera's tensorqrom.py not loadable (set DFTHC_REFIT_LERA_SRC): {type(e).__name__}")
        rng = np.random.default_rng(0)
        full = dr.full_block_delta(signed_table(rng, 100))
        rows, m, L = rc.anf_rows(full)
        C = tq.anf_tensor(full, 8, 4)          # (L, H, O)
        for h, r in enumerate(rows):
            ref = int(sum(int(x) << i for i, x in enumerate(C[1:, h + 1, :].reshape(-1))))
            self.assertEqual(r, ref)

    def test_structure_blocks_and_relations(self):
        m, L = 4, 6
        # two features in slots {0,1} with output 0b0011, two in slot {3} with 0b0100 / 0b1000
        terms = [(0b000011, 1, 0b000001, 2, 0b0011), (0b001000, 1, 0b001000, 2, 0b0100),
                 (0b001000, 4, 0, 0, 0b1000)]
        rows = [rc.column(0b000011, 0b0011, m), rc.column(0b000001, 0b0011, m), rc.column(0b001000, 0b1100, m)]
        st = rc.structure(terms, rows, m, L)
        self.assertEqual(st["products"], 3)
        self.assertEqual(st["features"], 5)
        self.assertEqual(st["dimW"], 4)              # (0b001000, 0b0100) appears twice
        self.assertEqual(st["relations"], L * m - 4)
        self.assertTrue(st["rows_in_W"])
        self.assertEqual(st["low_partition"], [[1, 2], [4]])
        self.assertEqual(st["pairing"]["single"], 1)


class RoundTripTest(unittest.TestCase):
    def test_emit_then_recover(self):
        km = dr.find_kickmix(None)
        if km is None:
            self.skipTest("kickmix not found")
        try:
            tq, _ = dr._load_tensorqrom(LERA)
        except Exception as e:
            self.skipTest(f"Lera's tensorqrom.py not loadable (set DFTHC_REFIT_LERA_SRC): {type(e).__name__}")
        rng = np.random.default_rng(3)
        A = 100
        delta = signed_table(rng, A)
        full = dr.full_block_delta(delta)
        terms = single_terms(full)
        n_sel = (A - 1).bit_length()
        circ = tq.build_circuit(full, terms, n_sel, 8, 4, groups=None, scratch=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "angle_k_0.kmb"
            circ.to_file(str(p), format="kmb")
            _, regs, ops = rc.read_loader(km, p)
            rec = rc.recover_terms(regs, ops)
            self.assertEqual(sorted(rec["terms"]), sorted(terms))
            tt = rc.truth_table(rec, len(full))
            self.assertTrue(np.array_equal(tt[:A], delta))
            rows, m, L = rc.anf_rows(tt)
            st = rc.structure(rec["terms"], rows, m, L)
            self.assertTrue(st["rows_in_W"])
            self.assertEqual(st["pairing"]["single"], len(terms))


class Family6566Test(unittest.TestCase):
    def test_coded_loader_4(self):
        km = dr.find_kickmix(None)
        if km is None or not (F6566 / "angle_k_4.kmb").is_file() or not W6566.is_file():
            self.skipTest("kickmix, the 6566cd395128 family (QHDL_DATASET) or its words (DFTHC_REFIT_6566_WORDS) missing")
        try:
            tq, _ = dr._load_tensorqrom(LERA)
        except Exception as e:
            self.skipTest(f"Lera's tensorqrom.py not loadable (set DFTHC_REFIT_LERA_SRC): {type(e).__name__}")
        words = np.load(W6566)
        rep, rec, rec_x, delta = rc.recover_loader(km, F6566 / "angle_k_4.kmb", words, 4)
        self.assertEqual(rep["num_toffoli_like"], 276)
        self.assertEqual(rep["products"], 196)
        self.assertTrue(rep["recovered_equals_delta_on_rows"])
        st = rep["structure"]
        self.assertEqual(st["relations"], 133)
        self.assertTrue(st["rows_in_W"])
        self.assertEqual(st["low_partition"], [[1, 6, 13], [2, 10, 15], [3, 4, 12], [5, 7, 14], [8, 9, 11]])
        self.assertEqual(st["output_sharing"], {"coded_only": 7, "noncoded_only": 186, "mixed": 3})
        with tempfile.TemporaryDirectory() as td:
            r = rc.reemit(km, tq, rec, rec_x, delta, Path(td) / "angle_k_4.kmb")
        self.assertEqual(r["num_toffoli_like"], 276)
        self.assertEqual(r["num_qubits"], 126)
        self.assertEqual(r["native_check"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
