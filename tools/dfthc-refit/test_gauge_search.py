# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for gauge_search.py: the exact symmetry group of a Givens tree and the per-slice moves."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gauge as gz  # noqa: E402
import gauge_search as gs  # noqa: E402


def binomial_edges(n_nodes):
    """A binomial Givens tree on n_nodes (a power of two): edges (p, p + 2**i) in rounds, as in aba448aeadb5."""
    edges = []
    step = 1
    while step < n_nodes:
        for p in range(step):
            edges.append((p, p + step))
        step *= 2
    return np.array(edges, np.int64)


def loader_delta_plain(words, k, j=5):
    """dfthc_refit.loader_delta without coded padding (enough for the synthetic tables here)."""
    w = np.asarray(words, np.int64)
    A, n = w.shape
    pad = np.zeros((A, -(-n // j) * j), np.int64)
    pad[:, :n] = w
    bk = pad[:, j * k:j * (k + 1)]
    return (bk.copy() if k == 0 else bk ^ pad[:, j * (k - 1):j * k]), []


class TestGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = binomial_edges(16)                       # 15 edges, 16 nodes
        cls.g = gs.group(cls.E)
        rng = np.random.default_rng(7)
        cls.W = rng.integers(0, 256, size=(64, len(cls.E)))

    def test_dims(self):
        d = self.g["dims"]
        self.assertEqual(d["group"], 15)
        self.assertTrue(d["ones_in_D_h"])
        # pure pi + negation projection = the whole group
        self.assertEqual(d["pure_pi"] + d["negation_projection"], d["group"])

    def test_every_f_with_its_negation_set_is_exact(self):
        rng = np.random.default_rng(1)
        Nf = self.g["Nf"].astype(np.int64)
        for _ in range(10):
            f = rng.integers(0, 2, (len(self.W), len(self.E)))
            n = (f @ Nf) % 2
            w2 = gs.apply_move(self.W, n, f)
            self.assertLess(gs.projector_change(self.W, w2, self.E), 1e-12)   # includes words 0, 64, 128, 192

    def test_pi_flip_alone_is_not_exact_in_general(self):
        f = np.zeros(len(self.E), np.int64)
        f[-1] = 1                                         # a leaf edge: flips two components only
        w2 = gs.apply_move(self.W, 0, f)
        self.assertGreater(gs.projector_change(self.W, w2, self.E), 1e-6)

    def test_pure_pi_kernel_leaves_planner_tensor_unchanged(self):
        Nf = self.g["Nf"].astype(np.int64)
        ker = gs._kernel_basis(Nf)
        self.assertEqual(len(ker), self.g["dims"]["pure_pi"])
        rng = np.random.default_rng(3)
        R = np.stack([(rng.integers(0, 2, len(ker)) @ np.array(ker)) % 2 for _ in range(len(self.W))])
        w2 = gs.apply_move(self.W, 0, R)
        self.assertLess(gs.projector_change(self.W, w2, self.E), 1e-12)
        for k in range(3):
            self.assertEqual(gs.slot_rows(self.W, k, loader_delta_plain)[0], gs.slot_rows(w2, k, loader_delta_plain)[0])

    def test_negation_changes_bits_above_lowest_set_bit_only(self):
        w = np.arange(256)
        x = gs.apply_move(w, 1, 0) ^ w
        self.assertTrue(np.all(x & 1 == 0))
        for v in range(1, 256):
            t = (v & -v).bit_length() - 1
            self.assertEqual(int(x[v]), (0xFF << (t + 1)) & 0xFF)

    def test_slice_moves_keep_low_pattern_and_symmetry(self):
        Nf = self.g["Nf"].astype(np.int64)
        W = np.where(self.W % 128 == 64, self.W + 1, self.W)      # no gauge-flipping words: every slice is free
        W = np.where(gz.off_gauge(W).astype(bool), W ^ 128, W)    # signed gauge start
        spaces = gs.allowed_spaces(W, Nf)
        self.assertTrue(all(len(B) == len(self.E) for B in spaces))
        F, w2 = gs.random_variant(W, Nf, spaces, np.random.default_rng(5))
        self.assertTrue(gs.gauge_ok(w2))
        self.assertLess(gs.projector_change(W, w2, self.E), 1e-12)
        self.assertTrue((w2 != W).any())

    def test_allowed_spaces_avoid_gauge_flipping_words(self):
        Nf = self.g["Nf"].astype(np.int64)
        W = self.W.copy()
        W[3, 2] = 64                                             # slice 3, column 2
        spaces = gs.allowed_spaces(W, Nf)
        for f in spaces[3]:
            self.assertEqual(int((f @ Nf)[2] % 2), 0)

    def test_block_bound(self):
        rows = [0b11, 0b01, 0b10, 0b100]
        self.assertEqual(gs.block_bound(rows, [0b11, 0b100], drop_last=False), 2 + 1)
        self.assertEqual(gs.block_bound(rows, [0b11, 0b100]), 2)

    def test_gauge_changes_and_regauge(self):
        Nf = self.g["Nf"].astype(np.int64)
        W = np.where(gz.off_gauge(self.W).astype(bool), self.W ^ 128, self.W)   # signed gauge: no changes
        off = gz.off_gauge(W)
        self.assertEqual(gs.gauge_changes(off, 1), [])
        self.assertEqual(gs.gauge_change_entries(off, 1), 0)
        ker = gs._kernel_basis(Nf)                     # a pure-pi element on slice 0 only
        f = ker[0]
        W2 = gs.apply_move(W, 0, np.where((np.arange(len(W)) % 16 == 0)[:, None], f[None, :], 0))
        self.assertLess(gs.projector_change(W, W2, self.E), 1e-12)
        before = sum(len(gs.gauge_changes(gz.off_gauge(W2), k)) for k in range(3))
        res = gs.regauge(W2, Nf, list(range(len(self.E))), [0, 1, 2], restarts=1)
        self.assertLessEqual(sum(len(c) for c in res["changes"].values()), before)
        self.assertEqual(res["score"][0], 0)           # the signed gauge is reachable again
        self.assertLess(gs.projector_change(W, res["words"], self.E), 1e-12)


if __name__ == "__main__":
    unittest.main()
