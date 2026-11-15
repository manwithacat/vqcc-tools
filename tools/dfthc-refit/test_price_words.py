# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for price_words (word-move pricing). Pure tests run anywhere (NumPy, SciPy); the integration test needs the
competition's verifiers directory ($DFTHC_REFIT_VERIFIER_DIR), PyTorch and a family ($DFTHC_PRICE_FAMILY: a dfthcsos
directory, with $DFTHC_PRICE_QUBITS / $DFTHC_PRICE_TOFFOLIS, default 427 / 15175). Lera's tensorqrom.py
($DFTHC_REFIT_LERA_SRC) is optional, for the ANF cross-check.

    python -m unittest test_price_words -v
"""
import importlib.util
import itertools
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import price_words as pw  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def brute_anf(delta, brot=8, lowbits=4):
    """ANF coefficient by definition: C[l, h, o] = XOR over sub-masks (h', l') of the truth table."""
    delta = np.asarray(delta, np.int64)
    count, J = delta.shape
    L = 1 << lowbits
    H = -(-count // L)
    O = J * (brot - 1)
    t = np.zeros((H, L, O), np.uint8)
    for a in range(count):
        for j in range(J):
            for k in range(brot - 1):
                t[a // L, a % L, j * (brot - 1) + k] = (delta[a, j] >> k) & 1
    C = np.zeros((L, H, O), np.uint8)
    for h in range(H):
        for l in range(L):
            acc = np.zeros(O, np.uint8)
            for h2 in range(h + 1):
                if h2 & h != h2:
                    continue
                for l2 in range(L):
                    if l2 & l == l2:
                        acc ^= t[h2, l2]
            C[l, h] = acc
    return C


def coded_table(rng, rows, cols, coded_cols, lowbits=4, brot=8):
    """Random signed-gauge words whose bit 0 is affine on every 16-row block in `coded_cols`."""
    w = rng.integers(0, 1 << (brot - 1), size=(rows, cols))
    w = w | (((w >> (brot - 2)) & 1) << (brot - 1))
    L = 1 << lowbits
    for c in coded_cols:
        for b0 in range(0, rows, L):
            cst, s = rng.integers(0, 2), rng.integers(0, 2, size=lowbits)
            for l in range(min(L, rows - b0)):
                bit = (cst + sum(int(s[i]) * ((l >> i) & 1) for i in range(lowbits))) % 2
                w[b0 + l, c] = (w[b0 + l, c] & ~1) | bit
    return w.astype(np.int64)


class TestGF2(unittest.TestCase):
    def test_rank(self):
        self.assertEqual(pw.gf2_rank([0b011, 0b101, 0b110]), 2)
        self.assertEqual(pw.gf2_rank([1, 2, 4, 7, 0]), 3)
        self.assertEqual(pw.gf2_rank([]), 0)

    def test_mobius_involution(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(16, 3)).astype(np.uint8)
        self.assertTrue(np.array_equal(pw.mobius(pw.mobius(x, 0), 0), x))

    def test_anf_matches_definition(self):
        rng = np.random.default_rng(1)
        delta = rng.integers(0, 256, size=(37, 2))          # 3 high chunks, partial last block
        self.assertTrue(np.array_equal(pw.anf_tensor(delta), brute_anf(delta)))

    @unittest.skipUnless(os.environ.get("DFTHC_REFIT_LERA_SRC"), "Lera's tensorqrom.py not given (set DFTHC_REFIT_LERA_SRC)")
    def test_anf_matches_lera(self):
        spec = importlib.util.spec_from_file_location(
            "tensorqrom_ref", Path(os.environ["DFTHC_REFIT_LERA_SRC"]) / "tensorqrom.py")
        tq = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(tq)
        except ModuleNotFoundError as ex:            # tensorqrom imports kickmix
            self.skipTest(f"tensorqrom not importable here: {ex}")
        rng = np.random.default_rng(2)
        delta = rng.integers(0, 256, size=(1220, 5))
        self.assertTrue(np.array_equal(pw.anf_tensor(delta, 8, 4), tq.anf_tensor(delta, 8, 4)))


class TestCodedRank(unittest.TestCase):
    def test_slopes_and_rank_bound(self):
        rng = np.random.default_rng(3)
        w = coded_table(rng, 16 * 40, 10, coded_cols=range(10))
        delta, cw = pw.dr.loader_delta(w, 1, 5)
        self.assertEqual(cw, [0, 1, 2, 3, 4])
        s = pw.block_slopes(delta, cw)
        b = delta[:16, 2] & 1
        self.assertEqual(int(s[2, 0, 3]), int(b[8] ^ b[0]))
        r, _ = pw.coded_mixed_rank(delta, cw)
        self.assertLessEqual(r, 20)
        self.assertGreaterEqual(r, 19)                      # random slopes over 39 high monomials: full rank

    def test_planted_dependency(self):
        """Slope (o=0, i=0) equal to (o=1, i=0) on every block: the coded mixed rank drops by one."""
        rng = np.random.default_rng(4)
        w = coded_table(rng, 16 * 40, 5, coded_cols=range(5))
        delta, cw = pw.dr.loader_delta(w, 0, 5)
        r0, _ = pw.coded_mixed_rank(delta, cw)
        for b0 in range(0, len(w), 16):
            s1 = (w[b0 + 1, 1] ^ w[b0, 1]) & 1
            s0 = (w[b0 + 1, 0] ^ w[b0, 0]) & 1
            if s0 != s1:
                for l in range(16):
                    if l & 1:
                        w[b0 + l, 0] ^= 1
        delta, cw = pw.dr.loader_delta(w, 0, 5)
        r1, _ = pw.coded_mixed_rank(delta, cw)
        self.assertEqual(r1, r0 - 1)

    def test_pad_free_never_higher(self):
        rng = np.random.default_rng(5)
        w = coded_table(rng, 16 * 20 + 4, 10, coded_cols=range(10))
        delta, cw = pw.dr.loader_delta(w, 1, 5)
        r, _ = pw.coded_mixed_rank(delta, cw, real_rows=len(w))
        rf, mask = pw.coded_mixed_rank(delta, cw, real_rows=len(w), pad_free=True)
        self.assertLessEqual(rf, r)
        self.assertEqual(pw._free_pad_slopes(16 * 20 + 4), [2, 3])
        self.assertEqual(pw._free_pad_slopes(16 * 20), [])


class TestRankDrop(unittest.TestCase):
    def test_cheapest_move_drops_rank(self):
        rng = np.random.default_rng(6)
        rows = 16 * 60 + 4                                   # 60 high monomials: 20 random slopes are independent
        w = coded_table(rng, rows, 15, coded_cols=range(15))
        fam = SimpleNamespace(words=w)
        price = {c: rng.random(rows) for c in range(15)}
        moves = pw.rank_drop_moves(fam, 2, row_price=price, max_terms=2, keep=5, restarts=20)
        self.assertTrue(moves)
        r0 = pw.loader_coded_ranks(w, loaders=[2])[2][1]
        self.assertEqual(r0, 20)                             # the drop is guaranteed from full rank
        prev = -1.0
        for mv in moves:
            self.assertGreaterEqual(mv.meta["predicted_rel"], prev)
            prev = mv.meta["predicted_rel"]
            w2 = pw.apply_move(w, mv)
            diff = w2 ^ w
            self.assertTrue(np.all((diff == 0) | (diff == 1)))          # only bit 0 moves
            self.assertTrue(pw.gauge_ok(w2).all())
            self.assertEqual(sorted(pw.dr._coded_columns(w2)), list(range(15)))
            self.assertLessEqual(pw.loader_coded_ranks(w2, loaders=[2])[2][1], r0 - 1)
            # predicted price = sum of the per-row prices of the toggled entries
            got = sum(price[c][r] for r, c, _ in mv.changes)
            self.assertAlmostEqual(got, mv.meta["predicted_rel"], places=9)

    def test_protect(self):
        rng = np.random.default_rng(7)
        w = coded_table(rng, 16 * 24, 15, coded_cols=range(15))
        fam = SimpleNamespace(words=w)
        moves = pw.rank_drop_moves(fam, 2, max_terms=1, keep=3, protect=(1, 3))
        for mv in moves:
            for _, c, _ in mv.changes:
                self.assertTrue(False, "every column of loader 2 also feeds loader 1 or 3")
        self.assertEqual(moves, [])

    def test_planner_callback(self):
        rng = np.random.default_rng(8)
        w = coded_table(rng, 16 * 30, 15, coded_cols=range(15))
        fam = SimpleNamespace(words=w)
        mv = pw.rank_drop_moves(fam, 2, max_terms=2, keep=1)[0]
        sv = pw.coded_rank_planner(pad_free=False)(w, pw.apply_move(w, mv), mv)
        a = pw.loader_coded_ranks(w, pad_free=False)
        b = pw.loader_coded_ranks(pw.apply_move(w, mv), pad_free=False)
        want = sum(-(-a[k][1] // 2) + (b[k][1] + 1) // 2 * -1 for k in a)
        self.assertEqual(sv, want)


class TestMovesAndPrices(unittest.TestCase):
    def test_apply_move_checks(self):
        w = np.zeros((4, 3), np.int64)
        with self.assertRaises(ValueError):
            pw.apply_move(w, pw.Move("bad", ((0, 0, 256),)))
        with self.assertRaises(ValueError):
            pw.apply_move(w, pw.Move("gauge", ((0, 0, 64),)))      # bit 6 without its bit-7 copy
        w2 = pw.apply_move(w, pw.Move("ok", ((1, 2, 192),)))
        self.assertEqual(int(w2[1, 2]), 192)
        self.assertEqual(int(w[1, 2]), 0)

    def test_parent_amplitude_dead(self):
        """A word whose parent amplitude is 0 does not change the decoded vector."""
        tree = np.array([[0, 1], [1, 2], [0, 3]])
        words = np.array([[0, 37, 5], [64, 11, 90]], np.int64)    # row 0: rotation 0 is identity, so node 1 is 0
        amp = pw.parent_amplitude(words, tree)
        self.assertEqual(amp[0, 1], 0.0)

        def decode(ws):
            th = np.pi * ws / 128.0
            v = np.zeros((len(ws), 4)); v[:, 0] = 1
            for e, (p, c) in enumerate(tree):
                pv, cv = v[:, p].copy(), v[:, c].copy()
                v[:, p] = np.cos(th[:, e]) * pv - np.sin(th[:, e]) * cv
                v[:, c] = np.sin(th[:, e]) * pv + np.cos(th[:, e]) * cv
            return v
        w2 = words.copy(); w2[0, 1] ^= 1
        self.assertTrue(np.array_equal(decode(words)[0], decode(w2)[0]))

    def test_first_order_and_units(self):
        base = dict(E_gap=4.0, Lambda=280.0, lambda_eff=math.sqrt(4.0 * (2 * 280.0 - 4.0)), error_1=0.015,
                    error_2=0.02)
        fam = SimpleNamespace(base=base, mu=dict(e1=0.0, e2=-30.0, gap=0.125))
        lam = lambda g, L: math.sqrt(g * (2 * L - g))  # noqa: E731
        h = 1e-6
        want = (lam(4.0, 280.0 + h) - lam(4.0, 280.0)) / lam(4.0, 280.0)
        self.assertAlmostEqual(float(pw.first_order_rel(fam, h, 0, 0, 0)), want, delta=1e-12)
        want = (lam(4.0 + h, 280.0) - lam(4.0, 280.0)) / lam(4.0, 280.0) - 0.125 * h
        self.assertAlmostEqual(float(pw.first_order_rel(fam, 0, h, 0, 0)), want, delta=1e-12)
        self.assertAlmostEqual(float(pw.first_order_rel(fam, 0, 0, 1e-6, 0)), 30e-6, delta=1e-15)
        self.assertEqual(float(pw.first_order_rel(fam, 0, 0, 0, 1.0)), 0.0)        # e1 slack: free
        fam2 = SimpleNamespace(base=dict(base), mu=fam.mu, qubits=427, toffolis=15000.0, diag=dict(
            kkt_rel_residual=0.0, wrong_sign=[], kkt_mu=[0, -30, 0.125]))
        fam2.cost = base["lambda_eff"] * 427 * 15000.0
        fam2.loader_toffoli_units = 2 * base["lambda_eff"] * 427
        sp = pw.shadow_prices(fam2)
        self.assertAlmostEqual(sp["units_per_uHa_e2"], fam2.cost * 30 * 2 * 0.02 * 1e-6)
        self.assertAlmostEqual(sp["uHa_e2_per_loader_toffoli"], fam2.loader_toffoli_units / sp["units_per_uHa_e2"])


@unittest.skipUnless(os.environ.get("DFTHC_REFIT_VERIFIER_DIR") and os.environ.get("DFTHC_PRICE_FAMILY")
                     and HAVE_TORCH,
                     "needs $DFTHC_REFIT_VERIFIER_DIR, $DFTHC_PRICE_FAMILY and PyTorch")
class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fam = pw.load_family(os.environ["DFTHC_PRICE_FAMILY"], os.environ["DFTHC_REFIT_VERIFIER_DIR"],
                                 qubits=int(os.environ.get("DFTHC_PRICE_QUBITS", 427)),
                                 toffolis=float(os.environ.get("DFTHC_PRICE_TOFFOLIS", 15175)))

    def test_dead_word_is_free(self):
        fam = self.fam
        amp = pw.parent_amplitude(fam.words, fam.tree)
        r, c = map(int, np.argwhere(amp == 0)[0])
        mv = pw.Move("dead", ((r, c, int(fam.words[r, c] ^ 2)),))
        row = pw.price_move(fam, mv, refit=False)
        self.assertTrue(row["metrics_unchanged"])
        self.assertEqual(row["first_order_rel"], 0.0)

    def test_row_model_tracks_exact(self):
        fam = self.fam
        rp = pw.RowPricer(fam)
        amp = pw.parent_amplitude(fam.words, fam.tree)
        rows = np.flatnonzero(amp[:, 40] > 0.1)[:3]
        for r in rows:
            cand = fam.words[:, 40].copy(); cand[r] ^= 1
            p, _ = rp.prices(40, cand)
            ex = pw.exact_fixed(fam, pw.apply_move(fam.words, pw.Move("t", ((int(r), 40, int(cand[r])),))))
            self.assertLess(abs(p[r] - ex["rel"]), 0.2 * abs(ex["rel"]) + 1e-9)


if __name__ == "__main__":
    unittest.main()
