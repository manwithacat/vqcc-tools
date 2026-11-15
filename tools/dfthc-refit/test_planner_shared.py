# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for planner_shared.py (block-partition planner with structural relations and shared products).

Pure tests run anywhere (NumPy only): GF(2) intersection, the padded-row rewrite (against dfthc_refit), and
end-to-end plans on synthetic tables built from a known paired plan (generic and coded). The integration test plans
loader 4 of the 6566cd395128 family with its recovered partition (needs its words, see test_recover.py; about one
CPU-minute) and checks <= 198 products, exact, real rows unchanged.

    "$QHDL_KICKMIX_PYTHON" -I -B test_planner_shared.py -v
"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfthc_refit as dr  # noqa: E402
import planner_shared as ps  # noqa: E402
import recover as rc  # noqa: E402

from testpaths import env_path  # noqa: E402

W6566 = env_path("DFTHC_REFIT_6566_WORDS")


def table_from_rows(rows, H, m=35, L=15, J=5, brot=8):
    """Full-block (16 H, J) signed-gauge table whose mixed ANF rows are `rows` (pure parts zero)."""
    T = np.zeros((H, L + 1, m), np.uint8)
    for h, r in enumerate(rows):
        for l in range(L):
            for o in range(m):
                T[h + 1, l + 1, o] = (r >> (l * m + o)) & 1
    for axis in (0, 1):
        t = np.swapaxes(T, axis, 0).copy()
        n = t.shape[0]
        ids = np.arange(n)
        for bit in range((n - 1).bit_length()):
            sel = ids[(ids & (1 << bit)) != 0]
            t[sel] ^= t[sel ^ (1 << bit)]
        T = np.swapaxes(t, axis, 0)
    bits = T.reshape(H * (L + 1), m)
    out = np.zeros((H * (L + 1), J), np.int64)
    for j in range(J):
        for b in range(brot - 1):
            out[:, j] |= bits[:, j * (brot - 1) + b].astype(np.int64) << b
        out[:, j] |= ((out[:, j] >> (brot - 2)) & 1) << (brot - 1)
    return out


def synthetic(seed, coded=False, H=24, n_pairs=16):
    """Rows drawn from the span of a known paired plan on random triples (singles spread)."""
    rng = random.Random(seed)
    m, L = 35, 15
    blocks = ps.random_triples(L, rng)
    forms = []
    for b in blocks:
        for code in range(1, 8):
            forms.append(sum(1 << b[i] for i in range(3) if code >> i & 1))
    singles = [0, 1, 3, 7]
    coded_mask = 1 if coded else 0          # output 0 = bit 0 of word 0
    feats = []
    for _ in range(n_pairs):
        f, g = rng.sample(forms, 2)
        v = rng.getrandbits(m) & ~coded_mask
        feats += [rc.column(f, v, m), rc.column(g, v, m)]
    if coded:
        for _ in range(3):
            a, b = rng.sample(singles, 2)
            v = coded_mask | (rng.getrandbits(m) if rng.random() < 0.5 else 0)
            feats += [rc.column(1 << a, v, m), rc.column(1 << b, v, m)]
    rows = []
    for _ in range(H - 1):
        r = 0
        for f in feats:
            if rng.getrandbits(1):
                r ^= f
        rows.append(r)
    return table_from_rows(rows, H), len(feats) // 2


class PureTests(unittest.TestCase):
    def test_intersect(self):
        rng = random.Random(0)
        m = 12
        common = [rng.getrandbits(m) for _ in range(2)]
        V1 = common + [rng.getrandbits(m) for _ in range(3)]
        V2 = common + [rng.getrandbits(m) for _ in range(3)]
        D = ps.intersect(V1, V2, m)
        self.assertGreaterEqual(len(D), rc.rank(common))
        b1, b2 = rc.Basis(), rc.Basis()
        for v in V1:
            b1.add(v)
        for v in V2:
            b2.add(v)
        for d in D:
            self.assertIsNotNone(b1.solve(d))
            self.assertIsNotNone(b2.solve(d))
        self.assertEqual(len(D), rc.rank(V1) + rc.rank(V2) - rc.rank(V1 + V2))

    def test_rewrite_pad_matches_dfthc_refit(self):
        rng = np.random.default_rng(1)
        A = 1220
        d = dr.full_block_delta(rng.integers(0, 1 << 7, size=(A, 5)))
        d = d | (((d >> 6) & 1) << 7)
        changes = {3: 0b101000001001000, 20: 0b111111111111000}
        a = ps.rewrite_pad(d, changes, A)
        b = dr.apply_pad_changes(d, changes, A)
        self.assertTrue(np.array_equal(a, b))

    def test_plan_synthetic_generic(self):
        table, n_true = synthetic(5)
        A = len(table) - 4
        res = ps.plan(table, A, (), seed=0, partitions=4, keep=2, admit_kw=dict(cpu_cap=20))
        self.assertIsNotNone(res["products"])
        self.assertTrue(ps.exact_rows(res["delta"], res["terms"]))
        self.assertTrue(np.array_equal(res["delta"][:A], table[:A]))

    def test_plan_synthetic_coded_known_partition(self):
        table, n_true = synthetic(7, coded=True)
        A = len(table) - 4
        rng = random.Random(7)
        blocks = ps.random_triples(15, rng)     # the same draw as synthetic(7)
        res = ps.plan(table, A, (0,), seed=0, blocks=blocks, admit_kw=dict(cpu_cap=20))
        self.assertIsNotNone(res["products"])
        self.assertTrue(ps.exact_rows(res["delta"], res["terms"]))
        self.assertTrue(np.array_equal(res["delta"][:A], table[:A]))
        self.assertLessEqual(res["products"], n_true)


def check_pairing(P, spaces, dims, pr):
    """Every end inside W (its form space), all ends independent per pairing group, spanning dims."""
    nb = 1 if P.pair_global else len(P.blocks)
    bases = [rc.Basis() for _ in range(nb)]
    for f, g, v in pr[0]:
        for x in (f, g):
            assert rc.Basis.solve(_basis(spaces[x]), v) is not None
            assert bases[0 if P.pair_global else P.blk[x]].add(P.col(x, v))
    for f, v in pr[1]:
        assert bases[0 if P.pair_global else P.blk[f]].add(P.col(f, v))
    return [len(B) for B in bases] == list(dims)


def _basis(V):
    B = rc.Basis()
    for v in V:
        B.add(v)
    return B


class ExchangeTests(unittest.TestCase):
    """Decompose, quotient augmentation, seeded pairing, exchange moves."""

    def test_decompose_linear(self):
        rng = random.Random(3)
        B = rc.Basis()
        for _ in range(20):
            B.add(rng.getrandbits(60))
        for _ in range(50):
            a, b = rng.getrandbits(60), rng.getrandbits(60)
            ra, ca = ps.decompose(B, a)
            rb, cb = ps.decompose(B, b)
            self.assertEqual(ps.decompose(B, a ^ b), (ra ^ rb, ca ^ cb))
            x = ra
            for i in rc.set_bits(ca):
                x ^= B.vectors[i]
            self.assertEqual(x, a)
            self.assertFalse(any(bit in B.piv for bit in rc.set_bits(ra)))

    def test_augment_quotient_two_for_one(self):
        # one block, basis e0..e3: selected line (e0, e1) blocks two lines (e0+e2, e1+e3)? no: lines (e0, e2) and
        # (e1, e3) can both enter once (e0, e1) leaves
        e = [1 << i for i in range(4)]
        ends = [((0, e[0]), (0, e[1])), ((0, e[0]), (0, e[2])), ((0, e[1]), (0, e[3]))]
        sel = ps.augment_quotient(ends, [0], 1, 4, random.Random(0), 10)
        self.assertEqual(sorted(sel), [1, 2])
        # a candidate outside p's span qualifies in the quotient: (e2 + e0 + e1, e3) after removing (e0, e1)
        ends = [((0, e[0]), (0, e[1])), ((0, e[0]), (0, e[2] ^ e[0] ^ e[1])), ((0, e[1]), (0, e[3]))]
        sel = ps.augment_quotient(ends, [0], 1, 4, random.Random(0), 10)
        self.assertEqual(sorted(sel), [1, 2])

    def test_quotient_pairing_valid_and_seeded(self):
        table, n_true = synthetic(11, coded=True)
        A = len(table) - 4
        rng = random.Random(11)
        blocks = ps.random_triples(15, rng)
        P = ps.SharedPlanner(table, A, (0,))
        n, rels, sp, pr = ps.structural(P, blocks, random.Random(0))
        dims = P.dims(rels)
        pq = P.pair_lines(sp, dims, random.Random(1), aug="quotient")
        self.assertTrue(check_pairing(P, sp, dims, pq))
        self.assertLessEqual(ps.products_of(pq), n + 1)
        ps2 = P.pair_lines(sp, dims, random.Random(2), aug="quotient", seed_lines=pq[0], trials=1)
        self.assertTrue(check_pairing(P, sp, dims, ps2))
        self.assertLessEqual(ps.products_of(ps2), ps.products_of(pq))

    def test_exchange_synthetic(self):
        table, n_true = synthetic(7, coded=True)
        A = len(table) - 4
        rng = random.Random(7)
        blocks = ps.random_triples(15, rng)
        P = ps.SharedPlanner(table, A, (0,))
        rng = random.Random(0)
        n, rels, sp, pr = ps.structural(P, blocks, rng)
        n2, rels2, pr2 = ps.admit_shared(P, rels, sp, pr, rng, cpu_cap=10, prefer_more=True)
        ex = ps.exchange(P, rels2, pr2, rng, cpu_cap=15)
        nb, rb, prb = ex["best"]
        self.assertLessEqual(nb, n2)
        self.assertGreaterEqual(len(rb), len(rels2) if nb == n2 else 0)
        ok, spb = P.feasible(rb)
        self.assertTrue(ok)
        self.assertTrue(check_pairing(P, spb, P.dims(rb), prb))
        rows, d, _ = P.complete(rb)
        terms = P.terms(rows, *prb)
        self.assertTrue(ps.exact_rows(d, terms))
        self.assertTrue(np.array_equal(d[:A], table[:A]))
        self.assertLessEqual(len(terms), n_true)

    def test_plan_with_exchange(self):
        table, n_true = synthetic(5)
        A = len(table) - 4
        res = ps.plan(table, A, (), seed=0, partitions=2, keep=1, admit_kw=dict(cpu_cap=10),
                      exchange_kw=dict(cpu_cap=10))
        self.assertIsNotNone(res["products"])
        self.assertIn("exchange", res)
        self.assertTrue(ps.exact_rows(res["delta"], res["terms"]))


class Family6566Test(unittest.TestCase):
    def test_loader_4_recovered_partition(self):
        if not W6566.is_file():
            self.skipTest("6566cd395128 words missing (set DFTHC_REFIT_6566_WORDS)")
        words = np.load(W6566)
        delta, cw = dr.loader_delta(words, 4)
        blocks = [[s - 1 for s in b] for b in ([1, 6, 13], [2, 10, 15], [3, 4, 12], [5, 7, 14], [8, 9, 11])]
        # structural start (full triple annihilators) pairs perfectly at 198; shared admission reached 196 (the
        # shipped count) only in some development runs and 197 in the registered B1 runs, so only <= 198 is asserted
        res = ps.plan(delta, len(words), cw, seed=0, blocks=blocks, admit_kw=dict(cpu_cap=30))
        self.assertTrue(ps.exact_rows(res["delta"], res["terms"]))
        self.assertTrue(np.array_equal(res["delta"][:len(words)], dr.full_block_delta(delta)[:len(words)]))
        self.assertLessEqual(res["products"], 198)


if __name__ == "__main__":
    unittest.main()
