# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Tests for chained_and.py. Optional real-target test: set CHAINED_AND_TARGET to a target JSON (ybits, outputs,
rows[{care, tt}]) for the coded bit-0 block of a loader (loader 4 of 6566cd395128 expects rank-2 bound 10, general
bound 6)."""
import os
import random
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chained_and as ca  # noqa: E402


def random_linear_target(n, m, rows, rng, partial=0):
    full = (1 << (1 << n)) - 1
    out = []
    for r in range(rows):
        tts = [ca.lin_tt(rng.getrandbits(n), n) ^ (full if rng.random() < 0.5 else 0) for _ in range(m)]
        out.append((full, tts))
    for _ in range(partial):
        care = rng.getrandbits(1 << n) | 1
        out.append((care, [rng.getrandbits(1 << n) for _ in range(m)]))
    return ca.Target(n=n, m=m, rows=out)


class TestChainedAnd(unittest.TestCase):
    def test_copy_gate_separation(self):
        c = ca.copy_gate_example()
        self.assertEqual(c.k, 2)
        self.assertEqual(c.depth(), 2)
        f = ca.linear_fiber_dim(c)
        self.assertEqual(f["dim"], 5)
        settings = []
        for s in range(1 << (2 * c.k)):
            sig, tau = ca.setting_bits(s, c.k)
            settings.append((sig, tau))
        T = ca.image_target(c, settings)
        # every setting gives outputs linear in y (the whole image is one fibre)
        for _, tts in T.rows:
            for t in tts:
                self.assertEqual(ca.split_linear(t, 3)[1], 0)
        self.assertEqual(ca.rank2_lower_bound(T)["bound"], 3)
        self.assertTrue(ca.verify(c, T)["all_pass"])
        self.assertTrue(ca.verify(c, T, rowbits=settings)["all_pass"])
        self.assertLessEqual(ca.general_lower_bound(T)["bound"], 2)

    def test_depth1_fibre_at_most_2k(self):
        rng = random.Random(1)
        for _ in range(30):
            n, m, k = 3, 2, rng.choice([1, 2])
            gates = [(rng.getrandbits(8) & ~1, 0, rng.getrandbits(8) & ~1, 0) for _ in range(k)]
            c = ca.Circuit(n=n, m=m, gates=gates, e=[rng.getrandbits(k) for _ in range(m)],
                           P=[rng.getrandbits(8) & ~1 for _ in range(m)])
            self.assertLessEqual(ca.linear_fiber_dim(c)["dim"], 2 * k)

    def test_rank2_construction_verifies(self):
        rng = random.Random(2)
        for n, m, rows in [(3, 2, 12), (4, 3, 30), (4, 5, 25)]:
            T = random_linear_target(n, m, rows, rng)
            c, rb = ca.construct_rank2(T)
            v = ca.verify(c, T, rowbits=rb)
            self.assertTrue(v["all_pass"])
            self.assertEqual(c.depth(), 1)
            self.assertGreaterEqual(c.k, ca.rank2_lower_bound(T)["bound"])

    def test_verify_rejects(self):
        rng = random.Random(3)
        T = random_linear_target(3, 2, 10, rng)
        c, rb = ca.construct_rank2(T)
        care, tts = T.rows[4]
        bad = ca.Target(n=3, m=2, rows=list(T.rows))
        bad.rows[4] = (care, [tts[0] ^ 0b00000110, tts[1]])   # not a constant shift
        v = ca.verify(c, bad, rowbits=rb)
        self.assertFalse(v["all_pass"])
        self.assertEqual(v["failed_rows"], [4])
        # exhaustive search also fails there: the bad row is not linear-in-y, depth-1 outputs with these P are
        self.assertIn(4, ca.verify(c, bad)["failed_rows"])

    def test_dont_cares(self):
        rng = random.Random(4)
        T = random_linear_target(4, 2, 20, rng, partial=3)
        c, rb = ca.construct_rank2(T)
        v = ca.verify(c, T)
        # partial rows are random on their care points; the check must at least run exhaustively and be exact
        self.assertEqual(v["rows"], 23)
        for r, ok in enumerate(v["rowbits"]):
            if ok is not None:
                care, tts = T.rows[r]
                self.assertTrue(ca.row_met(c.evaluate(*ok), care, tts, T.full))

    def test_bounds_inner_product(self):
        n = 4
        full = (1 << 16) - 1
        T = ca.Target(n=n, m=1, rows=[(full, [ca.lin_tt(x, n)]) for x in range(16)])
        self.assertEqual(ca.rank2_lower_bound(T), dict(affine_rank=4, bound=2))
        g = ca.general_lower_bound(T)
        self.assertEqual(g["output_dim"], 1)
        self.assertEqual(g["bound"], 2)

    def test_cnf_encoding_accepts_known_circuit(self):
        rng = random.Random(5)
        T = random_linear_target(2, 2, 5, rng)
        c, rb = ca.construct_rank2(T)
        rowbits = dict(enumerate(rb))
        for depth in (1, None):
            cnf = ca.encode_cnf(T, c.k, depth=depth)
            model = ca.assignment_from(cnf, c, T, rowbits)
            self.assertIsNotNone(model)
            self.assertTrue(ca.check_assignment(cnf, model))
            c2, rb2 = ca.decode_model(cnf, model, T, c.k)
            self.assertTrue(ca.verify(c2, T, rowbits=[rb2[r] for r in range(len(T.rows))])["all_pass"])
        # a chained circuit through the full-chain encoding
        cg = ca.copy_gate_example()
        settings = [ca.setting_bits(s, 2) for s in range(16)]
        Tc = ca.image_target(cg, settings[:6])
        cnf = ca.encode_cnf(Tc, 2)
        self.assertIsNotNone(ca.assignment_from(cnf, cg, Tc, dict(enumerate(settings[:6]))))
        # depth 1 cannot express the chain
        cnf1 = ca.encode_cnf(Tc, 2, depth=1)
        self.assertIsNone(ca.assignment_from(cnf1, cg, Tc, dict(enumerate(settings[:6]))))

    def test_cnf_rejects_wrong_rowbits(self):
        rng = random.Random(6)
        T = random_linear_target(2, 1, 4, rng)
        c, rb = ca.construct_rank2(T)
        cnf = ca.encode_cnf(T, c.k, depth=1)
        good = dict(enumerate(rb))
        self.assertIsNotNone(ca.assignment_from(cnf, c, T, good))
        for r in range(len(rb)):
            for s in range(1 << (2 * c.k)):
                bits = ca.setting_bits(s, c.k)
                if ca.row_met(c.evaluate(*bits), *T.rows[r], T.full):
                    continue
                bad = dict(good)
                bad[r] = bits
                self.assertIsNone(ca.assignment_from(cnf, c, T, bad))


    # ---- reductions R1-R7 (0.2.0)

    def random_chained_circuit(self, rng, n, m, k, chained):
        N = 1 << n
        gates = []
        for j in range(k):
            a, b = rng.getrandbits(N) & ~1, rng.getrandbits(N) & ~1
            cam = cbm = 0
            if j >= k - chained and j > 0:
                while not (cam | cbm):
                    cam, cbm = rng.getrandbits(j), rng.getrandbits(j)
            gates.append((a, cam, b, cbm))
        return ca.Circuit(n=n, m=m, gates=gates, e=[rng.getrandbits(k) for _ in range(m)],
                          P=[rng.getrandbits(N) & ~1 for _ in range(m)])

    def test_r1_mixed_matrix_and_plain_gate_rank(self):
        rng = random.Random(7)
        T = random_linear_target(4, 3, 30, rng)
        c, rb = ca.construct_rank2(T)
        M = ca.mixed_matrix(T)
        self.assertEqual(ca.rank(M), ca.rank2_lower_bound(T)["affine_rank"])
        self.assertEqual(ca.mixed_columns(T)["rank"], ca.rank(M))
        parts = ca.gate_mixed_parts(c, rb)
        self.assertTrue(all(ca.column_rank(pt) <= 2 for pt in parts))
        for w in range(T.m):                      # quotient identity: sum of gate mixed parts = target mixed part
            for zi, v in enumerate(M):
                gw = 0
                for j in range(c.k):
                    if c.e[w] >> j & 1:
                        gw ^= parts[j][zi]
                self.assertEqual(gw, (v >> (w * T.N)) & T.full)

    def test_r2_row_and_column_collapse(self):
        rng = random.Random(8)
        T = random_linear_target(3, 2, 8, rng)
        full = T.full
        rows = list(T.rows)
        rows.append((full, [rows[2][1][0] ^ full, rows[2][1][1]]))          # row 2 up to a constant
        rows.append((0b00001011, rows[5][1]))                               # partial row dominated by row 5
        T2 = ca.Target(n=3, m=2, rows=rows)
        classes = ca.row_classes(T2)
        self.assertEqual(len(classes), len(ca.row_classes(T)))      # the two added rows join existing classes
        Tr, cl = ca.reduce_rows(T2)
        c, rb = ca.construct_rank2(Tr)
        v = ca.verify(c, Tr)
        self.assertTrue(v["all_pass"])
        lifted = ca.lift_rowbits(v["rowbits"], cl, len(T2.rows))
        self.assertTrue(ca.verify(c, T2, rowbits=lifted)["all_pass"])
        # columns: make y = 5 a copy of y = 3 up to per-output constants on every row
        rows3 = []
        for care, tts in T.rows:
            new = []
            for w, t in enumerate(tts):
                bit = ((t >> 3) & 1) ^ (w & 1)
                new.append((t & ~(1 << 5)) | (bit << 5))
            rows3.append((care, new))
        T3 = ca.Target(n=3, m=2, rows=rows3)
        c, rb = ca.construct_rank2(T)
        merges = ca.column_merges(T3)
        self.assertEqual([(yk, yd, cs) for yk, yd, cs in merges], [(3, 5, [0, 1])])
        Tc, mg = ca.reduce_columns(T3)
        self.assertTrue(all(not (care >> 5 & 1) for care, _ in Tc.rows))
        # the construction for T solves the reduced instance (column 5 is a don't-care there); its lift solves T3
        self.assertTrue(ca.verify(c, Tc, rowbits=rb)["all_pass"])
        lifted = ca.lift_columns(c, mg)
        self.assertTrue(ca.verify(lifted, T3, rowbits=rb)["all_pass"])

    def test_r3_output_basis(self):
        rng = random.Random(9)
        T = random_linear_target(3, 3, 12, rng)
        c, rb = ca.construct_rank2(T)
        L = ca.output_basis(T)
        T3 = ca.apply_output_map(T, L)
        c3 = ca.map_circuit_outputs(c, L)
        self.assertTrue(ca.verify(c3, T3, rowbits=rb)["all_pass"])
        self.assertEqual(ca.rank(ca.mixed_matrix(T3)), ca.rank(ca.mixed_matrix(T)))
        back = ca.map_circuit_outputs(c3, ca.invert_output_map(L))
        self.assertEqual((back.e, back.P), (c.e, c.P))

    def test_r4_r5_canonical_form_verifies_and_is_accepted(self):
        rng = random.Random(10)
        for trial in range(60):
            n, m = rng.choice([2, 3]), rng.choice([1, 2])
            k = rng.choice([1, 2, 3, 4])
            I = rng.randint(0, max(0, k - 1))
            c = self.random_chained_circuit(rng, n, m, k, I)
            settings = [(rng.getrandbits(k), rng.getrandbits(k)) for _ in range(rng.randint(1, 6))]
            T = ca.image_target(c, settings)
            cc, rb = ca.canonicalise(c, settings)
            self.assertTrue(ca.verify(cc, T, rowbits=rb)["all_pass"])
            I2 = sum(1 for g in cc.gates if g[1] | g[3])
            self.assertEqual(I2, sum(1 for g in c.gates if g[1] | g[3]))
            cnf = ca.encode_shape(T, k, I2, gauge=True, lex=True, force_chain=True)
            model = ca.assignment_from(cnf, cc, T, dict(enumerate(rb)))
            self.assertIsNotNone(model)
            c2, rb2 = ca.decode_model(cnf, model, T, k)
            self.assertEqual((c2.gates, c2.e, c2.P), (cc.gates, cc.e, cc.P))
            # gauge clauses reject a swapped operand pair when the keys differ
            a, cam, b, cbm = cc.gates[0]
            if (cam, a) != (cbm, b) and I2 == 0 or (I2 and k - I2 > 0 and (cam, a) != (cbm, b)):
                bad = ca.Circuit(n=n, m=m, gates=[(b, cbm, a, cam)] + list(cc.gates[1:]), e=cc.e, P=cc.P)
                swapped = [(((s & ~1) | (t & 1)), ((t & ~1) | (s & 1))) for s, t in rb]
                self.assertTrue(ca.verify(bad, T, rowbits=swapped)["all_pass"])
                self.assertIsNone(ca.assignment_from(cnf, bad, T, dict(enumerate(swapped))))

    def test_r5_lex_rejects_out_of_order_plain_gates(self):
        rng = random.Random(12)
        T = random_linear_target(3, 2, 10, rng)
        c, rb = ca.construct_rank2(T)
        cc, rbc = ca.canonicalise(c, rb)
        cnf = ca.encode_shape(T, cc.k, 0)
        self.assertIsNotNone(ca.assignment_from(cnf, cc, T, dict(enumerate(rbc))))
        keys = [(g[0] >> 1, g[2] >> 1) for g in cc.gates]
        i = next(i for i in range(cc.k - 1) if keys[i] < keys[i + 1])
        perm = list(range(cc.k))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        bad = ca.reorder_gates(cc, perm)
        self.assertTrue(ca.verify(bad, T, rowbits=ca.reorder_rowbits(rbc, perm))["all_pass"])
        self.assertIsNone(ca.assignment_from(cnf, bad, T, dict(enumerate(ca.reorder_rowbits(rbc, perm)))))

    def test_lex_gadget_exact(self):
        # X >= Y lexicographically is satisfiable exactly when it holds, for all 3-bit pairs
        for X in range(8):
            for Y in range(8):
                c = ca.CNF()
                xs = [c.var() for _ in range(3)]
                ys = [c.var() for _ in range(3)]
                for i in range(3):
                    c.const(xs[i], bool(X >> (2 - i) & 1))
                    c.const(ys[i], bool(Y >> (2 - i) & 1))
                ca._lex_ge(c, xs, ys)
                self.assertEqual(self.brute_sat(c), X >= Y)

    @staticmethod
    def brute_sat(c):
        import itertools
        for bits in itertools.product([False, True], repeat=c.nv):
            true = {i + 1 for i, b in enumerate(bits) if b}
            if ca.check_assignment(c, true):
                return True
        return False

    def test_r6_shapes(self):
        sh = ca.shapes(9, 20)
        self.assertEqual([i for i, s in sh.items() if not s["rank_feasible"]], [0])
        self.assertEqual(sorted(sh), list(range(9)))
        self.assertTrue(ca.shapes(9, 18)[0]["rank_feasible"])

    def test_r7_spanning_rows(self):
        rng = random.Random(13)
        T = random_linear_target(4, 2, 25, rng, partial=1)
        sp = ca.spanning_rows(T)
        self.assertIn(25, sp)
        M = ca.mixed_matrix(T)
        full_idx = [r for r, (care, _) in enumerate(T.rows) if care == T.full]
        sub = [M[full_idx.index(r)] for r in sp if r in full_idx]
        self.assertEqual(ca.rank(sub), ca.rank(M))

    @unittest.skipUnless(shutil.which("kissat") or shutil.which("cadical"), "no SAT solver on PATH")
    def test_r7_cegar_with_solver(self):
        solver = "kissat" if shutil.which("kissat") else "cadical"
        import tempfile
        cg = ca.copy_gate_example()
        settings = [ca.setting_bits(s, 2) for s in range(16)]
        T = ca.image_target(cg, settings[:10])
        with tempfile.TemporaryDirectory() as d:
            res = ca.cegar(T, 2, 1, d, solver=solver, timeout_s=60, log=lambda m: None)
            self.assertEqual(res["status"], "SAT")
            self.assertTrue(res["verify"]["all_pass"])
            self.assertEqual(ca.Circuit.from_dict(res["circuit"]).depth(), 2)
            res0 = ca.cegar(T, 2, 0, d + "/u", solver=solver, timeout_s=60, log=lambda m: None)
            self.assertEqual(res0["status"], "UNSAT")      # rank-2 products need 3 here


    def test_depth1_exact_and_planted_control(self):
        rng = random.Random(14)
        # tight depth-1 case: a random 3-product circuit's image of rank 6 is recovered exactly
        gates = [(ca.lin_tt(1, 3), 0, ca.lin_tt(2, 3), 0), (ca.lin_tt(4, 3), 0, ca.lin_tt(3, 3), 0),
                 (ca.lin_tt(5, 3), 0, ca.lin_tt(6, 3), 0)]
        c = ca.Circuit(n=3, m=2, gates=gates, e=[0b011, 0b110], P=[0, 0])
        settings = [(rng.getrandbits(3), rng.getrandbits(3)) for _ in range(20)]
        T = ca.image_target(c, settings)
        if ca.rank(ca.mixed_matrix(T)) == 6:
            r = ca.depth1_exact(T)
            self.assertEqual(r["status"], "SAT")
            self.assertTrue(r["verify"]["all_pass"])
            self.assertEqual(ca.Circuit.from_dict(r["circuit"]).depth(), 1)
        # the copy-gate image has rank 5 with 2 gates: 2k = 4 < 5 is refused by rank, 3 products exist
        cg = ca.copy_gate_example()
        Tc = ca.image_target(cg, [ca.setting_bits(s, 2) for s in range(16)])
        self.assertEqual(ca.depth1_exact(Tc, k=2)["status"], "UNSAT")
        # planted loader-4 mirror: verified, affine rows, rank 20 with five rank-4 outputs, accepted by its shape
        T4, c4, rb4 = ca.plant_linear_chained(4, 5, 40, random.Random(15), partial_care=15)
        self.assertTrue(ca.verify(c4, T4, rowbits=rb4)["all_pass"])
        self.assertEqual((c4.k, c4.depth(), sum(1 for g in c4.gates if g[1] | g[3])), (9, 2, 2))
        self.assertTrue(all(ca.split_linear(t, 4)[1] == 0 for care, tts in T4.rows if care == T4.full for t in tts))
        self.assertEqual(ca.rank(ca.mixed_matrix(T4)), 20)
        self.assertEqual([ca.combo_rank(T4, 1 << w) for w in range(5)], [4] * 5)
        cc, rbc = ca.canonicalise(c4, rb4)
        cnf = ca.encode_shape(T4, 9, 2)
        self.assertIsNotNone(ca.assignment_from(cnf, cc, T4, dict(enumerate(rbc))))

    @unittest.skipUnless(os.environ.get("CHAINED_AND_TARGET"), "set CHAINED_AND_TARGET to a loader target JSON")
    def test_real_target(self):
        T = ca.Target.from_json(os.environ["CHAINED_AND_TARGET"])
        r2 = ca.rank2_lower_bound(T)
        g = ca.general_lower_bound(T)
        c, rb = ca.construct_rank2(T)
        v = ca.verify(c, T)
        self.assertTrue(v["all_pass"])
        self.assertGreaterEqual(c.k, r2["bound"])
        self.assertGreaterEqual(r2["bound"], g["bound"])
        if T.meta.get("k") == 4:
            self.assertEqual((r2["bound"], g["bound"], c.k), (10, 6, 10))


if __name__ == "__main__":
    unittest.main()
