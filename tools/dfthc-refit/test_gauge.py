"""Tests for gauge.py (dfthc_refit 1.5.0-dev): convention detector, signed-gauge working form, gauge-aware
emitter, refit default targets.

Pure tests run anywhere (NumPy). Emitter tests need kickmix and Lera's tensorqrom.py (`DFTHC_REFIT_LERA_SRC`). The
integration test needs the pi-gauge family aba448aeadb5 (`DFTHC_REFIT_ABA`, or `QHDL_DATASET`, the public circuits
dataset) and runs in about a minute.

    python -I -m unittest tools/dfthc-refit/test_gauge.py -v
"""
from __future__ import annotations

import sys
import tempfile
import warnings
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gauge  # noqa: E402

from testpaths import LERA, VERIFIERS, dataset_family  # noqa: E402

ABA = dataset_family("DFTHC_REFIT_ABA", "aba448aeadb5")

try:
    import kickmix as km  # noqa: F401
    HAVE_KM = True
except ImportError:
    HAVE_KM = False


def binomial_edges(n):
    """A binomial Givens tree on n + 1 nodes, edges in breadth order of the step size (as the public families)."""
    edges, step = [], 1
    while step <= n:
        for p in range(0, n + 1):
            c = p + step
            if p < step and c <= n:
                edges.append((p, c))
        step *= 2
    return np.array(edges[:n], np.int64)


def signed_table(rng, rows, cols, brot=8):
    w = rng.integers(0, 1 << (brot - 1), size=(rows, cols))
    w = w | (((w >> (brot - 2)) & 1) << (brot - 1))
    return w.astype(np.int64)


class PureTests(unittest.TestCase):
    def test_tree_kind(self):
        self.assertEqual(gauge.tree_kind([(0, 1), (1, 2), (2, 3)]), "path")
        self.assertEqual(gauge.tree_kind([(0, 1), (0, 2), (0, 3)]), "star")
        self.assertEqual(gauge.tree_kind(binomial_edges(7)), "binomial")

    def test_canonical_round_trip(self):
        rng = np.random.default_rng(0)
        w = signed_table(rng, 32, 6)
        pat = rng.integers(0, 2, size=w.shape).astype(np.uint8)
        wp = gauge.to_family_gauge(w, pat)
        np.testing.assert_array_equal(gauge.off_gauge(wp), pat)
        np.testing.assert_array_equal(gauge.canonical_words(wp), w)
        # signed storage: canonical form stays in [-128, 128)
        ws = np.where(wp >= 128, wp - 256, wp)
        cs = gauge.canonical_words(ws)
        self.assertTrue(cs.min() >= -128 and cs.max() < 128)
        np.testing.assert_array_equal(gauge.off_gauge(cs), 0)

    def test_classes(self):
        rng = np.random.default_rng(1)
        n = 7
        E = binomial_edges(n)
        w = signed_table(rng, 48, n)
        c = gauge.word_conventions(w, E)
        self.assertEqual(c["gauge"], "signed")
        # pi on the first edge (parent 0, the only support of e_0) negates the whole vector: exact symmetry
        pat = np.zeros_like(w, dtype=np.uint8)
        pat[np.arange(48) % 16 == 3, 0] = 1
        c = gauge.word_conventions(gauge.to_family_gauge(w, pat), E)
        self.assertEqual((c["gauge"], c["exact_symmetry"], c["pattern_low_bits"], c["patterns"]), ("pi", True, True, 1))
        q = gauge.low_pattern(pat)
        self.assertEqual(int(q[3, 0]), 1)
        self.assertEqual(int(q.sum()), 1)
        # pi on a late edge alone changes u u^T: free range; a row-dependent pattern is not a low-bit function
        pat2 = np.zeros_like(pat)
        pat2[5, n - 1] = 1
        c = gauge.word_conventions(gauge.to_family_gauge(w, pat2), E)
        self.assertEqual((c["gauge"], c["exact_symmetry"], c["pattern_low_bits"]), ("free", False, False))
        self.assertIsNotNone(gauge.refusal("replan", c))
        self.assertIsNotNone(gauge.refusal("price-words", c, moves=True))
        self.assertIsNone(gauge.refusal("price-words", c))
        self.assertIsNone(gauge.refusal("refit", c))

    @unittest.skipUnless((VERIFIERS / "dfthcsos" / "verify.py").is_file(), "needs the official verifiers (QHDL_OFFICIAL_ROOT)")
    def test_decode_matches_official(self):
        sys.path.insert(0, str(VERIFIERS))
        from dfthcsos.scoreinfra import dfthc_math as dm
        rng = np.random.default_rng(2)
        n = 7
        E = binomial_edges(n)
        w = gauge.to_family_gauge(signed_table(rng, 10, n), rng.integers(0, 2, size=(10, n)))
        u = dm.reconstruct_u_vectors(w / (1 << 7), E)       # as evaluate_dfthc_angle_words calls it
        np.testing.assert_array_equal(gauge.decode(w, E, 8), u)


class TargetTests(unittest.TestCase):
    """Deliverable 6: default refit targets hold a start that is past the error_1 cap or within 1e-6 mHa of the
    error_2 limit at its own values."""

    def setUp(self):
        import importlib
        try:
            self.dr = importlib.import_module("dfthc_refit")
        except ImportError as ex:  # SciPy missing
            self.skipTest(str(ex))

    def test_polished_start_is_held(self):
        dr = self.dr
        o0 = dict(error_1=19.9998606e-3, error_2=19.9999998e-3, E_gap=dr.E_GAP_FLOOR + 7e-7)
        q0 = dict(e1=19.99986, e2=19.99999, gap=dr.E_GAP_FLOOR + 7e-7)
        t, notes = dr.default_targets(o0, q0, e1t="free", e2t=19.999, gap_margin=1e-5, gap_floor=dr.E_GAP_FLOOR)
        self.assertAlmostEqual(t[0], q0["e1"], places=12)       # error_1 held at its own value (past the cap)
        self.assertAlmostEqual(t[1], q0["e2"], places=12)       # error_2 held (within 1e-6 mHa of the limit)
        self.assertAlmostEqual(t[2], q0["gap"], places=12)      # E_gap margin held (tighter than 1e-5)
        self.assertEqual(sorted(notes), ["e1_held_past_cap", "e2_held_at_limit"])

    def test_slack_start_unchanged(self):
        dr = self.dr
        o0 = dict(error_1=19.9e-3, error_2=19.99e-3, E_gap=dr.E_GAP_FLOOR + 1e-3)
        q0 = dict(e1=19.9, e2=19.99, gap=dr.E_GAP_FLOOR + 1e-3)
        t, notes = dr.default_targets(o0, q0, e1t="free", e2t=19.999, gap_margin=1e-5, gap_floor=dr.E_GAP_FLOOR)
        self.assertIsNone(t[0])
        self.assertAlmostEqual(t[1], 19.999, places=9)
        self.assertAlmostEqual(t[2], dr.E_GAP_FLOOR + 1e-5, places=9)
        self.assertEqual(notes, [])

    def test_explicit_targets_override_the_holds(self):
        """v1.7.0: an explicitly set target is never auto-held. A start on aba448aeadb5 (error_2
        19.99999978 mHa, within 1e-6 of the limit) with --e2t 19.9999997 targets 19.9999997 (the 1.4.0 rule);
        an explicit --e1t free stays the plain inequality past the cap."""
        dr = self.dr
        o0 = dict(error_1=19.99986e-3, error_2=19.99999978e-3, E_gap=dr.E_GAP_FLOOR + 7e-7)
        q0 = dict(e1=19.99986, e2=19.9999990, gap=dr.E_GAP_FLOOR + 7e-7)
        kw = dict(gap_margin=1e-5, gap_floor=dr.E_GAP_FLOOR)
        t, notes = dr.default_targets(o0, q0, e1t="free", e2t=19.9999997, e2t_explicit=True, e1t_explicit=True,
                                      **kw)
        self.assertIsNone(t[0])                                                   # explicit 'free': no hold
        self.assertAlmostEqual(t[1], q0["e2"] + (19.9999997 - 19.99999978), places=12)   # the explicit target
        self.assertEqual(notes, ["e2_explicit_at_limit"])
        # the same values left at their defaults are held, as in 1.5.0
        t, notes = dr.default_targets(o0, q0, e1t="free", e2t=19.9999997, **kw)
        self.assertAlmostEqual(t[0], q0["e1"], places=12)
        self.assertAlmostEqual(t[1], q0["e2"], places=12)
        self.assertEqual(sorted(notes), ["e1_held_past_cap", "e2_held_at_limit"])
        # a numeric --e1t was never held; it stays the explicit target
        t, _ = dr.default_targets(o0, q0, e1t=19.9998606, e2t=19.9999997, e1t_explicit=True, e2t_explicit=True,
                                  **kw)
        self.assertAlmostEqual(t[0], q0["e1"] + (19.9998606 - 19.99986), places=12)

    def test_cli_defaults_are_unset(self):
        """The CLI and run() pass None for an unset target, so only the defaults are held."""
        import inspect
        dr = self.dr
        sig = inspect.signature(dr.run).parameters
        self.assertIsNone(sig["e1t"].default)
        self.assertIsNone(sig["e2t"].default)
        sig = inspect.signature(dr.alternate).parameters
        self.assertIsNone(sig["e1t"].default)
        self.assertIsNone(sig["e2t"].default)
        self.assertEqual((dr.E1T_DEFAULT, dr.E2T_DEFAULT), ("free", 19.999))


class BrotTests(unittest.TestCase):
    """v1.7.0: brot = the npz field, then --brot, then the circuits' word-register width; else a
    clear error. Syed Asad Ali's eff6a4863f08 ships an npz without 'brot'."""

    def setUp(self):
        import importlib
        try:
            self.dr = importlib.import_module("dfthc_refit")
        except ImportError as ex:  # SciPy missing
            self.skipTest(str(ex))

    def npz(self, d, **arrays):
        p = Path(d) / "submission.npz"
        np.savez(p, **arrays)
        return np.load(p, allow_pickle=False)

    def test_order_and_conflicts(self):
        dr = self.dr
        with tempfile.TemporaryDirectory() as t:
            z = self.npz(t, angle_words=np.zeros((2, 3), np.int64), brot=np.array(8))
            self.assertEqual(dr.resolve_brot(z), (8, "npz field 'brot'"))
            self.assertEqual(dr.resolve_brot(z, 8, 8)[0], 8)
            with self.assertRaises(RuntimeError):
                dr.resolve_brot(z, brot=7)                     # --brot contradicts the npz field
        with tempfile.TemporaryDirectory() as t:
            z = self.npz(t, angle_words=np.zeros((2, 3), np.int64))
            self.assertEqual(dr.resolve_brot(z, brot=8), (8, "--brot"))
            self.assertEqual(dr.resolve_brot(z, circuit_brot=8), (8, "the circuits' word-register width"))
            with self.assertRaises(RuntimeError):
                dr.resolve_brot(z, brot=7, circuit_brot=8)
            with self.assertRaisesRegex(RuntimeError, "--brot"):
                dr.resolve_brot(z)

    def test_no_kickmix_fallback_without_brot_field(self):
        """recover_words' npz fallback used to raise KeyError('brot'); now it takes --brot or says to pass it."""
        dr = self.dr
        orig = dr.find_kickmix
        dr.find_kickmix = lambda *a, **k: None
        try:
            with tempfile.TemporaryDirectory() as t, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.npz(t, angle_words=np.array([[1, 200, 255], [0, 3, 128]], np.int64))
                with self.assertRaisesRegex(RuntimeError, "--brot"):
                    dr.recover_words(Path(t), Path(t))
                wr = dr.recover_words(Path(t), Path(t), brot=8)
                self.assertEqual((wr["brot"], wr["brot_source"]), (8, "--brot"))
                with self.assertRaises(RuntimeError):            # words do not fit in 7 bits
                    dr.recover_words(Path(t), Path(t), brot=7)
        finally:
            dr.find_kickmix = orig


@unittest.skipUnless(HAVE_KM and (LERA / "tensorqrom.py").is_file(), "needs kickmix and Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC)")
class EmitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_gauge_tq", LERA / "tensorqrom.py")
        cls.tq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tq)

    def table(self, seed, change_word=None, rows=40):
        """A 2-slot pi-gauge table (`rows` rows, 10 columns): gauge a function of the low address bits; slot 1 keeps
        the incoming gauge on every word except `change_word`."""
        rng = np.random.default_rng(seed)
        w = signed_table(rng, rows, 10)
        q = rng.integers(0, 2, size=(16, 5)).astype(np.uint8)
        q2 = q.copy()
        if change_word is not None:
            q2[:, change_word] ^= rng.integers(0, 2, size=16).astype(np.uint8)
            q2[0, change_word] ^= 1 if not (q2[:, change_word] ^ q[:, change_word]).any() else 0
        pat = np.concatenate([q[np.arange(rows) % 16], q2[np.arange(rows) % 16]], axis=1)
        return gauge.to_family_gauge(w, pat)

    def emit_slot(self, words, k, em_cls=gauge.GaugeEmitter):
        import kickmix as km
        B = gauge.batches(words)
        delta = B[k] ^ (B[k - 1] if k else 0)
        C = self.tq.anf_tensor(delta, 8, 4)
        terms = self.tq.per_output(C)
        pin, pout = gauge.slot_gauge(words, k)
        em = em_cls(self.tq, pin, pout)
        circ = em.build_circuit(delta, terms, (len(words) - 1).bit_length(), 8, 4, groups=None, scratch=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.kmb"
            circ.to_file(str(p), format="kmb")
            circ = km.Circuit.from_file(str(p), format="kmb")
            data = p.read_bytes()
        return circ, delta, terms, em, data

    def test_pi_gauge_slot_passes(self):
        import kickmix as km
        words = self.table(3)
        circ, delta, terms, em, _ = self.emit_slot(words, 1)
        self.assertFalse(em.signed)
        self.assertEqual(em.record()["lent_top_bits"], [0, 1, 2, 3, 4])
        chk = gauge.native_check(km, circ, delta, gauge.check_inputs(words, 1))
        self.assertEqual(chk["status"], "PASS", chk)
        self.assertEqual(chk["toffolis_per_application"][0], self.tq.cost(terms, 16, 3))
        # Lera's emitter on the same delta: either refuses or leaves a phase / wrong words on the gauged inputs
        try:
            lc = self.tq.build_circuit(delta, terms, 6, 8, 4, scratch=True)
            self.assertEqual(gauge.native_check(km, lc, delta, gauge.check_inputs(words, 1))["status"], "FAIL")
        except AssertionError:
            pass

    def test_gauge_change_keeps_top_bit(self):
        import kickmix as km
        words = self.table(4, change_word=2, rows=200)      # 13 high chunks: the high cache uses every lent bit
        circ, delta, terms, em, _ = self.emit_slot(words, 1)
        self.assertEqual(em.record()["lent_top_bits"], [0, 1, 3, 4])
        chk = gauge.native_check(km, circ, delta, gauge.check_inputs(words, 1))
        self.assertEqual(chk["status"], "PASS", chk)
        # one qubit more than a slot that lends every top bit
        circ2, *_ = self.emit_slot(self.table(4, rows=200), 1)
        self.assertEqual(circ.num_qubits, circ2.num_qubits + 1)

    def test_slot0_from_zero(self):
        import kickmix as km
        words = self.table(5)
        circ, delta, terms, em, _ = self.emit_slot(words, 0)
        chk = gauge.native_check(km, circ, delta, gauge.check_inputs(words, 0))
        self.assertEqual(chk["status"], "PASS", chk)
        cls = gauge.loader_classes  # the classifier on a two-slot family of emitted loaders
        circ1, *_ = self.emit_slot(words, 1)
        with tempfile.TemporaryDirectory() as td:
            ps = []
            for k, c in enumerate((circ, circ1)):
                p = Path(td) / f"angle_k_{k}.kmb"
                c.to_file(str(p), format="kmb")
                ps.append(p)
            got = cls(km, ps, words)
        self.assertEqual([s["loader"] for s in got], ["xor-in", "xor-in"])

    def test_recover_reads_gauge_loader(self):
        """recover.py with the incoming gauge reads a gauge-emitted loader back: same terms, function exact."""
        import kickmix as km  # noqa: F401
        import recover as rc
        words = self.table(7, rows=200)
        circ, delta, terms, em, data = self.emit_slot(words, 1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "angle_k_1.kmb"
            p.write_bytes(data)
            pin, _ = gauge.slot_gauge(words, 1)
            rep, rec, _, _ = rc.recover_loader(km, p, words, 1, gauge_in=pin)
            self.assertTrue(rep["recovered_equals_delta_on_rows"])
            self.assertEqual(sorted(map(tuple, rec["terms"])), sorted(terms))
            try:                                          # read as signed gauge: refused or a wrong function
                bad, *_ = rc.recover_loader(km, p, words, 1)
                self.assertFalse(bad["recovered_equals_delta_on_rows"] and bad["products"] == len(terms))
            except AssertionError:
                pass

    def test_signed_slot_is_leras_bytes(self):
        rng = np.random.default_rng(6)
        words = signed_table(rng, 40, 10)
        circ, delta, terms, em, data = self.emit_slot(words, 1)
        self.assertTrue(em.signed)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "b.kmb"
            self.tq.build_circuit(delta, terms, 6, 8, 4, groups=None, scratch=True).to_file(str(p), format="kmb")
            self.assertEqual(p.read_bytes(), data)


def monomial_pattern(m, lowbits=4):
    """(16,) 0/1: the low monomial with bit mask m evaluated on every low address."""
    return np.array([int((a & m) == m) for a in range(1 << lowbits)], np.uint8)


class FreeSourceModelTests(unittest.TestCase):
    """v1.8.0-dev: free low sources and the low-cache floor (pure NumPy)."""

    def test_free_monomials(self):
        pats = np.stack([monomial_pattern(m) for m in (3, 5, 6, 9, 10)], axis=1)
        self.assertEqual(gauge.free_monomials(pats, pats), {0: 3, 1: 5, 2: 6, 3: 9, 4: 10})
        self.assertEqual(gauge.low_cache_floor(pats, pats), 6)
        out = pats.copy()
        out[:, 1] = 0                                   # word 1's gauge changes: not a source
        self.assertEqual(gauge.free_monomials(pats, out), {0: 3, 2: 6, 3: 9, 4: 10})
        self.assertEqual(gauge.low_cache_floor(pats, out), 7)

    def test_affine_part_duplicates_and_mixed_terms(self):
        x0 = monomial_pattern(1)
        p = np.stack([monomial_pattern(6) ^ x0,          # x1x2 + x0: still the source of x1x2
                      monomial_pattern(6),               # duplicate monomial: one source only
                      monomial_pattern(3) ^ monomial_pattern(12),   # two nonlinear terms: not a source
                      np.zeros(16, np.uint8), monomial_pattern(15)], axis=1)
        self.assertEqual(gauge.free_monomials(p, p), {0: 6, 4: 15})
        self.assertEqual(gauge.low_cache_floor(p, p), 9)


@unittest.skipUnless(HAVE_KM and (LERA / "tensorqrom.py").is_file(), "needs kickmix and Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC)")
class FreeSourceEmitterTests(unittest.TestCase):
    """The free-source emitter (v1.8.0-dev) on a table whose gauge patterns are low monomials."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_gauge_tq_free", LERA / "tensorqrom.py")
        cls.tq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tq)

    emit_slot = EmitterTests.emit_slot

    def mono_table(self, seed, monos=(3, 5, 6, 9, 10), change_word=None, rows=200):
        rng = np.random.default_rng(seed)
        w = signed_table(rng, rows, 10)
        q = np.stack([monomial_pattern(m) for m in monos], axis=1)
        q2 = q.copy()
        if change_word is not None:
            q2[:, change_word] = 0
        pat = np.concatenate([q[np.arange(rows) % 16], q2[np.arange(rows) % 16]], axis=1)
        return gauge.to_family_gauge(w, pat)

    def free_emitter(self, tq, pin, pout):
        return gauge.GaugeEmitter(tq, pin, pout, free_sources=True)

    def test_free_sources_save_low_caches(self):
        import kickmix as km
        words = self.mono_table(11)
        base, delta, terms, _, base_bytes = self.emit_slot(words, 1)
        circ, _, _, em, _ = self.emit_slot(words, 1, em_cls=self.free_emitter)
        chk = gauge.native_check(km, circ, delta, gauge.check_inputs(words, 1))
        self.assertEqual(chk["status"], "PASS", chk)
        self.assertEqual(em.record()["free_monomials"], {0: 3, 1: 5, 2: 6, 3: 9, 4: 10})
        nb = base.operation_statistics().num_toffoli_like
        self.assertEqual(circ.operation_statistics().num_toffoli_like, nb - 5)
        self.assertEqual(chk["toffolis_per_application"][0], nb - 5)
        self.assertLessEqual(circ.num_qubits, base.num_qubits + 5)
        # off by default: the 1.7 emission is unchanged byte for byte
        _, _, _, _, again = self.emit_slot(words, 1)
        self.assertEqual(again, base_bytes)

    def test_changing_word_is_not_a_source(self):
        import kickmix as km
        words = self.mono_table(12, change_word=1)
        circ, delta, terms, em, _ = self.emit_slot(words, 1, em_cls=self.free_emitter)
        self.assertEqual(em.record()["free_monomials"], {0: 3, 2: 6, 3: 9, 4: 10})
        chk = gauge.native_check(km, circ, delta, gauge.check_inputs(words, 1))
        self.assertEqual(chk["status"], "PASS", chk)
        base, *_ = self.emit_slot(words, 1)
        self.assertEqual(circ.operation_statistics().num_toffoli_like,
                         base.operation_statistics().num_toffoli_like - 4)

    def test_recover_reads_free_source_loader(self):
        """recover.py reads the free-source loader back (same terms, 6 low caches) and reemit rebuilds it."""
        import kickmix as km  # noqa: F401
        import recover as rc
        words = self.mono_table(13)
        circ, delta, terms, em, data = self.emit_slot(words, 1, em_cls=self.free_emitter)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "angle_k_1.kmb"
            p.write_bytes(data)
            pin, pout = gauge.slot_gauge(words, 1)
            rep, rec, rec_x, d = rc.recover_loader(km, p, words, 1, gauge_in=pin)
            self.assertTrue(rep["recovered_equals_delta_on_rows"])
            self.assertEqual(rep["low_caches"], 6)
            self.assertEqual(sorted(map(tuple, rec["terms"])), sorted(terms))
            r = rc.reemit(km, gauge.GaugeEmitter(self.tq, pin, pout, free_sources=True), rec, rec_x, d,
                          Path(td) / "re.kmb", check_inputs=gauge.check_inputs(words, 1))
            self.assertEqual(r["native_check"]["status"], "PASS", r)
            self.assertEqual(r["num_toffoli_like"], rep["num_toffoli_like"])


@unittest.skipUnless(HAVE_KM and (ABA / "submission.npz").is_file(), "needs kickmix and the aba448aeadb5 family (QHDL_DATASET)")
class AbaIntegration(unittest.TestCase):
    def test_conventions(self):
        import kickmix as km
        z = np.load(ABA / "submission.npz")
        words = np.asarray(z["angle_words"], np.int64)
        loaders = [ABA / f"angle_k_{k}.kmb" for k in range(15)]
        rep = gauge.conventions(km, loaders, words, z["tree_edges"], int(z["brot"]))
        self.assertEqual((rep["gauge"], rep["exact_symmetry"], rep["pattern_low_bits"], rep["patterns"], rep["tree"]),
                         ("pi", True, True, 8, "binomial"))
        self.assertEqual(rep["off_gauge_entries"], 18867)
        # Atakan Aldemir's loaders are XOR oracles on words of the incoming gauge (not chained); on signed-gauge
        # words (e.g. zero) slots 1-14 are wrong on 533-685 rows, which a zero-input test reads as "chained"
        self.assertEqual(rep["loader_summary"]["xor-in"], list(range(15)))
        self.assertEqual([s["k"] for s in rep["slots"] if s["needs_incoming_gauge"]], list(range(1, 15)))
        self.assertEqual(rep["slots"][4]["runs"]["random_signed"]["rows_wrong"], 685)

    def test_recover_slot4_with_gauge(self):
        import kickmix as km
        import recover as rc
        z = np.load(ABA / "submission.npz")
        words = np.asarray(z["angle_words"], np.int64)
        pin, _ = gauge.slot_gauge(words, 4)
        rep, *_ = rc.recover_loader(km, ABA / "angle_k_4.kmb", words, 4, gauge_in=pin)
        self.assertTrue(rep["recovered_equals_delta_on_rows"])
        self.assertEqual((rep["products"], rep["caches"], rep["num_toffoli_like"]), (200, 76, 276))


if __name__ == "__main__":
    unittest.main()
