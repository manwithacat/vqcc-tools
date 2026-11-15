"""Tests for gauge_options.py and the in-place gauge change of gauge.build_circuit_gauge (dfthc_refit 1.6.0-dev).

Pure tests run anywhere (NumPy). Emitter tests need kickmix and Lera's tensorqrom.py (`DFTHC_REFIT_LERA_SRC` or the
testpaths.py). The integration test needs Dmitrii Gluzdov's family 726299a7f65b (QHDL_DATASET) and a replan output
of its coded slots 11-14 (DFTHC_REFIT_726299_PLANS) and runs in about 15 seconds.

    python -I -m unittest tools/dfthc-refit/test_gauge_options.py -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gauge  # noqa: E402
import gauge_options as go  # noqa: E402
import gauge_search as gs  # noqa: E402

from testpaths import LERA, VERIFIERS, dataset_family, env_path  # noqa: E402

G726 = dataset_family("DFTHC_REFIT_726299", "726299a7f65b")
REPLAN_726 = env_path("DFTHC_REFIT_726299_PLANS")     # a replan OUT_DIR of 726299a7f65b's coded slots 11-14

try:
    import kickmix as km  # noqa: F401
    HAVE_KM = True
except ImportError:
    HAVE_KM = False


def binomial_edges(n):
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


def gauge_table(seed, rows=200, change=None):
    """A 2-slot pi-gauge table (10 columns, gauge = f(low address bits)); `change` maps word -> 16-bit change pattern
    of slot 1 (g_j)."""
    rng = np.random.default_rng(seed)
    w = signed_table(rng, rows, 10)
    q = rng.integers(0, 2, size=(16, 5)).astype(np.uint8)
    q2 = q.copy()
    for j, h in (change or {}).items():
        q2[:, j] ^= np.asarray(h, np.uint8)
    pat = np.concatenate([q[np.arange(rows) % 16], q2[np.arange(rows) % 16]], axis=1)
    return gauge.to_family_gauge(w, pat)


class PureTests(unittest.TestCase):
    def test_basis_columns(self):
        g = np.zeros((16, 5), np.uint8)
        a = np.array([1, 0] * 8, np.uint8)
        b = np.array([1, 1, 0, 0] * 4, np.uint8)
        g[:, 1], g[:, 2], g[:, 3], g[:, 4] = a, b, a ^ b, a
        keep, comb = gauge.gf2_basis_columns(g)
        self.assertEqual(keep, [1, 2])
        self.assertEqual(comb, {3: [1, 2], 4: [1]})
        self.assertEqual(gauge.gf2_basis_columns(np.zeros((16, 5), np.uint8)), ([], {}))

    def test_width_model(self):
        self.assertEqual(gauge.gauge_width(1220), 126)
        self.assertEqual(gauge.gauge_width(1220, kept=4), 130)
        self.assertEqual(gauge.gauge_width(1220, kept=2), 128)

    def test_classification_partitions_the_group(self):
        edges = binomial_edges(20)
        g = gs.group(edges)
        Nf = g["Nf"].astype(np.int64)
        pure = go.kernel(Nf)
        for k in range(4):
            cl = go.classify(Nf, k, 20)
            d = cl["dims"]
            self.assertEqual(d["free"] + d["constrained"] + d["costed"], 20)
            W = cl["window"]
            self.assertEqual(d["costed"], gs.gf2_rank_rows([go._int(r) for r in Nf[:, W].T]))
            for f in cl["free"] + cl["constrained"]:          # tensor-neutral: no negation in the window
                self.assertFalse(((np.asarray(f) @ Nf) % 2)[W].any())
            for f in cl["free"]:                               # free: equal pi flips on both batches, word by word
                self.assertFalse(((np.asarray(f) @ go.contract_matrix(20, k, 20)) % 2).any())
            for f in pure:                                     # the pure-pi kernel is never costed
                self.assertFalse(((np.asarray(f) @ Nf) % 2)[W].any())

    def test_signed_contract_is_trivial(self):
        w = signed_table(np.random.default_rng(1), 64, 10)
        c = go.gauge_contract(w, 1)
        self.assertTrue(c["address_only"] and c["signed"])
        self.assertEqual((c["changing_words"], c["gauge_change_rank"]), ([], 0))

    def test_contract_rank(self):
        h = np.array([0, 1] * 8, np.uint8)
        w = gauge_table(2, rows=200, change={1: h, 2: h, 3: h})   # 13 high chunks: every lent bit is used
        c = go.gauge_contract(w, 1)
        self.assertEqual(c["changing_words"], [1, 2, 3])
        self.assertEqual(c["gauge_change_rank"], 1)
        self.assertEqual(c["width"]["kept_layout"] - c["width"]["inplace_layout"], 2)


@unittest.skipUnless(HAVE_KM and (LERA / "tensorqrom.py").is_file(), "needs kickmix and Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC)")
class InPlaceEmitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_gauge_options_tq", LERA / "tensorqrom.py")
        cls.tq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tq)

    def emit(self, words, k, inplace):
        import kickmix as km
        B = gauge.batches(words)
        delta = B[k] ^ (B[k - 1] if k else 0)
        terms = self.tq.per_output(self.tq.anf_tensor(delta, 8, 4))
        em = gauge.GaugeEmitter(self.tq, *gauge.slot_gauge(words, k), inplace=inplace)
        circ = em.build_circuit(delta, terms, (len(words) - 1).bit_length(), 8, 4, groups=None, scratch=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.kmb"
            circ.to_file(str(p), format="kmb")
            data = p.read_bytes()
            circ = km.Circuit.from_file(str(p), format="kmb")
        chk = gauge.native_check(km, circ, delta, gauge.check_inputs(words, k))
        return circ, chk, em, data

    def test_inplace_keeps_rank_top_bits(self):
        h = np.array([0, 1, 1, 0] * 4, np.uint8)
        h2 = np.array([1, 0] * 8, np.uint8)
        words = gauge_table(3, rows=200, change={0: h, 2: h, 3: h ^ h2, 4: h2})   # 4 changing words, rank 2
        kept, ck, ek, _ = self.emit(words, 1, "never")
        inpl, ci, ei, _ = self.emit(words, 1, "always")
        self.assertEqual(ck["status"], "PASS", ck)
        self.assertEqual(ci["status"], "PASS", ci)
        self.assertEqual(kept.num_qubits - inpl.num_qubits, 2)
        self.assertEqual(kept.operation_statistics().num_toffoli_like, inpl.operation_statistics().num_toffoli_like)
        rec = ei.record()
        self.assertEqual((rec["layout"], rec["gauge_change_rank"]), ("inplace", 2))
        self.assertEqual(sorted(rec["lent_top_bits"]), [1, 2, 4])     # keepers 0 and 3 (greedy basis)

    def test_inplace_slot0(self):
        w = gauge_table(4, rows=200)
        # slot 0's incoming words are zero: its gauge change is its own outgoing pattern
        _, chk, em, _ = self.emit(w, 0, "always")
        self.assertEqual(chk["status"], "PASS", chk)
        self.assertEqual(em.layout, "inplace")

    def test_auto_is_kept_bytes_under_the_cap(self):
        h = np.array([0, 1] * 8, np.uint8)
        words = gauge_table(5, rows=200, change={1: h, 2: h})
        _, ck, _, never = self.emit(words, 1, "never")
        _, ca, ea, auto = self.emit(words, 1, "auto")
        self.assertEqual(ck["status"], "PASS", ck)
        self.assertEqual(never, auto)
        self.assertEqual(ea.layout, "kept")

    def test_auto_switches_over_the_cap(self):
        h = np.array([0, 1] * 8, np.uint8)
        words = gauge_table(6, rows=200, change={1: h, 2: h, 3: h})
        circ_k, *_ = self.emit(words, 1, "never")
        em = gauge.GaugeEmitter(self.tq, *gauge.slot_gauge(words, 1), inplace="auto", cap=circ_k.num_qubits - 1)
        B = gauge.batches(words)
        delta = B[1] ^ B[0]
        terms = self.tq.per_output(self.tq.anf_tensor(delta, 8, 4))
        c = em.build_circuit(delta, terms, 8, 8, 4, groups=None, scratch=True)
        self.assertEqual(em.layout, "inplace")
        self.assertEqual(c.num_qubits, circ_k.num_qubits - 2)


@unittest.skipUnless(HAVE_KM and (LERA / "tensorqrom.py").is_file() and (G726 / "submission.npz").is_file()
                     and (REPLAN_726 / "work" / "plans" / "k12").is_dir() and VERIFIERS.is_dir(),
                     "needs kickmix, Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC), family 726299a7f65b "
                     "(QHDL_DATASET) and its replan plans (DFTHC_REFIT_726299_PLANS)")
class Gluzdov726299(unittest.TestCase):
    def test_slot12_plan_fits_in_place(self):
        import io
        import contextlib
        import dfthc_refit as dr
        with contextlib.redirect_stdout(io.StringIO()):
            rep = dr.gauge_options_family(G726, "12", words=REPLAN_726 / "work" / "words.npy", plans=REPLAN_726, lera_src=LERA,
                                          verifier_dir=VERIFIERS, verbose=False)
        s = rep["slots"][0]
        self.assertEqual(s["contract"]["changing_words"], [2, 3, 4])
        self.assertEqual(s["contract"]["gauge_change_rank"], 2)
        self.assertEqual(s["shipped_loader"]["qubits"], 128)
        p = s["plan"]
        # the 1.5 layout reproduces the replan's rejected 129-qubit loader byte for byte; the in-place one fits
        self.assertEqual(p["kept"]["kmb_sha256"], "6cfac65fc4dcb9b7ae3dfbbc41549424c461e0772c15101f230ab80c069cf14e")
        self.assertEqual((p["kept"]["qubits"], p["inplace"]["qubits"]), (129, 128))
        self.assertEqual(p["inplace"]["toffolis"], 276)
        self.assertEqual(p["inplace"]["native_check"], "PASS")
        self.assertTrue(any("in-place gauge change" in r for r in s["repairs"]))


if __name__ == "__main__":
    unittest.main()
