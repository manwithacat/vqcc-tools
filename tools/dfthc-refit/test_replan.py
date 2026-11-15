# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for `dfthc_refit.py replan` (loader re-planning, v1.3.0-dev).

Pure tests (cache-count model, family hash, job order, plan parsing) always run. The integration tests need a kickmix
build (run with the kickmix interpreter), the qrom-plan binary, Lera's tensorqrom.py, the competition verifiers and
the a56ec354e8b8 family from the public circuits dataset; each is located through an environment variable (see
testpaths.py), and the tests skip cleanly when one is missing:

    DFTHC_REFIT_QROM_PLAN     qrom-plan binary (default: `qrom-plan` on PATH)
    DFTHC_REFIT_LERA_SRC      directory with tensorqrom.py
    DFTHC_REFIT_VERIFIER_DIR  competition verifiers (default $QHDL_OFFICIAL_ROOT/verifiers)
    DFTHC_REFIT_STAGE         public circuits dataset (default $QHDL_DATASET)
    DFTHC_REFIT_REF_SUMS      optional: inputs.sha256 of the expected replan output
    DFTHC_REFIT_KERNEL_SRC    directory with qwertykeypadapple's kernel_search.py; with DFTHC_REFIT_A344 (the
                              a344f422f2e0 dfthcsos directory) for the kernel backend test
    DFTHC_REFIT_F429          the f4291619f2fd dfthcsos directory; with DFTHC_REFIT_F429_PADROW_PLAN (a saved padded-row plan
                              directory: arm_b/plans/armb_k4_padext_w0_s0.json, optional family/dfthcsos) for the
                              padded-row test

    "$QHDL_KICKMIX_PYTHON" -I test_replan.py -v
"""
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfthc_refit as dr  # noqa: E402
from testpaths import DATASET, LERA, VERIFIERS, env_path, unset  # noqa: E402

_QP = shutil.which("qrom-plan")
PLANNER = env_path("DFTHC_REFIT_QROM_PLAN", Path(_QP) if _QP else unset("DFTHC_REFIT_QROM_PLAN"))
PLANNER_SHA = "8224ed2bfb3f44b0cebc538435b2d1ec3dae9dd3a476636e6a23835fab1d3bdb"   # the reference qrom-plan build
STAGE = env_path("DFTHC_REFIT_STAGE", DATASET)
REF_SUMS = env_path("DFTHC_REFIT_REF_SUMS")
FAMILY = "a56ec354e8b8"
FAMILY_SHA = "a56ec354e8b802d5430099766537a281248d9fba6f8d62907067c8886636eb67"
REF_FAMILY_SHA = "c6d391fdc236d02a3c9dc3b6747b52f826bbadce0034995abf419da8c39b5a13"

# Reduced recipes: the reference recipe for the generic slots (block design d0, per 16, seeds 0-63, as the original
# search found it), one structured draw for the coded slots (which the planner does not beat).
QUICK = {"generic": [dict(name="block_8+3_d0", mode="block", groups=dr.GROUPS_8_3, design=dr.DESIGN_D0,
                          values=[16], seeds=[0, 63])],
         "coded": [dict(name="structured_8+3", mode="structured", groups=dr.GROUPS_8_3, values=[111],
                        seeds=[0, 15])]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def family_dir():
    """The a56ec354e8b8 family directory, found through the stage manifest."""
    man = STAGE / "manifest.csv"
    if not man.is_file():
        return None
    with open(man, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["task"] == "chemistry" and r["sha12"] == FAMILY]
    if not rows or rows[0]["sha256"] != FAMILY_SHA:
        return None
    d = STAGE / "circuits" / "chemistry" / FAMILY
    return d if (d / "submission.npz").is_file() else None


def ledger_words():
    """The family's word table from the chemistry parts ledger (families.csv table_key -> words/<key>.npy)."""
    fc = STAGE / "parts" / "families.csv"
    if not fc.is_file():
        return None
    with open(fc, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["family"] == FAMILY]
    p = STAGE / "parts" / "words" / f"{rows[0]['table_key']}.npy" if rows else None
    return np.load(p, allow_pickle=False).astype(np.int64) if p is not None and p.is_file() else None


def missing():
    try:
        import kickmix  # noqa: F401
        if not hasattr(kickmix, "Circuit"):
            return "kickmix build not importable"
    except ImportError:
        return "kickmix build not importable (run with the kickmix interpreter)"
    if not PLANNER.is_file():
        return "qrom-plan binary not found (set DFTHC_REFIT_QROM_PLAN or put qrom-plan on PATH)"
    if not (LERA / "tensorqrom.py").is_file():
        return "Lera's tensorqrom.py not found (set DFTHC_REFIT_LERA_SRC)"
    if not (VERIFIERS / "dfthcsos" / "verify.py").is_file():
        return "competition verifiers not found (set QHDL_OFFICIAL_ROOT or DFTHC_REFIT_VERIFIER_DIR)"
    if family_dir() is None:
        return f"family {FAMILY} not found through the dataset manifest (set QHDL_DATASET or DFTHC_REFIT_STAGE)"
    return None


SKIP = missing()


class TestPure(unittest.TestCase):
    def test_version(self):
        self.assertEqual(dr.__version__, "1.8.0-dev")

    def test_cache_count_model(self):
        # the planner hard-codes 89 (1,363-row tables); C = 45's 1,220-row table has 80
        self.assertEqual(dr.expected_cache_toffolis(1363), 89)
        self.assertEqual(dr.expected_cache_toffolis(1220), 80)
        self.assertEqual(dr.expected_cache_toffolis(1232), 80)       # coded padding to 16 rows changes nothing
        self.assertEqual(dr.PLANNER_CACHE_TOFFOLIS, 89)

    @unittest.skipUnless((LERA / "tensorqrom.py").is_file(), "Lera's tensorqrom.py not found (set DFTHC_REFIT_LERA_SRC)")
    def test_cache_count_matches_emitter_cost(self):
        try:
            tq, _ = dr._load_tensorqrom(LERA)
        except ImportError:
            self.skipTest("kickmix build not importable")
        for rows in (1220, 1232, 1363, 1024, 777):
            delta = np.zeros((rows, 5), np.int64)
            self.assertEqual(dr.cache_toffolis(tq, delta), dr.expected_cache_toffolis(rows), rows)

    def test_family_sha256(self):
        e = {"b.kmb": "1" * 64, "a.kmb": "2" * 64}
        text = f"a.kmb\t{'2' * 64}\nb.kmb\t{'1' * 64}\n"
        self.assertEqual(dr.family_sha256(e), hashlib.sha256(text.encode()).hexdigest())

    def test_parse_slots(self):
        self.assertEqual(dr._parse_slots(None, 3), [0, 1, 2])
        self.assertEqual(dr._parse_slots("0-3,7,2", 15), [0, 1, 2, 3, 7])

    def test_plan_jobs_order_and_templates(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = dr.plan_jobs(Path(d), 0, QUICK["generic"] + QUICK["coded"])
            self.assertEqual([(j["recipe"], j["value"], j["seeds"]) for j in jobs],
                             [("block_8+3_d0", 16, [0, 31]), ("block_8+3_d0", 16, [32, 63]),
                              ("structured_8+3", 111, [0, 15])])
            self.assertEqual(jobs[0]["template"].read_text().splitlines(),
                             ["k 0", "groups 3,5,6,7,11,13,14,15;9,10,12", "design 3,5,6;0,1,2;4,10,12;7,13,14;8,9,11",
                              "per 16"])
            self.assertEqual(jobs[2]["template"].read_text().splitlines()[-1], "count 111")

    def test_read_plan_txt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.txt"
            p.write_text("k 3\ngroups 1,2;3\ntoffolis 312\nrel 0x1f\nrel 0x2\nterm 1 2 3 4 5\nterm 6 7 0 0 9\n")
            pl = dr._read_plan_txt(p)
            self.assertEqual((pl["k"], pl["groups"], pl["planner_toffolis"], pl["relations"]),
                             (3, [[1, 2], [3]], 312, 2))
            self.assertEqual(pl["terms"], [(1, 2, 3, 4, 5), (6, 7, 0, 0, 9)])


@unittest.skipIf(SKIP, SKIP or "")
class TestReplanA56ec(unittest.TestCase):
    """Replan on a56ec354e8b8 reproduces the reference candidate (family c6d391fdc236): slots 0-3 at 303
    Toffolis per pass, slots 4-14 shipped; then, as the negative control, replan on that output changes nothing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fam = family_dir()
        cls.rep = dr.replan(cls.fam, Path(cls.tmp.name) / "c1", planner=PLANNER, lera_src=LERA,
                            verifier_dir=VERIFIERS, recipes=QUICK, procs=6, verbose=False)
        cls.out = Path(cls.tmp.name) / "c1" / "dfthcsos"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_planner_binary_is_the_reference_one(self):
        if self.rep["planner"]["sha256"] != PLANNER_SHA:
            self.skipTest("a different qrom-plan build than the reference 8224ed2b...")

    def test_words_recovered_equal_ledger(self):
        w = ledger_words()
        if w is None:
            self.skipTest("parts ledger words not found")
        got = np.load(Path(self.tmp.name) / "c1" / "work" / "words.npy").astype(np.int64)
        self.assertTrue(np.array_equal(got, w))

    def test_slots(self):
        rows = self.rep["slots"]
        for r in rows[:4]:
            self.assertEqual(r["kept"], "replanned", r)
            self.assertEqual(r["shipped"]["toffolis"], 307)
            rp = r["replanned"]
            self.assertEqual(rp["emitted_toffolis"], 303)
            self.assertEqual(rp["native_check"]["status"], "PASS")
            self.assertLessEqual(rp["qubits"], dr.QUBIT_CAP)
            # the planner's count is 9 too high on this 1,220-row table; corrected from the table, not hard-coded
            self.assertEqual((rp["planner_toffolis"], rp["cache_toffolis"], rp["planner_cache_offset"]), (312, 80, 9))
        for r in rows[4:]:
            self.assertEqual(r["kept"], "shipped", r)
            self.assertIn(r["shipped"]["toffolis"], (282, 284, 285))
            self.assertGreaterEqual(r["replanned"]["emitted_toffolis"], r["shipped"]["toffolis"])
        self.assertEqual(self.rep["changed_slots"], [0, 1, 2, 3])
        self.assertEqual(self.rep["toffolis_per_pass"], dict(shipped=4359, replanned=self.rep["toffolis_per_pass"]
                                                             ["replanned"], kept=4343, saving=16))

    def test_family_checks_and_cost(self):
        fc = self.rep["family_checks"]
        self.assertTrue(fc["family_ok"])
        self.assertFalse(fc["layout_mismatch"])
        self.assertFalse(fc["chain_break"])
        self.assertTrue(fc["metrics_unchanged"])
        c = self.rep["cost"]
        self.assertEqual(c["toffoli_after_official_evaluation"], 15427.0)
        self.assertAlmostEqual(c["after_official_evaluation"], 310821125.3617, delta=0.01)
        self.assertAlmostEqual(c["after_projected"], c["after_official_evaluation"], delta=1e-3)

    def test_output_reproduces_reference(self):
        self.assertEqual(self.rep["input_family_sha256"], FAMILY_SHA)
        self.assertEqual(sha(self.out / "submission.npz"), sha(self.fam / "submission.npz"))   # weights unchanged
        if REF_SUMS.is_file():
            want = dict(reversed(line.split()) for line in REF_SUMS.read_text().splitlines() if line.strip())
            self.assertEqual(self.rep["outputs_sha256"], want)
        self.assertEqual(self.rep["output_family_sha256"], REF_FAMILY_SHA)

    def test_negative_control_unchanged(self):
        """On the reproduced output the planner reaches 303 again on slots 0-3 (equal, not fewer) and loses on 4-14: nothing changes."""
        rep = dr.replan(self.out, Path(self.tmp.name) / "neg", planner=PLANNER, lera_src=LERA,
                        verifier_dir=VERIFIERS, recipes=QUICK, procs=6, verbose=False)
        self.assertEqual(rep["changed_slots"], [])
        self.assertTrue(all(r["kept"] == "shipped" for r in rep["slots"]))
        self.assertEqual([r["replanned"]["emitted_toffolis"] for r in rep["slots"][:4]], [303] * 4)
        self.assertEqual(rep["outputs_sha256"], rep["inputs_sha256"])
        self.assertEqual(rep["output_family_sha256"], REF_FAMILY_SHA)
        self.assertEqual(rep["toffolis_per_pass"]["saving"], 0)
        self.assertEqual(rep["family_checks"], {})


# Kernel backend (qwertykeypadapple's kernel_search.py, Apache-2.0).
KERNEL = env_path("DFTHC_REFIT_KERNEL_SRC")
KERNEL_SHA = "2904a6598725906f7048158e55e1d83ed818a2d1df717f2db8c5cd8aecf1f43e"
FAM_A344 = env_path("DFTHC_REFIT_A344")


class TestKernelLayout(unittest.TestCase):
    def test_sixes(self):
        perm, sizes = dr.kernel_layout(35, [])
        self.assertEqual(sizes, [6, 6, 6, 6, 6, 5])
        self.assertEqual(perm, list(range(35)))
        perm, sizes = dr.kernel_layout(35, [0, 1, 2, 3, 4])
        self.assertEqual(sizes, [6, 6, 6, 6, 6, 5])
        self.assertEqual(perm[-5:], [0, 7, 14, 21, 28])


@unittest.skipUnless(KERNEL.is_dir() and (KERNEL / "kernel_search.py").is_file() and LERA.is_dir()
                     and VERIFIERS.is_dir() and (FAM_A344 / "submission.npz").is_file(),
                     "needs kernel_search.py (DFTHC_REFIT_KERNEL_SRC), Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC), the "
                     "verifiers (QHDL_OFFICIAL_ROOT) and the a344f422f2e0 family (DFTHC_REFIT_A344)")
class TestKernelBackendA344(unittest.TestCase):
    """Inside replan: on qwertykeypadapple's own a344f422f2e0 family, slot 4 (coded) re-planned by his search at
    seed 0 reaches his 201 products (281 Toffolis per pass), a tie, so the shipped loader is kept."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rep = dr.replan(FAM_A344, Path(cls.tmp.name) / "k", lera_src=LERA, verifier_dir=VERIFIERS, slots="4",
                            procs=1, backend="kernel", kernel_src=KERNEL, kernel_seeds=(0,), verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_kernel_source_is_his(self):
        self.assertEqual(sha(KERNEL / "kernel_search.py"), KERNEL_SHA)
        self.assertEqual(self.rep["planner"]["backend"], "kernel")

    def test_reproduces_his_count(self):
        row = self.rep["slots"][4]
        rp = row["replanned"]
        self.assertEqual(rp["products"], 201)
        self.assertEqual(rp["emitted_toffolis"], 281)
        self.assertEqual(rp["emitted_toffolis"], row["shipped"]["toffolis"])
        self.assertEqual(rp["native_check"]["status"], "PASS")
        self.assertLessEqual(rp["qubits"], 128)
        self.assertTrue(rp["layout_equal_shipped"])
        self.assertEqual(row["kept"], "shipped")
        self.assertEqual(self.rep["changed_slots"], [])
        self.assertEqual(self.rep["outputs_sha256"], self.rep["inputs_sha256"])


# Padded rows (--pad-rows).
PADROW_PLANS = env_path("DFTHC_REFIT_F429_PADROW_PLAN")
F429 = env_path("DFTHC_REFIT_F429")


class TestPaddedRowsPure(unittest.TestCase):
    def test_padded_row_range(self):
        self.assertEqual(dr.padded_row_range(20), (20, 32))
        self.assertEqual(dr.padded_row_range(32), (32, 32))          # whole blocks: no never-read rows
        self.assertEqual(dr.padded_row_range(1220), (1220, 1232))
        self.assertEqual(dr.padded_row_range(1363), (1363, 1376))
        self.assertEqual(dr.padded_row_range(1), (1, 16))

    def test_full_block_delta(self):
        d = np.arange(20 * 3).reshape(20, 3)
        f = dr.full_block_delta(d)
        self.assertEqual(f.shape, (32, 3))
        self.assertTrue(np.array_equal(f[:20], d))
        self.assertFalse(f[20:].any())
        self.assertEqual(dr.full_block_delta(f).shape, (32, 3))

    def test_solve_fixed(self):
        feats = [0b0011, 0b0110, 0b1000]
        w = dr._solve_fixed(feats, 0b0101, 0b0111)
        self.assertEqual(w & 0b0111, 0b0101)
        self.assertIsNone(dr._solve_fixed([0b0011], 0b0001, 0b0011))


@unittest.skipUnless(LERA.is_dir(), "needs Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC)")
class TestPaddedRowsSynthetic(unittest.TestCase):
    """A 20-row table (two blocks; rows 20..31 never read): a padded-row change moves only the last high-chunk row
    of the ANF tensor, by exactly the requested low-monomial vector, and never a real row."""

    def test_apply_pad_changes(self):
        tq, _ = dr._load_tensorqrom(LERA)
        rng = np.random.default_rng(7)
        A = 20
        first, end = dr.padded_row_range(A)
        raw = rng.integers(0, 1 << 7, size=(A, 2))
        raw = raw | ((raw >> 6) & 1) << 7                            # signed gauge: bit 7 = bit 6
        d = dr.full_block_delta(raw)
        n0 = A % 16
        changes = {3: 0b101 << (n0 - 1), 6 + 7: 1 << 14}             # word 0 bit 3; word 1 bit 6 (gauge copy)
        new = dr.apply_pad_changes(d, changes, A)
        self.assertTrue(np.array_equal(new[:first], d[:first]))
        self.assertTrue(np.array_equal((new >> 7) & 1, (new >> 6) & 1))
        C0, C1 = (tq.anf_tensor(x, dr.LERA_BROT, dr.LERA_LOWBITS) for x in (d, new))
        diff = C0 ^ C1
        h = C0.shape[1] - 1
        self.assertFalse(diff[:, :h].any())
        self.assertFalse(diff[0].any())
        for o in range(C0.shape[2]):
            v = sum(int(diff[m, h, o]) << (m - 1) for m in range(1, C0.shape[0]))
            self.assertEqual(v, changes.get(o, 0), o)


@unittest.skipUnless(KERNEL.is_dir() and (KERNEL / "kernel_search.py").is_file() and LERA.is_dir()
                     and (F429 / "submission.npz").is_file() and (PADROW_PLANS / "arm_b" / "plans").is_dir(),
                     "needs kernel_search.py (DFTHC_REFIT_KERNEL_SRC), Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC), "
                     "the f4291619f2fd family (DFTHC_REFIT_F429) and the saved padded-row plan (DFTHC_REFIT_F429_PADROW_PLAN)")
class TestPaddedRowsF429(unittest.TestCase):
    """Loader 4 of f4291619f2fd, padded rows free, word group 0 first, seed 0: 200 products, the recorded plan
    and padded rows exactly, 280 Toffolis, byte-identical to the recorded family's loader (family 5619c37ef6bd;
    about 2 CPU-minutes)."""

    @classmethod
    def setUpClass(cls):
        words = np.load(F429 / "submission.npz")["angle_words"].astype(np.int64)
        cls.A = len(words)
        cls.words = words
        cls.rec = json.loads((PADROW_PLANS / "arm_b" / "plans" / "armb_k4_padext_w0_s0.json").read_text())
        d, cls.cw = dr.loader_delta(words, 4)
        cls.delta = dr.full_block_delta(d)
        cls.res = dr.kernel_plan(LERA, KERNEL, cls.delta, cls.cw, 0, perm=cls.rec["row"]["perm"],
                                 pad=dict(n0=cls.A % 16, real_rows=cls.A, extend=True, top=128, restarts=200))

    def test_detects_padded_rows(self):
        self.assertEqual(dr.padded_row_range(self.A), (1220, 1232))
        self.assertEqual(len(self.delta), 1232)

    def test_reproduces_recorded_plan(self):
        self.assertTrue(self.res["exact"])
        self.assertEqual(self.res["products"], 200)
        self.assertEqual([list(t) for t in self.res["terms"]], [list(t) for t in self.rec["terms"]])
        rows = {int(r): v for r, v in self.res["pad_rows"].items()}
        for r, v in self.rec["row"]["pad_rows"].items():
            self.assertEqual(rows[int(r)], v, r)
        self.assertEqual(self.res["pad"]["first_group"], "kernel+extend +pad")

    def test_emits_recorded_loader(self):
        try:
            import kickmix as km
        except ImportError:
            self.skipTest("needs a kickmix build")
        tq, _ = dr._load_tensorqrom(LERA)
        new = self.delta.copy()
        for r, v in self.res["pad_rows"].items():
            new[int(r)] = v
        self.assertTrue(np.array_equal(new[:self.A], self.delta[:self.A]))
        n_sel = (self.A - 1).bit_length()
        circ = tq.build_circuit(new, [tuple(t) for t in self.res["terms"]], n_sel, 8, 4, groups=None, scratch=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "angle_k_4.kmb"
            circ.to_file(p, format="kmb")
            circ = km.Circuit.from_file(p, format="kmb")
            self.assertEqual(int(circ.operation_statistics().num_toffoli_like), 280)
            self.assertLessEqual(int(circ.num_qubits), 128)
            self.assertEqual(dr.native_check(km, circ, new[:self.A])["status"], "PASS")
            ref = PADROW_PLANS / "family" / "dfthcsos" / "angle_k_4.kmb"
            if ref.is_file():
                self.assertEqual(sha(p), sha(ref))


if __name__ == "__main__":
    unittest.main()
