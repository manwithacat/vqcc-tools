# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for the v1.4.0-dev subcommands of dfthc_refit.py: plan, recover, price-words, chained-and, and the shared /
block backends of replan.

Pure tests (argument parsing, dispatch, partitions, move files, job layout) and small end-to-end runs on synthetic
tables (plan on a table built from a known paired plan; chained-and on a two-AND copy-gate target) run anywhere with
NumPy. The synthetic recover / plan-against-shipped test needs a kickmix build and Lera's tensorqrom.py (run with the
kickmix interpreter). Integration tests on real families run only when DFTHC_REFIT_CLI_INTEGRATION=1:

    DFTHC_REFIT_LERA_SRC      directory with tensorqrom.py
    DFTHC_REFIT_VERIFIER_DIR  competition verifiers (default $QHDL_OFFICIAL_ROOT/verifiers)
    DFTHC_REFIT_F429          the f4291619f2fd dfthcsos directory (replan --backend block, slot 0; a few CPU-minutes)
    DFTHC_PRICE_FAMILY        a dfthcsos directory for price-words (with DFTHC_PRICE_QUBITS / DFTHC_PRICE_TOFFOLIS
                              and DFTHC_PRICE_WORDS, a .npy of the words its loaders load; needs PyTorch)

    "$QHDL_KICKMIX_PYTHON" -I -B test_cli.py -v
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chained_and as ca  # noqa: E402
import dfthc_refit as dr  # noqa: E402
import planner_shared as ps  # noqa: E402
from test_planner_shared import synthetic  # noqa: E402
from test_recover import single_terms  # noqa: E402

from testpaths import LERA, VERIFIERS, env_path  # noqa: E402

F429 = env_path("DFTHC_REFIT_F429")
INTEGRATION = os.environ.get("DFTHC_REFIT_CLI_INTEGRATION") == "1"
SMALL = dict(partitions=2, keep=1, cpu_cap=5.0, ex_cap=5.0)


def quiet(fn, *a, **kw):
    """Run fn with stdout/stderr captured; returns (result, stdout)."""
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        r = fn(*a, **kw)
    return r, o.getvalue()


def exits(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            fn(*a)
        except SystemExit as ex:
            return ex.code
    return None


class TestParsing(unittest.TestCase):
    def test_version_and_bare_call(self):
        self.assertEqual(dr.__version__, "1.8.0-dev")
        _, out = quiet(dr.main, [])
        self.assertIn("1.8.0-dev", out)

    def test_help_lists_subcommands(self):
        self.assertEqual(exits(dr.main, ["--help"]), 0)
        for name in dr.SUBCOMMANDS:
            self.assertEqual(exits(dr.main, [name, "--help"]), 0, name)
        for name in ("plan", "recover", "price-words", "chained-and", "replan"):
            self.assertIn(name, dr.SUBCOMMAND_EPILOG)

    def test_parse_blocks(self):
        b = dr.parse_blocks("1,5,13/2,6,14/3,7,15/4,8,11/9,10,12")
        self.assertEqual(b[0], [0, 4, 12])
        self.assertTrue(dr.valid_partition(b))
        with self.assertRaises(ValueError):
            dr.parse_blocks("1,2,3/4,5,6")
        self.assertFalse(dr.valid_partition([list(range(15))]))          # one 15-wide block: too many forms
        self.assertFalse(dr.valid_partition(None))

    def test_replan_defaults_unchanged(self):
        """`replan` without the new flags passes exactly the pre-1.4 settings (seeds/blocks/exchange unset)."""
        seen = {}
        orig = dr.replan

        def fake(*a, **kw):
            seen.update(kw, args=a)
            return dict(toffolis_per_pass={}, changed_slots=[], cost={})
        dr.replan = fake
        try:
            quiet(dr.main, ["replan", "fam", "out", "--planner", "qp", "--lera-src", "lera"])
            self.assertEqual(seen["backend"], "qrom-plan")
            self.assertEqual((seen["seeds"], seen["blocks"], seen["exchange"], seen["planner_settings"]),
                             (None, None, False, None))
            self.assertEqual(seen["kernel_seeds"], [0, 1])
            seen.clear()
            quiet(dr.main, ["replan", "fam", "out", "--backend", "kernel", "--kernel-src", "ks", "--lera-src", "l",
                            "--pad-rows"])
            self.assertEqual((seen["backend"], seen["pad_rows"], seen["pad_screen"], seen["seeds"]),
                             ("kernel", True, True, None))
            seen.clear()
            quiet(dr.main, ["replan", "fam", "out", "--backend", "block", "--lera-src", "l", "--seeds", "0,3",
                            "--exchange", "--blocks", "recover", "--cpu-cap", "30", "--pad-rows"])
            self.assertEqual((seen["backend"], seen["seeds"], seen["blocks"], seen["exchange"]),
                             ("block", [0, 3], "recover", True))
            self.assertEqual(seen["planner_settings"], dict(cpu_cap=30.0))
        finally:
            dr.replan = orig

    def test_replan_flag_errors(self):
        self.assertEqual(exits(dr.main, ["replan", "f", "o", "--planner", "p", "--lera-src", "l", "--pad-rows"]), 2)
        self.assertEqual(exits(dr.main, ["replan", "f", "o", "--backend", "kernel", "--lera-src", "l",
                                         "--exchange"]), 2)
        self.assertEqual(exits(dr.main, ["replan", "f", "o", "--backend", "shared", "--lera-src", "l",
                                         "--no-pad-screen"]), 2)
        self.assertEqual(exits(dr.main, ["replan", "f", "o", "--backend", "nope", "--lera-src", "l"]), 2)
        with self.assertRaises(ValueError):
            dr.replan("f", "o", lera_src="l", planner="p", pad_rows=True)

    def test_plan_and_other_flag_errors(self):
        self.assertEqual(exits(dr.main, ["plan", "f"]), 2)                                  # --planner required
        self.assertEqual(exits(dr.main, ["plan", "f", "--planner", "kernel", "--exchange"]), 2)
        self.assertEqual(exits(dr.main, ["recover", "f", "--emit-check"]), 2)
        self.assertEqual(exits(dr.main, ["price-words", "f", "--moves", "m.json", "--rank-drop", "3"]), 2)
        self.assertEqual(exits(dr.main, ["chained-and", "t.json"]), 2)                      # action required

    def test_load_moves(self):
        import price_words as pw
        with tempfile.TemporaryDirectory() as td:
            for i, d in enumerate([[dict(name="a", changes=[[1, 2, 3]])], dict(moves=[dict(changes=[[4, 5, 6]])]),
                                   dict(b=[[7, 8, 9], [10, 11, 12]])]):
                p = Path(td) / f"m{i}.json"
                p.write_text(json.dumps(d))
                mv = dr.load_moves(p, pw)
                self.assertEqual(len(mv), 1)
                self.assertIsInstance(mv[0], pw.Move)
            self.assertEqual(mv[0].name, "b")
            self.assertEqual(mv[0].changes, ((7, 8, 9), (10, 11, 12)))

    def test_price_words_needs_torch(self):
        try:
            import torch  # noqa: F401
            self.skipTest("PyTorch present")
        except ImportError:
            pass
        if not (VERIFIERS / "dfthcsos" / "verify.py").is_file():
            self.skipTest("verifiers missing")
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "w.npy"
            np.save(w, np.zeros((16, 5), np.int64))
            with self.assertRaisesRegex(RuntimeError, "PyTorch"):
                dr.price_words_family(td, words=w, qubits=1, toffolis=1, verifier_dir=VERIFIERS, verbose=False)


class TestSharedJobs(unittest.TestCase):
    def test_job_layout(self):
        table, _ = synthetic(5)
        A = len(table) - 4
        words = table[:A]
        deltas = {0: dr.loader_delta(words, 0)}
        with tempfile.TemporaryDirectory() as td:
            jobs, args = dr.shared_jobs("shared", [0], deltas, A, Path(td), seeds=[0, 2], slot_blocks={0: None},
                                        pad_on=True, exchange=True, settings=SMALL)
            self.assertEqual([j["value"] for j in jobs], [0, 2])
            self.assertEqual(jobs[0]["recipe"], "shared_padrows_exchange")
            self.assertEqual(args[1][4], 2)
            self.assertTrue((Path(td) / "plans" / "k0").is_dir())


class TestPlanSynthetic(unittest.TestCase):
    """plan on a synthetic table built from a known paired plan: one slot, no loaders (words given)."""

    @classmethod
    def setUpClass(cls):
        cls.table, cls.n_true = synthetic(5)
        cls.A = len(cls.table) - 4
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fam = Path(cls.tmp.name) / "fam"
        cls.fam.mkdir()
        cls.words = Path(cls.tmp.name) / "words.npy"
        np.save(cls.words, cls.table[:cls.A])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shared_plan_function(self):
        delta, cw = dr.loader_delta(self.table[:self.A], 0)
        r = dr.shared_plan("shared", delta, self.A, cw, 0, pad_free=True, settings=SMALL)
        self.assertTrue(r["exact"])
        self.assertTrue(r["real_rows_unchanged"])
        self.assertEqual(r["planner"], "planner_shared.plan")
        self.assertEqual(set(r["pad_rows"]), set(range(self.A, len(self.table))))

    def test_plan_cli_shared(self):
        out = Path(self.tmp.name) / "plan_shared"
        rep, _ = quiet(dr.main, ["plan", str(self.fam), "--planner", "shared", "--words", str(self.words),
                                 "--out", str(out), "--pad-rows", "--partitions", "2", "--keep", "1",
                                 "--cpu-cap", "5", "--procs", "1"])
        self.assertTrue((out / "plans.json").is_file())
        row = rep["slots"][0]
        self.assertEqual(row["predicted_toffolis"], row["cache_toffolis"] + row["products"])
        self.assertEqual(row["cache_toffolis"], dr.expected_cache_toffolis(len(self.table)))
        self.assertIsNone(rep["toffolis_per_pass"])                 # no shipped loaders: no comparison
        self.assertTrue(rep["pad_rows"]["active"])
        kp = json.loads((out / row["plan"]).read_text())
        self.assertTrue(kp["exact"])
        self.assertTrue(ps.exact_rows(self._delta_with(kp), [tuple(t) for t in kp["terms"]]))

    def test_plan_block_given_partition(self):
        blocks = ps.random_triples(15, random.Random(5))      # the partition synthetic(5) drew
        spec = "/".join(",".join(str(s + 1) for s in b) for b in blocks)
        rep, _ = quiet(dr.plan_family, self.fam, None, planner="block", words=self.words, blocks=spec,
                       procs=1, planner_settings=dict(cpu_cap=5.0), exchange=False, verbose=True)
        row = rep["slots"][0]
        self.assertEqual(row["planner_module"], "planner_block.plan_generic")
        self.assertEqual(row["blocks"], [[s + 1 for s in b] for b in blocks])
        self.assertTrue(ps.exact_rows(dr.full_block_delta(dr.loader_delta(self.table[:self.A], 0)[0]),
                                      [tuple(t) for t in row["terms"]]))
        self.assertEqual(rep["blocks"]["source"], "given")

    def _delta_with(self, kp):
        d = dr.full_block_delta(dr.loader_delta(self.table[:self.A], 0)[0]).copy()
        for r, v in kp["pad_rows"].items():
            d[int(r)] = v
        return d


class TestChainedAndCli(unittest.TestCase):
    def test_bounds_verify_encode(self):
        circ = ca.copy_gate_example()
        T = ca.image_target(circ, [(s, t) for s in range(4) for t in range(4)])
        with tempfile.TemporaryDirectory() as td:
            tj, cj = Path(td) / "target.json", Path(td) / "circ.json"
            T.to_json(tj)
            cj.write_text(json.dumps(dict(circuit=circ.to_dict())))
            rep, _ = quiet(dr.main, ["chained-and", str(tj), "bounds", "--out", str(Path(td) / "b.json")])
            self.assertTrue(rep["rank2_construction"]["verify"]["all_pass"])
            self.assertEqual(rep["rank2"], ca.rank2_lower_bound(T))
            self.assertLessEqual(rep["general"]["bound"], 2)                  # 2 chained ANDs meet it
            self.assertGreaterEqual(rep["rank2_construction"]["ands"], rep["rank2"]["bound"])
            self.assertTrue((Path(td) / "b.json").is_file())
            rep, _ = quiet(dr.main, ["chained-and", str(tj), "verify", "--circuit", str(cj)])
            self.assertTrue(rep["verify"]["all_pass"])
            self.assertEqual(rep["verify"]["ands"], 2)
            cnf = Path(td) / "k2.cnf"
            rep, _ = quiet(dr.main, ["chained-and", str(tj), "encode", "--k", "2", "--cnf", str(cnf)])
            self.assertTrue(cnf.read_text().startswith("p cnf") or "p cnf" in cnf.read_text())
            self.assertEqual(rep["cnf"]["k"], 2)
            ok = ca.assignment_from(ca.encode_cnf(T, 2), circ, T,
                                    {r: (s, t) for r, (s, t) in enumerate((s, t) for s in range(4) for t in range(4))})
            self.assertIsNotNone(ok)
            self.assertEqual(exits(dr.main, ["chained-and", str(tj), "encode", "--k", "2", "--cnf", str(cnf)]), 2)
            shp = Path(td) / "shape.cnf"
            rep, _ = quiet(dr.main, ["chained-and", str(tj), "encode", "--k", "2", "--chained", "1",
                                     "--rows", "spanning", "--cnf", str(shp)])
            self.assertEqual(rep["cnf"]["chained"], 1)
            self.assertLessEqual(rep["cnf"]["n_rows"], len(T.rows))


@unittest.skipUnless((LERA / "tensorqrom.py").is_file(), "needs Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC)")
class TestRecoverAndPlanSyntheticFamily(unittest.TestCase):
    """A synthetic two-slot family (slot deltas built from known paired plans, test_planner_shared.synthetic) whose
    loaders carry one single product per nonzero ANF cell (Lera's emitter): recover reads every product back and the
    re-emitted loaders pass the native check; plan against the shipped loaders finds fewer products."""

    @classmethod
    def setUpClass(cls):
        cls.km = dr.find_kickmix(None)
        if cls.km is None:
            raise unittest.SkipTest("kickmix not found")
        cls.tq, _ = dr._load_tensorqrom(LERA)
        cls.tmp = tempfile.TemporaryDirectory()
        t0, _ = synthetic(5)
        t1, _ = synthetic(6)
        cls.A = len(t0) - 4
        words = np.concatenate([t0, t0 ^ t1], axis=1)[:cls.A]          # slot 0 loads t0, slot 1 turns t0 into t1
        cls.words = Path(cls.tmp.name) / "words.npy"
        np.save(cls.words, words)
        cls.fam = Path(cls.tmp.name) / "fam"
        cls.fam.mkdir()
        cls.terms = {}
        n_sel = (cls.A - 1).bit_length()
        for k in range(2):
            delta, cw = dr.loader_delta(words, k)
            assert not cw
            full = dr.full_block_delta(delta)
            cls.terms[k] = single_terms(full)
            circ = cls.tq.build_circuit(full, cls.terms[k], n_sel, 8, 4, groups=None, scratch=True)
            circ.to_file(str(cls.fam / f"angle_k_{k}.kmb"), format="kmb")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_recover_cli(self):
        out = Path(self.tmp.name) / "rec"
        rep, _ = quiet(dr.main, ["recover", str(self.fam), "--words", str(self.words), "--out", str(out),
                                 "--emit-check", "--lera-src", str(LERA)])
        self.assertEqual(rep["family"]["recovered"], 2)
        self.assertTrue(rep["family"]["function_ok_all"])
        for row in rep["slots"]:
            self.assertEqual(row["products"], len(self.terms[row["k"]]))
            self.assertEqual(row["pairing"]["single"], row["products"])
            self.assertEqual(row["reemit"]["status"], "PASS")
            self.assertTrue(row["reemit"]["toffolis_equal_shipped"])
        for f in ("recover.json", "words.npy", "family.json"):
            self.assertTrue((out / f).is_file(), f)
        fj = json.loads((out / "family.json").read_text())
        self.assertEqual(fj["words_npy_sha256"], dr.sha256_file(out / "words.npy"))
        self.assertTrue(np.array_equal(np.load(out / "words.npy"), np.load(self.words)))
        with self.assertRaises(FileExistsError):
            dr.recover_family(self.fam, out, words=self.words, verbose=False)

    def test_plan_against_shipped(self):
        rep, _ = quiet(dr.plan_family, self.fam, None, planner="shared", words=self.words, procs=1,
                       planner_settings=SMALL, pad_rows=True)
        self.assertTrue(rep["shipped_available"])
        pp = rep["toffolis_per_pass"]
        self.assertEqual(pp["slots"], [0, 1])
        for row in rep["slots"]:
            self.assertEqual(row["shipped_toffolis"], row["cache_toffolis"] + len(self.terms[row["k"]]))
            self.assertLess(row["products"], len(self.terms[row["k"]]))
            self.assertGreater(row["saving_vs_shipped"], 0)
            self.assertEqual(row["predicted_toffolis"], row["cache_toffolis"] + row["products"])
        self.assertEqual(pp["saving"], sum(r["saving_vs_shipped"] for r in rep["slots"]))


@unittest.skipUnless(INTEGRATION and (F429 / "submission.npz").is_file() and (LERA / "tensorqrom.py").is_file()
                     and (VERIFIERS / "dfthcsos" / "verify.py").is_file(),
                     "set DFTHC_REFIT_CLI_INTEGRATION=1 (needs the f4291619f2fd family, Lera's source, verifiers)")
class TestReplanBlockF429(unittest.TestCase):
    """replan --backend block on generic slot 0 of Syed Asad Ali's f4291619f2fd family, partition recovered from
    the shipped loader: the plan is emitted, natively checked and predicted exactly; the slot keeps the shipped
    loader unless strictly cheaper (the planner reaches Syed Asad Ali's 217 products, a tie)."""

    @classmethod
    def setUpClass(cls):
        if dr.find_kickmix(None) is None:
            raise unittest.SkipTest("kickmix not found")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "r"
        cls.rep, _ = quiet(dr.main, ["replan", str(F429), str(cls.out), "--backend", "block", "--lera-src", str(LERA),
                                     "--verifier-dir", str(VERIFIERS), "--slots", "0", "--procs", "1",
                                     "--pad-rows", "--cpu-cap", "30", "--quiet"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_slot0(self):
        rep = json.loads((self.out / "report.json").read_text())
        self.assertEqual(rep["planner"]["backend"], "block")
        self.assertEqual(rep["planner"]["blocks"]["per_slot"]["0"]["used"], "own")
        self.assertEqual(rep["planner"]["blocks"]["per_slot"]["0"]["blocks"],
                         [[1, 5, 13], [2, 6, 14], [3, 7, 15], [4, 8, 11], [9, 10, 12]])
        s0 = rep["slots"][0]
        rp = s0["replanned"]
        self.assertEqual(rp["native_check"]["status"], "PASS")
        self.assertEqual(rp["emitted_toffolis"], rp["predicted_toffolis"])
        self.assertLessEqual(rp["qubits"], 128)
        self.assertEqual(rp["planner_module"], "planner_block.plan_generic")
        if rp["emitted_toffolis"] >= s0["shipped"]["toffolis"]:
            self.assertEqual(s0["kept"], "shipped")
        for k in range(1, len(rep["slots"])):
            self.assertEqual(rep["slots"][k]["kept"], "shipped")


@unittest.skipUnless(INTEGRATION and os.environ.get("DFTHC_PRICE_FAMILY") and os.environ.get("DFTHC_PRICE_WORDS")
                     and (VERIFIERS / "dfthcsos" / "verify.py").is_file(),
                     "set DFTHC_REFIT_CLI_INTEGRATION=1, DFTHC_PRICE_FAMILY and DFTHC_PRICE_WORDS (needs PyTorch)")
class TestPriceWordsCli(unittest.TestCase):
    def test_rank_drop_table(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("needs PyTorch")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "pw"
            k = int(os.environ.get("DFTHC_PRICE_LOADER", 4))
            rep, _ = quiet(dr.main, ["price-words", os.environ["DFTHC_PRICE_FAMILY"], "--words",
                                     os.environ["DFTHC_PRICE_WORDS"], "--qubits",
                                     os.environ.get("DFTHC_PRICE_QUBITS", "427"), "--toffolis",
                                     os.environ.get("DFTHC_PRICE_TOFFOLIS", "15175"), "--verifier-dir",
                                     str(VERIFIERS), "--rank-drop", str(k), "--drops", "1", "--unit-row-prices",
                                     "--keep", "2", "--refit-top", "0", "--out", str(out)])
            self.assertTrue((out / "price_words.json").is_file())
            self.assertGreater(rep["shadow"]["loader_toffoli_units"], 0)
            self.assertEqual(rep["moves_source"]["rank_drop"], k)
            for r in rep["moves"]:
                self.assertEqual(r["price_tier"], "first_order")
                self.assertAlmostEqual(r["net_units"], r["saving_units"] - r["price_units"])


if __name__ == "__main__":
    unittest.main()
