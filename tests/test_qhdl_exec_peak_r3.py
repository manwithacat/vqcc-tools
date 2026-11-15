"""The rotation breakdown of `qhdl exec` and the two liveness models of `peak`/`runs`.

Unit checks run everywhere NumPy is present; the bce2948ff24e checks need the public-circuit dataset (skipped if
missing). Needs NumPy (system python3); skipped in the stdlib-only project environment.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
import vqcc_config as CFG  # noqa: E402
BCE = CFG.DATASET / "circuits/invsqrt/bce2948ff24e.kmb"
# recorded local official verifier summary for bce2948ff24e
REC = dict(toffoli_count=1325.481775804887, t_count=6.999871841622275, rotation_count=41.0,
           rotation_eps=2.434041480874084e-13, toffoli_equivalent=1622.0725676541888)


@unittest.skipUnless(HAS_NUMPY, "needs numpy")
class RotationPricingTests(unittest.TestCase):
    def test_zpow_classes(self):
        import qhdl_exec as X
        self.assertEqual(X.zpow_class(0), "clifford")
        self.assertEqual(X.zpow_class(1 << 63), "clifford")   # Z
        self.assertEqual(X.zpow_class(1 << 62), "clifford")   # S
        self.assertEqual(X.zpow_class(1 << 61), "t")          # T
        self.assertEqual(X.zpow_class(3 << 61), "t")
        self.assertEqual(X.zpow_class(1 << 60), "rotation")
        self.assertEqual(X.zpow_class(12345), "rotation")

    def test_reproduces_recorded_charge(self):
        import qhdl_exec as X
        eq = X.rotation_toffoli_eq(REC["rotation_count"], REC["t_count"], REC["rotation_eps"])
        self.assertAlmostEqual(REC["toffoli_count"] + eq, REC["toffoli_equivalent"], places=6)

    def test_no_eps_no_price(self):
        import qhdl_exec as X
        self.assertIsNone(X.rotation_toffoli_eq(3, 1, None))
        self.assertEqual(X.rotation_toffoli_eq(0, 4, None), 1.0)

    def test_executed_counts_conditional_rotations(self):
        import qhdl_bitsim as B
        import qhdl_exec as X
        lines = ["Z_POW q0 0.1", "X q30", "HMR q30 b0", "Z_POW q0 0.1 if b0", "Z_POW q1 0.25", "Z_POW q0 0.5"]
        ops = [B.parse_line(t) for t in lines]
        z = X.executed(ops, list(range(22)), 31, "zero")
        s = X.executed(ops, list(range(22)), 31, 3)
        n = z["rows"]
        self.assertEqual(z["zpow_per_op"], {0: n, 3: 0, 4: n, 5: n})
        self.assertTrue(0.45 * n < s["zpow_per_op"][3] < 0.55 * n)
        self.assertEqual([X.zpow_class(ops[i].angle) for i in (0, 4, 5)], ["rotation", "t", "clifford"])


@unittest.skipUnless(HAS_NUMPY and BCE.exists(), "needs numpy and the public-circuit dataset")
class LivenessModelTests(unittest.TestCase):
    def test_peak_models_on_bce2948(self):
        import qhdl_peak as P
        a = P.analyse(BCE)
        r = P.analyse(BCE, liveness="reset")
        self.assertEqual((a["liveness"], a["peak"], a["runs"], a["ops_at_level"]), ("allocator", 62, 92, 16129))
        self.assertEqual((r["liveness"], r["peak"], r["runs"], r["ops_at_level"]), ("reset", 62, 99, 15742))
        self.assertEqual(a["other_liveness"], dict(liveness="reset", peak=62, level=62, runs=99, ops_at_level=15742))
        self.assertEqual(r["other_liveness"]["runs"], 92)

    def test_bad_model_refused(self):
        import qhdl_peak as P
        with self.assertRaises(ValueError):
            P.analyse(BCE, liveness="nope")


@unittest.skipUnless(HAS_NUMPY and BCE.exists(), "needs numpy and the public-circuit dataset")
class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hashlib
        cls.tmp = tempfile.TemporaryDirectory()
        # a recorded local official-verifier report for bce2948ff24e (the summary REC above), so the rotation pricing
        # has the verifier's eps without a 15-minute verifier run
        outputs = Path(cls.tmp.name) / "outputs"
        (outputs / "bce-report").mkdir(parents=True)
        summ = dict(REC, circuit_sha256=hashlib.sha256(BCE.read_bytes()).hexdigest(), status="PASS",
                    verified_shots=3_133_452 * 64)
        (outputs / "bce-report" / "report.json").write_text(json.dumps(
            dict(kickmix_module="recorded", platform="macOS (recorded fixture)", settings=dict(salt=""), results=dict(invsqrt=dict(summary=summ)))))
        cls.env = dict(os.environ, QHDL_WORK=str(Path(cls.tmp.name) / "work"), QHDL_OUTPUTS=str(outputs))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_qhdl(self, *args):
        p = subprocess.run([sys.executable, str(SCRIPTS / "qhdl.py"), *args, "--all"], capture_output=True,
                           text=True, env=self.env, timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return json.loads(p.stdout)

    def test_runs_both_models(self):
        d = self.run_qhdl("runs", "-c", "bce2948ff24e")
        self.assertEqual(len(d["rows"]), 99)
        self.assertEqual(d["facts"]["other_liveness"]["runs"], 92)
        d = self.run_qhdl("runs", "-c", "bce2948ff24e", "--liveness", "allocator")
        self.assertEqual(len(d["rows"]), 92)
        self.assertIn("--liveness allocator", d["rerun"])

    def test_peak_reset_option(self):
        d = self.run_qhdl("peak", "-c", "bce2948ff24e", "--liveness", "reset", "--show", "runs")
        self.assertEqual(len(d["rows"]), 99)
        self.assertIn("allocator liveness", d["claim"])

    def test_exec_rotation_share(self):
        d = self.run_qhdl("exec", "-c", "bce2948ff24e")
        rot = d["facts"]["rotations"]
        self.assertEqual(rot["static_rotations"], 41)
        self.assertAlmostEqual(rot["mean_rotations"], 41.0)
        self.assertAlmostEqual(rot["charge_toffoli_eq"], REC["toffoli_equivalent"], delta=0.05)
        self.assertTrue(0.17 < rot["rotation_share"] < 0.19)
        k = self.run_qhdl("exec", "-c", "bce2948ff24e", "--by", "kind")
        kinds = {r["kind"] for r in k["rows"] if r["region"] == "all"}
        self.assertIn("ROTATION (Z_POW)", kinds)


if __name__ == "__main__":
    unittest.main()
