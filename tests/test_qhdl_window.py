"""qhdl window / windows / fdiff (harness plan 2026-11-01, workstream C): window functions on reachable inputs and
AND-count floors.

Synthetic circuits use the invsqrt register layout (x = q0..q21) so the verbs accept them; the library tests run on
x = 1..1023 (all patterns of x0..x9) for speed, the CLI test on all legal x. NumPy is required (system python3); the
tests are skipped in the stdlib-only project environment. The real-circuit test runs when the pinned kmx exports of
bce2948ff24e (Ehan Mallick) and its parent 67482db6c538 (yoyobar) are on disk (about 5 s).
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

HEADER = "".join(f"APPEND_TO_REGISTER q{i} r0\n" for i in range(22)) + 'REGISTER r0 "x"\n'

# u = q0, v = q1, n = q2, a = q3, b = q4; accumulator q30 (clean), second target q31 (clean).
# q30 ^= u(n+b) + v(n+u+a); q31 ^= v(a+b)  -- three ANDs (the shape of Ehan Mallick's windows)
THREE = """CX q4 q2
CCX q0 q2 q30
CX q4 q2
CX q0 q2
CX q3 q2
CCX q1 q2 q30
CX q3 q2
CX q0 q2
CX q4 q3
CCX q1 q3 q31
CX q4 q3
"""
# same two outputs with two ANDs: Q1 + Q2 = (u+w)(v+w) + w, w = n + b, so q30 = (u+w)(v+w) + w + Q2
TWO = """CX q4 q3
CCX q1 q3 q31
CX q4 q3
CX q31 q30
CX q4 q2
CX q2 q0
CX q2 q1
CCX q0 q1 q30
CX q2 q30
CX q2 q1
CX q2 q0
CX q4 q2
"""


def ops_of(body):
    import qhdl_bitsim as B
    return [B.parse_line(line) for line in body.strip().splitlines()]


def run(body, A=0, B_=None, **kw):
    import numpy as np
    import qhdl_window as W
    ops = ops_of(body)
    rows = np.arange(1, 1024, dtype=np.int64)
    return W.window_function(ops, list(range(22)), 40, A, len(ops) if B_ is None else B_, rows=rows,
                             branches=kw.pop("branches", ("zero",)), **kw)


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed (stdlib-only env); run with system python3")
class WindowFunctionTests(unittest.TestCase):
    def test_three_and_window_has_floor_two(self):
        r = run(THREE)
        self.assertEqual(r["io"]["toffoli"], 3)
        self.assertTrue(r["full_domain"])
        self.assertEqual(r["k"], 5)
        o = r["outputs"]
        self.assertEqual((o["q30"]["kind"], o["q30"]["rank"]), ("quadratic", 4))
        self.assertEqual((o["q31"]["kind"], o["q31"]["rank"]), ("quadratic", 2))
        self.assertTrue(o["q2"]["restored"] and o["q3"]["restored"])
        fl = r["floor"]
        self.assertEqual((fl["lb"], fl["ub"], fl["kind"]), (2, 2, "exact"))

    def test_two_and_rewrite_computes_the_same_function(self):
        import qhdl_window as W
        a, b = run(THREE), run(TWO)
        self.assertEqual(b["io"]["toffoli"], 2)
        for w in ("q30", "q31"):
            ta = W.anf_text(a["outputs"][w]["monos"], a["pivot_names"])
            tb = W.anf_text(b["outputs"][w]["monos"], b["pivot_names"])
            self.assertEqual(ta, tb, w)
        self.assertEqual(W.anf_text(a["outputs"]["q31"]["monos"], a["pivot_names"]), "q1*q3 + q1*q4")
        self.assertEqual((b["floor"]["lb"], b["floor"]["ub"]), (2, 2))

    def test_compute_copy_uncompute_has_slack_one(self):
        r = run("CCX q0 q1 q30\nCX q30 q31\nCCX q0 q1 q30\n")
        self.assertEqual(r["outputs"]["q30"]["kind"], "zero")
        self.assertEqual(r["outputs"]["q31"]["kind"], "quadratic")
        self.assertEqual((r["floor"]["lb"], r["floor"]["ub"]), (1, 1))
        self.assertEqual(r["io"]["toffoli"] - r["floor"]["ub"], 1)

    def test_rotation_and_measurement_consume_the_product(self):
        # the product is read by Z_POW and by the HMR kickback: no free lunch even though q30 ends at 0
        r = run("CCX q0 q1 q30\nZ_POW q30 0.25\nHMR q30 b5\n")
        self.assertIn("zpow@1:q30", r["outputs"])
        self.assertEqual(r["outputs"]["zpow@1:q30"]["kind"], "quadratic")
        self.assertEqual((r["floor"]["lb"], r["floor"]["ub"]), (1, 1))

    def test_dependent_inputs_are_reduced_and_dont_cares_counted(self):
        # q30 = x0 when the window starts, so q31 ^= x0 * q30 is just x0: affine, floor 0
        body = "CX q0 q30\nCCX q0 q30 q31\n"
        r = run(body, A=1)
        self.assertEqual(r["relations"], {"q30": "q0"})
        self.assertEqual(r["k"], 1)
        self.assertEqual(r["outputs"]["q31"]["kind"], "affine")
        self.assertEqual((r["floor"]["lb"], r["floor"]["ub"]), (0, 0))
        self.assertEqual(r["unreachable_raw"], 2)  # (q0, q30) in {00, 11}; q31 is fresh |0> at op 1 (round 3)

    def test_nonlinear_dont_cares_use_min_degree_completion(self):
        # q32 = x0 x1 before the window; q31 ^= q32 * q0 equals q32 on every reachable state: the CCX is a CX
        body = "CCX q0 q1 q32\nCCX q32 q0 q31\n"
        r = run(body, A=1)
        self.assertFalse(r["full_domain"])
        self.assertEqual((r["k"], r["reachable"], r["dont_cares_in_hull"]), (2, 3, 1))  # (q0, q32): 00, 10, 11
        self.assertEqual(r["outputs"]["q31"]["anf"] if "anf" in r["outputs"]["q31"] else
                         [r["pivot_names"][j] for j in range(2) if r["outputs"]["q31"]["monos"] == [1 << j]], ["q32"])
        self.assertEqual((r["floor"]["lb"], r["floor"]["ub"]), (0, 0))

    def test_declared_inputs_are_checked(self):
        r = run(THREE, declared=["q0", "q1", "q2"])
        self.assertGreater(r["not_determined_by_declared"]["q30"], 0)

    def test_nonlinear_rank_floor(self):
        # two independent products on two targets: rank 2
        r = run("CCX q0 q1 q30\nCCX q2 q3 q31\n")
        self.assertEqual(r["floor"]["floor_nonlinear_rank"], 2)
        self.assertEqual(r["floor"]["lb"], 2)
        # same-target sum of products x0x1 + x2x3: quadratic rank 4 -> 2 as well
        r = run("CCX q0 q1 q30\nCCX q2 q3 q30\n")
        self.assertEqual(r["floor"]["floor_nonlinear_rank"], 1)
        # three independent cubic/recursive products: rank 3
        r = run("CCX q0 q1 q30\nCCX q30 q2 q31\nCCX q31 q3 q32\n")
        self.assertEqual(r["floor"]["floor_nonlinear_rank"], 3)
        self.assertEqual(r["floor"]["lb"], 3)
        # redundant CX on an output must not raise it
        r = run("CCX q0 q1 q30\nCCX q2 q3 q31\nCX q30 q31\nCX q30 q31\n")
        self.assertEqual(r["floor"]["floor_nonlinear_rank"], 2)

    def test_nonlinear_rank_inverse_and_affine_are_zero(self):
        r = run("CCX q0 q1 q30\nCCX q0 q1 q30\n")
        self.assertEqual(r["floor"]["floor_nonlinear_rank"], 0)
        r = run("CX q0 q30\nCX q1 q30\nX q31\n")
        self.assertEqual(r["floor"]["floor_nonlinear_rank"], 0)

    def test_nonlinear_rank_never_lowers_existing_floor(self):
        import qhdl_window as W
        r = run(THREE)
        self.assertLessEqual(r["floor"]["floor_nonlinear_rank"], r["floor"]["lb"])
        self.assertEqual(r["floor"]["lb"], 2)
        self.assertEqual(W.nonlinear_rank([]), 0)

    def test_nonlinear_rank_not_applicable_with_hmr_or_condition(self):
        r = run("CCX q0 q1 q30\nZ_POW q30 0.25\nHMR q30 b5\n")
        self.assertIsNone(r["floor"]["floor_nonlinear_rank"])
        self.assertEqual((r["floor"]["lb"], r["floor"]["ub"]), (1, 1))
        r = run("CCX q0 q1 q30\nCZ q0 q1\n")
        self.assertIsNone(r["floor"]["floor_nonlinear_rank"])

    def test_cubic_output_floor_is_degree_bound(self):
        r = run("CCX q0 q1 q30\nCCX q30 q2 q31\nCCX q0 q1 q30\n")
        self.assertEqual(r["outputs"]["q31"]["deg"], 3)
        self.assertEqual(r["floor"]["lb"], 2)
        self.assertIsNone(r["floor"]["ub"])

    def test_scan_finds_the_three_and_window(self):
        import numpy as np
        import qhdl_window as W
        ops = ops_of(THREE + "CX q30 q33\n" + TWO.replace("q30", "q34").replace("q31", "q35"))
        res, skipped, _ = W.scan(ops, list(range(22)), 40, size=3, span=40, rows=np.arange(1, 1024, dtype=np.int64),
                                 branches=("zero",))
        best = {tuple(r["ops"]): r for r in res}
        r = best[(1, 10)]  # the three CCX of THREE
        self.assertEqual((r["io"]["toffoli"], r["floor"]["ub"]), (3, 2))
        self.assertEqual(r["floor"]["kind"], "exact")
        rows = W.collapse([W._scan_row(r) for r in res])
        self.assertGreaterEqual(rows[0]["charged"], 1.0)  # top row: a constructive saving
        two = best[(13, 20)]  # TWO's two CCX: at the floor
        self.assertEqual((two["io"]["toffoli"], two["floor"]["ub"]), (2, 2))


CLI = (
    "import sys; sys.path.insert(0, {scripts!r}); import qhdl\n"
    "qhdl.PLUGINS = tuple(qhdl.PLUGINS) + (() if 'qhdl_window' in qhdl.PLUGINS else ('qhdl_window',))\n"
    "sys.exit(qhdl.main(sys.argv[1:]))\n"
)


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed (stdlib-only env); run with system python3")
class WindowCliTests(unittest.TestCase):
    def cli(self, work, *args):
        env = dict(os.environ, QHDL_WORK=str(work))
        code = CLI.format(scripts=str(SCRIPTS))
        return subprocess.run([sys.executable, "-I", "-c", code, *args], capture_output=True, text=True, env=env)

    def test_window_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "three.kmx"
            src.write_text(HEADER + THREE)
            out = self.cli(tmp / "work", "window", "-c", str(src), "--ops", "0:11")
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertLessEqual(len(out.stdout.encode()), 2048)
            d = json.loads(out.stdout)
            self.assertEqual(d["tier"], "exact-all-x")
            if 'moved' in d.get('facts', {}):
                # Absolute replay commands take more of the bounded envelope;
                # --all must still expose the complete computed facts.
                full = self.cli(tmp / "work", "window", "-c", str(src), "--ops", "0:11", "--all")
                self.assertEqual(full.returncode, 0, full.stderr)
                d = json.loads(full.stdout)
            self.assertEqual(d["facts"]["floor"], {"lb": 2, "ub": 2, "kind": "exact", "model_lb": 2, "floor_nonlinear_rank": 2})
            self.assertEqual(d["facts"]["toffoli_static"], 3)
            self.assertIn("window", d["rerun"])
            ev = json.loads(Path(d["evidence"] if os.path.isabs(d["evidence"]) else ROOT / d["evidence"]).read_text())
            self.assertIn("q30", ev["outputs"])

    def test_refuses_non_invsqrt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "other.kmx"
            src.write_text('APPEND_TO_REGISTER q0 r0\nREGISTER r0 "y"\nCCX q0 q1 q2\n')
            out = self.cli(tmp / "work", "window", "-c", str(src), "--ops", "0:1")
            self.assertEqual(out.returncode, 2, out.stdout + out.stderr)
            d = json.loads(out.stdout)
            self.assertIn("invsqrt", d["error"])
            self.assertIn("hint", d)


KMX = Path(os.environ.get("QHDL_TEST_FIXTURES") or ROOT / "tests" / "fixtures") / "kmx"  # optional .kmx exports


@unittest.skipUnless(HAS_NUMPY and (KMX / "bce2948ff24e.kmx").exists() and (KMX / "67482db6c538.kmx").exists(),
                     "needs NumPy and the pinned invsqrt kmx exports")
class RealInvsqrtTests(unittest.TestCase):
    """Ehan Mallick's first 3-to-2 AND merge: three ANDs in yoyobar's 67482db6c538 (ops 21455..21473), two in
    bce2948ff24e (ops 21454..21481, 0-based kmx op indices); both have AND floor 2 on all 3,133,452 legal x."""

    def load(self, name):
        import qhdl_bitsim as B
        return B.load(str(KMX / f"{name}.kmx"))

    def test_and_merge_window_floor(self):
        import qhdl_window as W
        for name, (A, B_), toff in (("67482db6c538", (21455, 21474), 3), ("bce2948ff24e", (21454, 21482), 2)):
            ops, _, xq, nq = self.load(name)
            r = W.window_function(ops, xq, nq, A, B_)
            self.assertEqual(r["rows"], 3_133_452)
            self.assertEqual(r["io"]["toffoli"], toff, name)
            self.assertEqual((r["floor"]["lb"], r["floor"]["ub"], r["floor"]["kind"]), (2, 2, "exact"), name)
            o = r["outputs"]
            self.assertEqual(o["q19"]["rank"], 4)
            self.assertEqual(o["q14"]["rank"], 2)
            self.assertFalse(r["outcome_dependent"])


if __name__ == "__main__":
    unittest.main()
