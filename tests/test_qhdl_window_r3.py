"""qhdl window / fdiff, tool round 3 (2026-11-01): fresh scratch at the window start is not an input; fdiff --map
(edit map JSON or auto alignment), two measurement branches, and the whole-circuit mode.

Synthetic circuits use the invsqrt register layout (x = q0..q21). The CLI tests simulate all 3,133,452 legal x on
tiny circuits (a few seconds each). NumPy required (system python3).
"""

import hashlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

from test_qhdl_window import CLI, HEADER, THREE, TWO, ops_of  # noqa: E402

# phase (-1)^(x0 x1 x2), three equivalent forms and one with a missing measurement fix-up
P_TOFF = "CCX q0 q1 q30\nCZ q30 q2\nCCX q0 q1 q30\n"
P_CCZ = "CCZ q0 q1 q2\n"
P_MBU = "CCX q0 q1 q30\nCZ q30 q2\nHMR q30 b0\nCZ q0 q1 if b0\n"
P_MBU_NOFIX = "CCX q0 q1 q30\nCZ q30 q2\nHMR q30 b0\n"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed (stdlib-only env); run with system python3")
class FreshScratchTests(unittest.TestCase):
    def wf(self, body, A, B_):
        import numpy as np
        import qhdl_window as W
        ops = ops_of(body)
        return W.window_function(ops, list(range(22)), 40, A, B_, rows=np.arange(1, 1024, dtype=np.int64),
                                 branches=("zero",))

    def test_window_at_op0_reads_only_x(self):
        r = self.wf(P_TOFF + "CX q0 q31\n", 0, 4)
        self.assertEqual(sorted(r["input_names"]), ["q0", "q1", "q2"])
        self.assertEqual(r["io"]["fresh_zero"], 2)  # q30, q31: |0> at op 0

    def test_untouched_scratch_later_is_fresh_touched_is_not(self):
        body = "X q5\nCX q0 q31\n" + P_TOFF
        r = self.wf(body, 2, 5)  # q30 first touched at op 2: fresh; q31 written at op 1 but not read here
        self.assertNotIn("q30", r["input_names"])
        r = self.wf("CX q0 q30\n" + P_TOFF, 1, 4)  # q30 holds x0 at op 1: a real input
        self.assertIn("q30", r["input_names"])

    def test_scan_drops_fresh_inputs(self):
        import numpy as np
        import qhdl_window as W
        ops = ops_of(THREE)
        res, skipped, _ = W.scan(ops, list(range(22)), 40, size=3, span=40, rows=np.arange(1, 1024, dtype=np.int64),
                                 branches=("zero",))
        for r in res:  # q30 is first touched at op 1, q31 at op 9 (THREE): fresh before that
            A = r["ops"][0]
            self.assertFalse({nm for nm, first in (("q30", 1), ("q31", 9)) if A <= first} & set(r["input_names"]),
                             r["ops"])
        self.assertIn((1, 10), {tuple(r["ops"]) for r in res})


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed (stdlib-only env); run with system python3")
class FdiffR3CliTests(unittest.TestCase):
    def cli(self, work, *args):
        env = dict(os.environ, QHDL_WORK=str(work))
        out = subprocess.run([sys.executable, "-I", "-c", CLI.format(scripts=str(SCRIPTS)), *args],
                             capture_output=True, text=True, env=env)
        return out.returncode, (json.loads(out.stdout) if out.stdout.strip().startswith("{") else out.stdout + out.stderr)

    def write(self, tmp, name, body):
        p = Path(tmp) / f"{name}.kmx"
        p.write_text(HEADER + body)
        return p

    def test_whole_equivalent_and_branch_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.write(tmp, "toff", P_TOFF)
            for name, body, zero_ok, rand_ok in (("ccz", P_CCZ, True, True), ("mbu", P_MBU, True, True),
                                                 ("nofix", P_MBU_NOFIX, True, False)):
                b = self.write(tmp, name, body)
                # --all keeps "facts" inline; without it the envelope moves facts to the evidence file once stdout
                # exceeds its 2 KB budget (qhdl_envelope.py), and the facts checks below would find only "moved".
                rc, d = self.cli(Path(tmp) / "work", "fdiff", "-c", str(a), "-c2", str(b), "--whole", "--all")
                self.assertEqual(rc, 0, d)
                self.assertEqual(d["tier"], "exact-all-x")
                rows = {r["branch"]: r for r in d["rows"]}
                self.assertEqual(rows["zero"]["equivalent"], zero_ok, name)
                self.assertEqual(rows["random(seed 7)"]["equivalent"], rand_ok, name)
                self.assertEqual(("EQUIVALENT" in d["claim"]), zero_ok and rand_ok, name)
                self.assertIn("--whole", d["rerun"])
            # the missing fix-up shows as outcome-dependent phase on the second circuit only
            self.assertEqual(d["facts"]["outcome_dependent_phase_x"][0], 0)
            self.assertGreater(d["facts"]["outcome_dependent_phase_x"][1], 0)

    def test_map_auto_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.write(tmp, "three", "X q5\n" + THREE + "X q5\n")
            b = self.write(tmp, "two", "X q5\n" + TWO + "X q5\n")
            work = Path(tmp) / "work"
            rc, d = self.cli(work, "fdiff", "-c", str(a), "-c2", str(b), "--ops", "1:12", "--map", "auto")
            self.assertEqual(rc, 0, d)
            self.assertEqual(json.loads(Path(d["evidence"]).read_text())["ops2"], "1:13")
            self.assertTrue(all(not r["differs_at_end"] and not r["differs_at_end_random"] for r in d["rows"]), d)
            self.assertIn("--map auto", d["rerun"])
            hunks = [["equal", 0, 1, 0, 1], ["replace", 1, 12, 1, 13], ["equal", 12, 13, 13, 14]]
            m = Path(tmp) / "e.map.json"
            m.write_text(json.dumps(dict(old_sha256=sha(a), new_sha256=sha(b), old_ops=13, new_ops=14, hunks=hunks)))
            rc, d = self.cli(work, "fdiff", "-c", str(a), "-c2", str(b), "--ops", "1:12", "--map", str(m))
            self.assertEqual(rc, 0, d)
            self.assertEqual(json.loads(Path(d["evidence"]).read_text())["ops2"], "1:13")
            rc, d = self.cli(work, "fdiff", "-c", str(b), "-c2", str(a), "--ops", "1:13", "--map", str(m))
            self.assertEqual(rc, 2)
            self.assertIn("swap", d["error"] + d.get("hint", ""))
            rc, d = self.cli(work, "fdiff", "-c", str(a), "-c2", str(b), "--ops", "3:12", "--map", str(m))
            self.assertEqual(rc, 2)  # cut inside the edited ops
            self.assertIn("inside", d["error"])

    def test_old_form_still_works_and_wide_windows_compare_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            pre = "".join(f"X q{w}\n" for w in range(30, 41))  # 11 scratch wires set before the window
            body = pre + "".join(f"CX q{w} q50\n" for w in list(range(22)) + list(range(30, 41)))
            a = self.write(tmp, "wide", body)
            b = self.write(tmp, "wide2", body)
            n = len(body.strip().splitlines())
            rc, d = self.cli(Path(tmp) / "work", "fdiff", "-c", str(a), "-c2", str(b), "--ops", f"11:{n}",
                             "--ops2", f"11:{n}")
            self.assertEqual(rc, 0, d)
            self.assertIn("agrees", d["claim"])
            ev = json.loads(Path(d["evidence"]).read_text())
            self.assertTrue(ev["sides"][0]["kind"].startswith("unavailable"))


if __name__ == "__main__":
    unittest.main()
