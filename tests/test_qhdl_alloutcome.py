"""qhdl_alloutcome: activation-conditioned shared-outcome equivalence screen. Tiny synthetic circuits,
x = q0..q2 on rows 1..7. NumPy required (system python3)."""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    import qhdl_alloutcome as A
    import qhdl_bitsim as B

# q3 = x0&x1 ; six extra outcome bits b1..b6 from HMRs of |0> wires; HMR q3 -> b0 and the standard CZ fix-up.
PRE = ["CCX q0 q1 q3", "HMR q3 b0"] + [f"HMR q{3 + k} b{k}" for k in range(1, 7)]
FIX = ["CZ q0 q1 if b0"]
NEST = [f"PUSH_CONDITION if b{k}" for k in range(1, 7)]
POPS = ["POP_CONDITION"] * 6
RARE = ["CZ q0 q1 if b0"]        # fires only when b0..b6 are all 1 (nested): a net extra phase on x with x0 = x1 = 1
UNREACH_SETUP = ["BIT_INVERT b8", "BIT_INVERT b8 if b1"]   # b8 = 1 + b1
UNREACH = ["PUSH_CONDITION if b1", "PUSH_CONDITION if b8", "CZ q0 q1", "POP_CONDITION", "POP_CONDITION"]


def ops(lines):
    return [B.parse_line(s) for s in lines]


def run(parent, child):
    return A.check(ops(parent), ops(child), [0, 1, 2], nq1=10, nq2=10, rows=np.arange(1, 8), rand_runs=1)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class AllOutcome(unittest.TestCase):
    def test_polynomials_and_solver(self):
        # x0*x1 + x2 + 1 = 1 and x0 + x2 = 1  ->  affine part solvable, nonaffine counted
        f_aff = frozenset({1 << 0, 1 << 2})
        f_non = frozenset({(1 << 0) | (1 << 1), 0})
        sol = A.solve_activation([f_aff, f_non], 3)
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["nonaffine"], 1)
        self.assertEqual(A.solve_activation([frozenset({1}), frozenset({1, 0})], 3)["status"], "unreachable")
        self.assertEqual(A.p_eval(A.p_mul(f_aff, f_aff), 0b001), 1)

    def test_identical_circuits_equivalent(self):
        base = PRE + FIX + NEST + RARE + POPS
        r = run(base, base)
        self.assertEqual(r["verdict"], "EQUIVALENT_ON_ACTIVATING_SAMPLES")
        self.assertEqual(r["parent_only_ops"], 0)

    def test_rare_branch_deletion_rejected_only_by_targeting(self):
        parent = PRE + FIX + NEST + RARE + POPS
        child = PRE + FIX + NEST + POPS            # deletes the rare nested CZ
        r = run(parent, child)
        self.assertEqual(r["verdict"], "REJECT")
        self.assertEqual(r["parent_only_ops"], 1)
        self.assertTrue(r["runs"][-1]["label"].startswith("target"))
        self.assertGreater(r["runs"][-1]["phase_differs"], 0)
        # the three plain branches (zero + one random) do not see it with these seeds: the screen's blind spot
        self.assertTrue(all(x["equivalent"] for x in r["runs"] if x["kind"] != "sol"))

    def test_unreachable_deletion_certified(self):
        parent = PRE + FIX + UNREACH_SETUP + UNREACH
        child = PRE + FIX + UNREACH_SETUP
        r = run(parent, child)
        self.assertEqual(r["verdict"], "EQUIVALENT_ON_ACTIVATING_SAMPLES")
        self.assertGreaterEqual(r["provably_unreachable_targets"], 1)

    def test_unconditioned_change_checked_by_plain_runs(self):
        parent = PRE + FIX + ["CZ q0 q1"]
        child = PRE + FIX
        r = run(parent, child)
        self.assertEqual(r["verdict"], "REJECT")
        self.assertEqual(r["unconditioned_changed_ops"], 1)


if __name__ == "__main__":
    unittest.main()
