"""qhdl recompute: synthetic determination, witness, quadratic pricing with an exactly checked fragment, and the
negative control on the public 62-Q circuit 1c5b1ee28290 (q23 at the pre-op-15918 cut is not determined by the other live values)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qhdl_bitsim as B  # noqa: E402
import qhdl_recompute as R  # noqa: E402
import qhdl_valuegraph as G  # noqa: E402

ROWS = np.arange(1, 64)       # 6 input bits q0..q5


def run(lines, query, xq=(0, 1, 2, 3, 4, 5), rows=ROWS):
    ops = [B.parse_line(x) for x in lines]
    nq = 1 + max(max(o.q) for o in ops if o.q)
    values, _, _ = G.build(ops, list(xq), nq, None)
    return R.run_query(ops, list(xq), nq, values, query, rows=rows)


def q(target, avail, erase, restore, **kw):
    return dict(target_value_id=target, erase_after_op=erase, restore_before_op=restore, available_value_ids=avail,
                max_extra_qubits=0, max_and_cost=10, **kw)


class Determination(unittest.TestCase):
    def test_quadratic_target_is_determined_priced_exact_with_fragment(self):
        # q6 = x0*x1 + x2 (one AND); available x0, x1, x2
        lines = ['CCX q0 q1 q6', 'CX q2 q6', 'Z q6']
        r = run(lines, q('q6:v2', ['q0:v0', 'q1:v0', 'q2:v0'], 0, 3))
        self.assertTrue(r['determined'])
        self.assertEqual(r['and_cost'], 1)
        self.assertEqual(r['scratch_peak'], 0)
        self.assertEqual(r['price_status'], 'exact')
        self.assertEqual(r['phase_repair']['status'], 'unknown')
        frag = r['fragment']
        self.assertIsNotNone(frag)
        self.assertEqual(frag['exact_check']['ands'], 1)
        self.assertEqual(frag['exact_check']['rows_checked'], len(ROWS))
        self.assertTrue(r['feasible_within_budget'])

    def test_affine_target_costs_zero(self):
        r = run(['CX q0 q6', 'CX q1 q6', 'Z q6'], q('q6:v2', ['q0:v0', 'q1:v0'], 0, 3))
        self.assertTrue(r['determined'])
        self.assertEqual((r['and_cost'], r['price_status']), (0, 'exact'))
        self.assertEqual(sorted(r['dependency_ids']), ['q0:v0', 'q1:v0'])

    def test_missing_input_gives_a_reverified_witness(self):
        # q6 = x0 ^ x1 ^ x2 but only x0, x1 are available
        lines = ['CX q0 q6', 'CX q1 q6', 'CX q2 q6', 'Z q6']
        r = run(lines, q('q6:v3', ['q0:v0', 'q1:v0'], 0, 4))
        self.assertFalse(r['determined'])
        w = r['witness']
        self.assertTrue(w['reverified'])
        self.assertTrue(w['available_state_identical'])
        self.assertEqual(w['target_values'], [0, 1])
        x0, x1 = w['x_target0'], w['x_target1']
        self.assertEqual((x0 ^ x1) & 3, 0)          # same x0, x1 bits
        self.assertEqual((x0 ^ x1) >> 2 & 1, 1)     # differ in x2

    def test_hmr_kill_makes_the_erased_value_unavailable(self):
        # q6 = x0 AND x1, measured away; q6 itself is the target: determined from x0, x1; not from x0 alone
        lines = ['CCX q0 q1 q6', 'HMR q6 b1', 'Z q0']
        a = run(lines, q('q6:v1', ['q0:v0', 'q1:v0'], 1, 3))
        self.assertTrue(a['determined'])
        self.assertEqual(a['and_cost'], 1)
        b = run(lines, q('q6:v1', ['q0:v0'], 1, 3))
        self.assertFalse(b['determined'])

    def test_bare_target_and_target_removed_from_available(self):
        lines = ['CX q0 q6', 'Z q6']
        r = run(lines, q('q6', ['q6:v1', 'q0:v0'], 0, 2))
        self.assertTrue(r['determined'])
        self.assertEqual(r['target_value_ids'], ['q6:v1'])
        self.assertEqual(r['available_value_ids'], ['q0:v0'])
        self.assertTrue(any('target itself' in n for n in r['notes']))

    def test_cubic_target_is_bounded_not_exact(self):
        r = run(['CCX q0 q1 q6', 'CCX q6 q2 q7', 'Z q7'], q('q7:v1', ['q0:v0', 'q1:v0', 'q2:v0'], 0, 3))
        self.assertTrue(r['determined'])
        self.assertEqual(r['price_status'], 'bounded')
        self.assertGreaterEqual(r['and_cost'], 2)
        self.assertIsNone(r['fragment'])

    def test_budget_flag(self):
        r = run(['CCX q0 q1 q6', 'Z q6'], dict(q('q6:v1', ['q0:v0', 'q1:v0'], 0, 2), max_and_cost=0))
        self.assertTrue(r['determined'])
        self.assertFalse(r['feasible_within_budget'])


class Control(unittest.TestCase):
    SHA = '1c5b1ee282904faf4c1a09d2a440577de1866c47462ff7df9714768f9ed2a56a'
    LIVE = ['q13:v6', 'q4:v7', 'q5:v7', 'q6:v7', 'q7:v7', 'q8:v7', 'q9:v7', 'q10:v6', 'q11:v6', 'q12:v7', 'q0:v5',
            'q1:v7', 'q2:v9', 'q3:v7']

    def test_q23_at_pre_op_15918_is_not_determined(self):
        d = ROOT / 'work' / self.SHA[:12]
        if not (d / 'circuit.kmx').exists():
            self.skipTest('workspace of 1c5b1ee28290 not opened')
        ops, xq, nq = G.load_ops(d / 'circuit.kmx')
        values, _, _ = G.build(ops, xq, max(nq, 62), G.read_intervals(d))
        rows = np.random.default_rng(1).choice(np.arange(1, B.MAX_X + 1), 200_000, replace=False)
        r = R.run_query(ops, xq, max(nq, 62), values, dict(source_sha256=self.SHA, target_value_id='q23:v3',
                        erase_after_op=447, restore_before_op=15918, available_value_ids=['q23:v3'] + self.LIVE,
                        max_extra_qubits=0, max_and_cost=25), rows=rows)
        self.assertFalse(r['determined'])
        self.assertTrue(r['witness']['reverified'])
        self.assertEqual(r['evidence_tier'], 'exact on the supplied rows')
        self.assertEqual(len(r['available_value_ids']), 14)


if __name__ == '__main__':
    unittest.main()
