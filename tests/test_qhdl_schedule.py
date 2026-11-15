"""qhdl schedule: plateaus, hold, attribute, repairs on small synthetic circuits, and a positive control on
the public 62-qubit invsqrt 1c5b1ee28290 (recorded: A_c mean 239.75 with 23.0 repair; 28 plateau runs / 1,162 ops in
A_c, 37 repair blocks, no feasible avoid-the-recompute hold)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qhdl_schedule as S  # noqa: E402
import vqcc_config as CFG  # noqa: E402

T = S.Table.from_ops
PUSH, POP = 'PUSH_CONDITION_IF', 'POP_CONDITION'


def hmr(w, bit):
    return ('HMR', [w], bit, -1)


def push(bit):
    return (PUSH, [], -1, bit)


class Plateaus(unittest.TestCase):
    def test_runs_at_peak(self):
        # wires 0,1 always live; q2 live over ops 1..3 and 6..7 (width 3), q3 over op 2..3 (width 4 = peak)
        ops = [('X', [0]), ('X', [2]), ('CX', [2, 3]), ('CX', [2, 3]), ('CZ', [0, 1]), ('CZ', [0, 1]),
               ('X', [2]), ('X', [2])]
        ops[3:4] = [('HMR', [3], 0, -1)]
        t = T(ops, always_live=[0, 1])
        r = S.plateaus(t)
        self.assertEqual((r['peak'], r['level']), (4, 4))
        self.assertEqual([(x['start'], x['end']) for x in r['rows']], [(2, 3)])
        r3 = S.plateaus(t, level=3)
        self.assertEqual([(x['start'], x['end']) for x in r3['rows']], [(1, 1), (4, 7)])
        self.assertEqual(S.plateaus(t, 4, 8)['runs'], 0)         # range clipped away from the peak

    def test_clipped_runs(self):
        t = T([('X', [0])] * 6, always_live=[0])
        r = S.plateaus(t, 2, 4)
        self.assertEqual((r['runs'], r['ops']), (1, 2))


class Hold(unittest.TestCase):
    def setUp(self):
        # q5 measured at op 2; q6 is live over ops 4..7 and raises the width to the bound there
        self.t = T([('CX', [0, 5]), ('CX', [1, 5]), hmr(5, 0), ('X', [0]), ('X', [6]), ('X', [6]), ('X', [6]),
                    hmr(6, 1)], always_live=[0, 1, 2])

    def test_free_until_zeroed_then_costs_one(self):
        # widths: ops0-2: 4 (0,1,2,5), op3: 3, ops 4-7: 4
        self.assertEqual(list(self.t.prof), [4, 4, 4, 3, 4, 4, 4, 4])
        self.assertTrue(S.hold(self.t, [5], 0, 2, q=4)['feasible'])      # live and not yet zeroed
        r = S.hold(self.t, [5], 1, 3, q=4)                               # op 3 costs +1: 3 + 1 = 4 fits
        self.assertTrue(r['feasible'])
        r = S.hold(self.t, [5], 1, 5, q=4)                               # ops 4,5 are at 4 already: blocked
        self.assertFalse(r['feasible'])
        self.assertEqual([(x['start'], x['end']) for x in r['blocking_runs']], [(4, 5)])
        self.assertEqual(r['blocking_ops'], 2)
        self.assertEqual(r['max_width_needed'], 5)

    def test_bound_and_extra(self):
        self.assertTrue(S.hold(self.t, [5], 1, 5, q=5)['feasible'])
        self.assertFalse(S.hold(self.t, [5], 1, 3, q=4, extra=1)['feasible'])
        r = S.hold(self.t, [9], 3, 3, q=4)                               # never live: costs one everywhere
        self.assertFalse(r['wire_live_at_start']['q9'])
        self.assertTrue(S.hold(self.t, [9], 3, 3, q=4)['feasible'] is True)

    def test_second_value_on_a_reused_wire_is_not_free(self):
        t = T([('X', [3]), hmr(3, 0), ('X', [3]), ('X', [3]), ('X', [3]), hmr(3, 1)], always_live=[0])
        # q3 is re-written after the HMR at op 1; a held copy of the first value cannot ride on that wire
        self.assertEqual(list(t.prof), [2, 2, 2, 2, 2, 2])
        self.assertFalse(S.hold(t, [3], 0, 4, q=2)['feasible'])
        self.assertTrue(S.hold(t, [3], 0, 1, q=2)['feasible'])


def repair_circuit():
    """q0..q2 live inputs; q3 = a.b computed, measured; the repair block (opened on the outcome) recomputes it,
    applies CCZ(q3, c, d) and uncomputes."""
    return [('CCX', [0, 1, 3]), hmr(3, 7),                                # 0, 1
            push(7), ('CCX', [0, 1, 3]), ('CCZ', [3, 2, 4]), ('CCX', [0, 1, 3]), (POP, []),   # 2..6
            ('CX', [0, 2])]


class Attribute(unittest.TestCase):
    def test_categories(self):
        ops = [('CCX', [0, 1, 3]), ('CCX', [0, 1, 3]), hmr(3, 7),
               push(7), ('CCX', [0, 1, 3]), ('CCZ', [3, 2, 4]), ('CCX', [0, 1, 3]), (POP, []),     # repair block
               ('BIT_STORE1', [], 5), push(5), ('CCX', [0, 2, 3]), push(6), ('CCX', [0, 1, 3]), (POP, []), (POP, []),
               ('CCZ_IF', [0, 1, 2], -1, 5), ('CCX', [0, 1, 3])]
        t = T(ops, always_live=[0, 1, 2, 4])
        w = {i: 1.0 for i in range(len(ops))}
        for i in (4, 5, 6):
            w[i] = 0.5
        r = S.attribute_range(t, 0, len(ops), w, w)
        self.assertEqual((r['unconditional']['static'], r['conditional']['static'], r['repair']['static']), (3, 3, 3))
        self.assertAlmostEqual(r['repair']['mean'], 1.5)
        self.assertEqual(r['conditional']['nested'], 1)
        self.assertEqual(r['total']['static'], 9)
        r0 = S.attribute_range(t, 0, len(ops))
        self.assertNotIn('mean', r0['total'])

    def test_pair_difference_and_cx_ccx_cx_step(self):
        ops = [('CX', [9, 0]), ('CX', [9, 1]), ('CCX', [0, 1, 3]), ('CX', [9, 3]),            # one step
               ('CCX', [0, 1, 4])]
        t = T(ops, always_live=[0, 1, 9])
        res = S.attribute(t, {'A': (0, 4), 'B': (0, 5)})
        self.assertEqual(res['ranges']['A']['unconditional']['in_cx_ccx_cx_step'], 1)
        self.assertEqual(res['paired_difference']['unconditional']['static'], 1)

    def test_repair_needs_the_hmr_within_three_ops(self):
        ops = [hmr(3, 7), ('X', [0]), ('X', [0]), ('X', [0]), push(7), ('CCX', [0, 1, 4]), (POP, [])]
        t = T(ops, always_live=[0, 1])
        self.assertEqual(S.attribute_range(t, 0, 7)['repair']['static'], 0)
        self.assertEqual(S.attribute_range(t, 0, 7)['conditional']['static'], 1)


class Repairs(unittest.TestCase):
    def test_block_and_degree(self):
        t = T(repair_circuit(), always_live=[0, 1, 2, 4])
        rows = S.repairs(t, 0, t.n, q=5)
        self.assertEqual(len(rows), 1)
        b = rows[0]
        self.assertEqual((b['start'], b['end'], b['hmr'], b['hmr_wire'], b['toffolis']), (2, 6, 1, 3, 3))
        # q3 is dead at the block entry (zero): phase = (a.b).c.d on the live wires -> degree 4, one term
        self.assertEqual((b['phase']['computable'], b['phase']['degree'], b['phase']['terms']), (True, 4, 1))
        # the first CCX of the block repeats CCX(0,1->3) at op 0; q3 was measured at op 1, so holding it from op 1
        # to the block start costs a wire: width is 4 (0,1,2,4) + q3 live to op 1 -> +1 fits in 5, not in 4
        h = b['recompute_holds'][0]
        self.assertEqual((h['ccx'], h['earlier'], h['wire'], h['hmr']), (3, 0, 3, 1))
        self.assertTrue(h['feasible'])
        rows4 = S.repairs(t, 0, t.n, q=4)
        self.assertFalse(rows4[0]['recompute_holds'][0]['feasible'])  # the profile is 5 at op 1, above a bound of 4

    def test_hold_blocked_when_span_is_wider(self):
        ops = [('CCX', [0, 1, 3]), hmr(3, 7), ('X', [2]), ('X', [2]),
               push(7), ('CCX', [0, 1, 3]), ('CCZ', [3, 2, 4]), ('CCX', [0, 1, 3]), (POP, [])]
        t = T(ops, always_live=[0, 1, 2, 4])
        self.assertEqual(int(t.prof[2]), 4)
        h = S.repairs(t, 0, t.n, q=4)[0]['recompute_holds'][0]
        self.assertFalse(h['feasible'])
        self.assertGreater(h['blocking_ops'], 0)

    def test_phase_degrees(self):
        # linear phase: CZ(a, b) with both live -> degree 2; Z(a) -> degree 1
        for body, deg in ([('CZ', [0, 1])], 2), ([('Z', [0])], 1), ([('CCZ', [0, 1, 2]), ('CZ', [0, 1])], 3):
            t = T([hmr(5, 0), push(0)] + body + [(POP, [])], always_live=[0, 1, 2])
            self.assertEqual(S.phase_polynomial(t, 1, t.n)['degree'], deg)

    def test_cancelling_phase_is_zero(self):
        t = T([hmr(5, 0), push(0), ('CZ', [0, 1]), ('CZ', [0, 1]), (POP, [])], always_live=[0, 1])
        r = S.phase_polynomial(t, 1, t.n)
        self.assertEqual((r['computable'], r['degree'], r['phase_zero']), (True, None, True))

    def test_inner_hmr_and_nested_condition_enter_the_degree(self):
        # measure q0 inside the block: phase m' . q0 (degree 2: outcome variable times the value)
        t = T([hmr(5, 0), push(0), hmr(0, 1), (POP, [])], always_live=[0, 1])
        self.assertEqual(S.phase_polynomial(t, 1, t.n)['degree'], 2)
        # nested block on an outcome bit defined outside: gate under the nested bit adds that bit to the monomial
        t = T([hmr(5, 0), push(0), push(9), ('CZ', [0, 1]), (POP, []), (POP, [])], always_live=[0, 1])
        self.assertEqual(S.phase_polynomial(t, 1, t.n)['degree'], 3)

    def test_uncomputable_and_cap(self):
        t = T([hmr(5, 0), push(0), ('Z_POW', [0]), (POP, [])], always_live=[0])
        r = S.phase_polynomial(t, 1, t.n)
        self.assertFalse(r['computable'])
        self.assertIsNone(r['degree'])
        ops = [hmr(9, 0), push(0)] + [('CX', [i, i + 1]) for i in range(11)] + [('CCZ', [0, 5, 11]), (POP, [])]
        t = T(ops, always_live=list(range(12)))
        self.assertFalse(S.phase_polynomial(t, 1, t.n, cap=2)['computable'])
        self.assertTrue(S.phase_polynomial(t, 1, t.n)['computable'])


class Control1c5b(unittest.TestCase):
    """Positive control on maoxin123's 62-qubit invsqrt (needs the pulled dataset circuit; opens a qhdl workspace)."""
    SHA = '1c5b1ee28290'
    KMB = CFG.DATASET / 'circuits/invsqrt' / (SHA + '.kmb')

    @classmethod
    def setUpClass(cls):
        work = Path(os.environ.get('QHDL_WORK', ROOT / 'work'))
        cls.ws = work / cls.SHA
        if not cls.KMB.exists():
            raise unittest.SkipTest('1c5b1ee28290.kmb not available')
        env = dict(os.environ, OMP_NUM_THREADS='2')
        cp = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'schedule', 'attribute', '-c', str(cls.KMB),
                             '--range', 'A_b=23606:25567', '--range', 'A_c=27792:32341'],
                            capture_output=True, text=True, env=env, timeout=900)
        if cp.returncode != 0:
            raise AssertionError(cp.stderr[-2000:])
        cls.env_out = json.loads(cp.stdout)
        cls.ev = json.loads((cls.ws / 'schedule' / 'attribute.json').read_text())
        cls.t = S.Table.from_workspace(cls.ws)

    def test_attribution_matches_i10(self):
        self.assertTrue(cls_tier := self.env_out['tier'].startswith('sampled('), cls_tier)
        ac = self.ev['ranges']['A_c']
        self.assertEqual(ac['total']['static'], 308)
        self.assertAlmostEqual(ac['total']['mean'], 239.75, places=1)
        self.assertAlmostEqual(ac['repair']['mean'], 23.0, places=1)
        self.assertEqual((ac['repair']['static'], ac['unconditional']['static']), (46, 179))
        self.assertAlmostEqual(ac['conditional']['mean'], 37.75, places=1)
        self.assertEqual(self.ev['ranges']['A_b']['total']['mean'], 198.0)

    def test_plateaus_match_i12(self):
        r = S.plateaus(self.t, 27792, 32341)
        self.assertEqual((r['level'], r['runs'], r['ops']), (62, 28, 1162))

    def test_repairs_and_holds_match_i12(self):
        rows = S.repairs(self.t, 27792, 32341)
        self.assertEqual(len(rows), 37)
        holds = [h for b in rows for h in b['recompute_holds'] if 'feasible' in h]
        self.assertEqual(len(holds), 24)
        self.assertFalse(any(h['feasible'] for h in holds))
        self.assertTrue(all(b['phase']['computable'] for b in rows))

    def test_cli_verbs_run(self):
        env = dict(os.environ, OMP_NUM_THREADS='2')
        for args in (['plateaus', '-c', self.SHA, '--ops', '27792:32341'],
                     ['hold', '-c', self.SHA, '--wire', '41', '--from', '28057', '--to', '28123'],
                     ['repairs', '-c', self.SHA, '--ops', '27792:32341', '--no-exec']):
            cp = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'schedule', *args],
                                capture_output=True, text=True, env=env, timeout=600)
            self.assertEqual(cp.returncode, 0, cp.stderr[-1500:])
            d = json.loads(cp.stdout)
            self.assertIn('rerun', d)
            self.assertEqual(d['tier'], 'static')


if __name__ == '__main__':
    unittest.main()
