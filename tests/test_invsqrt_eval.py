"""Tests for scripts/invsqrt_eval.py. Fast tests need NumPy only; the slow ones run the engine on a public 62-qubit
invsqrt circuit and are skipped unless VQCC_SLOW=1:

    $QHDL_KICKMIX_PYTHON -I tests/test_invsqrt_eval.py -v
    VQCC_SLOW=1 $QHDL_KICKMIX_PYTHON -I tests/test_invsqrt_eval.py -v   # ~1 min
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import invsqrt_eval as E  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import vqcc_config as CFG  # noqa: E402
PICK = CFG.DATASET / 'circuits/invsqrt/19a0386ae02f.kmb'
FIXTURES = Path(os.environ.get('QHDL_TEST_FIXTURES') or ROOT / 'tests' / 'fixtures')  # optional recorded fixtures
PICK_REPORT = FIXTURES / 'invsqrt_19a0386ae02f/report.json'  # a recorded official-verifier report
SLOW = os.environ.get('VQCC_SLOW') == '1'


class Salts(unittest.TestCase):
    def test_n_salts_start_with_default(self):
        a = argparse.Namespace(salts=4, salt_list=None, salt_format='sweep-{}')
        self.assertEqual(E.salts_from(a), ['', 'sweep-1', 'sweep-2', 'sweep-3'])

    def test_explicit_list_keeps_empty_default(self):
        a = argparse.Namespace(salts=None, salt_list=',a,b', salt_format='sweep-{}')
        self.assertEqual(E.salts_from(a), ['', 'a', 'b'])


class SeedsDigest(unittest.TestCase):
    def row(self, salt, status, score=1.0, ratio=0.5, x=3, eps=1e-13):
        r = dict(salt=salt, status=status)
        if status == 'PASS':
            r.update(score=score, worst_error_over_tolerance=ratio, worst_x=x, rotation_eps=eps)
        else:
            r['error'] = 'x=1: phase error'
        return r

    def test_first_fail_and_worst_case(self):
        rows = [self.row('', 'PASS', 10.0, 0.9, 7), self.row('s1', 'PASS', 11.0, 0.95, 5, 2e-13),
                self.row('s2', 'FAIL')]
        d = E.seeds_digest(rows, ['', 's1', 's2', 's3'])
        self.assertEqual(d['first_fail']['salts_to_catch'], 3)
        self.assertFalse(d['all_pass'])
        self.assertEqual(d['worst_case']['max_cost_salt'], 's1')
        self.assertEqual(d['worst_case']['at_x'], 5)
        self.assertEqual(d['worst_case']['min_rotation_eps'], 1e-13)

    def test_all_pass_needs_every_salt_run(self):
        d = E.seeds_digest([self.row('', 'PASS')], ['', 's1'])
        self.assertFalse(d['all_pass'])
        d = E.seeds_digest([self.row('', 'PASS')], [''])
        self.assertTrue(d['all_pass'])


@unittest.skipIf(np is None, 'NumPy required')
class PerX(unittest.TestCase):
    def test_fold_wraps_at_max_x_and_records_salt(self):
        maxx = 5
        w = np.zeros(maxx, dtype=np.uint64)
        ws = np.full(maxx, -1, dtype=np.int32)
        # shots 3..9 -> x = 4,5,1,2,3,4,5
        E.fold_per_x(w, ws, 0, np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint64), 3, maxx)
        self.assertEqual(w.tolist(), [3, 4, 5, 6, 7])
        E.fold_per_x(w, ws, 1, np.array([9, 0, 0, 0, 0], dtype=np.uint64), 0, maxx)
        self.assertEqual(w.tolist(), [9, 4, 5, 6, 7])
        self.assertEqual(ws.tolist(), [1, 0, 0, 0, 0])

    def test_digest_histogram_and_extremes(self):
        tol = np.array([100, 100, 1000, 100, 100], dtype=np.uint64)
        w = np.array([10, 95, 999, 100, 101], dtype=np.uint64)
        ws = np.array([0, 0, 1, 0, 1], dtype=np.int32)
        d = E.per_x_digest(np, dict(worst_used=w, worst_salt=ws), tol, ['', 'sweep-1'], top=3)
        self.assertEqual(d['worst_ratio_x'], 5)
        self.assertEqual(d['worst_ratio_salt'], 'sweep-1')
        self.assertEqual(d['min_margin_units'], -1)
        counts = {h['bin']: h['x_count'] for h in d['histogram']}
        self.assertEqual(counts['[0.0, 0.5)'], 1)
        self.assertEqual(counts['[0.9, 0.99)'], 1)
        self.assertEqual(counts['[0.999, 0.9999)'], 1)
        self.assertEqual(counts['[0.9999, 1.0]'], 1)
        self.assertEqual(counts['> 1.0 (out of tolerance)'], 1)
        self.assertEqual(sum(counts.values()), 5)
        self.assertEqual([t['x'] for t in d['top']], [5, 4, 3])

    def test_half_turn_error_margin_does_not_overflow(self):
        tol = np.array([100, 100], dtype=np.uint64)
        w = np.array([2**63, 50], dtype=np.uint64)
        d = E.per_x_digest(np, dict(worst_used=w, worst_salt=np.zeros(2, dtype=np.int32)), tol, [''], top=2)
        self.assertEqual(d['min_margin_x'], 1)
        self.assertLess(d['min_margin_units'], -(2**62))

    def test_first_failure_priority_within_batch(self):
        n = 8
        flat_x = np.arange(1, n + 1, dtype=np.uint64)
        z = np.zeros(n, dtype=bool)
        bad_ph = z.copy(); bad_ph[1] = True
        dirty = z.copy(); dirty[5] = True
        used = np.full(n, 2**63, dtype=np.uint64)
        f = E.first_failure(flat_x, z, dirty, bad_ph, used, np.ones(n, dtype=np.uint64), 0, 4)
        self.assertEqual(f['kind'], 'phase_out_of_tolerance')  # batch 0 has only the phase failure
        f = E.first_failure(flat_x, z, dirty, bad_ph, used, np.ones(n, dtype=np.uint64), 0, 8)
        self.assertEqual(f['kind'], 'scratch_dirty')           # one batch: scratch outranks phase


class Compare(unittest.TestCase):
    def test_pass_fields_and_subset(self):
        off = {k: 1 for k in E.SUMMARY_FIELDS}
        off['status'] = 'PASS'
        eng = dict(off)
        self.assertTrue(E.compare_summaries(eng, off)['match'])
        eng['score'] = 2
        r = E.compare_summaries(eng, off)
        self.assertFalse(r['match'])
        self.assertIn('score', r['differences'])
        sub = {k: off[k] for k in ('status', 'score')}
        self.assertTrue(E.compare_summaries(sub, off)['match'])

    def test_fail_compares_message(self):
        off = {'status': 'FAIL', 'error': 'x was not restored.'}
        self.assertTrue(E.compare_summaries(dict(off, salt=''), off)['match'])
        self.assertFalse(E.compare_summaries({'status': 'FAIL', 'error': 'other'}, off)['match'])


@unittest.skipIf(np is None or not PICK.exists(), 'pick circuit not present')
class Fixtures(unittest.TestCase):
    def test_rare_branch_fixture_ops(self):
        import qhdl_kmb as K
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'rb.kmb'
            E.main(['fixture', 'rare-branch', str(PICK), '--k', '5', '--out', str(out)])
            a, b = K.read(str(PICK)), K.read(str(out))
            self.assertEqual(len(b.op_types), len(a.op_types) + 3 * 5 + 1)
            self.assertEqual(int(b.num_bits), int(a.num_bits) + 5)
            self.assertEqual(int(b.num_qubits), int(a.num_qubits))
            self.assertTrue((b.op_types[:len(a.op_types)] == a.op_types).all())
            with self.assertRaises(SystemExit):  # append-only: never overwrite
                E.main(['fixture', 'rare-branch', str(PICK), '--k', '5', '--out', str(out)])


@unittest.skipUnless(SLOW and PICK.exists() and PICK_REPORT.exists(), 'slow engine test: set VQCC_SLOW=1')
class SlowEngine(unittest.TestCase):
    def test_verify_equals_recorded_official(self):
        eng = E.Engine()
        s = eng.verify(PICK, '', threads=4)
        r = E.compare_summaries(s, E.official_summary(PICK_REPORT))
        self.assertTrue(r['match'], r)

    def test_margin_summary_equals_verify(self):
        eng = E.Engine()
        prog, sha, _ = eng.load(PICK)
        acc = dict(worst_used=np.zeros(eng.spec.max_x, dtype=np.uint64),
                   worst_salt=np.full(eng.spec.max_x, -1, dtype=np.int32), salt_index=0)
        row = E.margin_one_salt(eng, prog, sha, 'sweep-1', 4, acc)
        s = eng.verify(PICK, 'sweep-1', threads=4)
        r = E.compare_summaries(row, s)
        self.assertTrue(r['match'], r)


if __name__ == '__main__':
    unittest.main()
