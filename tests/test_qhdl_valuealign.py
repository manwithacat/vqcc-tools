"""qhdl valuealign: synthetic parent/child streams, then the recorded parent-child / self / unrelated controls on real KMBs.
Static op-stream evidence (+ sampled signatures with --sig); a match is a correspondence, never an equivalence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qhdl_bitsim as B  # noqa: E402
import qhdl_valuealign as A  # noqa: E402

import vqcc_config as CFG  # noqa: E402
KMB = CFG.DATASET / 'circuits/invsqrt'


def side(lines, xq, it, rows=0):
    ops = [B.parse_line(x) for x in lines]
    nq = 1 + max(max(o.q) for o in ops if o.q)
    return A.Side(ops, list(xq), nq, it, rows)


def run(par, chi, xq=(0, 1, 2), sig=False):
    it = A.Interner()
    P, C = side(par, xq, it), side(chi, xq, it)
    res = A.align(P, C, it)
    rows, summ = A.build_rows(P, C, res)
    return P, C, res, {(r['side'], r['id']): r for r in rows}


PAR = ['CCX q0 q1 q3', 'CX q2 q3', 'CCX q3 q1 q4', 'CX q3 q5', 'CX q4 q5']


class Synthetic(unittest.TestCase):
    def test_renamed_reordered_copy_shares_normal_form(self):
        # child: wires renamed (q3->q6 ...), the two CX that feed q3 swapped in order
        chi = ['CCX q0 q1 q6', 'CX q2 q6', 'CCX q6 q1 q7', 'CX q6 q8', 'CX q7 q8']
        P, C, res, rows = run(PAR, chi)
        self.assertEqual(rows[('parent', 'q5:v2')]['matches'][0]['value'], 'q8:v2')
        self.assertEqual(rows[('parent', 'q4:v1')]['matches'][0]['value'], 'q7:v1')
        ref = ['CX q2 q3', 'CCX q0 q1 q3']
        chi2 = ['CCX q0 q1 q3', 'CX q2 q3']
        _, _, _, r2 = run(ref, chi2)
        self.assertEqual(r2[('parent', 'q3:v2')]['status'], 'matched')   # XOR-linear updates commute

    def test_self_alignment_matches_every_value(self):
        P, C, res, rows = run(PAR, PAR)
        self.assertEqual({r['status'] for k, r in rows.items() if k[0] == 'parent'}, {'matched'})
        self.assertFalse([k for k in rows if k[0] == 'child'])
        self.assertTrue(all(r['matches'][0]['value'] == r['id'] for k, r in rows.items() if k[0] == 'parent'))

    def test_hard_kill_in_child_is_erased_not_equivalent(self):
        chi = PAR[:4] + ['HMR q3 b0', 'CX q4 q5']
        P, C, res, rows = run(PAR, chi)
        r = rows[('parent', 'q3:v2')]
        self.assertEqual(r['status'], 'erased_in_child')
        self.assertEqual(r['erase']['mechanism'], 'hard_kill_in_child')
        self.assertEqual(r['erase']['child_kill']['kind'], 'measured')
        for x in rows.values():
            self.assertIn(x['status'], ('matched', 'erased_in_child', 'unknown', 'new_in_child'))
        self.assertNotIn('equivalent', json.dumps(list(rows.values())))

    def test_unrelated_circuit_is_unknown_not_forced(self):
        other = ['X q0', 'CX q0 q1', 'CCX q1 q2 q3', 'CCX q0 q3 q4', 'CX q4 q5']
        P, C, res, rows = run(PAR, other)
        pstat = [r['status'] for k, r in rows.items() if k[0] == 'parent' and r['kind'] != 'input']
        self.assertGreater(pstat.count('unknown'), len(pstat) // 2)
        self.assertTrue([k for k in rows if k[0] == 'child'])

    def test_signature_mode_matches_functionally_equal_values_and_flags_zeroing(self):
        # same function, different gate structure: q3 = x0 AND x1 either way; child then XORs q3 back to zero
        par = ['CCX q0 q1 q3', 'CX q0 q4', 'CX q2 q4', 'CX q4 q5']
        chi = ['CCX q1 q0 q6', 'CX q2 q7', 'CX q0 q7', 'CX q7 q8', 'CCX q0 q1 q6']
        it = A.Interner()
        P, C = side(par, (0, 1, 2), it, 256), side(chi, (0, 1, 2), it, 256)
        res = A.align(P, C, it, use_sig=True)
        rows, _ = A.build_rows(P, C, res)
        rows = {(r['side'], r['id']): r for r in rows}
        r = rows[('parent', 'q3:v1')]
        self.assertEqual(r['matches'][0]['value'], 'q6:v1')
        self.assertEqual(r['status'], 'erased_in_child')
        self.assertEqual(r['erase']['mechanism'], 'zeroed_in_child')
        self.assertEqual(r['erase']['tier'], 'sampled(256)')

    def test_write_out_files(self):
        P, C, res, rows = run(PAR, PAR)
        it_rows, summ = A.build_rows(P, C, res)
        s = A.summarize(P, C, it_rows, summ, res, 'p' * 64, 'c' * 64)
        with tempfile.TemporaryDirectory() as t:
            A.write_out(Path(t) / 'o', it_rows, s)
            lines = (Path(t) / 'o' / 'alignment.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), len(it_rows))
            sm = json.loads((Path(t) / 'o' / 'summary.json').read_text())
            self.assertEqual((sm['schema'], sm['evidence_tier']), (A.SCHEMA, 'static'))
            with self.assertRaises(FileExistsError):
                A.write_out(Path(t) / 'o', it_rows, s)


class RealControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for s in ('85ead04e2462', '7bde83f48462', '1c5b1ee28290'):
            if not (KMB / f'{s}.kmb').exists():
                raise unittest.SkipTest('lineage KMBs not on file')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = dict(os.environ, OMP_NUM_THREADS='2', QHDL_WORK=cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def align(self, a, b, *extra):
        out = Path(self.tmp.name) / f'{a}-{b}-{len(extra)}'
        cp = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'valuealign', '-c', str(KMB / f'{a}.kmb'),
                             '-c2', str(KMB / f'{b}.kmb'), '--out', str(out), *extra], capture_output=True, text=True,
                            env=self.env, timeout=1800)
        if cp.returncode != 0:
            raise AssertionError(cp.stderr[-2000:])
        rows = [json.loads(x) for x in (out / 'alignment.jsonl').read_text().splitlines()]
        return json.loads((out / 'summary.json').read_text()), rows

    def test_e02_eight_address_values_with_signatures(self):
        sm, rows = self.align('85ead04e2462', '7bde83f48462', '--sig', '1024')
        by = {r['id']: r for r in rows if r['side'] == 'parent'}
        # the two address values the child measures (HMR q20/q21 at ops 7972/7973): static hard kills
        for pid, cid, op in (('q20:v8', 'q20:v8', 7972), ('q21:v7', 'q21:v10', 7973)):
            r = by[pid]
            self.assertEqual((r['status'], r['matches'][0]['value'], r['erase']['mechanism']), ('erased_in_child', cid, 'hard_kill_in_child'))
            self.assertEqual(r['erase']['child_kill']['op'], op)
        # q15..q19 are erased (sampled all-zero) in the child at medium confidence
        for w, v in ((19, 9), (18, 9), (17, 10), (16, 10), (15, 9)):
            r = by[f'q{w}:v{v}']
            self.assertEqual((r['status'], r['erase']['mechanism']), ('erased_in_child', 'zeroed_in_child'))
            self.assertIn(r['confidence'], ('medium', 'high'))
        # q14 is only a low-confidence hint (the parent keeps it to op 25168)
        r = by['q14:v9']
        self.assertEqual(r['status'], 'matched')
        self.assertEqual(r['erase_hint']['mechanism'], 'zeroed_in_child')
        self.assertEqual(sm['evidence_tier'], 'static + sampled(1024) signature evidence')

    def test_e02_static_only_does_not_find_the_eight(self):
        sm, rows = self.align('85ead04e2462', '7bde83f48462')
        erased = [r['id'] for r in rows if r['side'] == 'parent' and r['status'] == 'erased_in_child']
        self.assertFalse({'q20:v8', 'q21:v7'} & set(erased))
        self.assertLess(sm['matched_total'] / sm['parent_values'], 0.05)

    def test_self_alignment_all_matched(self):
        sm, rows = self.align('1c5b1ee28290', '1c5b1ee28290')
        self.assertEqual(sm['parent_status'], {'matched': sm['parent_values']})
        self.assertEqual(sm['child_new_in_child'], 0)

    def test_unrelated_static_mostly_unknown(self):
        sm, rows = self.align('1c5b1ee28290', '85ead04e2462')
        self.assertGreater(sm['parent_status'].get('unknown', 0), 0.95 * sm['parent_values'])


if __name__ == '__main__':
    unittest.main()
