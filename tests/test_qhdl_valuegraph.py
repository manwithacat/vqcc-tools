"""qhdl valuegraph: synthetic op streams, the HMR kill record and a control on the public 62-Q circuit 1c5b1ee28290
(op 456 `HMR q42 b483`: explicit kill, outcome value b483:v1, phase obligation unknown, no cone/recompute claim on q23)."""
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
import qhdl_valuegraph as G  # noqa: E402
import vqcc_config as CFG  # noqa: E402


def graph(lines, xq=(), intervals=None):
    ops = [B.parse_line(x) for x in lines]
    nq = 1 + max(max(o.q) for o in ops if o.q)
    v, u, info = G.build(ops, list(xq), nq, intervals)
    return v, u, info


class Synthetic(unittest.TestCase):
    def test_in_place_versions_and_cuts(self):
        v, u, _ = graph(['X q0', 'CX q0 q1', 'CCX q0 q1 q2', 'Z q2'], xq=[0])
        self.assertEqual(v['q0:v0']['kind'], 'input')
        self.assertEqual(v['q0:v1']['producer']['inputs'], ['q0:v0'])          # X: new version of the same wire
        self.assertEqual(v['q0:v1']['birth_cut'], 1)
        self.assertEqual(v['q0:v0']['kill_cut'], 1)
        self.assertEqual(v['q0:v0']['kill']['kind'], 'superseded_in_place')
        self.assertEqual(v['q1:v1']['producer']['inputs'], ['q1:v0', 'q0:v1'])  # control version used
        self.assertEqual([c['role'] for c in v['q0:v1']['consumers']], ['control', 'control'])
        self.assertEqual(v['q0:v1']['last_use_cut'], 2)
        # diagonal op: phase use, no new version
        self.assertNotIn('q2:v2', v)
        self.assertEqual(v['q2:v1']['consumers'][0]['role'], 'phase')
        self.assertEqual(v['q2:v1']['kill']['kind'], 'live_at_end')
        self.assertTrue(all(x['expression'] == 'unknown' for x in v.values()))

    def test_hmr_is_kill_outcome_and_obligation(self):
        v, u, info = graph(['CX q0 q1', 'HMR q1 b5', 'PUSH_CONDITION if b5', 'Z q0', 'POP_CONDITION', 'CX q0 q1'])
        k = v['q1:v1']
        self.assertEqual((k['kill_cut'], k['kill']['kind'], k['kill']['outcome_value']), (2, 'measured', 'b5:v1'))
        self.assertEqual(k['kill']['phase_obligation']['status'], 'unknown')
        self.assertEqual(k['kill']['phase_obligation']['guarded_blocks'], [dict(push_op=2, pop_op=4)])
        o = v['b5:v1']
        self.assertEqual((o['kind'], o['producer']['opcode'], o['outcome_of']), ('classical', 'HMR', 'q1:v1'))
        self.assertEqual(o['last_use_cut'], 4)                                  # held to the end of the block
        z = v['q1:v2']
        self.assertEqual((z['known_value'], z['post_reset']), (0, True))
        # guard ids on the guarded phase op
        zu = [x for x in u if x['opcode'] == 'Z']
        self.assertEqual(zu[0]['guards'], ['b5:v1'])
        self.assertEqual(zu[0]['condition_depth'], 1)

    def test_guarded_write_keeps_old_value_and_conditional_reset(self):
        v, u, _ = graph(['HMR q0 b1', 'X q1 if b1', 'R q2 if b1', 'R q2'])
        self.assertEqual(v['q1:v1']['producer']['inputs'], ['q1:v0', 'b1:v1'])
        self.assertTrue(v['q1:v1']['producer']['conditional'])
        self.assertIsNone(v['q2:v0']['kill_cut'])                              # guarded reset is not a certain kill
        self.assertIn('conditional_kill', v['q2:v0'])

    def test_swap_and_carriers(self):
        iv = [('q0#0', 0, 0, 2), ('q1#0', 1, 0, 2)]
        v, u, _ = graph(['CX q0 q1', 'SWAP q0 q1', 'R q1'], intervals=iv)
        self.assertEqual(v['q0:v1']['producer']['inputs'], ['q0:v0', 'q1:v1'])
        self.assertEqual(v['q1:v0']['carrier']['interval'], 'q1#0')
        self.assertEqual(v['q0:v1']['carrier']['interval'], 'q0#0')
        # semantic lifetime differs from the carrier interval: v0 ends at op 0, the carrier runs to op 2
        self.assertEqual(v['q1:v0']['kill_cut'], 1)
        self.assertEqual(v['q1:v0']['carrier']['end_op'], 2)

    def test_export_files(self):
        ops = [B.parse_line(x) for x in ['CX q0 q1', 'HMR q1 b0']]
        with tempfile.TemporaryDirectory() as t:
            man, v, u = G.export(ops, [0], 2, [], 'f' * 64, Path(t) / 'g')
            self.assertEqual((man['schema'], man['evidence_tier'], man['num_ops']), (G.SCHEMA, 'static', 2))
            rows = [json.loads(x) for x in (Path(t) / 'g' / 'values.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), man['value_count'])
            self.assertEqual(len(u), man['use_count'])

    def test_live_at(self):
        v, _, _ = graph(['CX q0 q1', 'CX q0 q1', 'CX q0 q1'], xq=[0])
        ids = {x['id'] for x in G.live_at(v, 1, 2)}
        self.assertIn('q0:v0', ids)
        self.assertNotIn('q1:v1', ids)                                         # killed at cut 2


class Pick(unittest.TestCase):
    SHA = '1c5b1ee28290'
    KMB = CFG.DATASET / 'circuits/invsqrt/1c5b1ee28290.kmb'

    @classmethod
    def setUpClass(cls):
        if not cls.KMB.exists():
            raise unittest.SkipTest('pick KMB not on file')
        cls.tmp = tempfile.TemporaryDirectory()
        env = dict(os.environ, OMP_NUM_THREADS='2', QHDL_WORK=cls.tmp.name)
        out = Path(cls.tmp.name) / 'g'
        cp = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'valuegraph', 'export', '-c', str(cls.KMB),
                             '--out', str(out)], capture_output=True, text=True, env=env, timeout=900)
        if cp.returncode != 0:
            raise AssertionError(cp.stderr[-2000:])
        cls.out = out
        cls.V = {}
        for line in (out / 'values.jsonl').read_text().splitlines():
            r = json.loads(line)
            cls.V[r['id']] = r

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_manifest(self):
        m = json.loads((self.out / 'manifest.json').read_text())
        self.assertEqual((m['num_qubits'], m['num_ops'], m['evidence_tier']), (62, 53818, 'static'))
        self.assertTrue(m['source_sha256'].startswith('1c5b1ee28290'))

    def test_op456_hmr_is_a_kill_with_outcome_and_obligation(self):
        k = self.V['q42:v10']
        self.assertEqual(k['kill']['op'], 456)
        self.assertEqual((k['kill']['kind'], k['kill']['outcome_value']), ('measured', 'b483:v1'))
        self.assertEqual(k['kill_cut'], 457)
        self.assertEqual(k['kill']['phase_obligation']['status'], 'unknown')
        o = self.V['b483:v1']
        self.assertEqual((o['producer']['op'], o['producer']['opcode'], o['outcome_of']), (456, 'HMR', 'q42:v10'))

    def test_q23_has_no_cone_or_recompute_claim(self):
        q23 = [r for r in self.V.values() if r.get('wire') == 23]
        self.assertTrue(q23)
        for r in q23:
            self.assertEqual(r['expression'], 'unknown')
            self.assertFalse([k for k in r if any(s in k for s in ('cone', 'inverse', 'recompute', 'determined'))])


if __name__ == '__main__':
    unittest.main()
