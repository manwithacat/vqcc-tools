"""Read-only QHDL predicate contracts, checked independently at source cuts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _optional import requires_kickmix  # noqa: E402
import qhdl_edit as ED
import qhdl_kmb as K
import qhdl_rewrite as R
import qhdl_rewrite_phase_fold as P
import qhdl_optimizations as O
import qhdl_predicates as QP


def build(directory, lines):
    empty = directory / 'empty.kmb'
    K.write_kmb(str(empty), np.array([], np.uint8), {},
                registers=[('x', [(True, i) for i in range(3)])],
                num_qubits=3, num_bits=0)
    path = directory / 'fixture.kmb'
    ED.edit_file(empty, 'insert 0 => ' + '; '.join(lines), path)
    return path


class PredicateIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def scan(self, lines, **kwargs):
        path = build(self.directory, lines)
        k = K.read(str(path))
        folds, stats = P.scan(k, R.Ops(k), include_predicates=True, **kwargs)
        plain, plain_stats = P.scan(k, R.Ops(k), **kwargs)
        self.assertEqual(folds, plain)
        self.assertEqual({k: v for k, v in stats.items() if k != 'predicates'}, plain_stats)
        return stats['predicates']

    def test_basis_pullback_and_measurement_condition_reset_cuts(self):
        rows = self.scan(['Z q0', 'CX q0 q1', 'Z q1', 'Z q0 if b0',
                          'Z q1', 'HMR q2 b1', 'Z q1', 'R q2', 'Z q1'])
        self.assertEqual([r['entry_cut'] for r in rows], [0, 0, 4, 6, 8])
        self.assertEqual([r['region_end'] for r in rows], [3, 3, 5, 7, 9])
        self.assertEqual(rows[1]['predicate'], ['0x1', '0x2'])
        contracts = [O.predicate_contract('a' * 64, row) for row in rows]
        phases = [QP.ExactPhaseTerm.from_dict(c) for c in contracts]
        self.assertEqual([p.to_dict() for p in phases], contracts)
        # The same physical wire at different cuts is not an equality proof.
        with self.assertRaises(ValueError):
            QP.phase_equivalent([phases[2]], [phases[3]])

    def test_span_and_capacity_cuts_preserve_prior_scope(self):
        rows = self.scan(['Z q1', 'X q0', 'Z q0'], max_terms=1)
        self.assertEqual([(r['entry_cut'], r['region_end'], r['cut_reason']) for r in rows],
                         [(0, 1, 'capacity'), (2, 3, 'query_end')])
        rows = self.scan(['Z q0', 'Z q1', 'Z q2'], max_span=1)
        self.assertEqual([r['entry_cut'] for r in rows], [0, 1, 2])

    def test_stack_conditions_are_not_reported_as_unconditional(self):
        rows = self.scan(['PUSH_CONDITION if b0', 'Z q0', 'POP_CONDITION', 'Z q1'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['ops'], [3])
        self.assertEqual(rows[0]['entry_cut'], 3)

    def test_exact_low_angle_bits_survive_adapter(self):
        row = dict(predicate=['0x1'], entry_cut=7, angle128=f'{(1 << 126) + 1:032x}')
        phase = QP.ExactPhaseTerm.from_dict(O.predicate_contract('a' * 64, row))
        self.assertEqual(phase.angle128, (1 << 126) + 1)

    @requires_kickmix
    def test_real_cli_catalog_and_profile(self):
        path = build(self.directory, ['CX q0 q1', 'Z q1'])
        before = path.read_bytes()
        env = dict(os.environ, QHDL_WORK=str(self.directory / 'work'))
        env.pop('QHDL_INVESTIGATION', None)
        for args in [('predicate-families', '--all'),
                     ('phase-profile', '-c', str(path), '--ops', '0:2', '--all')]:
            proc = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), *args],
                                  env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            envelope = json.loads(proc.stdout)
            self.assertEqual(envelope['tier'], 'static')
            evidence = Path(envelope['evidence'])
            if not evidence.is_absolute():
                evidence = ROOT / evidence
            payload = json.loads(evidence.read_text())
            if args[0] == 'phase-profile':
                self.assertEqual(len(payload['predicates']), 1)
                phase = QP.ExactPhaseTerm.from_dict(payload['predicates'][0]['contract'])
                self.assertEqual(phase.to_dict(), payload['predicates'][0]['contract'])
            else:
                self.assertGreaterEqual(len(payload['families']), 5)
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
