"""QHDL v1 rewrite registry on the real circuits (slow; minutes and several GB per DQI circuit). Run with
QHDL_SLOW_TESTS=1 and the system python3; skipped otherwise. Uses the default QHDL_WORK (shared caches) unless set.

Acceptance checks of the rewrite registry:
  - the semantic finder on busyaprime's 3779a3431e9c reports epoch 0's five blocks with the recorded spans and roles;
  - `rewrite --class and3-dirty-mbu` on yoyobar's 6d58bd6d0232 reproduces the recorded c253111974fc byte for byte;
  - `rewrite --class and-merge` on 67482db6c538 with the recorded replacement passes `fdiff --whole` at -2 Toffolis.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / 'scripts'))
HAS_NUMPY = importlib.util.find_spec('numpy') is not None
SLOW = os.environ.get('QHDL_SLOW_TESTS') == '1'
import vqcc_config as CFG  # noqa: E402
DQI = CFG.DATASET / 'circuits/dqi'
INV = CFG.DATASET / 'circuits/invsqrt'
AND3_MBU_SHA = 'c253111974fcd36937d3fa6eb480a573e4d2455d3f50054befcbe4675b4a9234'
ACCEPT = Path(os.environ.get('QHDL_TEST_FIXTURES') or ROOT / 'tests' / 'fixtures') / 'rewrite-acceptance'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def qhdl(*args, env=None):
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), *args], capture_output=True, text=True,
                         env=env or dict(os.environ), timeout=3600)
    return json.loads(out.stdout)


@unittest.skipUnless(HAS_NUMPY and SLOW, 'slow real-circuit tests: set QHDL_SLOW_TESTS=1 (NumPy needed)')
class Real(unittest.TestCase):
    @unittest.skipUnless((DQI / '3779a3431e9c.kmb').exists(), 'public circuit dataset not staged')
    def test_3779_epoch0_blocks(self):
        import qhdl_kmb as K
        import qhdl_rewrite as R
        k = K.read(str(DQI / '3779a3431e9c.kmb'))
        bl = R.pair_blocks(R.find_blocks(R.Ops(k), 56239607, 56243627))
        got = [(b['start'], b['end'], b['kind'], b['n'], b['toffolis'], b['target'], b['assumed_zero']) for b in bl]
        self.assertEqual(got, [(56240076, 56240128, 'chain', 12, 36, 845, [890, 891]),
                               (56240231, 56240253, 'chain', 6, 14, 890, [891]),
                               (56240280, 56240284, 'chain', 3, 4, 891, []),
                               (56242166, 56242171, 'mbu-partial', 3, 3, 891, []),
                               (56242202, 56242224, 'chain', 6, 14, 890, [891])])
        self.assertEqual([w for w, _ in bl[0]['controls']], [886, 867, 884, 851, 854, 866, 861, 885, 873, 848, 870, 868])
        self.assertEqual(bl[4]['partner']['start'], 56240231)
        self.assertEqual(R.TagIndex(k).rel(56240076), 'tag:unextract/companion_normalize[0]+469')

    @unittest.skipUnless((DQI / '6d58bd6d0232.kmb').exists(), 'public circuit dataset not staged')
    def test_hw06_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'hw06.kmb'
            e = qhdl('rewrite', '-c', '6d58bd6d0232', '--class', 'and3-dirty-mbu', '--out', str(out), '--certify', 'none')
            self.assertIn('12 site(s) of 12 fitting', e['claim'])
            self.assertEqual(sha(out), AND3_MBU_SHA)

    @unittest.skipUnless((INV / '67482db6c538.kmb').exists() and (ACCEPT / '04-and-merge-replacement.txt').exists(),
                         'public circuit dataset or the recorded replacement not available')
    def test_and_merge_fdiff_whole(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'merge.kmb'
            e = qhdl('rewrite', '-c', '67482db6c538', '--class', 'and-merge', '--replacement',
                     str(ACCEPT / '04-and-merge-replacement.txt'), '--out', str(out))
            self.assertEqual(e['tier'], 'exact-all-x')
            self.assertIn('EQUIVALENT', e['claim'])
            cert = json.loads(out.with_name(out.name + '.certificate.json').read_text())
            self.assertTrue(cert['observed']['equivalent'])
            z = cert['observed']['mean_toffoli']
            self.assertAlmostEqual(z[0] - z[1], 2.0, places=3)


if __name__ == '__main__':
    unittest.main()
