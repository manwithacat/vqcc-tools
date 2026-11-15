"""qhdl retain: synthetic circuits only (no recorded evidence needed). NumPy needed."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
CCX, CX, HMR, XG, CSWAP, SWAP = 0x31, 0x21, 0x50, 0x11, 0x32, 0x22
NQ = 40


def write(path, ops):
    import numpy as np
    import qhdl_kmb as KM
    ot = np.array([o for o, _ in ops], dtype=np.uint8)
    s = {n: [] for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0']}
    nb = 0
    for o, q in ops:
        c = o & 0x30
        if c == 0x30:
            for n, v in zip(['qqq0', 'qqq1', 'qqq2'], q):
                s[n].append(v)
        elif c == 0x20:
            for n, v in zip(['qq0', 'qq1'], q):
                s[n].append(v)
        elif c == 0x10:
            s['q0'].append(q[0])
        if o & 0x40:
            s['b0'].append(nb)
            nb += 1
    KM.write_kmb(str(path), ot, s, num_qubits=NQ, num_bits=max(nb, 1))
    return str(path)


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class Detector(unittest.TestCase):
    def setUp(self):
        import qhdl_kmb as KM
        import qhdl_retain as R
        self.KM, self.R = KM, R
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_ops(self, ops):
        p = write(Path(self.tmp.name) / 'c.kmb', ops)
        k = self.KM.read(p)
        return self.R.analyse(k, 0, len(ops))

    def kinds(self, r):
        return [(s['kind'], s['free'], s['price']) for s in r['sites']]

    def test_recompute_after_hmr_is_free_when_holder_untouched(self):
        r = self.run_ops([(CCX, (0, 1, 10)), (HMR, (10,)), (CX, (2, 3)), (CCX, (0, 1, 11))])
        self.assertEqual(self.kinds(r), [('retain', True, 1)])
        self.assertEqual(r['sites'][0]['removal'], 'hmr')

    def test_recompute_blocked_when_holder_reused_and_no_spare(self):
        r = self.run_ops([(CCX, (0, 1, 10)), (HMR, (10,)), (XG, (10,)), (CCX, (0, 1, 11))])
        self.assertEqual(self.kinds(r), [('retain', False, 1)])

    def test_spare_zero_wire_frees_a_reused_holder(self):
        r = self.run_ops([(HMR, (20,)), (CCX, (0, 1, 10)), (HMR, (10,)), (XG, (10,)), (CCX, (0, 1, 11))])
        self.assertEqual(self.kinds(r), [('retain', True, 1)])
        self.assertEqual(r['sites'][0]['spare'], 1)

    def test_changed_control_is_not_a_recompute(self):
        r = self.run_ops([(CCX, (0, 1, 10)), (HMR, (10,)), (CX, (2, 0)), (CCX, (0, 1, 11))])
        self.assertEqual(r['sites'], [])
        self.assertEqual(r['counts']['first'], 2)

    def test_affine_equal_controls_match(self):
        # wire 5 reset to |0> then copied from wire 0, so it carries the same form; its product is the same product
        r = self.run_ops([(HMR, (5,)), (CX, (0, 5)), (CCX, (0, 1, 10)), (HMR, (10,)), (CCX, (5, 1, 11))])
        self.assertEqual([s['kind'] for s in r['sites']], ['retain'])

    def test_pure_measure_out_needs_no_spare(self):
        r = self.run_ops([(HMR, (10,)), (CCX, (0, 1, 10)), (CX, (2, 3)), (CCX, (0, 1, 10))])
        self.assertEqual(self.kinds(r), [('measure-out', True, 1)])
        self.assertEqual(r['sites'][0]['target_form'], 'pure')

    def test_mixed_measure_out_needs_a_spare_wire(self):
        mixed = [(CX, (2, 10)), (CCX, (0, 1, 10)), (CX, (3, 4)), (CCX, (0, 1, 10))]
        self.assertEqual(self.kinds(self.run_ops(mixed)), [('measure-out', False, 1)])
        self.assertEqual(self.kinds(self.run_ops([(HMR, (20,))] + mixed)), [('measure-out', True, 1)])
        touched = [(HMR, (20,)), (CX, (2, 10)), (CCX, (0, 1, 10)), (XG, (20,)), (CCX, (0, 1, 10))]
        self.assertEqual(self.kinds(self.run_ops(touched)), [('measure-out', False, 1)])

    def test_ccx_removal_prices_two(self):
        r = self.run_ops([(CCX, (0, 1, 10)), (CCX, (0, 1, 10)), (CCX, (0, 1, 11))])
        self.assertEqual([(s['kind'], s['price']) for s in r['sites']], [('measure-out', 1), ('retain', 2)])

    def test_pack_does_not_double_book_a_spare_wire(self):
        # two overlapping mixed intervals, only one spare wire (20)
        ops = [(HMR, (20,)), (CX, (2, 10)), (CX, (2, 11)), (CCX, (0, 1, 10)), (CCX, (3, 4, 11)),
               (CX, (5, 6)), (CCX, (0, 1, 10)), (CCX, (3, 4, 11))]
        r = self.run_ops(ops)
        free = [s for s in r['sites'] if s['free']]
        self.assertEqual(len(free), 2)
        chosen, price = self.R.pack(r['sites'])
        self.assertEqual((len(chosen), price), (1, 1))

    def test_cswap_product_is_matched(self):
        r = self.run_ops([(CSWAP, (0, 1, 2)), (CSWAP, (0, 1, 2))])
        self.assertEqual(r['counts']['off'], 1)

    def test_deterministic(self):
        ops = [(CCX, (0, 1, 10)), (HMR, (10,)), (CCX, (0, 1, 11))]
        a, b = self.run_ops(ops), self.run_ops(ops)
        self.assertEqual(json.dumps(a['sites'], sort_keys=True, default=str), json.dumps(b['sites'], sort_keys=True, default=str))


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class Verb(unittest.TestCase):
    def test_envelope_on_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(Path(d) / 'c.kmb', [(CCX, (0, 1, 10)), (HMR, (10,)), (CCX, (0, 1, 11))])
            out = Path(d) / 'ev'
            env = dict(os.environ, QHDL_WORK=str(Path(d) / 'work'))
            r = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'retain', '-c', p, '--task', 'invsqrt', '--out', str(out)],
                               capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            env_json = json.loads(r.stdout)
            self.assertEqual(env_json['tier'], 'static')
            self.assertEqual(env_json['facts']['free_total'], 1)
            ev = json.loads((out / 'retain.json').read_text())
            self.assertEqual(ev['sites'][0]['kind'], 'retain')


if __name__ == '__main__':
    unittest.main()
