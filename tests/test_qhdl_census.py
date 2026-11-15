"""Tool round 3 (2026-11-01): qhdl census (dirty-borrow chains, provably-|0> targets), the track macros it relies on,
qhdl doctor (engine export check) and qhdl simulate (bit-parallel value simulation), on small synthetic circuits."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _optional import requires_kickmix  # noqa: E402
import qhdl_census as C  # noqa: E402
import qhdl_doctor as D  # noqa: E402
import qhdl_kmb as K  # noqa: E402
import qhdl_simulate as SIM  # noqa: E402
import qhdl_track as T  # noqa: E402

CCX, CZ, X, CX, HMR, CCZ_IF, H = 0x31, 0x20, 0x11, 0x21, 0x50, 0xB0, None


def build(path, ops, registers, num_qubits=None, num_bits=None):
    """ops: [(code, qubits tuple, b0 or None, bc or None)]."""
    st = {n: [] for n in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')}
    codes = []
    for code, q, b0, bc in ops:
        codes.append(code)
        if len(q) == 3:
            for n, v in zip(('qqq0', 'qqq1', 'qqq2'), q):
                st[n].append(v)
        elif len(q) == 2:
            st['qq0'].append(q[0])
            st['qq1'].append(q[1])
        elif len(q) == 1:
            st['q0'].append(q[0])
        if b0 is not None:
            st['b0'].append(b0)
        if bc is not None:
            st['bc'].append(bc)
    K.write_kmb(str(path), np.array(codes, np.uint8), {k: np.array(v, np.uint32) for k, v in st.items()},
                registers=registers, num_qubits=num_qubits, num_bits=num_bits)
    return path


def op(code, *q, b0=None, bc=None):
    return (code, tuple(q), b0, bc)


def and3_dirty(a, b, c, d, t):
    return [op(CCX, a, b, d), op(CCX, d, c, t), op(CCX, a, b, d), op(CCX, d, c, t)]


def barenco4(y1, y2, y3, y4, d1, d2, t):
    c1, c2, tt = op(CCX, y1, y2, d1), op(CCX, d1, y3, d2), op(CCX, d2, y4, t)
    return [c1, c2, tt, c2, c1, c2, tt, c2]


REGS = [('x', [(True, i) for i in range(8)])]   # q0..q7 registers (dirty-borrowable), q8.. scratch


class ChainFinder(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def circuit(self):
        ops = (and3_dirty(0, 1, 2, 6, 8) + [op(CZ, 8, 4)] + and3_dirty(0, 1, 2, 6, 8)      # dirty-borrow AND3 uncompute shape
               + barenco4(0, 1, 2, 3, 6, 7, 9) + [op(CZ, 9, 5), op(CX, 5, 6)]   # dirty Barenco 4-control pair; the dirty wire changes
               + barenco4(0, 1, 2, 3, 6, 7, 9)
               + [op(CCX, 0, 1, 10), op(CZ, 10, 4), op(CCX, 0, 1, 10)]                        # lone pair
               + [op(X, 0)] + and3_dirty(0, 1, 2, 6, 11) + [op(X, 0)])                        # negated control
        return build(self.dir / 'c.kmb', ops, REGS, num_qubits=12)

    def test_finds_exact_chains(self):
        k = K.read(str(self.circuit()))
        ch = C.find_chains(k)
        shapes = [(c['start'], c['toffolis'], len(c['controls']), c['target']) for c in ch]
        self.assertEqual(shapes[:4], [(0, 4, 3, 8), (5, 4, 3, 8), (9, 8, 4, 9), (19, 8, 4, 9)])
        self.assertEqual(ch[0]['borrowed'], [6])
        self.assertEqual(sorted(ch[2]['borrowed']), [6, 7])
        neg = ch[-1]
        self.assertEqual(neg['target'], 11)
        self.assertIn((0, True), neg['controls'])

    def test_rejects_non_chains(self):
        # a clean AND ladder relies on its ancilla being |0>: not valid for every value, so not a dirty chain
        ops = [op(CCX, 0, 1, 8), op(CCX, 8, 2, 9), op(CCX, 0, 1, 8)]
        k = K.read(str(build(self.dir / 'n.kmb', ops, REGS, num_qubits=10)))
        self.assertEqual(C.find_chains(k), [])

    def test_census_zero_after_and_prices(self):
        r = C.census(self.circuit())
        rows = {(x['op'], x['pattern']): x for x in r['rows']}
        self.assertFalse(rows[(0, 'chain3')]['zero_after'])           # compute
        u = rows[(5, 'mbu-chain3')]                                    # dirty AND3 uncompute: 4 -> 0.5
        self.assertTrue(u['zero_after'])
        self.assertEqual((u['now'], u['mbu'], u['saving']), (4, 0.5, 3.5))
        b = rows[(19, 'mbu-chain4')]                                   # Barenco n=4 uncompute: 8 -> mbu 0.5*(n-2)
        self.assertEqual(b['now'], 8)
        self.assertEqual(b['mbu'], 1.0 if b['free_clean'] >= 1 else 2.5)
        s = [x for x in r['rows'] if x['pattern'] == 'single']
        self.assertEqual([(x['op'], x['saving']) for x in s], [(29, 1)])
        self.assertLessEqual(r['peak'], 12)

    def test_track_macros_cancel_dirty_chains(self):
        p = self.circuit()
        k = K.read(str(p))
        ch = C.find_chains(k)
        macros = {c['start']: (c['end'], c['target'], c['controls']) for c in ch}
        plain = T.track(p)
        with_m = T.track(p, macros=macros)
        self.assertEqual([(s, t, z0, z1) for s, t, z0, z1 in with_m['macro_events']][:4],
                         [(0, 8, True, False), (5, 8, False, True), (9, 9, True, False), (19, 9, False, True)])
        # without macros the Barenco uncompute is not proved once a dirty wire changed in between (products are
        # compared by operand identity)
        self.assertNotIn(9, [w for _, w, *_ in plain['uncomputes']])
        self.assertIn(8, [w for _, w, *_ in plain['uncomputes']])

    def test_structural_proofs_agree_and_are_strict(self):
        p = self.circuit()
        tr = {(x['op'], x['zero_after']) for x in C.census(p, proof='track')['rows']}
        st = {(x['op'], x['zero_after']) for x in C.census(p, proof='structural')['rows']}
        self.assertEqual(tr, st)
        # a control written between compute and uncompute: the structural rule must not claim a proof
        ops = and3_dirty(0, 1, 2, 6, 8) + [op(CX, 3, 1)] + and3_dirty(0, 1, 2, 6, 8)
        q = build(self.dir / 'w.kmb', ops, REGS, num_qubits=9)
        rows = C.census(q, proof='structural')['rows']
        self.assertEqual([x['zero_after'] for x in rows], [False, False])
        rows = C.census(q, proof='track')['rows']
        self.assertEqual([x['zero_after'] for x in rows], [False, False])

    def test_classify_uncomputes_and3(self):
        # the track verb's classifier (as on 6d58 unextract, where its emitted script rebuilt the public
        # c253111974fc byte for byte): the dirty AND3 uncompute becomes HMR + CCZ-if-outcome
        ops = and3_dirty(0, 1, 2, 6, 8) + [op(CZ, 8, 4)] + and3_dirty(0, 1, 2, 6, 8)
        p = build(self.dir / 'a.kmb', ops, REGS, num_qubits=9)
        r = T.track(p)
        rows, nb = T.classify_uncomputes(p, r['uncomputes'])
        self.assertEqual([(x['mbu'], x['ops']) for x in rows], [('and3-dirty', '5..8')])
        self.assertEqual(rows[0]['edit'], 'replace 5:9 => HMR q8 b{NEW}; CCZ q0 q1 q2 if b{NEW}')

    def test_pricing_model(self):
        opts, best = C.price(6, 16, True, 3)
        self.assertEqual(opts['mbu'], (2.0, 3))
        self.assertEqual(best[0], 'mbu')
        opts, best = C.price(12, 40, False, 0)
        self.assertNotIn('mbu', opts)
        self.assertEqual(opts['cc'], (21, 0))
        self.assertEqual(opts['clean'], (11, 10))


class Doctor(unittest.TestCase):
    def test_engine_needs_parsing(self):
        text = ('from . import _engine\nfrom ._engine import Program, derive_keys, run as _run\n'
                'from ._engine import run_sequence as _run_sequence\n')
        self.assertEqual(D.engine_needs(text), ['Program', 'derive_keys', 'run', 'run_sequence'])

    def test_missing_export_fails_by_byte_scan(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / 'kickmix_engine'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('from ._engine import run, run_sequence\n')
            (pkg / '_engine.abi3.so').write_bytes(b'\x00run\x00other\x00')
            st, det, hint = D.check_engine(Path(td), python=None)
            self.assertEqual(st, 'FAIL')
            self.assertIn('run_sequence', det)
            self.assertIn('maturin', hint)
            (pkg / '_engine.abi3.so').write_bytes(b'\x00run\x00run_sequence\x00')
            os.utime(pkg / '__init__.py', (1, 1))
            st, _, _ = D.check_engine(Path(td), python=None)
            self.assertEqual(st, 'ok')


class Simulate(unittest.TestCase):
    def test_bit_parallel_values_and_conditions(self):
        with tempfile.TemporaryDirectory() as td:
            # registers S0 (2 qubits) + E0 (2 classical bits); scratch q2, q3
            regs = [('S0', [(True, 0), (True, 1)]), ('E0', [(False, 0), (False, 1)])]
            ops = [op(CCX, 0, 1, 2), op(0x81, bc=0), op(X, 3), op(0x01), op(HMR, 2, b0=2),
                   op(0xA0, 0, 1, bc=2), op(0x81, bc=0), op(X, 3), op(0x01)]
            p = build(Path(td) / 's.kmb', ops, regs, num_qubits=4, num_bits=3)
            lanes = 4
            bits = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], bool)       # lane l: q0 = l&1, q1 = l>>1
            mbuc = np.array([[0], [1], [2], [3]])                       # E0 bits = lane index
            inp = Path(td) / 'in.npz'
            np.savez(inp, bits=bits, qids=np.array([0, 1]), ebits=np.array([0, 1]), mbuc=mbuc, degree=2)
            r = SIM.simulate(p, inp, probes=[1, 4])
            self.assertEqual(r['state_at'][1][2], 0b1000)               # q2 = q0 q1: lane 3 only
            self.assertEqual(r['state_at'][4][3], 0b1010)               # X under E0 bit 0: lanes 1, 3
            self.assertEqual(r['end'][2], 0)                            # HMR zeroed
            self.assertEqual(r['end'][3], 0)
            self.assertEqual(r['end'][:2], r['inputs'][:2])
            cnt, rows = SIM.summarise(r['state_at'][1], 4, r['registers'], wires=[2])
            self.assertEqual(rows[0]['ones'], 0.25)


class CliSmoke(unittest.TestCase):
    @requires_kickmix
    def test_census_and_doctor_verbs(self):
        with tempfile.TemporaryDirectory() as td:
            p = ChainFinder.circuit(type('o', (), {'dir': Path(td)})())
            env = dict(os.environ, QHDL_WORK=str(Path(td) / 'work'))
            out = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'census', '-c', str(p), '--sites',
                                  'mbu'], capture_output=True, text=True, env=env, timeout=300)
            e = json.loads(out.stdout)
            self.assertIn('MBU candidates', e['claim'])
            self.assertEqual(e['rows_total'], 3)            # rows may be cut by the 2 KB stdout budget
            ev = json.loads(Path(e['evidence']).read_text())
            self.assertEqual(sorted(r['op'] for r in ev['rows'] if r['zero_after']), [5, 19, 29])
            out = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'doctor'], capture_output=True,
                                 text=True, env=env, timeout=300)
            e = json.loads(out.stdout)
            self.assertEqual(e['tier'], 'static')
            self.assertEqual([r['check'] for r in e['rows']][:2], ['engine', 'kickmix'])


if __name__ == '__main__':
    unittest.main()
