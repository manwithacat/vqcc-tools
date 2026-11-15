"""QHDL v1 rewrite registry (stream B, 2026-11-01): the semantic block finder, the clean-wire model, tag-relative
addresses, every class's match / price / apply on small synthetic circuits, and the census --class / rewrite verbs.
NumPy required (system python3); skipped in the stdlib-only environment."""
from __future__ import annotations

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _optional import requires_kickmix  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
HAS_NUMPY = importlib.util.find_spec('numpy') is not None

if HAS_NUMPY:
    import numpy as np
    import qhdl_kmb as K
    import qhdl_rewrite as R
    from test_qhdl_census import CCX, CX, CZ, HMR, REGS, X, and3_dirty, build, op

CZ_IF, CCZ_IF, TAG, SWAP = 0xA0, 0xB0, 0x06, 0x22


def barenco6(ys, ds, t):
    y1, y2, y3, y4, y5, y6 = ys
    d1, d2, d3, d4 = ds
    asc = [op(CCX, y1, y2, d1), op(CCX, d1, y3, d2), op(CCX, d2, y4, d3), op(CCX, d3, y5, d4), op(CCX, d4, y6, t)]
    return asc + asc[:-1][::-1] + asc[1:] + asc[1:-1][::-1]


def synthetic_ops():
    """One site per class: dirty AND3 pair (op 5), a clean reset (q10), dirty Barenco pair (op 27), a lone pair (op 45), a dirty
    AND3 compute + busyaprime-style partial MBU (op 51)."""
    ops = and3_dirty(0, 1, 2, 6, 8) + [op(CZ, 8, 4)] + and3_dirty(0, 1, 2, 6, 8)
    ops += [op(HMR, 10, b0=0)]
    ops += barenco6((0, 1, 2, 3, 4, 5), (6, 7, 12, 13), 9) + [op(CZ, 9, 3)] + barenco6((0, 1, 2, 3, 4, 5), (6, 7, 12, 13), 9)
    ops += [op(CCX, 0, 1, 11), op(CZ, 11, 4), op(CCX, 0, 1, 11)]
    ops += and3_dirty(0, 1, 2, 6, 14) + [op(CZ, 14, 5)]
    ops += [op(CCX, 0, 1, 6), op(CCX, 6, 2, 14), op(CCX, 0, 1, 6), op(HMR, 14, b0=1), op(CZ_IF, 6, 2, bc=1)]
    return ops


def build_tagged(path, ops, registers, tags, num_qubits=None, num_bits=None):
    """Like test_qhdl_census.build but ops may be ('TAG', text)."""
    st = {n: [] for n in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')}
    codes, texts = [], []
    for o in ops:
        if o[0] == 'TAG':
            codes.append(TAG)
            texts.append(o[1].encode())
            continue
        code, q, b0, bc = o
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
                tags=texts, registers=registers, num_qubits=num_qubits, num_bits=num_bits)
    return path


@unittest.skipUnless(HAS_NUMPY, 'NumPy not installed (stdlib-only env); run with system python3')
class Finder(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def blocks(self, ops, nq=16, nb=2):
        p = build(self.dir / 'c.kmb', ops, REGS, num_qubits=nq, num_bits=nb)
        k = K.read(str(p))
        return R.pair_blocks(R.find_blocks(R.Ops(k)))

    def test_synthetic_blocks_and_pairs(self):
        bl = self.blocks(synthetic_ops())
        shape = [(b['start'], b['end'], b['kind'], b['n'], b['toffolis'], b['target']) for b in bl]
        self.assertEqual(shape, [(0, 4, 'chain', 3, 4, 8), (5, 9, 'chain', 3, 4, 8), (10, 26, 'chain', 6, 16, 9),
                                 (27, 43, 'chain', 6, 16, 9), (46, 50, 'chain', 3, 4, 14),
                                 (51, 56, 'mbu-partial', 3, 3, 14)])
        self.assertEqual(bl[1]['partner']['start'], 0)
        self.assertEqual(bl[3]['partner']['start'], 10)
        self.assertEqual(bl[5]['partner']['start'], 46)       # the partial MBU pairs with the dirty AND3 compute
        self.assertEqual([w for w, _ in bl[2]['controls']], [0, 1, 2, 3, 4, 5])   # first-appearance order
        self.assertEqual(bl[2]['borrowed'], [6, 7, 12, 13])

    def test_cached_chain_with_in_block_measurement(self):
        """busyaprime's 14-Toffoli shape (3779a3431e9c epoch 0 u6, wires renamed): a product cached on the clean wire
        B inside the Barenco chain, routed by CX, measured out with a CZ fix-up. The block is a 6-input chain only
        under the assumption B = |0> at entry, which the finder records as `assumed_zero`."""
        y1, y2, y3, y4, y5, y6 = 0, 1, 2, 3, 4, 5
        d1, d2, d3, d4, B, A = 6, 7, 12, 13, 9, 8
        p1 = [op(CCX, y1, y2, d1), op(CCX, d1, y3, d2), op(CCX, d2, y4, d3), op(CCX, d3, y5, B), op(CX, B, d4),
              op(CCX, d4, y6, A), op(CX, B, d4), op(HMR, B, b0=0), op(CZ_IF, d3, y5, bc=0), op(CCX, d2, y4, d3),
              op(CCX, d1, y3, d2), op(CCX, y1, y2, d1)]
        p2 = [op(CCX, d1, y3, d2), op(CCX, d2, y4, d3), op(CCX, d3, y5, B), op(CX, B, d4), op(CCX, d4, y6, A),
              op(CX, B, d4), op(HMR, B, b0=0), op(CZ_IF, d3, y5, bc=0), op(CCX, d2, y4, d3), op(CCX, d1, y3, d2)]
        bl = self.blocks(p1 + p2, nq=14, nb=1)
        self.assertEqual(len(bl), 1, [(b['start'], b['end'], b['kind'], b['n']) for b in bl])
        b = bl[0]
        self.assertEqual((b['start'], b['end'], b['kind'], b['n'], b['toffolis'], b['target']), (0, 22, 'chain', 6, 14, A))
        self.assertEqual((b['assumed_zero'], b['hmr_wires']), ([B], [B]))
        self.assertEqual([w for w, _ in b['controls']], [y1, y2, y3, y4, y5, y6])
        self.assertEqual(b['borrowed'], sorted([d1, d2, d3, d4]))
        # the wrong fix-up (CZ on the wrong pair) breaks the phase obligation: no block at all
        bad = [o if not (o[0] == CZ_IF) else op(CZ_IF, d2, y5, bc=0) for o in p1 + p2]
        self.assertEqual(self.blocks(bad, nq=14, nb=1), [])

    def test_construction_prefix_is_a_cached_chain(self):
        """The S6 construction starts with CCX(y1,y2,F) CCX(F,y3,B) HMR F CZ(y1,y2) if m: the finder, which
        reports the smallest exact block from each start, sees that prefix as a 3-input chain into B with F assumed
        |0> (a clean ancilla released by measurement-based uncomputation)."""
        import qhdl_rewrite_cc_chain as CC
        n, anc, tgt, sops = CC.S6()
        names = {f'y{i + 1}': i for i in range(n)}
        names.update({tgt: n, anc[0]: n + 1})
        bits, ops = {}, []
        for o in sops:
            kd = o[0]
            if kd == 'CCX':
                ops.append(op(CCX, *(names[x] for x in o[1:])))
            elif kd == 'X':
                ops.append(op(X, names[o[1]]))
            elif kd == 'HMR':
                bits.setdefault(o[2], len(bits))
                ops.append(op(HMR, names[o[1]], b0=bits[o[2]]))
            elif kd == 'CZ_IF':
                ops.append(op(CZ_IF, *(names[x] for x in o[2:]), bc=bits[o[1]]))
            elif kd == 'CCZ_IF':
                ops.append(op(CCZ_IF, *(names[x] for x in o[2:]), bc=bits[o[1]]))
        regs = [('x', [(True, i) for i in range(n)])]
        p = build(self.dir / 's6.kmb', ops, regs, num_qubits=n + 2, num_bits=len(bits))
        bl = R.find_blocks(R.Ops(K.read(str(p))))
        self.assertTrue(bl)
        b = bl[0]
        self.assertEqual((b['start'], b['end'], b['kind'], b['n'], b['target'], b['assumed_zero']),
                         (0, 4, 'chain', 3, n + 1, [n]))

    def test_rejects_non_chains(self):
        ops = [op(CCX, 0, 1, 8), op(CCX, 8, 2, 9), op(CCX, 0, 1, 8)]   # clean AND ladder without a clean proof
        self.assertEqual(self.blocks(ops, nq=10), [])
        ops = [op(CCX, 0, 1, 8), op(CZ, 8, 4), op(CCX, 0, 1, 8)]        # lone pair: not a chain (census 'single')
        self.assertEqual(self.blocks(ops, nq=10), [])
        # a partial MBU whose fix-up is wrong is nothing
        ops = [op(CCX, 0, 1, 6), op(CCX, 6, 2, 14), op(CCX, 0, 1, 6), op(HMR, 14, b0=1), op(CZ_IF, 6, 1, bc=1)]
        self.assertEqual(self.blocks(ops), [])


@unittest.skipUnless(HAS_NUMPY, 'NumPy not installed (stdlib-only env); run with system python3')
class CleanAndTags(unittest.TestCase):
    def test_clean_model(self):
        with tempfile.TemporaryDirectory() as td:
            ops = [op(HMR, 10, b0=0), op(X, 11), op(CX, 0, 12), op(CCX, 0, 1, 13), op(0x12, 11), op(CZ, 10, 4)]
            p = build(Path(td) / 'c.kmb', ops, REGS, num_qubits=14, num_bits=1)
            k = K.read(str(p))
            cm = R.CleanModel(k, R.Ops(k))
            self.assertEqual(cm.status(10, 5), 'proved')      # erased by an unconditional HMR
            self.assertEqual(cm.status(11, 4), '')            # X'd
            self.assertEqual(cm.status(11, 5), 'proved')      # R
            self.assertEqual(cm.status(12, 5), '')            # holds a copy
            self.assertEqual(cm.status(9, 5), 'proved')       # never touched
            self.assertEqual(cm.status(0, 5), '')             # register
            cand = cm.clean_scratch(5, exclude=(9,), span_end=5)
            self.assertEqual([w for w, _, _ in cand][:3], [11, 10, 8])  # most recently erased first, never-touched last
            self.assertNotIn(10, [w for w, _, _ in cm.clean_scratch(5, span_end=6)])   # q10 is read inside [5, 6)

    def test_conditional_reset_inside_one_push_block(self):
        """A wire reset at depth 0 and then written and reset again inside one `if b` block is |0> after the block in
        every shot (the measurement-based ladders end this way); a conditional write without a reset is not."""
        with tempfile.TemporaryDirectory() as td:
            ops = [op(HMR, 9, b0=0), op(0x81, bc=0), op(CCX, 0, 1, 9), op(HMR, 9, b0=1), op(0x01), op(X, 8),
                   op(0x81, bc=0), op(CCX, 0, 1, 10), op(HMR, 10, b0=1), op(0x01), op(X, 8),
                   op(0x81, bc=0), op(CCX, 0, 1, 9), op(0x01)]
            p = build(Path(td) / 'c.kmb', ops, REGS, num_qubits=11, num_bits=2)
            k = K.read(str(p))
            cm = R.CleanModel(k, R.Ops(k))
            self.assertEqual(cm.status(9, 5), 'proved')     # after the block
            self.assertEqual(cm.status(9, 4), '')           # inside the block (not yet reset)
            self.assertEqual(cm.status(10, 10), 'proved')   # never written before the block
            self.assertEqual(cm.status(9, 14), '')          # conditional write, no reset

    def test_tag_relative_addresses(self):
        with tempfile.TemporaryDirectory() as td:
            ops = [('TAG', '@enter:a'), ('TAG', '@enter:b'), op(X, 8), ('TAG', '@exit:b'), ('TAG', '@enter:b'),
                   op(X, 8), op(X, 9), ('TAG', '@exit:b'), ('TAG', '@exit:a'), op(X, 8)]
            p = build_tagged(Path(td) / 't.kmb', ops, REGS, [], num_qubits=10)
            ti = R.TagIndex(K.read(str(p)))
            self.assertTrue(ti.tagged)
            self.assertEqual(ti.rel(2), 'tag:a/b[0]+1')
            self.assertEqual(ti.rel(6), 'tag:a/b[1]+2')
            self.assertEqual(ti.rel(8), 'tag:a[0]+8')
            self.assertIsNone(ti.rel(9))
            self.assertEqual(ti.resolve('tag:a/b[1]+2'), 6)
            with self.assertRaises(R.RewriteError):
                ti.resolve('tag:a/b[2]+0')
            with self.assertRaises(R.RewriteError):
                ti.resolve('tag:a/b[1]+9')

    def test_parse_replacement(self):
        d = R.parse_replacement('# c\nsite and-merge#0\nCX q1 q2\n@5 # copy\nops 7:9\nCCX q0 q1 q2\n')
        self.assertEqual(d, {'and-merge#0': ['CX q1 q2', '@5'], '7:9': ['CCX q0 q1 q2']})
        with self.assertRaises(R.RewriteError):
            R.parse_replacement('CX q1 q2\n')


@unittest.skipUnless(HAS_NUMPY, 'NumPy not installed (stdlib-only env); run with system python3')
class Templates(unittest.TestCase):
    def test_exhaustive_template_checks(self):
        import qhdl_rewrite_cc_chain as CC
        import qhdl_rewrite_flag_ladder_mbu as FL
        self.assertTrue(FL.check_ladder())
        for n, f in CC.SCHEMES.items():
            nn, anc, tgt, ops = f()
            ok, st, ex, msgs = CC.check(nn, anc, tgt, ops)
            self.assertTrue(ok, msgs)
            self.assertEqual((st, ex), CC.PRICE[n])
        # a broken scheme is caught
        nn, anc, tgt, ops = CC.S6()
        ok, *_ = CC.check(nn, anc, tgt, ops[:-1])   # drop the last fix-up
        self.assertFalse(ok)


@unittest.skipUnless(HAS_NUMPY, 'NumPy not installed (stdlib-only env); run with system python3')
class Cli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.td.name)
        cls.circuit = build(cls.dir / 's.kmb', synthetic_ops(), REGS, num_qubits=15, num_bits=2)
        cls.env = dict(os.environ, QHDL_WORK=str(cls.dir / 'work'))

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def qhdl(self, *args):
        out = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), *args], capture_output=True, text=True,
                             env=self.env, timeout=600)
        self.assertTrue(out.stdout.strip(), out.stderr)
        return json.loads(out.stdout)

    @requires_kickmix
    def test_census_class_all(self):
        e = self.qhdl('census', '-c', str(self.circuit), '--class', 'all', '--unfit', '--all')
        self.assertEqual(e['tier'], 'static')
        rows = {r['id']: r for r in e['rows']}
        self.assertEqual(rows['and3-dirty-mbu#0']['ops'], '5..8')
        self.assertEqual((rows['and3-dirty-mbu#0']['now'], rows['and3-dirty-mbu#0']['after'],
                          rows['and3-dirty-mbu#0']['saving'], rows['and3-dirty-mbu#0']['status']), (4, 0.5, 3.5, 'ok'))
        self.assertEqual((rows['flag-ladder-mbu#0']['ops'], rows['flag-ladder-mbu#0']['after'],
                          rows['flag-ladder-mbu#0']['status']), ('27..42', 2.25, 'ok'))
        self.assertEqual((rows['cc-chain#0']['ops'], rows['cc-chain#0']['after'], rows['cc-chain#0']['status']),
                         ('10..25', 7.5, 'ok'))
        self.assertEqual(rows['cc-chain#1']['status'], 'unfit')    # the uncompute chain is not a compute site
        self.assertEqual((rows['single-toffoli-mbu#0']['ops'], rows['single-toffoli-mbu#0']['saving']), ('45..45', 1.0))
        self.assertEqual((rows['u3-mbu#0']['ops'], rows['u3-mbu#0']['after'], rows['u3-mbu#0']['status']),
                         ('51..55', 0.5, 'ok'))
        ev = json.loads(Path(e['evidence']).read_text())
        s = next(x for x in ev['sites'] if x['id'] == 'flag-ladder-mbu#0')
        self.assertEqual(s['wires']['helper'], 'q10')              # the wire the HMR just erased
        self.assertEqual(s['anchor'], f"op@{e['circuit_sha'][:12]}:27")
        self.assertIsNone(s['tag_rel'])                             # untagged circuit: absolute addresses only
        self.assertEqual([o['status'] for o in s['obligations']], ['proved', 'proved'])

    @requires_kickmix
    def test_rewrite_each_class_and_scripts(self):
        want = {
            'and3-dirty-mbu': ['replace 5:9 => HMR q8 b2; CCZ q0 q1 q2 if b2'],
            'single-toffoli-mbu': ['replace 45 => HMR q11 b2; CZ q0 q1 if b2'],
            'u3-mbu': ['replace 51:56 => HMR q14 b2; CCZ q0 q1 q2 if b2'],
            'flag-ladder-mbu': ['replace 27:43 => HMR q9 b2; PUSH_CONDITION if b2; CCX q0 q1 q9; CCX q9 q2 q10; '
                                'HMR q9 b3; CZ q0 q1 if b3; CCX q10 q3 q9; CCZ q9 q4 q5; HMR q9 b4; CZ q10 q3 if b4; '
                                'HMR q10 b5; CCZ q0 q1 q2 if b5; POP_CONDITION'],
        }
        for cls, lines in want.items():
            out = self.dir / f'cand-{cls}.kmb'
            e = self.qhdl('rewrite', '-c', str(self.circuit), '--class', cls, '--out', str(out), '--certify', 'none')
            self.assertIn(cls, e['claim'])
            self.assertEqual(e['tier'], 'static')
            script = [l for l in (out.with_name(out.name + '.edit.txt')).read_text().splitlines() if not l.startswith('#')]
            self.assertEqual(script, lines)
            self.assertTrue(out.exists() and out.with_name(out.name + '.map.json').exists())
            cert = json.loads(out.with_name(out.name + '.certificate.json').read_text())
            self.assertEqual(cert['scope'], dict(domain='none', extent='none', branches='none'))
            self.assertEqual(cert['unproved_sites'], [])
            k = K.read(str(out))
            self.assertEqual(int(k.num_bits), 2 + len(R.load_classes([cls])[cls].bits))
        # cc-chain: only the proved compute is applied; the open uncompute needs --unproved
        out = self.dir / 'cand-cc.kmb'
        e = self.qhdl('rewrite', '-c', str(self.circuit), '--class', 'cc-chain', '--out', str(out), '--certify', 'none')
        self.assertIn('1 site(s) of 1 fitting', e['claim'])
        script = [l for l in (out.with_name(out.name + '.edit.txt')).read_text().splitlines() if not l.startswith('#')]
        self.assertTrue(script[0].startswith('replace 10:26 => CCX q0 q1 q9; CCX q9 q2 q10; HMR q9 b3; CZ q0 q1 if b3;'))
        e = self.qhdl('rewrite', '-c', str(self.circuit), '--class', 'cc-chain', '--sites', '#1', '--out',
                      str(self.dir / 'x.kmb'), '--certify', 'none')
        self.assertIn('do not fit the template', e['error'])    # an unfit site is never forced
        # an open obligation: the AND3 pair with a control rewritten between (no zero proof) needs --unproved
        ops = and3_dirty(0, 1, 2, 6, 8) + [op(CX, 3, 1)] + and3_dirty(0, 1, 2, 6, 8)
        w = build(self.dir / 'w.kmb', ops, REGS, num_qubits=9)
        e = self.qhdl('census', '-c', str(w), '--class', 'and3-dirty-mbu', '--all')
        self.assertEqual([r['status'] for r in e['rows']], ['open'])
        e = self.qhdl('rewrite', '-c', str(w), '--class', 'and3-dirty-mbu', '--out', str(self.dir / 'x.kmb'),
                      '--certify', 'none')
        self.assertIn('open obligations', e['error'])
        e = self.qhdl('rewrite', '-c', str(w), '--class', 'and3-dirty-mbu', '--out', str(self.dir / 'x.kmb'),
                      '--certify', 'none', '--unproved')
        self.assertIn('UNPROVED', e['claim'])
        cert = json.loads((self.dir / 'x.kmb.certificate.json').read_text())
        self.assertEqual(cert['unproved_sites'], ['and3-dirty-mbu#0'])
        # two classes in one rewrite are refused; a wrong site address is refused
        e = self.qhdl('rewrite', '-c', str(self.circuit), '--class', 'cc-chain,u3-mbu', '--out', str(self.dir / 'y.kmb'))
        self.assertIn('ONE class', e['error'])
        e = self.qhdl('rewrite', '-c', str(self.circuit), '--class', 'u3-mbu', '--sites', '5', '--out',
                      str(self.dir / 'y.kmb'))
        self.assertIn('no site', e['error'])

    @requires_kickmix
    def test_rewritten_circuit_is_recognised_as_done(self):
        """After and3-dirty-mbu the site is gone and the census of the candidate reports no and3 site."""
        out = self.dir / 'done.kmb'
        self.qhdl('rewrite', '-c', str(self.circuit), '--class', 'and3-dirty-mbu', '--out', str(out), '--certify', 'none')
        e = self.qhdl('census', '-c', str(out), '--class', 'and3-dirty-mbu', '--all')
        self.assertEqual(e.get('rows_total', 0), 0)


if __name__ == '__main__':
    unittest.main()
