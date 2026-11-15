"""qhdl avenues --task dqi: streaming decode, Toffoli clustering, repetition up to wire relabelling, phase/value
floors, pricing. Synthetic .kmb circuits only (qhdl_kmb.write_kmb). NumPy required (skipped in the stdlib-only project env)."""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

CCX, CCZ, CX, CZ, HMR, X, PUSH, POP = 0x31, 0x30, 0x21, 0x20, 0x50, 0x11, 0x81, 0x01
CZ_IF = 0xA0


def build(path, ops):
    """ops: list of (type, wires, bbit, cbit)."""
    import numpy as np
    import qhdl_kmb as K
    t, s = [], {n: [] for n in ('qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc')}
    for tv, w, b, c in ops:
        t.append(tv)
        if tv & 0x30 == 0x30:
            for i, n in enumerate(('qqq0', 'qqq1', 'qqq2')):
                s[n].append(w[i])
        elif tv & 0x30 == 0x20:
            s['qq0'].append(w[0])
            s['qq1'].append(w[1])
        elif tv & 0x30 == 0x10:
            s['q0'].append(w[0])
        if tv & 0x40:
            s['b0'].append(b)
        if tv & 0x80:
            s['bc'].append(c)
    K.write_kmb(str(path), np.array(t, dtype=np.uint8), {n: np.array(v, dtype=np.uint32) for n, v in s.items()})


def site(a, b, x, y, bit0, pad=0):
    """4 data wires a, b, x, y (x, y fresh products): two CCX into fresh wires, a CZ between the products, measure both out."""
    return [(CX, (a, b), -1, -1), (CCX, (a, b, x), -1, -1), (CCX, (b, a, y), -1, -1), (CZ, (x, y), -1, -1),
            (CZ, (x, a), -1, -1), (HMR, (x,), bit0, -1), (HMR, (y,), bit0 + 1, -1), (CZ_IF, (a, b), -1, bit0),
            (CZ_IF, (b, a), -1, bit0 + 1)]


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class AvenuesDqi(unittest.TestCase):
    def setUp(self):
        import qhdl_avenues_dqi as D
        self.D = D

    def make(self, ops):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "c.kmb"
        build(p, ops)
        import qhdl_kmb as K
        return D_fast(self.D, K.read(str(p)))

    def test_decode_matches_reference_decoder(self):
        import qhdl_kmb as K
        ops = [(X, (3,), -1, -1)] + site(0, 1, 5, 6, 10) + [(PUSH, (), -1, 4)] + site(2, 3, 7, 8, 12) + [(POP, (), -1, -1)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.kmb"
            build(p, ops)
            k = K.read(str(p))
            f = self.D.Fast(k)
            dec = K.Decoder(k)
            got = f.decode(0, len(ops))
            for i, o in enumerate(got):
                self.assertEqual(o[0], int(k.op_types[i]))
                self.assertEqual(list(o[1]), dec.qubits(i))
            self.assertEqual(f.depth_before(len(ops) - 1), 1)

    def test_clusters_and_repetition_up_to_relabelling(self):
        glue = [(CX, (0, 1), -1, -1)] * 100         # > CLUSTER_GAP ops between sites
        ops = site(0, 1, 5, 6, 10) + glue + site(2, 3, 7, 8, 12) + glue + site(4, 0, 9, 11, 14)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.kmb"
            build(p, ops)
            res = self.D.run(p, 'testtest0000', None, None, ledger=[])
        self.assertEqual(res['stats']['clusters'], 3)
        # the three sites are one class (wires and bit ids renamed): either all fresh-identical or split only by the
        # entry-freshness signature; their family is 3 whatever the split
        fams = {}
        for r in res['open'] + res['closed'] + [x for x in res['unfloored']]:
            fams.setdefault(r['component'].split(' sig ')[1].split('/')[0], r['sig_family_multiplicity'])
        self.assertIn(3, fams.values())

    def test_phase_degree_floor_and_pricing(self):
        D = self.D
        self.assertEqual([D.phase_degree_floor(d) for d in (0, 2, 3, 4, 6, 7, 12, 13)], [0, 0, 1, 2, 2, 3, 3, 4])
        # GM = (1/3) ln(T/(T-S)) in percent
        self.assertAlmostEqual(D.gm_pct(3, 100.0), 100 * math.log(100 / 97) / 3)
        self.assertEqual(D.gm_pct(0, 100.0), 0.0)
        self.assertEqual(D.gm_pct(5, None), 0.0)
        t, q = D.t_mean_from_manifest({'qubits': '10', 'dqi_zero_failure_cost': str(1.1913 * 100 * 7)})
        self.assertAlmostEqual(t, 7.0)

    def test_window_function_phase_degree_and_floor(self):
        D = self.D
        ops = site(0, 1, 5, 6, 10)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.kmb"
            build(p, ops)
            import qhdl_kmb as K
            f = D.Fast(K.read(str(p)))
            r = D.window_function(f, 1, len(ops))
        self.assertEqual(r['status'], 'exact_anf')
        self.assertEqual(r['toffoli'], 2)
        # x = a*b (deg 2) and y = a*b are the same function: CZ(x, y) = a*b, CZ(x, a) = a*b -> phase degree 2 or 3 at most
        self.assertLessEqual(r['phase_degree'], 4)
        self.assertGreaterEqual(r['floor'], 0)

    def test_unsupported_condition_on_outside_bit(self):
        D = self.D
        ops = [(CZ_IF, (0, 1), -1, 9), (CCX, (0, 1, 2), -1, -1)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.kmb"
            build(p, ops)
            import qhdl_kmb as K
            f = D.Fast(K.read(str(p)))
            r = D.window_function(f, 0, 2)
        self.assertEqual(r['status'], 'unsupported')

    def test_cubic_phase_needs_one_and(self):
        D = self.D
        ops = [(CCZ, (0, 1, 2), -1, -1)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.kmb"
            build(p, ops)
            import qhdl_kmb as K
            r = D.window_function(D.Fast(K.read(str(p))), 0, 1)
        self.assertEqual((r['phase_degree'], r['floor']), (3, 1))


def D_fast(D, k):
    return D.Fast(k)


if __name__ == "__main__":
    unittest.main()
