"""qhdl measure-out: synthetic circuits only. Exactness is checked by a small statevector simulator:
every HMR outcome branch of the rewritten circuit must give exactly the original circuit's final state. NumPy needed."""
import importlib.util
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
CCX, CX, HMR, XG, CZ = 0x31, 0x21, 0x50, 0x11, 0x20
NQ = 9


def write(path, ops, nbits=1):
    """ops: (code, qubits, bit_or_None, cond_bit_or_None)."""
    import numpy as np
    import qhdl_kmb as KM
    ot = np.array([o | (0x80 if c is not None else 0) for o, _, _, c in ops], dtype=np.uint8)
    s = {n: [] for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc']}
    for o, q, b, c in ops:
        cl = o & 0x30
        for names in ({0x30: ('qqq0', 'qqq1', 'qqq2'), 0x20: ('qq0', 'qq1'), 0x10: ('q0',)}.get(cl, ())),:
            for n, v in zip(names, q):
                s[n].append(v)
        if o & 0x40:
            s['b0'].append(b)
        if c is not None:
            s['bc'].append(c)
    KM.write_kmb(str(path), ot, s, num_qubits=NQ, num_bits=nbits)
    return str(path)


def read_ops(path):
    import qhdl_kmb as KM
    k = KM.read(str(path))
    d = KM.Decoder(k)
    out = []
    for i in range(len(k.op_types)):
        t = int(k.op_types[i])
        b = None
        if t & 0x40:
            b = int(d.s['b0'][d.cb[i]])
        c = int(d.s['bc'][d.cc[i]]) if t & 0x80 else None
        out.append((t & 0x7F, tuple(d.qubits(i)), b, c))
    return out, int(k.num_qubits)


def simulate(ops, nq, psi, outcomes):
    """Statevector over nq qubits (index bit w = qubit w). HMR bit b takes outcomes[b]; returns the state (normalised)."""
    import numpy as np
    psi = psi.copy()
    idx = np.arange(1 << nq)
    bits = {}
    for code, q, b, c in ops:
        if c is not None and not bits.get(c, 0):
            continue
        if code == CCX:
            m = ((idx >> q[0]) & 1) & ((idx >> q[1]) & 1)
            psi = psi[idx ^ (m << q[2])]
        elif code == CX:
            psi = psi[idx ^ (((idx >> q[0]) & 1) << q[1])]
        elif code == XG:
            psi = psi[idx ^ (1 << q[0])]
        elif code == CZ:
            psi = psi * (1 - 2 * (((idx >> q[0]) & 1) & ((idx >> q[1]) & 1)))
        elif code == HMR:
            w = q[0]
            m = outcomes[b]
            bits[b] = m
            sign = 1 - 2 * (((idx >> w) & 1) & m)            # <m|x> = (-1)^{m x} / sqrt2, then reset to |0>
            new = np.zeros_like(psi)
            lo = idx[((idx >> w) & 1) == 0]
            new[lo] = (psi[lo] * sign[lo] + psi[lo | (1 << w)] * sign[lo | (1 << w)])
            psi = new
            nrm = np.linalg.norm(psi)
            psi = psi / nrm
        else:
            raise AssertionError(code)
    return psi


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class MeasureOut(unittest.TestCase):
    def setUp(self):
        import qhdl_kmb as KM
        import qhdl_measure_out as MO
        import qhdl_retain as RT
        import qhdl_edit as ED
        self.KM, self.MO, self.RT, self.ED = KM, MO, RT, ED
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def rewrite(self, ops):
        import numpy as np
        src = write(self.dir / 'c.kmb', [(o, q, None, None) if o != HMR else (o, q, 0, None) for o, q in ops])
        k = self.KM.read(src)
        r = self.RT.analyse(k, 0, len(ops))
        chosen, sk = self.MO.select(k, r['sites'])
        lines, nb = self.MO.edit_lines(chosen, int(k.num_bits))
        if not chosen:
            return None, chosen, sk, 0
        out = self.dir / 'o.kmb'
        self.ED.edit_file(src, '\n'.join(lines) + '\n', out)
        return str(out), chosen, sk, nb

    def check_equivalent(self, ops, pre_zero):
        """Random state on the qubits not in pre_zero; zero wires |0>. Every outcome branch equals the original."""
        import numpy as np
        out, chosen, sk, nb = self.rewrite(ops)
        self.assertTrue(chosen)
        new_ops, nq2 = read_ops(out)
        self.assertEqual(nq2, NQ)                                   # no new qubits
        rng = np.random.default_rng(5)
        psi = rng.normal(size=1 << NQ) + 1j * rng.normal(size=1 << NQ)
        idx = np.arange(1 << NQ)
        for w in pre_zero:
            psi = np.where(((idx >> w) & 1) == 0, psi, 0)
        psi /= np.linalg.norm(psi)
        orig_ops = [(o, q, 0 if o == HMR else None, None) for o, q in ops]
        want = simulate(orig_ops, NQ, psi, {0: 0})
        # the original HMR on the prefix wire (if any) uses bit 0; rewritten ops use new bits from num_bits
        nbits = 1
        for outs in itertools.product((0, 1), repeat=nb):
            om = {0: 0}
            om.update({nbits + i: v for i, v in enumerate(outs)})
            got = simulate(new_ops, NQ, psi, om)
            self.assertTrue(np.allclose(got, want, atol=1e-9), f'outcomes {outs}')
        return chosen

    def test_mixed_target_with_spare_is_exact(self):
        ops = [(CX, (2, 6)), (CCX, (0, 1, 6)), (CX, (3, 4)), (CX, (5, 3)), (CCX, (0, 1, 6)), (XG, (4,))]
        # wire 7 starts |0> and is reset by an HMR so that the scan knows it is zero
        ops = [(HMR, (7,))] + ops
        ch = self.check_equivalent(ops, pre_zero=[7, 8])
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0]['_spare'], 7)

    def test_pure_target_needs_no_spare(self):
        ops = [(HMR, (6,)), (CCX, (0, 1, 6)), (CX, (2, 3)), (CCX, (0, 1, 6))]
        ch = self.check_equivalent(ops, pre_zero=[6])
        self.assertIsNone(ch[0]['_spare'])

    def test_off_with_affine_equal_controls_on_other_wires(self):
        # wire 5 reset then copied from wire 0: the cancelling CCX uses it, the CZ fix-up must too
        ops = [(HMR, (5,)), (HMR, (7,)), (CX, (0, 5)), (CX, (2, 6)), (CCX, (0, 1, 6)), (CX, (3, 4)), (CCX, (5, 1, 6))]
        ch = self.check_equivalent(ops, pre_zero=[5, 7])
        self.assertEqual(ch[0]['_off'][:2], (5, 1))

    def test_toffoli_down_hmr_up_no_qubits(self):
        import numpy as np
        ops = [(HMR, (7,)), (CX, (2, 6)), (CCX, (0, 1, 6)), (CX, (3, 4)), (CCX, (0, 1, 6))]
        out, chosen, _, _ = self.rewrite(ops)
        b, a = self.MO.counts_of(self.KM.read(str(self.dir / 'c.kmb'))), self.MO.counts_of(self.KM.read(out))
        self.assertEqual((a['toffoli_plain'], b['toffoli_plain']), (1, 2))
        self.assertEqual(a['hmr'] - b['hmr'], 1)
        self.assertEqual(a['num_qubits'], b['num_qubits'])
        self.assertEqual(a['num_bits'], b['num_bits'] + 1)

    def test_no_spare_means_no_site_applied(self):
        out, chosen, sk, _ = self.rewrite([(CX, (2, 6)), (CCX, (0, 1, 6)), (CX, (3, 4)), (CCX, (0, 1, 6))])
        self.assertEqual(chosen, [])
        self.assertEqual(sk['no_free_spare'], 1)

    def test_overlapping_sites_do_not_share_a_spare(self):
        ops = [(HMR, (7,)), (CX, (2, 5)), (CX, (2, 6)), (CCX, (0, 1, 5)), (CCX, (3, 4, 6)),
               (CX, (2, 8)), (CCX, (0, 1, 5)), (CCX, (3, 4, 6))]
        out, chosen, sk, _ = self.rewrite(ops)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(sk['spare_booked'], 1)

    def test_retain_rows_are_not_touched(self):
        out, chosen, sk, _ = self.rewrite([(CCX, (0, 1, 6)), (HMR, (6,)), (CX, (2, 3)), (CCX, (0, 1, 8))])
        self.assertEqual(chosen, [])

    def test_conditional_ccx_is_skipped(self):
        # a conditioned off CCX is invisible to the scan (fresh atom), so no site and no edit
        src = write(self.dir / 'c.kmb', [(HMR, (6,), 0, None), (CCX, (0, 1, 6), None, None), (CCX, (0, 1, 6), None, 0)])
        k = self.KM.read(src)
        r = self.RT.analyse(k, 0, 3)
        chosen, _ = self.MO.select(k, r['sites'])
        self.assertEqual(chosen, [])


if __name__ == '__main__':
    unittest.main()
