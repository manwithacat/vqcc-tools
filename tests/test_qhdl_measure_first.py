"""qhdl measure-first: write-only wire runs ending in an HMR, on small synthetic circuits."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import qhdl_measure_first as M  # noqa: E402
from test_qhdl_census import CCX, CX, CZ, HMR, REGS, X, build, op  # noqa: E402

PUSH, POP = 0x81, 0x01


class MeasureFirst(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, ops, **kw):
        return M.scan(build(self.dir / 'c.kmb', ops, REGS, num_qubits=12, num_bits=4), **kw)

    def test_write_only_run_counts_its_toffolis(self):
        # q8 is written by two CCX, one CX and one X, then measured: 2 Toffolis become conditioned CZs
        r = self.scan([op(CCX, 0, 1, 8), op(CX, 2, 8), op(X, 8), op(CCX, 3, 4, 8), op(HMR, 8, b0=0)])
        self.assertEqual((r['sites'], r['ccx_sites'], r['toffoli_saving_static'], r['cx_to_z']), (1, 1, 2, 1))
        self.assertEqual(r['rows'][0]['first_write'], 0)

    def test_a_read_cuts_the_run(self):
        # q8 is read (control of a CX) after the first CCX: only the second CCX is in the run
        r = self.scan([op(CCX, 0, 1, 8), op(CX, 8, 9), op(CCX, 3, 4, 8), op(HMR, 8, b0=0)])
        self.assertEqual(r['toffoli_saving_static'], 1)
        self.assertEqual(r['rows'][0]['first_write'], 2)

    def test_diagonal_and_unmeasured_wires_do_not_count(self):
        r = self.scan([op(CCX, 0, 1, 8), op(CZ, 8, 2), op(HMR, 8, b0=0),    # CZ on q8 is a read
                       op(CCX, 0, 1, 9)])                                    # q9 never measured
        self.assertEqual(r['toffoli_saving_static'], 0)

    def test_conditioned_writes_block_unless_allowed(self):
        ops = [op(PUSH, bc=1), op(CCX, 0, 1, 8), op(POP), op(CCX, 2, 3, 8), op(HMR, 8, b0=0)]
        self.assertEqual(self.scan(ops)['toffoli_saving_static'], 1)
        r = self.scan(ops, conditioned=True)
        self.assertEqual((r['toffoli_saving_static'], r['toffoli_conditioned']), (1, 1))

    def test_width_freed_on_the_run(self):
        # q8 live from op 0 to its HMR; freeing it on the run lowers the 3-wire peak
        r = self.scan([op(CCX, 0, 1, 8), op(CCX, 0, 1, 8), op(HMR, 8, b0=0)])
        self.assertLessEqual(r['live_peak_after'], r['live_peak'])

    # ---- the rewrite emitter (qhdl measure-first --apply)
    def plan(self, ops):
        src = build(self.dir / 'p.kmb', ops, REGS, num_qubits=12, num_bits=4)
        return src, M.plan(src, M.scan(src)['rows'])

    def test_emitter_rewrites_each_write_in_place(self):
        _, (script, done, refused) = self.plan(
            [op(CCX, 0, 1, 8), op(CX, 2, 8), op(X, 8), op(CCX, 3, 4, 8), op(HMR, 8, b0=0)])
        lines = [l for l in script.splitlines() if l and not l.startswith('#')]
        self.assertEqual(lines, ['insert 0 => HMR q8 b0', 'replace 0 => CZ q0 q1 if b0', 'replace 1 => Z q2 if b0',
                                 'replace 2 => NEG if b0', 'replace 3 => CZ q3 q4 if b0', 'delete 4'])
        self.assertEqual((len(done), refused), (1, []))

    def test_emitter_refuses_when_the_bit_is_used_inside_the_run(self):
        # an op between the first write and the HMR is conditioned on the HMR's bit b0
        _, (_, done, refused) = self.plan(
            [op(CCX, 0, 1, 8), op(PUSH, bc=0), op(CX, 2, 3), op(POP), op(CCX, 4, 5, 8), op(HMR, 8, b0=0)])
        self.assertEqual(done, [])
        self.assertIn('bit b0 is used', refused[0]['reason'])

    def test_emitter_output_has_no_run_left(self):
        src, _ = self.plan([op(CCX, 0, 1, 8), op(CX, 2, 8), op(CCX, 3, 4, 8), op(HMR, 8, b0=0)])
        out = self.dir / 'out.kmb'
        app = M.apply_rewrite(src, out, M.scan(src)['rows'])
        self.assertEqual((app['applied'], app['toffolis_removed']), (1, 2))
        self.assertEqual(M.scan(out)['toffoli_saving_static'], 0)


if __name__ == '__main__':
    unittest.main()
