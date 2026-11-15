"""qhdl_dirty_fanout: dirty-accumulator fan-out pairs that a clean helper reduces from 2 Toffolis to 1."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import qhdl_dirty_fanout as F  # noqa: E402
from test_qhdl_census import CCX, CX, HMR, REGS, X, build, op  # noqa: E402


class DirtyFanout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, ops, nq=12):
        return F.scan(build(self.dir / 'c.kmb', ops, REGS, num_qubits=nq, num_bits=4))

    def fan(self, a=0, b=1, w=2, targets=(3, 4)):
        return ([op(CX, w, x) for x in targets] + [op(CCX, a, b, w)] + [op(CX, w, x) for x in targets]
                + [op(CCX, a, b, w)])

    def test_pair_with_free_wire_saves_one(self):
        # wires 0..4 live; q11 never used -> free helper
        r = self.scan(self.fan())
        self.assertEqual((r['pair_sites'], r['toffoli_saving_static']), (1, 1))
        self.assertEqual(r['sites'][0]['ccx_in_segment'], 2)

    def test_write_to_control_inside_window_blocks(self):
        ops = self.fan()
        ops.insert(4, op(X, 0))   # a written between the two CCX
        self.assertEqual(self.scan(ops)['pair_sites'], 0)

    def test_other_use_of_accumulator_blocks(self):
        ops = self.fan()
        ops.insert(4, op(CCX, 2, 5, 6))   # w used as a CCX control inside the window
        self.assertEqual(self.scan(ops)['pair_sites'], 0)

    def test_no_free_wire_no_saving(self):
        # every wire live across the window: touch all 6 wires before and after
        nq = 6
        ops = [op(X, q) for q in range(nq)] + self.fan(targets=(3, 4, 5)) + [op(X, q) for q in range(nq)]
        r = self.scan(ops, nq=nq)
        self.assertEqual((r['pair_sites'], r['toffoli_saving_static']), (1, 0))



class Episodes(unittest.TestCase):
    def test_interleaved_products_close_once(self):
        with tempfile.TemporaryDirectory() as d:
            # w=2 accumulates p1=(0,1) and p2=(5,6) interleaved, returns to empty once: one saving
            ops = [op(CCX, 0, 1, 2), op(CX, 2, 3), op(CCX, 5, 6, 2), op(CX, 2, 4), op(CCX, 0, 1, 2), op(CX, 2, 3),
                   op(CCX, 5, 6, 2)]
            r = F.scan(build(Path(d) / 'e.kmb', ops, REGS, num_qubits=12, num_bits=4))
            self.assertEqual((r['pair_sites'], r['toffoli_saving_static']), (1, 1))
            self.assertEqual(r['sites'][0]['ccx_in_segment'], 4)


if __name__ == '__main__':
    unittest.main()
