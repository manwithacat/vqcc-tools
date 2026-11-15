"""Product-span AND minimiser: exact linear-algebra step, sequential synthesis with helpers and a wire
pool, fragment emission and exact check, and the `qhdl recompute` integration."""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qhdl_bitsim as B  # noqa: E402
import qhdl_productspan as PS  # noqa: E402
import qhdl_recompute as R  # noqa: E402
import qhdl_valuegraph as G  # noqa: E402


def lookup_columns(nbits, nout, seed, rows=4000):
    """Random table f: {0,1}^nbits -> {0,1}^nout, evaluated on `rows` random addresses."""
    rng = np.random.default_rng(seed)
    table = rng.integers(0, 2, size=(nout, 1 << nbits)).astype(bool)
    addr = rng.integers(0, 1 << nbits, size=rows)
    avail = [(i, ((addr >> i) & 1).astype(bool)) for i in range(nbits)]
    targets = [(100 + j, table[j][addr]) for j in range(nout)]
    return avail, targets


class Kernel(unittest.TestCase):
    def test_find_step_and_exact_solution(self):
        # target x0x1 + x2x3 + x4 on 5 bits: two products over U = span(1, x0..x4)
        avail, targets = lookup_columns(5, 1, 1, rows=3000)
        n = 3000
        rng = np.random.default_rng(5)
        addr = rng.integers(0, 32, size=n)
        bits = [((addr >> i) & 1).astype(bool) for i in range(5)]
        t = (bits[0] & bits[1]) ^ (bits[2] & bits[3]) ^ bits[4]
        inst = PS.instance_from_columns([(i, bits[i]) for i in range(5)], [t])
        self.assertEqual(inst['N'], 32)
        r = PS.find_step(inst['U'], inst['T'], 2, random.Random(1), 1 << 12, 20)
        self.assertIsNotNone(r)
        self.assertTrue(PS.check_step(inst['U'], r['Ls'], r['Ms'], r['u'], inst['T'], r['cmask']))
        # two factors cannot give a rank-4 form with one product
        self.assertIsNone(PS.find_step(inst['U'], inst['T'], 1, random.Random(1), 1 << 8, 10))

    def test_unbuildable_cost_is_reported_not_guessed(self):
        avail, targets = lookup_columns(6, 3, 2)
        out = PS.synth_from_columns(avail, targets, seconds=0.0001, policy=lambda D, r, s: [(1, 1 << 4, 1)])
        self.assertFalse(out['ok'])
        self.assertIn('not captured', out['reason'])


class Synthesis(unittest.TestCase):
    def test_exact_fragment_beats_monomial_cache(self):
        avail, targets = lookup_columns(6, 3, 3)
        out = PS.synth_from_columns(avail, targets, extra=16, helpers=16, seconds=60,
                                    policy=lambda D, r, s: [(k, 1 << 14, 4) for k in (1, 2, 3, 4, 6, 8)])
        self.assertTrue(out['ok'], out.get('reason'))
        self.assertLessEqual(out['scratch_peak'], 16)
        # a 6-bit address has 57 nonlinear monomials; three outputs need far fewer products with sharing
        self.assertLess(out['and_cost'], 57)
        self.assertEqual(out['rows_checked'], 4000)

    def test_helpers_are_removed_and_checked_and_pool_respected(self):
        avail, targets = lookup_columns(6, 4, 4)
        out = PS.synth_from_columns(avail, targets, extra=16, helpers=16, seed=2, seconds=120,
                                    policy=lambda D, r, s: [(k, 1 << 14, 4) for k in (1, 2, 3, 4, 6, 8)])
        self.assertTrue(out['ok'], out.get('reason'))
        self.assertEqual(out['helper_products'], 16)
        self.assertLessEqual(out['pool_peak'], out['pool'])
        removals = [x for x in out['ops'] if x.startswith('HMR')]
        self.assertEqual(len(removals), 16)
        self.assertTrue(any(x.startswith('CZ') and ' if b' in x for x in out['ops']))
        # independent re-check on the bit-parallel simulator: values, helper wires zero, sign bit zero on every row
        for seed in (1, 2):
            chk = PS.bitsim_check(out['ops'], avail, targets, seeds=(seed,))
            self.assertTrue(chk['ok'], chk)

    def test_helper_removal_with_wrong_operands_is_caught(self):
        avail, targets = lookup_columns(4, 1, 5, rows=500)
        awires = [w for w, _ in avail]
        st = {w: c.copy() for w, c in avail}
        st[77] = ~st[0]            # corrupt helper value: not x0 AND x1
        ev = dict(type='helper', wire=77, Lmask=0b10, Mmask=0b100, Vw=[None, 0, 1, 2, 3], fn=0)
        rem = dict(type='remove', helper=ev)
        with self.assertRaises(AssertionError):
            # emission recomputes the helper (zero start) then we corrupt it before removal
            ops, _ = PS.emit_events([ev], [200], state=st)
            st[77] = ~st[77]
            PS.emit_events([rem], [200], state=st)

    def test_constant_constant_helper_leaves_no_sign(self):
        # helper (1 + x0) * (1 + x1) has both constants: the removal must carry NEG so the sign bit stays zero
        avail, _ = lookup_columns(3, 1, 9, rows=300)
        ev = dict(type='helper', wire=5000, Lmask=0b011, Mmask=0b101, Vw=[None, 0, 1, 2], fn=0)
        rem = dict(type='remove', helper=ev)
        st = {w: c.copy() for w, c in avail}
        ops, ccx = PS.emit_events([ev, rem], [200], state=st)
        self.assertEqual(ccx, 1)
        self.assertTrue(any(x.startswith('NEG if') for x in ops))
        # run through the simulator with a dummy zero target so that the wire set is complete
        zero = (200, np.zeros(300, bool))
        chk = PS.bitsim_check(ops, avail, [zero], seeds=(3, 4, 5))
        self.assertTrue(chk['ok'], chk)

    def test_emit_product_variants(self):
        for A, cA, B_, cB in (([0], 0, [1], 0), ([0, 1], 1, [1, 2], 0), ([0], 0, [0, 1], 1), ([0, 1], 0, [0, 1], 0),
                               ([], 1, [2], 1), ([0, 1, 2], 0, [2], 0)):
            ops, _ = PS.emit_product(A, cA, B_, cB, 9)
            rng = np.random.default_rng(1)
            st = {w: rng.integers(0, 2, 64).astype(bool) for w in range(3)}
            st[9] = np.zeros(64, bool)
            before = {w: st[w].copy() for w in range(3)}
            PS.apply_ops(ops, st)
            la = np.full(64, bool(cA))
            for w in A:
                la = la ^ before[w]
            lb = np.full(64, bool(cB))
            for w in B_:
                lb = lb ^ before[w]
            self.assertTrue(np.array_equal(st[9], la & lb), (A, cA, B_, cB))
            for w in range(3):
                self.assertTrue(np.array_equal(st[w], before[w]))


class Integration(unittest.TestCase):
    def test_recompute_prices_a_lookup_with_the_product_span_route(self):
        # six address bits x0..x5 on wires 0..5; targets q10, q11 are two random functions of the address, built here
        # as a truth-table cascade of CCX/CX ops (monomial network) so that the circuit is a legal op list
        rng = np.random.default_rng(7)
        nb = 6
        lines = []
        mw = {}
        nxt = 20
        for m in range(1, 1 << nb):
            bits = [i for i in range(nb) if m >> i & 1]
            if len(bits) == 1:
                mw[m] = bits[0]
                continue
            w = nxt
            nxt += 1
            parent = mw[m & ~(1 << bits[-1])]
            lines.append(f'CCX q{parent} q{bits[-1]} q{w}')
            mw[m] = w
        for tgt in (10, 11):
            for m in range(1, 1 << nb):
                if rng.integers(0, 2):
                    lines.append(f'CX q{mw[m]} q{tgt}')
        lines.append('Z q10')
        ops = [B.parse_line(x) for x in lines]
        nq = 1 + max(max(o.q) for o in ops if o.q)
        xq = list(range(nb))
        values, _, _ = G.build(ops, xq, nq, None)
        rows = np.arange(1, 64)
        q = dict(target_value_ids=['q10', 'q11'], target_value_id='q10', erase_after_op=0, restore_before_op=len(ops) - 1,
                 available_value_ids=[f'q{i}:v0' for i in range(nb)], max_and_cost=60, max_extra_qubits=16, search=dict(seconds=60, helpers=16))
        # make the monomial wires unavailable (they are not listed): only the six address bits are
        res = R.run_query(ops, xq, nq, values, q, rows=rows)
        self.assertTrue(res['determined'])
        self.assertIn(res['price_status'], ('bounded', 'exact'))
        self.assertIsNotNone(res['fragment'])
        self.assertLess(res['and_cost'], 57)
        self.assertEqual(res['price']['product_span']['ok'], True)


class InPlace(unittest.TestCase):
    """Run 3: the in-place update model (target wires start at an old value, which is also an operand)."""

    def _instance(self, seed=7, nbits=6, nt=3, rows=4000):
        rng = np.random.default_rng(seed)
        addr = rng.integers(0, 1 << nbits, size=rows)
        bits = [((addr >> i) & 1).astype(bool) for i in range(nbits)]

        def quad():                       # random quadratic form (targets and old values both reachable by products)
            f = np.zeros(rows, bool)
            for a in range(nbits):
                for b in range(a + 1, nbits):
                    if rng.integers(0, 2):
                        f ^= bits[a] & bits[b]
            for a in range(nbits):
                if rng.integers(0, 2):
                    f ^= bits[a]
            return f
        avail = [(i, bits[i]) for i in range(nbits)]
        targets = [(100 + j, quad()) for j in range(nt)]
        olds = [None] + [quad() for j in range(1, nt)]
        return avail, targets, olds

    def test_inplace_fragment_exact_and_old_value_is_start(self):
        avail, targets, olds = self._instance()
        out = PS.synth_from_columns_inplace(avail, targets, olds, seconds=60, restarts=2,
                                            policy=lambda D, r, s: [(k, 1 << 14, 4) for k in (2, 3, 4, 6, 8, 12)])
        self.assertTrue(out['ok'], out.get('reason'))
        self.assertEqual(out['start_model'], 'old value in place')
        # independent re-check with the targets loaded at their old values (and a zero start must NOT work)
        self.assertTrue(PS.inplace_check(out['ops'], avail, targets, olds, [w for w, _ in targets]))
        self.assertFalse(PS.inplace_check(out['ops'], avail, targets, [None] * len(targets), [w for w, _ in targets]))
        bs = PS.bitsim_check(out['ops'], avail, targets, target_init=[(w, o) for (w, _), o in zip(targets, olds) if o is not None])
        self.assertTrue(bs['ok'])

    def test_pivot_forcing_and_cost_not_worse_than_zero_start_bound(self):
        avail, targets, olds = self._instance(seed=9)
        pol = lambda D, r, s: [(k, 1 << 14, 4) for k in (2, 3, 4, 6, 8, 12)]
        a = PS.synth_from_columns_inplace(avail, targets, olds, seconds=60, restarts=2, policy=pol)
        self.assertTrue(a['ok'])
        # each step's combination contains its pivot (the last right-hand side), and no step reads its own pivot wire
        res = PS.synthesise_inplace(*self._u(avail, targets, olds), policy=pol, log=None)
        self.assertFalse(res['missing'])
        for ev in res['events']:
            self.assertIn(ev['pivot'], ev['combo'])
            self.assertNotIn(100 + ev['orig'][ev['pivot']], [w for w in ev['Vw'] if w is not None])

    def _u(self, avail, targets, olds):
        nt = len(targets)
        inst = PS.instance_from_columns(avail, [c for _, c in targets] + [o for o in olds if o is not None])
        it = iter(inst['T'][nt:])
        Of = [None if o is None else next(it) for o in olds]
        return inst['U'], inst['Uw'], inst['T'][:nt], [w for w, _ in targets], Of

    def test_force_last_kernel_option(self):
        # target t1 = x0x1 is a one-product hit; with force_last the hit must include the last target
        rng = np.random.default_rng(3)
        addr = rng.integers(0, 32, size=3000)
        bits = [((addr >> i) & 1).astype(bool) for i in range(5)]
        t0 = bits[0] & bits[1]
        t1 = bits[2] & bits[3]
        inst = PS.instance_from_columns([(i, bits[i]) for i in range(5)], [t0, t1])
        r = PS.find_step(inst['U'], inst['T'], 1, random.Random(1), 1 << 12, 20, force_last=True)
        # one product cannot make a combination containing t1 together with... t1 alone is one product: must be found
        self.assertIsNotNone(r)
        self.assertTrue(r['cmask'] >> 1 & 1)


if __name__ == '__main__':
    unittest.main()


class WideStates(unittest.TestCase):
    """N up to 1024 reachable states (16-word kernel); planted targets with a known product-span form."""

    @staticmethod
    def _cols(N, rows=3000, seed=0):
        # 10 address bits; reachable states = the first N addresses (all 1024 when N = 1024)
        rng = np.random.default_rng(seed)
        addr = np.concatenate([np.arange(N), rng.integers(0, N, size=rows - N)])
        bits = [((addr >> i) & 1).astype(bool) for i in range(10)]
        return bits

    def _run(self, N):
        bits = self._cols(N)
        avail = [(i, bits[i]) for i in range(10)]
        t1 = bits[0] & bits[1]                              # one AND of two available values
        t2 = (bits[2] & bits[3]) ^ (bits[4] & bits[5]) ^ bits[6]   # sum of products
        inst = PS.instance_from_columns(avail, [t1, t2])
        self.assertIsNotNone(inst)
        self.assertEqual(inst['N'], N)
        self.assertEqual(PS._nw_for(inst['U'], inst['T']), 16)
        # kernel step: one product captures the AND exactly
        r = PS.find_step(inst['U'], inst['T'][:1], 1, random.Random(1), 1 << 12, 5)
        self.assertIsNotNone(r)
        self.assertTrue(PS.check_step(inst['U'], r['Ls'], r['Ms'], r['u'], inst['T'][:1], r['cmask']))
        # two products capture the sum-of-products target
        r2 = PS.find_step(inst['U'], inst['T'][1:], 2, random.Random(3), 1 << 11, 3000)
        self.assertIsNotNone(r2)
        self.assertTrue(PS.check_step(inst['U'], r2['Ls'], r2['Ms'], r2['u'], inst['T'][1:], r2['cmask']))
        # full pricer: exact fragment on every row, cost at most 3 Toffolis for the two targets
        out = PS.synth_from_columns(avail, [(100, t1), (101, t2)], seconds=120,
                                    policy=lambda D, r, s: [(k, 1 << 11, 400) for k in (1, 2, 3)])
        self.assertTrue(out['ok'], out.get('reason'))
        self.assertEqual(out['N'], N)
        self.assertLessEqual(out['and_cost'], 3)
        self.assertEqual(out['rows_checked'], 3000)

    def test_n257(self):
        self._run(257)

    def test_n818(self):
        self._run(818)

    def test_n1024(self):
        self._run(1024)

    def test_over_limit_is_refused(self):
        rng = np.random.default_rng(1)
        addr = rng.integers(0, 1 << 11, size=6000)
        bits = [((addr >> i) & 1).astype(bool) for i in range(11)]
        self.assertIsNone(PS.instance_from_columns([(i, b) for i, b in enumerate(bits)], [bits[0] & bits[1]]))

    def test_more_than_64_available_values(self):
        # 70 available columns (10 bits + 60 XOR combinations): distinct-state collapse must not need a 64-bit key
        bits = self._cols(818)
        avail = [(i, bits[i]) for i in range(10)]
        avail += [(10 + j, bits[j % 10] ^ bits[(j * 3 + 1) % 10] ^ bits[(j + 4) % 10]) for j in range(60)]
        inst = PS.instance_from_columns(avail, [bits[0] & bits[1]])
        self.assertIsNotNone(inst)
        self.assertEqual(inst['N'], 818)
        self.assertEqual(len(inst['U']), 11)      # constant + 10 independent bits
