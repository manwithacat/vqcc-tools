"""Independent exhaustive Boolean/phase checks for the typed predicate interface."""
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import qhdl_predicates as P


def assignments(variables):
    return (dict(zip(variables, bits)) for bits in itertools.product((0, 1), repeat=len(variables)))


class Predicates(unittest.TestCase):
    def setUp(self):
        self.q = tuple(P.Var('quantum', i) for i in range(4))
        self.c = tuple(P.Var('classical', i, 2) for i in range(2))
        self.context = P.Context('a' * 64, 17)
        self.one = P.ANF.constant(True)

    def term(self, expression, angle, guard=None, context=None):
        return P.ExactPhaseTerm(P.Predicate(context or self.context, expression),
                                self.one if guard is None else guard, angle)

    def test_boolean_algebra_independent_truths(self):
        a, b, c, d = (P.ANF.variable(v) for v in self.q)
        left = (a & b) ^ c ^ self.one
        right = (b & d) ^ a
        for values in assignments(self.q):
            x, y, z, w = (values[v] for v in self.q)
            l, r = (x & y) ^ z ^ 1, (y & w) ^ x
            self.assertEqual(left.evaluate(values), bool(l))
            self.assertEqual((left ^ right).evaluate(values), bool(l ^ r))
            self.assertEqual((left & right).evaluate(values), bool(l & r))
            self.assertEqual((~left).evaluate(values), not l)
        self.assertEqual(a & a, a)
        self.assertEqual(a ^ a, P.ANF.constant(False))
        self.assertEqual(P.monomial([self.q[0], self.q[0]]), a)
        self.assertEqual(P.parity([self.q[0], self.q[0]]), P.ANF.constant(False))
        self.assertEqual(a & ~a, P.ANF.constant(False))

    def test_family_catalogue_exhaustive(self):
        a, b, c = self.q[:3]
        for values in assignments((a, b, c)):
            x, y, z = (values[v] for v in (a, b, c))
            self.assertEqual(P.monomial((a, b, c)).evaluate(values), bool(x and y and z))
            self.assertEqual(P.parity((a, b, c)).evaluate(values), bool(x ^ y ^ z))
            self.assertEqual(P.cube((a, c), (b,)).evaluate(values), bool(x and not y and z))
            self.assertFalse(P.cube((a,), (a,)).evaluate(values))
            self.assertEqual(P.mux(P.ANF.variable(a), P.ANF.variable(b), P.ANF.variable(c)).evaluate(values), bool(y if x else z))
            for constant in range(8):
                self.assertEqual(P.equality_constant((a, b, c), constant).evaluate(values), (x + 2*y + 4*z) == constant)
            guards = P.priority_guards([P.ANF.variable(v) for v in (a, b, c)], include_default=True)
            actual = [int(g.evaluate(values)) for g in guards]
            expected = [int(bool(x)), int(not x and bool(y)), int(not x and not y and bool(z)), int(not x and not y and not z)]
            self.assertEqual(actual, expected)
            self.assertEqual(sum(actual), 1)
        self.assertTrue(P.monomial(()).evaluate({}))
        self.assertFalse(P.parity(()).evaluate({}))
        self.assertTrue(P.equality_constant((), 0).evaluate({}))
        with self.assertRaises(P.InvalidContract): P.equality_constant((a,), 2)
        with self.assertRaises(P.InvalidContract): P.equality_constant((a, a), 0)
        self.assertEqual(len(P.family_catalog()), 6)
        json.dumps(P.family_catalog(), allow_nan=False)

    def test_typed_support_versions_and_context(self):
        q0, c0, q_next = self.q[0], self.c[0], P.Var('quantum', 0, 1)
        expression = P.parity((q0, c0))
        info = expression.classify()
        self.assertEqual(info['kind'], 'parity')
        self.assertEqual((info['quantum_variables'], info['classical_variables']), (1, 1))
        self.assertEqual(expression.support(), tuple(sorted((q0, c0))))
        original = P.Predicate(self.context, P.ANF.variable(q0))
        self.assertTrue(original.equivalent(P.Predicate(self.context, P.ANF.variable(q0))))
        self.assertFalse(original.equivalent(P.Predicate(self.context, P.ANF.variable(q_next))))
        self.assertFalse(original.equivalent(P.Predicate(self.context, P.ANF.variable(P.Var('classical', 0)))))
        for other in (P.Context('b'*64,17), P.Context('a'*64,18)):
            with self.assertRaises(P.ContextMismatch): original.equivalent(P.Predicate(other, original.anf))
        for kwargs in ({'basis':'Hadamard'}, {'domain':'reachable_inputs'}):
            with self.assertRaises(P.InvalidContract): P.Context('a'*64,17,**kwargs)
        for arguments in (('a'*63,0), ('A'*64,0), ('a'*64,-1), ('a'*64,True)):
            with self.assertRaises(P.InvalidContract): P.Context(*arguments)
        for args in (('ancilla',0), ('quantum',True), ('classical',0,-1)):
            with self.assertRaises(P.InvalidContract): P.Var(*args)
        self.assertFalse(P.phase_equivalent([self.term(P.ANF.variable(c0), 1)],
                                           [self.term(P.ANF.variable(P.Var('classical',0,3)), 1)]))

    def test_capacity_is_rejection_not_truncation(self):
        a = P.ANF.variable(self.q[0], term_cap=1)
        b = P.ANF.variable(self.q[1], term_cap=1)
        with self.assertRaises(P.CapacityExceeded): a ^ b
        with self.assertRaises(P.CapacityExceeded): ~a
        with self.assertRaises(P.CapacityExceeded): P.parity(self.q[:2], term_cap=1)
        left = P.parity(self.q[:2], term_cap=2)
        right = P.parity(self.q[2:], term_cap=2)
        with self.assertRaises(P.CapacityExceeded): left & right
        with self.assertRaises(P.CapacityExceeded): P.ANF(left.terms, term_cap=1)
        with self.assertRaises(P.InvalidContract): P.ANF(frozenset(), term_cap=0)
        with self.assertRaises(P.InvalidContract): P.ANF([{self.q[0]}])

    def test_sign_phase_and_general_angle_counterexample(self):
        a, b = (P.ANF.variable(v) for v in self.q[:2])
        for theta in (P.HALF, 1, (1 << 97) + 3, P.MOD - 1):
            combined = self.term(a ^ b, theta)
            split = [self.term(a, theta), self.term(b, theta)]
            equal = P.phase_equivalent([combined], split)
            self.assertEqual(equal, theta == P.HALF)
            for values in assignments(self.q[:2]):
                expected = theta if values[self.q[0]] ^ values[self.q[1]] else 0
                self.assertEqual(combined.evaluate(values), expected)
                polynomial = P.phase_polynomial([combined])
                evaluated = sum(coef for m, coef in polynomial.items() if all(values[v] for v in m)) % P.MOD
                self.assertEqual(evaluated, expected)
        self.assertEqual(self.term(a, -1).angle128, P.MOD - 1)
        self.assertEqual(self.term(a, P.MOD + 1).angle128, 1)

    def test_global_phase_guards_and_context_fail_closed(self):
        zero, one = P.ANF.constant(False), self.one
        self.assertFalse(P.phase_equivalent([self.term(one, P.HALF)], [self.term(zero, P.HALF)]))
        self.assertTrue(P.phase_equivalent([self.term(one,P.HALF),self.term(one,P.HALF)], [], context=self.context))
        self.assertTrue(P.phase_equivalent([], [], context=self.context))
        with self.assertRaises(P.InvalidContract): P.phase_equivalent([], [])
        with self.assertRaises(P.InvalidContract): P.phase_equivalent([object()], [])
        with self.assertRaises(P.InvalidContract): P.phase_polynomial([object()])
        with self.assertRaises(P.ContextMismatch):
            P.phase_equivalent([self.term(zero,0)], [self.term(zero,0,context=P.Context('b'*64,17))])
        with self.assertRaises(P.InvalidContract): self.term(one, 1, guard=P.ANF.variable(self.q[0]))
        guarded = self.term(P.parity(self.q[:2]), 3, P.monomial(self.c))
        with self.assertRaises(P.InvalidContract):
            self.term(P.ANF.constant(False), 3, P.ANF.variable(self.c[0])).evaluate({})
        for values in assignments(self.q[:2] + self.c):
            x,y,b,c = (values[v] for v in self.q[:2] + self.c)
            self.assertEqual(guarded.evaluate(values), 3 if (x ^ y) and b and c else 0)
        four = P.parity(self.q, term_cap=4)
        with self.assertRaises(P.CapacityExceeded): P.phase_polynomial([self.term(four,1)],term_cap=4)
        self.assertEqual(len(P.phase_polynomial([self.term(four,P.HALF)],term_cap=4)),4)

    def test_phase_integer_expansion_against_direct_arithmetic(self):
        a,b,c,d = (P.ANF.variable(v) for v in self.q)
        g = P.ANF.variable(self.c[0])
        terms = [self.term((a & b) ^ c ^ self.one, 7, g),
                 self.term(P.mux(a,c,d), (1 << 119) + 13),
                 self.term(self.one, 1), self.term(b, -3, ~g)]
        polynomial = P.phase_polynomial(terms)
        for values in assignments(self.q + self.c[:1]):
            x,y,z,w,guard = (values[v] for v in self.q + self.c[:1])
            expected = (7 * (((x & y) ^ z ^ 1) & guard) +
                        ((1 << 119) + 13) * (z if x else w) + 1 - 3 * (y & (1-guard))) % P.MOD
            actual = sum(coef for m,coef in polynomial.items() if all(values[v] for v in m)) % P.MOD
            self.assertEqual(actual, expected)

    def test_strict_serialization_roundtrip_and_low_bits(self):
        expression = P.parity(self.q[:2] + self.c[:1]) ^ self.one
        predicate = P.Predicate(self.context, expression)
        phase = self.term(expression, (1 << 123) + 1, P.ANF.variable(self.c[1]))
        for item in (self.q[0], expression, self.context, predicate, phase):
            encoded = P.to_json(item)
            self.assertEqual(P.from_json(encoded), item)
            self.assertEqual(P.to_json(P.from_json(encoded)), encoded)
            self.assertEqual(type(item).from_dict(item.to_dict()), item)
        self.assertEqual(phase.to_dict()['angle128'], f'{(1 << 123) + 1:032x}')
        malformed = []
        value = expression.to_dict(); value['extra'] = 1; malformed.append(value)
        value = expression.to_dict(); value['terms'].append(value['terms'][0]); malformed.append(value)
        value = P.ANF.variable(self.q[0]).to_dict(); value['terms'][0].append(value['terms'][0][0]); malformed.append(value)
        value = phase.to_dict(); value['angle128'] = '1'; malformed.append(value)
        value = phase.to_dict(); value['angle128'] = 'A'*32; malformed.append(value)
        value = self.q[0].to_dict(); value['index'] = True; malformed.append(value)
        value = self.context.to_dict(); value['schema'] = 'unknown'; malformed.append(value)
        for value in malformed:
            with self.assertRaises(P.InvalidContract): P.from_json(json.dumps(value))
        for text in ('[]','{"schema":"a","schema":"b"}','{"schema":NaN}', '{broken'):
            with self.assertRaises(P.InvalidContract): P.from_json(text)

    def test_assignment_validation_and_immutability(self):
        expression = P.ANF.variable(self.q[0])
        with self.assertRaises(P.InvalidContract): expression.evaluate({})
        with self.assertRaises(P.InvalidContract): expression.evaluate({self.q[0]: 2})
        with self.assertRaises(P.InvalidContract): expression.evaluate({self.q[0]: 0.0})
        with self.assertRaises(AttributeError): expression.terms = frozenset()
        self.assertEqual(P.ANF.constant(False).classify()['kind'], 'constant')
        self.assertEqual(P.monomial(self.q[:2]).classify()['kind'], 'monomial')
        self.assertEqual((~expression).classify()['kind'], 'affine')


if __name__ == '__main__':
    unittest.main()
