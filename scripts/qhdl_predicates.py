"""Exact computational-basis Boolean predicates and context-bound phase contracts.

ANFs use GF(2); phase coefficients use integer turns modulo 2**128. These are
separate rings: a general-angle phase on XOR cannot be split into two phases.
This module does not infer reachable-state relations or cross measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Iterable, Mapping

MOD = 1 << 128
HALF = 1 << 127
DEFAULT_TERM_CAP = 256


class InvalidContract(ValueError):
    pass


class ContextMismatch(InvalidContract):
    pass


class CapacityExceeded(InvalidContract):
    pass


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidContract(f'{name} must be an integer >= {minimum}')
    return value


def _object(value, keys, schema):
    if type(value) is not dict or set(value) != set(keys) | {'schema'} or value.get('schema') != schema:
        raise InvalidContract(f'invalid {schema} object')


@dataclass(frozen=True, order=True)
class Var:
    kind: str
    index: int
    version: int = 0

    def __post_init__(self):
        if self.kind not in ('quantum', 'classical'):
            raise InvalidContract('variable kind must be quantum or classical')
        _integer(self.index, 'variable index')
        _integer(self.version, 'variable version')

    def to_dict(self):
        return {'schema': 'qhdl.var.v1', 'kind': self.kind, 'index': self.index, 'version': self.version}

    @classmethod
    def from_dict(cls, value):
        _object(value, ('kind', 'index', 'version'), 'qhdl.var.v1')
        return cls(value['kind'], value['index'], value['version'])


def _monomial_key(monomial):
    return len(monomial), tuple(sorted(monomial))


@dataclass(frozen=True)
class ANF:
    terms: frozenset[frozenset[Var]]
    term_cap: int = field(default=DEFAULT_TERM_CAP, compare=False)

    def __post_init__(self):
        _integer(self.term_cap, 'term cap', 1)
        if type(self.terms) is not frozenset or any(type(m) is not frozenset for m in self.terms):
            raise InvalidContract('ANF requires a frozenset of frozenset monomials')
        if any(not isinstance(v, Var) for m in self.terms for v in m):
            raise InvalidContract('ANF monomials require typed Var objects')
        if len(self.terms) > self.term_cap:
            raise CapacityExceeded(f'ANF exceeds {self.term_cap} terms')

    @classmethod
    def constant(cls, value: bool, *, term_cap=DEFAULT_TERM_CAP):
        if type(value) is not bool:
            raise InvalidContract('Boolean constant requires True or False')
        return cls(frozenset((frozenset(),)) if value else frozenset(), term_cap)

    @classmethod
    def variable(cls, variable: Var, *, term_cap=DEFAULT_TERM_CAP):
        return cls(frozenset((frozenset((variable,)),)), term_cap)

    @classmethod
    def from_monomials(cls, monomials: Iterable[Iterable[Var]], *, term_cap=DEFAULT_TERM_CAP):
        _integer(term_cap, 'term cap', 1)
        result = set()
        for monomial_vars in monomials:
            item = frozenset(monomial_vars)
            if item in result:
                result.remove(item)
            else:
                result.add(item)
                if len(result) > term_cap:
                    raise CapacityExceeded(f'ANF intermediate exceeds {term_cap} terms')
        return cls(frozenset(result), term_cap)

    def __xor__(self, other):
        if not isinstance(other, ANF):
            return NotImplemented
        return ANF(self.terms ^ other.terms, min(self.term_cap, other.term_cap))

    def __and__(self, other):
        if not isinstance(other, ANF):
            return NotImplemented
        cap, result = min(self.term_cap, other.term_cap), set()
        for left in sorted(self.terms, key=_monomial_key):
            for right in sorted(other.terms, key=_monomial_key):
                item = left | right
                if item in result:
                    result.remove(item)
                else:
                    result.add(item)
                    if len(result) > cap:
                        raise CapacityExceeded(f'ANF product intermediate exceeds {cap} terms')
        return ANF(frozenset(result), cap)

    def __invert__(self):
        return self ^ ANF.constant(True, term_cap=self.term_cap)

    def support(self):
        return tuple(sorted({v for monomial_vars in self.terms for v in monomial_vars}))

    def evaluate(self, assignment: Mapping[Var, bool | int]):
        bits = {}
        for variable in self.support():
            if variable not in assignment:
                raise InvalidContract(f'missing assignment for {variable}')
            value = assignment[variable]
            if type(value) not in (bool, int) or value not in (0, 1):
                raise InvalidContract('assignments must be Boolean or integer 0/1')
            bits[variable] = bool(value)
        return bool(sum(all(bits[v] for v in m) for m in self.terms) % 2)

    def classify(self):
        degree = max((len(m) for m in self.terms), default=0)
        constant = frozenset() in self.terms
        if not self.support():
            kind = 'constant'
        elif len(self.terms) == 1:
            kind = 'monomial'
        elif degree <= 1:
            kind = 'affine' if constant else 'parity'
        else:
            kind = 'anf'
        support = self.support()
        return {'kind': kind, 'terms': len(self.terms), 'degree': degree,
                'quantum_degree': max((sum(v.kind == 'quantum' for v in m) for m in self.terms), default=0),
                'classical_degree': max((sum(v.kind == 'classical' for v in m) for m in self.terms), default=0),
                'quantum_variables': sum(v.kind == 'quantum' for v in support),
                'classical_variables': sum(v.kind == 'classical' for v in support),
                'has_constant_term': constant}

    def to_dict(self):
        return {'schema': 'qhdl.anf.v1', 'term_cap': self.term_cap,
                'terms': [[v.to_dict() for v in sorted(m)] for m in sorted(self.terms, key=_monomial_key)]}

    @classmethod
    def from_dict(cls, value):
        _object(value, ('terms', 'term_cap'), 'qhdl.anf.v1')
        if type(value['terms']) is not list or any(type(m) is not list for m in value['terms']):
            raise InvalidContract('serialized ANF monomials must be lists')
        result = cls(frozenset(frozenset(Var.from_dict(v) for v in m) for m in value['terms']), value['term_cap'])
        if result.to_dict() != value:
            raise InvalidContract('serialized ANF is not canonical or has duplicate variables/terms')
        return result


def monomial(variables: Iterable[Var], *, term_cap=DEFAULT_TERM_CAP):
    return ANF(frozenset((frozenset(variables),)), term_cap)


def parity(variables: Iterable[Var], *, term_cap=DEFAULT_TERM_CAP):
    return ANF.from_monomials(((v,) for v in variables), term_cap=term_cap)


def cube(positive: Iterable[Var] = (), negative: Iterable[Var] = (), *, term_cap=DEFAULT_TERM_CAP):
    """Conjunction of positive and complemented literals; contradictions are zero."""
    result = ANF.constant(True, term_cap=term_cap)
    for variable in sorted(set(positive)):
        result = result & ANF.variable(variable, term_cap=term_cap)
    for variable in sorted(set(negative)):
        result = result & ~ANF.variable(variable, term_cap=term_cap)
    return result


def equality_constant(variables: Iterable[Var], value: int, *, term_cap=DEFAULT_TERM_CAP):
    """variables[0] is the least significant bit; duplicate word wires reject."""
    word = tuple(variables)
    if len(set(word)) != len(word):
        raise InvalidContract('equality word contains duplicate variables')
    _integer(value, 'equality constant')
    if value >= 1 << len(word):
        raise InvalidContract('constant does not fit equality word')
    return cube((v for i, v in enumerate(word) if (value >> i) & 1),
                (v for i, v in enumerate(word) if not (value >> i) & 1), term_cap=term_cap)


def mux(select: ANF, when_true: ANF, when_false: ANF):
    return when_false ^ (select & (when_true ^ when_false))


def priority_guards(predicates: Iterable[ANF], *, include_default=False, term_cap=DEFAULT_TERM_CAP):
    """First true predicate wins; optional last guard means none was true."""
    available = ANF.constant(True, term_cap=term_cap)
    guards = []
    for predicate in predicates:
        guards.append(available & predicate)
        available = available & ~predicate
    if include_default:
        guards.append(available)
    return tuple(guards)


@dataclass(frozen=True)
class Context:
    source_sha256: str
    entry_cut: int
    basis: str = 'computational'
    domain: str = 'all_inputs'

    def __post_init__(self):
        if type(self.source_sha256) is not str or re.fullmatch('[0-9a-f]{64}', self.source_sha256) is None:
            raise InvalidContract('source SHA-256 must be 64 lowercase hexadecimal digits')
        _integer(self.entry_cut, 'entry cut')
        if self.basis != 'computational' or self.domain != 'all_inputs':
            raise InvalidContract('only computational-basis all-input contracts are supported')

    def to_dict(self):
        return {'schema': 'qhdl.context.v1', 'source_sha256': self.source_sha256,
                'entry_cut': self.entry_cut, 'basis': self.basis, 'domain': self.domain}

    @classmethod
    def from_dict(cls, value):
        _object(value, ('source_sha256', 'entry_cut', 'basis', 'domain'), 'qhdl.context.v1')
        return cls(value['source_sha256'], value['entry_cut'], value['basis'], value['domain'])


@dataclass(frozen=True)
class Predicate:
    context: Context
    anf: ANF

    def __post_init__(self):
        if not isinstance(self.context, Context) or not isinstance(self.anf, ANF):
            raise InvalidContract('Predicate requires Context and ANF')

    def equivalent(self, other):
        return predicate_equivalent(self, other)

    def to_dict(self):
        return {'schema': 'qhdl.predicate.v1', 'context': self.context.to_dict(), 'anf': self.anf.to_dict()}

    @classmethod
    def from_dict(cls, value):
        _object(value, ('context', 'anf'), 'qhdl.predicate.v1')
        return cls(Context.from_dict(value['context']), ANF.from_dict(value['anf']))


def _same_context(left, right):
    if left != right:
        raise ContextMismatch('source, entry cut, basis and domain must match exactly')


def predicate_equivalent(left: Predicate, right: Predicate):
    _same_context(left.context, right.context)
    return left.anf == right.anf


@dataclass(frozen=True)
class ExactPhaseTerm:
    predicate: Predicate
    classical_guard: ANF
    angle128: int

    def __post_init__(self):
        if not isinstance(self.predicate, Predicate) or not isinstance(self.classical_guard, ANF):
            raise InvalidContract('ExactPhaseTerm requires Predicate and ANF guard')
        if any(v.kind != 'classical' for v in self.classical_guard.support()):
            raise InvalidContract('classical guard contains a quantum variable')
        if type(self.angle128) is not int:
            raise InvalidContract('angle128 must be an integer number of 2^-128 turns')
        object.__setattr__(self, 'angle128', self.angle128 % MOD)

    def evaluate(self, assignment):
        predicate_value = self.predicate.anf.evaluate(assignment)
        guard_value = self.classical_guard.evaluate(assignment)
        active = predicate_value and guard_value
        return self.angle128 if active else 0

    def to_dict(self):
        return {'schema': 'qhdl.phase-term.v1', 'predicate': self.predicate.to_dict(),
                'classical_guard': self.classical_guard.to_dict(), 'angle128': f'{self.angle128:032x}'}

    @classmethod
    def from_dict(cls, value):
        _object(value, ('predicate', 'classical_guard', 'angle128'), 'qhdl.phase-term.v1')
        angle = value['angle128']
        if type(angle) is not str or re.fullmatch('[0-9a-f]{32}', angle) is None:
            raise InvalidContract('serialized angle128 requires exactly 32 lowercase hex digits')
        return cls(Predicate.from_dict(value['predicate']), ANF.from_dict(value['classical_guard']), int(angle, 16))


def _add_coefficient(poly, monomial_vars, coefficient, cap):
    value = (poly.get(monomial_vars, 0) + coefficient) % MOD
    if value:
        poly[monomial_vars] = value
    else:
        poly.pop(monomial_vars, None)
    if len(poly) > cap:
        raise CapacityExceeded(f'exact phase polynomial exceeds {cap} terms')


def phase_polynomial(terms: Iterable[ExactPhaseTerm], *, context: Context | None = None,
                     term_cap=DEFAULT_TERM_CAP):
    """Exact multilinear integer phase polynomial, modulo 2**128 turns.

    For a Boolean XOR f XOR m, a*f becomes a*f + a*m - 2*a*f*m.
    The correction is essential for general angles. Overflow rejects the proof.
    """
    _integer(term_cap, 'term cap', 1)
    terms = tuple(terms)
    if any(not isinstance(term, ExactPhaseTerm) for term in terms):
        raise InvalidContract('phase contract requires ExactPhaseTerm values')
    if context is None:
        if not terms:
            raise InvalidContract('an empty phase contract needs an explicit Context')
        context = terms[0].predicate.context
    if not isinstance(context, Context):
        raise InvalidContract('phase context must be a Context')
    total = {}
    for term in terms:
        _same_context(context, term.predicate.context)
        predicate = ANF(term.predicate.anf.terms, min(term_cap, term.predicate.anf.term_cap))
        guard = ANF(term.classical_guard.terms, min(term_cap, term.classical_guard.term_cap))
        effective = predicate & guard
        scaled = {}
        for item in sorted(effective.terms, key=_monomial_key):
            previous = dict(scaled)
            _add_coefficient(scaled, item, term.angle128, term_cap)
            for old_item, coefficient in previous.items():
                _add_coefficient(scaled, old_item | item, -2 * coefficient, term_cap)
        for item, coefficient in scaled.items():
            _add_coefficient(total, item, coefficient, term_cap)
    return dict(sorted(total.items(), key=lambda item: _monomial_key(item[0])))


def phase_equivalent(left, right, *, context=None, term_cap=DEFAULT_TERM_CAP):
    left, right = tuple(left), tuple(right)
    if any(not isinstance(term, ExactPhaseTerm) for term in left + right):
        raise InvalidContract('phase contract requires ExactPhaseTerm values')
    if context is None:
        combined = left + right
        if not combined:
            raise InvalidContract('empty phase equivalence needs an explicit Context')
        context = combined[0].predicate.context
    return phase_polynomial(left, context=context, term_cap=term_cap) == phase_polynomial(right, context=context, term_cap=term_cap)


def family_catalog():
    return [
        {'name': 'monomial', 'meaning': 'AND of typed variables; empty product is true'},
        {'name': 'parity', 'meaning': 'XOR of typed variables; duplicates cancel'},
        {'name': 'cube', 'meaning': 'AND of positive and complemented literals'},
        {'name': 'equality_constant', 'meaning': 'little-endian word equals a fitting integer constant'},
        {'name': 'mux', 'meaning': 'select ? when_true : when_false'},
        {'name': 'priority_guards', 'meaning': 'first-true disjoint guards, optional default'},
    ]


def to_json(value):
    return json.dumps(value.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidContract(f'duplicate JSON key {key}')
        result[key] = value
    return result


def from_json(text):
    def bad_constant(value):
        raise InvalidContract(f'non-finite JSON value {value}')
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=bad_constant)
    except (TypeError, json.JSONDecodeError) as error:
        raise InvalidContract('invalid JSON') from error
    if type(value) is not dict:
        raise InvalidContract('predicate JSON must be an object')
    types = {'qhdl.var.v1': Var, 'qhdl.anf.v1': ANF, 'qhdl.context.v1': Context,
             'qhdl.predicate.v1': Predicate, 'qhdl.phase-term.v1': ExactPhaseTerm}
    kind = value.get('schema')
    if type(kind) is not str or kind not in types:
        raise InvalidContract('unknown predicate JSON schema')
    return types[kind].from_dict(value)
