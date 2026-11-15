"""Exact phase folding in bounded unconditional reversible regions.

The state at every analysis cut is arbitrary: all wires, including scratch,
are independent Boolean inputs. ANF equality is therefore an all-input proof.
Phases are unsigned 128-bit integers in units of a full turn / 2**128,
matching kickmix FixedPrecisionAngle128. No floating arithmetic is used.
Measurements, resets, classical conditions and unknown gates are barriers.
"""
from __future__ import annotations

import numpy as np
import qhdl_rewrite as R

MOD = 1 << 128
HALF = 1 << 127
MAX_TERMS = 256
MAX_SPAN = 2048
DIAGONAL = {0x10, 0x20, 0x30, 0x13}  # Z, CZ, CCZ, Z_POW
REVERSIBLE = {0x11, 0x21, 0x31, 0x22, 0x32}


def rotation_kind(angle):
    """Kickmix hierarchy: Clifford free, odd eighth-turn T, otherwise synthesized."""
    if angle % (1 << 126) == 0:
        return 'clifford'
    if angle % (1 << 125) == 0:
        return 't'
    return 'general'


class CapacityExceeded(Exception):
    pass


def product(a, b, cap):
    if len(a) * len(b) > cap * cap:
        raise CapacityExceeded
    out = set()
    for x in a:
        for y in b:
            out.symmetric_difference_update((x | y,))
            if len(out) > cap:
                raise CapacityExceeded
    return frozenset(out)


def scan(k, ops, A=0, B=None, max_terms=MAX_TERMS, max_span=MAX_SPAN, include_predicates=False):
    """Return independent folding groups plus coverage; no circuit is mutated."""
    if max_terms < 1 or max_span < 1:
        raise ValueError('symbolic budgets must be positive')
    B = len(k.op_types) if B is None else B
    if not 0 <= A <= B <= len(k.op_types):
        raise ValueError('invalid phase scan range')
    angle_ops = np.flatnonzero((k.op_types & 0x7f) == 0x13)
    off, n, _ = k.offsets['angles']
    raw = np.frombuffer(k.raw, dtype=np.uint8, count=16*n, offset=off).reshape(n, 16)
    angles = {int(i): int.from_bytes(bytes(raw[j]), 'little')
              for j, i in enumerate(angle_ops) if A <= i < B}
    state, groups, found = {}, {}, []
    stats = dict(phase_ops=0, supported_ops=0, barriers=0, capacity_cuts=0,
                 span_cuts=0, regions=0, max_terms=max_terms, max_span=max_span)
    start = A
    inventory = []

    def val(w):
        return state.setdefault(w, frozenset((1 << w,)))

    def finish(end, cut_reason):
        for pred, items in groups.items():
            total = sum(x['angle'] for x in items) % MOD if pred else 0
            # Retain global phase for a constant-one predicate, just as any other.
            if include_predicates:
                inventory.append(dict(region=stats['regions'], entry_cut=start,
                                      region_end=end, cut_reason=cut_reason,
                                      ops=[x['op'] for x in items],
                                      predicate=[hex(m) for m in sorted(pred)],
                                      degree=max((m.bit_count() for m in pred), default=0),
                                      terms=len(pred), angle128=f'{total:032x}',
                                      phase_ops=len(items), rotations=sum(x['code']==0x13 for x in items)))
            single = [x for x in items if x['code'] in (0x10, 0x13)]
            anchor = single[0] if single else items[0]
            if total == 0:
                replacement = None
            elif total == HALF:
                q = anchor['q']
                replacement = ('Z' if len(q) == 1 else 'CZ' if len(q) == 2 else 'CCZ') + ''.join(f' q{w}' for w in q)
            elif single:
                replacement = f"Z_POW q{anchor['q'][0]} angle128:{total:032x}"
            else:
                continue  # CZ/CCZ angles can only sum to zero or half a turn.
            before_t = sum(x['code'] == 0x30 for x in items)
            after_t = int(bool(replacement and replacement.startswith('CCZ ')))
            before_r = sum(x['code'] == 0x13 for x in items)
            after_r = int(bool(replacement and replacement.startswith('Z_POW ')))
            if len(items) == int(replacement is not None) and before_t == after_t and before_r == after_r:
                continue
            found.append(dict(ops=[x['op'] for x in items], anchor=anchor['op'], replacement=replacement,
                              predicate=sorted(pred), angle128=f'{total:032x}',
                              toffolis_before=before_t, toffolis_after=after_t,
                              rotations_before=before_r, rotations_after=after_r,
                              general_rotations_delta=(int(bool(after_r) and rotation_kind(total)=='general') -
                                  sum(x['code']==0x13 and rotation_kind(x['angle'])=='general' for x in items)),
                              t_gates_delta=(int(bool(after_r) and rotation_kind(total)=='t') -
                                  sum(x['code']==0x13 and rotation_kind(x['angle'])=='t' for x in items))))
        state.clear()
        groups.clear()
        stats['regions'] += 1

    # Small chunks avoid a Python record for every op of a large circuit.
    for lo in range(A, B, 1024):
        for i, (code, q, bit) in ops.records(range(lo, min(lo+1024, B))).items():
            if i - start >= max_span:
                finish(i, 'span'); start = i; stats['span_cuts'] += 1
            if code == 0x06:
                continue
            if ops.depth[i] or code not in DIAGONAL | REVERSIBLE:
                finish(i, 'barrier'); start = i+1; stats['barriers'] += 1
                continue
            try:
                if code in DIAGONAL:
                    p = frozenset((0,))
                    for w in q:
                        p = product(p, val(w), max_terms)
                    groups.setdefault(p, []).append(dict(op=i, code=code, q=q,
                                                          angle=angles[i] if code == 0x13 else HALF))
                    stats['phase_ops'] += 1
                elif code == 0x11:
                    state[q[0]] = val(q[0]) ^ {0}
                elif code == 0x21:
                    state[q[1]] = val(q[1]) ^ val(q[0])
                elif code == 0x31:
                    state[q[2]] = val(q[2]) ^ product(val(q[0]), val(q[1]), max_terms)
                elif code == 0x22:
                    state[q[0]], state[q[1]] = val(q[1]), val(q[0])
                elif code == 0x32:
                    a, b = val(q[1]), val(q[2])
                    delta = product(val(q[0]), a ^ b, max_terms)
                    state[q[1]], state[q[2]] = a ^ delta, b ^ delta
                if any(len(p) > max_terms for p in state.values()):
                    raise CapacityExceeded
                stats['supported_ops'] += 1
            except CapacityExceeded:
                # The offending operation stays untouched. After it, arbitrary
                # independent wire values overapproximate every reachable state.
                finish(i, 'capacity'); start = i+1; stats['capacity_cuts'] += 1
    finish(B, 'query_end')
    if include_predicates:
        stats['predicates'] = inventory
    return found, stats


class PhaseFold(R.RewriteClass):
    name = 'phase-fold'
    tasks = ('dqi', 'invsqrt', 'chemistry')
    needs = ('bounded symbolic phase predicates',)
    summary = 'combine equal Boolean phase predicates with exact 128-bit modular angles; barriers fail closed'
    certify = {'invsqrt': 'symbolic all-input identity; fdiff --whole is an independent cross-check',
               'dqi': 'symbolic identity; engine draws and official verifier for promotion',
               'chemistry': 'symbolic identity; full family verifier for promotion'}

    def match(self, ctx):
        groups, stats = scan(ctx.k, ctx.ops, ctx.A, ctx.B)
        ctx.notes.append({'phase_fold_coverage': stats})
        sites = []
        for g in groups:
            price = dict(static=g['toffolis_after'], expected=g['toffolis_after'],
                         saving=g['toffolis_before']-g['toffolis_after'],
                         rotations_delta=g['rotations_after']-g['rotations_before'],
                         general_rotations_delta=g['general_rotations_delta'], t_gates_delta=g['t_gates_delta'],
                         model='exact unconditional counts; rotation charge requires task synthesis margin')
            sites.append(R.make_site(ctx, self.name, (min(g['ops']), max(g['ops'])+1), None,
                                     len(g['predicate']), g['toffolis_before'], price, {},
                                     [R.obligation('phase identity', 'proved',
                                      'equal ANF predicates on arbitrary region-entry wires; angles add modulo 2^128'),
                                      R.obligation('barriers', 'proved',
                                      'no measurement/reset/condition/unsupported gate crossed')], fold=g))
        return sites

    def apply(self, site, ctx, bits):
        g = site['fold']
        return [f"replace {i} => {g['replacement']}" if i == g['anchor'] and g['replacement']
                else f'delete {i}' for i in g['ops']]

    def template_check(self):
        return 'diagonal phases on the same entry-state Boolean predicate add exactly modulo one turn'


CLASS = PhaseFold()
