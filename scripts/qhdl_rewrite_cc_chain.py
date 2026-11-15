"""Rewrite class `cc-chain`: a multi-input AND computed into a clean target by a dirty-borrow chain, replaced by a
construction with conditionally clean ancillae (Khattar & Gidney 2024, "Rise of conditionally clean ancillae for
optimizing quantum circuits"; constructions S12 / S6 designed and checked
exhaustively in an earlier prototype, then ported to busyaprime's cached chains).

Match (semantic): a block whose exact net action is t ^= AND(y1..yn) with every borrowed wire restored (40 Toffolis
for the plain 12-input Barenco chain, 36 with cached products and in-block measurements; 16 / 14 for n = 6), with
n in {12, 6}. Preconditions: the target is |0> at entry (the constructions host intermediate products on it), and k
clean ancillae are free at the site (k = 2 for S12: A, B; k = 1 for S6: B): wires |0> right before the block (reset,
or a proved zero-after site), not inputs or borrowed wires; cache wires the block itself measures come first (in
measurement order), else the most recently erased clean scratch wires.
Cost: S12 19 static / 18 expected (was 40 or 36); S6 8 / 7.5 (was 16 or 14). Both constructions are checked
exhaustively (all 2^n inputs, values and every measurement's phase) before any splice. Other n are priced at 2n - 3
(the conditionally clean ancilla figure) and reported; no template is built for them. Four new classical bits
shared by all sites (every fix-up is applied inside its own site).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_rewrite as R  # noqa: E402


# ---------------------------------------------------------------- the constructions (verbatim from the prototype)
def check(n, anc, target, ops):
    """Exhaustive bit-parallel check over all 2^n inputs (ancillae |0>): (ok, static cost, expected cost, messages)."""
    N = 1 << n
    full = (1 << N) - 1
    val = {}
    for i in range(1, n + 1):
        v = 0
        for x in range(N):
            if (x >> (i - 1)) & 1:
                v |= 1 << x
        val[f'y{i}'] = v
    for a in anc:
        val[a] = 0
    val[target] = 0
    prod = full
    for i in range(1, n + 1):
        prod &= val[f'y{i}']
    hmr_val, fix = {}, {}
    static = expected = 0.0
    msgs = []
    for op in ops:
        k = op[0]
        if k == 'CCX':
            _, a, b, t = op
            val[t] ^= val[a] & val[b]
            static += 1
            expected += 1
        elif k == 'X':
            val[op[1]] ^= full
        elif k == 'HMR':
            _, w, m = op
            assert m not in hmr_val, m
            hmr_val[m] = val[w]
            fix.setdefault(m, 0)
            val[w] = 0
        elif k == 'CZ_IF':
            _, m, a, b = op
            fix[m] = fix.get(m, 0) ^ (val[a] & val[b])
        elif k == 'CCZ_IF':
            _, m, a, b, c = op
            fix[m] = fix.get(m, 0) ^ (val[a] & val[b] & val[c])
            static += 1
            expected += 0.5
        else:
            raise ValueError(op)
    ok = True
    for m, v in hmr_val.items():
        if fix.get(m, 0) != v:
            ok = False
            msgs.append(f'phase fix-up for outcome {m} wrong on {bin(fix.get(m, 0) ^ v).count("1")} inputs')
    for i in range(1, n + 1):
        w = f'y{i}'
        exp = 0
        for x in range(N):
            if (x >> (i - 1)) & 1:
                exp |= 1 << x
        if val[w] != exp:
            ok = False
            msgs.append(f'input {w} not restored')
    for a in anc:
        if val[a] != 0:
            ok = False
            msgs.append(f'ancilla {a} not zero')
    if val[target] != prod:
        ok = False
        msgs.append(f'target wrong on {bin(val[target] ^ prod).count("1")} inputs')
    return ok, static, expected, msgs


def S12():
    """12-input AND into clean F with clean ancillae A, B (19 static, 18 expected)."""
    ops = [
        ('CCX', 'y1', 'y2', 'A'), ('CCX', 'A', 'y3', 'B'),
        ('HMR', 'A', 'mA1'), ('CZ_IF', 'mA1', 'y1', 'y2'),
        ('X', 'y1'), ('CCX', 'y4', 'y5', 'y1'),
        ('X', 'y4'), ('CCX', 'y6', 'y7', 'y4'),
        ('X', 'y2'), ('CCX', 'y1', 'y4', 'y2'),
        ('X', 'y6'), ('CCX', 'y8', 'y9', 'y6'),
        ('X', 'y3'), ('CCX', 'y2', 'y6', 'y3'),
        ('X', 'y8'), ('CCX', 'y10', 'y11', 'y8'),
        ('CCX', 'y3', 'y8', 'F'),
        ('CCX', 'F', 'y12', 'A'),
        ('HMR', 'F', 'mF'), ('CZ_IF', 'mF', 'y3', 'y8'),
        ('CCX', 'B', 'A', 'F'),
        ('HMR', 'A', 'mA2'), ('CCZ_IF', 'mA2', 'y3', 'y8', 'y12'),
        ('CCX', 'y10', 'y11', 'y8'), ('X', 'y8'),
        ('CCX', 'y2', 'y6', 'y3'), ('X', 'y3'),
        ('CCX', 'y8', 'y9', 'y6'), ('X', 'y6'),
        ('CCX', 'y1', 'y4', 'y2'), ('X', 'y2'),
        ('CCX', 'y6', 'y7', 'y4'), ('X', 'y4'),
        ('CCX', 'y4', 'y5', 'y1'), ('X', 'y1'),
        ('HMR', 'B', 'mB'), ('CCZ_IF', 'mB', 'y1', 'y2', 'y3'),
    ]
    return 12, ['A', 'B'], 'F', ops


def S6():
    """6-input AND into clean F with one clean ancilla B (8 static, 7.5 expected)."""
    ops = [
        ('CCX', 'y1', 'y2', 'F'), ('CCX', 'F', 'y3', 'B'),
        ('HMR', 'F', 'mF'), ('CZ_IF', 'mF', 'y1', 'y2'),
        ('X', 'y1'), ('CCX', 'y4', 'y5', 'y1'),
        ('X', 'y2'), ('CCX', 'y1', 'y6', 'y2'),
        ('CCX', 'B', 'y2', 'F'),
        ('CCX', 'y1', 'y6', 'y2'), ('X', 'y2'),
        ('CCX', 'y4', 'y5', 'y1'), ('X', 'y1'),
        ('HMR', 'B', 'mB'), ('CCZ_IF', 'mB', 'y1', 'y2', 'y3'),
    ]
    return 6, ['B'], 'F', ops


SCHEMES = {12: S12, 6: S6}
PRICE = {12: (19, 18.0), 6: (8, 7.5)}


def scheme_lines(ops, wmap, bits):
    """Scheme ops -> kmx-style op texts for a `qhdl edit` replace."""
    out = []
    for op in ops:
        kd = op[0]
        if kd == 'CCX':
            out.append('CCX ' + ' '.join(wmap[x] for x in op[1:]))
        elif kd == 'X':
            out.append(f'X {wmap[op[1]]}')
        elif kd == 'HMR':
            out.append(f'HMR {wmap[op[1]]} b{bits[op[2]]}')
        elif kd == 'CZ_IF':
            out.append('CZ ' + ' '.join(wmap[x] for x in op[2:]) + f' if b{bits[op[1]]}')
        elif kd == 'CCZ_IF':
            out.append('CCZ ' + ' '.join(wmap[x] for x in op[2:]) + f' if b{bits[op[1]]}')
        else:
            raise ValueError(op)
    return out


class CcChain(R.RewriteClass):
    name = 'cc-chain'
    tasks = ('dqi', 'invsqrt', 'chemistry')
    needs = ('census',)
    bits = ('mA1', 'mF', 'mA2', 'mB')
    certify = {'dqi': 'one engine smoke draw (sweep-0), then the 48-draw sweep and the official verifier',
               'invsqrt': 'fdiff --whole', 'chemistry': 'official verifier'}
    summary = 'n-input AND compute chain on a clean target -> conditionally clean construction (S12: 18, S6: 7.5 expected)'

    def match(self, ctx):
        sites = []
        for blk in ctx.blocks:
            if blk['kind'] != 'chain' or blk['n'] < 4:
                continue
            n, t = blk['n'], blk['target']
            lits = blk['controls']
            need = 2 if n == 12 else 1 if n == 6 else max(0, n - 4)
            if blk.get('partner') is not None:
                tz, thow = '', 'uncompute of an earlier identical chain: the target holds the value'
            else:
                tz, thow = ctx.clean_status(t, blk['start'])
            ob = [R.obligation('target |0> at entry', tz if tz else 'open', thow)]
            exclude = {t} | {w for w, _ in lits} | set(blk['borrowed'])
            anc, worst = [], 'proved'
            for w in blk.get('assumed_zero', []):
                st = ctx.clean.status(w, blk['start'])
                if w not in exclude and st and len(anc) < need:
                    anc.append((w, st, 'cache wire measured inside the block'))
            if len(anc) < need:
                for w, st, lw in ctx.clean.clean_scratch(blk['start'], exclude=exclude | {a for a, _, _ in anc},
                                                        span_end=blk['end']):
                    if len(anc) >= need:
                        break
                    anc.append((w, st, f'most recently erased clean scratch wire (last write op {lw})'))
            for _, st, _ in anc:
                if st.startswith('sampled'):
                    worst = st
            ob.append(R.obligation(f'{need} clean ancilla(e)', (worst if len(anc) >= need else 'open') if need else 'n/a',
                                   '; '.join(f'q{w}: {how}' for w, _, how in anc) if anc else
                                   f'{len(anc)} of {need} clean scratch wires free at the site'))
            fits, reason = True, ''
            if blk.get('partner') is not None:
                fits, reason = False, ('uncompute of an earlier identical chain (the target holds the value): a '
                                       'measurement-based uncomputation site (flag-ladder-mbu), not a compute')
            if n in PRICE:
                st_, ex = PRICE[n]
                price = dict(static=st_, expected=ex, saving=round(blk['toffolis'] - ex, 2),
                             model=f'S{n}: conditionally clean ancillae, exhaustive check')
                if not fits:
                    pass
                elif any(neg for _, neg in lits):
                    fits, reason = False, 'negated literal inside the block'
                elif len(anc) < need:
                    fits, reason = False, f'{len(anc)} of {need} clean ancillae free at the site'
            else:
                fits, reason = False, f'n = {n}: no construction built (priced 2n - 3 = {2 * n - 3}, Khattar & Gidney)'
                price = dict(static=2 * n - 3, expected=float(2 * n - 3), saving=round(blk['toffolis'] - (2 * n - 3), 2),
                             model='2n - 3 (not built)')
            roles = ['A', 'B'] if n == 12 else ['B']
            wires = dict(target=f'q{t}', inputs=[f'q{w}' for w, _ in lits],
                         ancillae={r: f'q{w}' for r, (w, _, _) in zip(roles, anc)})
            sites.append(R.make_site(ctx, self.name, (blk['start'], blk['end']), blk, n, blk['toffolis'], price,
                                     wires, ob, fits=fits, reason=reason))
        return sites

    def apply(self, site, ctx, bits):
        a_, b_ = site['ops']
        n = site['n']
        if n not in SCHEMES:
            raise R.RewriteError(f'cc-chain: no construction for n = {n}')
        _, _, _, ops = SCHEMES[n]()
        w = site['wires']
        wmap = {f'y{i + 1}': q for i, q in enumerate(w['inputs'])}
        wmap['F'] = w['target']
        wmap.update(w['ancillae'])
        return [f'replace {a_}:{b_} => ' + '; '.join(scheme_lines(ops, wmap, bits))]

    def template_check(self):
        parts = []
        for n, f in SCHEMES.items():
            nn, anc, tgt, ops = f()
            ok, st, ex, msgs = check(nn, anc, tgt, ops)
            parts.append(f'S{n} {"ok" if ok else "FAILED " + "; ".join(msgs)} ({st:g} static, {ex:g} expected) on all '
                         f'{1 << nn} inputs')
        return '; '.join(parts)


CLASS = CcChain()
