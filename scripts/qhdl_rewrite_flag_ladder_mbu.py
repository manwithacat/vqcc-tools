"""Rewrite class `flag-ladder-mbu` (first applied as c253111974fc -> 1779b4e5967f).

Match (semantic): a six-input dirty-borrow uncompute chain (net action t ^= AND(y1..y6), borrowed wires restored;
16 Toffolis as a Barenco chain, 14 with busyaprime's cached products and an in-block measurement) that is the
uncompute of an earlier identical chain.
Preconditions: (1) t is |0> right after the block (held AND(y) before): census proof, structural rule or the DQI lane
simulation (sampled: in the known circuits the controls are conditionally swapped between compute and uncompute, so
value-identity proofs fail); (2) one clean helper wire s at the site: a wire |0> right before the block (last
value-changing op an unconditional reset, or a proved zero-after site) that is not an input, the target or a borrowed
wire; a cache wire the block itself measures (assumed |0> at entry) is the first choice, else the most recently erased
clean scratch wire (on c253111974fc this reuses the wire the `and3-dirty-mbu` rewrite had just freed).
Replacement (a two-clean-wire ladder, from an earlier prototype):
    HMR t -> m
    if m:  CCX(y1,y2 -> t); CCX(t,y3 -> s); HMR t -> m3; CZ(y1,y2) if m3; CCX(s,y4 -> t); CCZ(t,y5,y6);
           HMR t -> m4; CZ(s,y4) if m4; HMR s -> m5; CCZ(y1,y2,y3) if m5
The phase applied when m = 1 is (-1)^(y1..y6), cancelling the HMR by-product; checked exhaustively on all 64 inputs
with both ancillae |0> before any splice. 16 (14) Toffolis -> 5 static, 2.25 expected. Four new bits shared by all
sites. Only n = 6 has a template here; other n are priced (census `mbu` model) and reported.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_census as C  # noqa: E402
import qhdl_rewrite as R  # noqa: E402


def ladder_phase(y):
    """Values and phase fix-ups of the conditional ladder on inputs y (t and s start |0>): (fix-ups ok, phase)."""
    y1, y2, y3, y4, y5, y6 = y
    q = s = 0
    ph = 0
    ok = True
    q ^= y1 & y2
    s ^= q & y3
    for m3 in (0, 1):
        ok &= ((m3 * q) ^ (m3 * y1 * y2)) == 0
    q = 0
    q ^= s & y4
    ph ^= q & y5 & y6
    for m4 in (0, 1):
        ok &= ((m4 * q) ^ (m4 * s * y4)) == 0
    q = 0
    for m5 in (0, 1):
        ok &= ((m5 * s) ^ (m5 * y1 * y2 * y3)) == 0
    return ok, ph


def check_ladder():
    for y in itertools.product((0, 1), repeat=6):
        ok, ph = ladder_phase(y)
        if not ok or ph != (y[0] & y[1] & y[2] & y[3] & y[4] & y[5]):
            return False
    return True


class FlagLadderMbu(R.RewriteClass):
    name = 'flag-ladder-mbu'
    tasks = ('dqi', 'invsqrt', 'chemistry')
    needs = ('census', 'simulate')
    bits = ('m', 'm3', 'm4', 'm5')
    certify = {'dqi': 'one engine smoke draw (sweep-0), then the 48-draw sweep and the official verifier',
               'invsqrt': 'fdiff --whole', 'chemistry': 'official verifier'}
    summary = '6-input flag uncompute chain (16/14 Toffolis) -> HMR + conditional two-wire ladder (2.25 expected)'

    def match(self, ctx):
        sites = []
        for blk in ctx.blocks:
            if blk['kind'] != 'chain' or blk['n'] < 4 or blk.get('partner') is None:
                continue
            n, t = blk['n'], blk['target']
            lits = blk['controls']
            z, how = ctx.zero_after(blk)
            ob = [R.obligation('target |0> after', z if z else 'open', how)]
            exclude = {t} | {w for w, _ in lits} | set(blk['borrowed'])
            helper, hstatus, hhow = None, '', ''
            for w in blk.get('assumed_zero', []):
                st = ctx.clean.status(w, blk['start'])
                if w not in exclude and st:
                    helper, hstatus, hhow = w, st, 'cache wire measured inside the block, |0> at entry'
                    break
            if helper is None:
                cands = ctx.clean.clean_scratch(blk['start'], exclude=exclude, span_end=blk['end'])
                if cands:
                    helper, hstatus, lw = cands[0]
                    hhow = f'most recently erased clean scratch wire (last write op {lw})'
            ob.append(R.obligation('one clean helper wire', hstatus if helper is not None else 'open',
                                   hhow if helper is not None else 'no clean scratch wire free at the site'))
            opts, _ = C.price(n, blk['toffolis'], True, 1 if helper is not None else 0)
            fits, reason = True, ''
            if n != 6:
                fits, reason = False, f'n = {n}: the ladder template is for n = 6 only (priced by the census model)'
                price = dict(static=None, expected=opts['mbu'][0], saving=round(blk['toffolis'] - opts['mbu'][0], 2),
                             model='census mbu price')
            else:
                price = dict(static=5, expected=2.25, saving=round(blk['toffolis'] - 2.25, 2),
                             model='HMR + conditional two-wire ladder (5 Toffoli-class ops, 2.25 expected)')
                if any(neg for _, neg in lits):
                    fits, reason = False, 'negated literal inside the block'
                elif helper is None:
                    fits, reason = False, 'no clean helper wire at the site'
            wires = dict(target=f'q{t}', inputs=[f'q{w}' for w, _ in lits],
                         helper=f'q{helper}' if helper is not None else None)
            sites.append(R.make_site(ctx, self.name, (blk['start'], blk['end']), blk, n, blk['toffolis'], price,
                                     wires, ob, fits=fits, reason=reason, pairs_with=blk['partner']['start']))
        return sites

    def apply(self, site, ctx, bits):
        a_, b_ = site['ops']
        w = site['wires']
        y1, y2, y3, y4, y5, y6 = w['inputs']
        Q, S = w['target'], w['helper']
        m, m3, m4, m5 = (f"b{bits[n]}" for n in ('m', 'm3', 'm4', 'm5'))
        ops = [f'HMR {Q} {m}', f'PUSH_CONDITION if {m}', f'CCX {y1} {y2} {Q}', f'CCX {Q} {y3} {S}', f'HMR {Q} {m3}',
               f'CZ {y1} {y2} if {m3}', f'CCX {S} {y4} {Q}', f'CCZ {Q} {y5} {y6}', f'HMR {Q} {m4}',
               f'CZ {S} {y4} if {m4}', f'HMR {S} {m5}', f'CCZ {y1} {y2} {y3} if {m5}', 'POP_CONDITION']
        return [f'replace {a_}:{b_} => ' + '; '.join(ops)]

    def template_check(self):
        return 'ladder checked exhaustively on all 64 inputs (values restored, phase (-1)^(y1..y6))' if check_ladder() \
            else 'LADDER CHECK FAILED'


CLASS = FlagLadderMbu()
