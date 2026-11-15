"""Rewrite class `and3-dirty-mbu` (first applied to yoyobar 6d58bd6d0232 -> c253111974fc).

Match (semantic): a block whose exact net action is t ^= a b c with one borrowed dirty wire restored, made of exactly
four consecutive Toffolis CCX(a,b,d) CCX(d,c,t) CCX(a,b,d) CCX(d,c,t) (the "toggle twice" dirty AND3), that is the
uncompute of an earlier identical block (same target and literals).
Precondition: t is |0> right after the block (so it held a b c before it): `qhdl census` proof (every shot), the
structural compute/uncompute rule, or the DQI lane simulation (sampled).
Replacement: HMR t -> m; CCZ a b c if m.  The X-basis measurement leaves the phase (-1)^(m * abc), cancelled by the
conditioned CCZ. 4 Toffolis -> 1 static, 0.5 expected (the fix-up runs when m = 1). One new classical bit shared by
all sites. On 6d58 this reproduces c253111974fc byte for byte through `qhdl edit`.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_rewrite as R  # noqa: E402


class And3DirtyMbu(R.RewriteClass):
    name = 'and3-dirty-mbu'
    tasks = ('dqi', 'invsqrt', 'chemistry')
    needs = ('census',)
    bits = ('m',)
    certify = {'dqi': 'one engine smoke draw (sweep-0), then the 48-draw sweep and the official verifier',
               'invsqrt': 'fdiff --whole', 'chemistry': 'official verifier'}
    summary = 'dirty-borrow AND3 uncompute (4 Toffolis) -> HMR + CCZ if m (0.5 expected)'

    @staticmethod
    def core_shape(ctx, block):
        """(core span, a, b, c, d, t) when the block's Toffolis are the 4-op toggle-twice shape, else (None, reason)."""
        rec = ctx.ops.records(range(block['start'], block['end']))
        ccx = [(i, r) for i, r in sorted(rec.items()) if r[0] == R.CCX]
        if len(ccx) != 4:
            return None, f'{len(ccx)} Toffolis in the block, not 4'
        idx = [i for i, _ in ccx]
        if idx[3] - idx[0] != 3:
            return None, 'the four Toffolis are not consecutive'
        q = [r[1] for _, r in ccx]
        if q[0] != q[2] or q[1] != q[3]:
            return None, 'not the toggle-twice shape CCX(a,b,d) CCX(d,c,t) CCX(a,b,d) CCX(d,c,t)'
        a, b, d = q[0]
        d2, c, t = q[1]
        if d2 != d:
            d2, c = c, d2
        if d2 != d or t != block['target'] or len({a, b, c, d, t}) != 5:
            return None, 'the second Toffoli does not read the first one\'s target'
        return (idx[0], idx[3] + 1), (a, b, c, d, t)

    def match(self, ctx):
        sites = []
        for blk in ctx.blocks:
            if blk['kind'] != 'chain' or blk['n'] != 3 or blk['toffolis'] != 4 or blk.get('hmr_wires'):
                continue
            if blk.get('partner') is None:
                continue   # a compute (first of its pair), not an uncompute site
            core, info = self.core_shape(ctx, blk)
            if core is None:
                sites.append(R.make_site(ctx, self.name, (blk['start'], blk['end']), blk, 3, blk['toffolis'],
                                         dict(static=None, expected=None, saving=None), {}, [], fits=False,
                                         reason=info))
                continue
            a, b, c, d, t = info
            if any(neg for _, neg in blk['controls']):
                sites.append(R.make_site(ctx, self.name, core, blk, 3, 4, dict(static=None, expected=None, saving=None),
                                         dict(target=f'q{t}'), [], fits=False,
                                         reason='negated literal inside the block (X between the Toffolis)'))
                continue
            z, how = ctx.zero_after(blk)
            ob = [R.obligation('target |0> after', z if z else 'open', how)]
            price = dict(static=1, expected=0.5, saving=3.5, model='HMR + CCZ if m: the fix-up runs in half the shots')
            sites.append(R.make_site(ctx, self.name, core, blk, 3, 4, price,
                                     dict(target=f'q{t}', controls=[f'q{a}', f'q{b}', f'q{c}'], dirty=f'q{d}'), ob,
                                     pairs_with=blk['partner']['start']))
        return sites

    def apply(self, site, ctx, bits):
        a_, b_ = site['ops']
        w = site['wires']
        m = bits['m']
        return [f"replace {a_}:{b_} => HMR {w['target']} b{m}; CCZ {' '.join(w['controls'])} if b{m}"]

    def template_check(self):
        return 'HMR t -> m then CCZ(a,b,c) if m cancels (-1)^(m abc) for every m, a, b, c (8 inputs, checked)'


CLASS = And3DirtyMbu()
