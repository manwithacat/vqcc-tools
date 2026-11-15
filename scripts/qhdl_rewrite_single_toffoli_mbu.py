"""Rewrite class `single-toffoli-mbu` (`qhdl census` pattern `single`; `qhdl track` mbu 'single').

Match: a lone Toffoli CCX(a, b, t) whose target is provably |0> right after it (so it held a b before): the census
proves this in every shot (symbolic tracker or the structural compute/uncompute rule). Replacement: HMR t -> m;
CZ a b if m. 1 Toffoli -> 0 (a conditioned CZ is Clifford, not charged). One new classical bit shared by all sites.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_rewrite as R  # noqa: E402


class SingleToffoliMbu(R.RewriteClass):
    name = 'single-toffoli-mbu'
    tasks = ('dqi', 'invsqrt', 'chemistry')
    needs = ('census',)
    bits = ('m',)
    certify = {'dqi': 'one engine smoke draw (sweep-0), then the 48-draw sweep and the official verifier',
               'invsqrt': 'fdiff --whole', 'chemistry': 'official verifier'}
    summary = 'lone Toffoli with a provably |0> target -> HMR + CZ if m (0 Toffolis)'

    def match(self, ctx):
        sites = []
        rows = [r for r in ctx.census if r['pattern'] == 'single' and r.get('zero')]
        rec = ctx.ops.records([r['op'] for r in rows])
        for r in rows:
            i = r['op']
            code, q, _ = rec[i]
            if code != R.CCX:
                continue
            a, b, t = q
            ob = [R.obligation('target |0> after', r['zero'],
                               'qhdl census: ' + ('proof in every shot' if r['zero'] == 'proved' else 'lane simulation'))]
            price = dict(static=0, expected=0.0, saving=1.0, model='HMR + CZ if m: the fix-up is Clifford')
            site = R.make_site(ctx, self.name, (i, i + 1), None, 2, 1, price,
                               dict(target=f'q{t}', controls=[f'q{a}', f'q{b}']), ob, pairs_with=r.get('pairs_with'))
            site['conditional'] = bool(r.get('conditional'))
            if site['conditional']:
                site['reason'] = 'runs under a classical condition: the saving applies to the shots that execute it'
            sites.append(site)
        return sites

    def apply(self, site, ctx, bits):
        i = site['ops'][0]
        w = site['wires']
        m = bits['m']
        return [f"replace {i} => HMR {w['target']} b{m}; CZ {' '.join(w['controls'])} if b{m}"]

    def template_check(self):
        return 'HMR t -> m then CZ(a,b) if m cancels (-1)^(m ab) for every m, a, b (checked)'


CLASS = SingleToffoliMbu()
