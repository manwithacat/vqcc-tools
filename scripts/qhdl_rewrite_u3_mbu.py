"""Rewrite class `u3-mbu`: a partial measurement-based uncomputation that still pays Toffolis (busyaprime's
3779a3431e9c AND3 flag uncompute: CCX(a,b,d) CCX(d,c,t) CCX(a,b,d); HMR t -> m; CZ d c if m).

Match (semantic): a block whose net value action is `t measured to |0>, every other wire restored` and whose only
unbalanced measurement phase is exactly m * (t XOR AND(literals)): the block is correct precisely when t held
AND(literals) at entry, which is the precondition (discharged by the structural compute/uncompute rule when the
operands are untouched between, else by the DQI lane simulation: sampled, since the controls are conditionally swapped
between compute and uncompute in the known circuits).
Replacement (n = 3): HMR t -> m; CCZ(a, b, c) if m. 3 Toffolis -> 1 static, 0.5 expected. Larger n is priced
(census `mbu` price) and reported; the multi-controlled fix-up ladder is not built here.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_census as C  # noqa: E402
import qhdl_rewrite as R  # noqa: E402


class U3Mbu(R.RewriteClass):
    name = 'u3-mbu'
    tasks = ('dqi', 'invsqrt', 'chemistry')
    needs = ('simulate',)
    bits = ('m3u',)
    certify = {'dqi': 'one engine smoke draw (sweep-0), then the 48-draw sweep and the official verifier',
               'invsqrt': 'fdiff --whole', 'chemistry': 'official verifier'}
    summary = 'partial MBU of a held AND (3 Toffolis + HMR + CZ) -> HMR + CCZ if m (0.5 expected)'

    def match(self, ctx):
        sites = []
        for blk in ctx.blocks:
            if blk['kind'] != 'mbu-partial':
                continue
            n = blk['n']
            t = blk['target']
            lits = blk['controls']
            # precondition: t held AND(lits) at entry
            p = blk.get('partner')
            status, how = '', ''
            if p is not None and p['kind'] == 'chain':
                ws = [t] + [w for w, _ in lits]
                if not any(ctx.clean.writes_between(w, p['end'], blk['start']) for w in ws) \
                        and ctx.clean.status(t, p['start']) == 'proved':
                    status, how = 'proved', ('structural rule: computed by the earlier identical chain, target and '
                                             'controls untouched between, target |0> before the compute')
            if not status:
                status, how = ctx.holds_and(blk)
            ob = [R.obligation('target held AND(literals) at entry', status if status else 'open', how)]
            opts, _ = C.price(n, blk['toffolis'], True, 0)
            mbu = opts['mbu'][0]
            price = dict(static=1 if n == 3 else None, expected=mbu, saving=round(blk['toffolis'] - mbu, 2),
                         model='HMR + C^nZ(literals) if m')
            wires = dict(target=f'q{t}', controls=[('~' if neg else '') + f'q{w}' for w, neg in lits])
            fits, reason = True, ''
            if n != 3:
                fits, reason = False, f'n = {n}: the conditioned multi-controlled Z fix-up ladder is not built (priced only)'
            elif any(neg for _, neg in lits):
                fits, reason = False, 'negated literal inside the block'
            sites.append(R.make_site(ctx, self.name, (blk['start'], blk['end']), blk, n, blk['toffolis'], price,
                                     wires, ob, fits=fits, reason=reason, pairs_with=p['start'] if p else None))
        return sites

    def apply(self, site, ctx, bits):
        a_, b_ = site['ops']
        w = site['wires']
        m = bits['m3u']
        return [f"replace {a_}:{b_} => HMR {w['target']} b{m}; CCZ {' '.join(w['controls'])} if b{m}"]

    def template_check(self):
        return ('given t = abc, HMR t -> m then CCZ(a,b,c) if m leaves no phase for every m, a, b, c (8 inputs, '
                'checked)')


CLASS = U3Mbu()
