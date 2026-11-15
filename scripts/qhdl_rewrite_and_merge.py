"""Rewrite class `and-merge` (invsqrt): Toffoli-led windows whose live outputs need fewer ANDs than the window spends
(Ehan Mallick's two 3-to-2 AND merges on 67482db6c538 -> bce2948ff24e).

Match: `qhdl windows` (size 4, span 48, max 20 inputs) rows with a constructive saving (charged = mean executed
Toffolis - constructive AND-count upper bound > 0; nested windows folded). Exact on all legal x in two outcome
branches; placement, width and phase are NOT checked by the ranking.
Apply is not automatic: the agent supplies the replacement window (--replacement FILE with a `site <id>` or `ops A:B`
section per site, one op per line, `@i` copies allowed) and the registry splices it with `qhdl edit`; the certificate
is `fdiff --whole` (all legal x, zero + one random branch: phase per x, x restored, scratch clean) and then the
official verifier. The class only states the floor; it does not synthesise the circuit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_rewrite as R  # noqa: E402

SIZE, SPAN, MAX_INPUTS = 4, 48, 20


class AndMerge(R.RewriteClass):
    name = 'and-merge'
    tasks = ('invsqrt',)
    needs = ('windows', 'window', 'fdiff')
    bits = ()
    certify = {'invsqrt': 'fdiff --whole, then the official verifier'}
    summary = 'window with Toffolis above its AND floor -> agent-supplied replacement window, checked by fdiff'

    def match(self, ctx):
        import qhdl_window as W
        api = ctx.api
        try:
            meta, L = W._open_invsqrt(api, ctx.sha12)
        except Exception as exc:  # noqa: BLE001  (DQI-sized or non-invsqrt: no windows here)
            ctx.notes.append(f'and-merge: {exc}')
            return []
        ns = argparse.Namespace(circuit=ctx.sha12, size=SIZE, span=SPAN, top=20, max_inputs=MAX_INPUTS, all=False)
        W.v_windows(api, ns)   # populates (or reuses) the windows cache
        ev = L.dir / 'window' / f'scan_s{SIZE}_p{SPAN}_i{MAX_INPUTS}.json'
        hit = json.loads(ev.read_text())
        sites = []
        for r in hit['rows']:
            if r.get('nested_in') or r.get('charged') is None or r['charged'] <= 0:
                continue
            A, B = (int(x) for x in r['ops'].split(':'))
            if not (A >= ctx.A and B <= ctx.B):
                continue
            ob = [R.obligation('replacement', 'open', 'supply the replacement window with --replacement; checked by fdiff'),
                  R.obligation('placement/width/phase', 'n/a', 'not checked by the ranking; fdiff --whole checks the circuit')]
            price = dict(static=r['ub'], expected=round(r['exec'] - r['charged'], 3), saving=r['charged'],
                         model='constructive AND-count upper bound (min-weight Dickson basis)')
            s = R.make_site(ctx, self.name, (A, B), None, len(r.get('outs', '').split(',')), r['toffoli'], price,
                            dict(outputs=r.get('outs', ''), floor=f"{r['lb']}..{r['ub']} ({r['floor']})"), ob,
                            window=r)
            sites.append(s)
        return sites

    def apply(self, site, ctx, bits):
        rep = site.get('replacement')
        if not rep:
            raise R.RewriteError(f"and-merge {site['id']}: no replacement supplied",
                                 'pass --replacement FILE with a `site <id>` section of op lines')
        A, B = site['ops']
        return [f'replace {A}:{B} => ' + '; '.join(rep)]

    def template_check(self):
        return ''


CLASS = AndMerge()
