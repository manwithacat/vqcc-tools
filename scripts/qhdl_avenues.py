#!/usr/bin/env python3
"""qhdl plugin: `qhdl avenues` -- a ranked list of places worth investigating in a circuit (guide:
docs/qhdl-avenues.md). Invsqrt here (enumerable domain); chemistry loaders and DQI in qhdl_avenues_chem/_dqi.

    qhdl avenues -c <sha> [--top N] [--task invsqrt] [--json] [--out DIR] [--refine K]

Generators (each emits candidate components with evidence; avenues are leads, not candidates, nothing is emitted):

  G1 cost attribution   Executed mean charged cost per op (Toffoli-like gates that run, mean over random measurement
                        outcomes, plus Z_POW rotations priced with the recorded verifier eps), summed per segment
                        (maximal unconditioned X/CX/CCX/SWAP "value runs", and the glue between them). The sum is
                        compared with the verifier's charge per qubit (cost = Q * charge); the verb reports the
                        attribution error and refuses to rank (KILL) when it exceeds 10%. Needs `qhdl exec -c` first
                        (its per-op counts are reused from work/<sha>/exec/exec_seed1.json).
  G2 floors             Sliding windows over each value run (Toffoli-count sizes 2..16, then a few larger ones). Exact
                        arbitrary-entry ANF by symbolic propagation (any input count; a per-wire term cap reports
                        'term_cap' instead of guessing), then the lower bounds of qhdl_window.floor: degree - 1,
                        Dickson rank/2 over output combinations (<= 10 nonlinear outputs), and the GF(2) rank of the
                        degree >= 2 output residuals. Windows start at a Toffoli and end after
                        one; scratch wires untouched before the window are |0> (not inputs). Outputs not live after
                        the window are ignored. Fast pass = degree + nonlinear rank; the --refine best candidates are
                        re-floored with the full qhdl_window.floor. The cubic-derivative and polar bounds are NOT
                        used in the fast pass, so a floor can be weaker than qhdl_window's and the gap larger: gap is an UPPER bound
                        on the possible saving, never a promise.
  G3 repetition         Windows are hashed up to wire relabelling (wires renamed by first use; op codes kept); the
                        multiplicity of a component is the number of non-overlapping windows with the same hash.
                        Affine renaming is not detected.
  pricing               gap * multiplicity Toffolis -> invsqrt cost Q * (charge - S) -> GM change = (1/3) ln ratio.
  negatives             Rows of an optional negative-results ledger CSV (docs/negative-results/ledger.csv; columns
                        id, task, outcome, mechanism, keywords, ...; none shipped, so no rows) (task invsqrt/all) matched by op numbers inside the
                        component or by window-search keywords with a stated window size >= the component's Toffolis;
                        matched rows sit below a divider (closed), not removed.

G4 (function matching) and G5 (contract slack) are not implemented here: those fields read 'unavailable'. Tier: static model / projection (floors are theorems on exact functions, but the price is arithmetic).
Standard library plus NumPy (via qhdl_window, qhdl_exec); no other dependencies. Written by an AI model (Claude).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_bitsim as B  # noqa: E402
import qhdl_envelope as E  # noqa: E402

VALUE = (B.C_X, B.C_CX, B.C_CCX, B.C_SWAP)
TERM_CAP = 6000           # terms per wire polynomial before a window is reported as term_cap
PRODUCT_CAP = 600_000       # |a| * |b| of one product
SIZES_SMALL = tuple(range(2, 17))
SIZES_LARGE = (20, 24, 28, 32, 40, 48, 64)
MAX_VARS = 62
KILL_TOLERANCE = 0.10
LEDGER = HERE.parent / 'docs/negative-results/ledger.csv'


# ---------------------------------------------------------------- pricing
def gm_effect_pct(saving_toffolis, charge):
    """Three-task GM change (percent) when `saving_toffolis` executed Toffoli-equivalents vanish from an invsqrt circuit
    of per-qubit charge `charge` (cost = Q * charge, so Q cancels): (1/3) ln(charge / (charge - S))."""
    if saving_toffolis <= 0:
        return 0.0
    if saving_toffolis >= charge:
        return float('inf')
    return 100.0 * math.log(charge / (charge - saving_toffolis)) / 3.0


def cost_after(q, charge, saving):
    return q * (charge - saving)


# ---------------------------------------------------------------- G1: attribution
def rotation_cost(angle, eps, X):
    """Toffoli-equivalents of one executed Z_POW of the given angle (verifier classes)."""
    cls = X.zpow_class(angle)
    if cls == 'clifford':
        return 0.0
    if cls == 't':
        return 1.0 / X.T_PER_TOFFOLI
    return X.rotation_t_count(eps) / X.T_PER_TOFFOLI


def value_runs(ops):
    """Maximal ranges [a, b) of unconditioned X/CX/CCX/SWAP ops."""
    runs, s = [], None
    for i, o in enumerate(ops):
        ok = o.code in VALUE and o.cond < 0
        if ok and s is None:
            s = i
        if not ok and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(ops)))
    return runs


def op_costs(ops, per_op_rows, zpow_rows, rows, eps, X):
    """Executed mean charged cost per op (Toffoli-equivalents). per_op_rows: {op: rows executed}; zpow_rows likewise."""
    cost = [0.0] * len(ops)
    for i, n in per_op_rows.items():
        cost[int(i)] += n / rows
    for i, n in zpow_rows.items():
        i = int(i)
        cost[i] += (n / rows) * rotation_cost(ops[i].angle, eps, X)
    return cost


def segments(ops, cost):
    """Partition 0..n into value runs (kind 'run') and glue between them (kind 'glue'); charged = sum of op costs."""
    out, pos = [], 0
    for a, b in value_runs(ops):
        if a > pos:
            out.append(dict(kind='glue', ops=[pos, a], charged=sum(cost[pos:a])))
        out.append(dict(kind='run', ops=[a, b], charged=sum(cost[a:b])))
        pos = b
    if pos < len(ops):
        out.append(dict(kind='glue', ops=[pos, len(ops)], charged=sum(cost[pos:])))
    return out


# ---------------------------------------------------------------- G2: symbolic arbitrary-entry ANF
class TermCap(Exception):
    pass


def pmul(a, b):
    if not a or not b:
        return set()
    if len(a) * len(b) > PRODUCT_CAP:
        raise TermCap
    r = set()
    for m in a:
        for n in b:
            k = m | n
            if k in r:
                r.remove(k)
            else:
                r.add(k)
    return r


class SymState:
    """Polynomials (sets of monomial bitmasks over input variables) of the wires of a value-gate window. A wire read
    before being written becomes a variable, unless it is a fresh scratch wire (|0>)."""

    def __init__(self, lv, A, xq):
        self.lv, self.A = lv, A
        self.xs = {f'q{w}' for w in xq}
        self.state, self.names, self.written = {}, [], []

    def get(self, w):
        nm = f'q{w}'
        p = self.state.get(w)
        if p is None:
            if nm not in self.xs and self.lv.first_touch(nm) >= self.A:
                p = set()
            else:
                if len(self.names) >= MAX_VARS:
                    raise TermCap
                p = {1 << len(self.names)}
                self.names.append(nm)
            self.state[w] = p
        return p

    def write(self, w, p):
        if len(p) > TERM_CAP:
            raise TermCap
        self.state[w] = p
        if w not in self.written:
            self.written.append(w)

    def apply(self, o):
        c, q = o.code, o.q
        if c == B.C_X:
            p = self.get(q[0])
            self.write(q[0], p ^ {0})
        elif c == B.C_CX:
            a, t = self.get(q[0]), self.get(q[1])
            self.write(q[1], t ^ a)
        elif c == B.C_CCX:
            a, b, t = self.get(q[0]), self.get(q[1]), self.get(q[2])
            self.write(q[2], t ^ pmul(a, b))
        elif c == B.C_SWAP:
            a, b = self.get(q[0]), self.get(q[1])
            self.write(q[0], b)
            self.write(q[1], a)
        else:
            raise ValueError('not a value gate')

    def outputs(self, Bend):
        """{name: monos} of written wires that are live after Bend and not simply restored/identity."""
        outs = {}
        for w in self.written:
            nm = f'q{w}'
            p = self.state[w]
            if not self.lv.live_after(nm, Bend):
                continue
            outs[nm] = sorted(p)
        return outs


def cheap_floor(outs):
    """Fast sound lower bound on the CCX count: max(deg - 1 over outputs, GF(2) rank of the degree >= 2 residuals)."""
    import qhdl_window as W
    lb_deg = 0
    for monos in outs.values():
        d = max((bin(m).count('1') for m in monos), default=-1)
        if d >= 2:
            lb_deg = max(lb_deg, d - 1)
    nlr = W.nonlinear_rank(list(outs.values()))
    return max(lb_deg, nlr), dict(degree=lb_deg, nonlinear_rank=nlr)


def full_floor(outs, nvars):
    """qhdl_window.floor on the exact arbitrary-entry ANFs (adds the Dickson rank over output combinations)."""
    import qhdl_window as W
    o = {}
    for nm, monos in outs.items():
        d = max((bin(m).count('1') for m in monos), default=-1)
        kind = {-1: 'zero', 0: 'constant', 1: 'affine', 2: 'quadratic'}.get(d, f'degree {d}')
        o[nm] = dict(monos=list(monos), deg=d, exact=True, live=True, kind=kind)
        if d == 2:
            o[nm]['rank'] = W.quad_info(monos, max(nvars, 1))[1]
    fl = W.floor(o, max(nvars, 1), 'exact', nl_rank_ok=True)
    return fl['lb'], dict(lb=fl['lb'], ub=fl['ub'], kind=fl['kind'], nonlinear_rank=fl.get('floor_nonlinear_rank'))


def window_floor(ops, lv, xq, A, Bend, refine=True):
    """Exact arbitrary-entry floor of one value-gate window (library entry, used by tests and the controls)."""
    st = SymState(lv, A, xq)
    try:
        for i in range(A, Bend):
            st.apply(ops[i])
    except TermCap:
        return dict(status='term_cap', lb=None)
    outs = st.outputs(Bend)
    lb, info = cheap_floor(outs)
    if refine:
        lb2, info2 = full_floor(outs, len(st.names))
        lb, info = max(lb, lb2), dict(info, **info2)
    return dict(status='exact_anf', lb=lb, vars=len(st.names), info=info,
                toffoli=sum(1 for i in range(A, Bend) if ops[i].code == B.C_CCX))


def scan_windows(ops, lv, xq, runs, sizes_small=SIZES_SMALL, sizes_large=SIZES_LARGE, time_budget=None):
    """Sliding windows over value runs: start at each Toffoli, end after the s-th Toffoli. Incremental symbolic state per
    start. -> list of dict(ops, toffoli, vars, lb, floor_parts) and a list of term_cap records."""
    wanted = set(sizes_small) | set(sizes_large)
    maxsize = max(wanted)
    found, capped = [], []
    t0 = time.time()
    for rid, (ra, rb) in enumerate(runs):
        tof = [i for i in range(ra, rb) if ops[i].code == B.C_CCX]
        for s, A in enumerate(tof):
            if time_budget and time.time() - t0 > time_budget:
                return found, capped, True
            st = SymState(lv, A, xq)
            cnt = 0
            try:
                for i in range(A, rb):
                    o = ops[i]
                    st.apply(o)
                    if o.code == B.C_CCX:
                        cnt += 1
                        if cnt in wanted:
                            outs = st.outputs(i + 1)
                            lb, parts = cheap_floor(outs)
                            found.append(dict(run=rid, ops=[A, i + 1], toffoli=cnt, vars=len(st.names), lb=lb,
                                              floor_parts=parts))
                        if cnt >= maxsize:
                            break
            except TermCap:
                capped.append(dict(run=rid, ops=[A, i + 1], toffoli=cnt + 1))
    return found, capped, False


# ---------------------------------------------------------------- G3: repetition up to wire relabelling
def canon_hash(ops, A, Bend, full=False):
    """Hash of ops[A:Bend] with wires renamed by first use. full=True also keys the rotation angle and whether the op is
    conditioned (glue blocks: Z_POW, HMR, R, ...); value windows have neither."""
    ren, seq = {}, []
    for i in range(A, Bend):
        o = ops[i]
        seq.append(o.code)
        if full:
            seq.append(int(o.angle) if getattr(o, 'angle', None) is not None and o.code == B.C_ZPOW else 0)
            seq.append(1 if o.cond >= 0 else 0)
        for w in o.q:
            if w not in ren:
                ren[w] = len(ren)
            seq.append(ren[w])
        seq.append(-1)
    return hashlib.sha1(json.dumps(seq).encode()).hexdigest()[:12]


def attach_multiplicity(ops, rows):
    """Each row gets sig (canonical hash) and multiplicity (non-overlapping windows with the same hash) and sites."""
    by = {}
    for r in rows:
        r['sig'] = canon_hash(ops, *r['ops'])
        by.setdefault(r['sig'], []).append(tuple(r['ops']))
    sites = {}
    for sig, ivs in by.items():
        ivs = sorted(set(ivs))
        kept, last = [], -1
        for a, b in sorted(ivs, key=lambda t: (t[1], t[0])):  # earliest-end greedy
            if a >= last:
                kept.append((a, b))
                last = b
        sites[sig] = sorted(kept)
    for r in rows:
        r['multiplicity'] = max(1, len(sites[r['sig']]))
        r['sites'] = [list(s) for s in sites[r['sig']][:12]]
    return rows


# ---------------------------------------------------------------- negatives
_NUM = re.compile(r'(?:\bops?\s*@?\s*|\[|@|\bcut\s*|\bop\s+)(\d{3,6})(?!\d)(?![.,]\d)', re.I)  # numbers written as op indices


def load_ledger(path=LEDGER, task='invsqrt'):
    rows = []
    try:
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if any(t in (task, 'all') for t in r['task'].split(';')):
                    rows.append(r)
    except OSError:
        pass
    return rows


def _window_size_words(text):
    """Largest 'N Toffolis' window size a ledger row states (window-search rows), 0 if none."""
    best = 0
    for m in re.finditer(r'(\d+)(?:\s*(?:and|to|\.\.|-)\s*(\d+))?\s*(?:-)?\s*Toffolis?(?:\s+per\s+window|\s+windows)?', text):
        for g in m.groups():
            if g and g.isdigit() and int(g) <= 16:
                best = max(best, int(g))
    return best


WINDOW_KEYS = ('window', 'resynthesis', 'and count', 'and-count', 'rank floor', 'peephole', 'dickson', 'quadratic-form',
               'and-merge')


def match_negatives(ledger, A, Bend, toffoli):
    """-> list of dict(id, outcome, covers, why). covers=True (the row CLOSES this component) only when the row's text
    names an op index inside [A, Bend) (written as 'op N', 'ops N', '[N', '@N' or 'cut N') and the outcome is a negative.
    Window-search rows whose stated window size is >= the component's Toffolis are listed as related (covers False):
    they searched the class, not this component. Stage/component names are not matched."""
    out = []
    for r in ledger:
        text = ' '.join(r.get(k, '') for k in ('mechanism', 'family_target', 'key_numbers', 'why_it_failed', 'keywords'))
        low = text.lower()
        hit_ops = [int(n) for n in _NUM.findall(text) if A <= int(n) < Bend]
        win = any(k in low for k in WINDOW_KEYS)
        size = _window_size_words(text) if win else 0
        if hit_ops:
            out.append(dict(id=r['id'], outcome=r['outcome'], covers=r['outcome'] in ('kill', 'null', 'negative'),
                            why=f'names op {hit_ops[0]} in this range'))
        elif win and size >= toffoli:
            out.append(dict(id=r['id'], outcome=r['outcome'], covers=False,
                            why=f'related: window search up to {size} Toffolis (not this range)'))
    return out


# ---------------------------------------------------------------- assembling the avenues
def first_test(sha12, A, Bend, toffoli, lb):
    cmd = f'python3 scripts/qhdl.py window -c {sha12} --ops {A}:{Bend}'
    return dict(command=cmd,
                stop=f'stop (close the lead) if the window floor reaches {toffoli} (no saving possible) or no equivalent '
                     f'map with fewer than {toffoli} CCX is found within the preregistered search budget',
                rung='verification ladder: local prescreen (qhdl fdiff on the window) -> BitSim zero/7/101 and 8 native '
                     'salts (invsqrt_eval.py seeds --salts 8) -> local official verifier')


def build_rows(ops, lv, xq, found, charge, q, ledger, sha12, refine=None):
    rows = [dict(r) for r in found]
    attach_multiplicity(ops, rows)
    for r in rows:
        r['gap'] = max(0, r['toffoli'] - r['lb'])
        r['floor_source'] = 'fast (degree, nonlinear rank)'
    # refine the best candidates with the full floor, then re-sort
    def key(r):
        return -gm_effect_pct(r['gap'] * r['multiplicity'], charge)
    todo = [r for r in rows if r['gap'] > 0]
    todo.sort(key=lambda r: (key(r), r['toffoli'], r['ops'][0]))
    for r in (todo if refine is None else todo[:refine]):
        res = window_floor(ops, lv, xq, r['ops'][0], r['ops'][1], refine=True)
        if res['status'] == 'exact_anf':
            r['lb'] = max(r['lb'], res['lb'])
            r['floor_parts'] = res['info']
            r['floor_source'] = 'full (degree, nonlinear rank, Dickson, cubic derivative rank, residual-span polar rank)'
        else:
            r['floor_source'] = 'term_cap in refinement; fast floor kept'
        r['gap'] = max(0, r['toffoli'] - r['lb'])
    for r in rows:
        S = r['gap'] * r['multiplicity']
        r['saving_toffolis'] = S
        r['gm_effect_pct'] = round(gm_effect_pct(S, charge), 5)
        r['cost_after'] = round(cost_after(q, charge, S), 2)
        r['negatives'] = match_negatives(ledger, r['ops'][0], r['ops'][1], r['toffoli'])
        r['closed'] = any(n['covers'] for n in r['negatives'])
        r['first_test'] = first_test(sha12, r['ops'][0], r['ops'][1], r['toffoli'], r['lb'])
    return rows


RANK_MAX_TOFFOLI = 16     # avenues are ranked over windows of at most this many Toffolis (larger ones: broad list only)


def regions(wins):
    """Merge overlapping windows (transitively) into maximal clusters. One row per cluster: its best window (open ones
    first, by gm_effect) with the cluster span and window count; the gap is the best window's, never a sum."""
    sk = lambda r: (-r['gm_effect_pct'], r['toffoli'], r['ops'][0])
    out, cur = [], None
    for r in sorted(wins, key=lambda r: tuple(r['ops'])):
        if cur is not None and r['ops'][0] < cur['span'][1]:
            cur['span'][1] = max(cur['span'][1], r['ops'][1])
            cur['wins'].append(r)
        else:
            cur = dict(span=list(r['ops']), wins=[r])
            out.append(cur)
    rows = []
    for c in out:
        openw = [r for r in c['wins'] if not r['closed']]
        best = dict(sorted(openw or c['wins'], key=sk)[0])
        best.update(region=c['span'], cluster_windows=len(c['wins']), closed=not openw,
                    cluster_best_gaps=sorted({r['gap'] for r in c['wins']}, reverse=True)[:4])
        rows.append(best)
    rows.sort(key=sk)
    return rows


def repeated(rows, top=15):
    """Components that repeat (same canonical hash, >= 2 non-overlapping sites), even at gap 0: ranked by Toffolis they
    account for (toffoli x multiplicity); sliding sub-windows of one repeated block are folded into the largest."""
    cand = sorted((r for r in rows if r['multiplicity'] >= 2 and r['toffoli'] >= 2),
                  key=lambda r: (-r['toffoli'] * r['multiplicity'], r['ops'][0]))
    kept = []
    for r in cand:
        sites = [tuple(x) for x in r['sites']]
        if any(any(min(b, kb) - max(a, ka) >= 0.5 * (b - a) for (a, b) in sites for (ka, kb) in map(tuple, k['sites']))
               for k in kept):
            continue
        kept.append(r)
        if len(kept) >= top:
            break
    return kept


def rank(rows):
    """-> (open regions, closed regions, broad windows, gap-0 count, positive windows, repeated components).
    Positive windows (gap > 0, <= RANK_MAX_TOFFOLI Toffolis) are merged into regions (see regions()); a region is closed
    only when every window in it is closed by a ledger row naming the range. Gap-0 windows are counted, not listed."""
    pos = [r for r in rows if r['gap'] > 0 and r['toffoli'] <= RANK_MAX_TOFFOLI]
    reg = regions(pos)
    broad = sorted([r for r in rows if r['gap'] > 0 and r['toffoli'] > RANK_MAX_TOFFOLI],
                   key=lambda r: (-r['gm_effect_pct'], r['toffoli'], r['ops'][0]))
    return ([r for r in reg if not r['closed']], [r for r in reg if r['closed']], broad,
            sum(1 for r in rows if r['gap'] == 0), pos, repeated(rows))


def public_row(r, charge):
    return dict(component=(f"region {r['region'][0]}:{r['region'][1]} ({r['cluster_windows']} windows); best window "
                           f"{r['ops'][0]}:{r['ops'][1]} sig {r['sig']}" if 'region' in r else
                           f"ops {r['ops'][0]}:{r['ops'][1]} run{r['run']} sig {r['sig']}"),
                region_gaps=r.get('cluster_best_gaps'),
                charged_cost=float(r['toffoli']), floor=r['lb'], floor_source=r['floor_source'],
                best_known='unavailable (G4)', gap=r['gap'], gap_is='upper bound on the possible saving, not a plan',
                multiplicity=r['multiplicity'], gm_effect_pct=r['gm_effect_pct'], slack='unavailable (G5)',
                negatives=[f"{n['id']} ({n['outcome']}; {n['why']})" for n in r['negatives']],
                first_test=r['first_test'], tier='static', tier_label='static model / projection')


# ---------------------------------------------------------------- the verb
def _load_exec(api, L):
    p = L.dir / 'exec' / 'exec_seed1.json'
    f = L.dir / 'exec' / 'exec_facts.json'
    if not (p.exists() and f.exists()):
        raise api.QhdlError('no per-op executed counts for this circuit',
                            f"run `{api.cmd('exec', '-c', L.meta['sha12'])}` first (qhdl avenues reuses its per-op counts)")
    d = json.loads(p.read_text())
    facts = json.loads(f.read_text())
    return d, facts


def avenues(api, L, top=10, refine=None, time_budget=None):
    import qhdl_exec as X
    import qhdl_window as W
    ops = L.ops
    d, facts = _load_exec(api, L)
    eps = facts['rotations']['eps']
    rows_n = d['rows']
    cost = op_costs(ops, d['sampled'], d['zpow_sampled'], rows_n, eps, X)
    total = sum(cost)
    q = L.nq
    charge_recorded = facts['rotations']['recorded_toffoli_equivalent']
    err = (total - charge_recorded) / charge_recorded
    segs = segments(ops, cost)
    att = dict(sum_op_costs=round(total, 4), recorded_charge_per_qubit=round(charge_recorded, 4),
               official_cost=round(q * charge_recorded, 2), attribution_error=round(err, 6), q=q,
               tolerance=KILL_TOLERANCE, verdict='pass' if abs(err) <= KILL_TOLERANCE else 'KILL',
               run_segments=sum(s['kind'] == 'run' for s in segs), glue_segments=sum(s['kind'] == 'glue' for s in segs),
               run_charged=round(sum(s['charged'] for s in segs if s['kind'] == 'run'), 3),
               glue_charged=round(sum(s['charged'] for s in segs if s['kind'] == 'glue'), 3))
    if abs(err) > KILL_TOLERANCE:
        return dict(attribution=att, rows=[], closed=[], capped=[], note='attribution KILL: no ranking')
    lv = W.Liveness(ops)
    runs = value_runs(ops)
    found, capped, timed_out = scan_windows(ops, lv, L.xq, runs, time_budget=time_budget)
    ledger = load_ledger()
    rows = build_rows(ops, lv, L.xq, found, charge_recorded, q, ledger, L.meta['sha12'], refine=refine)
    openr, closed, broad, gap0, pos, reps = rank(rows)
    groups = {}
    for sg in segs:
        if sg['kind'] != 'glue' or sg['charged'] <= 0:
            continue
        sig = canon_hash(ops, sg['ops'][0], sg['ops'][1], full=True)
        g = groups.setdefault(sig, dict(sig=sig, ops=sg['ops'], sites=0, charged_each=sg['charged'], total=0.0))
        g['sites'] += 1
        g['total'] += sg['charged']
    glue = sorted(groups.values(), key=lambda g: -g['total'])[:12]
    return dict(attribution=att, rows=openr, closed=closed, broad=broad, gap0=gap0, pos=pos, reps=reps, capped=capped, windows=len(found), timed_out=timed_out,
                glue_top=glue, runs=len(runs), charge=charge_recorded, q=q)


def v_avenues(api, a):
    import qhdl_window as W
    if a.task == 'chemistry':
        import qhdl_avenues_chem as C
        return C.v_chem(api, a)
    if a.task == 'dqi':
        import qhdl_avenues_dqi as D
        return D.v_dqi(api, a)
    if a.task not in (None, 'invsqrt'):
        raise api.QhdlError(f'--task {a.task}: invsqrt, chemistry and dqi only', 'see docs/qhdl-avenues.md')
    global TERM_CAP, PRODUCT_CAP
    TERM_CAP, PRODUCT_CAP = a.term_cap, a.term_cap * 100
    meta, L = W._open_invsqrt(api, a.circuit)
    led = E.Ledger()
    t0 = time.time()
    res = avenues(api, L, top=a.top, refine=a.refine)
    att = res['attribution']
    led.earn('static', 'per-op executed means from qhdl exec (sampled, seed 1) priced with the recorded verifier eps; '
                       'windows by symbolic ANF; prices are arithmetic')
    out_dir = L.dir / 'avenues'
    full = dict(schema='vqcc-qhdl-avenues/1', circuit_sha256=meta['sha256'], attribution=att,
                windows=res.get('windows'), runs=res.get('runs'), capped=res.get('capped'),
                open=[public_row(r, res.get('charge', 0)) | dict(sites=r['sites'], cluster_windows=r['cluster_windows'])
                      for r in res['rows']],
                closed=[public_row(r, res.get('charge', 0)) | dict(sites=r['sites'], cluster_windows=r['cluster_windows'])
                        for r in res['closed']],
                candidates=[dict(ops=r['ops'], toffoli=r['toffoli'], floor=r['lb'], gap=r['gap'],
                                 multiplicity=r['multiplicity'], gm_effect_pct=r['gm_effect_pct'], closed=r['closed'],
                                 ) for r in res['pos']],
                broad_windows=[dict(ops=r['ops'], toffoli=r['toffoli'], floor=r['lb'], gap=r['gap'],
                                    gm_effect_pct=r['gm_effect_pct']) for r in res['broad'][:100]],
                gap0_windows=res['gap0'],
                repeated=[dict(component=f"ops {r['ops'][0]}:{r['ops'][1]} sig {r['sig']}", charged_each=float(r['toffoli']),
                               floor=r['lb'], gap=r['gap'], multiplicity=r['multiplicity'], total_charged=r['toffoli'] * r['multiplicity'],
                               gm_effect_pct=r['gm_effect_pct'], sites=r['sites']) for r in res['reps']]
                         + [dict(component=f"glue ops {g['ops'][0]}:{g['ops'][1]} sig {g['sig']}", charged_each=round(g['charged_each'], 3),
                                 floor='unavailable', gap='unavailable', multiplicity=g['sites'], total_charged=round(g['total'], 3),
                                 gm_if_free_pct=round(gm_effect_pct(g['total'], res['charge']), 4)) for g in res['glue_top']],
                unfloored=dict(
                    note='components with no computable floor here (glue with rotations/HMR, term-capped windows): charged '
                         'cost only; gm_if_free is the ceiling if the whole component cost vanished, not a gap',
                    glue=[dict(first_site=g['ops'], sig=g['sig'], multiplicity=g['sites'],
                               charged_each=round(g['charged_each'], 3), charged_total=round(g['total'], 3),
                               gm_if_free_pct=round(gm_effect_pct(g['total'], res['charge']), 4))
                          for g in res['glue_top']], term_capped=res['capped'][:60]),
                seconds=round(time.time() - t0, 1))
    ev = Path(a.out) / 'avenues.json' if a.out else out_dir / 'avenues.json'
    api.write_json(ev, full)
    if att['verdict'] == 'KILL':
        raise api.QhdlError(f"cost attribution off by {100 * att['attribution_error']:.1f}% (> 10%): not ranking",
                            'check qhdl exec evidence')
    shown = res['rows'][:a.top]
    below = res['closed'][:max(0, a.top - len(shown))]
    rows = []

    def add(r, n):
        rows.append({'#': n, 'ops': f"{r['ops'][0]}:{r['ops'][1]}", 'cost': float(r['toffoli']), 'floor': r['lb'],
                     'gap': r['gap'], 'mult': r['multiplicity'], 'gm_pct': r['gm_effect_pct'],
                     'neg': ','.join(('' if n_['covers'] else 'rel:') + n_['id'] for n_ in r['negatives'][:3]) or 0, 'tier': 'static'})
    for i, r in enumerate(shown):
        add(r, i + 1)
        rows[-1]['ops'] = f"{r['region'][0]}:{r['region'][1]} best {r['ops'][0]}:{r['ops'][1]}"
    if below:
        rows.append({'#': '---', 'ops': 'closed by negatives (below the divider)', 'tier': 'static'})
        for i, r in enumerate(below):
            add(r, len(shown) + i + 1)
    reps = [dict(ops=f"{r['ops'][0]}:{r['ops'][1]}", cost=float(r['toffoli']), floor=r['lb'], gap=r['gap'],
                 mult=r['multiplicity'], gm_pct=r['gm_effect_pct'], tier='static') for r in res['reps'][:3]]
    reps += [dict(ops=f"glue {g['ops'][0]}:{g['ops'][1]}", cost=round(g['charged_each'], 3), floor='n/a', gap='n/a',
                  mult=g['sites'], gm_pct=round(gm_effect_pct(g['total'], res['charge']), 4), tier='static')
             for g in res['glue_top'][:3]]
    if reps:
        rows.append({'#': '---', 'ops': 'repeated components (gap may be 0; glue gm_pct = ceiling if free)', 'tier': 'static'})
        rows += reps
    claim = (f"{len(res['rows'])} open avenues (+{len(res['closed'])} closed by negatives) from {res['windows']} "
             f"windows over {res['runs']} value runs; attribution {att['sum_op_costs']} vs verifier charge "
             f"{att['recorded_charge_per_qubit']} per qubit ({100 * att['attribution_error']:+.3f}%, cost "
             f"{att['official_cost']:,}); gaps are upper bounds")
    facts = dict(attribution=att, fl_note='floor = degree, nonlinear rank, Dickson, cubic derivative rank, residual-span polar rank (all lower bounds)',
                 term_capped_windows=len(res['capped']), full_json=api.rel(ev))
    return E.Envelope(led, claim, domain='invsqrt: value-gate windows, exact arbitrary-entry ANF; executed means '
                                       'sampled(3133452)', evidence=api.rel(ev),
                      rerun=api.cmd('avenues', '-c', meta['sha12'], '--top', a.top, *(['--term-cap', a.term_cap] if a.term_cap != 6000 else [])), tool_sha=api.tool_sha(),
                      circuit_sha=meta['sha256'], facts=facts, rows=rows, table=api.rel(ev), next_cmds=[r['first_test']['command'] for r in (shown or below)[:2]])


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('avenues', help='ranked investigation avenues: cost attribution + floors + repetition + '
                                              'pricing + negatives (invsqrt, chemistry loaders, dqi)',
                              description=__doc__, formatter_class=__import__('argparse').RawDescriptionHelpFormatter)
    p.add_argument('-c', '--circuit', help='circuit: .kmb/.kmx path or sha prefix (invsqrt; see --task for chemistry and dqi)')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--task', default=None, help='invsqrt (default) or chemistry (-c = loader family sha prefix) or dqi')
    p.add_argument('--stage', help='chemistry: circuits dataset stage dir (default $QHDL_DATASET)')
    p.add_argument('--packet', help="chemistry: optional crossover packet.json (G7 rows; QHDL_CHEM_PACKET)")
    p.add_argument('--slots', help='chemistry: comma list of loader slots to analyse (default all 15)')
    p.add_argument('--regions', help='dqi: cached `workbench summarize` region tree JSON (names regions only; default: compute/cache)')
    p.add_argument('--ops', help='dqi: analyse one op window A:B (exact ANF of values and phase) instead of ranking')
    p.add_argument('--refine', type=int, default=None, help='re-floor only the best K gap>0 candidates (default: all)')
    p.add_argument('--term-cap', type=int, default=6000, help='max ANF terms per wire before a window is reported as '
                                                              'term_cap (default 6000; larger is slower)')
    p.add_argument('--out', help='directory for the full result JSON (default work/<sha>/avenues/)')
    p.add_argument('--json', action='store_true', help='(envelope is always JSON; the full table is in the evidence file)')
    p.add_argument('--all', action='store_true')
    p.set_defaults(fn=lambda a: v_avenues(api, a))
