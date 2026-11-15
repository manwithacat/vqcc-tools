#!/usr/bin/env python3
"""qhdl avenues, chemistry loaders (guide: docs/qhdl-avenues.md). Imported by qhdl_avenues.py.

    qhdl avenues --task chemistry -c <family sha prefix> [--stage DIR] [--packet packet.json] [--top N] [--out DIR]

Chemistry cost = lambda_eff * Q * T (official verify.py), T = angle-loader Toffolis (forward + reverse, i.e. 2 x the
per-pass sum) + modeled non-angle Toffolis (SpinFreeData_V1 heuristic). One ANGLE_k loader per slot k = 0..14.

  G1  per slot and region executed Toffolis. Loaders hold only unconditioned CCX, so executed = static; regions are the
      cache prefix (every CCX is preceded by an R of its fresh target), the product run, and the HMR/CZ phase tail.
      Control: sum over slots x 2 (two passes) + modeled non-angle vs the verifier T of the family (within 1%).
  G2  floors. (a) sliding windows over the loader's value runs (R counts as a zero-write, so cache wires are not free
      inputs); exact arbitrary-entry ANF; qhdl_window floors; sizes 2..16 CCX. (b) per-slot whole-region floors: the
      cache region, the product region (cache wires free), and the whole loader with only the register outputs required
      (words XOR table(selection)); with sampled-Dickson combination floors. These are lower bounds for the function
      over ALL 2048 addresses; the 828 unused addresses and 508 inactive words are not modelled (slack, generator G5),
      so with them the true floor can be lower. Whole-loader floors are WEAK (~ number of outputs): the loader is a
      lookup, and a proven floor on a 11-bit, 35-output table is tiny next to 268 products. Labelled, never hidden.
  G3  repeats: windows with the same canonical hash across all 15 slots (and the cache region itself).
  G4' best known: cheapest public loader with the SAME function (slot, previous words hash, words hash), from
      parts/angle_circuits.csv; a cheaper one is a realised gap (not an upper bound).
  G7  optional crossover packet rows (JSON, read-only; --packet or QHDL_CHEM_PACKET), priced against the family and marked dominated when their loader
      Toffolis per shot exceed the family's.
  negatives: rows of an optional negative-results ledger CSV (none shipped: then no rows) naming a slot or op range.
Tier: static model / projection. Gaps are upper bounds except the realised best-known gap.
"""
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_bitsim as B  # noqa: E402
import qhdl_avenues as A  # noqa: E402

ROOT = HERE.parent
import vqcc_config as CFG  # noqa: E402
DEFAULT_STAGE = CFG.DATASET
DEFAULT_PACKET = Path(os.environ.get('QHDL_CHEM_PACKET', '')) if os.environ.get('QHDL_CHEM_PACKET') else None
KMPY = CFG.KMPY
VERIFIERS = CFG.VERIFIERS
N_ORBITALS = 76
PASSES = 2                     # a loader runs in U_T and in U_T^dagger: per-shot Toffolis = 2 x static CCX
CHEM_SIZES = tuple(range(2, 17))
CHEM_MAX_VARS = 600
CHEM_TERM_CAP = 5000
CHEM_PRODUCT_CAP = 8_000_000
SLOTS = 15
CHEM_LEDGER_TASK = 'chemistry'


# ---------------------------------------------------------------- pricing
def chem_cost(lam, q, t):
    return lam * q * t


def gm_pct(saving_shot, t):
    """Three-task GM change (%) when the per-shot chemistry Toffolis T fall by `saving_shot` (cost ratio T/(T-s))."""
    if saving_shot <= 0:
        return 0.0
    if saving_shot >= t:
        return float('inf')
    return 100.0 * math.log(t / (t - saving_shot)) / 3.0


def price(saving_shot, fam):
    s = float(saving_shot)
    return dict(saving_toffoli_per_shot=s, saving_toffoli_per_pass=s / PASSES,
                delta_cost=round(fam['lambda_eff'] * fam['q'] * s, 2),
                cost_after=round(chem_cost(fam['lambda_eff'], fam['q'], fam['t'] - s), 2),
                gm_effect_pct=round(gm_pct(s, fam['t']), 5))


# ---------------------------------------------------------------- stage / family
def _csv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_family(stage, prefix):
    stage = Path(stage)
    fams = [r for r in _csv(stage / 'parts/families.csv') if r['family'].startswith(prefix)]
    if len(fams) != 1:
        raise ValueError(f'family prefix {prefix!r}: {len(fams)} matches in {stage}/parts/families.csv')
    fam = fams[0]
    ang = _csv(stage / 'parts/angle_circuits.csv')
    byid = {r['part_id']: r for r in ang}
    parts = [byid[p] for p in fam['parts'].split(';')]
    group = {}
    for r in ang:
        if int(r['num_qubits']) <= 128:
            group.setdefault((r['slot_k'], r['prev_words_sha256'], r['words_sha256']), []).append(r)
    fr = {r['npz_sha256']: r for r in _csv(stage / 'parts/factorizations.csv')}.get(fam['npz_sha256'], {})
    dirs = stage / 'circuits/chemistry' / fam['family']
    slots = []
    for k, r in enumerate(parts):
        p = dirs / f'angle_k_{k}.kmb'
        if not p.exists() or sha256_file(p) != r['file_sha256']:
            p = stage / r['dataset_path']          # same bytes live under the first family that published them
        if not p.exists() or sha256_file(p) != r['file_sha256']:
            raise ValueError(f'slot {k}: no .kmb under {stage} matches part {r["part_id"]} sha256')
        slots.append(dict(k=k, part=r, kmb=p))
    return fam, fr, slots, group


def heuristic_toffoli(rbc_brot, cache=None):
    """Modeled non-angle Toffolis (verify.heuristic_costs) via the kickmix interpreter; None if unavailable."""
    key = ','.join(map(str, rbc_brot))
    if cache is not None and key in cache:
        return cache[key]
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from dfthcsos import verify as v; "
            "print(v.heuristic_costs(%d, %s)['toffoli'])" % (N_ORBITALS, key))
    try:
        out = subprocess.run([KMPY, '-I', '-c', code, str(VERIFIERS)], capture_output=True, text=True, timeout=300)
        val = int(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else None
    except Exception:
        val = None
    if cache is not None:
        cache[key] = val
    return val


def to_kmx(kmb, kmx):
    if Path(kmx).exists():
        return
    Path(kmx).parent.mkdir(parents=True, exist_ok=True)
    code = ('import kickmix as km, sys; c = km.Circuit.from_file(sys.argv[1], format="auto"); '
            'c.to_file(sys.argv[2], format="kmx")')
    subprocess.run([KMPY, '-I', '-c', code, str(kmb), str(kmx)], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------- loader structure and G1
def chem_runs(ops):
    """Maximal ranges [a,b) of unconditioned X/CX/CCX/SWAP/R ops (R is a zero-write; loader caches are R + CCX pairs)."""
    ok = (B.C_X, B.C_CX, B.C_CCX, B.C_SWAP, B.C_R)
    runs, s = [], None
    for i, o in enumerate(ops):
        good = o.code in ok and o.cond < 0
        if good and s is None:
            s = i
        if not good and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(ops)))
    return runs


def loader_regions(ops):
    """Cache prefix [0, c): the chain of adjacent (R, CCX into that wire) pairs from the first R; product run [c, h):
    to the first HMR; tail [h, n): HMR + CZ phase repair (slot 0 of the f2069/e10 families runs a second compute after
    its first HMR, so its tail can hold CCX: reported, not floored). Returns dict(name -> (a, b))."""
    n = len(ops)
    rs = [i for i, o in enumerate(ops) if o.code == B.C_R]
    hm = [i for i, o in enumerate(ops) if o.code == B.C_HMR]
    c = 0
    if rs:       # the cache chain: adjacent (R q; CCX .. q) pairs from the first R (later R's, e.g. after an HMR, are not caches)
        i = rs[0]
        while i + 1 < n and ops[i].code == B.C_R and ops[i + 1].code == B.C_CCX and ops[i + 1].q[2] == ops[i].q[0]:
            i += 2
        c = i
    h = hm[0] if hm else n
    c = min(c, h)
    return dict(cache=(0, c), product=(c, h), tail=(h, n))


def count_ccx(ops, a, b):
    """Executed CCX in ops[a:b): unconditioned = 1; conditioned (none in these loaders) = 0.5 (flagged by caller)."""
    full = sum(1 for i in range(a, b) if ops[i].code == B.C_CCX and ops[i].cond < 0)
    cond = sum(1 for i in range(a, b) if ops[i].code in (B.C_CCX, B.C_CCZ) and ops[i].cond >= 0)
    return full, cond


class Loader:
    def __init__(self, slot, kmx):
        self.k = slot['k']
        self.part = slot['part']
        self.kmb = slot['kmb']
        self.ops, self.header, self.regwires = B.parse_kmx(str(kmx))
        self.lv = None
        self.regions = loader_regions(self.ops)
        self.ccx = {nm: count_ccx(self.ops, *ab) for nm, ab in self.regions.items()}
        self.static = sum(v[0] for v in self.ccx.values())
        self.cond = sum(v[1] for v in self.ccx.values())
        self.executed_per_pass = self.static + 0.5 * self.cond
        self.per_shot = PASSES * self.executed_per_pass


def attribution(fam, loaders, heur):
    loaders_shot = sum(L.per_shot for L in loaders)
    t_off = float(fam['T_official'])
    non_angle = heur if heur is not None else None
    total = loaders_shot + (non_angle if non_angle is not None else t_off - loaders_shot)
    err = (total - t_off) / t_off
    return dict(loader_toffoli_per_shot=loaders_shot, loader_toffoli_per_pass=loaders_shot / PASSES,
                non_angle_modeled=non_angle, non_angle_source=('verify.heuristic_costs' if heur is not None else
                                                              'derived = T_official - loaders (control circular)'),
                sum=total, verifier_T=t_off, attribution_error=round(err, 6), tolerance=0.01,
                verdict='pass' if (heur is not None and abs(err) <= 0.01) else ('circular' if heur is None else 'KILL'),
                csv_per_shot=[float(L.part['toffoli_per_shot']) for L in loaders],
                csv_match=[L.per_shot == float(L.part['toffoli_per_shot']) for L in loaders])


# ---------------------------------------------------------------- G2
class ChemState(A.SymState):
    def apply(self, o):
        if o.code == B.C_R:
            self.write(o.q[0], set())
        else:
            super().apply(o)


def _caps():
    A.MAX_VARS, A.TERM_CAP, A.PRODUCT_CAP = CHEM_MAX_VARS, CHEM_TERM_CAP, CHEM_PRODUCT_CAP


def scan_chem_windows(L, sizes=CHEM_SIZES):
    """Windows over the chem runs: start at each CCX (or its immediately preceding R of the same target), end after
    the s-th CCX. -> (found, capped)."""
    _caps()
    ops = L.ops
    lv = L.lv
    wanted, maxsize = set(sizes), max(sizes)
    found, capped = [], []
    for rid, (ra, rb) in enumerate(chem_runs(ops)):
        for i0 in range(ra, rb):
            if ops[i0].code != B.C_CCX:
                continue
            a = i0 - 1 if i0 > ra and ops[i0 - 1].code == B.C_R and ops[i0 - 1].q[0] == ops[i0].q[2] else i0
            st = ChemState(lv, a, L.regwires)
            cnt = 0
            try:
                for i in range(a, rb):
                    o = ops[i]
                    st.apply(o)
                    if o.code == B.C_CCX:
                        cnt += 1
                        if cnt in wanted:
                            outs = st.outputs(i + 1)
                            lb, parts = A.cheap_floor(outs)
                            found.append(dict(run=rid, ops=[a, i + 1], toffoli=cnt, vars=len(st.names), lb=lb,
                                              floor_parts=parts, slot=L.k))
                        if cnt >= maxsize:
                            break
            except A.TermCap:
                capped.append(dict(run=rid, ops=[a, i + 1], toffoli=cnt + 1, slot=L.k))
    return found, capped


def sampled_dickson(outs, nvars, samples=48, seed=1):
    """Lower bound (CCX) from the quadratic-form rank of random output combinations: any combination of outputs is
    computed with at most the same ANDs, and a quadratic form of rank 2r needs >= r. Needs every output degree <= 2;
    returns (bound, how many combos tried). Sound (a max over tried combinations)."""
    import qhdl_window as W
    names = list(outs)
    if not names or any(max((bin(m).count('1') for m in v), default=-1) > 2 for v in outs.values()):
        return 0, 0
    rng = random.Random(seed)
    best = 0
    combos = [[1] * len(names)] + [[rng.randint(0, 1) for _ in names] for _ in range(samples)]
    for c in combos:
        if not any(c):
            continue
        acc = {}
        for bit, nm in zip(c, names):
            if bit:
                for m in outs[nm]:
                    if bin(m).count('1') == 2:
                        acc[m] = acc.get(m, 0) ^ 1
        rows = [0] * nvars
        for m, v in acc.items():
            if v:
                lo = (m & -m).bit_length() - 1
                hi = m.bit_length() - 1
                rows[lo] |= 1 << hi
                rows[hi] |= 1 << lo
        best = max(best, W.gf2_rank(rows) // 2)
    return best, len(combos)


def region_floor(L, name, ab, register_only=False):
    """Exact arbitrary-entry floor of one region as a window. register_only: required outputs are the selection and
    word register wires only (garbage and HMR-read scratch not required). Other regions: live-after outputs."""
    _caps()
    a, b = ab
    ops = L.ops
    st = ChemState(L.lv, a, L.regwires)
    try:
        for i in range(a, b):
            st.apply(ops[i])
    except A.TermCap:
        return dict(region=name, ops=[a, b], status='term_cap', toffoli=count_ccx(ops, a, b)[0])
    regw = set(L.regwires)
    if register_only:
        outs = {f'q{w}': sorted(st.state[w]) for w in st.written if w in regw}
    else:
        outs = st.outputs(b)
    lb, parts = A.cheap_floor(outs)
    nv = len(st.names)
    dk, tried = sampled_dickson(outs, nv)
    return dict(region=name, ops=[a, b], status='exact_anf', toffoli=count_ccx(ops, a, b)[0], vars=nv,
                outputs=len(outs), floor=max(lb, dk), floor_parts=dict(parts, sampled_dickson=dk, combos=tried),
                required_outputs='registers only' if register_only else 'live after the region (HMR counts as a read)',
                floor_scope='arbitrary entry over all 2048 addresses; reachable-address slack not modelled')


# ---------------------------------------------------------------- negatives (slot- or op-range specific only)
def chem_negatives(ledger, slot, a, b):
    out = []
    tags = (f'angle_k_{slot}', f'angle_k{slot}', f'loader {slot}', f'loader k{slot:02d}', f'k{slot:02d}', f'slot {slot}',
            f'k={slot}')
    for r in ledger:
        text = ' '.join(r.get(k, '') for k in ('mechanism', 'family_target', 'key_numbers', 'why_it_failed', 'keywords'))
        low = text.lower()
        neg = r['outcome'] in ('kill', 'null', 'negative')
        ops_hit = [int(n) for n in A._NUM.findall(text) if a <= int(n) < b]
        named = any(t.lower() in low for t in tags)
        if ops_hit:
            out.append(dict(id=r['id'], outcome=r['outcome'], covers=neg, why=f'names op {ops_hit[0]} in this range'))
        elif named and ('loader' in low or 'angle' in low):
            out.append(dict(id=r['id'], outcome=r['outcome'], covers=False,
                            why=f'related: names slot {slot} (slot-specific, threshold-limited; not this range)'))
        elif any(k in low for k in ('loader', 'angle_k', 'qrom', 'lookup')) and neg and any(
                k in low for k in A.WINDOW_KEYS + ('products', 'cache', 'and-minimis', 'resynth')):
            out.append(dict(id=r['id'], outcome=r['outcome'], covers=False, why='related: loader keyword match'))
    return out


# ---------------------------------------------------------------- G7
def g7_rows(packet_path, fam, pick_loader_shot):
    try:
        p = json.loads(Path(packet_path).read_text())
    except Exception as e:  # unavailable is not zero
        return [], dict(status='unavailable', error=str(e)[:120])
    rows = []
    for s in p.get('shown', []):
        lt = float(s['loader_toffoli_per_shot'])
        save = pick_loader_shot - lt
        dominated = lt > pick_loader_shot
        r = dict(generator='G7', path_index=s['path_index'], prefix_table=s['prefix_table'],
                 suffix_table=s['suffix_table'], shared_block_slot=s['shared_block_slot'], authors=s.get('authors'),
                 loader_toffoli_per_shot=lt, family_loader_toffoli_per_shot=pick_loader_shot,
                 saving_vs_family=save, saving_vs_cheaper_parent=s.get('loader_saving_vs_cheaper_parent'),
                 dominated=dominated, actionable=s.get('actionable'), tier='static model / projection',
                 obligations=s.get('obligations'), screened_actionable_note=p.get('next_step'))
        r['price'] = price(max(save, 0.0), fam) if not dominated else dict(
            note='dominated: loader Toffolis per shot exceed the family, no saving to price',
            saving_toffoli_per_shot=0.0, gm_effect_pct=0.0)
        rows.append(r)
    return rows, dict(status='ok', schema=p.get('schema'), all_paths=p.get('all_paths'),
                      positive_loader_saving_paths=p.get('positive_loader_saving_paths'),
                      actionable_paths=p.get('actionable_paths'), catalogue_manifest_sha256=p.get('catalogue_manifest_sha256'))


# ---------------------------------------------------------------- driver
def first_test(fam, L, a, b, floor, toffoli, words_path):
    return dict(
        command=(f'{KMPY} scripts/chem_loader_checker.py {L.kmb} {words_path} {L.k}  # shipped loader must pass; '
                 f'window ops {a}:{b} of the slot-{L.k} kmx is the rewrite target'),
        stop=(f'stop if no rewrite of ops {a}:{b} with fewer than {toffoli} CCX (floor {floor}) verifies on all 1,220 '
              'valid addresses, scratch clean and phase clean within the preregistered budget'),
        rung='verification ladder: chem_loader_checker (exhaustive) -> chem_family_check / chem_assemble -> '
             'local official verifier (local_verify_reference.py --tasks dfthcsos)')


def run(fam_prefix, stage=DEFAULT_STAGE, packet=DEFAULT_PACKET, work=None, ledger_path=None, sizes=CHEM_SIZES,
        slots_only=None, refine=True, time_budget=None):
    t0 = time.time()
    fam, fr, slots, group = load_family(stage, fam_prefix)
    work = Path(work or ROOT / 'work/chem-avenues') / fam['family']
    loaders = []
    for s in slots:
        if slots_only is not None and s['k'] not in slots_only:
            continue
        kmx = work / f"angle_k_{s['k']}_{s['part']['file_sha256'][:12]}.kmx"
        to_kmx(s['kmb'], kmx)
        L = Loader(s, kmx)
        L.lv = __import__('qhdl_window').Liveness(L.ops)
        loaders.append(L)
    lam, q, t = float(fam['lambda_official']), int(fam['Q_official']), float(fam['T_official'])
    famd = dict(family=fam['family'], author=fam['author'], lambda_eff=lam, q=q, t=t,
                cost=float(fam['cost_official']), R=fr.get('R'), B=fr.get('B'), C=fr.get('C'),
                brot=fr.get('brot') or fam.get('brot') or 8)
    heur = None
    if fr.get('R') and slots_only is None:
        heur = heuristic_toffoli((fr['R'], fr['B'], fr['C'], famd['brot']))
    att = attribution(fam, loaders, heur) if slots_only is None else dict(partial=True)
    # G1 table
    g1 = []
    for L in loaders:
        g1.append(dict(slot=L.k, part=L.part['part_id'], ops=len(L.ops), static_ccx=L.static, conditioned_ccx=L.cond,
                       executed_per_pass=L.executed_per_pass, executed_per_shot=L.per_shot,
                       regions={nm: dict(ops=list(ab), ccx=L.ccx[nm][0]) for nm, ab in L.regions.items()},
                       share_of_T=round(L.per_shot / t, 5)))
    # best known (same slot/prev/words) and the realised gap
    known = []
    for L in loaders:
        p = L.part
        g = group.get((p['slot_k'], p['prev_words_sha256'], p['words_sha256']), [])
        best = min(g, key=lambda r: (float(r['toffoli_per_shot']), r['part_id'])) if g else None
        gap = L.per_shot - float(best['toffoli_per_shot']) if best else 0.0
        known.append(dict(slot=L.k, part=p['part_id'], per_shot=L.per_shot, same_function_parts=len(g),
                          best_known_part=best['part_id'] if best else None,
                          best_known_per_shot=float(best['toffoli_per_shot']) if best else None,
                          best_known_first_author=best.get('first_author') if best else None,
                          realised_gap_per_shot=max(gap, 0.0)))
    # G2b whole-region floors
    ledger = A.load_ledger(ledger_path or A.LEDGER, task=CHEM_LEDGER_TASK) if ledger_path != '' else []
    regs = []
    for L in loaders:
        for nm in ('cache', 'product'):
            r = region_floor(L, nm, L.regions[nm])
            r['slot'] = L.k
            regs.append(r)
        r = region_floor(L, 'whole loader (register outputs only)', (0, L.regions['product'][1]), register_only=True)
        r['slot'] = L.k
        regs.append(r)
    # G2a windows
    found_all, capped_all = [], []
    for L in loaders:
        f, c = scan_chem_windows(L, sizes)
        for r in f:
            r['canon'] = A.canon_hash(L.ops, *r['ops'])
        # per-slot multiplicity (non-overlapping), then sum across slots by signature
        rows = [dict(x) for x in f]
        A.attach_multiplicity(L.ops, rows)
        for r in rows:
            r['slot'] = L.k
        found_all += rows
        capped_all += c
    by_sig = {}
    seen = set()
    for r in found_all:
        key = (r['slot'], r['sig'])
        if key not in seen:
            seen.add(key)
            by_sig[r['sig']] = by_sig.get(r['sig'], 0) + r['multiplicity']
    slots_of = {}
    for r in found_all:
        slots_of.setdefault(r['sig'], set()).add(r['slot'])
    L_by = {L.k: L for L in loaders}
    for r in found_all:
        r['multiplicity_all_slots'] = by_sig[r['sig']]
        r['slots'] = sorted(slots_of[r['sig']])
        r['gap'] = max(0, r['toffoli'] - r['lb'])
        r['floor_source'] = 'fast (degree, nonlinear rank)'
    # refine the best candidates (full floor incl. cubic / polar bounds), ranked by global saving
    import qhdl_window as W
    todo = sorted([r for r in found_all if r['gap'] > 0], key=lambda r: (-r['gap'] * r['multiplicity_all_slots'],
                                                                      r['toffoli'], r['slot'], r['ops'][0]))
    if refine:
        _caps()
        done = {}
        for r in todo:
            k = (r['sig'], r['toffoli'])
            if k in done:           # identical canonical window: floor identical, reuse (floors are relabel invariant)
                res = done[k]
            else:
                res = _chem_window_floor(L_by[r['slot']], r['ops'][0], r['ops'][1])
                done[k] = res
            if res['status'] == 'exact_anf':
                r['lb'] = max(r['lb'], res['lb'])
                r['floor_source'] = 'full (degree, nonlinear rank, Dickson, cubic derivative, polar)'
                r['floor_parts'] = res.get('info')
            else:
                r['floor_source'] = 'term_cap in refinement; fast floor kept'
            r['gap'] = max(0, r['toffoli'] - r['lb'])
    for r in found_all:
        s = r['gap'] * r['multiplicity_all_slots'] * PASSES
        r['saving_toffoli_per_shot'] = s
        r['price'] = price(s, famd)
        r['gm_effect_pct'] = r['price']['gm_effect_pct']
        r['negatives'] = chem_negatives(ledger, r['slot'], *r['ops'])
        r['closed'] = any(n['covers'] for n in r['negatives'])
    # region merge per slot, then group identical best windows across slots into one avenue
    per_slot_regions = []
    for L in loaders:
        pos = [r for r in found_all if r['slot'] == L.k and r['gap'] > 0 and r['toffoli'] <= A.RANK_MAX_TOFFOLI]
        for rg in A.regions(pos):
            rg['slot'] = L.k
            per_slot_regions.append(rg)
    groups = {}
    for rg in per_slot_regions:
        g = groups.setdefault(rg['sig'], dict(best=rg, members=[]))
        g['members'].append(dict(slot=rg['slot'], region=rg['region'], best_window=rg['ops'], gap=rg['gap']))
        if rg['gm_effect_pct'] > g['best']['gm_effect_pct']:
            g['best'] = rg
    avenues = []
    for sig, g in groups.items():
        r = g['best']
        # at most one avenue per (sig); the saving counts every non-overlapping site on every slot
        avenues.append(dict(
            kind='window region', component=f"slot {r['slot']} ops {r['ops'][0]}:{r['ops'][1]} (region {r['region'][0]}:"
                                           f"{r['region'][1]}, {r['cluster_windows']} windows) sig {sig}",
            charged_cost=float(r['toffoli']), floor=r['lb'], floor_source=r['floor_source'], gap=r['gap'],
            gap_is='upper bound (arbitrary-entry floor; reachable-address slack not modelled)',
            multiplicity=r['multiplicity_all_slots'], slots=r['slots'], price=r['price'],
            gm_effect_pct=r['gm_effect_pct'], best_known='unavailable at window level (G4 is slot-level here)',
            slack='unavailable (G5)', negatives=[f"{n['id']} ({n['outcome']}; {n['why']})" for n in r['negatives']],
            closed=r['closed'], members=g['members'][:16], tier='static model / projection',
            first_test=first_test(fam, L_by[r['slot']], r['ops'][0], r['ops'][1], r['lb'], r['toffoli'],
                                  Path(stage) / 'parts/words' / f"{fam['table_key']}.npy")))
    avenues.sort(key=lambda a: (-a['gm_effect_pct'], a['charged_cost']))
    # per-slot region rows: product region floors are WEAK (not discriminating); ranked below the realised gaps
    region_av = []
    for r in regs:
        if r['region'] != 'product' or r.get('floor') is None:
            continue
        L = L_by[r['slot']]
        gap = max(0, r['toffoli'] - r['floor'])
        pr = price(gap * PASSES, famd)
        region_av.append(dict(kind='slot product region (WEAK floor)', component=f"slot {r['slot']} product ops "
                              f"{r['ops'][0]}:{r['ops'][1]}", charged_cost=float(PASSES * r['toffoli']),
                              charged_per_pass=float(r['toffoli']), floor=r['floor'], floor_per_shot=PASSES * r['floor'],
                              floor_source='nonlinear rank / sampled Dickson over the live outputs, cache wires free',
                              gap=PASSES * gap, gap_is='upper bound, NOT discriminating: the floor is about the output '
                              'count (<= 45) against 189-215 products; gm_effect is a ceiling, not a target',
                              price=pr, gm_effect_pct=pr['gm_effect_pct'], best_known='see realised rows (slot level)',
                              slack='unavailable (G5): 828 unused addresses, 508 inactive words not modelled',
                              negatives=[f"{n['id']} ({n['outcome']}; {n['why']})" for n in
                                         chem_negatives(ledger, r['slot'], *r['ops'])][:6],
                              closed=False, tier='static model / projection',
                              first_test=first_test(fam, L, r['ops'][0], r['ops'][1], r['floor'], r['toffoli'],
                                                    Path(stage) / 'parts/words' / f"{fam['table_key']}.npy")))
    region_av.sort(key=lambda a: (-a['gm_effect_pct'], a['component']))
    k4 = [n['id'] for n in chem_negatives(ledger, 4, 0, 0)]  # ledger rows naming slot 4 (none without a ledger)
    # realised best-known gaps (slot level)
    realised = []
    for kn in known:
        if kn['realised_gap_per_shot'] > 0:
            L = L_by[kn['slot']]
            realised.append(dict(kind='best-known loader (same function, cheaper public part)',
                                 component=f"slot {kn['slot']} loader {kn['part']}", charged_cost=kn['per_shot'],
                                 best_known=f"{kn['best_known_part']} at {kn['best_known_per_shot']} per shot "
                                            f"({kn['best_known_first_author']})",
                                 gap=kn['realised_gap_per_shot'], gap_is='REALISED elsewhere (not an upper bound)',
                                 price=price(kn['realised_gap_per_shot'], famd), gm_effect_pct=price(
                                     kn['realised_gap_per_shot'], famd)['gm_effect_pct'],
                                 floor='n/a', slack='unavailable (G5)',
                                 negatives=[f"{n['id']} ({n['outcome']}; {n['why']})" for n in
                                            chem_negatives(ledger, kn['slot'], 0, 0)][:4],
                                 first_test=dict(command=f"swap slot {kn['slot']} to part {kn['best_known_part']} "
                                                         f"(parts/angle_circuits.csv) and run chem_assemble.py + "
                                                         f"chem_family_check.py", stop='stop if the family check fails '
                                                         '(interface/previous-words mismatch) or the verifier T differs',
                                                 rung='chem_family_check -> local official verifier'),
                                 tier='static model / projection'))
    realised.sort(key=lambda a: -a['gap'])
    total_realised = sum(k['realised_gap_per_shot'] for k in known)
    pick_loader = sum(L.per_shot for L in loaders)
    g7, g7meta = g7_rows(packet, famd, pick_loader)
    # slot-level structural rows
    slot_rows = []
    for L in loaders:
        who = {r['region']: r for r in regs if r['slot'] == L.k}
        w = who.get('whole loader (register outputs only)')
        slot_rows.append(dict(slot=L.k, charged_per_shot=L.per_shot, static_ccx=L.static,
                              cache_ccx=L.ccx['cache'][0], product_ccx=L.ccx['product'][0],
                              cache_floor=who['cache'].get('floor'), product_floor=who['product'].get('floor'),
                              loader_floor_register_outputs=w.get('floor') if w else None,
                              loader_floor_status=w.get('status') if w else None,
                              structural_gap_upper_bound=(L.static - w['floor']) if w and w.get('floor') is not None else None,
                              structural_note='whole-loader floor is weak (about the output count); not discriminating',
                              dimension_count_reference=generic_reference(L),
                              slot_negatives=[f"{n['id']} ({n['outcome']}; {n['why']})"
                                              for n in chem_negatives(ledger, L.k, 0, 0)][:5]))
    repeats = []
    # cache region repeated across slots (identical canonical hash)
    cs = {}
    for L in loaders:
        a, b = L.regions['cache']
        cs.setdefault(A.canon_hash(L.ops, a, b, full=True), []).append(L.k)
    for sig, ks in cs.items():
        L0 = L_by[ks[0]]
        repeats.append(dict(component=f"cache region ops {L0.regions['cache'][0]}:{L0.regions['cache'][1]} sig {sig}",
                            charged_each_per_shot=PASSES * L0.ccx['cache'][0], slots=ks, multiplicity=len(ks),
                            total_charged_per_shot=PASSES * L0.ccx['cache'][0] * len(ks),
                            gm_if_free_pct=round(gm_pct(PASSES * L0.ccx['cache'][0] * len(ks), t), 4),
                            note='ceiling if the whole component were free, not a gap; carrying caches across loaders '
                                 'is a separate design question'))
    rep_w = {}
    for r in found_all:
        if r['toffoli'] >= 2 and len(r['slots']) >= 2:
            rep_w.setdefault(r['sig'], r)
    top_rep = sorted(rep_w.values(), key=lambda r: (-r['toffoli'] * r['multiplicity_all_slots'], r['ops'][0]))
    kept = []
    for r in top_rep:
        if len(kept) >= 12:
            break
        kept.append(dict(component=f"window sig {r['sig']} e.g. slot {r['slot']} ops {r['ops'][0]}:{r['ops'][1]}",
                         charged_each=float(r['toffoli']), floor=r['lb'], gap=r['gap'],
                         multiplicity=r['multiplicity_all_slots'], slots=r['slots'],
                         total_charged_per_shot=PASSES * r['toffoli'] * r['multiplicity_all_slots']))
    return dict(family=famd, attribution=att, g1=g1, known=known, total_realised_gap_per_shot=total_realised,
                realised=realised, region_avenues=region_av,
                slot4_closure_note=('negative-results ledger rows naming angle_k_4: ' + ', '.join(k4[:4])
                                    + ' (this family ships ' + str(L_by[4].static) + ' CCX per pass); read them before '
                                    're-searching slot 4'
                                    if (k4 and 4 in L_by) else None),
                slot_rows=slot_rows, regions_floors=regs, avenues=avenues, repeats=repeats + kept,
                g7=g7, g7_meta=g7meta, windows=len(found_all), capped=len(capped_all), capped_sample=capped_all[:40],
                gap0_windows=sum(1 for r in found_all if r['gap'] == 0), seconds=round(time.time() - t0, 1),
                pick_loader_per_shot=pick_loader,
                windows_all=[dict(slot=r['slot'], ops=r['ops'], toffoli=r['toffoli'], floor=r['lb'], gap=r['gap'],
                                  mult_all=r['multiplicity_all_slots'], sig=r['sig']) for r in found_all])


def _chem_window_floor(L, a, b):
    _caps()
    st = ChemState(L.lv, a, L.regwires)
    try:
        for i in range(a, b):
            st.apply(L.ops[i])
    except A.TermCap:
        return dict(status='term_cap', lb=None)
    outs = st.outputs(b)
    lb, info = A.cheap_floor(outs)
    lb2, info2 = A.full_floor(outs, len(st.names))
    return dict(status='exact_anf', lb=max(lb, lb2), info=dict(info, **info2))


def generic_reference(L, rows=1220, l=4, outputs=35, n_orb=76):
    """Dimension-count estimate (scripts/chem_c12_bounds.py `generic`) for the low/high-split product count:
    ceil(generic / 2) with generic = ceil(n1 n2 n3 / (n1 + n2 + n3 - 2)). HEURISTIC for generic tensors, NOT a proven
    floor for this table (the planners in that script predicted 196/197 for a shipped 195-product loader)."""
    lo = 1 << l
    hh = -(-rows // lo)
    n1, n2, n3 = hh - 1, lo - 1, outputs
    g = math.ceil(n1 * n2 * n3 / (n1 + n2 + n3 - 2))
    prod = math.ceil(g / 2)
    return dict(products_estimate=prod, products_actual=L.ccx['product'][0],
                estimate_gap=L.ccx['product'][0] - prod, label='heuristic estimate, not a bound', source='chem_c12_bounds.py')


# ---------------------------------------------------------------- the verb
def v_chem(api, a):
    import qhdl_exec  # noqa: F401  (keeps the import graph identical to the invsqrt path)
    import qhdl_envelope as E
    stage = Path(a.stage) if getattr(a, 'stage', None) else DEFAULT_STAGE
    packet = Path(a.packet) if getattr(a, 'packet', None) else DEFAULT_PACKET
    slots_only = {int(x) for x in a.slots.split(',')} if getattr(a, 'slots', None) else None
    if not a.circuit:
        raise api.QhdlError('--task chemistry needs -c <family sha prefix>', 'e.g. -c e10aded7c183')
    try:
        res = run(a.circuit, stage=stage, packet=packet, slots_only=slots_only)
    except ValueError as e:
        raise api.QhdlError(str(e), 'check -c (family) and --stage')
    led = E.Ledger()
    led.earn('static', 'executed loader Toffolis from the loader op streams (unconditioned CCX) x 2 passes; window floors by '
                       'symbolic ANF; prices are arithmetic through lambda_eff x Q x T')
    fam = res['family']
    out = (Path(a.out) / 'avenues_chem.json') if a.out else ROOT / 'work/chem-avenues' / fam['family'] / 'avenues_chem.json'
    api.write_json(out, dict(schema='vqcc-qhdl-avenues-chem/1', **{k: v for k, v in res.items()}))
    att = res['attribution']
    rows = []
    n = 0
    for r in (res['realised'] + res['avenues'] + res['region_avenues'])[:a.top]:
        n += 1
        rows.append({'#': n, 'component': r['component'][:70], 'cost': r['charged_cost'], 'floor': r.get('floor'),
                     'gap': r['gap'], 'gm_pct': r['gm_effect_pct'], 'gap_is': r['gap_is'][:22], 'tier': 'static'})
    rows.append({'#': '---', 'component': 'per-slot (G1/G2): charged per shot, whole-loader floor (weak)', 'tier': 'static'})
    for s in res['slot_rows'][:a.top]:
        rows.append({'#': f"k{s['slot']}", 'component': f"cache {s['cache_ccx']} + products {s['product_ccx']}",
                     'cost': s['charged_per_shot'], 'floor': s['loader_floor_register_outputs'],
                     'gap': s['structural_gap_upper_bound'], 'tier': 'static'})
    rows.append({'#': '---', 'component': 'G7 crossover packet rows (dominated = loader Toffolis > family)', 'tier': 'static'})
    for g in res['g7']:
        rows.append({'#': f"G7-{g['path_index']}", 'component': f"slot {g['shared_block_slot']} {g['prefix_table'][:6]}/"
                     f"{g['suffix_table'][:6]}", 'cost': g['loader_toffoli_per_shot'],
                     'gap': g['saving_vs_family'], 'gm_pct': g['price'].get('gm_effect_pct'),
                     'gap_is': 'DOMINATED' if g['dominated'] else 'candidate', 'tier': 'static'})
    claim = (f"family {fam['family']}: loaders {res['pick_loader_per_shot']:.0f}/shot of T {fam['t']:.0f} "
             f"(attribution {att.get('sum')} vs verifier T {att.get('verifier_T')}, {100 * att.get('attribution_error', 0):+.3f}%); "
             f"realised best-known loader gap {res['total_realised_gap_per_shot']:.0f}/shot; {res['windows']} windows, "
             f"{res['gap0_windows']} at floor, {len(res['avenues'])} open window avenues, {len(res['region_avenues'])} weak-floor region rows; gaps are upper bounds "
             f"except the realised best-known gap")
    facts = dict(attribution=att, full_json=api.rel(out), windows=res['windows'], term_capped=res['capped'])
    return E.Envelope(led, claim, domain='chemistry ANGLE_k loaders: executed = static CCX x 2 passes; exact arbitrary-entry '
                                         'ANF windows (R = zero-write)', evidence=api.rel(out),
                      rerun=api.cmd('avenues', '--task', 'chemistry', '-c', fam['family'], '--top', a.top),
                      tool_sha=api.tool_sha(), circuit_sha=fam.get('family'), facts=facts, rows=rows, table=api.rel(out),
                      next_cmds=[x['first_test']['command'] for x in (res['realised'] + res['avenues'] + res['region_avenues'])[:2]])
