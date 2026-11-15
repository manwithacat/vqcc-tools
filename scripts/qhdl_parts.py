"""qhdl parts: bounded, read-only agent verb over the PUBLIC modular parts catalogue
(Kaggle dataset https://www.kaggle.com/datasets/manwithacat/vqcc-modular-circuit-parts).

The catalogue (bundle_public/ of a release directory, or the bundle directory itself) is DATA: JSON-lines metadata
and typed interfaces for public invsqrt, chemistry and DQI circuit parts, with provenance. This module is a small
standard-library reader; it imports no catalogue code and executes nothing from the catalogue. The manifest
SHA-256s are verified on every load; a mismatch refuses to answer.

    qhdl parts summary
    qhdl parts source <sha-or-family prefix>
    qhdl parts module <module id>
    qhdl parts compose <left id> <right id>        left = the module that ends at the cut, right = the one that starts
    qhdl parts chain <id> <id> ...

Catalogue choice: --catalogue DIR (a release staging dir containing bundle_public/, or the bundle dir itself), else env
QHDL_PARTS_CATALOGUE, else the newest $QHDL_OUTPUTS/r-modular-parts-public-release-v*/ that has a bundle.

Tier: every answer is `static` metadata lookup (recorded evidence, nothing re-run, no circuit read). The recorded
evidence tier of each fact is carried in the facts, never promoted. Statuses for a pair:

  exact_source_continuity     byte-contiguous in the same public source (adjacent spans / shared cut)
  local_all_x_cut_pass        cross-source invsqrt join, SCOPED: this donor cut to this host cut only, all legal x
                              under zero/seed7/seed101 BitSim with the recorded repair ops; NOT a verified join, the
                              remaining obligations are listed. Where the catalogue annotates one exact assembled KMB
                              with a hosted official verifier PASS, that is reported as applying to that
                              named KMB only.
  rejected_at_this_cut        the tested cut failed (reason recorded)
  known_table_chain           chemistry pair: interface checks pass and a known word table is shared (pairwise only)
  chain_matches_without_known_table   chemistry pair interface matches but no shared known table
  structural_mismatch         interface fields, family or span order do not fit
  unknown_cross_source_join   no certificate in the catalogue (all cross-source DQI pairs; untested invsqrt cuts)

This verb never prints "verified" for a cross-source join.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import qhdl_envelope as E

ROOT = Path(__file__).resolve().parent.parent
RELEASE_GLOB = 'r-modular-parts-public-release-v*'
MAX_CHAIN = 64
MAX_ROWS = 20
CHEM_SLOTS = 15

FAMILY_FILES = {
    'invsqrt': dict(modules='modules.jsonl', sources='sources.jsonl', edges='source_edges.jsonl',
                    joins='cross_source_joins.jsonl', cuts='cuts.jsonl'),
    'chemistry': dict(modules='chemistry_modules.jsonl', tables='chemistry_word_tables.jsonl',
                      factorizations='chemistry_factorizations.jsonl', accuracy='chemistry_accuracy.jsonl',
                      control='chemistry_best_control.jsonl'),
    'dqi': dict(modules='dqi_modules.jsonl', sources='dqi_sources.jsonl', edges='dqi_source_edges.jsonl'),
}
# weakest first: a chain is only as strong as its weakest pair
STATUS_RANK = ['structural_mismatch', 'rejected_at_this_cut', 'unknown_cross_source_join',
               'chain_matches_without_known_table', 'nonadjacent_same_source_modules', 'local_all_x_cut_pass',
               'known_table_chain', 'exact_source_continuity']


class PartsError(ValueError):
    def __init__(self, msg, hint=None):
        super().__init__(msg)
        self.hint = hint


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _version(p: Path) -> int:
    m = re.search(r'-v(\d+)-', p.name)
    return int(m.group(1)) if m else -1


def default_catalogue(root: Path = ROOT) -> Path:
    env = os.environ.get('QHDL_PARTS_CATALOGUE')
    if env:
        return Path(env)
    cands = [p for p in (Path(os.environ.get('QHDL_OUTPUTS') or root / 'outputs')).glob(RELEASE_GLOB) if (p / 'bundle_public' / 'manifest.json').is_file()]
    if not cands:
        raise PartsError('no modular-parts catalogue found',
                         'download the Kaggle dataset manwithacat/vqcc-modular-circuit-parts and pass --catalogue DIR or set QHDL_PARTS_CATALOGUE')
    return max(cands, key=_version)


class Catalogue:
    """Manifest-verified reader. Nothing is imported or executed from the catalogue directory."""

    def __init__(self, path):
        p = Path(path)
        if (p / 'bundle_public' / 'manifest.json').is_file():
            self.release_dir, self.bundle = p, p / 'bundle_public'
        elif (p / 'manifest.json').is_file():
            self.release_dir, self.bundle = None, p
        else:
            raise PartsError(f'{p} is not a modular parts catalogue (no bundle_public/manifest.json or manifest.json)')
        self.path = p
        self.manifest = json.loads((self.bundle / 'manifest.json').read_text())
        self.manifest_sha256 = _sha(self.bundle / 'manifest.json')
        self.checked_files = self._verify()
        self._cache = {}
        self._index = None

    def _verify(self) -> int:
        n = 0
        rel = self.release_dir / 'release-manifest.json' if self.release_dir else None
        if rel is not None and rel.is_file():
            want = json.loads(rel.read_text()).get('bundle_manifest_sha256')
            if want and want != self.manifest_sha256:
                raise PartsError('release-manifest.json bundle_manifest_sha256 does not match bundle manifest.json',
                                 'catalogue is inconsistent; do not use it')
        fams = self.manifest.get('families') or {}
        if not fams:
            raise PartsError('manifest lists no families')
        for fam, spec in fams.items():
            for name, want in sorted(spec['files'].items()):
                f = self.bundle / fam / name
                if not f.is_file():
                    raise PartsError(f'manifest lists missing file {fam}/{name}')
                got = _sha(f)
                if got != want:
                    raise PartsError(f'SHA-256 mismatch for {fam}/{name}: manifest {want[:12]}, file {got[:12]}',
                                     'catalogue is corrupt or edited; do not use it')
                n += 1
            if spec.get('source_manifest_sha256') and spec['source_manifest_sha256'] != _sha(self.bundle / fam / 'manifest.json'):
                raise PartsError(f'{fam}/manifest.json does not match source_manifest_sha256')
        return n

    def rows(self, fam: str, key: str) -> list:
        k = (fam, key)
        if k not in self._cache:
            f = self.bundle / fam / FAMILY_FILES[fam][key]
            self._cache[k] = _rows(f) if f.is_file() else []
        return self._cache[k]

    def modules(self) -> dict:
        if self._index is None:
            idx = {}
            for fam in FAMILY_FILES:
                for r in self.rows(fam, 'modules'):
                    if r['module_id'] in idx:
                        raise PartsError(f"module ID collision: {r['module_id']}")
                    idx[r['module_id']] = (fam, r)
            self._index = idx
        return self._index

    def module(self, mid: str):
        idx = self.modules()
        if mid in idx:
            return idx[mid]
        hits = [k for k in idx if k.startswith(mid)] if len(mid) >= 6 else []
        if len(hits) == 1:
            return idx[hits[0]]
        if len(hits) > 1:
            raise PartsError(f'module prefix {mid!r} is ambiguous ({len(hits)} matches)', ', '.join(hits[:5]))
        raise PartsError(f'unknown module id {mid!r}', 'qhdl parts source <sha12> lists the modules of a source')

    def source_of(self, fam: str, rec: dict) -> dict | None:
        if fam == 'chemistry':
            return None
        for s in self.rows(fam, 'sources'):
            if s['source_sha256'] == rec['source_sha256']:
                return s
        return None


# ------------------------------------------------------------------ provenance helpers
def prov_of(cat: Catalogue, fam: str, rec: dict) -> dict:
    if fam == 'chemistry':
        p = rec.get('provenance') or {}
        return dict(author=p.get('first_author'), url=p.get('first_notebook'), version=p.get('first_notebook_version'),
                    license=None, file_sha256=rec.get('file_sha256'), note='first appearance; licence not in catalogue row')
    s = cat.source_of(fam, rec)
    if not s:
        return dict(author=None, url=None, version=None, license=None, note='source row missing')
    return dict(author=s.get('author'), url=s.get('source_notebook_url'), version=s.get('source_notebook_version'),
                license=s.get('license'), source_sha256=s['source_sha256'])


def module_brief(fam: str, r: dict) -> dict:
    res = r.get('static_resources') or {}
    if fam == 'invsqrt':
        return dict(module_id=r['module_id'], span=r['op_span'], role=r.get('module_role'),
                    toffoli=res.get('static_toffoli_like'), ops=res.get('operations'))
    if fam == 'dqi':
        return dict(module_id=r['module_id'], span=r['op_span'], role=r.get('stage_label'),
                    toffoli=res.get('toffoli_like'), ops=res.get('operations'))
    rs = r.get('resource') or {}
    return dict(module_id=r['module_id'], slot=r['entry']['slot'], toffoli=rs.get('static_toffoli'),
                qubits=rs.get('num_qubits'), tables=len(r.get('compatible_word_tables') or []))


# ------------------------------------------------------------------ pair logic
def _inv_pair(cat, left, right):
    same = left['source_sha256'] == right['source_sha256']
    if same:
        if left['exit_cut_id'] == right['entry_cut_id']:
            return dict(status='exact_source_continuity', evidence=dict(shared_cut=left['exit_cut_id'], tier='byte-contiguous in one public source'),
                        remaining_obligations=[])
        return dict(status='structural_mismatch', reason='nonadjacent_same_source_modules (left exit cut is not right entry cut)',
                    remaining_obligations=[])
    match = next((j for j in cat.rows('invsqrt', 'joins')
                  if j['donor_cut_id'] == left['exit_cut_id'] and j['host_cut_id'] == right['entry_cut_id']
                  and j['donor_source_sha256'] == left['source_sha256']
                  and j['host_source_sha256'] == right['source_sha256']), None)
    if match is None:
        return dict(status='unknown_cross_source_join', reason='no tested join record for this donor cut and host cut',
                    remaining_obligations=['cut interface test (quantum values, future classical records, phase, measurement history)',
                                           'constructive adapter', 'whole-circuit native salts', 'official whole-circuit verifier'])
    ev = match.get('evidence') or {}
    status = match['candidate_status']
    out = dict(status=status, scope=match.get('composition_scope'), evidence=ev)
    if status == 'local_all_x_cut_pass':
        out['repair_op_text'] = ev.get('repair_op_text')
        out['phase_exception_count'] = ev.get('phase_exception_count')
        out['remaining_obligations'] = ['constructive adapter (assemble prefix + recorded repair ops + tail as one KMB)',
                                        'whole-circuit native salts', 'official whole-circuit verifier']
        w = match.get('verified_whole_candidate')
        if w:
            out['named_whole_candidate'] = dict(
                name=w.get('name'), kmb_sha256=w.get('source_sha256'), assembly=w.get('assembly'),
                hosted_official_verifier=w.get('hosted_official_verifier'), hosted_version=w.get('hosted_version'),
                local_official_verifier=w.get('local_official_verifier'), native_salts=w.get('native_salts'),
                applies_to='this exact assembled KMB only; it does not certify this join in general, other joins or recompositions',
                url=w.get('hosted_evidence_url'))
            out['remaining_obligations'] = ['for any assembly other than the named KMB: ' + o
                                            for o in ('constructive adapter', 'whole-circuit native salts',
                                                      'official whole-circuit verifier')]
    else:
        out['reason'] = ev.get('reason')
        out['remaining_obligations'] = ['a different cut or a checked adapter; this cut failed']
    return out


def _dqi_pair(cat, left, right):
    same = left['source_sha256'] == right['source_sha256']
    if same:
        if left['op_span'][1] == right['op_span'][0]:
            return dict(status='exact_source_continuity', evidence=dict(shared_op_cut=left['op_span'][1],
                                                                        tier='byte-contiguous stages of one public source'),
                        remaining_obligations=[])
        return dict(status='structural_mismatch', reason='nonadjacent_same_source_modules (left end is not right start)',
                    remaining_obligations=[])
    return dict(status='unknown_cross_source_join', reason='DQI cross-source value/phase/record interfaces are not exported',
                remaining_obligations=['sampled value/record/phase interface', 'wire placement', 'multi-draw DQI cost',
                                       'official whole-circuit verifier'])


def _chem_pair(cat, left, right):
    le, lx, re_ = left['entry'], left['exit'], right['entry']
    checks = {
        'consecutive_slot': lx['slot'] == re_['slot'],
        'word_block': lx['word_block_sha256'] == re_['word_block_sha256'],
        'layout': lx['layout_sha256'] == re_['layout_sha256'],
        'J': le['J'] == re_['J'], 'brot': le['brot'] == re_['brot'],
        'selection_bits': le['selection_bits'] == re_['selection_bits'],
    }
    shared = sorted(set(left.get('compatible_word_tables') or []) & set(right.get('compatible_word_tables') or []))
    ob = ['all 15 slots', 'one compatible word table across them', 'passing factorization/table accuracy pair',
          'whole-assembly verification']
    if not all(checks.values()):
        return dict(status='structural_mismatch', failed_checks=sorted(k for k, v in checks.items() if not v),
                    remaining_obligations=ob)
    st = 'known_table_chain' if shared else 'chain_matches_without_known_table'
    return dict(status=st, checks=sorted(checks), known_common_word_tables=shared,
                scope='pairwise only; says nothing about the other slots', remaining_obligations=ob)


def compose_pair(cat: Catalogue, lid: str, rid: str) -> dict:
    lf, left = cat.module(lid)
    rf, right = cat.module(rid)
    base = dict(left=left['module_id'], right=right['module_id'], left_family=lf, right_family=rf)
    if lf != rf:
        return dict(base, status='structural_mismatch', reason='incompatible interface kinds (different families)',
                    remaining_obligations=[])
    res = {'invsqrt': _inv_pair, 'dqi': _dqi_pair, 'chemistry': _chem_pair}[lf](cat, left, right)
    res = dict(base, **res)
    if lf != 'chemistry':
        res['left_provenance'] = prov_of(cat, lf, left)['author']
        res['right_provenance'] = prov_of(cat, rf, right)['author']
    # guard: a cross-source status never reads as verified
    assert 'verified' not in res['status']
    return res


def weakest(statuses):
    return min(statuses, key=lambda s: STATUS_RANK.index(s) if s in STATUS_RANK else -1)


def chain_report(cat: Catalogue, ids: list) -> dict:
    if len(ids) < 2:
        raise PartsError('a chain needs at least two module ids')
    if len(ids) > MAX_CHAIN:
        raise PartsError(f'chain is bounded to {MAX_CHAIN} modules')
    pairs = [compose_pair(cat, a, b) for a, b in zip(ids, ids[1:])]
    statuses = [p['status'] for p in pairs]
    fams = {cat.module(i)[0] for i in ids}
    out = dict(modules=[cat.module(i)[1]['module_id'] for i in ids], pairs=pairs,
               pair_status_counts={s: statuses.count(s) for s in sorted(set(statuses))},
               chain_status=weakest(statuses), families=sorted(fams))
    if fams == {'chemistry'}:
        out['chemistry_model'] = chemistry_needs(cat, [cat.module(i)[1] for i in ids], statuses)
        out['chain_status'] = ('known_table_chain_pairwise_only' if all(s == 'known_table_chain' for s in statuses)
                               else weakest(statuses))
    else:
        obs = []
        for p in pairs:
            for o in p.get('remaining_obligations') or []:
                if o not in obs:
                    obs.append(o)
        out['remaining_obligations'] = obs
        named = [p['named_whole_candidate'] for p in pairs if p.get('named_whole_candidate')]
        if named:
            out['named_whole_candidates'] = named
    return out


def chemistry_needs(cat: Catalogue, mods: list, statuses: list) -> dict:
    slots = [m['entry']['slot'] for m in mods]
    have = sorted(set(slots))
    missing = [s for s in range(CHEM_SLOTS) if s not in have]
    common = None
    for m in mods:
        t = set(m.get('compatible_word_tables') or [])
        common = t if common is None else common & t
    common = sorted(common or [])
    acc = [a for a in cat.rows('chemistry', 'accuracy') if a['table_key'] in common]
    passing = [a for a in acc if a.get('passes_recorded_gate')]
    best = min(passing, key=lambda a: max(a['error_1_mHa'], a['error_2_mHa']), default=None)
    ctl = next(iter(cat.rows('chemistry', 'control')), None)
    ids = [m['module_id'] for m in mods]
    is_ctl = bool(ctl) and ids == ctl['module_ids']
    needs = []
    if missing or len(slots) != len(have):
        needs.append(f'all {CHEM_SLOTS} slots exactly once (missing {missing}; duplicates {len(slots) - len(have)})')
    if not common:
        needs.append('one word table compatible with every loader')
    if common and not passing:
        needs.append('a passing factorization/table accuracy pair for a common table')
    if not all(s == 'known_table_chain' for s in statuses):
        needs.append('every consecutive pair must be known_table_chain')
    needs.append('whole-assembly verification' + (' (a recorded PASS exists for the recorded best control, ledger tier, not re-run)'
                                                  if is_ctl else ' (none recorded for this chain)'))
    return dict(slots_present=have, slots_missing=missing, common_word_tables=common,
                recorded_accuracy_pairs=len(acc), recorded_passing_pairs=len(passing),
                best_recorded_pair=(dict(table_key=best['table_key'], factorization_npz_sha12=best['npz_sha256'][:12],
                                         error_1_mHa=round(best['error_1_mHa'], 4), error_2_mHa=round(best['error_2_mHa'], 4),
                                         lambda_eff=round(best['lambda_eff'], 4), tier=best['evidence_tier'])
                                    if best else None),
                matches_recorded_best_control=is_ctl,
                recorded_best_control=(dict(recorded_verified_status=ctl['recorded_verified_status'],
                                            recorded_verified_cost=ctl['recorded_verified_cost'],
                                            tier=ctl['evidence_tier']) if is_ctl else None),
                complete_model=False if needs else True,
                still_needs=needs)


# ------------------------------------------------------------------ queries
def q_summary(cat: Catalogue) -> dict:
    inv_j = cat.rows('invsqrt', 'joins')
    acc = cat.rows('chemistry', 'accuracy')
    ctl = cat.rows('chemistry', 'control')
    return dict(
        catalogue=dict(path=str(cat.path), manifest_sha256=cat.manifest_sha256, schema=cat.manifest.get('schema'),
                       files_verified=cat.checked_files, evidence_policy=cat.manifest.get('evidence_policy')),
        invsqrt=dict(sources=len(cat.rows('invsqrt', 'sources')), modules=len(cat.rows('invsqrt', 'modules')),
                     cuts=len(cat.rows('invsqrt', 'cuts')), source_edges=len(cat.rows('invsqrt', 'edges')),
                     joins=len(inv_j),
                     joins_by_status={s: sum(j['candidate_status'] == s for j in inv_j)
                                      for s in sorted({j['candidate_status'] for j in inv_j})},
                     joins_with_named_whole_candidate=sum(bool(j.get('verified_whole_candidate')) for j in inv_j)),
        chemistry=dict(loader_modules=len(cat.rows('chemistry', 'modules')), word_tables=len(cat.rows('chemistry', 'tables')),
                       factorizations=len(cat.rows('chemistry', 'factorizations')), accuracy_pairs=len(acc),
                       recorded_passing_pairs=sum(bool(a.get('passes_recorded_gate')) for a in acc),
                       best_control_modules=len(ctl[0]['module_ids']) if ctl else 0),
        dqi=dict(sources=len(cat.rows('dqi', 'sources')), stage_modules=len(cat.rows('dqi', 'modules')),
                 source_edges=len(cat.rows('dqi', 'edges'))))


def q_source(cat: Catalogue, prefix: str) -> dict:
    p = prefix.lower()
    if len(p) < 6 or not re.fullmatch(r'[0-9a-f]+', p):
        raise PartsError('source prefix must be at least 6 hex characters (source SHA-256 or chemistry family/file SHA)')
    hits = []
    for fam in ('invsqrt', 'dqi'):
        for s in cat.rows(fam, 'sources'):
            if s['source_sha256'].startswith(p):
                hits.append((fam, s))
    if len(hits) == 1:
        fam, s = hits[0]
        mods = sorted((r for r in cat.rows(fam, 'modules') if r['source_sha256'] == s['source_sha256']),
                      key=lambda r: r['op_span'][0])
        src = {k: s.get(k) for k in ('author', 'source_notebook_url', 'source_notebook_version', 'license', 'task',
                                     'family', 'qubits', 'operations', 'operation_count', 'local_task_score',
                                     'score_tier', 'evidence_tier', 'classical_bits', 'stage_count') if k in s}
        src['source_sha256'] = s['source_sha256']
        joins = []
        if fam == 'invsqrt':
            for j in cat.rows('invsqrt', 'joins'):
                if s['source_sha256'] in (j['donor_source_sha256'], j['host_source_sha256']):
                    joins.append(dict(role='donor' if j['donor_source_sha256'] == s['source_sha256'] else 'host',
                                      other=(j['host_source_sha256'] if j['donor_source_sha256'] == s['source_sha256']
                                             else j['donor_source_sha256'])[:12], status=j['candidate_status']))
        return dict(kind='source', family=fam, source=src, modules=[module_brief(fam, r) for r in mods],
                    cross_source_joins=joins)
    if len(hits) > 1:
        raise PartsError(f'source prefix {prefix!r} is ambiguous ({len(hits)} matches)',
                         ', '.join(s['source_sha256'][:12] for _, s in hits[:5]))
    mods = [r for r in cat.rows('chemistry', 'modules')
            if r['file_sha256'].startswith(p) or f'/{p}' in r['dataset_path'] or p in (r.get('provenance') or {}).get('also_in_families', [])
            or any(f.startswith(p) for f in (r.get('provenance') or {}).get('also_in_families', []))]
    if not mods:
        raise PartsError(f'no source, chemistry family or loader file matches {prefix!r}')
    mods.sort(key=lambda r: (r['entry']['slot'], r['module_id']))
    firsts = sorted({((r.get('provenance') or {}).get('first_author'), (r.get('provenance') or {}).get('first_notebook'),
                      (r.get('provenance') or {}).get('first_notebook_version')) for r in mods}, key=str)
    return dict(kind='chemistry_family_or_loader', family='chemistry',
                source=dict(match=p, loader_modules=len(mods), first_appearance=[dict(author=a, url=u, version=v) for a, u, v in firsts[:6]],
                            note='chemistry loaders are keyed by file SHA-256; one file can appear in several parameter families; licence is not in the loader rows'),
                modules=[module_brief('chemistry', r) for r in mods], cross_source_joins=[])


def q_module(cat: Catalogue, mid: str) -> dict:
    fam, r = cat.module(mid)
    d = dict(kind='module', family=fam, module_id=r['module_id'], provenance=prov_of(cat, fam, r))
    if fam == 'invsqrt':
        res = r.get('static_resources') or {}
        d.update(span=r['op_span'], role=r.get('module_role'), entry_cut=r['entry_cut_id'], exit_cut=r['exit_cut_id'],
                 semantic_transfer=r.get('semantic_transfer'), toffoli_like=res.get('static_toffoli_like'),
                 operations=res.get('operations'), hmr=res.get('hmr_operations'), phase_ops=res.get('phase_operations'),
                 entry_state_wires=len((r.get('entry') or {}).get('quantum_state_wires') or []),
                 exit_state_wires=len((r.get('exit') or {}).get('quantum_state_wires') or []),
                 exit_records_written=len((r.get('exit') or {}).get('records_written_in_module') or []),
                 source_future_live_records=len((r.get('exit') or {}).get('source_future_live_record_ids') or []),
                 evidence_tier='static source span; cross-source semantics unknown beyond recorded joins')
    elif fam == 'dqi':
        res = r.get('static_resources') or {}
        d.update(span=r['op_span'], stage=r.get('stage_label'), stage_index=r.get('stage_index'), source_tag=r.get('source_tag'),
                 entry=r.get('entry'), exit=r.get('exit'), static_resources=res, sampled_resources=r.get('sampled_resources'),
                 evidence_tier=r.get('evidence_tier'), composition_scope=r.get('composition_scope'))
    else:
        d.update(slot=r['entry']['slot'], entry=r['entry'], exit=r['exit'], resource=r.get('resource'),
                 compatible_word_tables=r.get('compatible_word_tables'), dataset_path=r.get('dataset_path'),
                 file_sha256=r.get('file_sha256'), also_in_families=len((r.get('provenance') or {}).get('also_in_families') or []))
    d['_full'] = r
    return d


# ------------------------------------------------------------------ verb
def _short(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith('_')}


def _write_full(api, out, name, payload):
    if not out:
        return None
    ev = Path(out) / name
    if ev.exists():
        raise api.QhdlError(f'{ev} exists; outputs are append-only', 'choose a new --out directory')
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + '\n')
    return ev


def _argv_catalogue(a):
    return ['--catalogue', api_rel(a.catalogue_path)]


def api_rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def v_parts(api, a):
    cat_path = Path(a.catalogue) if a.catalogue else default_catalogue()
    a.catalogue_path = cat_path
    try:
        cat = Catalogue(cat_path)
    except PartsError as exc:
        raise api.QhdlError(str(exc), exc.hint)
    led = E.Ledger()
    led.earn('static', f'recorded catalogue metadata lookup; manifest SHA-256 verified on {cat.checked_files} files; nothing re-run')
    sub = a.parts_cmd
    cat_arg = _argv_catalogue(a)
    msha = cat.manifest_sha256[:12]
    try:
        if sub == 'summary':
            res = q_summary(cat)
            claim = (f"catalogue {msha}: invsqrt {res['invsqrt']['sources']} sources / {res['invsqrt']['modules']} modules, "
                     f"chemistry {res['chemistry']['loader_modules']} loaders, DQI {res['dqi']['sources']} sources / "
                     f"{res['dqi']['stage_modules']} stages")
            rows = [dict(family=f, **{k: v for k, v in res[f].items() if not isinstance(v, dict)}) for f in ('invsqrt', 'chemistry', 'dqi')]
            facts = dict(manifest_sha256=cat.manifest_sha256, joins_by_status=res['invsqrt']['joins_by_status'],
                         named_whole_candidates=res['invsqrt']['joins_with_named_whole_candidate'],
                         recorded_passing_chem_pairs=res['chemistry']['recorded_passing_pairs'])
            rerun = api.cmd('parts', 'summary', *cat_arg)
            full, name = res, 'parts_summary.json'
        elif sub == 'source':
            res = q_source(cat, a.sha)
            s = res['source']
            who = s.get('author') or (s.get('first_appearance') or [{}])[0].get('author')
            claim = f"source {a.sha[:12]} ({res['family']}): {who}, {len(res['modules'])} module(s); provenance recorded"
            rows = res['modules']
            facts = {k: v for k, v in s.items() if k != 'first_appearance'}
            if s.get('first_appearance'):
                facts['first_appearance'] = s['first_appearance'][:2]
            if res['cross_source_joins']:
                facts['cross_source_joins'] = len(res['cross_source_joins'])
            rerun = api.cmd('parts', 'source', a.sha, *cat_arg)
            full, name = res, f'parts_source_{a.sha[:12]}.json'
        elif sub == 'module':
            res = q_module(cat, a.module_id)
            full = res
            facts = _short(res)
            facts.pop('kind', None)
            claim = f"module {res['module_id']} ({res['family']}): {res['provenance'].get('author')}; interface fields recorded"
            rows = []
            rerun = api.cmd('parts', 'module', a.module_id, *cat_arg)
            name = f"parts_module_{res['module_id']}.json"
        elif sub == 'compose':
            res = compose_pair(cat, a.left, a.right)
            full = res
            claim = f"{res['left']} -> {res['right']}: {res['status']} (recorded; scoped; not a verified join)" \
                if res['status'] == 'local_all_x_cut_pass' else f"{res['left']} -> {res['right']}: {res['status']} (recorded)"
            facts = {k: res[k] for k in ('status', 'scope', 'reason', 'repair_op_text', 'phase_exception_count',
                                         'failed_checks', 'known_common_word_tables', 'remaining_obligations',
                                         'named_whole_candidate', 'left_provenance', 'right_provenance') if k in res}
            rows = []
            rerun = api.cmd('parts', 'compose', a.left, a.right, *cat_arg)
            name = f"parts_compose_{res['left']}_{res['right']}.json"
        elif sub == 'chain':
            a.ids = [i for t in a.ids for i in re.split(r'[,\s]+', t) if i]  # ids may be comma/space joined
            res = chain_report(cat, a.ids)
            full = res
            claim = (f"chain of {len(res['modules'])} modules: {res['chain_status']} "
                     f"({', '.join(f'{k} x{v}' for k, v in res['pair_status_counts'].items())}); recorded, not a whole-circuit proof")
            rows = [dict(pair=f"{p['left']} -> {p['right']}", status=p['status']) for p in res['pairs']]
            facts = {k: res[k] for k in ('chain_status', 'remaining_obligations', 'named_whole_candidates') if k in res}
            if 'chemistry_model' in res:
                cm = res['chemistry_model']
                facts['chemistry_model'] = {k: cm[k] for k in ('slots_missing', 'common_word_tables', 'recorded_passing_pairs',
                                                               'matches_recorded_best_control', 'complete_model', 'still_needs')}
            rerun = api.cmd('parts', 'chain', *a.ids, *cat_arg)
            name = f"parts_chain_{hashlib.sha256(' '.join(a.ids).encode()).hexdigest()[:12]}.json"
        else:
            raise api.QhdlError('choose one of: summary, source, module, compose, chain')
    except PartsError as exc:
        raise api.QhdlError(str(exc), exc.hint)
    ev = _write_full(api, a.out, name, dict(catalogue_manifest_sha256=cat.manifest_sha256, result=full))
    evidence = api.rel(ev) if ev else api.rel(cat.bundle / 'manifest.json')
    rows = rows[:MAX_ROWS] if not a.all else rows
    env = E.Envelope(led, claim, domain='recorded public modular parts catalogue (metadata only; no circuit read)',
                     evidence=evidence, rerun=rerun, tool_sha=api.tool_sha(), circuit_sha=None, facts=facts, rows=rows,
                     table=api.rel(ev) if ev else None, next_cmds=[])
    env.stamp(provenance='recorded')
    return env


def qhdl_register(subparsers, api):
    p = subparsers.add_parser('parts', help='read-only lookup/composition triage over the public modular parts catalogue',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalogue', help='release staging dir (with bundle_public/) or bundle dir; default env '
                                       'QHDL_PARTS_CATALOGUE else newest $QHDL_OUTPUTS/r-modular-parts-public-release-v*/')
    p.add_argument('--out', help='write the full result JSON into this directory (append-only)')
    p.add_argument('--all', action='store_true')
    # the same options are accepted after the sub-command (SUPPRESS keeps a value given before it)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--catalogue', default=argparse.SUPPRESS)
    common.add_argument('--out', default=argparse.SUPPRESS)
    common.add_argument('--all', action='store_true', default=argparse.SUPPRESS)
    s = p.add_subparsers(dest='parts_cmd', required=True)
    s.add_parser('summary', parents=[common])
    x = s.add_parser('source', parents=[common])
    x.add_argument('sha')
    x = s.add_parser('module', parents=[common])
    x.add_argument('module_id')
    x = s.add_parser('compose', parents=[common])
    x.add_argument('left')
    x.add_argument('right')
    x = s.add_parser('chain', parents=[common])
    x.add_argument('ids', nargs='+')
    p.set_defaults(fn=lambda a: v_parts(api, a))
