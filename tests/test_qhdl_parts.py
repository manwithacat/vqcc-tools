"""qhdl parts: synthetic catalogue written here (no dependency on recorded evidence). Checks manifest verification,
provenance, every pair status, the refusal to call a cross-source join verified, the named-candidate scoping (one exact whole circuit hosted-verified),
the chemistry chain needs, bounds, and the envelope through the verb layer."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qhdl_envelope as E  # noqa: E402
import qhdl_parts as P  # noqa: E402

A, B, C = 'a' * 64, 'b' * 64, 'c' * 64
DA, DB = '1' * 64, '2' * 64
KMB = 'f' * 64


def w(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inv_mod(mid, src, a, b, ent, ext, role='seg'):
    return dict(module_id=mid, source_sha256=src, op_span=[a, b], entry_cut_id=ent, exit_cut_id=ext, module_role=role,
                schema='vqcc-modular-parts-v1-module', semantic_transfer='unknown_beyond_same_source_continuity',
                static_resources=dict(static_toffoli_like=5, operations=b - a),
                entry=dict(quantum_state_wires=[0, 1]), exit=dict(quantum_state_wires=[0, 1], records_written_in_module=[1]))


def chem_mod(mid, slot, inw, outw, tables=('T1',), layout='L', J=5):
    return dict(module_id=mid, schema='vqcc-chemistry-modules-v1-loader', family='chemistry_angle_loader',
                entry=dict(slot=slot, word_block_sha256=inw, layout_sha256=layout, J=J, brot=8, selection_bits=11),
                exit=dict(slot=slot + 1, word_block_sha256=outw, layout_sha256=layout),
                compatible_word_tables=list(tables), dataset_path=f'circuits/chemistry/fam1/angle_k_{slot}.kmb',
                file_sha256=hashlib.sha256(mid.encode()).hexdigest(),
                provenance=dict(first_author='Ch Author', first_notebook='https://example/chem', first_notebook_version='3',
                                also_in_families=['fam1']),
                resource=dict(static_toffoli=10, num_qubits=100))


def dqi_mod(mid, src, a, b):
    return dict(module_id=mid, source_sha256=src, op_span=[a, b], schema='vqcc-dqi-stage-modules-v1-stage',
                stage_label='st', stage_index=0, static_resources=dict(toffoli_like=1, operations=b - a),
                entry={}, exit={})


def build(root: Path, with_control=True):
    b = root / 'bundle_public'
    for fam in ('invsqrt', 'chemistry', 'dqi'):
        (b / fam).mkdir(parents=True)
    inv = b / 'invsqrt'
    w(inv / 'sources.jsonl', [dict(source_sha256=s, author=n, source_notebook_url=f'https://k/{n}', source_notebook_version='1',
                                   license='Apache-2.0', task='invsqrt') for s, n in ((A, 'Alice'), (B, 'Bob'), (C, 'Cy'))])
    w(inv / 'modules.jsonl', [
        inv_mod('mod-a0', A, 0, 10, 'cut-a0', 'cut-a1'), inv_mod('mod-a1', A, 10, 20, 'cut-a1', 'cut-a2'),
        inv_mod('mod-a2', A, 20, 30, 'cut-a2', 'cut-a3'), inv_mod('mod-b0', B, 0, 10, 'cut-b0', 'cut-b1'),
        inv_mod('mod-c0', C, 0, 10, 'cut-c0', 'cut-c1')])
    w(inv / 'source_edges.jsonl', [dict(from_module_id='mod-a0', to_module_id='mod-a1', status='exact_source_continuity')])
    w(inv / 'cuts.jsonl', [])
    w(inv / 'cross_source_joins.jsonl', [
        dict(donor_cut_id='cut-a1', host_cut_id='cut-b0', donor_source_sha256=A, host_source_sha256=B,
             candidate_status='local_all_x_cut_pass', composition_scope='one donor prefix to one host tail at these cuts only',
             evidence=dict(repair_op_text=['Z_POW q1 1.0'], phase_exception_count=1, tier='all_legal_x_zero_seed7_seed101_local_bitsim'),
             verified_whole_candidate=dict(name='HW-T', source_sha256=KMB, hosted_official_verifier='PASS', hosted_version=1,
                                           local_official_verifier='PASS', native_salts='8/8', assembly='A[0,10)+B[0,..)',
                                           scope='this exact assembled KMB only')),
        dict(donor_cut_id='cut-a2', host_cut_id='cut-b0', donor_source_sha256=A, host_source_sha256=B,
             candidate_status='local_all_x_cut_pass', composition_scope='scoped',
             evidence=dict(repair_op_text=[], phase_exception_count=0, tier='t')),
        dict(donor_cut_id='cut-b1', host_cut_id='cut-c0', donor_source_sha256=B, host_source_sha256=C,
             candidate_status='rejected_at_this_cut', composition_scope='scoped', evidence=dict(reason='future-live record differs'))])
    ch = b / 'chemistry'
    mods = [chem_mod(f'k{i:02d}-x', i, 'ZERO' if i == 0 else f'W{i}', f'W{i + 1}') for i in range(15)]
    mods.append(chem_mod('k99-badlayout', 15, 'W15', 'W16', layout='OTHER'))
    mods.append(chem_mod('k98-notable', 3, 'W3', 'W4', tables=('T2',)))
    w(ch / 'chemistry_modules.jsonl', mods)
    w(ch / 'chemistry_word_tables.jsonl', [dict(table_key='T1')])
    w(ch / 'chemistry_factorizations.jsonl', [])
    w(ch / 'chemistry_accuracy.jsonl', [dict(table_key='T1', npz_sha256='9' * 64, error_1_mHa=18.0, error_2_mHa=19.0,
                                            lambda_eff=40.0, passes_recorded_gate=True, evidence_tier='recorded_parts_ledger'),
                                       dict(table_key='T1', npz_sha256='8' * 64, error_1_mHa=30.0, error_2_mHa=19.0,
                                            lambda_eff=40.0, passes_recorded_gate=False, evidence_tier='recorded_parts_ledger')])
    w(ch / 'chemistry_best_control.jsonl', [dict(module_ids=[m['module_id'] for m in mods[:15]], recorded_verified_status='PASS',
                                                 recorded_verified_cost=1.5, evidence_tier='recorded_parts_ledger_assembly')] if with_control else [])
    dq = b / 'dqi'
    w(dq / 'dqi_sources.jsonl', [dict(source_sha256=DA, author='Dee', source_notebook_url='https://k/dee', source_notebook_version='2',
                                      license='Apache-2.0'),
                                 dict(source_sha256=DB, author='Eve', source_notebook_url='https://k/eve', source_notebook_version='1',
                                      license='Apache-2.0')])
    w(dq / 'dqi_modules.jsonl', [dqi_mod('dqi-mod-a0', DA, 0, 10), dqi_mod('dqi-mod-a1', DA, 10, 20), dqi_mod('dqi-mod-a2', DA, 30, 40),
                                 dqi_mod('dqi-mod-b0', DB, 0, 10)])
    w(dq / 'dqi_source_edges.jsonl', [])
    files = {}
    for fam in ('invsqrt', 'chemistry', 'dqi'):
        (b / fam / 'manifest.json').write_text(json.dumps(dict(family=fam)))
        files[fam] = dict(files={f.name: sha(f) for f in sorted((b / fam).iterdir())},
                          source_manifest_sha256=sha(b / fam / 'manifest.json'))
    (b / 'manifest.json').write_text(json.dumps(dict(schema='test', families=files, evidence_policy='synthetic')))
    (root / 'release-manifest.json').write_text(json.dumps(dict(bundle_manifest_sha256=sha(b / 'manifest.json'))))
    return root


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = build(Path(self.tmp.name))
        self.cat = P.Catalogue(self.root)

    def tearDown(self):
        self.tmp.cleanup()


class TestLoadAndLookup(Base):
    def test_manifest_verified(self):
        self.assertEqual(self.cat.checked_files, sum(len(f['files']) for f in self.cat.manifest['families'].values()))

    def test_tamper_refused(self):
        with open(self.root / 'bundle_public' / 'invsqrt' / 'modules.jsonl', 'a') as f:
            f.write('\n')
        with self.assertRaises(P.PartsError) as c:
            P.Catalogue(self.root)
        self.assertIn('mismatch', str(c.exception))

    def test_release_manifest_mismatch_refused(self):
        (self.root / 'release-manifest.json').write_text(json.dumps(dict(bundle_manifest_sha256='0' * 64)))
        with self.assertRaises(P.PartsError):
            P.Catalogue(self.root)

    def test_bundle_dir_accepted_directly(self):
        self.assertEqual(P.Catalogue(self.root / 'bundle_public').manifest_sha256, self.cat.manifest_sha256)

    def test_not_a_catalogue(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(P.PartsError):
            P.Catalogue(d)

    def test_summary(self):
        s = P.q_summary(self.cat)
        self.assertEqual(s['invsqrt']['joins_by_status'], {'local_all_x_cut_pass': 2, 'rejected_at_this_cut': 1})
        self.assertEqual(s['invsqrt']['joins_with_named_whole_candidate'], 1)
        self.assertEqual(s['chemistry']['recorded_passing_pairs'], 1)
        self.assertEqual(s['dqi']['stage_modules'], 4)

    def test_source_provenance(self):
        r = P.q_source(self.cat, 'aaaaaa')
        self.assertEqual((r['source']['author'], r['source']['license']), ('Alice', 'Apache-2.0'))
        self.assertEqual([m['module_id'] for m in r['modules']], ['mod-a0', 'mod-a1', 'mod-a2'])
        self.assertEqual({j['role'] for j in r['cross_source_joins']}, {'donor'})

    def test_source_short_or_unknown(self):
        with self.assertRaises(P.PartsError):
            P.q_source(self.cat, 'aaa')
        with self.assertRaises(P.PartsError):
            P.q_source(self.cat, 'dddddd')

    def test_source_chemistry_by_file_sha(self):
        r = P.q_source(self.cat, hashlib.sha256(b'k00-x').hexdigest()[:10])
        self.assertEqual((r['family'], len(r['modules'])), ('chemistry', 1))
        self.assertEqual(r['source']['first_appearance'][0]['author'], 'Ch Author')

    def test_module(self):
        r = P.q_module(self.cat, 'mod-b0')
        self.assertEqual(r['provenance']['author'], 'Bob')
        self.assertEqual(r['entry_cut'], 'cut-b0')
        with self.assertRaises(P.PartsError):
            P.q_module(self.cat, 'mod-zzz')


class TestCompose(Base):
    def test_exact_source_continuity(self):
        for fam_pair in (('mod-a0', 'mod-a1'), ('dqi-mod-a0', 'dqi-mod-a1')):
            self.assertEqual(P.compose_pair(self.cat, *fam_pair)['status'], 'exact_source_continuity')

    def test_nonadjacent_same_source_is_structural(self):
        self.assertEqual(P.compose_pair(self.cat, 'mod-a0', 'mod-a2')['status'], 'structural_mismatch')
        self.assertEqual(P.compose_pair(self.cat, 'dqi-mod-a1', 'dqi-mod-a2')['status'], 'structural_mismatch')

    def test_cross_source_join_scoped_never_verified(self):
        r = P.compose_pair(self.cat, 'mod-a0', 'mod-b0')
        self.assertEqual(r['status'], 'local_all_x_cut_pass')
        self.assertNotIn('verified', r['status'])
        self.assertEqual(r['repair_op_text'], ['Z_POW q1 1.0'])
        self.assertTrue(r['remaining_obligations'])
        self.assertTrue(all(o.startswith('for any assembly other than the named KMB') for o in r['remaining_obligations']))

    def test_named_candidate_applies_to_that_kmb_only(self):
        n = P.compose_pair(self.cat, 'mod-a0', 'mod-b0')['named_whole_candidate']
        self.assertEqual((n['name'], n['kmb_sha256'], n['hosted_official_verifier']), ('HW-T', KMB, 'PASS'))
        self.assertIn('this exact assembled KMB only', n['applies_to'])

    def test_join_without_named_candidate_keeps_full_obligations(self):
        r = P.compose_pair(self.cat, 'mod-a1', 'mod-b0')
        self.assertEqual(r['status'], 'local_all_x_cut_pass')
        self.assertNotIn('named_whole_candidate', r)
        self.assertEqual(r['remaining_obligations'], ['constructive adapter (assemble prefix + recorded repair ops + tail as one KMB)',
                                                      'whole-circuit native salts', 'official whole-circuit verifier'])

    def test_rejected_and_unknown_cross_source(self):
        self.assertEqual(P.compose_pair(self.cat, 'mod-b0', 'mod-c0')['status'], 'rejected_at_this_cut')
        self.assertEqual(P.compose_pair(self.cat, 'mod-c0', 'mod-b0')['status'], 'unknown_cross_source_join')
        self.assertEqual(P.compose_pair(self.cat, 'dqi-mod-a1', 'dqi-mod-b0')['status'], 'unknown_cross_source_join')

    def test_reversed_join_is_not_the_join(self):
        self.assertEqual(P.compose_pair(self.cat, 'mod-b0', 'mod-a0')['status'], 'unknown_cross_source_join')

    def test_mixed_families(self):
        self.assertEqual(P.compose_pair(self.cat, 'mod-a0', 'dqi-mod-a0')['status'], 'structural_mismatch')

    def test_chemistry_pairs(self):
        self.assertEqual(P.compose_pair(self.cat, 'k00-x', 'k01-x')['status'], 'known_table_chain')
        self.assertEqual(P.compose_pair(self.cat, 'k01-x', 'k03-x')['status'], 'structural_mismatch')  # slot gap
        bad = P.compose_pair(self.cat, 'k14-x', 'k99-badlayout')
        self.assertEqual(bad['status'], 'structural_mismatch')
        self.assertIn('layout', bad['failed_checks'])
        self.assertEqual(P.compose_pair(self.cat, 'k02-x', 'k98-notable')['status'], 'chain_matches_without_known_table')  # no shared table


class TestChain(Base):
    ids = [f'k{i:02d}-x' for i in range(15)]

    def test_chemistry_chain_pairs_and_needs(self):
        r = P.chain_report(self.cat, self.ids)
        self.assertEqual(r['pair_status_counts'], {'known_table_chain': 14})
        self.assertEqual(r['chain_status'], 'known_table_chain_pairwise_only')
        m = r['chemistry_model']
        self.assertEqual((m['slots_missing'], m['common_word_tables'], m['recorded_passing_pairs']), ([], ['T1'], 1))
        self.assertTrue(m['matches_recorded_best_control'])
        self.assertFalse(m['complete_model'])
        self.assertTrue(any('whole-assembly verification' in n for n in m['still_needs']))

    def test_short_chemistry_chain_lists_missing_slots(self):
        m = P.chain_report(self.cat, self.ids[:4])['chemistry_model']
        self.assertEqual(m['slots_missing'], list(range(4, 15)))
        self.assertTrue(any('all 15 slots' in n for n in m['still_needs']))
        self.assertFalse(m['matches_recorded_best_control'])

    def test_chain_without_common_table(self):
        r = P.chain_report(self.cat, ['k02-x', 'k98-notable'])
        self.assertEqual(r['chain_status'], 'chain_matches_without_known_table')
        self.assertIn('one word table compatible with every loader', r['chemistry_model']['still_needs'])

    def test_invsqrt_chain_is_weakest_pair(self):
        r = P.chain_report(self.cat, ['mod-a0', 'mod-b0'])
        self.assertEqual(r['chain_status'], 'local_all_x_cut_pass')
        self.assertEqual(r['named_whole_candidates'][0]['name'], 'HW-T')
        r = P.chain_report(self.cat, ['mod-a0', 'mod-a1'])
        self.assertEqual(r['chain_status'], 'exact_source_continuity')
        self.assertEqual(P.chain_report(self.cat, ['mod-a0', 'mod-b0', 'mod-c0'])['chain_status'], 'rejected_at_this_cut')

    def test_chain_bounds(self):
        with self.assertRaises(P.PartsError):
            P.chain_report(self.cat, ['mod-a0'])
        with self.assertRaises(P.PartsError):
            P.chain_report(self.cat, ['mod-a0'] * (P.MAX_CHAIN + 1))


class FakeApi:
    class QhdlError(Exception):
        def __init__(self, msg, hint=None):
            super().__init__(msg)
            self.hint = hint

    @staticmethod
    def rel(p):
        return str(p)

    @staticmethod
    def cmd(*parts):
        return 'qhdl ' + ' '.join(str(p) for p in parts)

    @staticmethod
    def tool_sha():
        return 'testtool'


class Args:
    def __init__(self, **kw):
        self.catalogue = None
        self.out = None
        self.all = False
        self.__dict__.update(kw)


class TestVerb(Base):
    def test_envelope_tier_rerun_and_bounds(self):
        env = P.v_parts(FakeApi, Args(catalogue=str(self.root), parts_cmd='compose', left='mod-a0', right='mod-b0'))
        d = json.loads(env.render())
        self.assertEqual(d['tier'], 'static')
        self.assertIn('--catalogue', d['rerun'])
        self.assertIn('scoped; not a verified join', d['claim'])
        self.assertLessEqual(len(env.render().encode()), E.MAX_BYTES)
        self.assertEqual(d['provenance'], 'recorded') if 'provenance' in d else None

    def test_rows_bounded(self):
        env = P.v_parts(FakeApi, Args(catalogue=str(self.root), parts_cmd='chain', ids=TestChain.ids))
        self.assertLessEqual(env.as_dict()['rows_shown'], P.MAX_ROWS)
        self.assertEqual(env.as_dict()['rows_total'], 14)

    def test_env_default_catalogue(self):
        old = os.environ.get('QHDL_PARTS_CATALOGUE')
        os.environ['QHDL_PARTS_CATALOGUE'] = str(self.root)
        try:
            env = P.v_parts(FakeApi, Args(parts_cmd='summary'))
            self.assertIn(self.cat.manifest_sha256[:12], env.d['claim'])
        finally:
            if old is None:
                del os.environ['QHDL_PARTS_CATALOGUE']
            else:
                os.environ['QHDL_PARTS_CATALOGUE'] = old

    def test_out_is_append_only(self):
        with tempfile.TemporaryDirectory() as o:
            a = Args(catalogue=str(self.root), parts_cmd='module', module_id='mod-b0', out=o)
            P.v_parts(FakeApi, a)
            with self.assertRaises(FakeApi.QhdlError):
                P.v_parts(FakeApi, a)

    def test_errors_become_qhdl_errors(self):
        with self.assertRaises(FakeApi.QhdlError):
            P.v_parts(FakeApi, Args(catalogue=str(self.root), parts_cmd='module', module_id='nope'))
        with self.assertRaises(FakeApi.QhdlError):
            P.v_parts(FakeApi, Args(catalogue=str(self.root / 'missing'), parts_cmd='summary'))

    def test_default_discovery_picks_newest_version(self):
        with tempfile.TemporaryDirectory() as d:
            for v in (2, 10, 9):
                build(Path(d) / 'outputs' / f'r-modular-parts-public-release-v{v}-2026-11-11')
            old = os.environ.pop('QHDL_PARTS_CATALOGUE', None)
            try:
                self.assertTrue(P.default_catalogue(Path(d)).name.startswith('r-modular-parts-public-release-v10-'))
            finally:
                if old is not None:
                    os.environ['QHDL_PARTS_CATALOGUE'] = old


if __name__ == '__main__':
    unittest.main()
