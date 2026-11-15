"""qhdl diff: synthetic circuits and a synthetic dataset directory only (no recorded evidence needed). NumPy needed."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
CCX, CX = 0x31, 0x21


def write(path, ops):
    """ops: list of (type, qubits tuple). Returns the path."""
    import numpy as np
    import qhdl_kmb as KM
    ot = np.array([o for o, _ in ops], dtype=np.uint8)
    s = {n: [] for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0']}
    for o, q in ops:
        if o & 0x30 == 0x30:
            for n, v in zip(['qqq0', 'qqq1', 'qqq2'], q):
                s[n].append(v)
        elif o & 0x30 == 0x20:
            for n, v in zip(['qq0', 'qq1'], q):
                s[n].append(v)
        elif o & 0x30 == 0x10:
            s['q0'].append(q[0])
    KM.write_kmb(str(path), ot, s, num_qubits=40)
    return str(path)


def body(n_sites, per_site):
    ops = []
    for s in range(n_sites):
        ops += [(CX, (s % 7, 8))] * 5
        ops += [(CCX, (s % 5, 6, 30 + k)) for k in range(per_site)]
        ops += [(CX, (9, s % 11))] * 200
    return ops


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class Align(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import qhdl_diff as D
        self.np, self.D = np, D

    def hashes(self, seq):
        return self.np.array(seq, dtype=self.np.uint64) * self.np.uint64(0x9E3779B97F4A7C15) + self.np.uint64(7)

    def test_identical_has_no_gaps(self):
        h = self.hashes(range(5000))
        self.assertEqual(self.D.align(h, h), ([], False))

    def test_insert_and_delete_gaps(self):
        a = list(range(10000))
        b = a[:3000] + [90001, 90002, 90003] + a[3000:7000] + a[7010:]
        gaps, unal = self.D.align(self.hashes(a), self.hashes(b))
        self.assertFalse(unal)
        self.assertEqual(gaps, [(3000, 3000, 3000, 3003), (7000, 7010, 7003, 7003)])

    def test_nearby_gaps_merge_into_one_site(self):
        a = list(range(4000))
        b = list(a)
        b[1000] = 50001
        b[1020] = 50002        # 19 matched ops between: one call site
        gaps, _ = self.D.align(self.hashes(a), self.hashes(b))
        self.assertEqual(len(gaps), 1)

    def test_unalignable_tail_is_flagged(self):
        a = list(range(3000))
        b = a[:1000] + list(range(70000, 73000))
        gaps, unal = self.D.align(self.hashes(a), self.hashes(b))
        self.assertTrue(unal)
        self.assertEqual(gaps[-1][1], 3000)
        self.assertEqual(gaps[-1][3], 4000)

    def test_hint_vocabulary_and_repeat_rule(self):
        mk = lambda o, n: dict(old=dict(toffoli=o), new=dict(toffoli=n))
        rows = [mk(5, 4)] * 9 + [mk(0, 0)] * 50
        tech, dom = self.D.hint_for_gaps(rows)
        self.assertEqual(tech, 'repeated-component')
        self.assertEqual((dom['old_toffoli'], dom['new_toffoli'], dom['sites']), (5, 4, 9))
        self.assertEqual(self.D.hint_for_gaps([mk(5, 4)] * 3)[0], 'component-resynthesis')
        self.assertEqual(self.D.hint_for_gaps([mk(1, 1)])[0], 'component-resynthesis')
        self.assertIn(self.D.hint_for_gaps(rows)[0], self.D.TECHNIQUES)


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class Circuits(unittest.TestCase):
    def setUp(self):
        import qhdl_diff as D
        import qhdl_kmb as KM
        self.D, self.KM = D, KM
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_op_hash_sensitive_to_operand_and_type_not_bit_ids(self):
        a = write(self.dir / 'a.kmb', [(CX, (1, 2)), (CCX, (1, 2, 3)), (CX, (4, 5))])
        b = write(self.dir / 'b.kmb', [(CX, (1, 2)), (CCX, (1, 2, 4)), (CX, (4, 5))])
        ha = self.D.op_hashes(self.KM.read(a), 0, 3)
        hb = self.D.op_hashes(self.KM.read(b), 0, 3)
        self.assertEqual(int(ha[0]), int(hb[0]))
        self.assertNotEqual(int(ha[1]), int(hb[1]))
        self.assertEqual(int(ha[2]), int(hb[2]))
        # a slice hashes like the same ops inside the whole circuit
        hs = self.D.op_hashes(self.KM.read(a), 1, 3)
        self.assertEqual([int(x) for x in hs], [int(x) for x in ha[1:]])

    def test_gap_signature_counts_toffolis_per_site(self):
        a = write(self.dir / 'a.kmb', body(12, 5))
        b = write(self.dir / 'b.kmb', body(12, 4))
        ka, kb = self.KM.read(a), self.KM.read(b)
        ha, hb = self.D.op_hashes(ka, 0, len(ka.op_types)), self.D.op_hashes(kb, 0, len(kb.op_types))
        gaps, unal = self.D.align(ha, hb)
        self.assertFalse(unal)
        sigs = self.D.gap_signatures(ka, kb, gaps)
        self.assertEqual(sum(s['new']['toffoli'] - s['old']['toffoli'] for s in sigs), -12)
        tech, dom = self.D.hint_for_gaps(sigs)
        self.assertEqual(tech, 'repeated-component')
        self.assertEqual((dom['old_toffoli'], dom['new_toffoli'], dom['sites']), (1, 0, 12))

    def test_self_diff_through_the_cli_reports_zero_changed(self):
        a = write(self.dir / 'a.kmb', body(3, 2))
        env = dict(os.environ, QHDL_WORK=str(self.dir / 'work'))
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'diff', '-c', a, '-c2', a, '--out', str(self.dir / 'o')],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual(d['tier'], 'static')
        self.assertIn('0 changed components', d['claim'])
        self.assertTrue((self.dir / 'o' / 'diff.json').exists())

    def test_task_unknown_is_an_error_not_a_guess(self):
        a = write(self.dir / 'a.kmb', body(3, 2))
        b = write(self.dir / 'b.kmb', body(3, 1))
        env = dict(os.environ, QHDL_WORK=str(self.dir / 'work'))
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/qhdl.py'), 'diff', '-c', a, '-c2', b],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2)
        self.assertIn('--task', r.stdout)


class Chemistry(unittest.TestCase):
    def build(self, root, fams, parts):
        (root / 'parts').mkdir(parents=True)
        for name, rows in (('families.csv', fams), ('angle_circuits.csv', parts)):
            with open(root / 'parts' / name, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        with open(root / 'manifest.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['task', 'sha256', 'sha12', 'author', 'source_notebook_url', 'source_notebook_version', 'public_since'])
            w.writeheader()

    def test_per_slot_deltas_flags_and_zero_self_diff(self):
        import qhdl_diff as D
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parts, fams = [], []
            for fam, st, npz, lam in (('famaaaaaaaaa', [10, 10, 7], 'n1', '5.0'), ('fambbbbbbbbb', [8, 10, 7], 'n2', '5.0')):
                ids = []
                for k, t in enumerate(st):
                    pid = f'{fam}-{k}'
                    ids.append(pid)
                    parts.append(dict(part_id=pid, file_sha256=f'f{fam}{k}' if t != 10 else f'same{k}', slot_k=k, words_sha256=f'w{k}',
                                      prev_words_sha256=f'w{k - 1}', num_qubits=100, static_toffoli=t, toffoli_per_shot=2 * t))
                fams.append(dict(family=fam, parts=';'.join(ids), T_official=sum(2 * t for t in st) + 100, lambda_official=lam,
                                 Q_official=100, table_key='T', npz_sha256=npz, cost_official=1000.0))
            self.build(root, fams, parts)
            r = D.diff_chem(None, 'famaaaaaaaaa', 'fambbbbbbbbb', root)
            self.assertEqual(r['whole_delta'], -4)
            self.assertEqual(r['component_sum'], -4)
            self.assertEqual(r['unmatched'], 0)
            self.assertEqual([(x['slot'], x['delta_per_pass']) for x in r['rows'] if x['delta_per_pass']], [(0, -2)])
            self.assertTrue(r['flags']['npz_changed'] and not r['flags']['table_changed'] and not r['flags']['words_changed'])
            self.assertEqual(set(r['hints']), {'component-resynthesis', 'weights-refit'})
            s = D.diff_chem(None, 'famaaaaaaaaa', 'famaaaaaaaaa', root)
            self.assertEqual((s['rows'], s['whole_delta'], s['hints']), ([], 0, []))


if __name__ == '__main__':
    unittest.main()


class TaglessDqiTest(unittest.TestCase):
    """A file with no INFO_TAG regions must not be read as 'every region removed'."""

    def test_tagless_refuses_region_attribution(self):
        import qhdl_diff as D

        class Api:
            WORK = '/tmp'
            @staticmethod
            def rel(p):
                return str(p)

        def tree(regions, tof):
            return {'tag_regions': regions, 'metrics': {'toffoli': tof, 'hmr': 5, 'conditional_toffoli': 0, 'ops': 100},
                    'circuit': {'qubits': 892}}
        tagged = tree([{'path': 'unextract', 'depth': 0, 'start': 0, 'end': 10,
                        'metrics': {'toffoli': 1000, 'conditional_toffoli': 0, 'ops': 100, 'hmr': 5}}], 1000)
        bare = tree([], 900)
        orig = D.regions_for
        D.regions_for = lambda api, path, sha12, force=False: (tagged if sha12 == 'a' * 12 else bare, 'x.json')
        try:
            res = D.diff_dqi(Api, {'sha': 'a' * 64, 'path': 'p'}, {'sha': 'b' * 64, 'path': 'q'})
        finally:
            D.regions_for = orig
        self.assertEqual(res['unmatched'], 0)
        self.assertEqual(len(res['rows']), 1)
        self.assertEqual(res['rows'][0]['delta'], -100)
        self.assertEqual(res['rows'][0]['tagless'], ['b' * 12])
