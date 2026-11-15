import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qhdl_strategy_eval import assess, digest, gap, leakage_groups, score


class StrategyEvaluationTest(unittest.TestCase):
    def test_geometric_mean_gap_is_cubed_for_single_task(self):
        g = gap(20, 22.8)
        self.assertAlmostEqual(g['one_task_cost_ratio'], 1/1.14**3)
        self.assertAlmostEqual(g['equal_each_task_cost_ratio'], 1/1.14)
        self.assertNotAlmostEqual(g['one_task_cost_ratio'], .86)

    def test_scores_reject_nan_boolean_number_and_duplicates(self):
        key = [dict(case_id='a', expected=1., absolute_tolerance=1e-8)]
        for answer in [True, float('nan'), float('inf'), None]:
            self.assertEqual(score(key, [dict(case_id='a', answer=answer, confidence=1)])['correct'], 0)
        with self.assertRaises(ValueError):
            score(key, [dict(case_id='a')]*2)
        with self.assertRaises(ValueError):
            score(key, [dict(case_id='foreign')])
        self.assertEqual(score(key, [])['diagnostic_accuracy'], 0)
        self.assertEqual(score(key, [dict(case_id='a', answer=1., confidence=float('nan'))])['correct'], 0)

    def test_false_boolean_answer_is_not_abstention(self):
        key = [dict(case_id='a', expected=False, absolute_tolerance=0)]
        got = score(key, [dict(case_id='a', answer=False, confidence=.9)])
        self.assertEqual(got['correct'], 1)
        self.assertEqual(got['attempted'], 1)
        self.assertIsNone(got['semantic_strategy_score'])

    def test_transitive_leakage_exclusions(self):
        def row(c, owner, factor=''):
            return dict(sha256=c*64, source_notebook_url=f'https://www.kaggle.com/code/{owner}/n', factorization_sha256=factor)
        catalog = [row('a','one'),row('b','one'),row('c','two','f'),row('d','three','f'),row('e','four')]
        episodes = [dict(child=dict(sha256='b'*64),parent_candidates=[dict(sha256='c'*64)])]
        g = leakage_groups(catalog, episodes)
        self.assertEqual(sorted(len(r['members']) for r in g['groups']), [1,4])
        self.assertEqual(g['split_status'], 'not assigned')

    def test_hash_receipt_and_large_claim_never_become_verified_gain(self):
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td)/'report.json'; evidence.write_text('{"claimed_pass": true}')
            p = dict(task='dqi', base_sha256='a'*64, mechanism='x',changed_interface='x',
                     preserved_outputs_and_phase='x',schedule_and_cleanup='x',falsification_test='x',
                     ratios=dict(R=1.,Q=2.,T=.1),evidence=[dict(path=str(evidence),sha256=digest(evidence))])
            got = assess(p, {'dqi':dict(sha256='a'*64)}, gap(20,22.8))
            self.assertAlmostEqual(got['projected_task_cost_ratio'], .4)
            self.assertTrue(got['formal_packet_complete'])
            self.assertFalse(got['validated_improvement'])
            self.assertEqual(got['semantic_review'], 'pending')
            evidence.write_text('changed')
            self.assertFalse(assess(p, {'dqi':dict(sha256='a'*64)},gap(20,22.8))['formal_packet_complete'])

    def test_partial_cost_model_and_wrong_base_are_blocked(self):
        p = dict(task='dqi',base_sha256='b',ratios=dict(Q=1,T=.5))
        got = assess(p, {'dqi':dict(sha256='a')},gap(20,22.8))
        self.assertIsNone(got['projected_task_cost_ratio'])
        self.assertIn('base SHA is not this frozen frontier',got['errors'])
        self.assertFalse(got['formal_packet_complete'])


if __name__ == '__main__':
    unittest.main()
