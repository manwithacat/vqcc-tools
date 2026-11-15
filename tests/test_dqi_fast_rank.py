"""Pure parts of scripts/dqi_fast_rank.py: R table, paired schedule, dT statistics, prediction, failure comparison
and sweep reading. The engine side (profile, rank) needs kickmix_engine and is not exercised here."""

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dqi_fast_rank", ROOT / "scripts" / "dqi_fast_rank.py")
fr = importlib.util.module_from_spec(spec)
sys.modules["dqi_fast_rank"] = fr
spec.loader.exec_module(fr)


def draw(t, ve=0, pe=0):
    return {"toffoli_mean": t, "value_errors": ve, "phase_errors": pe}


class RTable(unittest.TestCase):
    def test_table_matches_card(self):
        self.assertAlmostEqual(fr.r_of_k(0), 1.19126, places=5)
        self.assertAlmostEqual(fr.r_of_k(6), 1.2165566, places=6)  # verifier value; 1.21657 in a brief was a rounding slip
        self.assertTrue(all(fr.r_of_k(k) < fr.r_of_k(k + 1) for k in range(6)))

    def test_beyond_table_needs_verifier(self):
        with self.assertRaises(ValueError):
            fr.r_of_k(7)
        with self.assertRaises(ValueError):
            fr.r_of_k(-1)

    def test_verifier_object_is_used_when_given(self):
        class V:
            DEFAULT_LOG2_DELTA = -128.0

            @staticmethod
            def clopper_pearson_upper_bound(k, n, log2_delta):
                return k / n

            @staticmethod
            def repetition_overhead(eps):
                return 2.0 + eps
        self.assertEqual(fr.r_of_k(0, V()), 2.0)
        self.assertEqual(fr.r_of_k(9, V()), 2.0 + 9 / fr.SHOTS)


class Schedule(unittest.TestCase):
    def test_prefixes_nest_and_cover_every_draw(self):
        s64, s8 = fr.schedule(64), fr.schedule(8)
        self.assertEqual(s64[:8], s8)
        self.assertEqual({d for d, _ in s64}, set(range(48)))
        self.assertTrue(all(0 <= b < fr.BATCHES for _, b in s64))

    def test_no_repeated_batch_up_to_1200(self):
        s = fr.schedule(1200)
        self.assertEqual(len(set(s)), 1200)


class DeltaStats(unittest.TestCase):
    def test_constant_shift(self):
        base = [1000 * 256, 1003 * 256, 999 * 256]
        d = fr.delta_stats(base, [x - 294 * 256 for x in base])
        self.assertEqual(d["mean_per_shot"], -294.0)
        self.assertEqual(d["se_per_shot"], 0.0)
        self.assertTrue(d["constant"])
        self.assertEqual(d["shots"], 3 * 256)

    def test_varying_shift_has_standard_error(self):
        base = [0, 0, 0, 0]
        cand = [256 * x for x in (-2, -4, -2, -4)]  # batch means -2, -4, -2, -4
        d = fr.delta_stats(base, cand)
        self.assertEqual(d["mean_per_shot"], -3.0)
        self.assertAlmostEqual(d["se_per_shot"], math.sqrt((4 / 3) / 4))
        self.assertFalse(d["constant"])
        self.assertEqual(d["min_batch_mean"], -4.0)

    def test_single_batch_has_no_se(self):
        self.assertIsNone(fr.delta_stats([10], [8])["se_per_shot"])

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            fr.delta_stats([1, 2], [1])
        with self.assertRaises(ValueError):
            fr.delta_stats([], [])


class Predict(unittest.TestCase):
    def test_zero_shift_reproduces_the_base_cost(self):
        draws = [draw(400000.0), draw(400001.0, 1), draw(399999.0, 0, 2)]
        base_cost = sum(fr.r_of_k(d["value_errors"] + d["phase_errors"]) * 892 ** 2 * d["toffoli_mean"]
                        for d in draws) / 3
        p = fr.predict(draws, 0.0, 892)
        self.assertAlmostEqual(p["expected_cost"], base_cost, delta=1e-3)
        self.assertEqual(p["draws"], 3)

    def test_shift_and_qubits(self):
        draws = [draw(1000.0), draw(1000.0)]
        p = fr.predict(draws, -10.0, 10)
        self.assertAlmostEqual(p["expected_cost"], fr.R_TABLE[0] * 100 * 990.0)
        self.assertEqual(len(p["per_draw"]), 2)

    def test_rel_err(self):
        self.assertAlmostEqual(fr.rel_err(1.00001, 1.0), 1e-5)
        self.assertLessEqual(abs(fr.rel_err(1.0 + 5e-6, 1.0)), fr.REL_TIE)


class CompareFailures(unittest.TestCase):
    def test_identical_sets_match(self):
        b = {(3, 17): {"value": [5], "phase": []}, (40, 2): {"value": [], "phase": [200, 7]}}
        c = {(3, 17): {"value": [5], "phase": []}, (40, 2): {"value": [], "phase": [7, 200]}}
        r = fr.compare_failures(b, c)
        self.assertEqual((r["batches"], r["matched"], r["mismatches"]), (2, 2, []))

    def test_new_or_missing_failure_is_a_mismatch(self):
        b = {(3, 17): {"value": [5], "phase": []}}
        c = {(3, 17): {"value": [5], "phase": []}, (0, 3): {"value": [], "phase": [1]}}
        r = fr.compare_failures(b, c)
        self.assertEqual(r["matched"], 1)
        self.assertEqual([(m["draw"], m["batch"]) for m in r["mismatches"]], [(0, 3)])
        r2 = fr.compare_failures(b, {})
        self.assertEqual(r2["mismatches"][0]["candidate"], {"value": [], "phase": []})

    def test_type_change_keeps_the_set_and_is_reported(self):
        b = {(7, 142): {"value": [], "phase": [211]}}
        c = {(7, 142): {"value": [211], "phase": []}}  # same shot fails, value instead of phase: k_d unchanged
        r = fr.compare_failures(b, c)
        self.assertEqual((r["matched"], r["mismatches"]), (1, []))
        self.assertEqual([(t["draw"], t["batch"]) for t in r["type_changes"]], [(7, 142)])
        self.assertEqual(fr.compare_failures(b, b)["type_changes"], [])


class SweepReading(unittest.TestCase):
    def test_gate_json_and_csv(self):
        with tempfile.TemporaryDirectory() as td:
            g = Path(td) / "x.gate.json"
            rows = [{"draw": f"salt:sweep-{i}", "seed": str(i),
                     "candidate": {"score": 100.0 + i, "toffoli_count": 10.0, "value_errors": i % 2,
                                   "phase_errors": 0, "seconds": 2.0}} for i in range(4)]
            g.write_text(json.dumps({"rows": rows}))
            s = fr.full_sweep_cost(g)
            self.assertEqual(s["expected_cost"], 101.5)
            self.assertEqual(sum(s["failing_shots"]), 2)
            self.assertEqual(s["seconds"], 8.0)
            c = Path(td) / "draws_48.csv"
            c.write_text("salt,base_cost,base_value_errors,base_phase_errors,candidate_cost,candidate_value_errors,"
                         "candidate_phase_errors,dcost_pct\nsweep-0,1,0,0,200.0,0,1,0\nsweep-1,1,0,0,100.0,0,0,0\n")
            s = fr.full_sweep_cost(c)
            self.assertEqual(s["expected_cost"], 150.0)
            self.assertEqual(sum(s["failing_shots"]), 1)
            self.assertIsNone(s["seconds"])


class PreconditionPower(unittest.TestCase):
    """9fcda0c1aeb3 passed the old precondition at N = 16 (25 batches, 0.13% coverage) while failing 48 shots on
    96 draws against its base's 23. Recorded fixture: the rank report's precondition block and the 96-draw k vectors."""
    PRE = {"verdict": "PASS", "base_failing_batches": 9, "base_failing_shots": 9, "paired_batches": 16,
           "checked_batches": 25, "checked_shots": 6400, "total_shots": 4915200, "mismatches": [], "type_changes": []}
    K_PICK = [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0]
    K_CAND = [0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0]

    def test_old_logic_passes_but_k_differs(self):
        self.assertEqual(self.PRE["verdict"], "PASS")          # what the engine reported
        self.assertEqual([i for i, (a, b) in enumerate(zip(self.K_PICK, self.K_CAND)) if a != b], [8, 13])

    def test_new_verdict_is_underpowered(self):
        pw = fr.precondition_power(self.PRE)
        self.assertFalse(pw["powered"])
        self.assertGreater(pw["upper_extra_k_per_draw"], 10 * fr.MAX_UNDETECTED_EXTRA_K)
        self.assertEqual(fr.precondition_verdict(self.PRE), "UNDERPOWERED")

    def test_mismatch_fails_and_full_coverage_passes(self):
        self.assertEqual(fr.precondition_verdict(dict(self.PRE, mismatches=[{"draw": 1}])), "FAIL")
        full = dict(self.PRE, checked_shots=48 * fr.SHOTS, base_failing_shots=0)
        self.assertEqual(fr.precondition_verdict(full), "PASS")


if __name__ == "__main__":
    unittest.main()
