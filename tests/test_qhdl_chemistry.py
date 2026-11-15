"""Independent arithmetic, boundary and evidence regressions for family view."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qhdl_chemistry.py"
spec = importlib.util.spec_from_file_location("qhdl_chemistry", SCRIPT)
chemistry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chemistry)


def fixture():
    return {"id": "synthetic", "num_word_blocks": 15,
            "baseline": {"brot": 8, "batch_size": 5, "selection_qubits": 11,
                         "loader_qubits": 128, "prep_persistent": 85,
                         "prep_temporary": 303, "nonangle_toffolis": 6741,
                         "angle_toffolis_per_pass": 4746, "lambda_eff": 47,
                         "errors": [0.014, 0.0199], "tolerance": 0.02},
            "proposals": []}


def proposal(overrides=None, changed=None, status="unchanged", name="candidate"):
    return {"id": name, "overrides": overrides or {}, "changed_word_blocks": changed or [],
            "numerical_status": status}


class ChemistryTests(unittest.TestCase):
    def test_nonbinding_scratch_does_not_save_full_qubits(self):
        case = fixture()
        case["proposals"] = [proposal({"loader_qubits": 100})]
        untouched = copy.deepcopy(case)
        result = chemistry.analyze(case)
        self.assertEqual(case, untouched)
        self.assertEqual(result["baseline"]["Q_total"], 428)
        self.assertEqual(result["baseline"]["T_total"], 16233)
        self.assertEqual(result["baseline"]["cost"], 47 * 428 * 16233)
        self.assertEqual(result["proposals"][0]["cost_delta"], 0)

    def test_batch_benefit_and_gate_loss(self):
        case = fixture()
        case["proposals"] = [proposal({"batch_size": 3, "loader_qubits": 124}, name="small"),
                             proposal({"batch_size": 3, "loader_qubits": 124,
                                       "angle_toffolis_per_pass": 6000}, name="expensive")]
        good, bad = chemistry.analyze(case)["proposals"]
        self.assertEqual(good["Q_total"], 412)
        self.assertTrue(good["eligible_lower_cost"])
        self.assertAlmostEqual(good["maximum_angle_toffolis_per_pass_at_equal_cost"],
                               (428 * 16233 / 412 - 6741) / 2)
        self.assertAlmostEqual(good["residual_margins"][1], 0.0001)
        self.assertEqual(bad["Q_total"], 412)
        self.assertFalse(bad["lower_cost"])

    def test_binding_workspace_and_cap(self):
        case = fixture()
        case["baseline"]["prep_temporary"] = 50
        case["proposals"] = [proposal({"loader_qubits": 120}, name="smaller"),
                             proposal({"loader_qubits": 129}, name="too-wide")]
        result = chemistry.analyze(case)
        self.assertEqual(result["baseline"]["Q_total"], 202)
        self.assertTrue(result["baseline"]["loader_workspace_binding"])
        self.assertEqual(result["proposals"][0]["Q_total"], 194)
        self.assertFalse(result["proposals"][1]["eligible_for_further_validation"])

    def test_boundary_indices_against_independent_word_xor(self):
        # Exhaust all subsets; same mask 0b101 applied to otherwise arbitrary words.
        for subset in range(1 << 6):
            indices = [i for i in range(6) if subset >> i & 1]
            before = [2, 7, 4, 3, 0, 6]
            after = [word ^ (5 if i in indices else 0) for i, word in enumerate(before)]
            def deltas(words):
                return [word ^ (words[i - 1] if i else 0) for i, word in enumerate(words)]
            expected = [i for i, (a, b) in enumerate(zip(deltas(before), deltas(after))) if a != b]
            self.assertEqual(chemistry.affected_delta_blocks(indices, 6), expected)
        self.assertEqual(chemistry.affected_delta_blocks([5], 6), [5])
        self.assertEqual(chemistry.affected_delta_blocks([3, 4, 5], 6), [3])
        self.assertEqual(chemistry.affected_delta_blocks([2, 2], 6), [2, 3])

    def test_numerical_unknown_fail_and_threshold(self):
        case = fixture()
        case["proposals"] = [proposal({"angle_toffolis_per_pass": 1}, [3], "unknown", "unknown"),
                             proposal({"angle_toffolis_per_pass": 1}, [3], "fail", "failed"),
                             proposal({"errors": [0.02, 0.0200000001]}, [3], "pass", "over"),
                             proposal({"errors": [0.02, 0.02]}, [3], "pass", "at")]
        rows = chemistry.analyze(case)["proposals"]
        for row in rows[:3]:
            self.assertFalse(row["eligible_for_further_validation"])
        self.assertTrue(rows[0]["lower_cost"])
        self.assertEqual(rows[0]["accuracy_status"], "unknown")
        self.assertEqual(rows[1]["accuracy_status"], "fail")
        self.assertTrue(rows[3]["eligible_for_further_validation"])

    def test_invalid_quantities_and_widths(self):
        cases = [("lambda_eff", float("nan")), ("lambda_eff", float("inf")),
                 ("lambda_eff", True), ("lambda_eff", 0), ("brot", 0),
                 ("batch_size", False), ("batch_size", 2.5), ("loader_qubits", 50),
                 ("selection_qubits", -1), ("prep_temporary", -1),
                 ("nonangle_toffolis", -1), ("errors", [0.01, True]),
                 ("errors", [0.01]), ("tolerance", -1), ("loader_qubits", 10 ** 1000),
                 ("lambda_eff", 1e308)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                case = fixture()
                case["baseline"][key] = value
                with self.assertRaises(ValueError):
                    chemistry.analyze(case)
        for indices in ([True], [-1], [15], [1.0], "2"):
            with self.assertRaises(ValueError):
                chemistry.affected_delta_blocks(indices, 15)

    def test_report_extraction_retains_raw_residuals(self):
        report = {"results": {"dfthcsos": {"summary": {
            "status": "PASS", "brot": 8, "batch_size": 5, "num_circuits": 15,
            "num_qubits": 428, "toffoli_count": 16233, "score": 47 * 428 * 16233,
            "resources": {"angles_selection_qubits": 11, "angles_qubits": 128,
                          "heuristic_persistent_qubits": 85, "heuristic_temporary_qubits": 303,
                          "heuristic_toffoli": 6741, "angles_toffoli_per_pass": 4746},
            "metrics": {"lambda_eff": 47, "error_1": 0.01400000000003, "error_2": 0.0199999999}}}}}
        case = chemistry.report_to_case(report)
        self.assertEqual(case["baseline"]["errors"], [0.01400000000003, 0.0199999999])
        self.assertEqual(case["num_word_blocks"], 15)
        report["results"]["dfthcsos"]["summary"]["score"] += 100
        with self.assertRaisesRegex(ValueError, "disagrees"):
            chemistry.report_to_case(report)

    def test_cli_hashes_scope_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "case.json", Path(directory) / "view.json"
            source.write_text(json.dumps(fixture()))
            command = [sys.executable, str(SCRIPT), "--case", str(source), "--out", str(output)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertLess(len(run.stdout), 4096)
            result = json.loads(output.read_text())
            self.assertEqual(result["provenance"]["input_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(result["provenance"]["tool_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            self.assertFalse(result["provenance"]["verifier_run"])
            self.assertIn("not bytes-derived", result["scope"])
            frozen = output.read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(rerun.returncode, 2)
            self.assertEqual(output.read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
