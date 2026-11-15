# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for `dfthc_refit.py replan --plans-from` (plan replay, v1.7.0-dev).

Pure tests (plan-file layouts, the selection rule, CLI refusals) run anywhere with NumPy and SciPy. The integration
tests need a kickmix build, Lera's tensorqrom.py, the competition verifiers, the refit family 9706e2c7ecff
(DFTHC_REFIT_9706) and a replan output directory with saved shared-planner plans for slots 8-14
(DFTHC_REFIT_9706_PLANS); they skip when any is missing (see testpaths.py) and run in about a minute:

    "$QHDL_KICKMIX_PYTHON" -I -m unittest discover -s tools/dfthc-refit -p test_replay.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfthc_refit as dr  # noqa: E402

from testpaths import LERA, VERIFIERS, dataset_family, env_path  # noqa: E402

FAM_9706 = env_path("DFTHC_REFIT_9706")             # family 9706e2c7ecff (a weight refit of aba448aeadb5)
SAVED_PLANS = env_path("DFTHC_REFIT_9706_PLANS")    # a replan OUT_DIR with work/plans/k8 ... k14
SIGNED = dataset_family("DFTHC_REFIT_6566", "6566cd395128")
EXPECTED_KMB = {8: "6ec5b170e22c", 14: "725ca30bd967"}   # family 81db735f32b5's loaders 8 and 14 (gauge-aware)

try:
    import kickmix as km  # noqa: F401
    HAVE_KM = True
except ImportError:
    HAVE_KM = False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TestReplayPure(unittest.TestCase):
    def write(self, d, name, **kp):
        p = Path(d) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(kp))
        return p

    def test_layouts(self):
        with tempfile.TemporaryDirectory() as t:
            self.write(t, "work/plans/k3/shared_s0.json", terms=[], products=0, exact=True)
            self.write(t, "work/plans/k3/build.json", terms=[])          # never a plan
            self.assertEqual([f.name for f in dr.replay_plan_files(t, 3)], ["shared_s0.json"])
            self.assertEqual(dr.replay_plan_files(Path(t) / "work", 3)[0].name, "shared_s0.json")
            self.assertEqual(dr.replay_plan_files(Path(t) / "work" / "plans", 3)[0].name, "shared_s0.json")
            self.assertEqual(dr.replay_plan_files(t, 4), [])

    def test_choice_rule(self):
        """Fewest products; ties to the first file in sorted order; non-exact plans are skipped."""
        t3 = [[1, 1, 1, 1, 1]] * 3
        t2 = [[1, 1, 1, 1, 1]] * 2
        with tempfile.TemporaryDirectory() as t:
            fs = [self.write(t, "k0/shared_s0.json", terms=t3, products=3, exact=True),
                  self.write(t, "k0/shared_s1.json", terms=t2, products=2, exact=True),
                  self.write(t, "k0/shared_s2.json", terms=t2, products=2, exact=True),
                  self.write(t, "k0/shared_s3.json", terms=[[1, 1, 1, 1, 1]], products=1, exact=False),
                  self.write(t, "k0/shared_s4.json", terms=None, products=None, exact=False)]
            f, kp, skipped = dr.replay_choice(sorted(fs))
            self.assertEqual(f.name, "shared_s1.json")
            self.assertEqual(len(kp["terms"]), 2)
            self.assertEqual(sorted(s["file"] for s in skipped), ["shared_s3.json", "shared_s4.json"])
            self.assertEqual(dr.replay_choice([fs[3]])[0], None)

    def test_cli_refuses_planner_options(self):
        with tempfile.TemporaryDirectory() as t:
            for extra in (["--seeds", "0"], ["--backend", "shared"], ["--cpu-cap", "90"], ["--pad-rows"]):
                with self.assertRaises(SystemExit):
                    dr.replan_main([t, str(Path(t) / "out"), "--plans-from", t, "--lera-src", t] + extra)


@unittest.skipUnless(HAVE_KM and (LERA / "tensorqrom.py").is_file() and (VERIFIERS / "dfthcsos").is_dir()
                     and (FAM_9706 / "submission.npz").is_file() and (SAVED_PLANS / "work" / "plans" / "k8").is_dir(),
                     "needs kickmix, Lera's tensorqrom.py (DFTHC_REFIT_LERA_SRC), the verifiers (QHDL_OFFICIAL_ROOT), "
                     "family 9706e2c7ecff (DFTHC_REFIT_9706) and saved plans (DFTHC_REFIT_9706_PLANS)")
class TestReplay81db(unittest.TestCase):
    def test_reemits_81db_loaders(self):
        """Saved plans for slots 8 and 14 (14 changes a word's gauge) replay on family 9706e2c7ecff to family
        81db735f32b5's loader bytes; the never-worse family checks pass; the npz is copied unchanged."""
        with tempfile.TemporaryDirectory() as t:
            rep = dr.replan(FAM_9706, Path(t) / "out", lera_src=LERA, verifier_dir=VERIFIERS, slots="8,14",
                            verbose=False, plans_from=SAVED_PLANS)
            self.assertEqual(rep["planner"]["backend"], "replay")
            self.assertEqual(rep["changed_slots"], [8, 14])
            self.assertTrue(rep["family_checks"]["family_ok"])
            for k, want in EXPECTED_KMB.items():
                self.assertEqual(sha(Path(t) / "out" / "dfthcsos" / f"angle_k_{k}.kmb")[:12], want)
            self.assertEqual(rep["outputs_sha256"]["submission.npz"], rep["inputs_sha256"]["submission.npz"])
            self.assertEqual(rep["slots"][8]["replanned"]["replayed_plan"]["chosen"], "shared_s2.json")

    @unittest.skipUnless((SIGNED / "submission.npz").is_file(), "needs the 6566cd395128 family (QHDL_DATASET)")
    def test_wrong_family_is_refused(self):
        """A plan made for other words is not emitted: the slot keeps its shipped loader and the family is copied."""
        with tempfile.TemporaryDirectory() as t:
            rep = dr.replan(SIGNED, Path(t) / "out", lera_src=LERA, verifier_dir=VERIFIERS, slots="8",
                            verbose=False, plans_from=SAVED_PLANS)
            self.assertEqual(rep["changed_slots"], [])
            self.assertFalse(rep["planner"]["per_slot"]["8"]["exact_on_this_family"])
            self.assertTrue(rep["slots"][8]["reason"].startswith("replay:"))
            self.assertEqual(rep["output_family_sha256"], rep["input_family_sha256"])


if __name__ == "__main__":
    unittest.main()
