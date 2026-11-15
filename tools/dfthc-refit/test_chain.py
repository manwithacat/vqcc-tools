# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
"""Tests for chain.py: command order, out-dir refusal, plan-screen choice, replan adoption and the
REL_TIE refit acceptance, using a fake dfthc_refit that writes canned reports (no planners, no kickmix).

    python3 -I -B test_chain.py -v
"""
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chain  # noqa: E402

FAKE = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    args = sys.argv[1:]
    cfg = json.loads(Path(__file__).with_name("fake_cfg.json").read_text())
    def opt(name):
        return args[args.index(name) + 1] if name in args else None
    cmd = args[0]
    if cmd == "conventions":
        Path(opt("--out")).write_text("{}")
    elif cmd == "recover":
        d = Path(opt("--out")); d.mkdir(parents=True)
        (d / "recover.json").write_text(json.dumps({"per_slot": {"0": {"num_toffoli_like": 300}}}))
    elif cmd == "lowcache":
        Path(opt("--out")).write_text(json.dumps({"floor_total": 107, "gap_total": 0,
                                                  "changing_columns_untouchable": [51]}))
    elif cmd == "plan":
        p = opt("--planner"); d = Path(opt("--out")); d.mkdir(parents=True)
        sv = cfg["plan"][p]
        (d / "plans.json").write_text(json.dumps({"slots": [{"k": k, "saving_vs_shipped": s} for k, s in sv.items()],
                                                  "toffolis_per_pass": {"saving": sum(sv.values())}}))
    elif cmd == "replan":
        d = Path(args[2]); (d / "dfthcsos").mkdir(parents=True)
        (d / "dfthcsos" / "submission.npz").write_text("x")
        (d / "report.json").write_text(json.dumps({"toffolis_per_pass": {"saving": cfg["replan_saving"]},
                                                   "cost": {}, "verify_after": {"status": "PASS"}}))
    elif cmd == "refit":
        d = Path(args[2]); (d / "dfthcsos").mkdir(parents=True)
        (d / "dfthcsos" / "submission.npz").write_text("x")
        i = int(d.name.split("_")[1]) - 1
        rel = cfg["refit_rel"][i]
        (d / "report.json").write_text(json.dumps({"before": {"lambda_eff": 1.0}, "after": {"lambda_eff": 1.0 + rel},
                                                   "stop": "x", "outputs_sha256": {"submission.npz": "h"}}))
    with open(Path(__file__).with_name("calls.log"), "a") as f:
        f.write(" ".join(args[:3]) + "\\n")
''')


class ChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fam = self.tmp / "fam"
        self.fam.mkdir()
        (self.fam / "submission.npz").write_text("x")
        self.fake = self.tmp / "fake_tool.py"
        self.fake.write_text(FAKE)
        self.orig = chain.TOOL
        chain.TOOL = self.fake

    def tearDown(self):
        chain.TOOL = self.orig

    def cfg(self, **kw):
        c = dict(plan={"shared": {"4": 2, "5": 0}, "block": {"0": 1}}, replan_saving=2, refit_rel=[-5e-5, -1e-6, -1e-4])
        c.update(kw)
        (self.tmp / "fake_cfg.json").write_text(json.dumps(c))

    def run_chain(self, *extra):
        out = self.tmp / "out"
        rc = chain.main([str(self.fam), str(out), "--verifier-dir", "V", *extra])
        return rc, json.loads((out / "chain.json").read_text())

    def test_full_chain_choices(self):
        self.cfg()
        rc, s = self.run_chain()
        self.assertEqual(rc, 0)
        names = [st["step"] for st in s["steps"]]
        self.assertEqual(names, ["conventions", "recover", "lowcache", "plan_shared", "plan_block", "replan_shared",
                                 "refit_1", "refit_2"])
        self.assertEqual(s["lowcache"]["gap_total"], 0)
        self.assertEqual(s["replan"]["slots"], ["4"] if isinstance(s["replan"]["slots"][0], str) else [4])
        self.assertTrue(s["refit"][0]["accepted"])          # -5e-5 beats REL_TIE
        self.assertFalse(s["refit"][1]["accepted"])         # -1e-6 does not: chain stops
        self.assertTrue(s["final_family_dir"].endswith("refit_1/dfthcsos"))
        self.assertTrue(s["changed"])
        replan_cmd = s["steps"][5]["cmd"]
        self.assertIn("--verify", replan_cmd)
        self.assertIn("--exchange", replan_cmd)
        self.assertIn("--gauge-free-sources", replan_cmd)          # 0.2.0 default
        self.assertIn("--gauge-free-sources", s["steps"][3]["cmd"])

    def test_no_free_sources_flag(self):
        self.cfg()
        rc, s = self.run_chain("--no-free-sources", "--refit-runs", "0")
        self.assertEqual(rc, 0)
        self.assertFalse(any("--gauge-free-sources" in st["cmd"] for st in s["steps"]))

    def test_no_saving_skips_replan_and_unchanged(self):
        self.cfg(plan={"shared": {"4": 0}, "block": {"0": 0}}, refit_rel=[-1e-7])
        rc, s = self.run_chain()
        self.assertEqual(rc, 0)
        self.assertNotIn("replan_shared", [st["step"] for st in s["steps"]])
        self.assertIn("skipped", s["replan"])
        self.assertFalse(s["changed"])

    def test_dry_run_and_refusal(self):
        self.cfg()
        rc, s = self.run_chain("--dry-run", "--planners", "none", "--refit-runs", "1")
        self.assertEqual(rc, 0)
        self.assertEqual(s["status"], "dry-run")
        self.assertFalse((self.tmp / "calls.log").exists())
        rc2 = chain.main([str(self.fam), str(self.tmp / "out"), "--verifier-dir", "V"])
        self.assertEqual(rc2, 2)                            # out dir not empty: refused

    def test_failed_step_stops(self):
        self.cfg()
        self.fake.write_text("import sys; sys.exit(3)\n")
        rc, s = self.run_chain()
        self.assertEqual(rc, 1)
        self.assertTrue(s["status"].startswith("stopped: conventions failed"))
        self.assertEqual(len(s["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
