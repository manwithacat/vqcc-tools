#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""dfthc_refit chain: the whole per-family recipe as one reproducible command.

Runs, on any dfthcsos family, the README's "Processing a new public family" recipe in order and writes one summary:

    conventions -> recover -> lowcache (0.2.0) -> plan screen (per planner) -> replan (best planner, slots with a
      predicted saving) -> weight refit, chained while each run gains more than REL_TIE -> summary (chain.json)

    python3 -I chain.py FAMILY_DIR OUT_DIR --verifier-dir V [--kickmix-python K] [--torch-python P]
        [--lera-src L] [--planners shared,block | none] [--refit-runs 3] [--refit-rounds 10] [--refit-evals 1000]
        [--threads 3] [--kickmix-dir D] [--no-free-sources] [--dry-run]

0.2.0: a `lowcache` step (free gauge sources, the proved low-cache floor and the shipped gap per slot),
and plan/replan run with `--gauge-free-sources` by default, so the tool's plans pay the same cache price as loaders built
with the incoming gauge monomials as low sources (Syed Asad Ali's construction).

Two interpreters, as the README says: K (a kickmix build, no PyTorch) for conventions/recover/plan/replan, and P
(NumPy, SciPy, PyTorch) for the refit. Every step is a subprocess of dfthc_refit.py beside this file, with its stdout
and stderr kept under OUT_DIR/logs/. OUT_DIR must be new (or empty); nothing is written elsewhere. A step that fails
stops the chain and is recorded. Acceptance never weakens the tool's own rules: replan keeps a slot only on a strict
gain and checks the family; a refit run is accepted only if its lambda_eff gain exceeds REL_TIE (1e-5 relative), and
the chain stops at the first run that does not. --dry-run writes the command list and runs nothing.

Evidence tiers in chain.json are the tools' own: plan = static model / projection; replan "after" = the official
evaluate_circuits until its --verify passes locally; refit lambda_eff = official formulas. Only a hosted run of the
unmodified verifier is an official verifier PASS.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "dfthc_refit.py"
REL_TIE = 1e-5
__version__ = "0.2.0"


def default_python(env: str | None = None) -> str:
    """$env when set (e.g. QHDL_KICKMIX_PYTHON: a Python that can import kickmix), else this interpreter."""
    import os
    return (os.environ.get(env) if env else None) or sys.executable


class Chain:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out_dir)
        self.logs = self.out / "logs"
        self.summary = dict(tool=f"dfthc_refit chain {__version__}", family_dir=str(a.family_dir),
                            out_dir=str(self.out), rel_tie=REL_TIE, dry_run=bool(a.dry_run), steps=[])

    # ---------------------------------------------------------------- plumbing
    def run(self, name: str, py: str, args: list, report: Path | None = None) -> dict | None:
        cmd = [py, "-I", str(TOOL), *map(str, args)]
        step = dict(step=name, cmd=cmd)
        self.summary["steps"].append(step)
        if self.a.dry_run:
            step["status"] = "dry-run"
            return None
        t0 = time.time()
        with open(self.logs / f"{name}.stdout", "w") as fo, open(self.logs / f"{name}.stderr", "w") as fe:
            rc = subprocess.run(cmd, stdout=fo, stderr=fe).returncode
        step.update(rc=rc, wall_s=round(time.time() - t0, 1))
        if rc != 0:
            step["status"] = "failed"
            raise ChainStop(f"{name} failed (rc {rc}); see logs/{name}.stderr")
        step["status"] = "ok"
        if report is not None and report.is_file():
            step["report"] = str(report)
            return json.loads(report.read_text())
        return None

    def common(self) -> list:
        v = ["--verifier-dir", self.a.verifier_dir]
        return v

    # ---------------------------------------------------------------- steps
    def go(self) -> dict:
        a = self.a
        fam = Path(a.family_dir)
        K, P = a.kickmix_python, a.torch_python
        try:
            self.run("conventions", K, ["conventions", fam, *self.common(), "--out", self.out / "conventions.json",
                                        "--quiet"], self.out / "conventions.json")
            rec = self.run("recover", K, ["recover", fam, *self.common(), "--out", self.out / "recover", "--quiet"],
                           self.out / "recover" / "recover.json")
            if rec:
                self.summary["shipped_toffolis_per_slot"] = {
                    k: s.get("num_toffoli_like") for k, s in rec.get("per_slot", {}).items()}
            lc = self.run("lowcache", K, ["lowcache", fam, *self.common(), "--out", self.out / "lowcache.json",
                                          "--quiet"], self.out / "lowcache.json")
            if lc:
                self.summary["lowcache"] = {x: lc.get(x) for x in ("floor_total", "gap_total",
                                                                  "changing_columns_untouchable")}
            free = [] if a.no_free_sources else ["--gauge-free-sources"]
            cur = fam
            best = None
            planners = [] if a.planners in ("none", "") else a.planners.split(",")
            for p in planners:
                extra = ["--pad-rows"] if p in ("shared", "block") else []
                if p == "shared":
                    extra += ["--exchange"]
                rep = self.run(f"plan_{p}", K, ["plan", fam, "--planner", p, *extra, *free, *self.common(),
                                                "--out", self.out / f"plan_{p}", "--quiet"],
                               self.out / f"plan_{p}" / "plans.json")
                if rep is None:
                    continue
                slots = [r["k"] for r in rep.get("slots", []) if (r.get("saving_vs_shipped") or 0) > 0]
                saving = (rep.get("toffolis_per_pass") or {}).get("saving") or 0
                self.summary.setdefault("plan_screen", {})[p] = dict(predicted_saving_per_pass=saving, slots=slots)
                if slots and (best is None or saving > best[1]):
                    best = (p, saving, slots)
            if best is not None:
                p, _, slots = best
                extra = ["--pad-rows"] + (["--exchange"] if p == "shared" else [])
                if a.lera_src:
                    extra += ["--lera-src", a.lera_src]
                rep = self.run(f"replan_{p}", K, ["replan", cur, self.out / f"replan_{p}", "--backend", p,
                                                  "--slots", ",".join(map(str, slots)), "--gauge-inplace", "auto",
                                                  *extra, *free, *self.common(), "--verify", "--quiet"],
                               self.out / f"replan_{p}" / "report.json")
                if rep:
                    tpp = rep.get("toffolis_per_pass") or {}
                    self.summary["replan"] = dict(planner=p, slots=slots, toffolis_per_pass=tpp,
                                                  cost=rep.get("cost"), verify_after=rep.get("verify_after"))
                    if (tpp.get("saving") or 0) > 0:
                        cur = self.out / f"replan_{p}" / "dfthcsos"
            elif planners and not a.dry_run:
                self.summary["replan"] = "skipped: no planner predicts a saving on any slot"
            runs = []
            for i in range(a.refit_runs):
                name = f"refit_{i + 1}"
                args = ["refit", cur, self.out / name, *self.common(), "--arm", "B", "--rounds", a.refit_rounds,
                        "--max-evals", a.refit_evals, "--threads", a.threads, "--seed", 0]
                if a.kickmix_dir:
                    args += ["--kickmix-dir", a.kickmix_dir]
                if a.gauge_procs:
                    args += ["--gauge-procs", a.gauge_procs]
                rep = self.run(name, P, args, self.out / name / "report.json")
                if rep is None:
                    break
                lb, la = rep["before"]["lambda_eff"], rep["after"]["lambda_eff"]
                rel = la / lb - 1
                accepted = rel < -REL_TIE
                runs.append(dict(run=name, lambda_before=lb, lambda_after=la, rel=rel, stop=rep.get("stop"),
                                 accepted=accepted, npz_sha256=rep.get("outputs_sha256", {}).get("submission.npz"),
                                 verify_after=rep.get("verify_after")))
                if not accepted:
                    break
                cur = self.out / name / "dfthcsos"
            self.summary["refit"] = runs
            self.summary["final_family_dir"] = str(cur)
            self.summary["changed"] = Path(cur) != fam
            self.summary["status"] = "dry-run" if a.dry_run else "done"
        except ChainStop as e:
            self.summary["status"] = f"stopped: {e}"
        return self.summary


class ChainStop(RuntimeError):
    pass


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--verifier-dir", required=True)
    ap.add_argument("--kickmix-python", default=default_python("QHDL_KICKMIX_PYTHON"),
                    help="interpreter with a kickmix build (conventions, recover, plan, replan)")
    ap.add_argument("--torch-python", default=default_python(), help="interpreter with NumPy, SciPy, PyTorch (refit)")
    ap.add_argument("--kickmix-dir", help="passed to refit for word recovery and verify_after")
    ap.add_argument("--lera-src", help="directory holding Lera's tensorqrom.py (replan)")
    ap.add_argument("--planners", default="shared,block", help="comma list for the plan screen, or 'none'")
    ap.add_argument("--refit-runs", type=int, default=3, help="maximum chained refit runs (0: no refit)")
    ap.add_argument("--refit-rounds", type=int, default=10)
    ap.add_argument("--refit-evals", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--gauge-procs", type=int, default=None)
    ap.add_argument("--no-free-sources", action="store_true",
                    help="plan/replan without --gauge-free-sources (the 0.1.0 behaviour: full low cache)")
    ap.add_argument("--dry-run", action="store_true", help="write the command list only")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"refusing: {out} exists and is not empty (outputs are append-only)", file=sys.stderr)
        return 2
    if not Path(a.family_dir, "submission.npz").is_file():
        print(f"refusing: {a.family_dir} has no submission.npz", file=sys.stderr)
        return 2
    (out / "logs").mkdir(parents=True, exist_ok=True)
    s = Chain(a).go()
    (out / "chain.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    print(json.dumps({k: s[k] for k in ("status", "final_family_dir", "changed") if k in s}))
    return 0 if not str(s.get("status", "")).startswith("stopped") else 1


if __name__ == "__main__":
    sys.exit(main())
