#!/usr/bin/env python3
"""Fast DQI candidate ranking: reuse the base's failure profile, measure the Toffoli change on paired short runs.

The expected DQI cost over the fixed 48-draw set is

    E[cost] = mean_d  R(k_d) x Q^2 x T_d

with k_d the failing shots and T_d the per-shot mean Toffolis of draw d (102,400 shots each, seeds anchored on
40fa4ebf, salts sweep-0..47). For a candidate whose failing shots equal its
base's (k_d unchanged) the prediction is

    E[cost'] ~= mean_d  R(k_d) x Q'^2 x (T_d + dT)

where dT is the per-shot mean Toffoli change, estimated from N paired batches: identical verifier batches (same
sampler state, same simulator seeds, 256 shots each) through base and candidate. The base side comes from the
base's profile, which records every batch's Toffoli total and failing shots once (one full 48-draw run per base,
reused for every candidate); a determinism check re-runs the base on one paired batch and compares.

The precondition is checked, not assumed: the candidate is run on every batch in which the base fails and must
reproduce the same failing shots (value and phase) exactly; on the N paired batches it must add no failure. Shots
outside those batches are not checked (coverage is reported), so the verdict is "no difference found on the
checked batches", and the 48-draw sweep remains the confirmation before any release.

    profile  --base <kmb> --out <dir>                  one-time per base: <dir>/profile-<sha12>.json
    rank     --base <kmb> --candidate <kmb> --profile <json> --n 64 [--report-n 1,2,4,...] --out <json>
             [--sweep <gate.json or draws csv>]  compare with the candidate's full 48-draw sweep when one exists

Run with a Python that has kickmix and kickmix_engine: $QHDL_KICKMIX_PYTHON scripts/dqi_fast_rank.py ...
This is a thin wrapper over `kickmix_engine.diff` (`kickmix-engine profile` / `diff --profile`),
which generalises the method to any task plug-in; the CLI and output schema are unchanged.
The pure helpers below are standard library (tests/test_dqi_fast_rank.py). Tier of every number: engine verify
(sampled draws, parity-tested with kickmix), local; never a Kaggle score. Evidence is append-only: refuses to
overwrite. Written by an AI model (Claude).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402

OUTPUTS = CFG.OUTPUTS
VERIFIER = CFG.VERIFIERS / "dqi" / "verify_submission_dqi.py"
ENGINE_PKG = CFG.ENGINE_PKG  # '' = the installed kickmix_engine
DRAW_ANCHOR = "40fa4ebf25b2b9c81b16cf47981ef0f5fa2efe0cb7be54f0da83e68fea7d340e"
DRAWS = 48
BATCHES = 400          # 102,400 shots / 256 per engine batch
BATCH_SHOTS = 256
SHOTS = BATCHES * BATCH_SHOTS
REL_TIE = 1e-5
# R(k) = repetition_overhead(clopper_pearson_upper_bound(k, 102400, log2_delta=-128)) for k failing shots; the
# verifier's own functions are used when the engine is loaded, this table otherwise.
R_TABLE = (1.1912604648115832, 1.1967314991687559, 1.2013880602795501, 1.2055843238398525, 1.209464049815962,
           1.2131052140444296, 1.2165566016754124)


# ---------------------------------------------------------------- pure helpers (standard library, tested)

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r_of_k(k: int, verifier=None) -> float:
    """Repetition factor for k failing shots in 102,400."""
    if verifier is not None:
        return float(verifier.repetition_overhead(
            verifier.clopper_pearson_upper_bound(k, SHOTS, log2_delta=verifier.DEFAULT_LOG2_DELTA)))
    if 0 <= k < len(R_TABLE):
        return R_TABLE[k]
    raise ValueError(f"R({k}) needs the verifier (table covers k = 0..{len(R_TABLE) - 1})")


def schedule(n: int, draws: int = DRAWS, batches: int = BATCHES) -> list[tuple[int, int]]:
    """Fixed paired-batch schedule: item i is (draw i mod draws, batch (157 i + 3) mod batches). Prefixes nest, so
    one run at N gives every smaller N; it spreads over all draws and never repeats a batch for n <= 1200."""
    out = []
    for i in range(n):
        out.append((i % draws, (157 * i + 3) % batches))
    if len(set(out)) != len(out):
        raise ValueError("schedule repeats a batch")
    return out


def delta_stats(base_tof: list[int], cand_tof: list[int], batch_shots: int = BATCH_SHOTS) -> dict:
    """Per-shot mean Toffoli change over paired batches with its standard error (batch means as the sample)."""
    if len(base_tof) != len(cand_tof) or not base_tof:
        raise ValueError("paired lists must be non-empty and equal length")
    d = [(c - b) / batch_shots for b, c in zip(base_tof, cand_tof)]
    n = len(d)
    mean = sum(d) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in d) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = None
    return {"batches": n, "shots": n * batch_shots, "mean_per_shot": mean, "se_per_shot": se,
            "min_batch_mean": min(d), "max_batch_mean": max(d),
            "constant": all(abs(x - d[0]) < 1e-12 for x in d)}


def predict(profile_draws: list[dict], delta_t: float, q_cand: int, verifier=None) -> dict:
    """Predicted expected cost: mean over draws of R(k_d) Q'^2 (T_d + dT). profile_draws rows carry
    `toffoli_mean` (per shot), `value_errors`, `phase_errors`."""
    per = []
    for d in profile_draws:
        k = int(d["value_errors"]) + int(d["phase_errors"])
        per.append(r_of_k(k, verifier) * q_cand * q_cand * (float(d["toffoli_mean"]) + delta_t))
    return {"per_draw": per, "expected_cost": sum(per) / len(per), "draws": len(per), "Q": q_cand,
            "delta_t": delta_t}


def rel_err(pred: float, full: float) -> float:
    return pred / full - 1


def compare_failures(base_fail: dict, cand_fail: dict) -> dict:
    """base_fail/cand_fail: {(draw, batch): {"value": [...], "phase": [...]}} over the same checked batches.
    A batch matches when the set of failing shots (value or phase) is the same: the cost model charges
    R(value + phase errors), and a shot is never both, so an equal set means an equal k_d. A shot that fails in both
    but changes type (phase in the base, value in the candidate or the reverse) is reported in `type_changes`."""
    keys = sorted(set(base_fail) | set(cand_fail))
    mism, types = [], []
    for key in keys:
        b = base_fail.get(key, {"value": [], "phase": []})
        c = cand_fail.get(key, {"value": [], "phase": []})
        if set(b["value"]) | set(b["phase"]) != set(c["value"]) | set(c["phase"]):
            mism.append({"draw": key[0], "batch": key[1], "base": b, "candidate": c})
        elif sorted(b["value"]) != sorted(c["value"]):
            types.append({"draw": key[0], "batch": key[1], "base": b, "candidate": c})
    return {"batches": len(keys), "matched": len(keys) - len(mism), "mismatches": mism, "type_changes": types}


def full_sweep_cost(path) -> dict:
    """Mean per-draw cost of a 48-draw sweep: a qhdl_surgery gate JSON (rows[].candidate.score) or a
    draws_48.csv (candidate_cost column)."""
    p = Path(path)
    if p.suffix == ".csv":
        with p.open() as f:
            rows = list(csv.DictReader(f))
        costs = [float(r["candidate_cost"]) for r in rows]
        errs = [int(r["candidate_value_errors"]) + int(r["candidate_phase_errors"]) for r in rows]
        tofs = None
        seconds = None
    else:
        env = json.loads(p.read_text())
        rows = [r["candidate"] for r in env["rows"]]
        costs = [float(r["score"]) for r in rows]
        errs = [int(r["value_errors"]) + int(r["phase_errors"]) for r in rows]
        tofs = [float(r["toffoli_count"]) for r in rows]
        seconds = sum(float(r.get("seconds", 0)) for r in rows)
    return {"path": str(p), "sha256": sha256_file(p), "draws": len(costs), "expected_cost": sum(costs) / len(costs),
            "per_draw": costs, "failing_shots": errs, "toffoli_means": tofs, "seconds": seconds}


def find_sweep(sha: str):
    """A stored 48-draw sweep for this circuit SHA-256 under OUTPUTS, or None when there is none."""
    hits = sorted(OUTPUTS.glob(f"local-portfolio-*/draws/{sha[:12]}-sweep{DRAWS}@{DRAW_ANCHOR[:8]}.gate.json"))
    return hits[0] if hits else None


MAX_UNDETECTED_EXTRA_K = 0.1  # failing shots per draw the precondition must be able to exclude (a typical public decoder: 0.24 per draw)


def precondition_power(pre: dict, shots_per_draw: int = SHOTS, draws: int = DRAWS) -> dict:
    """What the precondition can actually exclude. It only compares batches where the base fails plus
    the N paired batches, so a candidate whose extra failures sit in other batches passes. The clean checked shots
    (paired batches where the base is clean) bound the candidate's per-shot extra-failure rate: with none found, the
    95% upper bound is 3/n (rule of three). Scaled to a draw this is the extra failures per draw that cannot be ruled
    out; `powered` is False when that exceeds MAX_UNDETECTED_EXTRA_K."""
    checked = int(pre["checked_shots"])
    clean = max(0, checked - int(pre.get("base_failing_shots", 0)) * BATCH_SHOTS)  # base-failing batches carry the base's own k
    if pre.get("mismatches"):
        return {"clean_checked_shots": clean, "upper_extra_k_per_draw": None, "powered": False, "mismatch": True}
    ub = 3.0 / clean * shots_per_draw if clean else float("inf")
    return {"clean_checked_shots": clean, "upper_extra_k_per_draw": ub,
            "powered": ub <= MAX_UNDETECTED_EXTRA_K, "mismatch": False}


def precondition_verdict(pre: dict) -> str:
    """PASS only when the check found no difference AND had the power to exclude MAX_UNDETECTED_EXTRA_K extra failures
    per draw; otherwise UNDERPOWERED (no difference found, but the k_d profile is not certified: run the sweep)."""
    if pre.get("mismatches"):
        return "FAIL"
    return "PASS" if precondition_power(pre)["powered"] else "UNDERPOWERED"


def write_new(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:  # evidence is append-only
        f.write(json.dumps(obj, indent=1, default=str) + "\n")
    return path


# ---------------------------------------------------------------- engine side (needs kickmix_engine)
#
# The profile and the paired rank live in the engine (`kickmix_engine.diff`): `kickmix-engine profile dqi` and
# `kickmix-engine diff dqi --profile` do the work for any task plug-in. This script is kept as a thin wrapper with its original CLI and
# output schema (the engine reproduces this script's five validation predictions exactly).
# The pure helpers above remain the standard-library reference the tests exercise.

def _engine():
    CFG.add_engine_path()
    import kickmix_engine.diff as ed
    from kickmix_engine.plugins import DrawSet, make_plugin
    return ed, DrawSet, make_plugin


def cmd_profile(a) -> None:
    ed, DrawSet, make_plugin = _engine()
    out_dir = Path(a.out).resolve()
    base = Path(a.base).resolve()
    plugin = make_plugin("dqi", str(VERIFIER))
    circ = ed.Circuit(base, plugin)
    out = out_dir / f"profile-{circ.sha[:12]}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    print(f"profile {circ.sha[:12]} Q={circ.q} load {circ.load_s:.1f}s -> {out}", flush=True)
    sweep = find_sweep(circ.sha) if not a.sweep else Path(a.sweep)
    sweep_rows = json.loads(Path(sweep).read_text())["rows"] if sweep else None
    prof = ed.profile(plugin, circ, DrawSet(anchor=DRAW_ANCHOR, salt_format="sweep-{}", count=a.draws),
                      threads=a.threads, chunk=a.chunk, log=lambda m: print(m, flush=True))
    for d, rec in enumerate(prof["draws"]):
        if sweep_rows:
            s = sweep_rows[d]["candidate"]
            rec["sweep_match"] = (abs(s["toffoli_count"] - rec["toffoli_mean"]) < 1e-6
                                  and s["value_errors"] == rec["value_errors"]
                                  and s["phase_errors"] == rec["phase_errors"]
                                  and abs(s["score"] / rec["score"] - 1) < 1e-12
                                  and str(sweep_rows[d]["seed"]) == rec["seed"])
    prof["sweep"] = str(sweep) if sweep else None
    prof["sweep_match_all"] = all(r.get("sweep_match") for r in prof["draws"]) if sweep_rows else None
    prof["rerun"] = " ".join(["dqi_fast_rank.py"] + sys.argv[1:])
    write_new(out, prof)
    print(json.dumps({k: prof[k] for k in prof if k != "draws"}, indent=1))


def cmd_rank(a) -> None:
    ed, DrawSet, make_plugin = _engine()
    t_all = time.perf_counter()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    report_n = sorted({int(x) for x in a.report_n.split(",")} | {a.n}) if a.report_n else [a.n]
    if max(report_n) > a.n:
        raise SystemExit("--report-n values must be <= --n")
    sweep = a.sweep
    if not sweep:
        hit = find_sweep(sha256_file(a.candidate))
        sweep = str(hit) if hit else None
    rep = ed.diff("dqi", Path(a.base).resolve(), Path(a.candidate).resolve(), verifier_path=str(VERIFIER), n=a.n,
                  report_n=report_n, profile_json=a.profile, threads=a.threads, structural_layer=False,
                  compare_state=True, sweep=sweep)
    pr, sem = rep["prediction"], rep["semantic"]
    prof_path = Path(a.profile).resolve()
    rec = {"tier": "engine verify (sampled draws, fixed seeds; parity-tested with kickmix), local; "
                   "predicted expected cost is a static model / projection",
           "base": {"path": rep["a"]["path"], "sha256": rep["a"]["sha256"], "Q": pr["profile"]["Q"],
                    "expected_cost": pr["profile"]["expected_cost"], "profile": str(prof_path),
                    "profile_sha256": sha256_file(prof_path)},
           "candidate": {"path": rep["b"]["path"], "sha256": rep["b"]["sha256"], "Q": rep["b"]["Q"]},
           "draw_set": pr["profile"]["draw_set"], "schedule": "draw i mod 48, batch (157 i + 3) mod 400",
           "N": a.n, "by_n": pr["by_n"], "delta_t": pr["delta_t"], "delta_t_se": pr["delta_t_se"],
           "predicted_expected_cost": pr["predicted_expected_cost"], "predicted_per_draw": pr["predicted_per_draw"],
           "predicted_dcost_pct": pr["predicted_dcost_pct"],
           "precondition": pr["precondition"],
           "precondition_power": precondition_power(pr["precondition"]),
           "precondition_verdict": precondition_verdict(pr["precondition"]),
           "tie_rule": "a tie or a cost ranking between near-equal files needs a 96-draw sweep, not this precondition", "determinism_check": pr["determinism_check"],
           "paired": pr["paired"], "full_sweep": pr["full_sweep"],
           "paired_agreement": {k: sem[k] for k in ("agree_fraction", "phase_equal_fraction", "interface_equal_fraction",
                                                   "verdict_equal_fraction", "disagreements_total")},
           "timing": {"candidate_load_s": rep["b"]["load_s"], "simulate_s": round(sem["seconds"]["simulate"], 2),
                      "total_s": round(time.perf_counter() - t_all, 2), "threads": a.threads,
                      "speedup_vs_sweep": ((pr["full_sweep"]["seconds"] / (time.perf_counter() - t_all))
                                           if pr["full_sweep"] and pr["full_sweep"]["seconds"] else None)},
           "engine": {"version": rep["engine_version"], "command": "kickmix-engine diff dqi --profile"},
           "rerun": " ".join(["dqi_fast_rank.py"] + sys.argv[1:])}
    write_new(out, rec)
    print(json.dumps({k: rec[k] for k in rec if k not in ("paired", "predicted_per_draw", "by_n")}
                     | {"by_n": pr["by_n"]}, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="base failure profile over the 48 fixed draws (per-batch Toffolis, failing shots)")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True, help="directory; writes profile-<sha12>.json")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--chunk", type=int, default=100, help="batches per engine call")
    p.add_argument("--sweep", help="the base's 48-draw gate JSON to cross-check (default: found by sha)")
    p.set_defaults(fn=cmd_profile)
    p = sub.add_parser("rank", help="paired short run of a candidate against a profiled base")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--n", type=int, default=64, help="paired batches (256 shots each)")
    p.add_argument("--report-n", help="comma list of prefix sizes to report too, e.g. 1,2,4,8,16,32")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--sweep", help="candidate's full 48-draw sweep (gate JSON or draws_48.csv) for validation")
    p.add_argument("--skip-determinism", action="store_true",
                   help="kept for compatibility; the engine checks the base on every paired batch")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_rank)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
