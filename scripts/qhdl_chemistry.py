#!/usr/bin/env python3
"""Chemistry-family arithmetic companion: supplied models, never circuit certification.

Q = prep_persistent + batch_size*brot + max(prep_temporary,
    loader_qubits - selection_qubits - batch_size*brot)
T = nonangle_toffolis + 2*angle_toffolis_per_pass; cost = lambda_eff*Q*T.
Errors and tolerance must use the same units. Report extraction uses Hartree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

TIER = "static model / projection"
INTEGER_FIELDS = {"brot", "batch_size", "selection_qubits", "loader_qubits",
                  "prep_persistent", "prep_temporary"}
REAL_FIELDS = {"nonangle_toffolis", "angle_toffolis_per_pass", "lambda_eff", "tolerance"}
MODEL_FIELDS = INTEGER_FIELDS | REAL_FIELDS | {"errors"}
DELTA_ASSUMPTION = ("Each listed word block receives the SAME nonzero XOR mask; "
                    "delta[k] = word[k] XOR word[k-1], with word[-1] = 0. "
                    "Different masks require actual words; no cancellation is inferred for them.")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    _number(value, name)
    return value


def _number(value, name, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{name} must be a nonempty string of at most 200 characters")
    return value


def _model(model):
    if not isinstance(model, dict) or set(model) != MODEL_FIELDS:
        raise ValueError(f"model requires exactly {sorted(MODEL_FIELDS)}")
    for key in INTEGER_FIELDS:
        _integer(model[key], key, 1 if key in {"brot", "batch_size", "loader_qubits"} else 0)
    for key in REAL_FIELDS:
        _number(model[key], key, key == "lambda_eff")
    errors = model["errors"]
    if not isinstance(errors, list) or len(errors) != 2:
        raise ValueError("errors must contain exactly two residuals")
    for value in errors:
        _number(value, "error")
    persistent = model["batch_size"] * model["brot"]
    scratch = model["loader_qubits"] - model["selection_qubits"] - persistent
    if scratch < 0:
        raise ValueError("loader_qubits cannot be less than selection_qubits + batch_size*brot")
    q = model["prep_persistent"] + persistent + max(model["prep_temporary"], scratch)
    t = model["nonangle_toffolis"] + 2 * model["angle_toffolis_per_pass"]
    try:
        cost = model["lambda_eff"] * q * t
    except OverflowError as exc:
        raise ValueError("projected cost exceeds finite numeric range") from exc
    for key, value in (("Q_total", q), ("T_total", t), ("cost", cost)):
        _number(value, key)
    return {"Q_total": q, "T_total": t, "cost": cost,
            "angle_persistent_qubits": persistent, "angle_temporary_qubits": scratch,
            "temporary_peak_qubits": max(model["prep_temporary"], scratch),
            "loader_workspace_binding": scratch > model["prep_temporary"],
            "loader_within_128_cap": model["loader_qubits"] <= 128,
            "errors_within_tolerance": all(e <= model["tolerance"] for e in errors),
            "residual_margins": [model["tolerance"] - e for e in errors]}


def affected_delta_blocks(changed_word_blocks, num_word_blocks):
    """Exact boundary indices under DELTA_ASSUMPTION, not arbitrary word edits."""
    _integer(num_word_blocks, "num_word_blocks", 1)
    if not isinstance(changed_word_blocks, list):
        raise ValueError("changed_word_blocks must be a list")
    changed = set()
    for index in changed_word_blocks:
        _integer(index, "changed word index")
        if index >= num_word_blocks:
            raise ValueError("changed word index outside num_word_blocks")
        changed.add(index)
    successors = {i + 1 for i in changed if i + 1 < num_word_blocks}
    return sorted(changed ^ successors)


def _row(identifier, model, status, changed, count):
    row = _model(model)
    accuracy = "unknown" if status == "unknown" else (
        "fail" if status == "fail" or not row["errors_within_tolerance"] else "pass")
    row.update(id=identifier, model=dict(model), numerical_status=status,
               accuracy_status=accuracy,
               eligible_for_further_validation=accuracy == "pass" and row["loader_within_128_cap"],
               changed_word_blocks=sorted(set(changed)),
               affected_delta_blocks=affected_delta_blocks(changed, count))
    return row


def analyze(case):
    """Project explicitly supplied family models; do not read files or verify bytes."""
    if not isinstance(case, dict) or set(case) != {"id", "baseline", "proposals", "num_word_blocks"}:
        raise ValueError("case requires id, baseline, proposals, num_word_blocks")
    identifier = _identifier(case["id"], "case id")
    count = _integer(case["num_word_blocks"], "num_word_blocks", 1)
    baseline = _row("baseline", case["baseline"], "pass", [], count)
    proposals = case["proposals"]
    if not isinstance(proposals, list) or len(proposals) > 1000:
        raise ValueError("proposals must be a list with at most 1000 entries")
    rows, ids = [], {"baseline"}
    for proposal in proposals:
        if not isinstance(proposal, dict) or set(proposal) != {
                "id", "overrides", "changed_word_blocks", "numerical_status"}:
            raise ValueError("proposal requires id, overrides, changed_word_blocks, numerical_status")
        pid = _identifier(proposal["id"], "proposal id")
        if pid in ids:
            raise ValueError(f"duplicate/reserved proposal id: {pid}")
        ids.add(pid)
        overrides = proposal["overrides"]
        if not isinstance(overrides, dict) or not set(overrides) <= MODEL_FIELDS:
            raise ValueError("overrides must be a subset of baseline model fields")
        status = proposal["numerical_status"]
        if status not in ("unchanged", "unknown", "pass", "fail"):
            raise ValueError("invalid numerical_status")
        changed = proposal["changed_word_blocks"]
        affected_delta_blocks(changed, count)  # Validate before sorting/deduplication.
        row = _row(pid, {**case["baseline"], **overrides}, status, changed, count)
        row["cost_delta"] = row["cost"] - baseline["cost"]
        row["cost_change_percent"] = (100 * row["cost_delta"] / baseline["cost"]
                                       if baseline["cost"] else None)
        row["maximum_angle_toffolis_per_pass_at_equal_cost"] = (
            baseline["cost"] / row["model"]["lambda_eff"] / row["Q_total"]
            - row["model"]["nonangle_toffolis"]) / 2
        if not math.isfinite(row["maximum_angle_toffolis_per_pass_at_equal_cost"]):
            raise ValueError("break-even threshold exceeds finite numeric range")
        row["lower_cost"] = row["cost"] < baseline["cost"]
        row["eligible_lower_cost"] = row["lower_cost"] and row["eligible_for_further_validation"]
        if row["cost_change_percent"] is not None and not math.isfinite(row["cost_change_percent"]):
            raise ValueError("cost percentage exceeds finite numeric range")
        rows.append(row)
    return {"schema_version": 1, "id": identifier, "tier": TIER,
            "scope": "Supplied-model arithmetic; not bytes-derived or verifier acceptance.",
            "numerical_scope": "Declared status and supplied residuals only; unchanged is a caller assertion, not a check.",
            "delta_assumption": DELTA_ASSUMPTION,
            "baseline": baseline, "proposals": rows}


def report_to_case(report, *, case_id="report-baseline", tolerance=0.02):
    """Extract a local report's declared baseline; does not authenticate its PASS.

    Only PASS reports with the full resource breakdown are supported. Raw
    error_1/error_2 are used, never rendered or rounded mHa values.
    """
    try:
        summary = report["results"]["dfthcsos"]["summary"]
        resources, metrics = summary["resources"], summary["metrics"]
        if summary["status"] != "PASS":
            raise ValueError("report baseline requires a declared PASS")
        model = {"brot": summary["brot"], "batch_size": summary["batch_size"],
                 "selection_qubits": resources["angles_selection_qubits"],
                 "loader_qubits": resources["angles_qubits"],
                 "prep_persistent": resources["heuristic_persistent_qubits"],
                 "prep_temporary": resources["heuristic_temporary_qubits"],
                 "nonangle_toffolis": resources["heuristic_toffoli"],
                 "angle_toffolis_per_pass": resources["angles_toffoli_per_pass"],
                 "lambda_eff": metrics["lambda_eff"],
                 "errors": [metrics["error_1"], metrics["error_2"]], "tolerance": tolerance}
        case = {"id": case_id, "baseline": model, "proposals": [],
                "num_word_blocks": summary["num_circuits"]}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported report schema: {exc}") from exc
    projected = analyze(case)["baseline"]
    for field, source in (("Q_total", "num_qubits"), ("T_total", "toffoli_count"), ("cost", "score")):
        if source not in summary:
            raise ValueError(f"report missing {source}")
        expected = _number(summary[source], source)
        if not math.isclose(projected[field], expected, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"report {source} disagrees with resource projection")
    return case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", type=Path, help="explicit model JSON")
    source.add_argument("--report", type=Path, help="local report JSON; extract baseline only")
    parser.add_argument("--out", type=Path, required=True, help="fresh JSON output (never overwritten)")
    args = parser.parse_args(argv)
    path = args.case or args.report
    try:
        raw = path.read_bytes()
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("input exceeds 8 MiB limit")
        supplied = json.loads(raw)
        case = supplied if args.case else report_to_case(supplied, case_id=path.stem)
        result = analyze(case)
        result["provenance"] = {"input_path": str(path.resolve()),
                                "input_sha256": hashlib.sha256(raw).hexdigest(),
                                "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "input_kind": "case" if args.case else "declared_local_report",
                                "hash_scope": "Input and tool files only; no circuit bytes authenticated.",
                                "verifier_run": False}
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        with args.out.open("x") as output:
            output.write(encoded)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"qhdl_chemistry: {exc}\n")
    print(json.dumps({"tier": TIER, "case_id": result["id"], "out": str(args.out),
                      "proposals": len(result["proposals"]),
                      "eligible_lower_cost_count": sum(p["eligible_lower_cost"] for p in result["proposals"]),
                      "input_sha256": result["provenance"]["input_sha256"],
                      "tool_sha256": result["provenance"]["tool_sha256"],
                      "verifier_run": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
