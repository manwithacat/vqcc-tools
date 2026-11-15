# Chemistry family view

`scripts/qhdl_chemistry.py` is a standalone QHDL companion for the chemistry resource model. It connects loader width, persistent word registers, the shared workspace maximum, full cost, declared numerical margins and adjacent XOR-delta dependencies. It is not yet a verb in `scripts/qhdl.py`.

Use a recorded local verifier report to inspect a family:

```sh
python3 scripts/qhdl_chemistry.py \
  --report <local-verifier-report.json> \
  --out <fresh-dir>/chemistry-family-view.json
```

The output path must be fresh. The compact stdout envelope points to the full JSON, hashes the input and tool, and says that no verifier ran. Report extraction checks that its resource decomposition reproduces its reported Q, Toffolis and cost. It does not authenticate a report's PASS or hash circuit members. It uses raw residual values rather than rounded display values.

For an explicit proposed design use `--case case.json --out fresh-view.json`. A case has `id`, `num_word_blocks`, a `baseline` model, and `proposals`. Each proposal has an `id`, `overrides` of baseline fields, `changed_word_blocks`, and `numerical_status` (`unchanged`, `unknown`, `pass`, `fail`). Models require:

```json
{
  "brot": 8, "batch_size": 5, "selection_qubits": 11,
  "loader_qubits": 128, "prep_persistent": 85, "prep_temporary": 303,
  "nonangle_toffolis": 6741, "angle_toffolis_per_pass": 4746,
  "lambda_eff": 47.035246076743576,
  "errors": [0.014376098866782544, 0.019999990003371285],
  "tolerance": 0.02
}
```

These numbers come from one recorded public-family report; an arbitrary override is a hypothesis, not a realizable circuit. Residuals and tolerance must use the same units (Hartree in official reports).

The view computes:

- Persistent loader wires: `batch_size*brot`.
- Loader temporary wires: `loader_qubits-selection_qubits-batch_size*brot`.
- Full Q: preparation persistent plus loader persistent plus the larger temporary workspace.
- Total Toffolis: modeled non-angle charge plus twice the per-pass loader charge.
- Cost: `lambda_eff*Q*Toffolis`.
- Break-even loader charge for each proposed resource/lambda combination, and both residual margins.

`eligible_for_further_validation` requires the stated accuracy evidence and the 128-wire cap to pass. `eligible_lower_cost` additionally requires strict cost improvement. Both are *static model / projection* results; neither is acceptance by the official verifier. Unknown numerical status never qualifies; explicit failure remains failure even if a caller supplies small residual numbers. `unchanged` is a caller assertion, not a word/factor equivalence check. The helper does not validate SOS gap constraints, phase cleanup, actual loader semantics, preparation shapes, coefficient feasibility, or whether a different precision/batch configuration is realizable. A physical candidate still needs the official equations and full verifier.

For the dependency view, every listed word block must receive the **same nonzero XOR mask**. Under that condition, changing a block normally touches that delta and its successor; equal changes on consecutive blocks cancel the interior delta changes. The final block has no successor delta. The helper checks this exact boundary identity; it cannot infer cancellation for different masks or translate XOR support into chemistry error. Batch-size changes require a separately established word-block layout before using this view on actual words.

The helper was exercised on six constructed diagnostic cases with an independent arithmetic grader. That tests contract reasoning and evidence gates, not optimization discovery; a correct diagnostic answer implies no circuit improvement.
