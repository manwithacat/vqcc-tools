# invsqrt evaluation service (contract)

`scripts/invsqrt_eval.py` checks invsqrt candidates for search loops. It runs the official invsqrt verifier logic
(every x in 1..3,133,452, 64 shots each, the verifier's seeds, chunks, reference table, tolerance and cost model) with
kickmix-engine as the simulator.

## Evidence tier

Every result is **local prescreen (engine verify, parity-proven)**, and every JSON line says so in `tier`. The
engine has been checked against kickmix shot for shot, and its `verify` against the official verifier field for
field (see the kickmix-engine README and the `compare` subcommand below). It is **not** *official verifier PASS*.
The promotion gate is unchanged: the unmodified official verifier on a kickmix build
(`scripts/local_verify_reference.py`), then a hosted Kaggle run; only the official verifier speaks to a score. The
`models` subcommand has its own tier: it is a comparator for fixed-seed BitSim-style outcome models, not the official
shot schedule.

## Running it

Interpreter: a Python that can import kickmix (`$QHDL_KICKMIX_PYTHON`), always with `-I` (the script adds the
engine package path itself and refuses an engine imported from anywhere else). Run from the repository root.

```sh
PY=$QHDL_KICKMIX_PYTHON
$PY -I scripts/invsqrt_eval.py verify CAND.kmb [--salt S] [--threads 4] [--out new.jsonl]
$PY -I scripts/invsqrt_eval.py seeds  CAND.kmb --salts 8 [--stop-on-fail]        # or --salt-list ',a,b'
$PY -I scripts/invsqrt_eval.py margin CAND.kmb [--salts 4] [--screen] [--per-x-out new.npz]
$PY -I scripts/invsqrt_eval.py models CAND.kmb                                  # fixed-seed models 7/101
$PY -I scripts/invsqrt_eval.py verify --candidates DIR --out new.jsonl          # batch: every DIR/*.kmb
$PY -I scripts/invsqrt_eval.py compare --engine new.jsonl --official report.json   # field identity
$PY -I scripts/invsqrt_eval.py fixture rare-branch SRC.kmb --k 27 --out new.kmb   # coverage-test fixture
```

Settings: the engine package is the `python/` directory of a kickmix-engine checkout, given with `--engine-python
DIR` or `VQCC_ENGINE_PY` (set it explicitly; it is used read only). The official verifiers come from the competition
data and are not included in this repository: the default is `data/official_full/verifiers` under the repository
root, overridden with `KICKMIX_ENGINE_VERIFIERS` (point it at `$QHDL_OFFICIAL_ROOT/verifiers`). 4 threads. `--salts N` means the default
(empty) salt followed by `sweep-1` … `sweep-(N-1)` (`--salt-format` changes the pattern). `--out` never overwrites:
an existing path is refused.

### Resident mode

Two forms, both keep compiled programs and the reference tables in memory between candidates:

1. **One process, many candidates** (recommended for loops): `--candidates DIR`, or a Python caller that holds one
   `invsqrt_eval.Engine()` and calls `eng.verify(path, salt)` / `margin_one_salt(...)` repeatedly. The reference
   table (`reference_values`, about 1.2 s) is computed once per process; programs are kept by content hash.
2. **A socket server** for callers in other processes:

   ```sh
   $PY -I scripts/invsqrt_eval.py serve --socket /tmp/ie.sock &                 # kickmix-engine serve
   $PY -I scripts/invsqrt_eval.py verify CAND.kmb --via /tmp/ie.sock
   $PY -I scripts/invsqrt_eval.py seeds  CAND.kmb --salts 8 --via /tmp/ie.sock
   ```

   The server runs the engine's own `verify invsqrt` CLI (output identical to the one-shot CLI). Only `verify` and
   `seeds` go through the socket; `margin` and `models` need per-shot arrays and always run in-process. The server
   serves one request at a time and trusts its own hash of an unchanged path. Stop it with
   `PYTHONPATH=<engine python dir> python -m kickmix_engine --via /tmp/ie.sock stop` or by ending the process.

## Inputs

A canonical `.kmb` invsqrt circuit (one register `x` of 22 qubits; everything else scratch). The engine seeds from
the SHA-256 of the file bytes; the official verifier seeds from kickmix's re-serialisation. They agree for canonical
files (kickmix-written, or `qhdl_edit.py` output). A non-canonical file would get
different measurement draws; convert it with kickmix first.

## Outputs (one JSON line per candidate)

Common envelope: `schema` (`vqcc-invsqrt-eval/1`), `command`, `tier`, `gate`, `circuit`, `circuit_sha256`,
`threads`, `mode` (in-process or the socket), `engine` (version, package path, extension SHA-256, kernel, verifier
path and SHA-256; socket path only in socket mode).

- **verify**: `summary` = `salt` plus the official summary fields, same names and values as the official CLI:
  `status, score, toffoli_equivalent, toffoli_count, max_toffoli_count, t_count, rotation_count, rotation_eps,
  num_qubits, reaction_depth, num_ops, verified_shots, worst_error_over_tolerance, worst_x, circuit_sha256`. On
  failure: `status: FAIL` and `error` with the official message text (first failure in the verifier's order).
  `timing`: `wall_s`, `cpu_s` (process user+sys; "unavailable" in socket mode, where `server_s` is given),
  `peak_rss_mb`.
- **seeds**: `per_salt` (one verify summary per salt), `salts_requested`, `salts_run`, `n_pass`, `n_fail`,
  `all_pass` (every requested salt run and passed), `first_fail` (`index`, `salts_to_catch` = index + 1, `salt`,
  `error`), `worst_case` over passing salts (`max_cost` and its salt, `min_cost`, the largest
  `worst_error_over_tolerance` with its x and salt, `min_rotation_eps`), `timing_total`.
- **margin**: `per_salt` rows (the same official summary fields except the static ones, plus `min_margin_units`, and
  — because it never stops at the first failure — `failing_shots` by type, `n_failing_x_seen`,
  `failing_x_sample`, `first_failure`), and `per_x`:
  - `worst_ratio`, `worst_ratio_x`, `worst_ratio_salt`: the largest |phase error| / tol(x) over all x, shots and
    salts;
  - `min_margin_units` (and `_turns`), `min_margin_x`, `min_margin_salt`: the smallest tol(x) − |error| (this is
    the quantity that sets `rotation_eps`, so it moves the cost);
  - `histogram` of the per-x worst ratio: `[0, 0.5)`, `[0.5, 0.9)`, `[0.9, 0.99)`, `[0.99, 0.999)`,
    `[0.999, 0.9999)`, `[0.9999, 1.0]`, `> 1.0`; `near_cap_x` counts (≥ 0.99, 0.999, 0.9999); `top` (the 20 worst
    x with ratio, margin and salt); optional `.npz` with the per-x arrays.
  With `--screen`, a one-screen table goes to stderr.
- **models**: per seed `valid`, `failing_rows`, `first_failing_x`, `failing_x_sample`; `valid_on_models`.
- **compare**: `match`, `fields_compared`, `differences`, file hashes. Exit code 3 on a mismatch.

Exit codes: 0 when the command ran (a FAIL verdict is data, not an error); `compare` 3 on mismatch.

### How the margin is computed

The engine's on-engine check (`run(..., check=(target, tol))`) returns only per-chunk folds (min margin, worst
ratio, first failure), not per-x values. `margin` therefore runs the same schedule *without* the on-engine check: the
engine returns each shot's phase (`phase_units`, the verifier's 2⁻⁶⁴-turn integer), restored x and scratch flag,
and the script applies the verifier's own arithmetic: `error = (theta − target[x−1])` as wrapping int64,
`|error|` as uint64, compared with `tol[x−1]` from the official `reference_values` (tolerance 2⁻²⁴ relative).
Shot i has x = i mod 3,133,452 + 1, so each 64-shot pass over the domain folds into a per-x maximum without
sorting. Shots that fail the x or scratch check carry no meaningful phase and are counted, not folded. Check:
on a public 62-qubit circuit, `margin`'s per-salt summary equals the official report field for field (score, Toffolis,
`rotation_eps`, worst ratio and x), so the per-shot path and the on-engine check agree.

## Runtime (Apple M2 Pro, 4 threads, shared machine)

About 24–31 s per full verify at 4 threads (200,540,928 shots), about 25 s per salt for `margin`, about 2 s for a
candidate that fails early, under 1 s for `models`. Peak RSS ≈ 0.55 GB per process. The official verifier on
kickmix takes ≈ 130 s at 8 workers on a 62-qubit circuit (617 s at 2 workers on a 215-qubit circuit, where the engine
takes 67 s).

## What it does and does not catch

- The official schedule gives every x 64 shots with independent measurement outcomes. A fault that needs a
  particular measurement branch at a particular x is caught when any of those shots takes the branch. The
  fixed-seed models give each x two random shots (plus the all-zero branch in BitSim), so a fault confined to
  one x on a 1/2-probability branch passes them with probability 1/4; a negative-control fixture of exactly that
  kind passed them.
- One salt is one draw of the measurement outcomes. A fault on a branch rarer than about one shot in 2×10⁸ can
  pass the default salt and fail the grader's salt. `seeds --salts N` runs N independent draws; the synthetic
  rare-branch fixture (`fixture rare-branch`) (k = 27) passed the default salt and was caught at the second salt.
- **Recommended gate: `seeds --salts 8 --stop-on-fail` before the official verifier** (≈ 3.2 min).
  Eight clean salts catch a fault firing at least once per ~3 salts with 95% probability and bound the per-salt
  firing rate at about 3/8; use more salts for a candidate headed for a submission.
- No number of salts proves the absence of a rare-branch fault. For promotion, the official verifier at the
  default salt remains necessary; `seeds` is the cheap way to look for salt-dependent failures before it.

## Failure modes

- `kickmix_engine imported from …`: run with `python -I`; the script inserts the engine path itself.
- `official invsqrt verifier not found`: set `KICKMIX_ENGINE_VERIFIERS` to a directory with `invsqrt/`.
- `no resident engine answering on SOCKET`: start `serve` first; AF_UNIX paths are limited to ~100 bytes.
- `… exists (append-only)`: `--out`, `--per-x-out` and fixture outputs never overwrite.
- `ValueError: Invalid register schema` / `Expected 22 x qubits`: the verifier's register rules, reported as FAIL.
- A stale engine extension (a different build) changes `engine.extension_sha256`; re-run the
  parity control before trusting new numbers.
- Qubit ids must be below 65,527 (engine constant slots).
