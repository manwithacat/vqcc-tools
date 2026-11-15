# dfthc_refit

Version 1.8.0-dev. The Kaggle utility script `manwithacat/dfthc-refit` carries an earlier release (1.1.1, weight refit only).

## Commands

One command per family and question. Only `refit` and `replan` write a new `dfthcsos` directory; the others read a
family and report. Run `python dfthc_refit.py SUBCOMMAND --help` for every option.

| Command | What it does | Writes a family? | Needs | Evidence tier of its numbers |
| --- | --- | --- | --- | --- |
| `conventions FAMILY_DIR` | Word conventions (v1.5.0-dev, `gauge.py`): storage range, tree, gauge class (signed / pi / free range), loader class per slot (XOR-in / chained), which subcommands refuse the family. Every family subcommand runs it first | no | NumPy; kickmix for the loaded words and the loader classes (else the npz words, no classes) | exact (decoder arithmetic; native runs) |
| `gauge-options FAMILY_DIR --slot K` | Per-loader gauge characterisation (v1.6.0-dev, `gauge_options.py`): the slot's exact word symmetries split into free / constrained / costed directions (bases, dimensions), its gauge contract (p_in, p_out, address-only, the change g and its rank), qubit prices of constrained moves, optional structural prices of costed variants, suggested repairs; `--plans DIR` emits a replan's best plan per slot in both gauge layouts and checks it natively | no (`--emit-dir` writes the emitted loaders) | NumPy; kickmix for the loaded words, loader classes and `--plans`; Lera's source for `--plans` | exact (GF(2) algebra on the tree's symmetry group); widths a static model unless emitted; native check |
| `[refit] SUBMISSION_DIR OUT_DIR` | Refit the DF-THC weights on the fixed words (the default mode; the word `refit` is optional) | yes: new `submission.npz`, loaders copied | NumPy, SciPy, PyTorch; kickmix for word recovery and `verify_after` | official `evaluate_circuits`; local verifier with kickmix |
| `replan FAMILY_DIR OUT_DIR --backend B` | Re-plan every ANGLE_k loader on its fixed words, emit with Lera's emitter, check natively, keep a slot only on a strict gain (positive-gain acceptance), then family checks. `B` = `qrom-plan` (default), `kernel`, `shared` or `block` | yes: never worse per slot | kickmix, Lera's `tensorqrom.py`; the qrom-plan binary or `kernel_search.py` for those backends | official `evaluate_circuits` (static model / projection) until `--verify` passes |
| `replan FAMILY_DIR OUT_DIR --plans-from DIR` | Plan replay (v1.7.0-dev): no planner runs; each slot's plan is read from saved plan JSONs (a replan `OUT_DIR`, its `work/plans`, or a directory of `k<slot>/`), re-checked exact on this family's words, then emitted and checked exactly as a searched plan (gauge-aware emitter, native check, strict gain, family checks). Deterministic, no CPU caps | yes: never worse per slot | kickmix, Lera's `tensorqrom.py`, the plan files | as `replan` |
| `plan FAMILY_DIR --planner P` | Per-slot plans with one planner (`kernel`, `shared`, `block`): products, predicted Toffolis (cache + products), saving against the shipped loaders. Nothing is emitted | no | NumPy; kickmix for the shipped counts, recovered partitions and words (or `--words`) | static model / projection |
| `recover FAMILY_DIR` | Structure census of the shipped loaders (`recover.py`): products read back from the circuit, function check, linear relations among ANF rows, partitions, pairing, shared products; `--emit-check` re-emits and checks natively | no (`--out` writes `words.npy` + `family.json` for `price-words`) | kickmix; Lera's source for `--emit-check` | exact (GF(2) algebra on the circuit); native check |
| `price-words FAMILY_DIR` | Shadow prices (KKT multipliers) and, with `--moves FILE` or `--rank-drop K`, a priced word-move table: AND-count saving, accuracy price, net (`price_words.py`) | no | NumPy, SciPy, PyTorch, the verifiers | static model / projection (first order or bounded re-refit of the official metrics) |
| `lowcache FAMILY_DIR` | Low-cache floor per loader (v1.8.0-dev, `lowcache.py`): free gauge sources (incoming top bits of words whose pi gauge does not change, holding one low monomial), the floor `nonlinear monomials − free sources` and its proof (GF(2) rank of the mixed ANF rows on real addresses), the shipped loader's low caches and the gap, and which gauge-losing columns a pure pi-gauge symmetry can touch | no | NumPy; kickmix for the shipped counts | exact |
| `chain.py FAMILY_DIR OUT_DIR` | The recipe below as one command (conventions → recover → plan screen → replan → chained refit), one `chain.json` summary | yes, under OUT_DIR only | both interpreters (K for loaders, P for refit) | as each step |
| `chained-and TARGET.json ACTION` | `bounds` (AND-count lower bounds, verified rank-2 construction), `verify` (exhaustive), `encode` (DIMACS CNF) for one GF(2) block (`chained_and.py`) | no | NumPy | exact (exhaustive verification / bounds); no solver bundled |

## Environment

Nothing outside this directory is shipped. The commands and tests find their inputs through environment variables:

| Variable | What it points at | When unset |
| --- | --- | --- |
| `QHDL_KICKMIX_PYTHON` | a Python that can import kickmix | `chain.py --kickmix-python` defaults to the current interpreter; run the loader tests with it |
| `QHDL_ANALYSIS_PYTHON` | an interpreter with the engine analysis extension (`kickmix_analysis`) | the GF(2) backend falls back to Python |
| `QHDL_OFFICIAL_ROOT` | the directory containing the competition `verifiers/` (not shipped: download the competition data) | `<repo>/data/official_full/verifiers`, then `/kaggle/input`; tests that need it skip |
| `QHDL_DATASET` | the public circuits dataset directory (on Kaggle `/kaggle/input/vqcc-apache-public-circuits`; https://www.kaggle.com/datasets/manwithacat/vqcc-apache-public-circuits) | tests that need a public family skip |
| `DFTHC_REFIT_VERIFIER_DIR` | the verifiers directory itself | `$QHDL_OFFICIAL_ROOT/verifiers` |
| `DFTHC_REFIT_LERA_SRC` | the directory holding Lera's `tensorqrom.py` | `--lera-src` is required; tests skip |
| `DFTHC_REFIT_QROM_PLAN` | the qrom-plan binary | `qrom-plan` on `PATH`, else the qrom-plan backend is unavailable |
| `DFTHC_REFIT_KERNEL_SRC` | the directory holding qwertykeypadapple's `kernel_search.py` | `--kernel-src` is required; tests skip |
| `CMS_PYTHON` | an interpreter with pycryptosat (`chained_and` solver `cms`) | the current interpreter |
| other `DFTHC_REFIT_*`, `DFTHC_PRICE_*` | specific families, words or saved plans for integration tests (named in each test file) | those tests skip |

The kickmix interpreter (`$QHDL_KICKMIX_PYTHON`) has no PyTorch, so `refit` and
`price-words` run in a NumPy/SciPy/PyTorch interpreter; `recover --out DIR` hands the recovered words and the official
resources to `price-words --recover-dir DIR`. All subcommands work under `python -I`.

## Processing a new public family (recipe)

```sh
K="$QHDL_KICKMIX_PYTHON"                                # kickmix interpreter (no PyTorch)
P=python3                                               # NumPy, SciPy, PyTorch
FAM=path/to/family/dfthcsos  V="$QHDL_OFFICIAL_ROOT/verifiers"  LERA=DIR_WITH_tensorqrom.py  KS=DIR_WITH_kernel_search.py
R=runs/<new-dir>                                        # a new directory per run; never overwrite earlier results

# 0. Which conventions (gauge, loader class)? Every subcommand below refuses what it cannot handle.
$K -I dfthc_refit.py conventions $FAM --verifier-dir $V --out $R/conventions.json

# 0b. Pi-gauge family: which symmetries matter to each loader, and does a gauge change fit in 128 qubits?
$K -I dfthc_refit.py gauge-options $FAM --slot 0-14 --verifier-dir $V --out $R/gauge_options.json

# 1. What is this family? Per loader: products, coded words, low-monomial partition, relations, pairing, products
#    shared between coded and non-coded outputs; re-emit each loader from its recovered terms and check it natively.
$K -I dfthc_refit.py recover $FAM --verifier-dir $V --emit-check --lera-src $LERA --out $R/recover

# 2. Which planner fits? Static screen, nothing emitted (`toffolis_per_pass.saving` in each plans.json).
#    Triple-structured coded loaders suggest `shared`; generic loaders with a triple partition suggest `block`.
$K -I dfthc_refit.py plan $FAM --planner shared --pad-rows --verifier-dir $V --out $R/plan_shared
$K -I dfthc_refit.py plan $FAM --planner block  --pad-rows --verifier-dir $V --out $R/plan_block
$K -I dfthc_refit.py plan $FAM --planner kernel --pad-rows --verifier-dir $V --kernel-src $KS --lera-src $LERA \
    --out $R/plan_kernel

# 3. Re-plan with the best backend: emit, native check, positive-gain acceptance per slot, family checks, local
#    official verifier. To combine backends, run replan again on the previous output: each pass keeps strict gains.
$K -I dfthc_refit.py replan $FAM $R/replan --backend block --pad-rows --lera-src $LERA --verifier-dir $V --verify

# 4. Price word moves on the result (approximate logic synthesis of the angle table): recover the new family's words
#    and resources, then price, e.g., the cheapest double rank drop of coded loader 4.
$K -I dfthc_refit.py recover $R/replan/dfthcsos --verifier-dir $V --out $R/recover_after
$P -I dfthc_refit.py price-words $R/replan/dfthcsos --recover-dir $R/recover_after --verifier-dir $V \
    --rank-drop 4 --refit-method penalty --out $R/price
```

### Free gauge sources (v1.8.0-dev)

In a pi-gauge family the incoming top bit of a word whose gauge does not change in a slot holds its gauge pattern
(a function of the low address bits) on both passes the verifier runs. When that pattern is one low monomial, the
loader can use the top bit as that monomial's low source instead of caching it (Syed Asad Ali's construction in
`da4d23796b13`: 75 cache Toffolis instead of 80 on slots 1–9). `plan`/`replan --gauge-free-sources` predict and emit
with it (`gauge.GaugeEmitter(free_sources=True)`); `recover --emit-check` re-emits such partial low caches. Off by
default, so earlier emissions are byte-identical. `lowcache` reports the floor; on Heliosli's `1f0eb655dd13` every
shipped loader already meets it (gap 0), and no pure pi regauge can add sources (the losing columns are outside the
pure-pi kernel's support).

### The recipe as one command (`chain.py`, 0.2.0)

```sh
python3 -I chain.py $FAM $R/chain --verifier-dir $V --kickmix-python $K --torch-python $P --lera-src $LERA \
    [--planners shared,block | none] [--refit-runs 3] [--kickmix-dir D] [--no-free-sources] [--dry-run]
```

Runs conventions → recover → `lowcache` (0.2.0) → `plan` screen per planner (with `--gauge-free-sources` unless
`--no-free-sources`) → `replan` with the planner that predicts the largest saving,
on the slots where it predicts one (`--verify`, strict gain per slot) → arm-B `refit` chained while each run gains more
than REL_TIE (1e-5 relative), and writes `$R/chain/chain.json` (every command, return code, wall time, per-slot
shipped Toffolis, plan screen, replan result, refit runs with acceptance, final family). Logs go to `$R/chain/logs/`;
the out dir must be new; a failed step stops the chain and is recorded. `--dry-run` writes only the command list.
Tests: `test_chain.py` (a fake tool exercises order, choice, acceptance and refusal).

Record every report's hashes, and report each number at its evidence tier (`plan` and `price-words` are a static model / projection; `replan`'s
"after" cost is the official `evaluate_circuits` until `--verify` passes locally; only a hosted run of the unmodified
verifier is an official verifier PASS).

Credits for what these commands run: qwertykeypadapple (kernel-relation search, `kernel_search.py`, Apache-2.0,
loaded unmodified; padded-row baseline); Lera / marculera (the cached-product emitter `tensorqrom.py`, Apache-2.0,
loaded unmodified; it credits mrmeows' cached-product emitter and Heliosli's signed scratch); Syed Asad Ali (the block
partition of their generic loaders, recovered from the shipped loaders, which `block` uses); "i want to run gemma 9b" (the
coded-loader construction recovered from their loaders, which `shared` implements, and the soft-word idea of pricing word
moves against loader AND count, which `price-words` implements; dataset `iwanttorungemma9b/vqcc-dfthc-soft-word-loaders`
v2, Apache-2.0); lightningv08 (the pairing method behind the relation model). The planners, census, pricing and
chained-AND modules are this tool's own code.

Refits the DF-THC weights of a `dfthcsos` submission for the Kaggle competition *Google Quantum AI – Verifiable
Quantum Circuit Compilation*, keeping the angle words and the ANGLE_k loaders fixed. The qubit and Toffoli counts
therefore do not change; only the two-body weights `w` (R, B, C) and offsets `W` (R, C) in `submission.npz` move, to
lower

$$\lambda_{\mathrm{eff}} = \sqrt{E_{\mathrm{gap}}\,(2\Lambda - E_{\mathrm{gap}})}$$

while `error_1` and `error_2` stay at or below 20 mHa and `E_gap` stays above its floor.

## Method

Two steps, alternated:

1. **Constrained refit.** Projected L-BFGS on log λ_eff with `error_2` and `E_gap` held as equalities (at 19.999 mHa
   and 1e-5 Ha above the floor, or at the input's own values when those are already tighter) and `error_1` as an
   inequality (`--e1t free`, at most 19.9995 mHa; or held with `start` / a number), a Gauss–Newton retraction onto the
   constraint set after every step, Armijo backtracking, and a KKT check that releases weights held at zero when that
   lowers λ_eff. Eigenvalues of h1′ within 1e-6 of zero are kinks of Λ and `E_gap`; the block of h1′ on their
   eigenvectors is held too, so steps move along the kink rather than across it. Only exact zeros are held at zero:
   public weights often carry thousands of deliberate tiny entries.
2. **Gauge step.** For each rank, right-multiplying `[w_r ; w_B,r]` by an orthogonal C × C matrix leaves `error_1`,
   `error_2` and `E_gap` unchanged but changes Λ. Exact Jacobi rotation sweeps (plus a smoothed-L1 L-BFGS start and
   two small random kicks) minimise Λ over this freedom.

The tool stops when a round's realised gain and the first-order gain predicted from the KKT multipliers are both below
1e-5 relative, after `--patience` rounds that each gain less than `--min-gain` (1e-7), or after `--rounds` rounds.
Random gauge kicks are tried in the first gauge round only (`--kick-rounds`); later rounds start from the previous
rotation. The gauge runs per rank on up to `--gauge-procs` processes. `--arm A` runs the refit alone. The output is never worse than the input:
the input is the first incumbent when it passes the accuracy gates.

Every reported metric comes from the official verifier code (`dfthcsos.scoreinfra.dfthc_math`, unmodified), imported
from the competition's `verifiers/` directory. The words are the ones the loaders actually load, recovered with the
official `dfthcsos.verify.evaluate_circuits` (this needs a kickmix build). If no kickmix build is found, the tool falls
back to the `angle_words` stored in `submission.npz` and prints a loud warning: that copy can be stale.

## Inputs and outputs

- Input: a submission directory with `angle_k_*.kmb` (or `.kmx`) and `submission.npz`.
- Output: `OUT_DIR/dfthcsos/` (the loaders copied unchanged and the refit `submission.npz`), `OUT_DIR/report.json`
  (before and after `error_1`, `error_2`, `E_gap`, Λ, λ_eff, cost; per-round log; settings; wall time; versions;
  SHA-256 of every input and output) and `OUT_DIR/log.jsonl`.
- With a kickmix build the tool also runs the unmodified official `verify()` on the output (`report.json` →
  `verify_after`). Without it the "after" cost is a projection: new λ_eff × the input's Toffolis × qubits.

Same input and settings (including `--threads`) give the same `submission.npz` bytes.

## Usage

Locally (Python ≥ 3.10 with NumPy, SciPy and PyTorch; a kickmix build for word recovery and verification):

```sh
python dfthc_refit.py SUBMISSION_DIR OUT_DIR --verifier-dir COMPETITION/verifiers \
    [--kickmix-dir DIR_WITH_kickmix.so] [--max-evals 1000] [--rounds 10] [--arm B] \
    [--e1t free] [--e2t 19.999] [--gap-margin 1e-5] [--threads 3] [--seed 0] [--min-gain 1e-7] [--patience 3]
```

On Kaggle, attach the competition data and this utility script (`manwithacat/dfthc-refit`), then:

```python
import glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(sorted(glob.glob("/kaggle/usr/lib/**/dfthc_refit.py", recursive=True))[0]).parent))
import dfthc_refit
report = dfthc_refit.run("/kaggle/input/.../dfthcsos", "/kaggle/working/refit", max_evals=300, rounds=3, threads=4)
```

The verifiers directory and the organisers' kickmix build are found under `/kaggle/input` automatically (also through
`kickmix_tools` when it is attached). One gauge step takes a few minutes for a 9-rank, 39-column table on 3 CPU
cores; one refit round of 1,000 evaluations takes about a minute.

Tests (synthetic 6-orbital problem; set `DFTHC_REFIT_VERIFIER_DIR` to also check against the official math):

```sh
python test_dfthc_refit.py -v
```

All tests: `python -I -m unittest discover -s tools/dfthc-refit -v` from the repository root, with the kickmix
interpreter for the loader tests and a PyTorch interpreter for `test_dfthc_refit.py` (its refit tests skip
without PyTorch). Tests whose inputs are absent skip with a message naming the variable to set (section
"Environment"; `testpaths.py`). `test_cli.py` covers the v1.4 subcommands: argument parsing and dispatch, `replan` defaults
unchanged, `plan` on synthetic tables built from known paired plans, `recover --emit-check` and `plan` against the
shipped loaders of a synthetic two-slot family, and `chained-and` on a two-AND copy-gate target. With
`DFTHC_REFIT_CLI_INTEGRATION=1` it also runs `replan --backend block` on slot 0 of f4291619f2fd (`DFTHC_REFIT_F429`)
and, with `DFTHC_PRICE_FAMILY` / `DFTHC_PRICE_WORDS`, `price-words --rank-drop` (PyTorch).

## Loader re-planning (`replan`, v1.3.0-dev, not released)

`replan` lowers the angle-loader Toffolis of a family without touching its words or weights. For each slot k it
recovers the words the loaders load (official `evaluate_circuits`), re-plans loader k with the **qrom-plan** relation
planner, emits the plan with Lera's unmodified emitter (`tensorqrom.build_circuit`: 4 low / 7 high address bits,
grouped low cache, borrowed top bits) and checks it natively (zero, table and random signed-gauge words, two
applications: XOR in, restore, no phase, selection unchanged, scratch clean).

**Never-worse rule.** A slot takes the re-planned loader only if it

- has **strictly fewer** Toffolis than the shipped loader (a tie keeps the shipped file),
- passes the native check, uses at most 128 qubits, and keeps the shipped register layout, and
- has exactly the Toffolis the cost model predicts (caches + products).

The assembled family must then load exactly the input words (no chain break), have one register layout across all
slots (no layout mismatch), leave `error_1`, `error_2`, Λ, `E_gap` and λ_eff unchanged to the digit, pass the
accuracy gates and not cost more under the official `evaluate_circuits`. If any check fails, the shipped family is
written unchanged. `submission.npz` is always copied byte for byte.

**Planner cache count.** qrom-plan adds a fixed 89 cache Toffolis to its counts (right for 1,363-row tables). The tool
ranks plans by products and computes the cache Toffolis from the table itself (11 low + H − 1 − bits(H − 1) high,
H = ⌈rows / 16⌉: 80 at 1,220 rows), so the predicted count is right for any table; `report.json` records the offset.

The planner is a separate Rust binary (`qrom-plan`, kickmix-engine branch `qrom-plan`); it is **not** in the public
kickmix-engine wheel yet. Lera's `tensorqrom.py` (Lera / marculera, Apache-2.0, `dfthc-compact-pipeline` revision 5e865a8b; it credits
mrmeows' cached-product emitter and Heliosli's signed scratch) is passed with `--lera-src`; the tool records its SHA-256.

```sh
python dfthc_refit.py replan FAMILY_DIR OUT_DIR --planner PATH/qrom-plan --lera-src DIR_WITH_tensorqrom.py \
    --verifier-dir COMPETITION/verifiers [--kickmix-dir DIR] [--recipes recipes.json] [--slots 0-3] \
    [--procs 6] [--trials 12] [--samples 24] [--repair 60] [--verify]
```

- Default recipes (`DEFAULT_RECIPES`): generic loaders (no coded words) use block designs on the 8 + 3 low-cache layout,
  per 13–17, seeds 0–63; coded loaders use structured draws, counts 109–121 x seeds 0–95 (8 + 3) and 111–119 odd x
  seeds 0–63 (one 11-monomial group). Jobs run in that order; a later plan replaces the incumbent only with strictly
  fewer products, so results are deterministic for a given planner build. `--recipes` takes a JSON file of the same
  shape (`{"generic": [...], "coded": [...]}`, each recipe `name`, `mode` block/structured, `groups`, `design` for
  block, `values`, inclusive `seeds`).
- Output: `OUT_DIR/dfthcsos/` (the chosen loaders + the unchanged npz), `OUT_DIR/report.json` (per slot: shipped,
  re-planned and kept Toffolis and qubits, recipe, seed, relations, products, planner and table cache counts, native
  check, hashes; per-pass totals; family checks; cost before and after; input/output family hashes),
  `OUT_DIR/log.jsonl` and `OUT_DIR/work/` (words, every planner template, plan and JSON, every emitted loader).
- `--verify` runs the unmodified official `verify()` on the output (`report.json` → `verify_after`). Without it the
  "after" cost is the official `evaluate_circuits` at its default seed plus an arithmetic projection.
- Supported layout: brot 8 and 5 words per slot (Lera's emitter). Needs NumPy, SciPy and a kickmix build (no PyTorch).
- v1.5.0: signed-gauge families use Lera's emitter unchanged; pi-gauge families (pattern a function of the low address
  bits) use the gauge-aware emitter and the gauged native check (section "Word conventions" below); free-range
  families are refused before any planning.

### Kernel-relation backend (`--backend kernel`)

```sh
python dfthc_refit.py replan FAMILY_DIR OUT_DIR --backend kernel --kernel-src DIR_WITH_kernel_search.py \
    --lera-src DIR_WITH_tensorqrom.py --verifier-dir COMPETITION/verifiers [--kernel-seeds 0,1] [--procs 6] [--verify]
```

This backend uses **qwertykeypadapple's kernel-relation search** in place of the qrom-plan binary. The search is
`kernel_search.py` from the dataset `qwertykeypadapple/qcc-kernel-search-portfolio` v1, Apache-2.0, Copyright
qwertykeypadapple. It is loaded unmodified from `--kernel-src` and is not copied into this tool. How it works:

- Relations of each output group's ANF rows are admitted one at a time.
- Each relation is scored by the greedy pairing of rank-one features it leaves.
- All 11 nonlinear low monomials are cached at once (the full low cache, with borrowed top bits).
- Output groups are sixes, with the coded bit-0 outputs last.

Each plan is emitted with Lera's emitter (`groups=None`, `scratch=True`, as their `marculera_plan_build.py` does). It then
goes through the same native check and never-worse rule as a qrom-plan plan. About 2 CPU-minutes per slot and seed.

On their own a344f422f2e0 family it reproduces their counts: 201 products, 281 Toffolis per pass on the coded slots.
It reaches that at 126 qubits. On 1,363-row tables (C = 39) the full cache needs 135 qubits, so the backend does not
apply there.

### Padded rows (`--pad-rows`, kernel backend, v1.3.0-dev)

```sh
python dfthc_refit.py replan FAMILY_DIR OUT_DIR --backend kernel --kernel-src DIR_WITH_kernel_search.py \
    --lera-src DIR_WITH_tensorqrom.py --verifier-dir COMPETITION/verifiers --pad-rows [--no-pad-screen] \
    [--kernel-seeds 0,1] [--procs 8] [--verify]
```

Lera's emitter addresses whole 16-address blocks, so a table of A rows is loaded as ⌈A / 16⌉ × 16 rows. The rows
from A up to that size are **never read** (the verifier's selection stays below A), so their values are free. The tool
takes A from the family's own words; it never assumes a row count. If A is a multiple of 16 there are no such rows,
and the option falls back to the plain search (`report.json` → `planner.pad_rows.active`).

The free rows reach only the last high-monomial row of the ANF tensor, and only its low monomials ≥ A mod 16. With
`--pad-rows`, each loader goes through three steps:

1. **Screen.** Each word's bits 1–6 are tried as the first output group, with the last high row left free. Each is
   scored by an exhaustive one-step extension of its row span (top 64, 200 pairing restarts, one RNG at seed 0 over
   the words in order). The best group goes first (ties go to the lower word). The other free outputs follow, then
   the coded bit-0 outputs, in the sixes layout. `--no-pad-screen` keeps the plain sixes order.
2. **Search.** Their `shared_kernel` runs unmodified for each `--kernel-seeds` seed. For the first group only, the
   tool wraps its `kernel_features` lookup during the call; their file is not changed and its SHA-256 is recorded.
   The wrapped step:
   - runs their search with the last high row left out;
   - then tries the one-step extension (top 128, 200 restarts) and keeps it if it has fewer products;
   - then sets the row to a vector of the found feature span that matches the original on the low monomials the
     padded addresses cannot change.

   If no such vector exists, the group is searched as usual.
3. **Rewrite.** The padded table entries are rewritten to produce that row (a Möbius transform over the free low
   addresses; a bit 6 flip carries its bit 7 signed-gauge copy).

Every plan is checked exact against the rewritten tensor. It is emitted on the rewritten delta and native-checked on
the real rows. It then goes through the same never-worse rule, 128-qubit check, layout check and family checks as
any other plan. The real rows of every delta are asserted unchanged. The words and `submission.npz` stay byte for
byte. `report.json` records, per slot, the screen, the first group, which padded rows changed and their values.
Cost: about 1 CPU-minute per loader for the screen plus about 2 per loader and seed.

The option is opt-in. It changes which plan a seed returns, so earlier `--backend kernel` runs stay reproducible
without it.

On Syed Asad Ali's f4291619f2fd (1,220 rows; rows 1,220–1,231 never read), seeds 0,1 give 200 products (280 Toffolis) on every coded loader. The
shipped loaders have 282. The generic loaders stay shipped (222 products against 217). That makes 4,268 Toffolis per
pass, and the family is byte-identical to the hand-built family `5619c37ef6bd…`. The test re-derives loader 4
at seed 0 byte for byte.

### Block-partition backends (`--backend shared` / `--backend block`, v1.4.0-dev)

```sh
python dfthc_refit.py replan FAMILY_DIR OUT_DIR --backend shared|block --lera-src DIR_WITH_tensorqrom.py \
    --verifier-dir COMPETITION/verifiers [--seeds 0,1] [--blocks recover|search|1,5,13/2,6,14/...] [--exchange] \
    [--pad-rows] [--partitions 6] [--keep 2] [--cpu-cap S] [--ex-cap S] [--procs 2] [--verify]
```

These run this tool's own planners (sections below) in place of the qrom-plan binary or the kernel search:

- `shared`: `planner_shared.plan` on every slot (block partition, full structural annihilator, shared products
  admitted while the pairing does not get worse).
- `block`: `planner_block.plan_generic` on generic slots (the pairing-scored planner on a block partition); coded slots use
  `planner_shared.plan`. `report.json` names the module per slot.
- `--blocks`: `recover` (block default) reads each shipped loader's low-monomial partition with `recover.py`; a slot
  whose partition is not a partition into blocks of at most 4 monomials takes the family's most common valid
  partition for slots of its kind, else a partition search. `search` (shared default) draws random triples. An
  explicit partition applies to every slot.
- `--exchange` adds exchange moves (drop j relations of a block, re-admit j + 1; zero-cost replacements allowed).
- `--pad-rows` treats the never-read padded rows as external input don't-cares, as for the kernel backend; the
  planner completes the last high row and rewrites those entries. Off by default.
- CPU caps: shared admission 90 s per partition (`--cpu-cap`), exchange 300 s (`--ex-cap`); block uses the module's
  phase caps unless `--cpu-cap` is given, exchange 600 s. These are the settings of the original development runs.

Plans go through the same emission (`groups=None`, `scratch=True`), native check, cost-model check, 128-qubit and
layout checks, positive-gain acceptance and family checks as the other backends. Development results: on the
6566cd395128 coded loaders `shared` reaches 196–197 products against the shipped 196; on generic loaders `block`
reaches 217 products on three tables, equal to Syed Asad Ali's loaders, so those slots tie and stay shipped.
`plan --planner shared|block` runs the same search without emitting, for a quick screen.

Tests (`test_replan.py`, run with the kickmix interpreter; integration tests skip without the planner, Lera's source,
the verifiers, the a56ec354e8b8 family, or, for the kernel backend, their `kernel_search.py` (`DFTHC_REFIT_KERNEL_SRC`)
and the a344f422f2e0 family (`DFTHC_REFIT_A344`); the padded-row test needs the f4291619f2fd family
(`DFTHC_REFIT_F429`) and a saved padded-row plan (`DFTHC_REFIT_F429_PADROW_PLAN`)). The kernel test checks that slot 4 of a344f422f2e0 at seed 0
reaches 201 products / 281 Toffolis (a tie, so the shipped loader is kept). On Agniv Banerjee's a56ec354e8b8 family (Ayodeji's loaders, Lera's
C = 45 table) the re-plan reproduces a recorded reference candidate (family `c6d391fdc236`) byte for byte (slots 0–3
at 303 instead of 307 Toffolis per pass, 4–14 shipped); re-planning that output changes nothing.

## Chained products (`chained_and.py`, 0.2.0, not released)

A standalone module for multi-output GF(2) blocks f_w(y, z) with a small low address y (n bits), a row index z (the
16-address block) and don't-cares, under the cached-monomial model: every function of y alone and of z alone is free
(XOR of cached wires), an AND gate multiplies two operands of the form a(y) + s(z) + (XOR of earlier gates), and
Toffolis = gates. Empty chain masks are Lera's rank-2 products; nonempty ones are chained (deep) products.

```sh
python chained_and.py TARGET.json [--out report.json] [--cnf-k 9 --cnf out.cnf [--depth 1]]
python dfthc_refit.py chained-and TARGET.json bounds [--depth1-exact SECONDS] [--out report.json]
python dfthc_refit.py chained-and TARGET.json verify --circuit circuit.json [--out report.json]
python dfthc_refit.py chained-and TARGET.json encode --k 9 [--depth 1 | --chained I] [--rows all|spanning] --cnf out.cnf
python test_chained_and.py -v        # CHAINED_AND_TARGET=target.json adds a real-loader test
```

- Target JSON: `ybits`, `outputs`, `rows: [{care, tt}]` (care mask and one truth table per output over the 2^n
  values of y; a row is met up to one constant per output). A typical target is the coded bit-0 block of a loader,
  read from the shipped `.kmb` (native kickmix run on every address).
- `rank2_lower_bound`: rank-2 products >= ceil(r / 2), r = affine rank of the normalised full rows.
  `general_lower_bound`: any chained circuit needs >= d ANDs (d = output dimension modulo free functions), + 1 when
  no output combination has mixed rank <= 2.
- `construct_rank2`: per-output rank-2 products; `verify`: exhaustive over every row and care point, with given row
  bits or a search over all 4^k row settings.
- `linear_fiber_dim` and `copy_gate_example`: a 2-AND chained circuit whose outputs stay linear in y and span an
  affine dimension of 5, where rank-2 products need 3 — so the rank-2 bound is not a bound for chained circuits.
- `encode_cnf` / `decode_model` / `assignment_from`: a complete CNF (DIMACS) for "k ANDs, chain depth 1 or any
  (or an explicit layering), these rows". No SAT solver is bundled; UNSAT on any subset of rows is a proof for the
  whole target, a timeout is not. The tests check that known circuits satisfy the encoding and wrong row bits do not.

On loader 4 of `6566cd395128`: rank-2 bound 10 (affine rank 20), general
bound 6, 10-product construction verified on all 77 rows; whether 9 chained ANDs suffice is unresolved.

### Search-space reductions R1-R7 (0.2.0)

Exact reductions (each preserves SAT/UNSAT; soundness arguments in the docstrings), all generic in the target:

- R1 `mixed_matrix`, `mixed_columns`, `gate_mixed_parts`, `column_rank`: the doubly normalised quotient by the free
  functions; a depth-1 gate's matrix has column rank <= 2.
- R2 `row_classes` / `reduce_rows` / `lift_rowbits` (dominated rows) and `column_merges` / `reduce_columns` /
  `lift_columns` (columns equal up to per-output constants become don't-cares, lifted by copying the operand column).
- R3 `output_basis` / `apply_output_map` / `map_circuit_outputs` / `invert_output_map`: canonical output basis.
- R4 `gauge_canonical` / `gauge_rowbits` / `canonicalise`: one GL(2,2) representative per operand pair (key =
  chain mask then y-part; first operand largest, second smallest), pushes into later gates and outputs.
- R5 `canonical_order` / `reorder_gates`: plain gates first in non-decreasing (a, b) order, chained gates after.
- R6 `shapes` and `encode_shape(target, k, chained, rows, gauge, lex, force_chain)`: shape = number I of chained
  gates (the last I gates carry chain variables, the first k - I use a direct 8-clause product encoding); I = 0 is
  excluded exactly when 2k < mixed rank. The gauge and lex clauses are one-directional lexicographic constraints.
- R7 `cegar(target, k, chained, workdir, solver, timeout_s, ...)`: row-subset loop with kissat or cadical
  (`run_solver`; a DRAT proof file when asked), exhaustive verification of every model on the whole target, failing
  rows added; `spanning_rows` picks a row basis of the mixed row space plus the partial rows as the start.
- Exact depth-1 decision `depth1_exact` / `pure_tensors` (tight case 2k = rank: a pure-tensor basis of the row
  space, DFS in milliseconds on the loaders) and a planted positive control `plant_linear_chained` /
  `LOADER4_MIRROR_PLAN` (7 plain + 2 copy gates, five rank-4 outputs, mixed rank 20, rows exactly affine in y).

On the loaders: no row or column collapses exist (77 classes, no merges); loader 14 has an exact 9-product depth-1
circuit (first built here, verified on all 77 rows); loader 4 needs 10 at depth 1 (2 x 9 < 20) and 10 exist.


## Word-move pricing (`price_words.py`, 0.1.0, not released)

A module, and the `price-words` subcommand, that prices **word moves**, edits `(row, column) -> new word` of a family's angle-word table,
by their accuracy cost, and ranks them against the loader Toffolis a planner says they save. It imports `dfthc_refit` and does not change it. Needs NumPy, SciPy and
PyTorch; no kickmix (pass the words the loaders load, e.g. from `recover_words` in a kickmix interpreter).

```python
import price_words as pw
fam = pw.load_family(FAMILY_DIR, VERIFIER_DIR, words=words, qubits=427, toffolis=15175)
pw.shadow_prices(fam)                    # KKT prices: score units per uHa of error_2 / error_1 / E_gap
pw.finite_response(fam, (1, 2, 5, 20), ladder=True, method="penalty")   # measured price of tightening error_2
mv = pw.Move("m", ((row, col, new_word), ...))
pw.price_move(fam, mv, max_evals=600, method="penalty")                 # exact fixed-weight + bounded refit
rows = pw.rank_moves(fam, moves, pw.coded_rank_planner(), refit_top=10)  # net = saving - price, best first
```

Score units are chemistry cost units (λ_eff × Q × T); one loader Toffoli per pass is 2 λ_eff Q units (about 40,300
on the current tables), because T counts every loader twice.

Three price tiers, each a relative λ_eff change:

- **first order** (`RowPricer`): per entry, vectorised over a column. Exact decode of the changed row; error_2² to
  second order in that row's projector change; exact rank-2 update of h1′ for Λ, E_gap and error_1 with the official
  matching. Converted to λ_eff with the KKT multipliers. It matches the exact single-entry change (correlation
  0.99999999997 over 40 sampled bit-0 toggles). For sets of a few hundred entries it stays within about 10% on
  error_1² and error_2². Negative per-entry prices are first-order credits and should not be summed.
- **exact fixed** (`exact_fixed`): official `dfthc_math` metrics of the moved table at the base weights, plus the
  same KKT correction.
- **refit** (`bounded_refit`, used by `price_move`): a bounded constrained refit from the base weights on the moved
  table, with error_2 and the E_gap margin held at the base's values and error_1 free below the cap. The price is
  λ_eff after the refit against the better of the base and a refit of the unmoved table at the same budget (the
  control). Only points that pass the official gates count.
  - `method="retract"` is `dfthc_refit.refit`.
  - `method="penalty"` is `penalty_refit`, a quadratic-penalty L-BFGS-B on the free coordinates. It is needed
    where the Gauss–Newton retraction fails. On 6566cd395128 the error_2 and E_gap constraint normals are nearly
    anti-parallel (cosine −0.9997), and every retraction that tightens error_2 fails.

KKT multipliers can be ill-conditioned or have the wrong sign at a public point; `diag["wrong_sign"]` and
`kkt_e1_inactive` show this. Before using first-order prices, re-estimate them at a polished point or measure them
(`finite_response(..., ladder=True)`), and set them with `set_e2_price`.

The planner side is static (GF(2) arithmetic only):

- `anf_tensor` uses the same convention as Lera's `tensorqrom.anf_tensor`; the test checks this when
  `DFTHC_REFIT_LERA_SRC` is set and kickmix imports.
- `coded_mixed_rank` and `loader_coded_ranks` give the GF(2) rank of the coded bit-0 outputs' mixed tensor:
  "low-bit mixed rank" in the soft-word write-up by "i want to run gemma 9b". `pad_free=True` lets the
  never-read padded rows choose the free slopes of the last block.
- `coded_rank_planner` is a callback: saving = Σ_k ⌈r_old/2⌉ − ⌈r_new/2⌉. It is a lower-bound model of a rank-2
  pairing planner and is realised only when a loader is re-planned and emitted. Any callable
  `(old_words, new_words, move) -> Toffolis per pass` can replace it.
- `rank_drop_moves(fam, k, row_price=...)` finds bit-0 toggles (word XOR 1) that lower loader k's coded rank by
  one:
  - Only bit 0 moves, so the signed gauge and every free output are untouched.
  - For a combination x of slope vectors (output o, low bit i), the rank falls when the XOR of x's slopes is
    constant over the 16-address blocks.
  - Each mismatched block is fixed by toggling 8 rows of one word column (the cheaper of B_k / B_(k−1), and of
    l_i = 1 / l_i = 0).
  - Combinations are enumerated up to `max_terms`, plus a local search with `restarts`.
  - `protect` keeps loaders out of reach. `frozen_terms` keeps an earlier drop's dependency when a second drop is
    built.

On the soft-word family `6566cd395128` the coded ranks are 20 on loaders 4–13. Loader 14 is at 19, or 18 with the
padded slopes free, which matches the write-up's "20 → 18" and that loader's 275 Toffolis.

Tests: `python test_price_words.py -v`. The pure tests (GF(2), ANF against the definition, coded rank, rank-drop
construction, moves, price arithmetic) run anywhere. The integration tests need `DFTHC_REFIT_VERIFIER_DIR` and
`DFTHC_PRICE_FAMILY` (a dfthcsos directory, plus `DFTHC_PRICE_QUBITS` and `DFTHC_PRICE_TOFFOLIS`).

### `price-words` subcommand

```sh
python dfthc_refit.py price-words FAMILY_DIR [--recover-dir DIR | --words W.npy --qubits Q --toffolis T] \
    [--moves moves.json | --rank-drop K [--drops 2] [--keep 8] [--max-terms 3] [--unit-row-prices]] \
    [--refit-method retract|penalty|ladder] [--max-evals 300] [--refit-top 10] [--out DIR]
```

Always prints the shadow prices (`shadow_prices`) and writes the coded mixed ranks per loader. With `--moves` (a JSON
list of `{name, changes: [[row, column, new_word], ...]}`) or `--rank-drop K` (bit-0 toggle moves that lower loader
K's coded rank by `--drops`, chosen by the per-entry first-order price with no credit for negative prices; two drops are two successive
single drops), it ranks the moves with the static coded-rank planner: saving (Toffolis per pass), price
(first order, or a bounded re-refit for the `--refit-top` best by first-order net) and net, best first.

## Word conventions and the gauge-aware emitter (`gauge.py`, v1.5.0-dev)

The official decoder reads word w as θ/π = w / 2^(brot−1) on the family's Givens tree, applied to e₀; only u uᵀ enters
the score. Lera's emitter, the planners and `recover` assumed the **signed gauge** (bit 7 = bit 6 on every word). Two
other conventions exist:

- **pi gauge** (Atakan Aldemir's tree and loaders: `aba448aeadb5`, `3f2407425257`, `726299a7f65b`): 20.6–26.1% of
  the words carry bit 7 = bit 6 ⊕ 1, i.e. θ + π. Flipping those top bits back (the canonical signed form) leaves every
  row's u uᵀ unchanged, so it is an exact symmetry. The flip p(row, column) depends only on the low four address
  bits (row mod 16): 8 row patterns.
- **free range**: off-gauge words whose canonical form changes u uᵀ, so not a symmetry. In the 2026-11-05 census the
  three such families (`88fbbb2cc7f5`, `d10ea5bbb6a9`, `4af60d870d82`) are free range only in their **stale**
  `submission.npz` `angle_words`. The words their loaders load are in the signed gauge.

```sh
python -I dfthc_refit.py conventions FAMILY_DIR [--out FILE.json] [--no-loaders] [--words W.npy]
```

`conventions` reports the storage range (unsigned / signed), the tree (path, star, binomial or a hash), the gauge class
with its pattern group, whether the pattern is a function of the low address bits, and, with kickmix, the loader
class per slot:

- **xor-in**: the loader maps x to x ⊕ (b_k ⊕ b_(k−1)) cleanly on the verifier's inputs (b_(k−1) forward, b_k in
  reverse) and on random words of the incoming gauge;
- **chained**: correct only on b_(k−1) ↔ b_k;
- **broken**: otherwise.

`needs_incoming_gauge` marks an XOR-in loader that is not an XOR oracle on random signed-gauge words. That is a
loader whose borrowed top bit expects the incoming gauge. Atakan Aldemir's loaders 1–14 are such loaders. On zero
or signed-gauge input, slot 4 is wrong on 685 of 1,220 rows. A zero-input test reads this as "chained"; they are
not chained. Words come from the loaders (official `evaluate_circuits`) when kickmix is present, else from the npz. The
npz copy can be stale.

Every family subcommand runs the detector first and refuses what it cannot handle, with a one-line reason. It does
not fail inside.

| Command | Signed | Pi gauge | Free range |
| --- | --- | --- | --- |
| `refit` | yes | yes (conventions recorded in `report.json`) | yes |
| `recover` | yes | yes: reads each loader with the incoming gauge seeded on the top bit (`recover_loader(..., gauge_in=...)`); `--emit-check` re-emits only full-low-cache plans | yes; chained slots are listed and skipped |
| `plan` | yes | yes (plans use bits 0..6 of the XOR delta) | refused |
| `replan` | Lera's emitter, unchanged | gauge-aware emitter (below) when the pattern is a function of the low address bits, else refused | refused |
| `price-words` | yes | on the canonical signed form; moves are translated in and out (`moves_family_gauge` in the report) | shadow prices only; moves refused |

**Gauge-aware emitter** (`gauge.build_circuit_gauge`, wrapped by `gauge.GaugeEmitter`). This is Lera's
cached-product emitter (`tensorqrom.build_circuit`, Lera / marculera, Apache-2.0; it credits mrmeows' cached-product
emitter and Heliosli's signed scratch). It keeps Lera's planning model, products, pure corrections and Toffoli count
(caches + products), and changes four things:

1. The low-monomial cache is built first, on fresh qubits.
2. The incoming gauge p_in(low bits) is cleared from each borrowed top bit through that cache before the bit is lent
   to the high cache.
3. At the end, bit 6 of the new word and p_out are put back, after the high cache is erased and before the low cache
   is.
4. A word whose gauge changes across the slot (p_in ≠ p_out) keeps its top bit as an output that also receives
   p_in ⊕ p_out. This costs one qubit per such word (the 1.5 "kept" layout).

**In-place gauge change (v1.6.0-dev; `build_circuit_gauge(..., inplace=True)`).** After bit 7 ^= bit 6 the
top bit of word j holds p_in,j(l) ⊕ dir·g_j(l), with g = p_in ⊕ p_out and dir = 0 on the forward pass, 1 on the reverse
pass. The dir part is real information on the addresses where g_j = 1, so that bit cannot be lent clean on its own.
But if g_j is the XOR of other changing words' columns (the keepers, a GF(2) basis of g's columns), XORing their top
bits into word j's cancels dir and leaves an address function, which the low cache clears: word j's top bit is lent
like an unchanged word's. Only rank(g) top bits stay unlent; at the end the dependents are rebuilt from the keepers,
every changing word receives g(l) and every word its new bit 6. Toffolis are unchanged (CNOTs only). With CNOTs and
address-controlled X gates only, rank(g) unlent bits is also the minimum (any such schedule maps the dir part to
M·span{g(l)}, an r-dimensional space that needs r nonzero coordinates). The construction is the one in Dmitrii
Gluzdov's loaders (`726299a7f65b`, slots 11–13: CX ladders among the top bits at both ends); the shipped widths of
Atakan Aldemir's and Dmitrii Gluzdov's loaders equal 126 + rank(g) on every slot.

`replan --gauge-inplace auto|never|always` (default `auto`): `auto` emits the 1.5 layout and switches to the in-place
one only when the 1.5 circuit exceeds 128 qubits, so every loader that fitted before is emitted byte for byte as
before (previously replanned loaders re-emit identically). Qubits do not enter the score below the cap
(Q_total = max(heuristic temporary, loader temporary)).

Only the full low cache (`groups=None`, `scratch=True`: the shared, block and kernel backends) is supported. For a
signed-gauge slot `GaugeEmitter.build_circuit` calls Lera's function unchanged, so signed families emit byte for
byte as before. The native check for a gauged slot (`gauge.native_check`) uses b_(k−1), b_k and random words of the
incoming gauge as inputs.

Tests: `test_gauge.py` covers the classes, canonical round trips, the decoder against the official
`reconstruct_u_vectors`, and the default refit targets. With kickmix and Lera's source it also covers:

- a synthetic pi-gauge slot that passes the gauged native check (Lera's emitter fails on it);
- a gauge change that keeps one top bit;
- slot 0 from zero;
- signed slots that are byte-identical to Lera's output;
- `recover` reading a gauge-emitted loader back.

With `DFTHC_REFIT_ABA` (or `QHDL_DATASET`) it checks the `aba448aeadb5` census and recovers its loader 4.
`test_gauge_options.py` (v1.6) covers the keeper basis, the width model, the free / constrained / costed partition of
the group, trivial signed contracts, the in-place layout (rank(g) unlent bits, native PASS, Toffolis equal), slot 0,
the `auto` policy (byte-identical under the cap, in place over it) and, with `DFTHC_REFIT_726299_PLANS` (a saved replan
of the coded slots), a slot-12 plan on `726299a7f65b` (1.5 layout byte-identical to that replan's 129-qubit loader;
in place 128 qubits, 276 Toffolis, PASS).

On `aba448aeadb5`: `replan --backend shared --pad-rows --exchange --slots 8,11,12,14 --seeds 0,1,2,3`
emits 196-product plans on all four slots (276 Toffolis each, against the shipped 277). Each passes the gauged
native check at 126–127 qubits. The family passes the family checks (words, metrics and layout unchanged) and the
local official verifier: 4,216 loader Toffolis per pass and cost 303,706,604.20 against 303,866,734.23. This takes
13 min wall and 35 CPU-min. `recover` with the gauge reads all 15 of Atakan Aldemir's loaders (function check
exact; coded slots 200–201 products + 76–77 caches). `price-words --rank-drop 8` prices on the canonical form, where
λ_eff is identical to the digit.

### Per-loader gauge options (`gauge-options`, `gauge_options.py`, v1.6.0-dev)

```sh
python -I dfthc_refit.py gauge-options FAMILY_DIR --slot 11-13 [--plans REPLAN_OUT --lera-src DIR [--emit-dir D]]
    [--structural N] [--max-basis M] [--out FILE.json]
```

The exact symmetries of the table (`gauge_search.py`) are, per row, Z_2^75: each pi-flip set f with its
unique negation set n(f); the pure-pi kernel (n(f) = 0) has 32 dimensions on the 75-column binomial tree. Slot k sees
a move only through the columns W of batches k−1 and k. Per slot:

- **costed**: n(f)|W ≠ 0. A negation changes bits 1–6, so the planners' tensor changes (Toffolis can move).
- **constrained**: n(f)|W = 0 but the pi flips differ between the two batches on some word: the gauge change g moves,
  so the width moves (1.5 layout: +1 qubit per changing word; in place: rank(g)).
- **free**: neither: n(f)|W = 0 and equal pi flips on both batches word by word.

The report adds the emitter's per-slice constraint (one move per low slice, avoiding words 64 / 192:
`allowed_per_slice`), the contract (p_in, p_out, address-only, g, changing words, rank, keepers and dependents, static
widths of both layouts), per constrained basis direction the best single-slice qubit price on slots k−1..k+1, with
`--structural N` planner_shared's structural count on N random allowed variants, and the suggested repairs (in-place
gauge change; re-gauge when even that exceeds 128; costed moves to price with the planner). `--plans` takes a replan
`OUT_DIR` (best plan per slot: fewest products, first file on ties, as replan) and emits it in both layouts.

Results (local runs):

| Family, slot | free / constrained / costed | changing words, rank | width 1.5 / in place (static) | shipped |
| --- | --- | --- | --- | --- |
| `726299a7f65b` 11 | 68 / 5 / 2 | 4, 2 | 130 / 128 | 278 Toffolis, 128 Q |
| `726299a7f65b` 12 | 66 / 5 / 4 | 3, 2 | 129 / 128 | 278, 128 |
| `726299a7f65b` 13 | 68 / 5 / 2 | 3, 2 | 129 / 128 | 279, 128 |
| `aba448aeadb5` 8 | 62 / 5 / 8 | 0, 0 | 126 / 126 | 277, 126 |
| `6566cd395128` (signed) 0–14 | 60–70 / 5 / 0–10 | 0, 0 | 126 / 126 | 275–296, 126 |

On `726299a7f65b` a replan's rejected slot 11–13 plans (277 / 276 / 277 Toffolis; 130 / 129 / 129 Q in the 1.5 layout,
re-emitted byte-identical) emit in place at 128 Q, pass the gauged native check, and the family with them (on that replan's
final output) passes the local official verifier at 306,847,819.52 (λ_eff unchanged, Q 427, T 15,223) against
307,049,388.08. On `aba448aeadb5` the classification reproduces `gauge_search.py`'s analysis: the pure-pi kernel is never costed, slots
11–14 have no costed direction at all, and slot 8's costed variants stay at the structural 198. The kept layout of
`aba448aeadb5` slots 0–3 would need 129–130 Q (Atakan Aldemir's shipped loaders: 127–128 = in place), so a generic-
slot replan there needs the in-place layout too.

**Refit default targets (v1.5.0).** A start that the first step could only leave by giving ground is held at its own
value instead of stalling the retraction:

- error_2 within 1e-6 mHa of the limit;
- with `--e1t free`, error_1 above the inequality cap (19.9995 mHa).

`report.json` → `targets_held_at_start` names the held targets. Before 1.5.0 the same effect needed
`--e1t <input> --e2t <input>` by hand.

**Explicit targets are never held (v1.7.0).** The two holds apply only to a target left at its default. An
explicit `--e2t` is the error_2 target even when the start is within 1e-6 mHa of the limit (the 1.1.0–1.4.0 rule,
noted `e2_explicit_at_limit`), and an explicit `--e1t free` is the plain inequality. 1.5.0 and 1.6.0 held error_2 even
against an explicit `--e2t`, so a recipe with an explicit target at the limit (`--e1t 19.9998606 --e2t 19.9999997` on
`aba448aeadb5`, three chained runs) gave a different npz there; 1.7.0 reproduces the 1.4.0 npz `2d53f957…` byte for
byte. `report.json` →
`settings.e1t_explicit` / `e2t_explicit`.

**`--brot` (v1.7.0-dev).** Some public npz files have no `brot` field
(Syed Asad Ali's `eff6a4863f08`); the official verifier reads brot from the circuits' word registers. `refit` resolves
it in this order: the npz field, then `--brot`, then the circuits' word-register width (with a kickmix build); any two
that are given must agree; with none, it stops and asks for `--brot`. `report.json` → `words.brot_source`.

## Plan replay (`replan --plans-from`, v1.7.0-dev)

`replan` searches with planners whose CPU-second caps (`--cpu-cap`, `--ex-cap`) can bind, so a search on a slower
machine or on the Python GF(2) backend can find different plans. Replay removes the search: every plan JSON that the
shared, block and kernel backends write (`work/plans/k<slot>/*.json`, with `terms`, `pad_rows`, `exact`) can be
emitted again.

```sh
$K -I dfthc_refit.py replan FAMILY_DIR OUT_DIR --plans-from PLANS_DIR --slots 8,11,12,14 \
    --lera-src DIR_WITH_tensorqrom.py --verifier-dir V [--verify]
```

Per slot: the plan files of the first layout found (`PLANS_DIR/work/plans/k<slot>`, `PLANS_DIR/plans/k<slot>`,
`PLANS_DIR/k<slot>`); the fewest-product plan marked exact (ties: the first file in sorted order, as replan chooses
among seeds); the plan's padded rows applied (rows below the table's row count are refused); exactness re-checked
against this family's delta (`planner_shared.exact_rows`), so a plan made for other words is refused and the slot keeps
its shipped loader. Emission, native check, strict-gain rule and family checks are those of a searched plan. Planner
options (`--backend`, `--seeds`, `--blocks`, `--exchange`, caps, `--pad-rows`, `--recipes`) are refused with
`--plans-from`. `report.json` → `planner.backend = "replay"`, `planner.per_slot` (files, choice, hash, exactness), and
per slot `replanned.replayed_plan`.

Check: four saved plans for slots 8, 11, 12, 14 (shared planner, seeds 2, 1, 0, 2) replayed on family `9706e2c7ecff`
(a weight refit of `aba448aeadb5`) give family `81db735f32b5…` byte for byte in about 40 s with the Python GF(2)
backend (the full saved plan directory, 16 files, gives the same choice). Tests: `test_replay.py`.

## Limitations

- It is a local method on a non-convex problem (L1 cusps in Λ, an eigenvalue assignment in `error_1`, a quartic
  `error_2`, the gauge freedom). No lower bound on λ_eff is computed, so nothing is claimed about global optimality.
- Weights that have already been refitted often sit where one or more eigenvalues of h1′ are almost exactly zero. The
  tool holds that block (v1.1.0); before that the retraction oscillated there and only the gauge step moved.
- Gains on recent public weights are small (1e-5 to 1e-4 relative). The "after" cost in `report.json` is a projection
  unless `verify_after` shows the official verifier's PASS; the competition's own scoring uses its own salt.
- `replan` reaches the shipped counts only where the planner's template class covers the shipped plans: on the C = 45
  coded loaders the shipped forms use 3–4 low monomials, the planner at most 2, so those slots stay shipped.
  The planner's repair step has a time budget (`--repair`), so a heavily loaded machine could in principle change
  which plan wins.
- Bit-for-bit reproducibility holds for the same library versions and thread count; it is not promised across
  platforms or versions.

Method and results: technical note "Refitting DF-THC weights on a fixed angle table", https://doi.org/10.5281/zenodo.23930466

Licence: Apache-2.0 (see `LICENSE`). Written with an AI model (Claude) under the author's direction. Not affiliated
with Google, Qualtran or Kaggle.

## Loader plan recovery (`recover.py`, v1.3.0-dev, not released)

```sh
python -I recover.py FAMILY/angle_k_K.kmb WORDS.npy K [--lera-src DIR --emit OUT_DIR] [--json OUT.json]
python -I dfthc_refit.py recover FAMILY_DIR [--slots 0-3] [--words W.npy] [--emit-check --lera-src DIR] [--out DIR]
```

Reads a shipped ANGLE_k loader emitted in the cached-product style of Lera's `tensorqrom.build_circuit` (any family,
generic or coded, any table size) and recovers its plan. `WORDS.npy` holds the words the family loads (for example
`dfthc_refit.recover_words(...)["words"]`).

- **Terms.** The circuit is executed symbolically over GF(2). A CCX onto a zero qubit with two single-monomial
  controls is a cache; every other CCX is a product, read back as Lera's term (a1, b1, a2, b2, s). A selection-only
  CX/X frame undone by its mirror is supported. The tool asserts that scratch ends clean and the words are restored.
- **Function.** The recovered truth table must equal the loader delta on every real row. The never-read padded rows
  that the shipped loader filled differently from the default padding are listed.
- **Structure** (exact, `report["structure"]`):
  - the feature space W and its relations;
  - whether W contains every row;
  - the coordinate blocks and their relations against each block's row annihilator;
  - the low-monomial and output partitions;
  - "cells" (low block × non-coded / coded outputs);
  - which blocks each product's two features fall in, and how many products carry coded outputs only,
    non-coded outputs only, or both.
- `--emit` rebuilds the recovered terms with Lera's unmodified emitter and runs the native check.

Tests: `test_recover.py`. The pure and round-trip tests use a synthetic table; the integration test uses loader 4 of
the 6566cd395128 family (environment variables in the file).

## Shared-product block planner (`planner_shared.py`, v1.3.0-dev, not released)

This planner implements the construction recovered from the coded loaders of "i want to run gemma 9b"'s soft-word
family 6566cd395128 (their loaders descend from Ayodeji's and
Syed Asad Ali's coded loaders). It is parameterised only by the loader's delta, its row count and its coded words:

1. The 15 low monomials are split into five triples, with the single-bit monomials 1, 2, 4 and 8 in different
   triples. The partition is searched (random draws, scored by the structural plan), or given.
2. The coded outputs (bit 0 of a coded word, affine in the low address bits) are zero off the single-bit monomials.
   Those coordinates are admitted as trivial relations.
3. Per triple, the annihilator of the rows on its non-coded coordinates is admitted in full (`struct_drop=0`), or
   minus a random codimension-`struct_drop` subspace.
4. Further forms are the combinations of two or more single-bit monomials, with any outputs (`singles_forms="any"`,
   as in loader 14 of 6566cd395128) or coded outputs only.
5. Relations from each triple's kernel that also involve coded coordinates are admitted one at a time while the
   paired product count does not rise. The kernel quotient is enumerated when it is small; these are the shared
   products.
6. The never-read padded rows are don't-cares: the last high row is completed into W and the padded entries are
   rewritten.
7. Pairing uses `pair_lines` from this tool's pairing-scored planner: exact candidate lines, randomised greedy selection, 1-out/2-in
   augmentation, then singles.

Plans are exact against the (rewritten) delta and are emitted with `tensorqrom.build_circuit(..., groups=None,
scratch=True)`. That gives the full low cache at 126 qubits on the five-word layout, with Toffolis = cache Toffolis
+ products. The search is randomised: results depend on the seed and the
CPU cap.

**Status: the construction is not yet fully reproduced.** On loader 4 of 6566cd395128, whose shipped
plan has 196 products, the structural start pairs perfectly at 198. With the recovered partition, shared admission
reached 196 in some development runs. From the words alone (partition search, seeds 0 and 1) the registered runs
stopped at **197** (277 Toffolis), one short of the shipped 276. The shipped plans differ in two ways:

- they admit 12–15 (not all 14–15) of each triple's non-coded annihilator;
- they admit more relations that mix coded and non-coded coordinates (6 on loader 4; 8, plus 2 coded-only, on loader 14).

Admission from the full annihilator stalls at 133 relations.

Tests: `test_planner_shared.py`. These cover synthetic generic and coded tables built from a known paired plan, the
padded-row rewrite against `apply_pad_changes`, and loader 4 of 6566cd395128 with its recovered partition.

### Exchange moves (`planner_shared.exchange`, v1.3.0-dev)

Monotone admission stalls one or two relations short of the shipped plans. Each exchange move picks a block, drops a
random codimension-j subspace of its admitted relations (j in `drops`, default 0, 1, 1, 2) and re-admits up to j+1
relations from the block's kernel. The first re-admitted relation must lie outside the dropped span. Each step keeps
the candidate with the best achieved pairing among the `patience` lowest by the pairing-scored kill score, and only while the
product count does not rise and the pairing stays within `max_excess` of ceil(dim W / 2).

- A move that ends with more relations is progress. Its pairing is re-paired at once if it is not perfect.
- A move that ends with as many relations and a new span is a neutral (plateau) step.
- `scope="own"` draws candidates from the block's kernel on its own coordinates. On the 6566cd395128 coded loaders
  the shipped relations all lie there, and the quotient is small enough to enumerate. On generic loaders it is the
  whole block kernel. `scope="global"` adds every coded coordinate, as shared admission does.

Supporting changes, all optional, with defaults that keep the earlier behaviour:

- `pair_lines(..., aug="quotient")` runs the 1-out/2-in augmentation in the quotient by the remaining pairs and skips
  it once the pairs span W. That made a pairing 20–200 times faster in development.
- `pair_lines(..., seed_lines=...)` warm-starts the pairing from a previous one.
- `lines(..., cache=...)` memoises the form-space intersections.
- `plan(..., exchange_kw=...)` runs exchange moves after shared admission.

## Generic-loader block planner (`planner_block.py`, v1.3.0-dev)

This is the tool's pairing-scored planner ported onto `planner_shared` for generic loaders (no coded
outputs) on a given block partition. For Syed Asad Ali's f4291619f2fd table that partition is 1,5,13 / 2,6,14 /
3,7,15 / 4,8,11 / 9,10,12; on other families, read it from a shipped loader with `recover.py`. It runs in
three phases:

1. The form-space cut rule. Relations are ranked by how few form spaces they shrink and kept while the pairing stays
   perfect. The pairing is `pair_lines`.
2. Pairing-scored admission (`admit_shared` with template-protecting candidates).
3. Optionally, exchange moves.

`plan_generic(delta, real_rows, blocks, seed=..., exchange_kw=...)` returns terms that are exact against the
rewritten delta, for `tensorqrom.build_circuit(..., groups=None, scratch=True)`. Tests: `test_planner_block.py`, plus the
`ExchangeTests` in `test_planner_shared.py`.

**Status.** On loader 4 of 6566cd395128, exchange moves after shared admission reached the
shipped **196** on seed 0, but stopped at 197 on seed 1. Both plans are exact, 126 qubits, native PASS. With the
generic block planner, every run on three tables (5619c37ef6bd, 160fc1ec, 6566cd395128) ends at exactly **217 products /
91 relations**. That equals Syed Asad Ali's generic loaders and is one above the best public generic loaders seen (216/215), whose
loaders 1–3 are not triple-structured.

## Rust backend for the GF(2) primitives (`backend="rust"`)

The GF(2) work in `recover.py` (`Basis`, `rank`, `nullspace`, `structure`), `planner_shared.py` (`intersect`, the
pairing passes of `pair_lines` and `augment_quotient`, `terms`, the kernels and form spaces), `planner_block.py`
(`plan_generic`) and `price_words.gf2_rank` can run in Rust: `backend="rust"` on `plan`, `plan_generic`,
`SharedPlanner`, `intersect`, `rank`, `nullspace`, `structure`, `make_basis` and `gf2_rank`. The default is
`"auto"`: Rust when `kickmix_analysis` imports, Python otherwise (`"python"` stays the reference). Choose explicitly
with the global CLI flag `--gf2-backend {python,rust,auto}` (any subcommand, any position; it sets the variable for
worker processes) or `DFTHC_GF2_BACKEND`; the test suite reads the variable. Pivots, reduced vectors and combination masks match the Python bit for bit. That keeps
the planners' random paths, and so their plans, byte-identical. The planning rules (pairing order, admission,
exchange) stay in Python. Only the elimination moved, as object-level calls on whole bases and line tables.

The Rust code is the `kickmix_analysis` module (crate `crates/kmx-analysis` of the kickmix engine's analysis
extension). It is not part of the verify wheel and not on Kaggle. Two ways to run it:

```sh
# an interpreter with the analysis extension installed (no kickmix needed)
"$QHDL_ANALYSIS_PYTHON" -I -B test_gf2_backend.py -v
# any interpreter, e.g. the kickmix one for the loader tests: point at the package (read even under python -I)
DFTHC_GF2_BACKEND=rust KICKMIX_ANALYSIS_PATH=ENGINE_DIR/crates/kmx-analysis/python \
    "$QHDL_KICKMIX_PYTHON" -I -B -m unittest discover -s tools/dfthc-refit -v
```

Without the module, `backend="rust"` raises an ImportError that names both options. Parity covers 3 × 10⁵ randomized basis operations, plus nullspace, rank,
intersect and line-table cases, all identical. 180 plans were identical in products, terms and rewritten deltas: 15
loaders × seeds 0–3 for the 5619c37ef6bd block planner, 5619c37ef6bd shared and 6566cd395128 shared. The loader-4 control (6566,
seed 0) went from 35.2 to 5.8 CPU-s with the same 197-product plan. Across the sweep the median per plan was 22× on
coded loaders and 4.7× with the generic block planner (`augment_quotient`). `test_gf2_backend.py` compares the two
backends directly and skips when the module is missing.
