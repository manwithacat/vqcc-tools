# vqcc-tools

Circuit-analysis and chemistry tooling for the Kaggle competition
[Google Quantum AI – Verifiable Quantum Circuit Compilation](https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation)
(VQCC). The competition scores three fixed tasks compiled to [kickmix](https://github.com/quantumlib/Qualtran/tree/main/kickmix)
circuits: `invsqrt`, `dqi` and `dfthcsos` (FeMoCo chemistry). This repository holds:

- **QHDL** (`scripts/qhdl.py` + `scripts/qhdl_*.py`): one agent-facing command line that answers questions about a
  `.kmb`/`.kmx` circuit (width peak, executed Toffolis, what a block of ops computes, rewrite targets) and prints every
  answer as one JSON envelope with an evidence tier and a `rerun` command.
- **dfthc-refit** (`tools/dfthc-refit/`): refit of the DF-THC weights on fixed angle words, ANGLE_k loader planner,
  loader recovery and word pricing for the `dfthcsos` task. See its own [README](tools/dfthc-refit/README.md).
- **Analysis scripts** (`scripts/`): an exhaustive chemistry loader checker, the invsqrt evaluation service, a local
  runner for the official verifiers, chemistry parts ledger/assembler, static resource models (`d1_` … `d3_`) and
  chemistry calculators/searches (`d4_` … `d11_`).

The official verifiers are **not** included. Download the competition data from Kaggle and point
`QHDL_OFFICIAL_ROOT` at the directory that contains `verifiers/`.

## Maturity

| Label | Meaning |
| --- | --- |
| **Release** | deterministic, versioned, tested, stable interface |
| **Beta** | deterministic and tested, but interfaces may still change and there is no versioned release |
| **Alpha** | research code, work in progress; interfaces and results may change |

| Tool | Maturity | Note |
| --- | --- | --- |
| QHDL read-only analysis verbs: `circuits`, `open`, `peak`, `exec`, `profile`, `runs`, `idle`, `explain`, `evidence`, `addr`, `window`, `windows`, `fdiff`, `track`, `census`, `measure-first`, `retain`, `diff`, `schedule`, `valuegraph`, `doctor`, `verify` | Beta | covered by `tests/test_qhdl_*`; JSON envelope fields are stable in practice but unversioned |
| QHDL exploratory analysis verbs: `avenues`, `parts`, `recompute`, `valuealign`, `predicate-families`, `phase-profile`, `optimizations`, `simulate`, `inv` | Alpha | newer, research-driven; output fields still move |
| QHDL rewrite / edit verbs: `rewrite`, `measure-out`, `recolour`, `edit`, and `qhdl_surgery.py`, `qhdl_park_*.py`, `qhdl_rewrite_*.py` | Alpha | each emitted circuit must still pass the official verifier |
| `tools/dfthc-refit` | Beta | carries a version string (1.8.0-dev) and its own test suite, but no tagged release |
| `chem_loader_checker.py`, `invsqrt_eval.py` | Beta | deterministic, tested |
| `local_verify_reference.py` | Beta | thin runner over the unmodified official verifiers |
| `dqi_fast_rank.py` | Beta | pure helpers tested; engine side needs `kickmix_engine` |
| `chem_parts_ledger.py`, `chem_parts_schema.py`, `chem_family_check.py`, `chem_assemble.py`, `chem_word_tables.py` | Beta | tested where the public dataset is present |
| `chem_c12_bounds.py`, `chem_c12_walk.py`, `d1_` … `d11_` models, calculators and searches | Alpha | research calculators; mostly untested here |

## Evidence tiers

Every number these tools print belongs to exactly one tier. Never present a lower tier as a higher one.

| Tier | Meaning |
| --- | --- |
| **static model / projection** | arithmetic on counts or formulas; no circuit is run |
| **local prescreen** | the official equations or a simulator (BitSim, kickmix engine, sampled or exact) run locally |
| **official verifier** | the unmodified official verifier run on a circuit (locally with kickmix, or hosted on Kaggle) |

QHDL envelopes use finer labels (`static`, `sampled(n)`, `exact-all-x`, `engine-parity`, `official-local`,
`official-hosted`), lowest to highest. **The official verifier is the only authority on a score**; a hosted Kaggle
run is the authority for anything that will be submitted, and leaderboard salts differ from local runs.

## Install

Python 3.12+ with NumPy. SciPy is optional (some chemistry calculators); PyTorch is needed only for
`dfthc_refit.py refit` / `price-words`.

```sh
git clone <this repository> vqcc-tools && cd vqcc-tools
python3 -m pip install numpy            # or: python3 -m pip install -e '.[chem,test]'
python3 scripts/vqcc_config.py          # prints the resolved settings
```

Some features need extra software, each optional:

- **kickmix** (official simulator; needed for `.kmx` conversion, `verify`, loader checks):
  the [kickmix-macos fork](https://github.com/manwithacat/kickmix-macos) builds natively on macOS; on Kaggle use the
  wheels in the dataset `manwithacat/kickmix-fork-wheels`. Point `QHDL_KICKMIX_PYTHON` at an interpreter that can
  `import kickmix`.
- **kickmix-engine** (fast verifier engine for `invsqrt_eval.py`, `dqi_fast_rank.py`, DQI gate draws):
  [manwithacat/kickmix-engine](https://github.com/manwithacat/kickmix-engine); Kaggle wheels in the dataset
  `manwithacat/kickmix-engine-wheels`. Install the wheel, or set `QHDL_ENGINE_PKG` to a checkout's `python/` directory.
- **Public circuits dataset** (circuit lookup by SHA prefix, chemistry parts):
  [vqcc-apache-public-circuits](https://www.kaggle.com/datasets/manwithacat/vqcc-apache-public-circuits).
- **Official verifiers**: from the competition data (not redistributed here).

## Configuration

All paths are environment variables (see `scripts/vqcc_config.py`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `QHDL_DATASET` | `data/vqcc-apache-public-circuits`, else `/kaggle/input/vqcc-apache-public-circuits` | public circuits dataset (`manifest.csv`, `circuits/<task>/`, `parts/`) |
| `QHDL_OFFICIAL_ROOT` | `data/official_full`, else `/kaggle/input/google-verifiable-quantum-circuit-compilation` | directory containing the official `verifiers/` |
| `QHDL_WORK` | `work/` | QHDL workspaces and caches |
| `QHDL_OUTPUTS` | `outputs/` | where recorded verifier reports are looked up |
| `QHDL_KICKMIX_PYTHON` | the running interpreter | a Python that can `import kickmix` |
| `QHDL_ENGINE_PKG` | empty (installed `kickmix_engine`) | a kickmix-engine checkout's `python/` directory |
| `QHDL_ANALYSIS_PYTHON` | the running interpreter | interpreter with the engine analysis extension (`kickmix_analysis`) |
| `QHDL_VIEWER_DIR` | `external/kickmix-macos/tools/circuit-viewer` | the fork's circuit viewer (optional stage maps) |
| `QHDL_WORKBENCH` | `$QHDL_VIEWER_DIR/workbench` | the fork's `workbench` launcher (DQI-sized region summaries for `qhdl diff`) |
| `QHDL_PARTS_CATALOGUE` | none | the modular-parts catalogue for `qhdl parts` (Kaggle dataset `manwithacat/vqcc-modular-circuit-parts`) |

`dfthc-refit` reads its own `DFTHC_REFIT_*` variables; see its README.

In a Kaggle notebook, attach the public circuits dataset, the competition data and the kickmix wheels, then:

```python
import os
os.environ['QHDL_DATASET'] = '/kaggle/input/vqcc-apache-public-circuits'
os.environ['QHDL_OFFICIAL_ROOT'] = '/kaggle/input/google-verifiable-quantum-circuit-compilation'
os.environ['QHDL_WORK'] = '/kaggle/working/qhdl-work'
```

## Quick examples

```sh
# Open a circuit (a path, or a SHA-256 prefix of a circuit in the dataset), then follow the `next` suggestions.
python3 scripts/qhdl.py open -c path/to/invsqrt.kmb
python3 scripts/qhdl.py peak -c 47b318
python3 scripts/qhdl.py runs -c 47b318
python3 scripts/qhdl.py window -c 47b318 --ops 1200:1260
python3 scripts/qhdl.py doctor                      # what optional software is available

# Run the unmodified official verifiers locally (needs kickmix and the competition data)
$QHDL_KICKMIX_PYTHON scripts/local_verify_reference.py --root "$QHDL_OFFICIAL_ROOT" \
    --submission DIR_WITH_KMBS --out outputs/run-1/report.json --tasks invsqrt

# invsqrt evaluation service (kickmix-engine); see docs/invsqrt-evaluation-service.md
python3 -I scripts/invsqrt_eval.py verify CANDIDATE.kmb

# Exhaustive classical check of one chemistry ANGLE_k loader
$QHDL_KICKMIX_PYTHON -I scripts/chem_loader_checker.py --help
```

The question → command index is [docs/agent-circuit-tools.md](docs/agent-circuit-tools.md). Other docs:
[invsqrt evaluation service](docs/invsqrt-evaluation-service.md), [qhdl avenues](docs/qhdl-avenues.md),
[optimization library](docs/qhdl-optimization-library.md), [phase predicates](docs/qhdl-phase-predicates.md),
[chemistry family view](docs/qhdl-chemistry-family-view.md).

## Safety

Circuits and pulled notebooks are untrusted data. `qhdl.py` re-executes itself under `python -I`; run other scripts
with `-I` when they read downloaded files. QHDL refuses to write its investigation trail into `outputs/` and many
scripts refuse to overwrite an existing output directory.

## Tests

```sh
python3 -m pytest -q
```

Tests that need the official verifiers, kickmix, the engine or the public dataset skip with a reason when those are
absent. Set the variables above to run them.

## License and credits

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Public circuits and methods by other competitors are
credited in the code where they are used or re-implemented, and in NOTICE.
