# `qhdl avenues`: a ranked list of places worth investigating in a circuit

Status: implemented for invsqrt, chemistry loaders and DQI (`scripts/qhdl_avenues.py`, `qhdl_avenues_chem.py`,
`qhdl_avenues_dqi.py`). Usage per task is in [agent-circuit-tools.md](agent-circuit-tools.md).

## Why

Narrow tools answer narrow questions: `qhdl window` asks what one block costs and what it must cost; a parts
crossover asks whether an exactly shared block gives a cheaper stitched path. When the answer is no, an agent has no
next move. Several public gains in this competition came from a different pattern:

- maoxin123 rebuilt all 15 chemistry loaders with a better search: same table, same interface, −52 Toffolis.
- Atakan Aldemir replaced one 6-Toffoli DQI phase component with a 5-Toffoli one at every call site.
- Two public invsqrt halves (Gengsr's prefix, maoxin123's tail) were joined at a cut with two free Clifford gates.

In each case a component costs more than it must, it is found and rebuilt, and the saving is multiplied by how often
it appears. `qhdl avenues` surfaces such components, ranked by expected score effect.

## Output

`python3 scripts/qhdl.py avenues -c <sha> [--top N] [--task invsqrt|dqi|chemistry] [--out DIR]`

It returns one row per avenue. Each row has:

| Field | Meaning |
| --- | --- |
| `component` | Op range(s), stage tag or loader slot; a function hash when available |
| `charged_cost` | Executed mean Toffolis (plus rotations for invsqrt), not static counts |
| `floor` | Best proven lower bound for this component, with its source: degree, nonlinear rank, Dickson rank, window floors |
| `best_known` | The cheapest public implementation of the same function across the dataset (generator G4, when present; otherwise `unavailable`) |
| `gap` | `charged_cost − max(floor, best_known_if_cheaper)`, labelled as an **upper bound on the possible saving** |
| `multiplicity` | The number of call sites of this component, up to wire relabelling |
| `gm_effect` | Static estimate of the score change if the gap closes at every site: `gap × multiplicity` priced through the task's cost model (invsqrt Q × charge; DQI Q² × mean Toffolis; chemistry λ_eff × Q × T), as `(1/3) ln(cost / (cost − saving))` |
| `slack` | Contract slack from generator G5: unreachable inputs, unread outputs, later-cancelled phase (otherwise `unavailable`) |
| `negatives` | Matching rows from an optional negative-results ledger CSV (task, keyword and op range) |
| `first_test` | A concrete command, a stop line, and the verification step that decides it |
| `tier` | Always *static model / projection* until a test runs |

Rows are sorted by `gm_effect`. Rows already closed by a negative result are shown below a divider, not removed.

## Generators (pluggable; each emits candidate components with evidence)

| Id | Generator | Status |
| --- | --- | --- |
| G1 | **Cost attribution.** Executed charged cost per stage or component: invsqrt per-op executed means (checked against the verifier charge; the run stops if they differ by more than 10%), chemistry loader slots and regions (cache / product / HMR tail) | implemented |
| G2 | **Floors.** Exact arbitrary-entry ANF windows of 2..16 Toffolis with degree, nonlinear-rank and Dickson-rank floors; DQI adds a phase-degree floor (degree d needs k with 3·2^(k−1) ≥ d; weak for high-degree phase) | implemented |
| G3 | **Repetition.** Repeated subcircuits up to wire and classical-bit relabelling, with call counts (DQI: Toffoli-like ops clustered ≤ 64 ops apart plus their phase/measure tail) | implemented |
| G4 | **Function matching.** Two byte-different public blocks computing the same function (canonical ANF or truth table under a declared interface), so each function's cheapest public version is known. Chemistry uses the same-function best-known loader from the parts table | chemistry only; invsqrt/DQI read `unavailable` |
| G5 | **Contract slack.** Reachable-input domain versus arbitrary entry; outputs never read; phases cancelled downstream | not wired; reads `unavailable` |
| G6 | **Diff explainer.** For a new public file, which components changed against a base and with what technique | separate verb: `qhdl diff` |
| G7 | **Crossovers.** A chemistry crossover packet (`--packet packet.json`), one row per stitched path; dominated when its loader Toffolis exceed the family's | implemented (optional input) |

## Controls

- invsqrt: on the 62-qubit public circuit `6c6a734dbcb6`, a known 12-CCX lookup block (ops [13714,13969), floor 5)
  ranks in the top 10, and windows already at their floors show `gap = 0`. Every row carries a runnable
  `first_test`.
- DQI: on `bb4287e77eb6`, Atakan Aldemir's 5-Toffoli component ranks first (floor 2, 1,750 sites plus 14 in a sibling
  class).
- Chemistry: per-slot executed Toffolis (two passes) are checked against the verifier's T within 1%.

## Non-goals

- No candidate emission. Avenues are starting points; `rewrite` and the verification steps still decide.
- No scoring claims. `gap` is an upper bound on the possible saving, `gm_effect` is a static model, and only the
  official verifier decides a score.
- Generators exchange JSON; no generator imports another's research code.
