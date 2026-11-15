# Circuit identity registry

`identities.json` lists the gate identities and rewrite patterns we know save Toffolis or qubits, in one structured
place that tools and agents can read. `scripts/qhdl_identities.py` lists them, checks them, and runs every
tool-backed detector on a circuit. Started 2026-11-15.

```sh
python3 scripts/qhdl_identities.py list [--task invsqrt|dqi|chemistry]
python3 scripts/qhdl_identities.py show ID-005
python3 scripts/qhdl_identities.py validate             # schema + exhaustive check of every exact fixture
python3 scripts/qhdl_identities.py scan -c CIRCUIT.kmb --task invsqrt [--run] [--out DIR]
```

## Fields

| Field | Meaning |
| --- | --- |
| `id` | `ID-NNN`, never reused. |
| `name`, `summary` | Plain words: what the rewrite does. |
| `validity` | `exact`: equal as operators for every input and every measurement outcome; the fixture is checked exhaustively. `reachable`: equal only on states the circuit reaches; every use must be certified on the whole circuit (`qhdl fdiff --whole` for invsqrt, 48-draw sweep for DQI, the loader checker for chemistry) and by the official verifier. |
| `lhs`, `rhs` | Minimal fixture in kickmix text (`CCX q0 q1 q2`, `HMR q2 b0`, `CZ q0 q1 if b0`). Empty for `reachable` identities that have no context-free form. |
| `fixture_zero` | Wires the fixture assumes clean on entry (for example a free ancilla). |
| `toffoli_delta`, `qubit_delta` | Change per application. The validator checks `toffoli_delta` against the fixture. |
| `preconditions` | What must hold at a site. A measurement bit written only by `rhs` is a fresh record, unused elsewhere. |
| `tasks` | Tasks where it applies or has been tried. |
| `credit` | Originator first (contestant, paper or "standard"); our tool second. |
| `evidence` | Circuits, entries and cards where it was used, with tier. |
| `detector`, `emitter` | Shell commands with `{circuit}` (.kmb), `{kmx}` (text form) and `{out}` placeholders. Empty when no tool exists yet. |
| `status` | `tool-backed` (a detector exists) or `catalogued` (recorded, no tool yet). |

The fixture checker covers X, CX, CCX, Z, CZ, CCZ, SWAP, CSWAP, HMR, R and `if bN` conditions, with phase as a sign.
It does not model `Z_POW` angles.

## Adding an identity

1. Append an entry with the next free ID. Write the smallest fixture that shows it; for a `reachable` identity, say in
   `preconditions` what the certification must establish.
2. Run `python3 scripts/qhdl_identities.py validate` and `python3 tests/test_qhdl_identities.py`.
3. When a detector exists, add its command and set `status` to `tool-backed`; `scan` then runs it on any circuit.

Credit follows the project rule: a re-implementation of a known form needs no attribution; an idea another entrant
originated is credited to them (Apache-2.0 where code is reused).
