# QHDL optimization library and refactoring rubric

QHDL has a reusable library of **scoped transformations**, not a universal optimizer. A rule can be used on any supported KMB circuit whose local obligations are met; a claim about one task's score does not transfer to another task. Start with the catalog, select a region and preserve the evidence chain from source bytes to candidate.

```sh
PY=$QHDL_KICKMIX_PYTHON   # a Python that can import kickmix; run from the repository root
$PY scripts/qhdl.py optimizations
$PY scripts/qhdl.py optimizations --class phase-fold
$PY scripts/qhdl.py phase-profile -c bce2948ff24e --ops 15240:15590
$PY scripts/qhdl.py census -c bce2948ff24e --class phase-fold
$PY scripts/qhdl.py rewrite -c INPUT.kmb --class phase-fold --out NEW.kmb --certify none
```

`optimizations` is generated from the implemented class registry. Its full evidence JSON records applicability, obligations, footprint, cost units and certificates. `phase-profile` exports predicates for planning a phase-basis refactor; it does not perform approximate synthesis. No rule is inferred safe merely because an input sampler did not find a counterexample.

## Reading and evidence map

| Need | Source |
| --- | --- |
| Find a command and its evidence tier | [Agent circuit tools](agent-circuit-tools.md) |
| Understand the implemented rewrite contract | `qhdl optimizations --class NAME`, then its evidence JSON |
| Reproduce the phase-fold validation | `tests/test_qhdl_phase_fold.py`, then the frontier calibration below |
| Exact predicate API | [Boolean phase predicates](qhdl-phase-predicates.md) |

## Implemented rules

| Class | Mechanism | Conditions that matter |
| --- | --- | --- |
| `and3-dirty-mbu` | Replace dirty-borrow AND3 uncompute with measurement and phase correction | Exact matched function, target zero after, restored borrowed wires |
| `flag-ladder-mbu` | Replace six-input flag uncompute with a measured clean ladder | Correct target function and available helper lifetimes |
| `u3-mbu` | Simplify a partial measurement-based AND3 uncompute | Entry function and all phase fixups match |
| `cc-chain` | Conditionally clean 6- and 12-input AND compute templates (Khattar and Gidney 2024) | Clean target and proved conditional cleanliness; widths must fit |
| `single-toffoli-mbu` | Measure a known AND target instead of uncomputing it | Target zero after original gate; exact CZ correction |
| `and-merge` | Rank windows whose Boolean outputs admit a cheaper construction | Replacement is agent-supplied; ranking does not prove placement, phase or width |
| `phase-fold` | Combine/cancel diagonal phases with identical Boolean predicates | Exact ANF equality inside a bounded unconditional reversible region |

The older MBU rules can report sampled obligations as evidence and the CLI permits explicit unproved builds. Review each site's obligation statuses: **proved**, **sampled**, **open** and **failed** are different. A built candidate and a smoke draw are a local prescreen, not official verifier acceptance.

## Exact phase folding

A diagonal gate applies `exp(2π i a f)` where `f` is a Boolean predicate of the region's entry wires. Classical reversible gates change the predicate carried by a wire; they do not prevent its phase contribution from being compared with a later contribution. If the predicates are exactly equal, their angles add modulo one turn. The rule can recognize equality across SWAPs, nonlinear compute/uncompute and other supported reversible operations, not just adjacent gates on the same wire.

Representation: ANF is an XOR of monomials. In the profile JSON, mask bit `w` denotes entry wire `qw`; mask zero denotes constant one. All entry wires, including scratch, are arbitrary independent variables. This deliberately avoids a task-specific clean-state or sampled-reachability assumption. Equality is valid for superpositions too because both the basis permutation and phase on every basis state are preserved. Global phase is preserved, not silently discarded.

Angles use kickmix's exact unsigned 128-bit full-turn representation; integer addition wraps modulo `2^128`. `qhdl edit` now accepts `Z_POW qN angle128:<32 hex digits>` as well as copying an existing angle with `@i`. There is no floating-point angle round-trip.

Supported analysis operations are X, CX, CCX, SWAP, CSWAP, Z, CZ, CCZ and Z_POW, plus transparent INFO_TAG metadata. Measurement, reset, classical conditions, condition-stack boundaries and any unsupported opcode cut the region. Predicates are not matched across cuts. Default limits are 256 ANF terms and 2,048 ops per region; growth beyond a limit starts a fresh independent region without changing the intervening circuit. A capacity cut sacrifices opportunities, not correctness. `phase-profile` accepts explicit limits and refuses an unbounded query above two million ops; `census --class phase-fold` uses the documented defaults.

The rewrite touches only the selected phase ops. Registers, data operations, tags, resets and controls stay intact; no wire or measurement bit is added. The original quantum-width header is retained, so a disappearing use is not claimed as a width saving. Multiple matching groups may have overlapping enclosing intervals, but their actual edited phase op sets are disjoint.

Cost reporting separates Toffolis, raw Z_POW gates, general synthesized rotations and T-angle gates. Clifford-angle Z_POW gates are free in the invsqrt contract. A reduction in raw gate count can therefore have zero cost benefit. A general rotation's cost depends on the available error margin and the total rotation count; do not convert it to a fixed Toffoli price. Phase folding is exact, so it does not spend approximation error, but the official cost calculation remains authoritative.

## Worked example: recognize the predicate, not the wire number

Consider this unconditional region, with every wire allowed to be dirty:

```text
CCZ q0 q1 q2
SWAP q2 q3
X q4
CCZ q0 q1 q3
```

The first CCZ contributes a half-turn when entry values `q0*q1*q2` are one. After the swap, wire q3 carries the original q2, so the second CCZ contributes a half-turn to the same predicate. The X on q4 does not change it. The sum is one full turn and cancels exactly; the replacement is:

```text
SWAP q2 q3
X q4
```

This removes two unconditional Toffoli-class gates, preserves the wire permutation and every basis state's phase, and adds no workspace. This is a synthetic correctness example, not a discovered competition improvement. The independent scalar tests enumerate all 32 five-wire input states, including dirty values.

Contrast `CCZ q0 q1 q2; X q2; CCZ q0 q1 q2`: the predicates differ and this rule must not cancel them. Nor may it match across an intervening HMR or classical condition, even when a sample happens to show the same values.

After a real rewrite, inspect `OUT.kmb.sites.json` for obligations and edited ops, `OUT.kmb.map.json` for original-to-new addresses, and `OUT.kmb.certificate.json` for validation scope. With `--certify none`, no whole-circuit check has run. With invsqrt `--certify fdiff`, the raw results are saved as `OUT.kmb.fdiff.json`. A static saving, an all-x two-branch check and official acceptance are distinct evidence.

## A rubric for larger refactoring

Use the following gates in order. They are planning heuristics, not an automatic search or acceptance certificate.

| Gate | Required evidence | Reject or redirect when |
| --- | --- | --- |
| Contract | Input domain, preserved outputs, phase, allowed gates, error tolerance and cost equation | The proposed algorithm solves a relaxed or different problem |
| Whole-circuit bottleneck | Current source SHA; executed charge where available; peak coverage under stated liveness model | A local win cannot affect the global bottleneck, or uses static counts as expected counts |
| Consumer demand | Exact values/phases consumed at the block boundary; scratch and measurement history | A cheaper representation omits information the next block needs |
| Algebraic opportunity | Shared products, phase predicates, repeated submaps or approximation freedom derived from bytes | The proposed tensor/function is only guessed from source labels |
| Schedule | Concrete reversible placement, cleanup and measurement corrections | Extra live wires or reconstruction erase the projected benefit |
| Full price | Q × charge, R × Q² × mean Toffolis, or chemistry lambda × Q × Toffolis, as appropriate | Only one favorable count falls while total cost rises; unknown costs treated as zero |
| Independent validation | Exact small-domain proof, then whole-circuit checks and task verification | A sampled match is presented as universal or an approximate phase change lacks a full-domain error check |
| Promotion | Frozen candidate, provenance, full verifier and required fixed-draw comparisons | Retuning after official outcomes or selecting a lucky measurement salt |

Useful search order:

1. Exact local identities with no added workspace: `phase-fold`, justified Boolean cancellations and MBU templates. Use bounded regions on large circuits.
2. Sharing across nearby consumers: expose a joint map, then price storing or recomputing intermediates. Optimize both adjacent chemistry delta loaders when one word block changes.
3. Lifetime and representation changes: inventory **every** binding peak before claiming fewer qubits. A wire idle at the peak may still carry indispensable history.
4. Joint approximation and implementation: change predicates/coefficient grids only with an explicit error budget, a shared compute/uncompute schedule and actual rotation/loader pricing.
5. Whole-algorithm replacement: require a byte-derived interface and a conservative resource model before a full build.

A negative search result closes the tested mechanism and budget. A rank proxy, bounded symbolic cut, failed placement or one optimizer start is not a general impossibility proof.

## Phase-basis refactoring: what this provides, and what remains

The new `phase-profile` gives a hash-linked inventory of exact phase predicates, their degree, ANF terms, angle sum, gate positions and region identity. It is a starting point for selecting a bounded nonlinear predicate library. Entry variables at separate cuts are independent interfaces: identical masks across cuts do **not** prove equal values. Full JSON includes the basis convention, coverage and limitations.

A future phase-basis search should pin one block; reproduce its current phase; propose shared predicates; fit coefficients on every reachable input with explicit synthesis margin; schedule compute/phase/uncompute; and compare full Q × charge. Approximating predicates separately or minimizing rotations without charging predicate computation is insufficient. Approximate minimax fitting and global phase-basis synthesis are **not implemented** by this change.

Frontier calibration: the pinned Ehan Mallick `bce2948ff24e` circuit has two half-turn Z_POW gates canonicalized to Z. Independent `fdiff --whole` agrees on every legal x under zero and random(seed 7) branch models. Both removed Z_POWs were Clifford-priced, so the result is a representation simplification, **not a score improvement**. The bounded profile saw 48 predicates, 18,660 barriers and 182 capacity cuts. No conclusion about better predicates across those cuts follows.

## Adding a reusable rule

Implement `qhdl_rewrite_<name>.py` with `CLASS`, `match`, `apply` and the existing class interface; register its name in `qhdl_rewrite.CLASSES` and its contract in `qhdl_optimizations.CONTRACTS`. Sites must carry a source hash, original op addresses, explicit obligations, edit footprint and typed resource changes. Use the existing edit/map/certificate machinery. Do not add a second candidate-writing path.

Before treating it as reusable, require an independent exhaustive small-circuit semantics check, adversarial nonmatches, dirty-wire tests, measurement/control boundaries, resource limits, and CLI/edit/hash roundtrips. Test disjoint composition and shared-helper lifetimes; never add overlapping edits merely because their headline savings sum. For angle rules, include 128-bit low-bit and wraparound cases and distinguish global phase from a relative phase.

A broader pass manager is deferred until class cost vectors, interference and certificate composition are explicit. Today one rewrite class produces one candidate and one certificate, making failures attributable and results reviewable.


## Roadmap and completion criteria

The seven-class library, exact phase profiler and exact phase-fold implementation are in place. The following are separate future increments, not hidden capabilities of the current commands.

| Increment | Required output | Completion gate |
| --- | --- | --- |
| Bounded phase-basis experiment | One pinned block, reachable-domain contract, shared predicate library, coefficients and reversible schedule | Reproduce the control; then beat full Q × charge at the same accuracy after selection and cleanup |
| Reusable approximate phase rule | Explicit error allocation, domain certificate, parameter serialization and task-specific cost adapter | Independent exhaustive small cases plus whole-domain error and task-verifier acceptance; never promote sampling to proof |
| Additional exact rewrite class | Registry module, typed resource changes, obligations, edit map and independent tests | Positive and adversarial fixtures, dirty/control/measurement cases, and one bounded real-circuit calibration |
| Cross-class pass manager | Site conflict model, helper lifetime accounting, order/dependency rules and certificate composition | Refuse conflicts and unproved obligations; reproduce standalone passes and preserve full-circuit behavior on composed fixtures |

Start with the first increment only as a new, preregistered experiment. The frontier calibration null does not justify widening the search automatically. The optimization library is complete for this increment; the nonlinear phase-basis search remains open.


## Typed predicate contracts

The shared Boolean family library and exact phase contract are implemented. Use `qhdl predicate-families` and the typed `contract` field in `phase-profile`; old fields remain available. See the [API guide](qhdl-phase-predicates.md). This adds no rewrite class or automatic synthesis pass. The phase-basis search, cross-measurement analysis and external compiler translation remain separate future work.
