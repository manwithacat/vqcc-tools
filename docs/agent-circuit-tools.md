# Circuit tools for agents: question → command

This file maps an optimization question to the command that answers it. Two tool families, one rule each:

- **qhdl** (`scripts/qhdl.py`): one bounded JSON envelope per call (≤ 20 rows / 2 KB; full tables go to files).
  Every answer carries a `tier`, an `evidence` path and a `rerun` string: quote the rerun string as your evidence.
  Run commands from the repository root as `python3 scripts/qhdl.py ...`; verbs that import kickmix or the engine
  need an interpreter that can (`$QHDL_KICKMIX_PYTHON`, with NumPy). `doctor --all` checks plugin imports; emitted
  commands preserve the interpreter and script path. Set `QHDL_WORK=<scratch dir>` to keep workspaces out of the
  repository. Circuits can be named by SHA-256 prefix (looked up in the manifest of the public-circuits dataset,
  `$QHDL_DATASET`). `python3 scripts/vqcc_config.py` prints every resolved setting.
- **workbench** (the kickmix-macos fork's `tools/circuit-viewer/workbench`, located by `$QHDL_WORKBENCH` /
  `$QHDL_VIEWER_DIR`): structure queries on any size of circuit and the exact bilinear (GF(2) multiplier) tools.
  It runs with the fork's own environment; set `WORKBENCH_PYTHON` when that is not found. Never `cat` a DQI region
  report (several MB); use `query`. See the fork's own agent notes.
- **Official verifier** (the only authority on a score): the unmodified verifiers from the competition data (not
  included in this repository; set `$QHDL_OFFICIAL_ROOT` to the directory holding `verifiers/`), run with
  `$QHDL_KICKMIX_PYTHON scripts/local_verify_reference.py ...` or `qhdl.py verify -c <sha>`. A local run is
  *local official verifier PASS*; the hosted Kaggle run remains the final authority. DQI costs depend on the
  measurement draw: compare DQI candidates on an expected cost over many fixed draws (for example with
  `scripts/dqi_fast_rank.py`), never one draw.

## Evidence tiers

Three tiers apply to any claim about cost or correctness: **static model / projection** (arithmetic on op counts),
**local prescreen** (BitSim, engine or exhaustive checks; not the verifier), and **official verifier PASS** (the
unmodified verifier). Only the last speaks to a score. Within qhdl envelopes the `tier` field is finer:

| Tier | Means |
| --- | --- |
| `exact-all-x` | checked on every legal input (invsqrt: all 3,133,452 x, zero measurement branch unless stated) |
| `sampled(n)` | n random inputs/outcomes; carries a standard error |
| `static` | counted from the op stream, not executed; an upper bound on charged Toffolis |
| `heuristic` | a search or ranking; absence of a hit is not a proof |
| `recorded` | quoted from an earlier official-verifier report (path given), not re-run |

A claim with mixed tiers lists them separately in `parts`.

## Question → command

All commands below are `python3 scripts/qhdl.py <verb> ...` unless another script is named.

| Question | Command | Tier | Size limit |
| --- | --- | --- | --- |
| What is this circuit; what can I ask? | `open -c <sha>` (then follow `next`) | static | any |
| Which public circuits are in the dataset? | `circuits [--task invsqrt\|dqi\|dfthcsos] [--prefix P]` | static | — |
| Where is the width peak and what is live there? (DQI-sized too) | `peak -c <sha> [--show regions\|paths\|runs\|idle]` | static | any |
| Which Toffolis are actually charged, by region / gate kind? | `exec -c <sha> [--by region\|kind]` | invsqrt: sampled over all x (matches the verifier mean); DQI: static/unconditional bounds only | any |
| invsqrt: per-stage profile, peak runs and idle wires | `profile`, `runs`, `idle` | static | ≤ 2 M ops |
| Why do `peak` and `runs` give different run counts? | `peak -c <sha> --liveness reset` or `runs -c <sha> --liveness allocator` (each reports the other model in `facts.other_liveness`) | static | peak any size; runs ≤ 2 M ops |
| Is this wire / run's value needed or recomputable? | `explain wire q49`, `explain run R1` | exact-all-x (+ heuristic recompute search in `parts`) | invsqrt |
| Can qubits be renamed onto fewer wires (lifetime recolouring)? | `recolour -c <sha> [--no-register-gaps]` | static (width); verifier status reported separately | ≤ 2 M ops |
| Which reusable optimizations exist, with what obligations? | `optimizations [--class phase-fold]` | static implementation contracts | no circuit required |
| What exact Boolean predicates receive phase in a region? | `phase-profile -c <sha> [--ops A:B / --region R] [--max-terms 256] [--max-span 2048]`; families: `predicate-families` | static exact algebra on arbitrary entry wires, per region; no sampled proof | ≤ 2 M ops per query; capacity/control/measurement barriers reported |
| What does this block of ops compute; how many ANDs does it really need? | `window -c <sha> --ops A:B [--out q19,q14] [--in q8,q9]` | exact on reachable inputs (outcome-dependent inputs: sampled, upper bound withheld). Floors include `floor_nonlinear_rank` (GF(2) rank of the degree ≥ 2 ANF residuals; only for exact-ANF unconditioned X/CX/CCX/SWAP windows, else null; never lowers `lb` or touches `ub`) | invsqrt |
| Window from op 0 / before a wire's first use | `window -c <sha> --ops 0:B` (untouched scratch is \|0>, reported as `fresh_zero`) | exact on reachable inputs | invsqrt |
| Where could a rewrite pay (Toffolis above the AND floor)? | `windows -c <sha> [--size 4] [--span 48] [--top 20] [--max-inputs 20]` (~1 min on invsqrt) | ranked by charged slack; floors exact or bounds, labelled | invsqrt |
| Where is a component costlier than it must be, ranked by score effect? | `avenues -c <sha> [--top 10] [--term-cap 6000] [--refine K] [--out DIR]` (invsqrt; run `exec` first; ~45 s); `avenues --task chemistry -c <family sha prefix> [--stage DIR] [--packet packet.json] [--slots 4,5]` (~25 min for 15 loader slots); `avenues --task dqi -c <sha> [--regions tree.json\|none]`, or `--ops A:B` for the exact value/phase ANF of one window (~25 s on a 58 M-op file; works on untagged files). Fields, generators and pricing: [qhdl-avenues.md](qhdl-avenues.md) | static model / projection; **`gap` is an upper bound, not a saving** | invsqrt, chemistry loaders, DQI |
| What changed between a base circuit and a new public file of the same task: which components, by how many Toffolis, at how many call sites, with a technique hint? | `diff -c <base> -c2 <new> [--task dqi\|invsqrt\|chemistry] [--top 20] [--deep] [--out DIR]`. DQI: top-level tag-path regions from `workbench summarize` (cached in `work/_diff/regions`), then op-hash alignment of each changed region; the aligned gaps are the call sites (e.g. a `5T>4T x1764` component change in `unextract`). invsqrt: aligned changed ranges plus a provenance map naming the public files the new one is built from (splices). Chemistry: `-c`/`-c2` are family SHA prefixes; per-slot Toffolis per pass, table/words/NPZ SHAs, λ_eff. Hints: component-resynthesis, repeated-component, splice, weights-refit, table-change, width-change, unknown. `unmatched` = whole-file delta minus the sum of component deltas. Classical bit ids are ignored when aligning. A region with unchanged counts is assumed unchanged unless `--deep`. ~15 s on DQI, ~2 s on invsqrt | static | any (DQI needs the workbench for the region tree) |
| Which products are uncomputed by a Toffoli or measured out and later recomputed, where one spare qubit would remove the Toffoli? (retained products) | `retain -c <circuit> [--region NAME] [--range A:B] [--top 10] [--out DIR]` (chemistry: `--task chemistry -c <family>`, scans the 15 loader files). Exact GF(2)-affine symbolic scan (128-bit Zobrist form hashes, so CX frames around a product do not hide it). Site classes: `measure-out` (a CCX cancels a product held in a mixed wire; computing it in a spare wire first lets an HMR remove it, −1 Toffoli), `retain` (HMR or CCX removal then identical recompute, −1 or −2), `copy` (duplicate while alive). A site is `free` only if a qubit known zero and untouched across the whole interval exists; `packed_total` greedily books one spare wire per overlapping site. Conditional blocks are not analysed (invsqrt: use `schedule repairs`). Positive control: Areeb Ali's DQI `8eccde65e8b7` against its parent `eae396baf0fa`, 133 of 133 realised Toffolis flagged (`python3 scripts/qhdl_retain.py control -c PARENT -c2 CHILD --out DIR`). Flagged sites are candidates: phase fix-ups and the spare wire are not verified. Missed: products equal only on the reachable state set with different operands (e.g. `CCX q11 q6 q8` vs `CCX q6 q10 q38` in a 46-qubit invsqrt build); confirm such pairs with a full-domain BitSim probe. ~2 min on a 58 M-op DQI file | static | any (DQI needs the region tree) |
| Apply the `measure-out` rewrite: a product removed by a cancelling CCX is instead removed by HMR plus a conditioned CZ, using a spare zero wire | `measure-out -c <circuit> --out X.kmb [--region R] [--dry-run]`. Writes the candidate, edit script, op map and sites JSON; one Toffoli less, one HMR and one CZ more per site, Q unchanged; skips CSWAP products, shared `on` ops and sites with no free spare. Exactness tested by statevector simulation of every HMR outcome (`tests/test_qhdl_measure_out.py`). Control: reproduces 98 of the 133 Toffolis of `8eccde65e8b7` from `eae396baf0fa` (the other 35 are `retain`-mechanism sites) | static (the candidate still needs the verifier) | any |
| Toffolis that a measure-first rewrite removes (write-only wire runs ending in an HMR: each CCX becomes a CZ conditioned on the outcome), and the live-width peak if those wires are freed | `measure-first -c <sha>` (`--all` for every row); bulk: `python3 scripts/qhdl_measure_first.py *.kmb --json out.json [--conditioned]` | static (the rewrite is an exact identity) | any (≈ 20 s on a whole DQI file) |
| Do two circuits compute the same thing on a block? | `fdiff -c A -c2 B --ops A:B [--ops2 A2:B2]` | exact on reachable inputs | invsqrt |
| Same block across an edit (cuts carried by the edit map), both measurement branches | `fdiff -c A -c2 B --ops A:B --map <edit.map.json\|auto> [--seed 7]` | exact-all-x, zero + random branch (draws not matched) | invsqrt |
| Is a whole candidate equivalent to its parent (phase per x, x restored, scratch clean, Toffolis per x)? | `fdiff -c <parent> -c2 <cand.kmb> --whole` (~10 s) | exact-all-x, both branches; tolerance via the BitSim contract model (the verifier stays the authority) | invsqrt |
| Is a candidate equivalent to its parent over the measurement outcomes that activate the changed conditioned gates (catches rare-branch faults that `fdiff --whole` misses)? | `python3 scripts/qhdl_alloutcome.py <parent.kmb> <cand.kmb> --out <new dir>` (about 25 s + 10 s per conditioned target; exit 1 REJECT, 2 INCONCLUSIVE); follow with `invsqrt_eval.py seeds --salts 8` (32 when conditioned ops changed), see [invsqrt-evaluation-service.md](invsqrt-evaluation-service.md) | local prescreen: both circuits run on identical outcomes (aligned HMRs share variables) sampled inside the solved activating subspace over all legal x; per-row phase, state, restoration; not a proof | invsqrt |
| How much of the charge is rotations? | `exec -c <sha> [--by kind]` (`facts.rotations`; full facts in `exec/exec_facts.json`) | Z_POW counts sampled over all x; eps quoted from a recorded official report (unavailable without one) | invsqrt; DQI static only |
| Which Toffolis leave their target provably \|0> (MBU candidates)? Which wires idle across peak runs are CNOT-recomputable? | `track -c <sha> [--region R \| --ops A:B] [--idle] [--emit-script e.txt]` (DQI: `--region unextract` ~3 min, 7 GB with `--idle`; a whole DQI circuit takes about an hour) | static (sound proof: true in every shot) | any |
| Where does an uncompute pay full price on a dirty wire, and what would MBU / a clean ladder / a conditionally clean construction save? | `census -c <sha> [--region R] [--proof auto\|track\|local\|structural] [--sampled 1024] [--sites mbu\|chains\|all]` (DQI ~100 s, ~7 GB) | chain functions and `zero=proved` exact (every shot); `zero=sampled(N)` sampled (DQI simulation); prices and free wires a static model | any |
| Which rewrite-class sites does this circuit have (AND3 dirty-borrow MBU, flag-ladder MBU, partial u3 MBU, 12- and 6-input conditionally clean AND chains, lone-Toffoli MBU, invsqrt AND merges, phase folding), with obligations, prices and tag-relative addresses? | `census -c <sha> --class all\|<class> [--region R \| --ops A:B] [--unfit]` (DQI ~3 min, ~5 GB) | matches exact (block ANF over all wire values); `zero`/clean proofs every shot; `sampled(N)` obligations are evidence; prices a static model | any |
| Build one class's rewrite and get its cheapest certificate (map, edit script, sites and certificate JSON beside the output) | `rewrite -c <sha> --class <class> [--sites ids] --out cand.kmb [--replacement F] [--unproved] [--certify auto\|none\|smoke\|fdiff]` (one class per candidate; `and-merge` needs an agent-supplied replacement window) | DQI: one engine smoke draw; invsqrt: `fdiff --whole` exact-all-x; never official | any |
| Splice ops by original index | `edit` (insert/delete/replace script → new `.kmb` + op-index map) | static | any |
| DQI: value of any wire at any op over valid syndromes | `simulate -c <dqi> [--syndromes 4096] [--at OP,...] [--wires q885,...] [--weights official\|uniform]` (~20 s at 4,096 lanes on a 60 M-op circuit; states saved as .npz) | sampled(N): official sampler; phases ignored; absence is not a proof | DQI |
| Which values of a child circuit correspond to which values of its parent (matched / erased_in_child / unknown / new_in_child), with per-match evidence; never an equivalence claim | `valuealign -c <parent> -c2 <child> [--sig 1024] [--out DIR]` → `alignment.jsonl` + `summary.json` | static (op-stream anchors, producer/consumer structure, intervals); with `--sig N` match keys and erasures are `sampled(N)` | any two enumerable circuits (~2 s for 30k ops) |
| Which op runs sit at the peak live width (where a width bound binds)? | `schedule plateaus -c <sha> [--ops A:B \| --stage NAME] [--level Q]` | static | any opened circuit |
| Can wire(s) W be held live over ops I..J without exceeding width Q? (width test only, not correctness; `--extra N` adds anonymous words) | `schedule hold -c <sha> --wire 41 --from I --to J [--q Q] [--extra N]` | static | any opened circuit |
| Toffolis (static and mean executed per shot) per op range, split into unconditional, conditional-branch and repair-block work; two ranges give the paired difference | `schedule attribute -c <sha> --range A=23606:25567 --range B=27792:32341 [--no-exec]` | static counts; means `sampled(n)` via the cached `exec` | invsqrt (~15 s after the workspace exists) |
| Every HMR-then-recompute repair block: op range, Toffolis, mean executed, live width, GF(2) degree of its repaired phase, and whether holding the earlier target would avoid the recompute within Q | `schedule repairs -c <sha> [--ops A:B] [--no-exec] [--no-holds] [--term-cap N]` | static (phase algebra, holds); means `sampled(n)` | any opened circuit; means invsqrt only |
| Static value graph (schema `qhdl-value-graph/v1`): wire-level SSA values with birth/last-use/kill cuts, producers, guards, consumers, HMR as kill + classical outcome + `unknown` phase obligation | `valuegraph export -c <sha> [--out DIR]`, `valuegraph show -c <sha> --op 456`, `valuegraph live -c <sha> --cut K [--through K2] [--min-span N]` | static (expressions and recomputability are `unknown`) | any opened circuit under 2 M ops (invsqrt ~3 s) |
| Can a value erased at one cut be restored at a later cut from other live values? `determined` true/false over all legal x (a witness pair if false; a minimal dependency set and a priced encoding if true) | `recompute -c <sha> --query q.json [--seed 7] [--out DIR]` or `--target q23 --erase-after I --restore-before J --available id,id,...` | exact-all-x (zero HMR branch; optional random branches); phase repair `unknown`; cache networks of degree ≥ 3 are upper bounds | invsqrt (~3 s) |
| Can these public parts be joined (provenance, pair/chain status, what a complete chemistry model still needs)? | `parts summary\|source <sha12>\|module <id>\|compose <left> <right>\|chain <id> ... [--catalogue DIR] [--out DIR]` (catalogue: a public modular-parts `bundle_public/` directory via `--catalogue` or `QHDL_PARTS_CATALOGUE`; manifest SHA-256s verified on load) | static lookup of recorded catalogue metadata (nothing re-run, no circuit read). Cross-source invsqrt joins are at most `local_all_x_cut_pass` (scoped to that cut, obligations listed); a verifier PASS applies only to the one named KMB; DQI cross-source = `unknown_cross_source_join`; chemistry chains are pairwise `known_table_chain` only | rows ≤ 20, chain ≤ 64 modules |
| Is the tool chain healthy (engine exports `run_sequence`, kickmix importable, workbench found)? | `doctor [--all]` | static | — |
| Official result already recorded for these bytes? | `evidence -c <sha>` (scans `$QHDL_OUTPUTS`) | recorded | any |
| Run the official verifier locally | `verify -c <sha> [--task invsqrt\|dqi] [--salt S] [--workers 6]` (invsqrt ~7 min, DQI ~4 min) | official verifier (local) | any |
| Address a site so it survives edits (DQI tagged circuits) | `--ops 'tag:unextract/companion_normalize[0]+187:+400'` on any `--ops`/`--ops2`/`--at` verb; `addr <addr> -c <sha>` prints the absolute op and every enclosing `tag:<path>[k]+off` (k 0-based, a unique path suffix is enough) | static | any `.kmb` with tags |
| Carry an op index across an edit | `--ops op@<sha12>:A:B --as-of <edit.map.json>[,<map2>]` (qhdl edit maps or span-list maps; a cut inside an edit is refused) | static | any |
| Start an investigation (resumable) | `inv new <slug> --task dqi --question "..." --base <sha>=<label>`, then `export QHDL_INVESTIGATION=investigations/<date>-<slug>` before the first probe (every verb then appends to its `trail.jsonl`; envelopes carry `claim_id`, `provenance`, certificate `scope`) | — | — |
| Record a hypothesis, then its outcome | `inv hypothesis --id H1 --statement ... --kill-if ...` (kill_if required, fixed once written); `inv kill H1 --outcome ... --tier sampled(4096) --probe <claim_id>` / `inv confirm` / `inv park`; `inv candidate --sha ... --domain draws:48@<seed> --extent whole --branches engine-sampled`; `inv log --call ...` for a helper script | as recorded | — |
| Pick up someone else's investigation cold | `inv resume investigations/<dir>` (one envelope ≤ 4 KB: question, open hypotheses with kill_if, candidates with scope, outcomes, `next`); `inv status` for the row list | recorded | — |
| Draft negative-result rows from an investigation | `inv ledger-stage <dir>` → `ledger-stage.json` and `leads-stage.md` (drafts for a human to file) | as recorded | — |
| Region tree and static Toffolis per tag (DQI) | `workbench summarize X.kmb --out r.json` then `workbench query r.json --depth 0 --sort toffoli` | static | any |
| Aggregate repeated region instances (e.g. 8,050 `scan/...`) | `workbench query r.json --depth 1 --group path` | static | any |
| Ops around a position | `workbench cut X.kmb <op> --radius 4` | static | any |
| GF(2) multiplier: what product tensor does this KMB compute? | `workbench bilinear-cert --challenge manifest.json --id <id> --out cert.json --tensor-out t.json` | exact | small |
| GF(2) multiplier: certified local rewrites | `workbench bilinear-corner` / `-mechanisms` / `-carrier` (same `--challenge/--id`, `--candidate-out c.kmb`) | exact (re-certified) | small |
| GF(2) multiplier: invariants that suggest a decomposition | `workbench bilinear-structure --modulus 0x13`, `bilinear-isotopy` | exact | small |
| Compile a term list to KMB | `workbench bilinear-synth ...` | exact | small |

## How to work

1. `open`, then follow its `next`. On DQI-sized circuits that is `peak`, `exec`, `evidence` and the workbench.
2. Form a hypothesis from an exact or sampled result, not from a static count alone (static Toffolis overstate the
   charged cost: on one public invsqrt circuit, 1,495 static vs 1,325.48 executed).
3. Check whether the mechanism has already been tried (`avenues` lists matching rows from an optional
   negative-results ledger below a divider).
4. Build the candidate, run the prescreens (`fdiff --whole`, `qhdl_alloutcome.py`, `invsqrt_eval.py seeds` for
   invsqrt; an expected cost over fixed draws for DQI), then the official verifier. Record the `rerun` strings.

Run `doctor` first when an engine-backed script fails to import the engine: a stale `_engine.abi3.so` without
`run_sequence` is the usual cause.

Not wired into qhdl: the surgery tools (`scripts/qhdl_surgery.py`, own CLI: `library`, `matrix`, `compat`,
`adaptor`, `splice`, `gate`) and the DQI draw ranking (`scripts/dqi_fast_rank.py`). `qhdl link` is hidden until the
viewer honours its parameters.

Calibration: `windows` on the public invsqrt `67482db6c538` ranks exactly Ehan Mallick's two 3-to-2 AND merges as the
only charged savings (1 Toffoli each); on `bce2948ff24e` it finds none in that window class. Upper bounds are
algebraic counts (placement, width and phase not checked); windows reading more than `--max-inputs` are skipped.

Optimization library and larger-refactor rubric: [qhdl-optimization-library.md](qhdl-optimization-library.md). The
`phase-fold` class is available through `census --class` / `rewrite --class`. It uses exact 128-bit phases and
reports raw rotations, synthesized rotations and T-angle gates separately. It adds no wires; gate simplification
does not imply score improvement. Predicate API: [qhdl-phase-predicates.md](qhdl-phase-predicates.md).

## Chemistry family companion

For chemistry full-model width, cost, declared accuracy margins and same-mask neighbouring-delta dependencies, use
`python3 scripts/qhdl_chemistry.py --report <local-report.json> --out <fresh-view.json>` or `--case
<explicit-model.json>`. This is a separate companion, not a qhdl verb: supplied-model arithmetic and declared report
evidence only, with no circuit-byte authentication, numerical recomputation or verifier acceptance. See
[qhdl-chemistry-family-view.md](qhdl-chemistry-family-view.md).
