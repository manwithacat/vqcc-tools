# Boolean phase predicates in QHDL

The shared module [`qhdl_predicates.py`](../scripts/qhdl_predicates.py) formalizes established Boolean and phase-polynomial structures. It is a representation and exact comparison library, not a new optimizer. The families follow established phase-polynomial and Boolean-predicate structures used by tools such as Feynman, ZX/ZH calculus, TKET and Qualtran.

## Inspect the families and a circuit

From the repository root, with a Python that can import kickmix:

```sh
$QHDL_KICKMIX_PYTHON scripts/qhdl.py predicate-families --all
$QHDL_KICKMIX_PYTHON scripts/qhdl.py phase-profile -c PATH_TO_CIRCUIT.kmb --ops 0:200 --all
```

Both commands are read-only with respect to the circuit. They write ordinary QHDL evidence files and return a `static` envelope. The profile retains its previous ANF masks and exact `angle128`, and adds:

- `entry_cut`, `region_end`, `cut_reason`: where the symbolic input coordinates apply; source ranges are half-open.
- `contract`: a versioned `qhdl.phase-term.v1` object containing source SHA, cut, computational basis, arbitrary-input domain, typed ANF, classical guard and exact angle.

The existing profiler stops at direct classical conditions, condition scopes, measurements, resets and unsupported operations. Its exported guards are therefore explicitly true. The library can represent classical guards, but the profiler does not yet extract guarded programs or infer relations among measurement bits.

## Six reusable families

| Constructor | Boolean meaning | Example use |
| --- | --- | --- |
| `monomial([a,b,c])` | a AND b AND c | CCZ sign predicate |
| `parity([a,b,c])` | a XOR b XOR c | Parity phase gadget |
| `cube(positive=[a], negative=[b])` | a AND NOT b | A selected computational-basis pattern |
| `equality_constant([a,b,c], 5)` | Little-endian word equals 101 | Equality-controlled phase; first variable is least significant |
| `mux(s,t,f)` | If s then t, otherwise f | Boolean selector, with ANF arguments |
| `priority_guards([s0,s1,...])` | First true selector wins | Mutually exclusive branches, optional default |

Constructors return canonical Boolean ANFs: XOR of monomials, with repeated variables in a product idempotent. A zero ANF has no monomials; the constant-one ANF has one empty monomial. This is different from an integer-valued weighted phase polynomial.

## Python example

The scripts directory must be on Python's import path. Here the source hash is an explicitly synthetic example identity; a circuit-derived contract must use the actual circuit SHA-256.

```python
from qhdl_predicates import (
    ANF, Var, Context, Predicate, ExactPhaseTerm, HALF,
    monomial, parity, phase_equivalent, to_json, from_json,
)

a, b = Var('quantum', 0), Var('quantum', 1)
m = Var('classical', 7, version=0)
context = Context('0' * 64, entry_cut=0)

cz = ExactPhaseTerm(
    Predicate(context, monomial([a, b])),
    ANF.variable(m),       # recorded classical guard
    HALF,                 # half a full turn: a sign change
)
assert from_json(to_json(cz)) == cz

def sign(function):
    return ExactPhaseTerm(Predicate(context, function), ANF.constant(True), HALF)

assert phase_equivalent(
    [sign(parity([a, b]))],
    [sign(ANF.variable(a)), sign(ANF.variable(b))],
)
```

`phase_equivalent` compares **sequences of phase terms**, so each argument is a list or tuple. `predicate_equivalent` compares two context-bound predicates. A mismatched source, cut, basis or domain raises `ContextMismatch`; different variable kinds, indices or versions remain different inputs. The only supported basis/domain pair in this version is computational basis / all inputs. No clean-scratch or reachable-state assumption is implicit.

## Exactness and limits

Angles are integer multiples of one full turn divided by 2^128, normalized modulo a full turn and serialized as exactly 32 lowercase hexadecimal digits. Floating-point angles are rejected. A constant predicate contributes global phase, which is retained even when a physical application might choose to ignore it.

`phase_polynomial` converts Boolean XOR to an exact multilinear integer expression using `f XOR g = f + g - 2fg`. This permits comparison at arbitrary angles without applying the sign-only XOR identity incorrectly. Both ANFs and expanded phase polynomials have explicit term caps; overflow raises `CapacityExceeded`, meaning the attempted comparison is unavailable, not that the functions differ. Boolean `&`, `^` and `~` are conjunction, XOR and complement; arithmetic over phase angles belongs to the separate phase API.

The source hash is provenance, not a proof by itself. A caller constructing its own predicate must justify the mapping from source gates to that function. The existing phase profiler supplies that mapping only within its admitted regions. Cross-measurement reasoning, arbitrary quantum effect operators, external compiler adapters, automatic synthesis, reachable-state invariants and competition cost prediction are outside this interface; costs remain a matter for the static model, local prescreens and, finally, the official verifier.

For a new rewrite, a predicate equality is one obligation. Output permutation, restored workspace, classical control history and full-circuit cost still require their own checks. No circuit rewriting or score improvement is implied by a successful contract comparison.
