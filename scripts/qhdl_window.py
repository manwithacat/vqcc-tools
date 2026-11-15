#!/usr/bin/env python3
"""qhdl plugin: what does an op window compute, and how many Toffolis must it cost?

Verbs (registered into scripts/qhdl.py through qhdl_register(subparsers, api)):

  window  -c C --ops A:B [--out q19,q14] [--in q8,q9,...]
      The window is ops A..B-1 (0-based kmx op indices, header lines excluded). Inputs are the wires, classical bits
      and condition-stack levels that the window reads before it overwrites them; outputs are what it writes.
      The input states that actually occur at op A are taken from bitset simulations of all 3,133,452 legal x in
      two outcome branches (every HMR returns 0; independent random outcomes, seed 7: this is what makes
      measurement-conditioned fix-up blocks visible), and the window is run on each distinct state with its own
      HMRs in the zero branch. Result:
      every output bit as a Boolean function of the inputs on the reachable states, its algebraic normal form
      (ANF), degree, support, and an AND-count floor for the live outputs next to the window's Toffoli count.
  windows -c C [--size N] [--span S] [--top K]
      Scans every Toffoli-led window (up to N consecutive Toffoli-class gates within S ops) and ranks them by slack
      = Toffolis - floor: the places where a rewrite could pay.
  fdiff   -c C1 -c2 C2 --ops A:B [--ops2 A2:B2 | --map EDIT.map.json | --map auto] [--seed S]
      Matched cuts in two circuits: per output, on how many legal x the values differ at the end of the windows,
      given whether the inputs already differed at the start, in the zero branch and a random-outcome branch (seed S,
      default 7; draws are not matched across circuits); plus each side's Toffolis and floor (floor 'unavailable'
      when a side reads > 30 inputs; the values are still compared). --map takes the cut map `qhdl edit` writes
      (-c = the edit's input) or aligns the two op streams itself (auto); an insertion exactly at a cut is inside.
  fdiff   -c C1 -c2 C2 --whole [--seed S]
      Whole circuits on all legal x, zero branch and one random branch each: per-x total phase (sign + Z_POW units,
      compared exactly), x restored, scratch clean, executed Toffolis per x; plus each side's invsqrt tolerance
      check (qhdl_bitsim.contract model). ~10 s for two invsqrt circuits.

A window that starts before a scratch wire's (or bit's) first touch, e.g. at op 0, does not count it as an input:
it is |0> there (fresh_zero).

Reachable inputs and don't-cares. Input states never reached at A are freedom a rewrite may exploit. Linear
dependencies are removed exactly: the reachable states span an affine subspace of dimension k, and the outputs are
re-expressed over k pivot inputs (the other inputs are affine in them; listed as relations). This loses nothing: any
circuit agreeing on the reachable states, composed with the affine embedding, is a circuit for the reduced function
with the same AND count. If the reachable states fill the whole subspace (2^k states) the ANF is unique and exact;
otherwise the remaining don't-cares are handled by a minimum-degree completion (GF(2) solve, degree <= 3) and the
floor becomes a bound.

AND-count floor (multiplicative complexity, MC), over the live outputs (static liveness: a written wire is dead only
if its next touch is an unconditional R; HMR counts as a read because its phase kickback depends on the value):
  * degree bound (Schnorr): MC(f) >= deg(f) - 1, for every linear combination of the outputs; sound in general.
  * quadratic forms (Dickson; Mirwald-Schnorr): a single quadratic form needs exactly rank(B)/2 ANDs, B its
    alternating matrix; the max over the span of the outputs is a sound lower bound for the whole window.
  * dimension of the span of the outputs' quadratic parts: a lower bound in the degree-2 circuit model.
  * upper bound (constructive): a minimum-weight basis of that span, sum of rank/2 (matroid greedy), when every
    live output is quadratic. Placement, width and phase are NOT checked: ub is an algebraic count, not a rewrite.
  floor_kind: 'exact' (sound lb == ub), 'exact (degree-2 model)', or 'bound'.
Phase: Z/CZ/CCZ/NEG inside the window give a sign function (zero branch); degree >= 3 needs at least one
Toffoli-class gate (counted in lb). Z_POW rotations and the HMR kickback in the m = 1 branch are not modelled: a
rewrite must keep each HMR's fix-up pairing (check with qhdl_bitsim random-outcome seeds and the verifier).

Tiers: the functions are exact-all-x on the observed states (all x; outcome bits from two branches, so a state
reachable only under another outcome pattern can be missing: that can lower lb, and ub is then only checked on the
observed states); liveness and Toffoli counts are static; floors are theorems
applied to exact functions, labelled exact or bound. invsqrt only (enumerable domain); DQI-sized circuits are
refused with a hint.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qhdl_bitsim as B  # noqa: E402
import qhdl_envelope as E  # noqa: E402

TOFF = (B.C_CCX, B.C_CCZ, B.C_CSWAP)
PHASE = (B.C_Z, B.C_CZ, B.C_CCZ, B.C_NEG)
MAX_INPUTS = 30          # raw inputs of one window (keys are uint64; bincount up to 22)
ANF_FULL_MAX = 20        # pivots for an exact ANF when the reachable states fill the subspace
ANF_DC_MAX = 16          # pivots for the don't-care (min-degree completion) solve
SPAN_MAX = 10            # nonlinear outputs for span enumeration
DC_MAX_DEG = 3
TAPPED = (B.C_ZPOW, B.C_HMR)  # value consumed by a rotation angle / an HMR kickback: a tap output
VALUE_GATES = (B.C_X, B.C_CX, B.C_CCX, B.C_SWAP)  # nonlinear-rank floor applies only to windows made of these
BRANCHES = ('zero', 7)    # outcome models whose states at A are united: zero branch + one random branch (seed 7)


# ---------------------------------------------------------------- static structure
def op_io(o):
    """(reads, writes, kills) as names 'qN' / 'bN', zero-branch semantics. kills = fully overwritten."""
    c, q = o.code, o.q
    reads, writes, kills = [], [], []
    if o.cond >= 0:
        reads.append(f'b{o.cond}')
    if c in (B.C_PUSH, B.C_POP, B.C_INFO, B.C_NEG, B.C_ZPOW):
        pass
    elif c == B.C_X:
        reads.append(f'q{q[0]}'); writes.append(f'q{q[0]}')
    elif c == B.C_CX:
        reads += [f'q{q[0]}', f'q{q[1]}']; writes.append(f'q{q[1]}')
    elif c == B.C_CCX:
        reads += [f'q{q[0]}', f'q{q[1]}', f'q{q[2]}']; writes.append(f'q{q[2]}')
    elif c in (B.C_Z, B.C_CZ, B.C_CCZ):
        reads += [f'q{x}' for x in q]
    elif c == B.C_SWAP:
        reads += [f'q{q[0]}', f'q{q[1]}']; writes += [f'q{q[0]}', f'q{q[1]}']
    elif c == B.C_CSWAP:
        reads += [f'q{x}' for x in q]; writes += [f'q{q[1]}', f'q{q[2]}']
    elif c == B.C_R:
        writes.append(f'q{q[0]}'); kills.append(f'q{q[0]}')
    elif c == B.C_HMR:  # zero branch: q := 0, b := 0, no phase
        writes += [f'q{q[0]}', f'b{o.bit}']; kills += [f'q{q[0]}', f'b{o.bit}']
    elif c in (B.C_BST0, B.C_BST1):
        writes.append(f'b{o.bit}'); kills.append(f'b{o.bit}')
    elif c == B.C_BINV:
        reads.append(f'b{o.bit}'); writes.append(f'b{o.bit}')
    return reads, writes, kills


def tap_name(o, i):
    return f"{'zpow' if o.code == B.C_ZPOW else 'hmr'}@{i}:q{o.q[0]}"


def window_io(ops, A, Bend, depth_A=0):
    """Static inputs (read before overwritten) and outputs (written) of ops[A:Bend]; condition-stack levels s0..
    (outermost first) are inputs when the window starts inside a PUSH_CONDITION. cones: for each output (and
    'phase'), the inputs its value can depend on (static forward propagation, zero branch)."""
    inputs, outputs, defined, seen_in, seen_out = [], [], set(), set(), set()
    toff = phase = hmr = zpow = conditioned = 0
    dep = {}
    stack = [{f's{j}'} for j in range(depth_A)]
    pcone = set()

    def D(nm):
        return dep[nm] if nm in dep else {nm}
    for i in range(A, Bend):
        o = ops[i]
        c = o.code
        r, w, k = op_io(o)
        if c == B.C_PUSH:
            r = [f'b{o.cond}']
        for nm in r:
            if nm not in defined and nm not in seen_in:
                seen_in.add(nm); inputs.append(nm)
        for nm in w:
            if nm not in seen_out:
                seen_out.add(nm); outputs.append(nm)
        defined.update(k)
        toff += c in TOFF
        phase += c in PHASE
        hmr += c == B.C_HMR
        zpow += c == B.C_ZPOW
        conditioned += o.cond >= 0 and c not in (B.C_PUSH, B.C_POP, B.C_INFO)
        # dependency cones
        if c == B.C_PUSH:
            stack.append(set(D(f'b{o.cond}')))
            continue
        if c == B.C_POP:
            if stack:
                stack.pop()
            continue
        ctx = set().union(*stack) if stack else set()
        if o.cond >= 0:
            ctx |= D(f'b{o.cond}')
        q = [f'q{x}' for x in o.q]
        if c in TAPPED:
            tn = tap_name(o, i)
            outputs.append(tn)
            dep[tn] = D(q[0]) | ctx
            if q[0] not in defined and q[0] not in seen_in:
                seen_in.add(q[0]); inputs.append(q[0])
        if c in (B.C_X, B.C_CX, B.C_CCX):
            dep[q[-1]] = D(q[-1]) | ctx | set().union(*[D(x) for x in q[:-1]])
        elif c in (B.C_SWAP, B.C_CSWAP):
            if c == B.C_SWAP and not ctx:
                da, db = D(q[0]), D(q[1])
                dep[q[0]], dep[q[1]] = db, da
            else:
                u = ctx | set().union(*[D(x) for x in q])
                for x in q[-2:]:
                    dep[x] = set(u)
        elif c in (B.C_R, B.C_HMR):
            dep[q[0]] = (D(q[0]) | ctx) if ctx else set()
            if c == B.C_HMR:
                b = f'b{o.bit}'
                dep[b] = (D(b) | ctx) if ctx else set()
        elif c in (B.C_BST0, B.C_BST1):
            b = f'b{o.bit}'
            dep[b] = (D(b) | ctx) if ctx else set()
        elif c == B.C_BINV:
            b = f'b{o.bit}'
            dep[b] = D(b) | ctx
        elif c in PHASE:
            pcone |= ctx.union(*[D(x) for x in q])
    inputs = [f's{j}' for j in range(depth_A)] + inputs
    cones = {nm: sorted(D(nm)) for nm in outputs}
    cones['phase'] = sorted(pcone)
    return dict(inputs=inputs, outputs=outputs, cones=cones, toffoli=toff, phase_ops=phase, hmr=hmr, zpow=zpow,
                conditioned=conditioned)


def drop_fresh(io, lv, A, xq):
    """Remove inputs that are certainly at their initial value at A (untouched scratch wires and bits): before the
    first touch a scratch wire is |0>, so a window starting at op 0 (or before a wire's first use) does not read it.
    Their cones are kept (they name constants, which relevant_inputs ignores). Returns io with `fresh_zero`."""
    fresh = set(lv.fresh_at(io['inputs'], A, xq))
    if fresh:
        io = dict(io, inputs=[nm for nm in io['inputs'] if nm not in fresh])
    io['fresh_zero'] = len(fresh)
    return io


def relevant_inputs(io, oc, phase):
    """Inputs in the static cones of the outputs that are not constant on the reachable states."""
    need = set()
    for nm, col in oc.items():
        if col.any() and not col.all():
            need.update(io['cones'].get(nm, []))
    if phase is not None and phase.any() and not phase.all():
        need.update(io['cones']['phase'])
    return [nm for nm in io['inputs'] if nm in need]


class Liveness:
    """Static next-use liveness after a cut. A wire is dead after B only if its next touch is an unconditional
    R at condition depth 0; HMR counts as a read (its m = 1 phase kickback depends on the value)."""

    def __init__(self, ops):
        self.ops = ops
        n = len(ops)
        depth = np.zeros(n, np.int32)
        d = 0
        self.wire_touch, self.bit_touch = {}, {}
        for i, o in enumerate(ops):
            depth[i] = d
            r, w, k = op_io(o)
            if o.code == B.C_PUSH:
                r = [f'b{o.cond}']
            uncond = o.cond < 0 and d == 0
            for nm in dict.fromkeys(r + w):
                if nm[0] == 'q':
                    dead = o.code == B.C_R and uncond  # HMR reads the value through its phase kickback
                else:
                    dead = nm in k and nm not in r and uncond
                (self.wire_touch if nm[0] == 'q' else self.bit_touch).setdefault(nm, []).append((i, dead))
            if o.code == B.C_PUSH:
                d += 1
            elif o.code == B.C_POP:
                d = max(0, d - 1)
        self.depth = depth

    def first_touch(self, nm):
        tl = (self.wire_touch if nm[0] == 'q' else self.bit_touch).get(nm)
        return tl[0][0] if tl else len(self.ops)

    def fresh_at(self, names, A, xq):
        """Names in `names` that hold their initial value at cut A for certain: scratch wires (not x) and classical
        bits never touched by ops[:A] are |0> / 0 there, so they are constants, not window inputs."""
        xs = {f'q{w}' for w in xq}
        return [nm for nm in names if nm[0] in 'qb' and nm not in xs and self.first_touch(nm) >= A]

    def live_after(self, nm, Bend):
        import bisect
        if '@' in nm:
            return True  # taps: the value was consumed inside the window
        tl = (self.wire_touch if nm[0] == 'q' else self.bit_touch).get(nm, [])
        j = bisect.bisect_left(tl, (Bend, False))
        if j >= len(tl):
            return nm[0] == 'q'  # wires: end state matters (x restored, scratch clean); bits: unused
        return not tl[j][1]


# ---------------------------------------------------------------- reachable states
def snap_cols(sim, names):
    """Packed uint64 column per input name from a running BitSim."""
    cols = []
    for nm in names:
        if nm[0] == 'q':
            cols.append(sim.q[int(nm[1:])].copy())
        elif nm[0] == 'b':
            cols.append(sim.bits.get(int(nm[1:]), np.zeros(sim.W, np.uint64)).copy())
        else:  # condition-stack level
            j = int(nm[1:])
            cols.append(sim.stack[j].copy() if j < len(sim.stack) else sim.valid.copy())
    return cols


_T7, _T14, _T28 = (np.uint64(0x00AA00AA00AA00AA), np.uint64(0x0000CCCC0000CCCC), np.uint64(0x00000000F0F0F0F0))


def keys_from_cols(cols, N):
    """Per row: input key (bit j = input j). Eight wires at a time: their packed bytes side by side form one uint64
    per byte position, and an 8x8 bit-matrix transpose turns it into eight per-row key bytes (row order)."""
    n = len(cols)
    dt = np.uint8 if n <= 8 else np.uint16 if n <= 16 else np.uint32 if n <= 32 else np.uint64
    if n == 0:
        return np.zeros(N, dt)
    key = None
    nb = len(cols[0]) * 8
    for g in range(0, n, 8):
        M = np.zeros((nb, 8), np.uint8)
        for t, c in enumerate(cols[g:g + 8]):
            M[:, t] = c.view(np.uint8)
        x = M.view(np.uint64).ravel()
        t = (x ^ (x >> np.uint64(7))) & _T7
        x = x ^ t ^ (t << np.uint64(7))
        t = (x ^ (x >> np.uint64(14))) & _T14
        x = x ^ t ^ (t << np.uint64(14))
        t = (x ^ (x >> np.uint64(28))) & _T28
        x = x ^ t ^ (t << np.uint64(28))
        b = x.view(np.uint8)[:N]
        if key is None:
            key = b.astype(dt)
        else:
            key |= b.astype(dt) << dt(g)
    return key


def unique_keys(key, n):
    if n <= 22:
        cnt = np.bincount(key.astype(np.int64), minlength=1 << n)
        u = np.flatnonzero(cnt)
        return u.astype(np.uint64), cnt[u]
    u, c = np.unique(key, return_counts=True)
    return u.astype(np.uint64), c


def run_window(ops, nq, A, Bend, names, ukeys, mult, outputs):
    """Run ops[A:Bend] on each distinct reachable input state (zero branch). Returns output columns (bool arrays
    over the distinct states), phase column, mean executed Toffolis per row, weighting each state by `mult` (callers
    pass the random-branch row counts: the verifier charges executed Toffolis under random outcomes)."""
    R = len(ukeys)
    sim = B.BitSim(ops, rows=np.zeros(R, np.int64), num_qubits=nq, x_qubits=[], outcome='zero')
    stack = {}
    for j, nm in enumerate(names):
        col = B.pack_bits(((ukeys >> np.uint64(j)) & np.uint64(1)).astype(bool))
        if nm[0] == 'q':
            sim.q[int(nm[1:])] = col
        elif nm[0] == 'b':
            sim.bits[int(nm[1:])] = col
        else:
            stack[int(nm[1:])] = col
    sim.stack = [stack[j] for j in sorted(stack)]
    sim._recompute_act()
    sim.pos = A
    w = mult.astype(np.float64)
    executed = 0.0
    taps = {}
    for i in range(A, Bend):
        o = ops[i]
        if o.code in TOFF or o.code in TAPPED:
            sim.run(stop=i)
            act = sim.act
            if o.cond >= 0:
                cb = sim._bit(o.cond)
                act = cb if act is None else (act & cb)
            if o.code in TOFF:
                executed += w.sum() if act is None else w[B.unpack_bits(act, R)].sum()
            else:
                v = sim.q[o.q[0]] if act is None else (sim.q[o.q[0]] & act)
                taps[tap_name(o, i)] = B.unpack_bits(v, R)
    sim.run(stop=Bend)
    out = {}
    for nm in outputs:
        if nm in taps:
            out[nm] = taps[nm]
        elif nm[0] == 'q':
            out[nm] = B.unpack_bits(sim.q[int(nm[1:])], R)
        else:
            out[nm] = B.unpack_bits(sim.bits.get(int(nm[1:]), np.zeros(sim.W, np.uint64)), R)
    phase = B.unpack_bits(sim.sign, R)
    return out, phase, executed / max(1.0, float(mult.sum()))


# ---------------------------------------------------------------- GF(2) algebra
def bool_to_int(b):
    """bool array -> python int, bit r = row r."""
    if len(b) == 0:
        return 0
    return int.from_bytes(np.packbits(b.astype(np.uint8), bitorder='little').tobytes(), 'little')


def mobius(t):
    a = t.astype(np.uint8).copy()
    h, n = 1, len(a)
    while h < n:
        v = a.reshape(-1, 2, h)
        v[:, 1, :] ^= v[:, 0, :]
        h *= 2
    return a


def gf2_rank(rows):
    basis, r = {}, 0
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                r += 1
                break
    return r


def nonlinear_rank(monos_list):
    """GF(2) rank of the degree >= 2 residuals (constant and degree-1 terms dropped) of the given ANFs, as vectors
    over monomials. Lower bound on the CCX count of an unconditioned X/CX/CCX/SWAP window computing these outputs
    X/CX/SWAP keep the span of the residuals, each CCX adds at most one product function."""
    idx, rows = {}, []
    for monos in monos_list:
        v = 0
        for m in monos:
            if m & (m - 1):  # popcount >= 2
                v ^= 1 << idx.setdefault(m, len(idx))
        rows.append(v)
    return gf2_rank(rows)


def cubic_derivative_floor(monos_list):
    """Cubic-derivative dimension floor. For outputs of degree
    <= 3: MC-k functions have essential affine dimension <= 2k (Turan-Peralta 2021), so the GF(2) rank r of the
    quadratic coefficient vectors of the partial derivatives of the cubic homogeneous part is <= 2k. Returns
    (per_output_rank, joint_rank, floor) with floor = ceil(max(per_output, joint) / 2); (0, 0, 0) when any output has
    degree > 3. The joint rank (derivatives of all outputs' cubic parts stacked) subsumes the per-output ranks.
    Sound lower bound on the CCX count of an unconditioned X/CX/CCX/SWAP window with exact ANFs."""
    if any(m.bit_count() > 3 for monos in monos_list for m in monos):
        return 0, 0, 0
    best, idx, joint = 0, {}, {}
    for o, monos in enumerate(monos_list):
        cubic = [m for m in monos if m.bit_count() == 3]
        if not cubic:
            continue
        loc, rows = {}, []
        for j in sorted({b for m in cubic for b in range(m.bit_length()) if m >> b & 1}):
            v = jv = 0
            for m in cubic:
                if m >> j & 1:
                    q = m ^ (1 << j)
                    v ^= 1 << loc.setdefault(q, len(loc))
                    jv ^= 1 << idx.setdefault((o, q), len(idx))
            rows.append(v)
            joint.setdefault(j, 0)
            joint[j] ^= jv
        best = max(best, gf2_rank(rows))
    jr = gf2_rank(list(joint.values()))
    return best, jr, (max(best, jr) + 1) // 2


def span_polar_floor(monos_list, k, max_basis=10):
    """Max over the span of the outputs' nonlinear residuals that has degree <= 2 of rank(alternating
    matrix)/2 (Dickson / Mirwald-Schnorr). Unlike span_analysis it enumerates the span of the residual basis, so
    any number of outputs is allowed when their residual rank is <= max_basis. Returns floor, or None if rank > max_basis."""
    idx, vecs, basis = {}, [], {}
    for monos in monos_list:
        v = 0
        for m in monos:
            if m & (m - 1):
                v ^= 1 << idx.setdefault(m, len(idx))
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                vecs.append(v)
                break
    if len(vecs) > max_basis:
        return None
    rev = {i: m for m, i in idx.items()}
    best = 0
    for mask in range(1, 1 << len(vecs)):
        v = 0
        for j, b in enumerate(vecs):
            if mask >> j & 1:
                v ^= b
        ms = [rev[i] for i in range(v.bit_length()) if v >> i & 1]
        if ms and max(m.bit_count() for m in ms) <= 2:
            best = max(best, quad_info(ms, max(k, 1))[1] // 2)
    return best


def value_gates_only(ops, A, Bend):
    """True if ops[A:Bend] are all unconditioned X/CX/CCX/SWAP (the nonlinear-rank floor is sound there)."""
    return all(o.code in VALUE_GATES and o.cond < 0 for o in ops[A:Bend])


def quad_info(monos, k):
    """monos: iterable of monomial masks. -> (quadratic part as int over pair ids, rank of alternating matrix)."""
    rows = [0] * k
    qint = 0
    for m in monos:
        if bin(m).count('1') == 2:
            i = (m & -m).bit_length() - 1
            j = m.bit_length() - 1
            rows[i] ^= 1 << j
            rows[j] ^= 1 << i
            qint ^= 1 << (i * k + j)
    return qint, gf2_rank(rows)


def anf_text(monos, names, limit=None):
    if not monos:
        return '0'
    ms = sorted(monos, key=lambda m: (-bin(m).count('1'), [j for j in range(len(names)) if m >> j & 1]))
    terms = ['1' if m == 0 else '*'.join(names[j] for j in range(len(names)) if m >> j & 1) for m in ms]
    s = ' + '.join(terms)
    if limit and len(s) > limit:
        cut = s[:limit].rsplit(' + ', 1)[0]
        s = f'{cut} + ... ({len(terms)} terms)'
    return s


def deg_of(monos):
    return max((bin(m).count('1') for m in monos), default=-1)


def solve_gf2(features, target):
    """features [(id, int)], target int -> list of ids XORing to target, or None (big-int elimination)."""
    basis = {}
    for idx, (_, v) in enumerate(features):
        combo = 1 << idx
        while v:
            h = v.bit_length() - 1
            if h in basis:
                bv, bc = basis[h]
                v ^= bv
                combo ^= bc
            else:
                basis[h] = (v, combo)
                break
    v, combo = target, 0
    while v:
        h = v.bit_length() - 1
        if h not in basis:
            return None
        bv, bc = basis[h]
        v ^= bv
        combo ^= bc
    return [features[i][0] for i in range(len(features)) if (combo >> i) & 1]


def min_degree_completion(pcols, target, R, max_deg=DC_MAX_DEG):
    """Lowest-degree polynomial over the pivot columns agreeing with target on the R reachable rows.
    -> (monos or None, degree tried up to)."""
    from itertools import combinations
    k = len(pcols)
    full = (1 << R) - 1
    feats = [(0, full)]
    for d in range(0, max_deg + 1):
        if d >= 1:
            for comb in combinations(range(k), d):
                v = full
                for j in comb:
                    v &= pcols[j]
                feats.append((sum(1 << j for j in comb), v))
        sol = solve_gf2(feats, target)
        if sol is not None:
            return sorted(sol), d
    return None, max_deg


# ---------------------------------------------------------------- the analysis
def analyse(names, ukeys, mult, outcols, phase, live, declared=None, anf_full_max=ANF_FULL_MAX,
            anf_dc_max=ANF_DC_MAX, nl_rank_ok=False):
    """Function of each output on the reachable input states. names: input names (bit j of a key); ukeys: distinct
    reachable keys; outcols: {output: bool array over ukeys}; live: {output: bool}. declared: express in terms of
    these input names only (checked: conflicts reported)."""
    R0 = len(ukeys)
    res = dict(inputs_raw=len(names), reachable_raw=int(R0))
    var_idx = list(range(len(names)))
    keys = ukeys
    oc = dict(outcols)
    if phase is not None and phase.any():
        oc['phase'] = phase
    conflicts = {}
    if declared is not None:
        miss = [d for d in declared if d not in names]
        var_idx = [names.index(d) for d in declared if d in names]
        res['declared_not_read'] = miss
        mask = np.uint64(sum(1 << j for j in var_idx))
        dk = keys & mask
        order = np.argsort(dk, kind='stable')
        sk = dk[order]
        first = np.ones(len(sk), bool)
        first[1:] = sk[1:] != sk[:-1]
        grp = np.cumsum(first) - 1
        newkeys = sk[first]
        newoc = {}
        for nm, col in oc.items():
            v = col[order]
            g_or = np.zeros(len(newkeys), bool)
            g_and = np.ones(len(newkeys), bool)
            np.logical_or.at(g_or, grp, v)
            np.logical_and.at(g_and, grp, v)
            conflicts[nm] = int((g_or & ~g_and).sum())
            newoc[nm] = g_or
        newmult = np.bincount(grp, weights=mult[order].astype(np.float64)).astype(np.int64)
        keys, oc, mult = newkeys, newoc, newmult
    R = len(keys)
    vnames = [names[j] for j in var_idx]
    cols = [((keys >> np.uint64(j)) & np.uint64(1)).astype(bool) for j in var_idx]
    constants, varying = {}, []
    for nm, c in sorted(zip(vnames, cols), key=lambda t: ('sbq'.index(t[0][0]), int(t[0][1:]))):
        if c.all() or not c.any():
            constants[nm] = int(c[0]) if R else 0
        else:
            varying.append((nm, c))
    # affine hull: pivots over varying inputs, rows translated by the first reachable state
    s0 = {nm: bool(c[0]) for nm, c in varying}
    basis = {}  # leading bit -> (vec, combo over pivots, pivot slot)
    pivots, relations = [], {}
    for nm, c in varying:
        v = bool_to_int(c ^ c[0])
        combo = 0
        while v:
            h = v.bit_length() - 1
            if h in basis:
                bv, bc = basis[h]
                v ^= bv
                combo ^= bc
            else:
                break
        if v:
            slot = len(pivots)
            basis[v.bit_length() - 1] = (v, combo ^ (1 << slot))
            pivots.append((nm, c))
        else:  # nm = s0[nm] + sum over combo of (pivot + s0[pivot])
            const = s0[nm] ^ (sum(s0[pivots[j][0]] for j in range(len(pivots)) if combo >> j & 1) & 1)
            relations[nm] = ' + '.join([pivots[j][0] for j in range(len(pivots)) if combo >> j & 1]
                                       + (['1'] if const else [])) or '0'
    k = len(pivots)
    pn = [p[0] for p in pivots]
    full = R == (1 << k)
    res.update(inputs_used=len(vnames), constants=constants, relations=relations, pivots=pn, k=k,
               reachable=int(R), domain=1 << k if k < 63 else None, dont_cares_in_hull=(1 << k) - R if k < 63 else None,
               unreachable_raw=(1 << len(vnames)) - R if len(vnames) < 63 else None, full_domain=bool(full))
    if conflicts:
        res['not_determined_by_declared'] = {nm: c for nm, c in conflicts.items() if c}
    outs = {}
    if full and k <= anf_full_max:
        idx = np.zeros(R, np.int64)
        for j, (_, c) in enumerate(pivots):
            idx |= c.astype(np.int64) << j
        res['anf_kind'] = 'exact (reachable states fill the affine hull)'
        for nm, col in oc.items():
            t = np.zeros(1 << k, np.uint8)
            t[idx] = col
            monos = [int(m) for m in np.flatnonzero(mobius(t))]
            outs[nm] = dict(monos=monos, deg=deg_of(monos), exact=True)
    elif not full and k <= anf_dc_max:
        pc = [bool_to_int(c) for _, c in pivots]
        res['anf_kind'] = f'min-degree completion on reachable states (degree <= {DC_MAX_DEG} searched)'
        res['quad_completion_unique'] = quad_unique(pc, R)
        for nm, col in oc.items():
            monos, d = min_degree_completion(pc, bool_to_int(col), R)
            outs[nm] = dict(monos=monos, deg=deg_of(monos) if monos is not None else None,
                            min_deg_at_least=None if monos is not None else d + 1, exact=False)
    else:
        res['anf_kind'] = f'not computed: {k} pivots (limit {anf_full_max} full / {anf_dc_max} with don\'t-cares)'
        for nm, col in oc.items():
            outs[nm] = dict(monos=None, deg=None, exact=False, constant=bool(col.all() or not col.any()))
    for nm, o in outs.items():
        o['live'] = True if nm == 'phase' else bool(live.get(nm, True))
        if o['monos'] is not None:
            o['support'] = [pn[j] for j in range(k) if any(m >> j & 1 for m in o['monos'])]
            o['kind'] = {-1: 'zero', 0: 'constant', 1: 'affine', 2: 'quadratic'}.get(o['deg'], f"degree {o['deg']}")
            if o['deg'] == 2:
                o['rank'] = quad_info(o['monos'], k)[1]
        else:
            o['kind'] = (f"degree >= {o['min_deg_at_least']}" if o.get('min_deg_at_least') else
                         ('constant' if o.get('constant') else 'unknown'))
        if nm in oc and nm in names:  # restored?
            j = names.index(nm)
            if j in var_idx:
                c = cols[var_idx.index(j)]
                o['restored'] = bool(np.array_equal(c, oc[nm]))
    res['outputs'] = outs
    mode = 'exact' if res.get('anf_kind', '').startswith('exact') else (
        'dc-unique2' if res.get('quad_completion_unique') else 'dc')
    res['floor'] = floor(outs, k, mode, nl_rank_ok=nl_rank_ok)
    res['pivot_names'] = pn
    return res


def quad_unique(pcols, R):
    """True if no nonzero polynomial of degree <= 2 vanishes on the reachable rows (degree-<=2 completions unique)."""
    from itertools import combinations
    k = len(pcols)
    full = (1 << R) - 1
    feats = [full] + list(pcols) + [pcols[i] & pcols[j] for i, j in combinations(range(k), 2)]
    return gf2_rank(feats) == len(feats)


def span_analysis(nonlin, k):
    """Enumerate the span of the nonlinear outputs' ANFs: max degree, max(deg-1, rank/2), dim and min-weight basis
    of the quadratic parts."""
    names_nl = sorted(nonlin)
    sets = [set(nonlin[nm]['monos']) for nm in names_nl]
    elems, maxdeg, lb = [], 0, 0
    for mask in range(1, 1 << len(sets)):
        s = set()
        for j in range(len(sets)):
            if mask >> j & 1:
                s ^= sets[j]
        d = deg_of(s)
        maxdeg = max(maxdeg, d)
        lb = max(lb, d - 1)
        if d == 2:
            qi, r = quad_info(s, k)
            lb = max(lb, r // 2)
            elems.append((r, qi))
    dim = ub = None
    if nonlin and maxdeg == 2:
        elems.sort()
        basis, dim, ub = {}, 0, 0
        for r, qi in elems:
            v = qi
            while v:
                h = v.bit_length() - 1
                if h in basis:
                    v ^= basis[h]
                else:
                    basis[h] = v
                    dim += 1
                    ub += r // 2
                    break
    return dict(lb=lb, maxdeg=maxdeg, dim=dim, ub=ub)


def floor(outs, k, mode='exact', nl_rank_ok=False):
    """AND-count lower/upper bounds for the live outputs; see the module docstring for the theorems used.
    mode: 'exact' (unique ANF on the affine hull), 'dc-unique2' (don't-cares, but the degree-<=2 completion is
    unique, so any other completion has a component of degree >= 3 and needs >= 2 ANDs), 'dc'."""
    val = {nm: o for nm, o in outs.items() if o['live'] and nm != 'phase'}
    ph = outs.get('phase')
    notes = []
    lb, lb_model, ub = 0, None, None
    unknown = [nm for nm, o in val.items() if o['monos'] is None and o['kind'] != 'constant']
    nonlin = {nm: o for nm, o in val.items() if o['monos'] is not None and o['deg'] >= 2}
    for o in val.values():  # min-degree completions: sound per output
        if o['monos'] is not None and o['deg'] >= 2:
            lb = max(lb, o['deg'] - 1)
        elif o.get('min_deg_at_least'):
            lb = max(lb, o['min_deg_at_least'] - 1)
    if not unknown and len(nonlin) <= SPAN_MAX:
        sp = span_analysis(nonlin, k)
        if not nonlin:
            ub = 0
        elif sp['ub'] is not None:
            ub = sp['ub']
            notes.append(f"quadratic parts span dimension {sp['dim']}; ub = min-weight basis (Dickson), placement "
                         'unchecked')
        if mode == 'exact':
            lb = max(lb, sp['lb'])
            if sp['dim'] is not None:
                lb_model = max(lb, sp['dim'])
            notes.append('lb: max over the span of the live outputs of max(deg-1, rank/2); exact functions')
        elif mode == 'dc-unique2':
            lb = max(lb, min(sp['lb'], 2))
            if sp['dim'] is not None:
                lb_model = max(lb, min(sp['dim'], 2))
            notes.append("lb: degree-<=2 completion unique; span max(deg-1, rank/2) capped at 2 (any other "
                         'completion has a degree >= 3 component)')
        else:
            notes.append("lb: per-output min degree - 1 (don't-cares; ranks over completions not minimised)")
    else:
        notes.append('lb: per-output degree only' + (f'; unknown outputs {",".join(unknown[:4])}' if unknown else ''))
    # nonlinear-output rank: lower bound only, never lowers lb or touches ub. Needs the exact ANF (unique
    # on the affine hull) of every live output and a window of unconditioned X/CX/CCX/SWAP only.
    nlr = None
    cubic_fl = spf = None
    if not nl_rank_ok:
        notes.append('nonlinear-output rank: not applicable (window has gates other than unconditioned X/CX/CCX/SWAP)')
    elif mode != 'exact' or unknown:
        notes.append('nonlinear-output rank: not computed (ANF not exact on the affine hull)')
    else:
        nlr = nonlinear_rank([o['monos'] for o in val.values()])
        lb = max(lb, nlr)
        notes.append(f'nonlinear-output rank {nlr} (lower bound; GF(2) rank of the degree>=2 ANF residuals of all live '
                     'outputs)')
        mlist = [o['monos'] for o in val.values()]
        cd = cubic_derivative_floor(mlist)
        cubic_fl = cd[2] if cd[1] or cd[0] else None
        if cubic_fl is not None:
            lb = max(lb, cubic_fl)
            notes.append(f'cubic derivative rank {cd[0]} per output, {cd[1]} joint -> floor {cubic_fl} (lower bound; '
                         'MC-k functions have essential affine dimension <= 2k; Turan-Peralta 2021)')
        spf = span_polar_floor(mlist, k)
        if spf is not None:
            lb = max(lb, spf)
            notes.append(f'residual-span polar rank floor {spf} (lower bound)')
    if ph is not None:
        pd = ph['deg'] if ph['monos'] is not None else ph.get('min_deg_at_least')
        if pd is None or pd >= 3:
            ub = None
            if pd is not None:
                lb = max(lb, 1)
                notes.append(f'phase degree {pd} >= 3 needs a Toffoli-class gate; no construction offered')
    if ub is not None and lb == ub:
        kind = 'exact'
    elif ub is not None and lb_model is not None and lb_model == ub:
        kind = 'exact (degree-2 model)'
    else:
        kind = 'bound'
    return dict(lb=lb, lb_degree2_model=lb_model, ub=ub, kind=kind, notes=notes, floor_nonlinear_rank=nlr,
                floor_cubic_derivative=cubic_fl, floor_span_polar=spf)


# ---------------------------------------------------------------- one window end to end (library)
def mark_outcome_dependent(res, C):
    """States seen only in the random branch: the reachable set is then sampled over outcome patterns. lb stays a
    valid lower bound (a subset of states has fewer constraints); ub is only checked on the observed states."""
    od = C.shape[1] > 1 and bool((C[:, 0] == 0).any())
    res['outcome_dependent'] = od
    if od:
        fl = res['floor']
        fl['ub_observed'] = fl['ub']
        fl['ub'] = None
        fl['kind'] = 'bound (outcome-sampled states)'
        fl['notes'].append('input states depend on measurement outcomes (seen only in the random branch): ub '
                           'withheld; check with more qhdl_bitsim seeds')
    return res


def merge_keys(parts):
    """Union of per-branch (ukeys, counts) -> (ukeys, C) with C[:, j] = rows of branch j in each state."""
    u = np.concatenate([p[0] for p in parts])
    uu, inv = np.unique(u, return_inverse=True)
    C = np.zeros((len(uu), len(parts)), np.int64)
    off = 0
    for j, (k, c) in enumerate(parts):
        C[inv[off:off + len(k)], j] = c
        off += len(k)
    return uu, C


def window_function(ops, xq, nq, A, Bend, rows=None, outs=None, declared=None, liveness=None,
                    branches=BRANCHES):
    """Simulate to A in each outcome branch, collect reachable input states (union), run the window, analyse."""
    if liveness is None:
        liveness = Liveness(ops)
    io = drop_fresh(window_io(ops, A, Bend, int(liveness.depth[A]) if A < len(ops) else 0), liveness, A, xq)
    names = io['inputs']
    if len(names) > MAX_INPUTS:
        raise ValueError(f'window reads {len(names)} inputs (> {MAX_INPUTS}); narrow --ops or pass --in')
    parts = []
    for br in branches:
        sim = B.BitSim(ops, rows=rows, num_qubits=nq, x_qubits=xq, outcome=br)
        got = {}
        sim.run(stop=A, snaps=[A], on_snap=lambda i, s: got.setdefault('cols', snap_cols(s, names)))
        parts.append(unique_keys(keys_from_cols(got['cols'], sim.n), len(names)))
    ukeys, C = merge_keys(parts)
    mult = C.sum(1)
    outputs = io['outputs']
    if outs:
        bad = [o for o in outs if o not in outputs]
        if bad:
            raise ValueError(f"--out {','.join(bad)} not written in ops {A}..{Bend - 1} (written: "
                             f"{','.join(outputs[:12])}{'...' if len(outputs) > 12 else ''})")
        outputs = list(outs)
    oc, phase, executed = run_window(ops, nq, A, Bend, names, ukeys, C[:, -1], outputs)
    live = {nm: liveness.live_after(nm, Bend) for nm in outputs}
    auto = declared is None
    if auto:
        declared = relevant_inputs(io, oc, phase if not outs else None)
    res = analyse(names, ukeys, mult, oc, phase if not outs else None, live, declared=declared,
                  nl_rank_ok=value_gates_only(ops, A, Bend))
    mark_outcome_dependent(res, C)
    res['inputs_mode'] = 'static cones of non-constant outputs' if auto else '--in'
    res.update(ops=[A, Bend], io={k: v for k, v in io.items() if k not in ('inputs', 'outputs', 'cones')},
               input_names=names, output_names=outputs, executed_toffoli=executed, rows=int(sim.n))
    return res


def summarise_outputs(res, limit=70):
    rows = []
    pn = res['pivot_names']
    for nm, o in res['outputs'].items():
        r = dict(out=nm, live=o['live'], kind=o['kind'])
        if o.get('restored'):
            r['restored'] = True
        if o.get('support') is not None:
            r['support'] = len(o['support'])
        if o.get('rank') is not None:
            r['rank'] = o['rank']
        if o['monos'] is not None:
            r['anf'] = anf_text(o['monos'], pn, limit)
        rows.append(r)
    order = {'phase': 0}
    rows.sort(key=lambda r: (not r['live'], r.get('restored', False), -(r.get('rank') or 0),
                             r['kind'] in ('constant', 'zero', 'affine'), order.get(r['out'], 1), r['out']))
    return rows


# ---------------------------------------------------------------- scan
def candidate_windows(ops, size=4, span=48):
    """Toffoli-led windows: from each Toffoli-class gate T[a], every run T[a..b] of 1..size consecutive
    Toffoli-class gates with T[b] - T[a] <= span; window = [T[a], T[b] + 1). Grouped by start: {A: [B, ...]}."""
    T = [i for i, o in enumerate(ops) if o.code in TOFF]
    out = {}
    for a in range(len(T)):
        ends = [T[a] + 1]
        b = a
        while b + 1 < len(T) and b + 1 - a < size and T[b + 1] - T[a] <= span:
            b += 1
            ends.append(T[b] + 1)
        out[T[a]] = ends
    return out


def project_keys(ukeys, mult, names, sub):
    """Distinct states of a subset of the inputs (sub must be a subset of names), with multiplicities."""
    if list(sub) == list(names):
        return ukeys, mult
    k = np.zeros(len(ukeys), np.uint64)
    for j, nm in enumerate(sub):
        i = names.index(nm)
        k |= ((ukeys >> np.uint64(i)) & np.uint64(1)) << np.uint64(j)
    u, inv = np.unique(k, return_inverse=True)
    out = np.zeros((len(u), mult.shape[1]), np.int64)
    np.add.at(out, inv, mult)
    return u, out


def scan(ops, xq, nq, size=4, span=48, max_inputs=20, rows=None, branches=BRANCHES):
    """All candidate windows; per start one key extraction (for the largest window within max_inputs) per
    outcome branch; shorter windows from the same start project those states. Returns (results, skipped, timing)."""
    t0 = time.time()
    lv = Liveness(ops)
    cands = candidate_windows(ops, size, span)
    base, subs, skipped = {}, {}, []
    for A, ends in cands.items():
        ios = [(Bend, drop_fresh(window_io(ops, A, Bend, int(lv.depth[A])), lv, A, xq)) for Bend in ends]
        ok = [(Bend, io) for Bend, io in ios if len(io['inputs']) <= max_inputs]
        skipped += [(A, Bend, len(io['inputs'])) for Bend, io in ios if len(io['inputs']) > max_inputs]
        if ok:
            base[A] = ok[-1][1]['inputs']  # inputs grow with the window
            subs[A] = ok
    pending = {}

    def cb(i, s):
        names = base[i]
        pending.setdefault(i, []).append(unique_keys(keys_from_cols(snap_cols(s, names), s.n), len(names)))
    n = 0
    for br in branches:
        sim = B.BitSim(ops, rows=rows, num_qubits=nq, x_qubits=xq, outcome=br)
        sim.run(snaps=sorted(base), on_snap=cb)
        n = sim.n
    t_sim = time.time() - t0
    results, seen = [], set()
    for A in sorted(subs):
        uk, mu = merge_keys(pending.pop(A))
        for Bend, io in subs[A]:
            if (A, Bend) in seen:
                continue
            seen.add((A, Bend))
            names = io['inputs']
            ukeys, C = project_keys(uk, mu, base[A], names)
            mult = C.sum(1)
            oc, phase, executed = run_window(ops, nq, A, Bend, names, ukeys, C[:, -1], io['outputs'])
            live = {nm: lv.live_after(nm, Bend) for nm in io['outputs']}
            res = analyse(names, ukeys, mult, oc, phase, live, declared=relevant_inputs(io, oc, phase),
                          anf_full_max=16, anf_dc_max=12, nl_rank_ok=value_gates_only(ops, A, Bend))
            mark_outcome_dependent(res, C)
            res.update(ops=[A, Bend], io={k: v for k, v in io.items() if k not in ('inputs', 'outputs', 'cones')},
                       input_names=names, output_names=io['outputs'], executed_toffoli=executed, rows=int(n))
            results.append(res)
    return results, skipped, dict(simulate=round(t_sim, 1), analyse=round(time.time() - t0 - t_sim, 1))


# ---------------------------------------------------------------- CLI glue
def parse_ops(s):
    try:
        a, b = s.split(':')
        a, b = int(a), int(b)
    except ValueError:
        raise ValueError(f'--ops takes A:B (0-based op indices, B exclusive), got {s!r}')
    if not 0 <= a < b:
        raise ValueError(f'--ops {s}: need 0 <= A < B')
    return a, b


def parse_names(s):
    if not s:
        return None
    out = []
    for t in s.split(','):
        t = t.strip()
        if t.isdigit():
            t = 'q' + t
        if not (len(t) > 1 and t[0] in 'qbs' and t[1:].isdigit()):
            raise ValueError(f'bad wire/bit name {t!r} (use q14, b8367, s0)')
        out.append(t)
    return out


def _open_invsqrt(api, ref):
    c = api.resolve(ref)
    row = c.get('manifest') or {}
    task = row.get('task')
    if task and task != 'invsqrt':
        raise api.QhdlError(
            f"{c['sha'][:12]} is a {task} circuit: window/windows/fdiff need the enumerable invsqrt domain "
            '(all 3,133,452 legal x)',
            'for DQI use `qhdl peak` / `qhdl open`, the fork workbench (`workbench query --sort toffoli`) or '
            'scripts/qhdl_dqi_peak_anatomy.py')
    meta = api.ensure_open(ref)
    if 'ops_table' in meta or meta.get('task') != 'invsqrt':
        raise api.QhdlError(
            f"{meta['sha12']}: window analysis needs an invsqrt circuit with an op table (task {meta.get('task')}, "
            f"{api.fmt_int(meta.get('num_ops', 0))} ops)",
            'DQI-sized circuits: `workbench query` in kickmix-macos tools/circuit-viewer, or '
            'scripts/qhdl_dqi_peak_anatomy.py')
    L = api.Loaded(meta)
    if L.nq > 64:
        raise api.QhdlError(f'{L.nq} wires > 64: not enumerable')
    return meta, L


def _tool_sha(api):
    files = [HERE / f for f in getattr(api, 'TOOL_FILES', []) if (HERE / f).exists()] + [Path(__file__)]
    return E.files_sha(files)


DOMAIN = ('all {} legal x in two outcome branches (all HMR = 0; random, seed 7); functions on the input states '
          'reached at op {} (in-window HMRs: zero branch)')


def _floor_text(fl, toff):
    if fl['ub'] is not None and fl['kind'].startswith('exact'):
        s = f"AND floor {fl['ub']} ({fl['kind']})"
    else:
        s = f"AND floor >= {fl['lb']}" + (f", <= {fl['ub']} constructive" if fl['ub'] is not None else '')
    return s + f" vs {toff} Toffolis"


def v_window(api, a):
    led = E.Ledger()
    A, Bend = parse_ops(a.ops)
    meta, L = _open_invsqrt(api, a.circuit)
    if Bend > len(L.ops):
        raise api.QhdlError(f'--ops {a.ops}: circuit has {len(L.ops)} ops')
    outs, declared = parse_names(a.out), parse_names(a.inputs)
    tsha = _tool_sha(api)
    key = dict(tool_sha=tsha, verb='window', ops=[A, Bend], out=outs, inp=declared)
    tag = f"{A}_{Bend}" + (f"_out{'-'.join(outs)}" if outs else '') + (f"_in{'-'.join(declared)}" if declared else '')
    ev = L.dir / 'window' / f'{tag[:120]}.json'
    hit = api.cached(ev, key)
    t0 = time.time()
    if hit is None:
        res = window_function(L.ops, L.xq, L.nq, A, Bend, outs=outs, declared=declared)
        res['cache_key'] = key
        res['seconds'] = round(time.time() - t0, 1)
        for o in res['outputs'].values():
            if o.get('monos') is not None:
                o['anf'] = anf_text(o['monos'], res['pivot_names'])
        api.write_json(ev, res)
        hit = res
    res = hit
    led.earn('static', 'window inputs/outputs, liveness and Toffoli counts from the op stream')
    led.earn('exact-all-x', f'bitset simulation of all {api.fmt_int(api.MAX_X)} legal x to op {A} (zero + random seed-7 branch); '
                            'window run on every distinct reachable input state')
    fl = res['floor']
    toff = res['io']['toffoli']
    live_nl = [nm for nm, o in res['outputs'].items() if o['live'] and o['kind'] not in
               ('zero', 'constant', 'affine') and not o.get('restored')]
    dc = res.get('dont_cares_in_hull')
    claim = (f"ops {A}..{Bend - 1}: {len(res['input_names'])} inputs read, reachable states span a {res['k']}-dim "
             f"affine space ({api.fmt_int(res['reachable'])} of {api.fmt_int(res['domain'] or 0)} states"
             + (f", {api.fmt_int(dc)} don't-cares" if dc else ', no don\'t-cares') + f"); {len(live_nl)} live "
             f"nonlinear output(s) {','.join(live_nl[:4])}; " + _floor_text(fl, toff)
             + (f"; slack {toff - fl['ub']} sure" if fl['ub'] is not None and toff > fl['ub'] else '')
             + (f", {toff - fl['lb']} max" if toff > fl['lb'] else ''))
    facts = dict(toffoli_static=toff, toffoli_mean_executed=round(res['executed_toffoli'], 3),
                 floor=dict(lb=fl['lb'], ub=fl['ub'], kind=fl['kind'], model_lb=fl['lb_degree2_model'],
                            floor_nonlinear_rank=fl.get('floor_nonlinear_rank')),
                 floor_nonlinear_rank=(f"{fl['floor_nonlinear_rank']} (lower bound; folded into lb)"
                                       if fl.get('floor_nonlinear_rank') is not None else 'not applicable'),
                 pivots=','.join(res['pivot_names']), anf=res['anf_kind'].split(' (')[0],
                 constants=len(res['constants']), relations=len(res['relations']),
                 phase_ops=res['io']['phase_ops'], hmr=res['io']['hmr'])
    if res['io'].get('fresh_zero'):
        facts['fresh_zero'] = res['io']['fresh_zero']  # untouched scratch wires/bits: |0> at A, not inputs
    if res.get('outcome_dependent'):
        facts['outcome_dependent'] = 'states seen only under random outcomes: sampled; ub withheld'
    if res['io']['hmr'] or res['io']['zpow']:
        facts['caveat'] = 'in-window HMRs in the zero branch; values at HMR/Z_POW are tap outputs'
    if res.get('not_determined_by_declared'):
        facts['not_determined_by_in'] = res['not_determined_by_declared']
    rows = summarise_outputs(res)
    for r in rows:
        r['tier'] = 'exact-all-x'
    rr = api.cmd('window', '-c', meta['sha12'], '--ops', f'{A}:{Bend}', *(['--out', a.out] if a.out else []),
                 *(['--in', a.inputs] if a.inputs else []))
    return E.Envelope(led, claim, domain=DOMAIN.format(api.fmt_int(api.MAX_X), A), evidence=api.rel(ev), rerun=rr,
                      tool_sha=tsha, circuit_sha=meta['sha256'], facts=facts, rows=rows,
                      next_cmds=[api.cmd('windows', '-c', meta['sha12']),
                                 api.cmd('link', '-c', meta['sha12'], '--op', A)])


def _scan_row(res, stage=''):
    fl = res['floor']
    toff = res['io']['toffoli']
    nl = [nm for nm, o in res['outputs'].items() if o['live'] and o['kind'] not in ('zero', 'constant', 'affine')
          and not o.get('restored')]
    return dict(ops=f"{res['ops'][0]}:{res['ops'][1]}", toffoli=toff, exec=round(float(res['executed_toffoli']), 2),
                lb=fl['lb'], ub=fl['ub'], floor=fl['kind'], nlrank=fl.get('floor_nonlinear_rank'),
                sure=(toff - fl['ub']) if fl['ub'] is not None else None,
                charged=round(float(res['executed_toffoli'] - fl['ub']), 2) if fl['ub'] is not None else None,
                max=toff - fl['lb'],
                k=res['k'], dc=res.get('dont_cares_in_hull'), outs=','.join(nl[:4]), stage=stage)


def collapse(rows):
    """Rank by positive charged slack (mean executed Toffolis - ub), then windows with a constructive ub by static
    sure slack, then max slack (static - proved lb), exact floors first, fewer Toffolis; a window overlapping a better-or-equal ranked
    one is marked nested_in (kept in the table, hidden from stdout)."""
    def rng(r):
        a, b = r['ops'].split(':')
        return int(a), int(b)
    def nz(v):
        return v if v is not None else -99
    rows.sort(key=lambda r: (-max(nz(r['charged']), 0), r['ub'] is None, -nz(r['sure']), -r['max'],
                             r['floor'] != 'exact', r['toffoli'], rng(r)[0]))
    kept = []
    for r in rows:
        a, b = rng(r)
        host = next((k for k in kept if rng(k)[0] < b and a < rng(k)[1]), None)
        if host is not None:
            r['nested_in'] = host['ops']
        else:
            r['nested_in'] = ''
            kept.append(r)
    return rows


def v_windows(api, a):
    led = E.Ledger()
    meta, L = _open_invsqrt(api, a.circuit)
    tsha = _tool_sha(api)
    key = dict(tool_sha=tsha, verb='windows', size=a.size, span=a.span, max_inputs=a.max_inputs)
    ev = L.dir / 'window' / f'scan_s{a.size}_p{a.span}_i{a.max_inputs}.json'
    tsv = ev.with_suffix('.tsv')
    hit = api.cached(ev, key)
    if hit is None:
        t0 = time.time()
        results, skipped, timing = scan(L.ops, L.xq, L.nq, size=a.size, span=a.span, max_inputs=a.max_inputs)
        rows = [_scan_row(r, L.stage(r['ops'][0])) for r in results]
        rows = collapse(rows)
        full = []
        for r in results:
            d = {k: v for k, v in r.items() if k not in ('outputs',)}
            d['outputs'] = {nm: {k: v for k, v in o.items() if k != 'monos'} | (
                {'anf': anf_text(o['monos'], r['pivot_names'], 400)} if o.get('monos') is not None else {})
                for nm, o in r['outputs'].items()}
            full.append(d)
        hit = dict(cache_key=key, seconds=round(time.time() - t0, 1), timing=timing, windows=len(results),
                   skipped_too_many_inputs=len(skipped), skipped=skipped[:200], rows=rows)
        api.write_json(ev.with_name(ev.stem + '_full.json'), dict(cache_key=key, results=full))
        api.write_tsv(tsv, list(rows[0]) if rows else ['ops'], [list(r.values()) for r in rows])
        api.write_json(ev, hit)
    led.earn('static', 'candidate windows, liveness and Toffoli counts from the op stream')
    led.earn('exact-all-x', f'bitset simulation of all {api.fmt_int(api.MAX_X)} legal x (zero + random seed-7 branch); each '
                            'window run on its distinct reachable input states')
    rows = [r for r in hit['rows'] if not r.get('nested_in')]
    sure = [r for r in rows if r['charged'] is not None and r['charged'] > 0]
    maybe = [r for r in rows if r['max'] > 0]
    top = [{k: v for k, v in r.items() if k != 'nested_in'} for r in rows[:a.top]]
    for r in top:
        r['tier'] = 'exact-all-x'
    claim = (f"{hit['windows']} Toffoli-led windows (1..{a.size} Toffolis within {a.span} ops) analysed in "
             f"{hit['seconds']} s; after folding nested windows: {len(sure)} with a constructive algebraic saving "
             f"(charged = mean executed Toffolis - ub > 0; placement, width and phase unchecked), {len(maybe)} with Toffolis above the proved floor; "
             f"{hit['skipped_too_many_inputs']} skipped (> {a.max_inputs} inputs)")
    facts = dict(charged_slack_total=round(sum(r['charged'] for r in sure), 2),
                 legend='sure = static Toffolis - ub; charged = mean executed (random branch) - ub; max = static - lb; '
                        'dc = don\'t-cares in the affine hull', timing_s=hit.get('timing'))
    nxt = [api.cmd('window', '-c', meta['sha12'], '--ops', r['ops']) for r in top[:2]]
    return E.Envelope(led, claim, domain=f"all {api.fmt_int(api.MAX_X)} legal x in two outcome branches (zero; random seed 7)",
                      evidence=api.rel(tsv), rerun=api.cmd('windows', '-c', meta['sha12'], '--size', a.size, '--span',
                                                           a.span), tool_sha=tsha, circuit_sha=meta['sha256'],
                      facts=facts, rows=top, table=api.rel(tsv), next_cmds=nxt)


FDIFF_SEED = 7  # the random measurement branch of fdiff (same seed as window/windows)


def _map_ops(api, a, m1, m2, L1, L2, A1, B1):
    """--map PATH (the JSON `qhdl edit` writes) or --map auto (align the two op-text streams): cut A1/B1 of the first
    circuit -> cut in the second. Returns (A2, B2, source)."""
    sys.path.insert(0, str(HERE))
    import qhdl_edit as ED
    if a.map == 'auto':
        hunks = ED.auto_hunks([o.text for o in L1.ops], [o.text for o in L2.ops])
        om, src = ED.OpMap(hunks, len(L1.ops), len(L2.ops)), 'auto (aligned op texts)'
    else:
        try:
            om, d = ED.OpMap.load(a.map)
        except (OSError, ValueError, KeyError) as exc:
            raise api.QhdlError(f'cannot read --map {a.map}: {exc}', 'pass the .map.json `qhdl edit` wrote, or --map auto')
        olds, news = d.get('old_sha256'), d.get('new_sha256')
        if olds and news and (olds, news) != (m1['sha256'], m2['sha256']):
            if (olds, news) == (m2['sha256'], m1['sha256']):
                raise api.QhdlError('--map runs from -c2 to -c (edit input/output reversed)',
                                    'swap -c and -c2: -c is the circuit the edit read, -c2 the one it wrote')
            raise api.QhdlError(f'--map is for {olds[:12]} -> {news[:12]}, not {m1["sha12"]} -> {m2["sha12"]}')
        src = a.map
    def cut(x, end):
        # an insertion exactly at a cut belongs to the window on both sides (the edit's new ops are inside)
        ins = [h for h in om.h if h[0] == 'insert' and h[1] == x]
        if ins:
            return ins[-1][4] if end else ins[0][3]
        return om.cut(x)
    inside = sum(1 for h in om.edits() if A1 <= h[1] and h[2] <= B1)
    try:
        return cut(A1, False), cut(B1, True), f'{src}; {inside} edit hunk(s) inside the window'
    except ED.EditError as exc:
        raise api.QhdlError(str(exc), 'widen --ops so both cuts lie outside the edited ops (`qhdl edit` rows list them)')


def _final_columns(L, outcome):
    """Run a whole invsqrt circuit on all legal x. Per x: phase (2^-64 turns: sign + Z_POW), x restored, scratch
    clean; plus mean executed Toffolis."""
    nq = max([L.nq] + [w + 1 for w in L.xq])
    sim = B.BitSim(L.ops, num_qubits=nq, x_qubits=L.xq, outcome=outcome, count_toffoli=True).run()
    bad_x = np.zeros(sim.W, np.uint64)
    for k, qq in enumerate(L.xq):
        bad_x |= sim.q[qq] ^ B.pack_bits(((sim.rows >> k) & 1).astype(bool))
    xs = set(L.xq)
    dirty = np.zeros(sim.W, np.uint64)
    for w in range(sim.nq):
        if w not in xs:
            dirty |= sim.q[w]
    return dict(phase=sim.phase_units(), bad_x=B.unpack_bits(bad_x, sim.n), dirty=B.unpack_bits(dirty, sim.n),
                toffoli=sim.toffoli_rows / sim.n, n=sim.n)


def fdiff_whole(L1, L2, seed=FDIFF_SEED):
    """Whole-circuit comparison on all legal x in the zero branch and one seeded random-outcome branch per side
    (draws are not matched across circuits: a correct MBU circuit's end state is outcome-independent). Phase is compared
    exactly per x (sign + rotation units); the official tolerance check of each side is reported separately."""
    out = dict(branches={}, contract=[])
    finals = {}
    for br in ('zero', seed):
        f1, f2 = _final_columns(L1, br), _final_columns(L2, br)
        finals[br] = (f1['phase'], f2['phase'])
        d = f1['phase'] != f2['phase']
        delta = (f2['phase'] - f1['phase']).view(np.int64)
        out['branches'][str(br)] = dict(
            phase_differs=int(d.sum()), max_abs_phase_delta=int(np.abs(delta).max()) if d.any() else 0,
            first_x_phase_differs=int(np.flatnonzero(d)[0]) + 1 if d.any() else None,
            x_not_restored=[int(f1['bad_x'].sum()), int(f2['bad_x'].sum())],
            scratch_dirty=[int(f1['dirty'].sum()), int(f2['dirty'].sum())],
            mean_toffoli=[round(f1['toffoli'], 4), round(f2['toffoli'], 4)], n=f1['n'])
        if br == 'zero':
            try:
                tgt, tol = B.invsqrt_reference()
                if f1['n'] == len(tgt):
                    for f in (f1, f2):
                        used = np.abs((f['phase'] - tgt).view(np.int64)).view(np.uint64)
                        out['contract'].append(dict(phase_ok=bool((used <= tol).all()),
                                                    worst_ratio=round(float((used / tol).max()), 10)))
            except Exception as exc:  # the reference table is a convenience, not part of the comparison
                out['contract_error'] = repr(exc)
    out['outcome_independent'] = [int((finals['zero'][j] != finals[seed][j]).sum()) for j in (0, 1)]
    return out


def v_fdiff(api, a):
    led = E.Ledger()
    m1, L1 = _open_invsqrt(api, a.circuit)
    m2, L2 = _open_invsqrt(api, a.circuit2)
    tsha = _tool_sha(api)
    seed = a.seed
    if a.whole:
        return _v_fdiff_whole(api, a, led, m1, L1, m2, L2, tsha, seed)
    if not a.ops:
        raise api.QhdlError('fdiff needs --ops A:B (or --whole)', 'e.g. --ops 21455:21474 --map auto, or --whole')
    A1, B1 = parse_ops(a.ops)
    if a.map and a.ops2:
        raise api.QhdlError('pass --map or --ops2, not both')
    map_src = None
    if a.map:
        A2, B2, map_src = _map_ops(api, a, m1, m2, L1, L2, A1, B1)
    else:
        A2, B2 = parse_ops(a.ops2 or a.ops)
    for L, A_, B_, m in ((L1, A1, B1, m1), (L2, A2, B2, m2)):
        if B_ > len(L.ops):
            raise api.QhdlError(f"{m['sha12']}: ops {A_}:{B_} beyond its {len(L.ops)} ops")
    key = dict(tool_sha=tsha, verb='fdiff', c2=m2['sha256'], ops=[A1, B1], ops2=[A2, B2], seed=seed)
    ev = L1.dir / 'window' / f'fdiff_{A1}_{B1}_vs_{m2["sha12"]}_{A2}_{B2}_s{seed}.json'
    hit = api.cached(ev, key)
    if hit is None:
        sides = []
        pre = []
        for L, A, Bend in ((L1, A1, B1), (L2, A2, B2)):
            lv = Liveness(L.ops)
            pre.append((L, A, Bend, lv, drop_fresh(window_io(L.ops, A, Bend, int(lv.depth[A])), lv, A, L.xq)))
        wires = sorted({int(n[1:]) for p in pre for n in p[4]['inputs'] + p[4]['outputs'] if n[0] == 'q'})
        for L, A, Bend, lv, io in pre:
            ws = [w for w in wires if w < L.nq]
            per = {}
            n = 0
            for br in ('zero', seed):
                snaps = {}
                sim = B.BitSim(L.ops, num_qubits=L.nq, x_qubits=L.xq, outcome=br)
                sim.run(stop=Bend, snaps=[A, Bend],
                        on_snap=lambda i, s, ws=ws, snaps=snaps: snaps.__setitem__(i, {w: s.q[w].copy() for w in ws}))
                per[str(br)] = (snaps[A], snaps[Bend])
                n = sim.n
            try:
                fn = window_function(L.ops, L.xq, L.nq, A, Bend, liveness=lv)
            except ValueError as exc:  # too many inputs for the function analysis: values are still compared
                fn = dict(error=str(exc))
            sides.append(dict(io=io, per=per, fn=fn, n=n))
        s1, s2 = sides
        outs = sorted({n for n in s1['io']['outputs'] + s2['io']['outputs'] if n[0] == 'q'}, key=lambda n: int(n[1:]))
        ins = sorted({n for n in s1['io']['inputs'] + s2['io']['inputs'] if n[0] == 'q'}, key=lambda n: int(n[1:]))

        def ndiff(x, y):
            return int(np.bitwise_count(x ^ y).sum())

        def dd(w, br, j):
            x, y = s1['per'][br][j], s2['per'][br][j]
            return ndiff(x[w], y[w]) if (w in x and w in y) else None
        rb = str(seed)
        in_rows = []
        for nm in ins:
            w = int(nm[1:])
            d, dr = dd(w, 'zero', 0), dd(w, rb, 0)
            if d or dr:
                in_rows.append(dict(wire=nm, differs_at_start=d, differs_at_start_random=dr))
        rows = []
        for nm in outs:
            w = int(nm[1:])
            o1 = s1['fn'].get('outputs', {}).get(nm, {})
            o2 = s2['fn'].get('outputs', {}).get(nm, {})
            rows.append(dict(out=nm, differs_at_end=dd(w, 'zero', 1), differs_at_start=dd(w, 'zero', 0),
                             differs_at_end_random=dd(w, rb, 1), differs_at_start_random=dd(w, rb, 0),
                             in1=nm in s1['io']['outputs'], in2=nm in s2['io']['outputs'],
                             kind1=o1.get('kind'), kind2=o2.get('kind'), live1=o1.get('live'), live2=o2.get('live')))
        rows.sort(key=lambda r: (-((r['differs_at_end'] or 0) + (r['differs_at_end_random'] or 0)), int(r['out'][1:])))

        def summ(s):
            if 'error' in s['fn']:
                return dict(toffoli=s['io']['toffoli'], exec=None, lb=None, ub=None, kind='unavailable: '
                            + s['fn']['error'])
            f = s['fn']['floor']
            return dict(toffoli=s['io']['toffoli'], exec=round(s['fn']['executed_toffoli'], 3), lb=f['lb'], ub=f['ub'],
                        kind=f['kind'])
        hit = dict(cache_key=key, rows=rows, inputs_differing=in_rows, sides=[summ(s) for s in sides], n=s1['n'],
                   map=map_src, branches=['zero', seed], ops=f'{A1}:{B1}', ops2=f'{A2}:{B2}')
        api.write_json(ev, hit)
    led.earn('static', 'window inputs/outputs and Toffoli counts from both op streams'
             + (f'; cuts mapped by {hit["map"]}' if hit.get('map') else ''))
    led.earn('exact-all-x', f'bitset simulation of both circuits, all legal x, zero + random (seed {seed}) branch; '
                            'per-x wire values at the matched cuts')
    rows = hit['rows']
    for r in rows:
        r['tier'] = 'exact-all-x'
    differ = [r['out'] for r in rows if r['differs_at_end']]
    differ_r = [r['out'] for r in rows if r.get('differs_at_end_random')]
    s1, s2 = hit['sides']
    claim = (f"{m1['sha12']} ops {A1}..{B1 - 1} vs {m2['sha12']} ops {A2}..{B2 - 1}"
             + (' (cuts from --map)' if hit.get('map') else '') + ': '
             + (f"{len(differ)} output wire(s) differ at the end on the zero branch ({','.join(differ[:5])})" if differ
                else 'every output wire agrees on all legal x at the end on the zero branch')
             + (f", {len(differ_r)} on the random branch ({','.join(differ_r[:5])})" if differ_r
                else ' and on the random branch')
             + (f"; {len(hit['inputs_differing'])} input wire(s) already differ at the start" if hit['inputs_differing']
                else '; inputs agree at the start')
             + f"; Toffolis {s1['toffoli']} vs {s2['toffoli']}, floors {s1['lb']}..{s1['ub']} vs {s2['lb']}..{s2['ub']}")
    rr = api.cmd('fdiff', '-c', m1['sha12'], '-c2', m2['sha12'], '--ops', f'{A1}:{B1}',
                 *(['--map', a.map] if a.map else ['--ops2', f'{A2}:{B2}']),
                 *(['--seed', seed] if seed != FDIFF_SEED else []))
    return E.Envelope(led, claim, domain=f"all {api.fmt_int(api.MAX_X)} legal x, zero branch and random branch (seed "
                      f"{seed}) per-x values; floors as in `window`", evidence=api.rel(ev), rerun=rr,
                      tool_sha=tsha, circuit_sha=m1['sha256'],
                      facts=dict(ops2=f'{A2}:{B2}', toffoli=[x['toffoli'] for x in hit['sides']],
                                 floor_kind=[x['kind'][:40] for x in hit['sides']],
                                 inputs_differing=len(hit['inputs_differing']), map=hit.get('map')),
                      rows=rows,
                      next_cmds=[api.cmd('fdiff', '-c', m1['sha12'], '-c2', m2['sha12'], '--whole'),
                                 api.cmd('window', '-c', m1['sha12'], '--ops', f'{A1}:{B1}'),
                                 api.cmd('window', '-c', m2['sha12'], '--ops', f'{A2}:{B2}')])


def _v_fdiff_whole(api, a, led, m1, L1, m2, L2, tsha, seed):
    import time as _t
    key = dict(tool_sha=tsha, verb='fdiff-whole', c2=m2['sha256'], seed=seed)
    ev = L1.dir / 'window' / f'fdiff_whole_vs_{m2["sha12"]}_s{seed}.json'
    hit = api.cached(ev, key)
    if hit is None:
        t0 = _t.time()
        hit = fdiff_whole(L1, L2, seed)
        hit.update(cache_key=key, seconds=round(_t.time() - t0, 1))
        api.write_json(ev, hit)
    n = next(iter(hit['branches'].values()))['n']
    led.earn('exact-all-x', f'bitset simulations of both whole circuits on all {api.fmt_int(n)} legal x in two '
                            f'outcome branches (all HMR = 0; random, seed {seed}); per-x phase (sign + Z_POW units), '
                            'x restoration and scratch cleanliness')
    rows, ok = [], True
    for br, d in hit['branches'].items():
        good = (d['phase_differs'] == 0 and not any(d['x_not_restored']) and not any(d['scratch_dirty']))
        ok &= good
        rows.append(dict(branch='zero' if br == 'zero' else f'random(seed {br})', equivalent=good,
                         phase_differs_x=d['phase_differs'], max_abs_phase_delta=d['max_abs_phase_delta'],
                         first_x=d['first_x_phase_differs'], x_not_restored=d['x_not_restored'],
                         scratch_dirty=d['scratch_dirty'], mean_toffoli=d['mean_toffoli'], tier='exact-all-x'))
    z = hit['branches']['zero']['mean_toffoli']
    r_ = hit['branches'][str(seed)]['mean_toffoli']
    claim = (f"{m1['sha12']} vs {m2['sha12']} (whole circuits, all {api.fmt_int(n)} legal x): "
             + ('EQUIVALENT in both branches: identical phase per x, x restored, scratch clean'
                if ok else 'DIFFER: see rows (phase / x / scratch per branch)')
             + f"; executed Toffolis per x {z[0]:,.2f} vs {z[1]:,.2f} (zero branch), {r_[0]:,.2f} vs {r_[1]:,.2f} "
               f"(random branch, one draw: not the verifier mean)")
    facts = dict(outcome_dependent_phase_x=hit['outcome_independent'], seconds=hit.get('seconds'))
    if hit.get('contract'):
        facts['tolerance'] = hit['contract']
        led.earn('exact-all-x', 'zero-branch phase checked against the invsqrt target and 2^-25-scaled tolerance '
                                '(qhdl_bitsim.contract model; the official verifier stays the authority)')
    return E.Envelope(led, claim, domain=f'all {api.fmt_int(n)} legal x; zero branch and one random branch (seed '
                      f'{seed}) per circuit, draws not matched', evidence=api.rel(ev),
                      rerun=api.cmd('fdiff', '-c', m1['sha12'], '-c2', m2['sha12'], '--whole',
                                    *(['--seed', seed] if seed != FDIFF_SEED else [])),
                      tool_sha=tsha, circuit_sha=m1['sha256'], facts=facts, rows=rows,
                      next_cmds=[api.cmd('verify', '-c', m2['sha12'])])


def qhdl_register(subparsers, api):
    import argparse
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--all', action='store_true', help='print every row (default: <= 20 rows, <= 2 KB)')
    common.add_argument('-c', '--circuit', help='circuit: .kmb/.kmx path or sha prefix (>= 6 hex); invsqrt only')

    def wrap(fn):
        def run(a):
            try:
                return fn(api, a)
            except MemoryError:
                raise api.QhdlError('out of memory', 'narrow --ops or lower --max-inputs')
        return run
    p = subparsers.add_parser(
        'window', parents=[common],
        help='Boolean function (ANF, degree, AND floor) of ops A..B-1 on the reachable inputs (exact, invsqrt)',
        description='Example: qhdl window -c bce2948ff24e --ops 21454:21482 (Ehan Mallick\'s first AND merge)')
    p.add_argument('--ops', required=True, help='A:B, 0-based op indices, B exclusive')
    p.add_argument('--out', help='only these outputs, e.g. q19,q14 (default: everything written)')
    p.add_argument('--in', dest='inputs', help='express outputs over these inputs only (determination checked)')
    p.set_defaults(fn=wrap(v_window))
    p = subparsers.add_parser(
        'windows', parents=[common],
        help='rank Toffoli-led windows by slack = Toffolis - AND floor on reachable inputs (hypothesis list)')
    p.add_argument('--size', type=int, default=4, help='max Toffoli-class gates per window (default 4)')
    p.add_argument('--span', type=int, default=48, help='max ops from first to last Toffoli (default 48)')
    p.add_argument('--top', type=int, default=20, help='rows to print (default 20)')
    p.add_argument('--max-inputs', type=int, default=20, help='skip windows reading more inputs (default 20)')
    p.set_defaults(fn=wrap(v_windows))
    p = subparsers.add_parser(
        'fdiff', parents=[common], help='compare matched windows (or whole circuits) of two invsqrt circuits per x',
        description='Examples: qhdl fdiff -c 67482db6c538 -c2 cand.kmb --ops 21455:21474 --map auto | '
                    'qhdl fdiff -c 67482db6c538 -c2 cand.kmb --whole')
    p.add_argument('-c2', '--circuit2', required=True, help='second circuit')
    p.add_argument('--ops', help='A:B in the first circuit (required unless --whole)')
    p.add_argument('--ops2', help='A:B in the second circuit (default: same as --ops, or mapped by --map)')
    p.add_argument('--map', help="op-index map: the .map.json `qhdl edit` wrote (-c = its input, -c2 = its output), "
                                 "or 'auto' to align the two op streams; derives --ops2 from --ops")
    p.add_argument('--whole', action='store_true', help='compare the whole circuits: per-x phase, x restored, '
                                                        'scratch clean (no input limit)')
    p.add_argument('--seed', type=int, default=FDIFF_SEED, help=f'random-branch outcome seed (default {FDIFF_SEED})')
    p.set_defaults(fn=wrap(v_fdiff))
