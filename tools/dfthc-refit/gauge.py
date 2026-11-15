"""Word conventions of a dfthcsos family: detector, signed-gauge working form and a gauge-aware loader emitter
(dfthc_refit 1.5.0-dev; in-place gauge
change 1.6.0-dev).

The official decoder reads word w of row r, column i as theta / pi = w / 2**(brot - 1) on the family's Givens tree
applied to e_0; only u u^T enters the score. Lera's emitter (and every planner in this tool) assumes the **signed
gauge**: the top bit of every word copies the bit below (bit 7 = bit 6 at brot 8). Two other conventions are in use:

- **pi gauge** (Atakan Aldemir's tree and loaders, e.g. `aba448aeadb5`): some words carry bit 7 = bit 6 XOR 1, i.e.
  theta + pi. Flipping those top bits back (the canonical signed form) leaves every row's u u^T unchanged: an exact
  symmetry. The per-entry flip p(row, column) takes a few row patterns.
- **free range**: off-gauge words whose canonical form changes u u^T: not a symmetry; the off-gauge words are data.

What is here:

- `word_conventions(words, tree_edges, brot)`: storage range, tree kind, gauge class with its pattern group, and
  whether the pattern is a function of the low address bits (what the gauge-aware emitter needs).
- `canonical_words` / `to_family_gauge`: the signed-gauge working form of a pi-gauge table and back.
- `loader_classes(km, loaders, words)`: per slot, XOR-in (out = in XOR f(address) on every input the verifier can
  feed, and on random words of the incoming gauge) or chained (correct only on the table's own words).
- `build_circuit_gauge(tq, delta, terms, ...)`: Lera's cached-product emitter (`tensorqrom.build_circuit`, Lera /
  marculera, Apache-2.0; it credits mrmeows' cached-product emitter and Heliosli's signed scratch) with the borrowed
  top bit made gauge-aware. The planning model, the products, the corrections and the Toffoli count are Lera's; the
  differences are: the low-monomial cache is built first (on fresh qubits), the incoming gauge p_in(low bits) is
  cleared from each borrowed top bit through that cache before the bit is lent to the high cache, and p_out is put
  back at the end; a word whose gauge changes across the slot (p_in != p_out) keeps its top bit as an output that
  also receives p_in XOR p_out. `GaugeEmitter` wraps Lera's module and calls their `build_circuit` unchanged whenever
  the gauge of the slot is the signed one, so signed-gauge families emit byte for byte as before.
"""
from __future__ import annotations

import hashlib

import numpy as np

GAUGE_TOL = 1e-9          # max |u u^T - u_c u_c^T| for the canonical form to count as an exact symmetry
LOWBITS = 4


# ------------------------------------------------------------------------------------------------------- decoder


def decode(words, edges, brot):
    """u vectors (rows, N) of an angle-word table, as the official decoder: theta / pi = word / 2**(brot - 1) on the
    Givens tree `edges` (parent, child) in order, applied to e_0."""
    a = np.asarray(words, np.float64) / float(1 << (int(brot) - 1))
    v = np.zeros((a.shape[0], a.shape[1] + 1))
    v[:, 0] = 1.0
    for i, (p, c) in enumerate(np.asarray(edges, np.int64)):
        pv, cv = v[:, p].copy(), v[:, c].copy()
        t = np.pi * a[:, i]
        v[:, p] = np.cos(t) * pv - np.sin(t) * cv
        v[:, c] = np.sin(t) * pv + np.cos(t) * cv
    return v


def tree_kind(edges) -> str:
    E = np.asarray(edges, np.int64)
    if all(int(p) == i and int(c) == i + 1 for i, (p, c) in enumerate(E)):
        return "path"
    if all(int(p) == 0 for p, _ in E):
        return "star"
    if all((d := int(c) - int(p)) > int(p) and d & (d - 1) == 0 for p, c in E):
        return "binomial"
    return "tree:" + hashlib.sha256(E.tobytes()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------- gauge classes


def off_gauge(words, brot=8):
    """(rows, cols) uint8: 1 where the top bit differs from the bit below (the word is off the signed gauge)."""
    w = np.asarray(words, np.int64)
    return (((w >> (brot - 1)) ^ (w >> (brot - 2))) & 1).astype(np.uint8)


def canonical_words(words, brot=8):
    """The signed-gauge form: every off-gauge word moved by pi (top bit flipped), kept in the table's storage range
    (unsigned [0, 2**brot) by XOR; signed [-2**(brot-1), 2**(brot-1)) by -/+ 2**(brot-1))."""
    w = np.asarray(words, np.int64)
    off = off_gauge(w, brot).astype(bool)
    half = 1 << (brot - 1)
    if w.min() < 0:
        c = np.where(w >= 0, w - half, w + half)
    else:
        c = w ^ half
    return np.where(off, c, w)


def to_family_gauge(words_c, pattern, brot=8):
    """Re-apply a family's gauge pattern (rows, cols of 0/1) to signed-gauge words (unsigned storage)."""
    w = np.asarray(words_c, np.int64)
    return np.where(np.asarray(pattern, bool), w ^ (1 << (brot - 1)), w)


def low_pattern(pattern, lowbits=LOWBITS):
    """(2**lowbits, cols) table q with pattern[r, c] = q[r mod 2**lowbits, c] on every row, or None if the pattern is
    not a function of the low address bits alone."""
    P = np.asarray(pattern, np.uint8)
    L = 1 << lowbits
    low = np.arange(P.shape[0]) % L
    q = np.zeros((L, P.shape[1]), np.uint8)
    for v in range(L):
        rows = P[low == v]
        if len(rows) == 0:
            continue
        if np.any(rows != rows[0]):
            return None
        q[v] = rows[0]
    return q


def word_conventions(words, tree_edges, brot=8, *, lowbits=LOWBITS) -> dict:
    """Storage range, tree, gauge class (signed / pi / free) of a table. JSON-ready."""
    w = np.asarray(words, np.int64)
    off = off_gauge(w, brot)
    rows_off = off.any(axis=1)
    pats = {}
    for r in np.flatnonzero(rows_off):
        pats.setdefault(tuple(int(c) for c in np.flatnonzero(off[r])), []).append(int(r))
    rep = dict(shape=list(w.shape), brot=int(brot), storage=[int(w.min()), int(w.max())],
               storage_kind="signed" if w.min() < 0 else "unsigned",
               tree=tree_kind(tree_edges) if tree_edges is not None else "unknown (no tree_edges)",
               off_gauge_entries=int(off.sum()), off_gauge_share=float(off.mean()), rows_off_gauge=int(rows_off.sum()),
               patterns=len(pats))
    if not off.any():
        rep.update(gauge="signed", exact_symmetry=None, pattern_low_bits=True)
        return rep
    if tree_edges is None:   # no tree: exactness cannot be tested
        rep.update(gauge="unknown", exact_symmetry=None, pattern_low_bits=low_pattern(off, lowbits) is not None)
        return rep
    u = decode(w, tree_edges, brot)
    uc = decode(canonical_words(w, brot), tree_edges, brot)
    dproj = np.abs(u[:, :, None] * u[:, None, :] - uc[:, :, None] * uc[:, None, :]).max(axis=(1, 2))
    exact = bool(dproj.max() < GAUGE_TOL)
    q = low_pattern(off, lowbits)
    rep.update(gauge="pi" if exact else "free", exact_symmetry=exact, max_projector_change=float(dproj.max()),
               rows_projector_changed=int((dproj >= GAUGE_TOL).sum()), pattern_low_bits=q is not None)
    if len(pats) <= 64:
        low = np.arange(w.shape[0]) % (1 << lowbits)
        rep["pattern_group"] = [dict(columns=list(k), rows=len(v), low_addresses=sorted({int(low[r]) for r in v}))
                                for k, v in sorted(pats.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    return rep


# --------------------------------------------------------------------------------------------- loader classes


def batches(words, j=5):
    """(slots, rows, j): the words padded with zero columns to whole batches."""
    w = np.asarray(words, np.int64)
    A, n = w.shape
    s = -(-n // j)
    pad = np.zeros((A, s * j), np.int64)
    pad[:, :n] = w
    return pad.reshape(A, s, j).transpose(1, 0, 2)


def _run(km, circ, inputs, n_sel, brot, seed=0):
    """Run `circ` once on every row of `inputs` (rows, J) at its own address; return (out, phase, dirty)."""
    count, J = inputs.shape
    sim = km.Simulator(batch_size=count, seed=seed, counting_operations=True)
    sim.use_same_registers_as(circ)
    a = np.arange(count)
    sim.write_across_shots("selection", ((a[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for jj in range(J):
        sim.write_across_shots(f"word_{jj}", ((inputs[:, jj][None, :] >> np.arange(brot)[:, None]) & 1).astype(bool))
    sim.do(circ)
    p = np.int64(1) << np.arange(brot)
    out = np.stack([p @ sim.read_across_shots(f"word_{jj}").astype(np.int64) for jj in range(J)], axis=1)
    phase = np.asarray(sim.read_phase_flipped_across_shots()).astype(bool)
    total = sim.read_across_shots(km.array([km.q(i) for i in range(circ.num_qubits)]))
    dirty = np.asarray(total[n_sel + brot * J:]).any(axis=0) if total.shape[0] > n_sel + brot * J else \
        np.zeros(count, bool)
    return out, phase, dirty


def gauge_random(rng, pattern_in, brot=8):
    """Random words whose top bit is bit brot-2 XOR the given incoming gauge pattern (rows, J)."""
    pin = np.asarray(pattern_in, np.int64)
    r = rng.integers(0, 1 << (brot - 1), size=pin.shape)
    return r | ((((r >> (brot - 2)) & 1) ^ pin) << (brot - 1))


def loader_classes(km, loaders, words, *, brot=8, j=5, seed=4242, slots=None) -> list:
    """Per slot k: 'xor-in' when the loader maps every input x to x XOR (b_k XOR b_(k-1)) on the inputs the verifier
    feeds (b_(k-1) forward, b_k in reverse) and on random words of the incoming gauge, all cleanly (no phase, scratch
    clean); 'chained' when it maps b_(k-1) <-> b_k cleanly but is not XOR-in; 'broken' otherwise.
    `needs_incoming_gauge`: the incoming gauge is not the signed one and the loader is not an XOR oracle on random
    signed-gauge words (a loader with a gauge-aware borrowed top bit). Zero is such an off-gauge input for k >= 1, so
    a zero-input test alone mistakes these loaders for input-dependent ones (as an earlier analysis did)."""
    B = batches(words, j)
    A = B.shape[1]
    n_sel = (A - 1).bit_length()
    P = off_gauge(B.transpose(1, 0, 2).reshape(A, -1), brot).reshape(A, -1, j).transpose(1, 0, 2)
    rng = np.random.default_rng(seed)
    out = []
    for k, path in enumerate(loaders):
        if slots is not None and k not in slots:   # v1.6: classify only these slots (the random-gauge inputs then
            continue                               # differ from a full run's, which draws them slot after slot)
        path = str(path)
        circ = km.Circuit.from_file(path, format="kmb" if path.endswith(".kmb") else "kmx")
        prev = B[k - 1] if k else np.zeros_like(B[0])
        pin = P[k - 1] if k else np.zeros_like(P[0])
        delta = B[k] ^ prev
        rnd = gauge_random(rng, pin, brot)
        rec = dict(k=k, file=path.rsplit("/", 1)[-1], qubits=int(circ.num_qubits),
                   toffolis=int(circ.operation_statistics().num_toffoli_like))
        res = {}
        runs = [("forward", prev, B[k]), ("reverse", B[k], prev), ("random_gauge", rnd, rnd ^ delta)]
        if pin.any():   # diagnostic: random words in the signed gauge (off the incoming gauge on some rows)
            rs = gauge_random(rng, np.zeros_like(pin), brot)
            runs.append(("random_signed", rs, rs ^ delta))
        for name, x, want in runs:
            o, ph, dt = _run(km, circ, x, n_sel, brot)
            res[name] = dict(correct=bool(np.array_equal(o, want)), rows_wrong=int((o != want).any(axis=1).sum()),
                             phase=int(ph.sum()), dirty=int(dt.sum()))

        def ok(n):
            return res[n]["correct"] and not res[n]["phase"] and not res[n]["dirty"]
        chain_ok = ok("forward") and ok("reverse")
        rec.update(runs=res, gauge_in_off=int(pin.sum()), gauge_change=int((pin ^ P[k]).sum()),
                   loader="xor-in" if chain_ok and ok("random_gauge") else ("chained" if chain_ok else "broken"),
                   needs_incoming_gauge=bool(pin.any() and not ok("random_signed")))
        out.append(rec)
    return out


def conventions(km, loaders, words, tree_edges, brot=8, j=5) -> dict:
    """The full detector: word conventions plus per-slot loader classes (when a kickmix module is given)."""
    rep = word_conventions(words, tree_edges, brot)
    if km is not None and loaders:
        rep["slots"] = loader_classes(km, loaders, words, brot=brot, j=j)
        rep["loader_summary"] = {c: [s["k"] for s in rep["slots"] if s["loader"] == c]
                                 for c in ("xor-in", "chained", "broken")}
    return rep


# Which subcommand handles which convention. A refusal names the reason; anything else runs.
SUPPORT = {
    "refit": "every gauge (the metrics read the words as the decoder does)",
    "recover": "XOR-in cached-product loaders; chained slots are listed and skipped",
    "plan": "signed and pi gauge (plans use bits 0..brot-2 of the XOR delta); free range refused",
    "replan": "signed gauge (Lera's emitter); pi gauge with a low-address pattern (gauge-aware emitter); "
              "free range refused",
    "price-words": "signed gauge; pi gauge on the canonical signed form (exact symmetry); free range: shadow prices "
                   "only",
}


def refusal(command, conv, *, moves=False) -> str | None:
    """None if `command` supports the family's conventions, else a one-line reason."""
    g = conv["gauge"]
    if command == "replan" and g == "unknown":
        return "replan: off-gauge words and no tree_edges to test whether the gauge is an exact symmetry"
    if command == "price-words" and moves and g == "unknown":
        return "price-words: off-gauge words and no tree_edges; word moves refused"
    if command in ("plan", "replan") and g == "free":
        return (f"{command}: the table is free range ({conv['off_gauge_entries']} words off the signed gauge, not an "
                f"exact symmetry): the loaders' top bits are data, which neither the planners' tensor nor the emitter "
                f"model")
    if command == "replan" and g == "pi" and not conv["pattern_low_bits"]:
        return ("replan: pi-gauge pattern is not a function of the low address bits; the gauge-aware emitter clears "
                "the borrowed top bit through the low-monomial cache only")
    if command == "price-words" and moves and g == "free":
        return ("price-words: word moves on a free-range table are not supported (the signed-gauge check would "
                "reject the table itself); shadow prices only")
    return None


# ------------------------------------------------------------------------------------------- gauge-aware emitter


def slot_gauge(words, k, j=5, brot=8, lowbits=LOWBITS):
    """(p_in, p_out): (2**lowbits, j) incoming / outgoing gauge of slot k as functions of the low address bits, or
    None when either is not such a function."""
    B = batches(words, j)
    A = B.shape[1]
    P = off_gauge(B.transpose(1, 0, 2).reshape(A, -1), brot).reshape(A, -1, j).transpose(1, 0, 2)
    pin = low_pattern(P[k - 1], lowbits) if k else np.zeros((1 << lowbits, j), np.uint8)
    pout = low_pattern(P[k], lowbits)
    if pin is None or pout is None:
        return None
    return pin, pout


def gf2_basis_columns(g):
    """Greedy GF(2) basis of the nonzero columns of g (rows = low addresses, columns = words), in column order.

    Returns (keepers, combos): `keepers` are the word indices whose columns form the basis; `combos[j]` (for every
    other nonzero column j) is the list of keepers whose columns XOR to column j. len(keepers) = rank of g."""
    g = np.asarray(g, np.uint8)
    basis = []                    # (pivot bit, reduced int, set of keepers it is the XOR of)
    keepers, combos = [], {}
    for jj in range(g.shape[1]):
        v = int(sum(1 << i for i, x in enumerate(g[:, jj]) if x))
        if not v:
            continue
        used = set()
        for piv, bv, comb in basis:
            if v >> piv & 1:
                v ^= bv
                used ^= comb
        if v:
            basis.append((v.bit_length() - 1, v, used ^ {jj}))
            basis.sort(key=lambda t: -t[0])
            keepers.append(jj)
        else:
            combos[jj] = sorted(used)
    return keepers, combos


def gauge_width(n_rows, J=5, brot=8, lowbits=LOWBITS, kept=0) -> int:
    """Qubits of the cached-product emitter with the full low cache and borrowed top bits (Lera's layout): selection
    + words + low cache (2**lowbits - 1 - lowbits) + high cache (H - 1 - bits(H - 1)) - the J - kept lent top bits.
    1,220 rows: 11 + 40 + 11 + 69 - 5 = 126 at kept = 0. Static model; the emitted circuit is the authority."""
    n_sel = max(1, (int(n_rows) - 1).bit_length())
    L = 1 << lowbits
    H = -(-int(n_rows) // L)
    high = (H - 1) - max(1, (H - 1).bit_length())
    return n_sel + brot * J + (L - 1 - lowbits) + max(0, high - (J - kept))


def free_monomials(p_in, p_out, lowbits=LOWBITS) -> dict:
    """Free low sources (v1.8.0-dev): {word: m} for every word whose gauge does not change in this slot
    (p_in == p_out on every low address) and whose p_in has exactly one nonlinear term m in its ANF over the low
    monomials. After `bit 7 ^= bit 6` and clearing the affine part, that word's top bit holds the monomial m on both
    passes (the verifier feeds p_in forward and p_out = p_in in reverse), so it can serve as the low source X[m]
    instead of a cached AND. One word per monomial (the lowest-numbered); words with other patterns are not used."""
    L = 1 << lowbits
    p_in = np.asarray(p_in, np.uint8).reshape(L, -1)
    p_out = np.asarray(p_out, np.uint8).reshape(L, -1)
    anf = p_in.copy()
    bit = 1
    while bit < L:
        for a in range(L):
            if a & bit:
                anf[a] ^= anf[a ^ bit]
        bit <<= 1
    out, seen = {}, set()
    for jj in range(p_in.shape[1]):
        if (p_in[:, jj] != p_out[:, jj]).any():
            continue
        nl = [m for m in range(L) if anf[m, jj] and m & (m - 1)]
        if len(nl) == 1 and nl[0] not in seen:
            out[jj] = nl[0]
            seen.add(nl[0])
    return out


def low_cache_floor(p_in, p_out, lowbits=LOWBITS) -> int:
    """Low-cache ANDs the free-source emitter spends: every nonlinear low monomial minus the free ones. It is also a
    lower bound for any cached-product loader whose mixed ANF rows span the whole low space."""
    L = 1 << lowbits
    return (L - 1 - lowbits) - len(free_monomials(p_in, p_out, lowbits))


def build_circuit_gauge(tq, delta, terms, n_sel, brot, lowbits, p_in, p_out, inplace=False, free_sources=False):
    """The XOR oracle for the table `delta` from the decomposition `terms` (Lera's planning model, full low cache),
    for incoming words with top bit = bit brot-2 XOR p_in(low address bits) and outgoing words with p_out.

    Words with p_in == p_out lend their top bit to the high cache: one CNOT from bit brot-2 leaves p_in(low), which
    is cleared through the low-monomial cache (built first, on fresh qubits), and restored the same way at the end.
    Words whose gauge changes keep the top bit as an output: it receives bit brot-2's delta and p_in XOR p_out.
    Valid for incoming words of either gauge pattern in the second case and of p_in in the first, which is what the
    official verifier feeds (zero for slot 0, the previous batch forward, this batch in reverse). Bit brot-1 of
    `delta` is not read: bits 0..brot-2 define the tensor (as in Lera's anf_tensor).

    `inplace=True` (v1.6) makes the gauge change in place instead of keeping one top bit per changing word.
    After `bit 7 ^= bit 6` the top bit of word j holds p_in_j(l) XOR dir * g_j(l), where dir = 0 on the forward pass
    and 1 on the reverse pass: a bit of direction information on the low addresses where g_j = 1. If g_j is the XOR
    of the columns of some other changing words (the keepers, a GF(2) basis of g's columns), XORing their top bits
    into word j's cancels the dir part and leaves a function of the low address bits, which the low cache clears:
    word j's top bit is lent like an unchanged word's. Only the keepers (rank of g, not the number of changing words)
    stay unlent; they are not outputs during the products. At the end the dependents are rebuilt from the keepers,
    every changing word receives g(l) and every word receives its new bit 6. Toffolis are unchanged (CNOTs only).
    The same construction is in Dmitrii Gluzdov's loaders (family 726299a7f65b, slots 11-13: CX ladders among the
    top bits at both ends), which is where this tool took it from. With CNOTs and address functions only, rank(g)
    unlent top bits is also the minimum.

    `free_sources=True` (v1.8.0-dev; not with `inplace`) keeps the top bit of every word in
    `free_monomials(p_in, p_out)` holding its gauge monomial m and uses it as the low source X[m]: m is not cached,
    so the low cache costs `low_cache_floor(p_in, p_out)` ANDs instead of 11. This is the construction in Syed Asad
    Ali's loaders (family da4d23796b13). The low cache is erased before those top bits are restored. Off by default,
    so earlier emissions are unchanged byte for byte."""
    import kickmix as km
    delta = np.asarray(delta, dtype=np.int64)
    count, J = delta.shape
    L = 1 << lowbits
    p_in = np.asarray(p_in, np.uint8).reshape(L, J)
    p_out = np.asarray(p_out, np.uint8).reshape(L, J)
    g = p_in ^ p_out
    lend = [jj for jj in range(J) if not g[:, jj].any()]
    keep = [jj for jj in range(J) if g[:, jj].any()]
    combos = {}
    if inplace:                               # keepers: a GF(2) basis of g's columns; dependents are lent
        keep, combos = gf2_basis_columns(g)
        lend = [jj for jj in range(J) if jj not in keep]
    free = free_monomials(p_in, p_out, lowbits) if (free_sources and not inplace) else {}
    assert all(jj in lend for jj in free)
    C = tq.anf_tensor(delta, brot, lowbits)
    L_, H, O = C.shape
    assert L_ == L
    assert np.array_equal(tq.mixed_of(terms, L - 1, H - 1, O), C[1:, 1:]), 'decomposition does not cover the tensor'

    # pure parts, exactly as Lera's build_circuit
    low = C[:, 0].copy()
    high = C[0].copy()
    high[0] = 0
    vl, vh = tq._monomial_values(L, L), tq._monomial_values(H, H)
    hb = max(1, (H - 1).bit_length())
    for a1, b1, a2, b2, s in terms:
        outs = np.array(tq._bits(s))
        if a2:
            pl = tq.mobius((vl @ np.array(tq._bits_vec(a1, L - 1)) & 1) & (vl @ np.array(tq._bits_vec(a2, L - 1)) & 1))
            low[:, outs] ^= pl[:, None]
        if b2:
            th = np.zeros(1 << hb, np.uint8)
            th[:H] = (vh @ np.array(tq._bits_vec(b2, H - 1)) & 1) & (vh @ np.array(tq._bits_vec(b1, H - 1)) & 1)
            ph = tq.mobius(th)[:H]
            high[:, outs] ^= ph[:, None]
    low[0] ^= high[0]
    high[0] = 0
    anf_in = tq.mobius(p_in, 0)               # ANF of p_in over the low monomials, per word
    anf_g = tq.mobius(g, 0)
    q_dep = p_in.copy()                       # in place: what a dependent's top bit holds after the keeper XORs
    for jj, comb in combos.items():
        for ii in comb:
            q_dep[:, jj] ^= p_in[:, ii]
    anf_clear = tq.mobius(q_dep, 0)

    b = km.CircuitBuilder()
    sel = b.create_quantum_register(n_sel, 'selection')
    words = [b.create_quantum_register(brot, f'word_{jj}') for jj in range(J)]
    outputs = [[words[jj][k]] + ([words[jj][brot - 1]] if (k == brot - 2 and jj in keep and not inplace) else [])
               for jj in range(J) for k in range(brot - 1)]
    X = {m: sel[m.bit_length() - 1] for m in range(1, L) if m & (m - 1) == 0}

    def xor_low(coeffs, t, *, affine=True, nonlinear=True) -> None:
        for m in range(L):
            if not coeffs[m]:
                continue
            lin = m == 0 or m & (m - 1) == 0
            if lin and affine:
                b.x(t) if m == 0 else b.cx(X[m], t)
            elif not lin and nonlinear:
                b.cx(X[m], t)

    # 1. lent top bits: bit 7 ^= bit 6 leaves p_in(low); clear its affine part
    if inplace:   # every top bit: bit 7 ^= bit 6; dependents ^= their keepers (the dir part cancels)
        for jj in range(J):
            b.cx(words[jj][brot - 2], words[jj][brot - 1])
        for jj, comb in combos.items():
            for ii in comb:
                b.cx(words[ii][brot - 1], words[jj][brot - 1])
        for jj in lend:
            xor_low(anf_clear[:, jj], words[jj][brot - 1], nonlinear=False)
    for jj in ([] if inplace else lend):
        b.cx(words[jj][brot - 2], words[jj][brot - 1])
        xor_low(anf_in[:, jj], words[jj][brot - 1], nonlinear=False)
    for jj, m in free.items():             # v1.8: this top bit now holds the monomial m: a free low source
        X[m] = words[jj][brot - 1]
    # 2. low-monomial cache on fresh qubits (alive for the whole circuit)
    made_x = []
    for mask in range(1, L):
        if mask & (mask - 1) == 0 or mask in X:
            continue
        top = mask.bit_length() - 1
        t = b.alloc_qubits(1)[0]
        b.init_and(X[mask ^ (1 << top)], sel[top], t)
        X[mask] = t
        made_x.append((X[mask ^ (1 << top)], sel[top], t))
    # 3. clear the nonlinear part of p_in: the lent top bits are now |0> (free sources keep their monomial)
    for jj in lend:
        if jj not in free:
            xor_low((anf_clear if inplace else anf_in)[:, jj], words[jj][brot - 1], affine=False)
    pool = [words[jj][brot - 1] for jj in lend if jj not in free]
    lent: list = []

    def take():
        if pool:
            lent.append(pool[-1])
            return pool.pop()
        return b.alloc_qubits(1)[0]

    def give(t) -> None:
        if any(t is q for q in lent):
            lent[:] = [q for q in lent if q is not t]
            pool.append(t)
        else:
            b.free(t)

    # 4. high-monomial cache (Lera's cache(H, lowbits, made))
    Y, made_y = {}, []
    for mask in range(1, H):
        top = mask.bit_length() - 1
        if mask == 1 << top:
            Y[mask] = sel[lowbits + top]
        else:
            t = take()
            b.init_and(Y[mask ^ (1 << top)], sel[lowbits + top], t)
            Y[mask] = t
            made_y.append((Y[mask ^ (1 << top)], sel[lowbits + top], t))

    def qubit(q: int):
        return X[q + 1] if q < L - 1 else Y[q - L + 2]

    def targets(s) -> list:
        return [q for o in s for q in outputs[o]]

    def fan_ccx(c1, c2, ts: list) -> None:
        first, rest = ts[0], ts[1:]
        for t in rest:
            b.cx(first, t)
        b.ccx(c1, c2, first)
        for t in reversed(rest):
            b.cx(first, t)

    def product(a1, b1, a2, b2, s) -> None:
        m1, m2 = a1 | (b2 << (L - 1)), a2 | (b1 << (L - 1))
        if not m1 or not m2 or m1 == m2:
            return
        p1 = tq._bits(m1)[0]
        ops = [(q, p1) for q in tq._bits(m1) if q != p1]
        if m2 >> p1 & 1:
            m2 ^= m1 ^ (1 << p1)
        p2 = next(q for q in tq._bits(m2) if q != p1)
        ops += [(q, p2) for q in tq._bits(m2) if q != p2]
        for src, dst in ops:
            b.cx(qubit(src), qubit(dst))
        fan_ccx(qubit(p1), qubit(p2), targets(tq._bits(s)))
        for src, dst in reversed(ops):
            b.cx(qubit(src), qubit(dst))

    # 5. products and pure corrections (Lera's order for the full low cache)
    for t in terms:
        product(*t)
    for m in range(L):
        ts = targets(np.flatnonzero(low[m]))
        if ts:
            if m == 0:
                b.x(km.array(ts))
            else:
                b.cx(X[m], km.array(ts))
    for m in range(1, H):
        ts = targets(np.flatnonzero(high[m]))
        if ts:
            b.cx(Y[m], km.array(ts))
    # 6. kept top bits: the gauge change p_in XOR p_out
    for jj in ([] if inplace else keep):
        xor_low(anf_g[:, jj], words[jj][brot - 1])
    # 7. erase the high cache (the lent top bits return to |0>)
    for c1, c2, t in reversed(made_y):
        b.del_and(c1, c2, t)
        give(t)
    assert not lent and len(pool) == len(lend) - len(free)
    # 8. restore the lent top bits: bit 6 of the new word, then p_out(low) (= p_in(low))
    if inplace:   # dependents back to p_in ^ dir g; every changing word ^= g(low); every word ^= its new bit 6
        for jj, comb in combos.items():
            xor_low(anf_clear[:, jj], words[jj][brot - 1])
            for ii in comb:
                b.cx(words[ii][brot - 1], words[jj][brot - 1])
        for jj in range(J):
            if g[:, jj].any():
                xor_low(anf_g[:, jj], words[jj][brot - 1])
        for jj in range(J):
            b.cx(words[jj][brot - 2], words[jj][brot - 1])
        for jj in lend:
            if jj not in combos:
                xor_low(anf_in[:, jj], words[jj][brot - 1])
    for jj in ([] if inplace else lend):
        if jj not in free:
            b.cx(words[jj][brot - 2], words[jj][brot - 1])
            xor_low(anf_in[:, jj], words[jj][brot - 1])
    # 9. erase the low cache
    for c1, c2, t in reversed(made_x):
        b.del_and(c1, c2, t)
        b.free(t)
    # 10. free sources (v1.8): the top bit holds m; add p_in's affine part and the new bit 6
    for jj in free:
        xor_low(anf_in[:, jj], words[jj][brot - 1], nonlinear=False)
        b.cx(words[jj][brot - 2], words[jj][brot - 1])
    return b.finish_circuit()


QUBIT_CAP = 128   # per ANGLE_k circuit (dfthcsos.verify.QUBIT_CAP)
INPLACE_POLICIES = ("auto", "never", "always")


class GaugeEmitter:
    """Lera's emitter module with a gauge for one slot. `build_circuit` calls Lera's unmodified function when the slot
    is in the signed gauge (p_in = p_out = 0), else `build_circuit_gauge` (full low cache, borrowed top bits only).
    Every other attribute is Lera's.

    `inplace` (v1.6): "never" keeps one top bit per gauge-changing word (the 1.5 layout); "always" changes the gauge
    in place (rank(g) unlent top bits); "auto" (default) emits the 1.5 layout and switches to the in-place one only
    when the 1.5 circuit exceeds `cap` qubits and the in-place one is narrower, so every loader that fitted before
    is emitted byte for byte as before. Toffolis are the same in both layouts."""

    def __init__(self, tq, p_in, p_out, inplace="auto", cap=QUBIT_CAP, free_sources=False):
        if inplace not in INPLACE_POLICIES:
            raise ValueError(f"inplace must be one of {INPLACE_POLICIES}")
        self._tq = tq
        self.p_in = np.asarray(p_in, np.uint8)
        self.p_out = np.asarray(p_out, np.uint8)
        self.signed = not (self.p_in.any() or self.p_out.any())
        self.inplace = inplace
        self.cap = int(cap)
        self.free_sources = bool(free_sources)   # v1.8: gauge monomials as free low sources
        self.layout = None                 # set by build_circuit: "lera", "kept" or "inplace"
        self.widths = {}

    def __getattr__(self, name):
        return getattr(self._tq, name)

    def build_circuit(self, delta, terms, n_sel, brot, lowbits, groups=None, scratch=False):
        if self.signed:
            return self._tq.build_circuit(delta, terms, n_sel, brot, lowbits, groups=groups, scratch=scratch)
        if groups is not None or not scratch:
            raise ValueError("the gauge-aware emitter supports the full low cache with borrowed top bits only "
                             "(groups=None, scratch=True)")
        changing = bool((self.p_in ^ self.p_out).any())
        if self.inplace == "always" and changing:
            c = build_circuit_gauge(self._tq, delta, terms, n_sel, brot, lowbits, self.p_in, self.p_out, inplace=True)
            self.layout, self.widths = "inplace", dict(inplace=int(c.num_qubits))
            return c
        c = build_circuit_gauge(self._tq, delta, terms, n_sel, brot, lowbits, self.p_in, self.p_out,
                                free_sources=self.free_sources)
        self.layout, self.widths = "kept", dict(kept=int(c.num_qubits))
        if self.inplace == "auto" and changing and c.num_qubits > self.cap:
            c2 = build_circuit_gauge(self._tq, delta, terms, n_sel, brot, lowbits, self.p_in, self.p_out, inplace=True)
            self.widths["inplace"] = int(c2.num_qubits)
            if c2.num_qubits < c.num_qubits:
                self.layout = "inplace"
                return c2
        return c

    def record(self) -> dict:
        g = self.p_in ^ self.p_out
        keepers, combos = gf2_basis_columns(g)
        inpl = self.layout == "inplace"
        return dict(module="gauge.build_circuit_gauge (this tool; after Lera's tensorqrom.build_circuit)"
                    if not self.signed else "tensorqrom.build_circuit (Lera, unmodified)",
                    signed=self.signed, inplace_policy=self.inplace, layout=self.layout, widths=self.widths,
                    changing_words=[int(jj) for jj in range(g.shape[1]) if g[:, jj].any()],
                    gauge_change_rank=len(keepers), keepers=[int(x) for x in keepers],
                    dependents={int(k): [int(x) for x in v] for k, v in combos.items()},
                    lent_top_bits=[int(jj) for jj in range(g.shape[1])
                                   if not g[:, jj].any() or (inpl and jj in combos)],
                    free_sources=self.free_sources,
                    free_monomials=({int(k): int(v) for k, v in free_monomials(self.p_in, self.p_out).items()}
                                    if self.free_sources and self.layout != "inplace" else {}),
                    p_in=["".join(map(str, self.p_in[:, jj])) for jj in range(self.p_in.shape[1])],
                    p_out=["".join(map(str, self.p_out[:, jj])) for jj in range(self.p_out.shape[1])])


def check_inputs(words, k, j=5, brot=8, seed=4242):
    """Native-check inputs for slot k of a gauged family: the incoming batch b_(k-1), the outgoing batch b_k (reverse
    pass) and random words of the incoming gauge, each on every real row."""
    B = batches(words, j)
    A = B.shape[1]
    P = off_gauge(B.transpose(1, 0, 2).reshape(A, -1), brot).reshape(A, -1, j).transpose(1, 0, 2)
    prev = B[k - 1] if k else np.zeros_like(B[0])
    pin = P[k - 1] if k else np.zeros_like(P[0])
    rnd = gauge_random(np.random.default_rng(seed), pin, brot)
    return np.concatenate([prev, B[k], rnd]).astype(np.int64)


def native_check(km, circuit, delta, inputs, brot=8, seed=4242) -> dict:
    """dfthc_refit.native_check with explicit inputs (rows a multiple of the table's): two applications on every
    row at its address; the first must XOR delta in, the second restore the input; no phase, selection unchanged,
    scratch clean."""
    delta = np.asarray(delta, np.int64)
    count, J = delta.shape
    n_sel = (count - 1).bit_length()
    initial = np.asarray(inputs, np.int64)
    assert len(initial) % count == 0
    addresses = np.tile(np.arange(count), len(initial) // count)
    sim = km.Simulator(batch_size=len(addresses), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots("selection", ((addresses[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for jj in range(J):
        sim.write_across_shots(f"word_{jj}", ((initial[:, jj][None, :] >> np.arange(brot)[:, None]) & 1).astype(bool))
    powers = np.int64(1) << np.arange(brot)
    toffolis = []
    try:
        for application in range(2):
            before = sim.operation_statistics().num_toffoli_like
            sim.do(circuit)
            toffolis.append((sim.operation_statistics().num_toffoli_like - before) / len(addresses))
            got = np.stack([powers @ sim.read_across_shots(f"word_{jj}").astype(np.int64) for jj in range(J)], axis=1)
            expected = initial ^ delta[addresses] if application == 0 else initial
            assert np.array_equal(got, expected), "incorrect XOR"
            assert not np.any(sim.read_phase_flipped_across_shots()), "phase"
            decoded = (np.int64(1) << np.arange(n_sel)) @ sim.read_across_shots("selection").astype(np.int64)
            assert np.array_equal(decoded, addresses), "selection changed"
            total = sim.read_across_shots(km.array([km.q(i) for i in range(circuit.num_qubits)]))
            assert not np.any(total[n_sel + brot * J:]), "scratch not clean"
    except AssertionError as ex:
        return {"status": "FAIL", "error": str(ex), "shots": int(len(addresses)), "toffolis_per_application": toffolis,
                "inputs": "gauge (b_(k-1), b_k, random of the incoming gauge)"}
    return {"status": "PASS", "shots": int(len(addresses)), "toffolis_per_application": toffolis,
            "inputs": "gauge (b_(k-1), b_k, random of the incoming gauge)"}
