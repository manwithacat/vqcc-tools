# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Low-cache floor of a family's ANGLE_k loaders (dfthc_refit 1.8.0-dev).

In the cached-product loader model (Lera's `tensorqrom.build_circuit` and this tool's gauge emitter; every shipped
public loader recovered so far), every product control's low part is a XOR of low *source* qubits: the low address
bits, cached low monomials (one AND each) and, in a pi-gauge family, the incoming top bits of words whose gauge does
not change in that slot (`gauge.free_monomials`: after `bit 7 ^= bit 6` such a top bit holds its gauge pattern, a
function of the low address bits, on both passes the verifier runs). Two facts follow, per slot:

- **Floor.** If the mixed ANF matrix (low monomial x (high monomial, output bit), real addresses only: the last high
  row, which holds the never-read padded addresses, is left out) has full GF(2) row rank over the low monomials,
  every decomposition's low forms span the whole low space, so the sources must span all 2**lowbits - 1 - lowbits
  nonlinear monomials: **low-cache ANDs >= nonlinear monomials - free gauge monomials**. With lower rank the bound
  is reported as not proved.
- **Restoring sources.** A word whose gauge pattern changes (or is absent) can become a free source only through a
  different gauge choice. Pure pi-gauge moves are exact symmetries that leave bits 0..brot-2 (every plan) unchanged
  (`gauge_search.group`); a column that no pure-pi kernel element touches cannot be re-gauged that way.

`family_lowcache` reports both, plus the shipped low-cache count when the loaders are given (`recover`). Read only.
"""
from __future__ import annotations

import numpy as np

import gauge as gz
from recover import anf_rows, rank

LOWBITS = 4


def anf_terms(col, lowbits=LOWBITS) -> list[int]:
    """ANF terms (low monomial masks) of a 0/1 function of the low address bits."""
    L = 1 << lowbits
    a = np.asarray(col, np.uint8).reshape(L).copy()
    bit = 1
    while bit < L:
        for x in range(L):
            if x & bit:
                a[x] ^= a[x ^ bit]
        bit <<= 1
    return [m for m in range(L) if a[m]]


def low_rank(full_delta, brot=8, lowbits=LOWBITS, drop_last=True) -> int:
    """GF(2) rank of the low monomials' rows of the mixed ANF matrix (each row: one low monomial, columns
    (high monomial, output bit)); `drop_last` leaves out the last high row (padded addresses)."""
    rows, m, Lm = anf_rows(np.asarray(full_delta, np.int64), brot, lowbits)
    if drop_last:
        rows = rows[:-1]
    mask = (1 << m) - 1
    vecs = []
    for low in range(Lm):
        v = 0
        for h, r in enumerate(rows):
            v |= ((r >> (low * m)) & mask) << (h * m)
        vecs.append(v)
    return rank(vecs)


def slot_lowcache(words, k, *, j=5, brot=8, lowbits=LOWBITS, full_delta=None) -> dict:
    """Free gauge sources, the floor and its proof status for slot k."""
    pin, pout = gz.slot_gauge(words, k, j)[:2]
    free = gz.free_monomials(pin, pout, lowbits)
    nonlinear = (1 << lowbits) - 1 - lowbits
    row = dict(k=k, p_in_anf=[anf_terms(pin[:, c], lowbits) for c in range(pin.shape[1])],
               p_out_anf=[anf_terms(pout[:, c], lowbits) for c in range(pout.shape[1])],
               changing_words=[int(c) for c in range(pin.shape[1]) if (pin[:, c] != pout[:, c]).any()],
               free_monomials={int(a): int(b) for a, b in free.items()},
               floor=nonlinear - len(free))
    if full_delta is not None:
        r = low_rank(full_delta, brot, lowbits)
        row.update(low_rank=r, floor_proved=r == (1 << lowbits) - 1)
    return row


def pure_pi_support(edges, n_cols, brot=8) -> list[int]:
    """Columns that some pure pi-gauge symmetry (n(f) = 0) of the Givens tree touches."""
    import gauge_search as gs
    g = gs.group(np.asarray(edges, np.int64), brot)
    ker = gs._kernel_basis(g["Nf"].astype(np.int64))
    if not ker:
        return []
    K = np.array(ker, np.int64)
    return [int(c) for c in np.flatnonzero(K.any(axis=0)) if c < n_cols]


def family_lowcache(words, deltas=None, *, edges=None, shipped_low=None, j=5, brot=8, lowbits=LOWBITS) -> dict:
    """Per slot: free sources, floor (proved or not), shipped low caches and the gap; the columns that lose a gauge
    pattern and whether a pure pi move can touch them. `deltas`: {k: full-block delta} for the rank test;
    `shipped_low`: {k: low caches recovered from the shipped loader}."""
    words = np.asarray(words, np.int64)
    n = -(-words.shape[1] // j)
    rows = []
    for k in range(n):
        r = slot_lowcache(words, k, j=j, brot=brot, lowbits=lowbits,
                          full_delta=None if deltas is None else deltas.get(k))
        if shipped_low is not None and k in shipped_low:
            r.update(shipped_low_caches=int(shipped_low[k]), gap=int(shipped_low[k]) - r["floor"])
        rows.append(r)
    lost = sorted({j * r["k"] + c for r in rows for c in r["changing_words"] if j * r["k"] + c < words.shape[1]})
    # columns whose (outgoing) gauge pattern is not one nonlinear monomial: they can never be free sources as stored
    plain = sorted({j * r["k"] + c for r in rows for c, t in enumerate(r["p_out_anf"])
                    if len([m for m in t if m & (m - 1)]) != 1 and j * r["k"] + c < words.shape[1]})
    rep = dict(slots=rows, floor_total=sum(r["floor"] for r in rows),
               gap_total=sum(r.get("gap", 0) for r in rows) if shipped_low is not None else None,
               columns_with_changing_gauge=lost, columns_without_single_monomial=plain)
    if edges is not None:
        supp = set(pure_pi_support(edges, words.shape[1], brot))
        rep.update(pure_pi_support=sorted(supp),
                   changing_columns_untouchable=[c for c in lost if c not in supp],
                   plain_columns_untouchable=[c for c in plain if c not in supp])
    return rep
