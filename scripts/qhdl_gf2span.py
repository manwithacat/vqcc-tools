"""GF(2) span test over sampled wire values.

span_relation(S, w, candidates) -> (subset, const) such that S[w] == XOR(S[p] for p in subset) ^ const on every
sample, or None. S: bool array [num_wires, num_samples].
"""
import numpy as np


def _ints(S):
    packed = np.packbits(S, axis=1)
    return [int.from_bytes(r.tobytes(), 'big') for r in packed]


def span_relation(S, w, candidates):
    ns = S.shape[1]
    cols = _ints(S)
    ones = int.from_bytes(np.packbits(np.ones(ns, bool)).tobytes(), 'big')
    basis = {}  # pivot bit -> (vector, mask of contributing items)
    items = [('1', ones)] + [(p, cols[p]) for p in candidates if p != w]
    names = [n for n, _ in items]
    for idx, (_, v) in enumerate(items):
        m = 1 << idx
        while v:
            pb = v.bit_length() - 1
            if pb in basis:
                bv, bm = basis[pb]
                v ^= bv
                m ^= bm
            else:
                basis[pb] = (v, m)
                break
    v = cols[w]
    m = 0
    while v:
        pb = v.bit_length() - 1
        if pb not in basis:
            return None
        bv, bm = basis[pb]
        v ^= bv
        m ^= bm
    sub = [names[i] for i in range(len(names)) if m >> i & 1]
    const = '1' in sub
    sub = [p for p in sub if p != '1']
    # verify
    acc = np.zeros(ns, bool) ^ const
    for p in sub:
        acc ^= S[p]
    assert np.array_equal(acc, S[w])
    return sub, const


def gf2_rank(S, rows=None):
    """Rank over GF(2) of the wire-value vectors S[rows] (plus the all-ones vector) and the list of dependent rows."""
    rows = list(range(S.shape[0])) if rows is None else list(rows)
    cols = _ints(S[rows])
    ones = int.from_bytes(np.packbits(np.ones(S.shape[1], bool)).tobytes(), 'big')
    basis = {}
    dependent = []
    for name, v in [('1', ones)] + list(zip(rows, cols)):
        while v:
            pb = v.bit_length() - 1
            if pb in basis:
                v ^= basis[pb]
            else:
                basis[pb] = v
                break
        if v == 0 and name != '1':
            dependent.append(name)
    return len(basis), dependent
