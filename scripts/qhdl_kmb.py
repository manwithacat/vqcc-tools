"""Minimal NumPy reader/patcher for kickmix binary circuits (.kmb, format version 2).

Written for the lifetime allocator (qhdl_lifetime_allocator.py).
Layout follows kickmix `Circuit::write_kmb_to` (Qualtran kickmix src/kickmix/circuit/circuit.cc): a 16-byte magic,
little-endian u32/u64 header fields, register packets, then one packet per stream (op types, qqq0..2, qq0..1, q0,
b0, bc, angles, info tags). Qubit ids in the streams are untagged u32.

`patch_qubits` writes a copy whose only differences are the qubit streams, the register qubit ids and the header
`num_qubits`; every other byte is copied from the input.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

# Operation-type bit layout (kickmix src/kickmix/circuit/op_type.h).
OP_HAS_1Q, OP_HAS_2Q, OP_HAS_1B, OP_HAS_COND = 0x10, 0x20, 0x40, 0x80
POP_CONDITION, INFO_TAG = 0x01, 0x06
R, Z_POW, HMR = 0x12, 0x13, 0x50
SWAP = 0x22
PUSH_CONDITION = 0x81
TOFFOLI_LIKE = (0x30, 0x31, 0x32, 0xB0, 0xB1, 0xB2)  # CCZ CCX CSWAP and their _IF forms

MAGIC = bytes.fromhex('d750c7d5c329d326e3cc9f6834f2b8bf')
STREAMS = ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc', 'angles']


@dataclass
class Kmb:
    path: str
    raw: np.ndarray  # whole file as uint8 (memory-mapped)
    num_qubits: int
    num_bits: int
    counts: dict
    registers: list  # [(name, [(is_qubit, id, byte_offset_of_u32)])]
    offsets: dict = field(default_factory=dict)  # stream -> (byte offset, element count, dtype)
    op_types: np.ndarray | None = None

    def stream(self, name):
        off, n, dt = self.offsets[name]
        return np.frombuffer(self.raw, dtype=dt, count=n, offset=off)

    def info_tags(self):
        off, n, _ = self.offsets['info_tags']
        ends = np.frombuffer(self.raw, dtype='<u8', count=n, offset=off)
        chars = bytes(self.raw[off + 8 * n: off + 8 * n + self.counts['info_tag_chars']])
        out, start = [], 0
        for e in ends.tolist():
            out.append(chars[start:e].decode('utf-8', 'replace'))
            start = e
        return out


class KmbFormatError(ValueError):
    """The file is not a version-2 kickmix binary circuit (or is truncated)."""


def _need(cond, path, what):
    if not cond:
        raise KmbFormatError(f'{path}: not a valid version-2 .kmb ({what})')


def read(path: str) -> Kmb:
    try:
        raw = np.memmap(path, dtype=np.uint8, mode='r')
    except ValueError:  # empty file
        raise KmbFormatError(f'{path}: empty file, not a .kmb')
    buf = memoryview(raw)
    _need(len(raw) >= 16 + 16 + 72, path, 'file shorter than the header')
    _need(bytes(raw[:16]) == MAGIC, path, 'wrong magic bytes; a .kmx text file or another format?')
    pos = 16
    version, num_qubits, num_bits, num_regs = struct.unpack_from('<4I', buf, pos)
    _need(version == 2, path, f'format version {version}')
    pos += 16
    keys = ['ops', 'qqq', 'qq', 'q', 'b', 'c', 'angle', 'info_tags', 'info_tag_chars']
    counts = dict(zip(keys, struct.unpack_from('<9Q', buf, pos)))
    pos += 72
    registers = []
    try:
        for _ in range(num_regs):
            ptype, size = struct.unpack_from('<2Q', buf, pos)
            _need(ptype == 0, path, 'register packet expected')
            pos += 16
            name_len, reg_len = struct.unpack_from('<2I', buf, pos)
            pos += 8
            name = bytes(buf[pos:pos + name_len]).decode()
            pos += name_len
            contents = []
            for k in range(reg_len):
                (v,) = struct.unpack_from('<I', buf, pos)
                contents.append((bool(v & 0x80000000), v & 0x7FFFFFFF, pos))
                pos += 4
            registers.append((name, contents))
    except (struct.error, UnicodeDecodeError):
        raise KmbFormatError(f'{path}: not a valid version-2 .kmb (truncated register packets)')
    k = Kmb(path, raw, num_qubits, num_bits, counts, registers)

    def packet(expected_type, n, dt, name):
        nonlocal pos
        _need(pos + 16 <= len(raw), path, f'truncated before stream {name}')
        ptype, size = struct.unpack_from('<2Q', buf, pos)
        _need(ptype == expected_type, path, f'stream {name}: packet type {ptype}, expected {expected_type}')
        itemsize = np.dtype(dt).itemsize
        _need(size == n * itemsize and pos + 16 + size <= len(raw), path, f'stream {name}: bad size')
        k.offsets[name] = (pos + 16, n, dt)
        pos += 16 + size

    packet(1, counts['ops'], 'u1', 'op_types')
    for i, (nm, cnt) in enumerate([('qqq0', 'qqq'), ('qqq1', 'qqq'), ('qqq2', 'qqq'), ('qq0', 'qq'), ('qq1', 'qq'),
                                    ('q0', 'q'), ('b0', 'b'), ('bc', 'c')]):
        packet(2 + i, counts[cnt], '<u4', nm)
    packet(10, counts['angle'], 'V16', 'angles')
    _need(pos + 24 <= len(raw), path, 'truncated before info tags')
    ptype, size = struct.unpack_from('<2Q', buf, pos)
    _need(ptype == 11, path, 'info-tag packet expected')
    (ntags,) = struct.unpack_from('<Q', buf, pos + 16)
    _need(ntags == counts['info_tags'], path, 'info-tag count mismatch')
    k.offsets['info_tags'] = (pos + 24, ntags, '<u8')
    pos += 16 + size
    _need(pos == len(raw), path, f'{len(raw) - pos} trailing bytes')
    k.op_types = k.stream('op_types')
    return k


def op_qubits(k: Kmb):
    """Per qubit reference: (op index, qubit id, slot) for every qubit operand, in stream order.

    Returns (ops, qubits) as int64/uint32 arrays covering 3Q, 2Q and 1Q operands.
    """
    t = k.op_types
    qcls = (t & 0x30)
    idx3 = np.flatnonzero(qcls == 0x30)
    idx2 = np.flatnonzero(qcls == 0x20)
    idx1 = np.flatnonzero(qcls == 0x10)
    assert len(idx3) == k.counts['qqq'] and len(idx2) == k.counts['qq'] and len(idx1) == k.counts['q']
    ops = np.concatenate([idx3, idx3, idx3, idx2, idx2, idx1])
    qs = np.concatenate([k.stream('qqq0'), k.stream('qqq1'), k.stream('qqq2'), k.stream('qq0'), k.stream('qq1'),
                         k.stream('q0')])
    return ops, qs, (idx3, idx2, idx1)


def condition_depth(t: np.ndarray) -> np.ndarray:
    """Condition-stack depth in force at each op (PUSH_CONDITION +1 after, POP_CONDITION -1 after)."""
    delta = np.zeros(len(t), dtype=np.int32)
    delta[t == PUSH_CONDITION] = 1
    delta[t == POP_CONDITION] = -1
    after = np.cumsum(delta, dtype=np.int64)
    before = after - delta
    # A PUSH itself is unconditional; ops between PUSH and POP see depth >= 1.
    return before


def patch_qubits(k: Kmb, out_path: str, new_streams: dict, register_map: dict, num_qubits: int) -> None:
    """Write a copy of `k` with replaced qubit streams (qqq0..2, qq0..1, q0), remapped register qubit ids and header."""
    data = np.array(k.raw, copy=True)
    struct.pack_into('<I', data, 20, num_qubits)  # header: magic(16) version(4) num_qubits(4)
    for name, arr in new_streams.items():
        off, n, dt = k.offsets[name]
        arr = np.ascontiguousarray(arr, dtype='<u4')
        assert len(arr) == n
        data[off:off + 4 * n] = arr.view(np.uint8)
    for name, contents in k.registers:
        for is_q, qid, pos in contents:
            if is_q:
                struct.pack_into('<I', data, pos, register_map.get(qid, qid) | 0x80000000)
    data.tofile(out_path)


NAMES = {0x00: 'NEG', 0x01: 'POP_CONDITION', 0x06: 'INFO_TAG', 0x40: 'BIT_STORE0', 0x41: 'BIT_STORE1',
         0x42: 'BIT_INVERT', 0x10: 'Z', 0x11: 'X', 0x12: 'R', 0x13: 'Z_POW', 0x50: 'HMR', 0x20: 'CZ', 0x21: 'CX',
         0x22: 'SWAP', 0x30: 'CCZ', 0x31: 'CCX', 0x32: 'CSWAP', 0x81: 'PUSH_CONDITION'}
# Base names only: a conditioned op is base | 0x80 (0x80 itself is NEG_IF), so look names up with `code & 0x7F` (and
# special-case PUSH_CONDITION, 0x81). Until 2026-11-01 this table also mapped 0x80 to 'NEG'.


class Decoder:
    """Random-access text rendering of ops (for inspection only)."""

    def __init__(self, k: Kmb):
        self.k = k
        t = k.op_types.astype(np.uint8)
        q = t & 0x30
        self.c3 = np.cumsum(q == 0x30) - (q == 0x30)
        self.c2 = np.cumsum(q == 0x20) - (q == 0x20)
        self.c1 = np.cumsum(q == 0x10) - (q == 0x10)
        hb = (t & 0x40) > 0
        self.cb = np.cumsum(hb) - hb
        hc = (t & 0x80) > 0
        self.cc = np.cumsum(hc) - hc
        ht = t == INFO_TAG
        self.ct = np.cumsum(ht) - ht
        self.s = {n: k.stream(n) for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc']}
        self.tags = None

    def qubits(self, i):
        t = int(self.k.op_types[i]); q = t & 0x30
        if q == 0x30:
            j = self.c3[i]; return [int(self.s['qqq0'][j]), int(self.s['qqq1'][j]), int(self.s['qqq2'][j])]
        if q == 0x20:
            j = self.c2[i]; return [int(self.s['qq0'][j]), int(self.s['qq1'][j])]
        if q == 0x10:
            return [int(self.s['q0'][self.c1[i]])]
        return []

    def text(self, i):
        t = int(self.k.op_types[i])
        if t == INFO_TAG:
            if self.tags is None:
                self.tags = self.k.info_tags()
            return f'INFO_TAG "{self.tags[self.ct[i]]}"'
        base = t & ~0x80 if (t & 0x80 and t != PUSH_CONDITION) else t
        name = NAMES.get(base, hex(base))
        parts = [name] + [f'q{x}' for x in self.qubits(i)]
        if t & 0x40:
            parts.append(f'b{int(self.s["b0"][self.cb[i]])}')
        if t & 0x80:
            parts.append(('if ' if t != PUSH_CONDITION else '') + f'b{int(self.s["bc"][self.cc[i]])}')
        return ' '.join(parts)


def write_kmb(path, op_types, streams, angles=b'', tags=(), registers=(), num_qubits=None, num_bits=None):
    """Write a version-2 .kmb from arrays. streams: dict of the 8 u32 streams; tags: list of bytes."""
    ot = np.ascontiguousarray(op_types, dtype=np.uint8)
    s = {n: np.ascontiguousarray(streams.get(n, np.zeros(0, np.uint32)), dtype='<u4') for n in
         ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc']}
    qmax = -1
    for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0']:
        if len(s[n]):
            qmax = max(qmax, int(s[n].max()))
    bmax = -1
    for n in ['b0', 'bc']:
        if len(s[n]):
            bmax = max(bmax, int(s[n].max()))
    for _, contents in registers:
        for is_q, i in contents:
            if is_q:
                qmax = max(qmax, i)
            else:
                bmax = max(bmax, i)
    nq = qmax + 1 if num_qubits is None else num_qubits
    nb = bmax + 1 if num_bits is None else num_bits
    tag_bytes = [bytes(t) for t in tags]
    ends = np.cumsum([len(t) for t in tag_bytes]).astype('<u8') if tag_bytes else np.zeros(0, '<u8')
    chars = b''.join(tag_bytes)
    angles = bytes(angles)
    out = bytearray()
    out += MAGIC
    out += struct.pack('<4I', 2, nq, nb, len(registers))
    out += struct.pack('<9Q', len(ot), len(s['qqq0']), len(s['qq0']), len(s['q0']), len(s['b0']), len(s['bc']),
                       len(angles) // 16, len(tag_bytes), len(chars))
    for name, contents in registers:
        nm = name.encode()
        out += struct.pack('<2Q', 0, len(nm) + 4 * len(contents) + 8)
        out += struct.pack('<2I', len(nm), len(contents)) + nm
        for is_q, i in contents:
            out += struct.pack('<I', i | (0x80000000 if is_q else 0))
    out += struct.pack('<2Q', 1, len(ot)) + ot.tobytes()
    for i, n in enumerate(['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc']):
        out += struct.pack('<2Q', 2 + i, 4 * len(s[n])) + s[n].tobytes()
    out += struct.pack('<2Q', 10, len(angles)) + angles
    out += struct.pack('<2Q', 11, 8 * (len(tag_bytes) + 1) + len(chars)) + struct.pack('<Q', len(tag_bytes))
    out += ends.tobytes() + chars
    with open(path, 'wb') as f:
        f.write(out)


def to_file(path, op_types, streams, angles=b'', tags=(), registers=(), num_qubits=None, num_bits=None,
            format=None):
    """Write a circuit given as arrays (see write_kmb). The format follows the path suffix unless given: '.kmb'
    writes KMB bytes (always, never kmx text); '.kmx' writes kickmix text through the kickmix venv (a KMB written
    to a temporary file and converted by kickmix itself, so the text is kickmix's own)."""
    from pathlib import Path
    path = Path(path)
    fmt = format or path.suffix.lstrip('.').lower() or 'kmb'
    if fmt == 'kmb':
        write_kmb(str(path), op_types, streams, angles, tags, registers, num_qubits, num_bits)
        return str(path)
    if fmt != 'kmx':
        raise ValueError(f'unknown circuit format {fmt!r} (use .kmb or .kmx)')
    import os
    import subprocess
    import sys
    import tempfile
    kmpy = os.environ.get('QHDL_KICKMIX_PYTHON') or os.environ.get('VQCC_KMPY') or sys.executable
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / 'c.kmb'
        write_kmb(str(tmp), op_types, streams, angles, tags, registers, num_qubits, num_bits)
        code = ('import kickmix as km, sys; c = km.Circuit.from_file(sys.argv[1], format="kmb"); '
                'c.to_file(sys.argv[2], format="kmx")')
        subprocess.run([kmpy, '-I', '-c', code, str(tmp), str(path)], check=True, capture_output=True, text=True)
    return str(path)


def magic_of(k):
    return bytes(k.raw[:16])


class Slicer:
    """Cut [a, b) op ranges of a circuit into standalone .kmb segments (no registers, no tags, num ids kept)."""

    def __init__(self, k: Kmb):
        self.k = k
        t = k.op_types.astype(np.uint8)
        q = t & 0x30
        z = lambda m: np.concatenate([[0], np.cumsum(m)])
        self.c3, self.c2, self.c1 = z(q == 0x30), z(q == 0x20), z(q == 0x10)
        self.cb, self.cc = z((t & 0x40) > 0), z((t & 0x80) > 0)
        self.ca = z((t & 0x7F) == 0x13)
        self.ct = z(t == INFO_TAG)

    def write(self, path, a, b, replace=None):
        """replace: optional (op_types, streams) for the segment instead of the original slice."""
        k = self.k
        if replace is None:
            ot = np.array(k.op_types[a:b])
            st = {}
            for n, c in [('qqq0', self.c3), ('qqq1', self.c3), ('qqq2', self.c3), ('qq0', self.c2), ('qq1', self.c2),
                         ('q0', self.c1), ('b0', self.cb), ('bc', self.cc)]:
                st[n] = np.array(k.stream(n)[c[a]:c[b]])
        else:
            ot, st = replace
        # drop info tags (no effect on behaviour) to keep segments simple
        keep = ot != INFO_TAG
        ot = ot[keep]
        off, n, _ = k.offsets['angles']
        ang = bytes(k.raw[off + 16 * self.ca[a]: off + 16 * self.ca[b]]) if replace is None else b''
        write_kmb(path, ot, st, ang, (), ())
