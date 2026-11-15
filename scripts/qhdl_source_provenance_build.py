#!/usr/bin/env python3
"""Source-line provenance build for a kickmix DQI decoder.

Rebuilds a decoder with its builder wrapped in a proxy. Before each builder call whose source key changes, the proxy
emits an extra INFO_TAG "@src:<key>". The key is the innermost stack frame inside the ENTRY file (line and function)
plus the innermost frame of any file (basename:function) that made the builder call. Nothing else changes: removing
the "@src:" tags gives back the plain build op for op (checked by --check-against, which compares op-type streams and
qubit streams of the stripped build with a reference .kmb).

Usage (run with the kickmix interpreter, -I, from an empty working directory):
  python -I qhdl_source_provenance_build.py SOURCE_DIR OUT.kmb [--entry final_decoder/decode.py]
      [--check-against REF.kmb]
Writes OUT.kmb (tagged) and OUT.srcmap.npz: for every op of the PLAIN build, the index of its source key, plus the
key table (OUT.srcmap.json).

SOURCE_DIR is the block's source folder (the one holding build_897.py). Treat it as untrusted code: read it first.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


class Proxy:
    def __init__(self, b, entry_path):
        self._b = b
        self._entry = str(entry_path)
        self._last = None
        self.keys = {}

    def _key(self):
        f = sys._getframe(2)
        inner = f'{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}'
        while f is not None and f.f_code.co_filename != self._entry:
            f = f.f_back
        if f is None:
            return f'?|{inner}'
        return f'L{f.f_lineno}:{f.f_code.co_name}|{inner}'

    def __getattr__(self, name):
        attr = getattr(self._b, name)
        if not callable(attr) or name in ('mark', 'alloc_qubits', 'alloc_bits', 'free', 'num_allocated_qubits',
                                           'num_free_qubits', 'max_qubits', 'finish_circuit', 'info_tag'):
            return attr

        def call(*a, **k):
            key = self._key()
            if key != self._last:
                self._b.info_tag('@src:' + key)
                self._last = key
            return attr(*a, **k)
        return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source_dir')
    ap.add_argument('out')
    ap.add_argument('--entry', default='final_decoder/decode.py')
    ap.add_argument('--check-against')
    args = ap.parse_args()
    import kickmix as km
    sys.modules['circ'] = km
    src = Path(args.source_dir).resolve()
    entry = src / args.entry
    t0 = time.time()
    spec = importlib.util.spec_from_file_location('decoder_under_test', entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules['decoder_under_test'] = module
    spec.loader.exec_module(module)
    raw = km.CircuitBuilder()
    S = [raw.alloc_qubits(12) for _ in range(70)]
    E = [raw.alloc_bits(12) for _ in range(4095)]
    proxy = Proxy(raw, entry.resolve())
    module.decode(proxy, S, E)
    metadata = []
    for i in range(70):
        metadata.append(f'REGISTER r{i} "S{i}"')
        metadata.extend(f'APPEND_TO_REGISTER q{12 * i + k} r{i}' for k in range(12))
    for j in range(4095):
        metadata.append(f'REGISTER r{70 + j} "E{j}"')
        metadata.extend(f'APPEND_TO_REGISTER b{12 * j + k} r{70 + j}' for k in range(12))
    c = km.Circuit('\n'.join(metadata)) + raw.finish_circuit()
    out = Path(args.out)
    c.to_file(str(out), format='kmb')
    print(f'built tagged circuit in {time.time() - t0:.0f} s: Q {c.num_qubits}, ops {len(c)}', flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import qhdl_kmb as K
    k = K.read(str(out))
    t = k.op_types
    tag_idx = np.flatnonzero(t == K.INFO_TAG)
    tags = k.info_tags()
    is_src = np.array([s.startswith('@src:') for s in tags], dtype=bool)
    src_ops = tag_idx[is_src]
    keep = np.ones(len(t), dtype=bool)
    keep[src_ops] = False
    key_names, key_index = [], {}
    seg_key = np.empty(len(src_ops), dtype=np.int32)
    for n, s in enumerate(np.array(tags, dtype=object)[is_src]):
        name = s[5:]
        if name not in key_index:
            key_index[name] = len(key_names)
            key_names.append(name)
        seg_key[n] = key_index[name]
    # key of each plain op = key of the last @src tag before it (-1 before the first)
    pos = np.searchsorted(src_ops, np.flatnonzero(keep), side='left') - 1
    plain_key = np.where(pos >= 0, seg_key[np.maximum(pos, 0)], -1).astype(np.int32)
    np.savez_compressed(str(out.with_suffix('.srcmap.npz')), plain_key=plain_key)
    out.with_suffix('.srcmap.json').write_text(json.dumps(dict(entry=str(args.entry), keys=key_names,
                                                                src_tags=int(is_src.sum()),
                                                                plain_ops=int(keep.sum())), indent=0))
    print(f'{int(is_src.sum())} @src tags, {len(key_names)} distinct keys, plain ops {int(keep.sum())}')
    if args.check_against:
        ref = K.read(args.check_against)
        ok = ref.op_types.shape[0] == int(keep.sum()) and np.array_equal(ref.op_types, t[keep])
        ops_a, qs_a, _ = K.op_qubits(ref)
        ops_b, qs_b, _ = K.op_qubits(k)
        mb = keep[ops_b]
        ok_q = np.array_equal(qs_a, qs_b[mb])
        print(f'check against {args.check_against}: op types equal {ok}, qubit streams equal {ok_q}')
        if not (ok and ok_q):
            sys.exit(2)


if __name__ == '__main__':
    main()
