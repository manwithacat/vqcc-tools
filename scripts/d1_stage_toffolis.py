#!/usr/bin/env python3
# Copyright 2026 James Barlow. SPDX-License-Identifier: Apache-2.0. AI-assisted (Claude, Anthropic).
"""Static Toffoli-like gates per builder stage of a stage-tagged DQI build (tags '@stage:NAME', written after
each stage by the launcher wrapper). Usage: python3 scripts/d1_stage_toffolis.py TAGGED.kmb OUT.json
Each region is the ops between two consecutive stage tags, attributed to the later tag (the stage that just ended).
Static counts (an upper bound on the per-shot mean; conditioned gates are counted separately)."""
import collections, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qhdl_kmb as K

k = K.read(sys.argv[1]); t = k.op_types; base = t & np.uint8(0x7F)
tof = np.isin(base, [0x30, 0x31, 0x32]) & (t != K.PUSH_CONDITION)
dep = K.condition_depth(t)
cond = tof & ((dep > 0) | (t >= 0x80))
idx = np.flatnonzero(t == K.INFO_TAG); tags = k.info_tags()
ct = np.r_[0, np.cumsum(tof)]; cc = np.r_[0, np.cumsum(cond)]
rows, prev = [], 0
for o, s in zip(idx, tags):
    if not s.startswith('@stage:'): continue
    rows.append(dict(stage=s[7:], ops=int(o - prev), toffoli=int(ct[o] - ct[prev]), conditioned=int(cc[o] - cc[prev])))
    prev = int(o)
rows.append(dict(stage='(after last stage)', ops=int(len(t) - prev), toffoli=int(ct[-1] - ct[prev]), conditioned=int(cc[-1] - cc[prev])))
def group(n):
    n = re.sub(r':\d+$', '', n); n = re.sub(r'_\d\d$', '', n); return n
g = collections.OrderedDict()
for r in rows:
    a = g.setdefault(group(r['stage']), dict(stages=0, ops=0, toffoli=0, conditioned=0))
    a['stages'] += 1
    for x in ('ops', 'toffoli', 'conditioned'): a[x] += r[x]
tot = int(ct[-1])
Path(sys.argv[2]).write_text(json.dumps(dict(total_toffoli=tot, total_ops=int(len(t)), groups=g, rows=rows), indent=1))
print('total static Toffoli-like', tot, 'ops', len(t))
for n, a in sorted(g.items(), key=lambda kv: -kv[1]['toffoli']):
    print(f"{a['toffoli']:>8} {100*a['toffoli']/tot:5.1f}%  cond {a['conditioned']:>7}  x{a['stages']:<4} {n}")
