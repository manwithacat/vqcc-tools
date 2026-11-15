"""QHDL v1 Phase 0: claim identity, the automatic trail, the investigation container and stable addresses
(the `qhdl inv` and `qhdl addr` core verbs).

Imported by scripts/qhdl.py (core, not a plugin). Everything here is off unless asked for, so every existing command
behaves as before:

* Trail. With QHDL_INVESTIGATION=<dir> set, every qhdl verb call appends one JSON line to <dir>/trail.jsonl:
  {at, claim_id, verb, argv, rerun, tier, provenance, scope, evidence, circuit_sha, tool_sha, seconds, exit,
   addresses?, error?}. The wall-clock stamp lives only in the trail, never in the envelope.
* Identity. claim_id = sha12 of (verb, rerun string without env prefix, tool_sha, circuit_sha). provenance is
  computed | recorded | derived. Certify verbs (fdiff, verify) also get a certificate scope
  {domain: all-x | draws:<n>@<anchor> | inputs:<n>@<seed>, extent: whole | window A:B,
   branches: zero | zero+random(<seed>) | all-outcomes | engine-sampled | official}.
* Addresses (resolved before the verb runs, on --ops, --ops2 and --at):
    tag:<path>[k]          the k-th (0-based; -1 = last) instance of an INFO_TAG path, as the range enter..exit+1
    tag:<path>[k]+off      the op `off` ops after that instance's @enter tag (INFO_TAG ops count, as in op indices)
    tag:<path>[k]+a:+b     range a..b relative to the same instance;  ...:tag:<path2>[j]+b  or  ...:<abs>
    op@<sha12>:<i>         an op of a named circuit; op@<sha12>:A:B a range there
    --as-of MAP[,MAP2...]  translate op@ addresses (and bare A:B) from the map chain's first circuit into the
                           target circuit, through `qhdl edit` .map.json files or span-list maps
                           ([{old: [a, b], new: [c, d]}, ...]); a cut strictly inside an edited range is refused.
  A path that is not a full tag path may be a unique suffix (`tag:companion_normalize[0]`). Untagged circuits (every
  public invsqrt) have no tag addresses: use absolute indices plus --as-of.
* Investigation verbs: qhdl inv new | hypothesis | kill | confirm | park | candidate | log | status | resume |
  ledger-stage. Directory layout (investigations/<date>-<slug>/):
    investigation.json  question, task, base circuits, card (free-text reference to a plan or note), budget, stop rule, tool_sha at start
    hypotheses.jsonl    one full hypothesis snapshot per line, append-only; the last line per id is current
    candidates.jsonl    one candidate/certificate snapshot per line; the last line per sha is current
    trail.jsonl         automatic verb trail (and `inv log` rows for helper runs)
    notes.md            free prose for the next agent (<= 1 page)
    ledger-stage.json   derived by `inv ledger-stage`: negative-results row drafts (JSON) for a ledger kept outside
                        qhdl (a human adds them; qhdl never writes docs/)
    leads-stage.md      derived: leads-table row drafts (Markdown)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

import qhdl_envelope as E

SCHEMA = 1
STATUSES = ('open', 'killed', 'confirmed', 'parked', 'superseded')
LEDGER_OUTCOMES = ('kill', 'null', 'partial', 'open idea', 'observation')
EXTRA_TIERS = ('recorded', 'structural')  # outcome tiers beyond the envelope ladder
CERT_VERBS = ('fdiff', 'verify', 'gate')
RESUME_MAX_BYTES = 4096  # resume carries a whole investigation; still one bounded envelope
TRAIL_TAIL = 3

# ---------------------------------------------------------------- identity


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


_ENV_PREFIX = re.compile(r'^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+')


def strip_env(rerun: str) -> str:
    """Drop leading NAME=value assignments (QHDL_WORK=...) so a claim id does not depend on scratch paths."""
    return _ENV_PREFIX.sub('', rerun or '').strip()


def claim_id(verb: str, rerun: str, tool_sha: str, circuit_sha: str | None) -> str:
    key = json.dumps([verb or '', strip_env(rerun), tool_sha or '', circuit_sha or ''], separators=(',', ':'))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def default_provenance(verb: str) -> str:
    if verb == 'evidence':
        return 'recorded'
    if verb in ('inv', 'addr'):
        return 'derived'
    return 'computed'


def default_scope(verb: str, a, env_dict: dict) -> dict | None:
    """Certificate scope for the certify verbs (fdiff, verify); None for observe/transform verbs."""
    if verb == 'fdiff':
        dom = env_dict.get('domain') or ''
        seed = getattr(a, 'seed', None)
        branches = f'zero+random({seed})' if 'random branch' in dom else 'zero'
        if getattr(a, 'whole', False):
            extent = 'whole'
        else:
            extent = f"window {getattr(a, 'ops', None) or '?'}"
        return {'domain': 'all-x' if 'legal x' in dom or 'all ' in dom else 'inputs:reachable',
                'extent': extent, 'branches': branches}
    if verb == 'verify':
        return {'domain': f"official-salt:{getattr(a, 'salt', '') or 'default'}", 'extent': 'whole',
                'branches': 'official'}
    return None


def scope_is_local(scope: dict | None) -> bool:
    """A window-extent or zero-branch certificate is local evidence, never a PASS."""
    if not scope:
        return True
    return str(scope.get('extent', '')).startswith('window') or scope.get('branches') == 'zero'


def check_scope(scope: dict) -> dict:
    dom, ext, br = scope.get('domain', ''), scope.get('extent', ''), scope.get('branches', '')
    if not (dom == 'all-x' or re.match(r'^(draws|inputs):\d+@\S+$', dom) or dom.startswith('official-salt:')
            or dom == 'inputs:reachable'):
        raise ValueError(f'scope domain {dom!r}: use all-x, draws:<n>@<anchor> or inputs:<n>@<seed>')
    if not (ext == 'whole' or re.match(r'^window \d+:\d+$', ext)):
        raise ValueError(f"scope extent {ext!r}: use whole or 'window A:B'")
    if not (br in ('zero', 'all-outcomes', 'engine-sampled', 'official') or re.match(r'^zero\+random\(\d+\)$', br)):
        raise ValueError(f'scope branches {br!r}: zero | zero+random(<seed>) | all-outcomes | engine-sampled | '
                         'official')
    return {'domain': dom, 'extent': ext, 'branches': br}


# ---------------------------------------------------------------- files


def _retry(fn, tries=4, wait=2.0):
    """The external SSD sometimes refuses access (EPERM) for a while: wait and retry the same step."""
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def append_jsonl(path: Path, row: dict):
    line = json.dumps(row, ensure_ascii=False, separators=(', ', ': ')) + '\n'

    def go():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)
    _retry(go)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(_retry(lambda: path.read_text(encoding='utf-8')).splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path.name} line {n} is not JSON: {exc}')
    return out


def latest_by(rows, key):
    cur = {}
    for r in rows:
        cur[r[key]] = r
    return cur


def trail_dir():
    """The investigation directory from QHDL_INVESTIGATION, or None. Refuses outputs/ (append-only evidence)."""
    v = os.environ.get('QHDL_INVESTIGATION')
    if not v:
        return None
    p = Path(v).expanduser()
    if 'outputs' in p.resolve().parts:
        sys.stderr.write('qhdl: QHDL_INVESTIGATION points inside outputs/ (append-only evidence); trail not '
                         'written. Use investigations/<date>-<slug>/.\n')
        return None
    return p


def record(verb, argv, env_dict, seconds, exit_code, error=None, addresses=None):
    """Append one trail row if QHDL_INVESTIGATION is set. Never raises (the verb's answer matters more)."""
    d = trail_dir()
    if d is None or verb == 'inv':
        return None
    env_dict = env_dict or {}
    row = dict(at=now_utc(), claim_id=env_dict.get('claim_id'), verb=verb, argv=list(argv),
               rerun=env_dict.get('rerun'), tier=env_dict.get('tier'), provenance=env_dict.get('provenance'),
               scope=env_dict.get('scope'), evidence=env_dict.get('evidence'),
               circuit_sha=env_dict.get('circuit_sha'), tool_sha=env_dict.get('tool_sha'),
               seconds=round(float(seconds), 2), exit=exit_code)
    if addresses:
        row['addresses'] = addresses
    if error:
        row['error'] = str(error)[:300]
    try:
        append_jsonl(d / 'trail.jsonl', row)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f'qhdl: trail not written ({exc})\n')
    return row


# ---------------------------------------------------------------- tag-relative and as-of addresses
_TAG = re.compile(r'^tag:(?P<path>[^\[\]\s]+)(?:\[(?P<k>-?\d+)\])?(?:\+(?P<off>\d+))?$')
_OP = re.compile(r'^op@(?P<sha>[0-9a-f]{12}):(?P<i>\d+)$')
_OPRANGE = re.compile(r'^op@(?P<sha>[0-9a-f]{12}):(?P<a>\d+):(?P<b>\d+)$')
ADDR_FLAGS = {'--ops': ('range', 'c'), '--ops2': ('range', 'c2'), '--at': ('list', 'c')}


def tag_instances(path) -> dict:
    """{full tag path: [(start, end_exclusive)]} for every INFO_TAG instance (enter op .. exit op + 1). .kmb only."""
    import numpy as np
    import qhdl_kmb as K
    k = K.read(str(path))
    t = k.op_types
    pos = np.flatnonzero(t == K.INFO_TAG).tolist()
    tags = k.info_tags()
    n = len(t)
    stack, inst = [], {}
    for o, txt in zip(pos, tags):
        kind, _, name = txt.strip().strip('"').partition(':')
        if kind == '@enter':
            stack.append((name, o))
        elif kind == '@exit' and stack:
            nm, s = stack.pop()
            inst.setdefault('/'.join([x for x, _ in stack] + [nm]), []).append((s, o + 1))
    while stack:  # unclosed tags run to the end
        nm, s = stack.pop()
        inst.setdefault('/'.join([x for x, _ in stack] + [nm]), []).append((s, n))
    return {p: sorted(v) for p, v in inst.items()}


def innermost(instances: dict, i: int):
    """Tag-relative forms of op i, outermost first: [(path, k, start, end, off)]."""
    hits = []
    for p, spans in instances.items():
        for k, (s, e) in enumerate(spans):
            if s <= i < e:
                hits.append((p, k, s, e, i - s))
    return sorted(hits, key=lambda h: (h[0].count('/'), h[2]))


class _Resolver:
    def __init__(self, api, circuit_ref, as_of):
        self.api, self.ref, self.as_of = api, circuit_ref, as_of
        self._c = None
        self._inst = None
        self._maps = None

    @property
    def circuit(self):
        if self._c is None:
            if self.ref is None:
                raise self.api.QhdlError('a tag: or op@ address needs the circuit (-c / -c2) on the same command')
            self._c = self.api.resolve(self.ref)
        return self._c

    def instances(self):
        if self._inst is None:
            p = Path(self.circuit['path'])
            if p.suffix != '.kmb':
                raise self.api.QhdlError('tag addresses need a .kmb circuit', 'use absolute indices on .kmx')
            self._inst = tag_instances(p)
        return self._inst

    def maps(self):
        if self._maps is None:
            self._maps = load_maps(self.as_of) if self.as_of else []
        return self._maps

    def tag(self, text):
        m = _TAG.match(text)
        if not m:
            raise ValueError(f'bad tag address {text!r} (form: tag:<path>[k]+off)')
        path, k, off = m.group('path'), int(m.group('k') or 0), m.group('off')
        inst = self.instances()
        if path not in inst:
            cands = [p for p in inst if p.endswith('/' + path)]
            if len(cands) != 1:
                if not inst:
                    raise self.api.QhdlError(f'circuit {self.circuit["sha"][:12]} has no INFO_TAG regions',
                                             'use absolute op indices (with --as-of across edits)')
                hint = ', '.join(sorted(cands)[:4]) if cands else ', '.join(sorted(inst)[:6])
                raise self.api.QhdlError(f'tag path {path!r} not found' + (' (ambiguous suffix)' if cands else ''),
                                         f'paths include: {hint}')
            path = cands[0]
        spans = inst[path]
        try:
            s, e = spans[k]
        except IndexError:
            raise self.api.QhdlError(f'tag:{path} has {len(spans)} instances; [{k}] is out of range')
        kk = k if k >= 0 else len(spans) + k
        if off is not None:
            o = int(off)
            if s + o > e:
                raise self.api.QhdlError(f'offset +{o} is past the end of tag:{path}[{kk}] ({e - s} ops)')
            return dict(path=path, k=kk, start=s, end=e, op=s + o)
        return dict(path=path, k=kk, start=s, end=e, op=None)

    def op(self, sha12, i):
        tgt = self.circuit['sha']
        if tgt.startswith(sha12) and not self.as_of:
            return i
        if not self.as_of:
            raise self.api.QhdlError(f'op@{sha12}:{i} names another circuit than {tgt[:12]}',
                                     'pass --as-of <edit .map.json> to translate it')
        return translate(self.maps(), sha12, tgt, i)

    def bare(self, i):
        """A bare index: translated only when --as-of is given and the target is the map chain's end."""
        if not self.as_of:
            return i
        ms = self.maps()
        tgt = self.circuit['sha']
        start, end = ms[0].get('old'), ms[-1].get('new')
        if end and end != tgt:
            return i  # e.g. fdiff -c <the edit's input>: its indices are already its own
        if start and start == tgt:
            return i
        return translate(ms, None, tgt, i)


def load_maps(spec: str) -> list:
    """--as-of a.map.json[,b.map.json]: [{old, new, cut(fn)}] in chain order."""
    import qhdl_edit as ED
    out = []
    for p in [x for x in spec.split(',') if x]:
        if p == 'auto':
            raise ValueError('--as-of auto is not supported (use fdiff --map auto, or the .map.json qhdl edit wrote)')
        d = json.loads(_retry(lambda: Path(p).read_text()))
        if isinstance(d, dict) and 'hunks' in d:
            om = ED.OpMap(d['hunks'], d.get('old_ops'), d.get('new_ops'))
            out.append(dict(path=p, old=d.get('old_sha256'), new=d.get('new_sha256'), cut=om.cut))
        elif isinstance(d, list) and all(isinstance(x, dict) and 'old' in x and 'new' in x for x in d):
            out.append(dict(path=p, old=None, new=None, cut=_span_map(d)))
        else:
            raise ValueError(f'{p}: not a qhdl edit map (hunks) or a span-list map ([{{old: [a,b], new: [c,d]}}])')
    if not out:
        raise ValueError('--as-of needs a map path')
    for a, b in zip(out, out[1:]):
        if a['new'] and b['old'] and a['new'] != b['old']:
            raise ValueError(f"--as-of maps do not chain: {a['path']} ends at {a['new'][:12]}, "
                             f"{b['path']} starts at {b['old'][:12]}")
    return out


def _span_map(spans):
    spans = sorted(([int(x['old'][0]), int(x['old'][1]), int(x['new'][0]), int(x['new'][1])] for x in spans))

    def cut(a):
        delta = 0
        for o1, o2, n1, n2 in spans:
            if a < o1:
                break
            if a == o1:
                return n1
            if o1 < a < o2:
                raise ValueError(f'cut @{a} falls inside edited ops {o1}..{o2 - 1}; choose a cut outside the edit')
            delta = n2 - o2
        return a + delta
    return cut


def translate(maps, sha12, target_sha, i):
    if sha12 and maps[0]['old'] and not maps[0]['old'].startswith(sha12):
        raise ValueError(f"op@{sha12}: the --as-of chain starts at {maps[0]['old'][:12]}")
    if maps[-1]['new'] and maps[-1]['new'] != target_sha:
        raise ValueError(f"--as-of chain ends at {maps[-1]['new'][:12]}, not the target circuit {target_sha[:12]}")
    x = i
    for m in maps:
        try:
            x = m['cut'](x)
        except Exception as exc:  # EditError from qhdl_edit
            raise ValueError(str(exc))
    return x


def _point(r: _Resolver, text: str, role: str):
    """One address -> absolute index. role 'start' / 'end' / 'op' picks the end of a bare tag instance."""
    t = text.strip()
    if t.startswith('tag:'):
        h = r.tag(t)
        if h['op'] is not None:
            return h['op'], h
        return (h['end'] if role == 'end' else h['start']), h
    m = _OP.match(t)
    if m:
        return r.op(m.group('sha'), int(m.group('i'))), None
    if t.lstrip('@').isdigit():
        return r.bare(int(t.lstrip('@'))), None
    raise ValueError(f'unrecognised op address {text!r}')


def resolve_range(r: _Resolver, value: str) -> str:
    v = value.strip()
    m = _OPRANGE.match(v)
    if m:
        return f"{r.op(m.group('sha'), int(m.group('a')))}:{r.op(m.group('sha'), int(m.group('b')))}"
    if v.startswith('tag:'):
        m = re.match(r'^(tag:[^\[\]\s:]+(?:\[-?\d+\])?(?:\+\d+)?)(?::(.+))?$', v)
        if not m:
            raise ValueError(f'bad tag range {value!r}')
        head, tail = m.group(1), m.group(2)
        a, h = _point(r, head, 'start')
        if tail is None:
            if h['op'] is not None:
                raise ValueError(f'{value!r} is a single op; give a range: {head}:+<off> or the bare tag:<path>[k]')
            return f"{h['start']}:{h['end']}"
        if tail.startswith('+') and tail[1:].isdigit():
            b = h['start'] + int(tail[1:])
            if b > h['end']:
                raise ValueError(f"end {tail} is past the end of tag:{h['path']}[{h['k']}] ({h['end'] - h['start']} ops)")
        else:
            b, _ = _point(r, tail, 'end')
        return f'{a}:{b}'
    if v.startswith('op@'):
        m = re.match(r'^(op@[0-9a-f]{12}:\d+):(.+)$', v)
        if not m:
            raise ValueError(f'bad op range {value!r}')
        a, _ = _point(r, m.group(1), 'start')
        b, _ = _point(r, m.group(2), 'end')
        return f'{a}:{b}'
    if r.as_of and re.match(r'^\d+:\d+$', v):
        A, B = v.split(':')
        return f'{r.bare(int(A))}:{r.bare(int(B))}'
    return value


def resolve_list(r: _Resolver, value: str) -> str:
    return ','.join(str(_point(r, x, 'op')[0]) if x.strip() else x for x in value.split(','))


def _needs_rewrite(argv) -> bool:
    for i, t in enumerate(argv):
        if t == '--as-of' or t.startswith('--as-of='):
            return True
        flag, eq, val = t.partition('=')
        if flag in ADDR_FLAGS:
            v = val if eq else (argv[i + 1] if i + 1 < len(argv) else '')
            if 'tag:' in v or 'op@' in v:
                return True
    return False


def rewrite_argv(argv, api):
    """Resolve tag:/op@ addresses and --as-of in argv before argparse sees it -> (argv', addresses dict).

    argv without any of these is returned unchanged (every existing command keeps its exact behaviour)."""
    argv = list(argv)
    if (argv and argv[0] in ('inv', 'addr')) or not _needs_rewrite(argv):
        return argv, {}
    out, as_of, refs = [], None, {'c': None, 'c2': None}
    i = 0
    while i < len(argv):
        t = argv[i]
        if t == '--as-of':
            if i + 1 >= len(argv):
                raise ValueError('--as-of needs a map path')
            as_of = argv[i + 1]
            i += 2
            continue
        if t.startswith('--as-of='):
            as_of = t.split('=', 1)[1]
            i += 1
            continue
        if t in ('-c', '--circuit') and i + 1 < len(argv):
            refs['c'] = argv[i + 1]
        elif t.startswith('--circuit='):
            refs['c'] = t.split('=', 1)[1]
        elif t in ('-c2', '--circuit2') and i + 1 < len(argv):
            refs['c2'] = argv[i + 1]
        elif t.startswith('--circuit2='):
            refs['c2'] = t.split('=', 1)[1]
        out.append(t)
        i += 1
    res = {k: _Resolver(api, v, as_of) for k, v in refs.items()}
    addresses = {}
    final = []
    i = 0
    while i < len(out):
        t = out[i]
        flag, eq, val = t.partition('=')
        if flag in ADDR_FLAGS and (eq or i + 1 < len(out)):
            kind, which = ADDR_FLAGS[flag]
            raw = val if eq else out[i + 1]
            r = res[which]
            new = resolve_range(r, raw) if kind == 'range' else resolve_list(r, raw)
            if new != raw:
                addresses[flag] = f'{raw} -> {new}' + (f' (as of {as_of})' if as_of and ('op@' in raw or raw[:1].isdigit()) else '')
            final.extend([f'{flag}={new}'] if eq else [flag, new])
            i += 1 if eq else 2
            continue
        final.append(t)
        i += 1
    if as_of and not addresses:
        raise ValueError('--as-of given but no --ops/--ops2/--at address to translate')
    return final, addresses


# ---------------------------------------------------------------- `qhdl addr`
def v_addr(api, a):
    led = E.Ledger()
    c = api.resolve(a.circuit)
    r = _Resolver(api, a.circuit, a.as_of)
    text = a.address.strip()
    rows, facts = [], {}
    if text.startswith('tag:') and _TAG.match(text) and '+' not in text:
        h = r.tag(text)
        A, B = h['start'], h['end']
        facts = dict(path=h['path'], instance=h['k'], instances=len(r.instances()[h['path']]), ops=f'{A}:{B}',
                     length=B - A)
        claim = f"tag:{h['path']}[{h['k']}] is ops {A}:{B} ({B - A:,} ops, INFO_TAGs included) of {c['sha'][:12]}"
        target = A
    else:
        if ':' in text.replace('tag:', '', 1).replace('op@', '', 1).split(':', 1)[-1] and text.count(':') >= 2 \
                and not text.startswith('tag:'):
            raise api.QhdlError('addr takes one op address (tag:<path>[k]+off, op@<sha12>:<i>, @<i>) or a bare tag')
        target, _ = _point(r, text, 'op')
        claim = f'{text} is op {target} of {c["sha"][:12]}'
    led.earn('static', 'INFO_TAG instance table of the circuit bytes' + (' and the --as-of map chain'
                                                                          if a.as_of else ''))
    try:
        inst = r.instances()
    except api.QhdlError:
        inst = {}
    for p, k, s, e, off in innermost(inst, target):
        rows.append(dict(address=f'tag:{p}[{k}]+{off}', ops=f'{s}:{e}'))
    if rows:
        claim += f"; innermost tag form {rows[-1]['address']}"
    elif not inst:
        claim += '; the circuit has no INFO_TAG regions (absolute addresses only)'
    facts['op'] = target
    facts['absolute'] = f"op@{c['sha'][:12]}:{target}"
    return E.Envelope(led, claim, domain='op stream (static)', evidence=None,
                      rerun=api.cmd('addr', text, '-c', c['sha'][:12], *(['--as-of', a.as_of] if a.as_of else [])),
                      tool_sha=api.tool_sha(), circuit_sha=c['sha'], facts=facts, rows=rows,
                      next_cmds=[api.cmd('peak', '-c', c['sha'][:12])])


# ---------------------------------------------------------------- investigation container
def inv_dir(api, a) -> Path:
    v = getattr(a, 'dir', None) or os.environ.get('QHDL_INVESTIGATION')
    if not v:
        raise api.QhdlError('no investigation directory', 'pass it as the first argument or set QHDL_INVESTIGATION')
    p = Path(v)
    if not p.is_absolute() and not p.exists() and (api.ROOT / p).exists():
        p = api.ROOT / p
    return p


def load_inv(api, d: Path) -> dict:
    f = d / 'investigation.json'
    if not f.exists():
        raise api.QhdlError(f'{d} is not an investigation (no investigation.json)', 'create one with qhdl inv new')
    return json.loads(_retry(lambda: f.read_text(encoding='utf-8')))


def hyps(d: Path) -> list:
    """Current hypotheses (last snapshot per id), in creation order."""
    rows = read_jsonl(d / 'hypotheses.jsonl')
    order, cur = [], {}
    for r in rows:
        if r['id'] not in cur:
            order.append(r['id'])
        cur[r['id']] = r
    return [cur[i] for i in order]


def cands(d: Path) -> list:
    rows = read_jsonl(d / 'candidates.jsonl')
    order, cur = [], {}
    for r in rows:
        if r['sha'] not in cur:
            order.append(r['sha'])
        cur[r['sha']] = r
    return [cur[s] for s in order]


def _check_outcome_tier(t):
    if t is None:
        return None
    if t in EXTRA_TIERS:
        return t
    E.parse_tier(t)
    return t


def _short(s, n):
    s = ' '.join(str(s or '').split())
    return s if len(s) <= n else s[:n - 1] + '…'


def _env(api, led_basis, claim, d, rerun_parts, facts=None, rows=None, nxt=None, evidence=None):
    led = E.Ledger()
    led.earn('static', led_basis)
    env = E.Envelope(led, claim, domain=f'investigation {d.name} (files, not circuits)',
                     evidence=evidence or api.rel(d), rerun=api.cmd('inv', *rerun_parts),
                     tool_sha=api.tool_sha(), circuit_sha=None, facts=facts, rows=rows, next_cmds=nxt)
    env.stamp(provenance='derived')
    return env


def v_new(api, a):
    slug = re.sub(r'[^a-z0-9-]+', '-', a.slug.lower()).strip('-')
    if not slug:
        raise api.QhdlError('empty slug')
    base = Path(a.root) if a.root else api.ROOT / 'investigations'
    date = a.date or dt.date.today().isoformat()
    d = base / (slug if re.match(r'^\d{4}-\d{2}-\d{2}-', slug) else f'{date}-{slug}')
    if (d / 'investigation.json').exists():
        raise api.QhdlError(f'{api.rel(d)} already exists', 'resume it: qhdl inv resume ' + api.rel(d))
    bases = []
    for b in a.base or []:
        sha, _, label = b.partition('=')
        bases.append(dict(sha=sha, label=label))
    meta = dict(schema=SCHEMA, id=d.name, question=a.question, task=a.task, base=bases, card=a.card or '',
                budget=a.budget or '', stop_rule=a.stop_rule or '', outputs=a.outputs or [], created_utc=now_utc(),
                tool_sha=api.tool_sha())
    d.mkdir(parents=True, exist_ok=True)
    _retry(lambda: (d / 'investigation.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False) + '\n'))
    for f in ('hypotheses.jsonl', 'candidates.jsonl', 'trail.jsonl'):
        (d / f).touch()
    if not (d / 'notes.md').exists():
        (d / 'notes.md').write_text(f'# {d.name}\n\n{a.question}\n\nWhat I believe and why (<= 1 page):\n')
    return _env(api, 'wrote investigation.json and empty hypotheses/candidates/trail files',
                f'investigation {d.name} created; tool_sha {meta["tool_sha"]} pinned', d,
                ['new', a.slug, '--question', a.question, '--task', a.task],
                facts=dict(dir=api.rel(d)),
                nxt=[f'export QHDL_INVESTIGATION={api.rel(d)}',
                     api.cmd('inv', 'hypothesis', api.rel(d), '--id', 'H1', '--statement', '...', '--kill-if', '...')])


def _trail(d):
    return read_jsonl(d / 'trail.jsonl')


def v_hypothesis(api, a):
    d = inv_dir(api, a)
    meta = load_inv(api, d)
    cur = {h['id']: h for h in hyps(d)}
    if not a.kill_if or not a.kill_if.strip():
        raise api.QhdlError('--kill-if is required: write the kill criterion before the first probe')
    if a.id in cur and not a.supersedes:
        raise api.QhdlError(f'hypothesis {a.id} exists; kill_if is fixed once written',
                            'add a new id with --supersedes ' + a.id)
    if a.supersedes and a.supersedes not in cur:
        raise api.QhdlError(f'--supersedes {a.supersedes}: no such hypothesis')
    predicted = json.loads(a.predicted) if a.predicted else None
    row = dict(id=a.id, investigation=meta['id'], rev=1, at=a.at or now_utc(), statement=a.statement,
               mechanism=a.mechanism or '', predicted=predicted, kill_if=a.kill_if, reopen_if=a.reopen_if or '',
               circuit=a.circuit or '', next=a.next or '', probes=[], evidence=[], status='open', outcome_tier=None,
               outcome='', ledger=None, trail_mark=len(_trail(d)), supersedes=a.supersedes or None,
               keywords=a.keywords or '', family_target=a.family_target or '')
    append_jsonl(d / 'hypotheses.jsonl', row)
    if a.supersedes:
        old = dict(cur[a.supersedes])
        old.update(rev=old.get('rev', 1) + 1, at=row['at'], status='superseded', outcome=f'superseded by {a.id}')
        append_jsonl(d / 'hypotheses.jsonl', old)
    return _env(api, 'appended one hypothesis row', f'hypothesis {a.id} open; kill_if: {_short(a.kill_if, 160)}', d,
                ['hypothesis', api.rel(d), '--id', a.id], facts=dict(trail_mark=row['trail_mark']))


def _close(api, a, status):
    d = inv_dir(api, a)
    load_inv(api, d)
    cur = {h['id']: h for h in hyps(d)}
    if a.id not in cur:
        raise api.QhdlError(f'no hypothesis {a.id}', 'qhdl inv status ' + api.rel(d))
    h = dict(cur[a.id])
    if h['status'] in ('killed', 'confirmed', 'superseded') and status != 'parked' and not a.amend:
        raise api.QhdlError(f'{a.id} is already {h["status"]}', 'pass --amend to append a corrected outcome')
    tier = _check_outcome_tier(a.tier)
    trail = _trail(d)
    ids = {t.get('claim_id'): n for n, t in enumerate(trail) if t.get('claim_id')}
    for p in a.probe or []:
        if p not in ids:
            raise api.QhdlError(f'probe {p} is not a claim id in {d.name}/trail.jsonl',
                                'cite claim ids printed by verbs run with QHDL_INVESTIGATION set (or inv log rows)')
        if ids[p] < h.get('trail_mark', 0) and not a.prior_probe:
            raise api.QhdlError(f'probe {p} ran before {a.id} was written (kill_if must come first)',
                                'pass --prior-probe to cite it anyway; it is then marked as prior')
    if status in ('killed', 'confirmed') and not (a.outcome or '').strip():
        raise api.QhdlError('--outcome is required: the finding in one or two sentences')
    h.update(rev=h.get('rev', 1) + 1, at=a.at or now_utc(), status=status,
             outcome=a.outcome or h.get('outcome', ''), outcome_tier=tier or h.get('outcome_tier'),
             probes=list(h.get('probes') or []) + list(a.probe or []),
             evidence=list(h.get('evidence') or []) + list(a.evidence or []))
    if a.prior_probe and a.probe:
        h['prior_probes'] = list(a.probe)
    if getattr(a, 'next', None):
        h['next'] = a.next
    if getattr(a, 'reopen_if', None):
        h['reopen_if'] = a.reopen_if
    if getattr(a, 'key_numbers', None):
        h['key_numbers'] = a.key_numbers
    if status == 'killed':
        h['ledger'] = dict(kind='negative', outcome=a.ledger_outcome or 'kill', row=None)
    elif status == 'confirmed':
        h['ledger'] = dict(kind='lead', outcome='confirmed', row=None)
    append_jsonl(d / 'hypotheses.jsonl', h)
    return _env(api, 'appended one hypothesis row', f'{a.id} {status}: {_short(h["outcome"], 200)}', d,
                [status if status != 'killed' else 'kill', api.rel(d), a.id],
                facts=dict(outcome_tier=h['outcome_tier'], probes=len(h['probes']), evidence=len(h['evidence'])))


def v_kill(api, a):
    return _close(api, a, 'killed')


def v_confirm(api, a):
    return _close(api, a, 'confirmed')


def v_park(api, a):
    return _close(api, a, 'parked')


def v_candidate(api, a):
    d = inv_dir(api, a)
    load_inv(api, d)
    if a.hypothesis and a.hypothesis not in {h['id'] for h in hyps(d)}:
        raise api.QhdlError(f'no hypothesis {a.hypothesis}')
    if not re.match(r'^[0-9a-f]{12,64}$', a.sha):
        raise api.QhdlError('--sha must be 12..64 hex digits')
    scope = None
    if a.domain or a.extent or a.branches:
        scope = check_scope(dict(domain=a.domain or '', extent=a.extent or 'whole', branches=a.branches or ''))
    tier = _check_outcome_tier(a.tier)
    row = dict(sha=a.sha, at=a.at or now_utc(), parent=a.parent or '', hypothesis=a.hypothesis or '',
               kmb=a.kmb or '', map=a.map or '', status=a.status,
               certificate=dict(scope=scope, tier=tier, local=scope_is_local(scope) if scope else None,
                                metrics=json.loads(a.metrics) if a.metrics else None,
                                versus=json.loads(a.versus) if a.versus else None,
                                evidence=a.evidence or '', rerun=a.rerun or ''),
               next=a.next or '', note=a.note or '')
    append_jsonl(d / 'candidates.jsonl', row)
    return _env(api, 'appended one candidate row', f'candidate {a.sha[:12]} {a.status}'
                + (f"; scope {scope['domain']} / {scope['extent']} / {scope['branches']}" if scope else ''), d,
                ['candidate', api.rel(d), '--sha', a.sha[:12]])


def v_log(api, a):
    """Record a call that did not go through a qhdl verb (an agent's helper script, a gate run): one trail row."""
    d = inv_dir(api, a)
    load_inv(api, d)
    tier = _check_outcome_tier(a.tier)
    cid = claim_id('log', a.call, '', a.circuit or '')
    row = dict(at=a.at or now_utc(), claim_id=cid, verb='log', argv=[], rerun=a.call, tier=tier,
               provenance=a.provenance, scope=None, evidence=a.evidence or '', circuit_sha=a.circuit or None,
               tool_sha=None, seconds=None, exit=0, note=a.note or '')
    append_jsonl(d / 'trail.jsonl', row)
    return _env(api, 'appended one trail row', f'logged {cid}: {_short(a.call, 120)}', d,
                ['log', api.rel(d), '--call', a.call], facts=dict(claim_id=cid))


def _hyp_row(h, wide):
    r = dict(id=h['id'], status=h['status'])
    if h['status'] == 'open':
        r['statement'] = _short(h['statement'], 100 if wide else 70)
        r['kill_if'] = _short(h['kill_if'], 150 if wide else 90)
    else:
        r['outcome'] = _short(h.get('outcome') or h['statement'], 150 if wide else
                              (80 if h['status'] == 'confirmed' else 60))
        if h.get('outcome_tier'):
            r['outcome_tier'] = h['outcome_tier']
    return r


def _cand_row(c):
    cert = c.get('certificate') or {}
    sc = cert.get('scope') or {}
    r = dict(candidate=c['sha'][:12], status=c['status'], hypothesis=c.get('hypothesis', ''))
    if sc:
        r['scope'] = f"{sc.get('domain')}/{sc.get('extent')}/{sc.get('branches')}"
    if cert.get('tier'):
        r['cert_tier'] = cert['tier']
    if c.get('note'):
        r['note'] = _short(c['note'], 70)
    return r


def v_status(api, a):
    d = inv_dir(api, a)
    meta = load_inv(api, d)
    H, C = hyps(d), cands(d)
    counts = {s: sum(h['status'] == s for h in H) for s in STATUSES}
    rows = [_hyp_row(h, False) for h in H] + [_cand_row(c) for c in C]
    claim = (f"{meta['id']}: {len(H)} hypotheses (" + ', '.join(f'{v} {k}' for k, v in counts.items() if v)
             + f'); {len(C)} candidates; {len(_trail(d))} trail rows')
    return _env(api, 'read of investigation files', claim, d, ['status', api.rel(d)], rows=rows,
                nxt=[api.cmd('inv', 'resume', api.rel(d))])


def v_resume(api, a):
    """One bounded envelope a cold agent can act on: question, open hypotheses with kill_if, outcomes, candidates
    with certificate scope, last trail rows, tool_sha drift, and next = the command for the top open hypothesis."""
    d = inv_dir(api, a)
    meta = load_inv(api, d)
    H, C, T = hyps(d), cands(d), _trail(d)
    now_sha = api.tool_sha()
    open_h = [h for h in H if h['status'] == 'open']
    shut = ([h for h in H if h['status'] == 'confirmed'] + [h for h in H if h['status'] == 'killed']
            + [h for h in H if h['status'] == 'parked'])
    facts = dict(question=_short(meta.get('question'), 260), task=meta.get('task'),
                 base=[f"{b['sha'][:12]}" + (f" ({_short(b['label'], 24)})" if b.get('label') else '')
                       for b in meta.get('base', [])],
                 counts={s: sum(h['status'] == s for h in H) for s in STATUSES if any(h['status'] == s for h in H)})
    if meta.get('tool_sha') and meta['tool_sha'] != now_sha:
        facts['tool_sha_changed'] = f"{meta['tool_sha']} -> {now_sha} (re-run probes get new claim ids)"
    if meta.get('stop_rule'):
        facts['stop_rule'] = _short(meta['stop_rule'], 100)
    tail = [t for t in T if t.get('rerun')][-TRAIL_TAIL:]
    facts['trail_tail'] = [f"{t.get('claim_id') or '-'} {t.get('tier') or ''} "
                           f"{_short(re.sub(r'outputs/[^ ]*/helpers/', 'helpers/', strip_env(t['rerun'])), 60)}"
                           for t in tail]
    rows = ([_hyp_row(h, True) for h in open_h] + [_cand_row(c) for c in C]
            + [_hyp_row(h, False) for h in shut])
    by_claim = {t.get('claim_id'): t for t in T}
    nxt = []
    for h in open_h:
        last = next((by_claim[p]['rerun'] for p in reversed(h.get('probes') or []) if p in by_claim), None)
        cmd = h.get('next') or last
        if cmd and cmd not in nxt:
            nxt.append(cmd)
    for c in C:
        if c.get('status') in ('pending', 'under-verification') and c.get('next') and c['next'] not in nxt:
            nxt.append(c['next'])
    nxt.append(api.cmd('inv', 'status', api.rel(d)))
    claim = (f"{meta['id']}: {len(open_h)} open, " + ', '.join(f"{h['id']} {h['status']}" for h in shut)
             + (f"; {len(C)} candidates" if C else '') + '; prose in notes.md')
    env = _env(api, 'read of investigation files; each row carries the outcome tier it was recorded with', claim, d,
               ['resume', api.rel(d)], facts=facts, rows=rows, nxt=nxt[:4])
    env.max_bytes = RESUME_MAX_BYTES
    return env


TIER_TO_LEDGER = {'static': 'static model', 'sampled': 'measured by us', 'exact-all-x': 'measured by us',
                  'engine-parity': 'measured by us', 'official-local': 'official verifier',
                  'official-hosted': 'official verifier', 'structural': 'paper only', 'recorded': 'measured by us'}


def ledger_tier(t):
    if not t:
        return 'paper only'
    if t in TIER_TO_LEDGER:
        return TIER_TO_LEDGER[t]
    return TIER_TO_LEDGER[E.parse_tier(t)[0]]


def v_ledger_stage(api, a):
    d = inv_dir(api, a)
    meta = load_inv(api, d)
    H = hyps(d)
    task = {'dfthcsos': 'chemistry'}.get(meta.get('task'), meta.get('task') or 'all')
    nr, leads, rows = [], [], []
    for h in H:
        date = (h.get('at') or now_utc())[:10]
        if h['status'] == 'killed':
            e = dict(task=task, mechanism=_short(h.get('mechanism') or h['statement'], 200),
                     family_target=h.get('family_target') or ', '.join(b['sha'][:12] for b in meta.get('base', [])),
                     date_tested=date, source='ours', link='', evidence_tier=ledger_tier(h.get('outcome_tier')),
                     outcome=(h.get('ledger') or {}).get('outcome') or 'kill',
                     key_numbers=h.get('key_numbers', ''), why_it_failed=h.get('outcome', ''),
                     reopen_if=h.get('reopen_if') or 'not stated (add before staging)',
                     related_cards=';'.join(x for x in [meta.get('card'), f"investigations/{meta['id']}"] if x),
                     dataset_files='', keywords=h.get('keywords', ''), licence='Apache-2.0', supersedes='')
            nr.append(e)
            rows.append(dict(id=h['id'], kind='NR', tier=None, evidence_tier=e['evidence_tier'],
                             mechanism=_short(e['mechanism'], 70)))
        elif h['status'] == 'confirmed':
            leads.append(f"| — | {_short(h.get('mechanism') or h['statement'], 120)} | `{meta.get('card') or ''}` | "
                         f"(price) | — | (gate) | CONFIRMED {date} ({ledger_tier(h.get('outcome_tier'))}): "
                         f"{_short(h.get('outcome'), 200)} | {_short(h.get('next') or '', 120)} |")
            rows.append(dict(id=h['id'], kind='lead', evidence_tier=ledger_tier(h.get('outcome_tier')),
                             mechanism=_short(h.get('mechanism') or h['statement'], 70)))
    for r in rows:
        r.pop('tier', None)
    out = d / 'ledger-stage.json'
    _retry(lambda: out.write_text(json.dumps({'rows': nr}, indent=1, ensure_ascii=False) + '\n'))
    (d / 'leads-stage.md').write_text('# Leads-table row drafts (paste by hand into your leads table)\n\n'
                                      + '\n'.join(leads) + '\n')
    claim = (f'{len(nr)} negative-results rows and {len(leads)} lead drafts staged; a human adds them '
             f'(qhdl never writes docs/)')
    return _env(api, 'derived from hypotheses.jsonl; evidence_tier mapped by TIER_TO_LEDGER', claim, d,
                ['ledger-stage', api.rel(d)], rows=rows, evidence=api.rel(out),
                nxt=[f'python3 -m json.tool {api.rel(out)}'])


def register(sub, api):
    """Core verbs `inv` and `addr` (called from qhdl.build_parser)."""
    common = __import__('argparse').ArgumentParser(add_help=False)
    common.add_argument('--all', action='store_true', help='print every row')
    p = sub.add_parser('addr', parents=[common], help='resolve an address (tag:<path>[k]+off, op@<sha12>:<i> '
                                                      '[--as-of MAP]) and print its absolute and tag-relative forms')
    p.add_argument('address')
    p.add_argument('-c', '--circuit')
    p.add_argument('--as-of', dest='as_of', help='edit map(s), comma-separated, from the address circuit to -c')
    p.set_defaults(fn=lambda a: v_addr(api, a))

    p = sub.add_parser('inv', help='investigation container: new | hypothesis | kill | confirm | park | candidate | '
                                   'log | status | resume | ledger-stage')
    isub = p.add_subparsers(dest='inv_verb', metavar='<inv verb>')

    def dirarg(q):
        q.add_argument('dir', nargs='?', help='investigation directory (default $QHDL_INVESTIGATION)')
        q.add_argument('--at', help='timestamp override (ISO 8601 UTC), for reconstructed records')

    q = isub.add_parser('new', parents=[common], help='create investigations/<date>-<slug>/')
    q.add_argument('slug')
    q.add_argument('--question', required=True)
    q.add_argument('--task', required=True, choices=['invsqrt', 'dqi', 'dfthcsos', 'all'])
    q.add_argument('--base', action='append', metavar='SHA[=LABEL]')
    q.add_argument('--card', help='free-text reference to a plan or design note (stored, not read)')
    q.add_argument('--budget')
    q.add_argument('--stop-rule', dest='stop_rule')
    q.add_argument('--outputs', action='append', help='outputs/ directory holding the heavy artefacts')
    q.add_argument('--date', help='YYYY-MM-DD (default today)')
    q.add_argument('--root', help='parent directory (default investigations/)')
    q.set_defaults(fn=lambda a: v_new(api, a))

    q = isub.add_parser('hypothesis', aliases=['hyp'], parents=[common],
                        help='add a hypothesis; --kill-if is required and fixed once written')
    dirarg(q)
    q.add_argument('--id', required=True)
    q.add_argument('--statement', required=True)
    q.add_argument('--kill-if', dest='kill_if', required=True)
    q.add_argument('--mechanism')
    q.add_argument('--predicted', help='JSON, e.g. {"metric": "qubits", "delta": -1}')
    q.add_argument('--reopen-if', dest='reopen_if')
    q.add_argument('--circuit', help='circuit sha the hypothesis is about')
    q.add_argument('--next', help='the command a resuming agent should run next for this hypothesis')
    q.add_argument('--supersedes')
    q.add_argument('--keywords')
    q.add_argument('--family-target', dest='family_target')
    q.set_defaults(fn=lambda a: v_hypothesis(api, a))

    for name, fn, hlp in (('kill', v_kill, 'the kill criterion was met'),
                          ('confirm', v_confirm, 'the hypothesis held at the stated tier'),
                          ('park', v_park, 'set aside without an outcome')):
        q = isub.add_parser(name, parents=[common], help=hlp)
        dirarg(q)
        q.add_argument('id')
        q.add_argument('--outcome', help='the finding in one or two sentences')
        q.add_argument('--tier', help='outcome tier: static, sampled(n), exact-all-x, engine-parity, official-local, '
                                      'official-hosted, structural or recorded')
        q.add_argument('--probe', action='append', help='claim id from trail.jsonl that answered kill_if')
        q.add_argument('--evidence', action='append', help='evidence file path (outputs/...)')
        q.add_argument('--prior-probe', dest='prior_probe', action='store_true',
                       help='allow citing a probe that ran before the hypothesis was written')
        q.add_argument('--amend', action='store_true', help='append a corrected outcome to a closed hypothesis')
        q.add_argument('--reopen-if', dest='reopen_if')
        q.add_argument('--key-numbers', dest='key_numbers')
        q.add_argument('--next')
        q.add_argument('--ledger-outcome', dest='ledger_outcome', choices=LEDGER_OUTCOMES)
        q.set_defaults(fn=lambda a, fn=fn: fn(api, a))

    q = isub.add_parser('candidate', parents=[common], help='record a candidate and its certificate scope')
    dirarg(q)
    q.add_argument('--sha', required=True)
    q.add_argument('--parent')
    q.add_argument('--hypothesis')
    q.add_argument('--kmb')
    q.add_argument('--map')
    q.add_argument('--status', required=True, choices=['built', 'pending', 'under-verification', 'verified',
                                                       'failed', 'superseded'])
    q.add_argument('--domain', help='all-x | draws:<n>@<anchor> | inputs:<n>@<seed>')
    q.add_argument('--extent', help="whole | 'window A:B'")
    q.add_argument('--branches', help='zero | zero+random(<seed>) | all-outcomes | engine-sampled | official')
    q.add_argument('--tier')
    q.add_argument('--metrics', help='JSON')
    q.add_argument('--versus', help='JSON')
    q.add_argument('--evidence')
    q.add_argument('--rerun')
    q.add_argument('--next')
    q.add_argument('--note')
    q.set_defaults(fn=lambda a: v_candidate(api, a))

    q = isub.add_parser('log', parents=[common], help='record a non-qhdl call (helper script, gate) in the trail')
    dirarg(q)
    q.add_argument('--call', required=True, help='the command line, as a rerun string')
    q.add_argument('--tier')
    q.add_argument('--provenance', choices=['computed', 'recorded', 'derived'], default='recorded')
    q.add_argument('--evidence')
    q.add_argument('--circuit')
    q.add_argument('--note')
    q.set_defaults(fn=lambda a: v_log(api, a))

    for name, fn, hlp in (('status', v_status, 'one row per hypothesis and candidate'),
                          ('resume', v_resume, 'everything a cold agent needs, in one bounded envelope'),
                          ('ledger-stage', v_ledger_stage, 'draft negative-results rows (killed) and lead rows (confirmed)')):
        q = isub.add_parser(name, parents=[common], help=hlp)
        q.add_argument('dir', nargs='?', help='investigation directory (default $QHDL_INVESTIGATION)')
        q.set_defaults(fn=lambda a, fn=fn: fn(api, a))

    def no_inv_verb(a):
        raise api.QhdlError('qhdl inv needs a verb', 'new | hypothesis | kill | confirm | park | candidate | log | '
                                                     'status | resume | ledger-stage')
    p.set_defaults(fn=no_inv_verb)
