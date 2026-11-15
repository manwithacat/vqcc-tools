#!/usr/bin/env python3
"""qhdl: one agent-facing command line over the kickmix circuit-analysis tools.

    python3 scripts/qhdl.py <verb> [<noun> <address>] [-c CIRCUIT] [--all]

CIRCUIT is a .kmb/.kmx path or a SHA-256 prefix (>= 6 hex digits) of a circuit in the public-circuit dataset,
in an opened workspace, or produced by a qhdl verb (e.g. a recoloured candidate). Every verb prints ONE JSON
envelope:

    claim        the finding in one sentence
    tier         static | sampled(n) | exact-all-x | engine-parity | official-local | official-hosted
                 (lowest to highest; a verb never prints a tier above what it computed or checked)
    domain       what the claim was computed over (e.g. all 3,133,452 legal x, zero measurement branch)
    evidence     file with the full result;  rerun  the command that reproduces it
    tool_sha     hash of the tool code;  circuit_sha  full SHA-256 of the circuit
    basis        how the tier was earned;  facts / rows  key numbers (<= 20 rows, <= 2 KB; --all for every row)
    next         suggested follow-up commands

Addresses (stable for one circuit sha): op@<sha12>:<index>, wire q7, interval q7#2 (k-th live interval of q7),
run R4 (k-th maximal run of ops at the peak width, op order, from R0), cut @1200 (before op 1200),
stage <name>, tag <path>. Across edits and circuits (--ops, --ops2, --at; `qhdl addr` prints both forms):
tag:<path>[k]+off (k-th instance of a tag path, 0-based; stable across edits outside the region) and
op@<sha12>:<i> --as-of <edit .map.json>[,<map2>] (translated through the edit maps).

Identity and trail: every envelope also carries claim_id (sha12 of verb, rerun, tool_sha, circuit_sha) and
provenance (computed | recorded | derived); certify verbs add scope {domain, extent, branches}. With
QHDL_INVESTIGATION=<investigations/<date>-<slug>> set, every call is appended to its trail.jsonl; `qhdl inv` manages
the investigation (new, hypothesis --kill-if, kill, confirm, park, candidate, log, status, resume, ledger-stage).

Workspace: work/<sha12>/ (git-ignored; QHDL_WORK overrides the root): circuit.kmb (a copy for small circuits; large
ones are referenced by path and SHA-256 in meta.json), canonical circuit.kmx, ops.tsv (index, kind, operands, bit,
cond, depth, tag path, stage), intervals.tsv, profile.npy (live width per op), stages.json, meta.json, and the
evidence files of each verb. Derived results are cached by (circuit sha, tool sha, args).

Exact (exact-all-x) analysis is available for invsqrt circuits (enumerable domain, <= 64 wires for the
information test). Circuits over 2,000,000 ops (DQI) get no op table: use peak, exec and evidence on them.
A claim that mixes tiers lists them in `parts` (one tier per part).
"""
# Wraps qhdl_bitsim, qhdl_idle_wires, qhdl_cheap_function, qhdl_width_census, qhdl_kmb, qhdl_lifetime_allocator and
# local_verify_reference without changing their maths.
from __future__ import annotations

import os
import sys

if not sys.flags.isolated:  # circuits are untrusted data: always run isolated (-I)
    os.execv(sys.executable, [sys.executable, '-I', os.path.abspath(__file__)] + sys.argv[1:])

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import qhdl_envelope as E  # noqa: E402

try:  # QHDL v1 Phase 0 (identity, trail, investigations, tag/op@ addresses); the core verbs work without it
    import qhdl_investigation as INV  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    INV = None
    sys.stderr.write(f'qhdl: investigation layer not loaded: {_exc}\n')

import vqcc_config as CFG  # noqa: E402  (paths and interpreters: environment variables, see scripts/vqcc_config.py)

WORK = CFG.WORK
DATASET = CFG.DATASET
MANIFEST = DATASET / 'manifest.csv'
OUTPUTS = CFG.OUTPUTS
VIEWER_DIR = CFG.VIEWER_DIR
VIEWER_URL = 'https://manwithacat.github.io/kickmix-macos/viewer/'
KMPY = CFG.KMPY
OFFICIAL_ROOT = CFG.OFFICIAL_ROOT
MAX_OPS_TABLE = 2_000_000
MAX_X = 3_133_452
MAX_OPS_TABLE = int(os.environ.get('QHDL_MAX_OPS_TABLE', MAX_OPS_TABLE))  # test hook: force the large-circuit path
REF_BYTES = int(os.environ.get('QHDL_REF_BYTES', 32 << 20))  # open references, not copies, files larger than this
TOOL_FILES = ['qhdl.py', 'qhdl_envelope.py', 'qhdl_bitsim.py', 'qhdl_idle_wires.py', 'qhdl_cheap_function.py',
              'qhdl_width_census.py', 'qhdl_kmb.py', 'qhdl_lifetime_allocator.py', 'local_verify_reference.py', 'vqcc_config.py',
              'qhdl_peak.py', 'qhdl_exec.py', 'qhdl_window.py', 'qhdl_track.py', 'qhdl_edit.py', 'qhdl_census.py',
              'qhdl_doctor.py', 'qhdl_simulate.py', 'qhdl_rewrite.py', 'qhdl_rewrite_phase_fold.py', 'qhdl_optimizations.py',
              'qhdl_predicates.py', 'qhdl_avenues.py', 'qhdl_diff.py', 'qhdl_retain.py', 'qhdl_measure_out.py']
# Optional verb modules: define qhdl_register(subparsers, api); a missing or broken one is skipped. (qhdl_surgery is
# not one: it needs the kickmix engine venv and prints its own envelopes; run scripts/qhdl_surgery.py directly.)
PLUGINS = ('qhdl_window', 'qhdl_track', 'qhdl_edit', 'qhdl_census', 'qhdl_doctor', 'qhdl_simulate', 'qhdl_rewrite', 'qhdl_optimizations',
           'qhdl_measure_first', 'qhdl_schedule', 'qhdl_valuegraph', 'qhdl_recompute', 'qhdl_valuealign', 'qhdl_parts', 'qhdl_avenues', 'qhdl_diff', 'qhdl_retain', 'qhdl_measure_out')
WORKBENCH = CFG.WORKBENCH
EXEC_SEED = 1
INVSQRT_SHOTS = MAX_X * 64
DQI_SHOTS = 102_400


class QhdlError(Exception):
    def __init__(self, msg, hint=None):
        super().__init__(msg)
        self.hint = hint


_TOOL_SHA = None


def tool_sha():
    global _TOOL_SHA
    if _TOOL_SHA is None:
        _TOOL_SHA = E.files_sha([HERE / f for f in TOOL_FILES if (HERE / f).exists()])
    return _TOOL_SHA


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def cmd(*parts):
    pre = f"QHDL_WORK={shlex.quote(os.environ['QHDL_WORK'])} " if os.environ.get('QHDL_WORK') else ''
    return pre + shlex.quote(sys.executable) + ' ' + shlex.quote(str(HERE / 'qhdl.py')) + ' ' + ' '.join(shlex.quote(str(p)) for p in parts if p is not None and p != '')


def fmt_int(v):
    return f'{int(v):,}'


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp',
                                     delete=False) as f:
        tmp = Path(f.name)
        f.write(json.dumps(obj, indent=1, default=str) + '\n')
    tmp.replace(path)


def write_tsv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                     dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as f:
        tmp = Path(f.name)
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)


def cached(path, key):
    """Return the cached result at path if its cache key matches, else None."""
    p = Path(path)
    if p.exists():
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        if d.get('cache_key') == key:
            return d
    return None


# ---------------------------------------------------------------- circuit resolution
def manifest_rows():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline='') as f:
        return list(csv.DictReader(f))


def candidates_index():
    p = WORK / '_index' / 'candidates.json'
    return json.loads(p.read_text()) if p.exists() else {}


def register_candidate(sha, path, parent, how):
    idx = candidates_index()
    idx[sha] = dict(path=rel(path), parent=parent, how=how)
    write_json(WORK / '_index' / 'candidates.json', dict(sorted(idx.items())))


def resolve(ref):
    """-> dict(sha, path, source, manifest_row or None). Hash-checks every file it returns."""
    if ref is None:
        raise QhdlError('no circuit given', 'pass -c CIRCUIT (a .kmb path or a sha prefix; `qhdl circuits` lists '
                                            'the public ones)')
    p = Path(ref)
    if p.exists() and p.is_file():
        sha = E.file_sha(p)
        row = next((r for r in manifest_rows() if r['sha256'] == sha), None)
        return dict(sha=sha, path=p.resolve(), source='file', manifest=row)
    pre = ref.lower()
    if pre.startswith('op@'):
        pre = pre[3:].split(':')[0]
    if not (len(pre) >= 6 and all(c in '0123456789abcdef' for c in pre)):
        raise QhdlError(f'circuit {ref!r} is neither a file nor a sha prefix (>= 6 hex digits)')
    hits = {}
    for r in manifest_rows():
        if r['sha256'].startswith(pre):
            hits[r['sha256']] = ('manifest', DATASET / 'circuits' / r['task'] / f"{r['sha12']}.kmb", r)
    for sha, c in candidates_index().items():
        if sha.startswith(pre) and sha not in hits:
            hits[sha] = ('candidate', ROOT / c['path'], None)
    if WORK.exists():
        for m in sorted(WORK.glob('*/meta.json')):
            d = json.loads(m.read_text())
            if d['sha256'].startswith(pre) and d['sha256'] not in hits:
                hits[d['sha256']] = ('workspace', circuit_path(d, m.parent), None)
    if not hits:
        raise QhdlError(f'no circuit with sha prefix {ref}', 'see `qhdl circuits` or pass a file path')
    if len(hits) > 1:
        raise QhdlError(f'sha prefix {ref} is ambiguous: ' + ', '.join(sorted(h[:12] for h in hits)))
    sha, (src, path, row) = next(iter(hits.items()))
    if not path.exists():
        raise QhdlError(f'circuit file for {sha[:12]} not found at {rel(path)}')
    got = E.file_sha(path)
    if got != sha:
        raise QhdlError(f'hash mismatch for {rel(path)}: expected {sha[:12]}, file is {got[:12]}')
    if row is None:
        row = next((r for r in manifest_rows() if r['sha256'] == sha), None)
    return dict(sha=sha, path=path, source=src, manifest=row)


# ---------------------------------------------------------------- workspace
def ws_dir(sha):
    return WORK / sha[:12]


def circuit_path(meta, ws=None):
    """The workspace's circuit file: its copy, or the referenced original (large circuits; meta circuit_ref)."""
    ref = meta.get('circuit_ref')
    if ref:
        p = Path(ref)
        return p if p.is_absolute() else ROOT / p
    return (ws or ws_dir(meta['sha256'])) / meta['circuit_file']


def to_kmx(kmb, kmx):
    code = ('import kickmix as km, sys; c = km.Circuit.from_file(sys.argv[1], format="auto"); '
            'c.to_file(sys.argv[2], format="kmx")')
    try:
        subprocess.run([KMPY, '-I', '-c', code, str(kmb), str(kmx)], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as e:
        tail = (getattr(e, 'stderr', '') or str(e)).strip().splitlines()[-1:] or ['?']
        raise QhdlError(f'cannot convert {Path(kmb).name} to .kmx with {KMPY}: {tail[0][:200]}',
                        'opening a .kmb needs kickmix: set QHDL_KICKMIX_PYTHON to a Python that can `import kickmix` '
                        '(or pass a .kmx file)') from None


def tag_paths(ops):
    """Per op: INFO_TAG path ('' outside tags) and top-level region bounds [(name, start, end_exclusive)]."""
    import qhdl_bitsim as B
    stack, paths, regions = [], [], []
    for i, o in enumerate(ops):
        if o.code == B.C_INFO:
            txt = o.text.split(None, 1)[1].strip().strip('"') if len(o.text.split(None, 1)) > 1 else ''
            kind, _, name = txt.partition(':')
            if kind == '@enter':
                if not stack:
                    regions.append([name, i, None])
                stack.append(name)
            elif kind == '@exit' and stack:
                stack.pop()
                if not stack and regions:
                    regions[-1][2] = i + 1
        paths.append('/'.join(stack))
    return paths, [(n, s, e if e is not None else len(ops)) for n, s, e in regions]


def make_stages(sha, ops, regions):
    """Stage map: viewer stages.json for the viewer's reference circuit, else top-level INFO_TAG regions (gaps
    named 'untagged'), else ten equal slices seg0..seg9."""
    n = len(ops)
    vops = VIEWER_DIR / 'ops.json'
    if vops.exists():
        try:
            src = json.loads(vops.read_text()).get('source_kmb_sha256')
        except json.JSONDecodeError:
            src = None
        if src == sha:
            st = json.loads((VIEWER_DIR / 'stages.json').read_text())['stages']
            return 'viewer stages.json (documented/inferred, boundaries good to about +-100 ops)', [
                dict(name=s['id'], start=int(s['start']), end=int(s['end']), title=s.get('title', '')) for s in st]
    if regions:
        out, pos, seen = [], 0, {}
        for name, s, e in regions:
            if s > pos:
                out.append(['untagged', pos, s])
            out.append([name, s, e])
            pos = e
        if pos < n:
            out.append(['untagged', pos, n])
        stages = []
        for name, s, e in out:
            seen[name] = seen.get(name, 0) + 1
            stages.append(dict(name=name if seen[name] == 1 else f'{name}.{seen[name]}', start=s, end=e, title=''))
        return 'top-level INFO_TAG regions', stages
    k = 10
    return 'equal slices (no tags, no stage map)', [
        dict(name=f'seg{j}', start=n * j // k, end=n * (j + 1) // k, title='') for j in range(k)]


def stage_of(stages, i):
    for s in stages:
        if s['start'] <= i < s['end']:
            return s['name']
    return ''


def ensure_open(ref, ledger=None):
    """Open (or reuse) the workspace for a circuit; returns meta dict."""
    c = resolve(ref)
    d = ws_dir(c['sha'])
    m = d / 'meta.json'
    if m.exists():
        meta = json.loads(m.read_text())
        if meta.get('open_tool_sha') == tool_sha() and meta.get('sha256') == c['sha']:
            meta['_reused'] = True
            return meta
    return open_workspace(c)


def open_workspace(c):
    import numpy as np
    import qhdl_bitsim as B
    import qhdl_idle_wires as I
    sha, src = c['sha'], Path(c['path'])
    d = ws_dir(sha)
    d.mkdir(parents=True, exist_ok=True)
    is_kmx = src.suffix == '.kmx'
    meta = dict(sha256=sha, sha12=sha[:12], source=rel(src), source_kind=c['source'], open_tool_sha=tool_sha())
    if src.stat().st_size > REF_BYTES and src.resolve().parent != d.resolve():
        circ = src  # reference by path + SHA-256 (resolve() re-hashes it on every use); no copy
        meta['circuit_ref'] = rel(src)
        meta['circuit_file'] = None
    else:
        circ = d / ('circuit.kmx' if is_kmx else 'circuit.kmb')
        if not circ.exists() or E.file_sha(circ) != sha:
            shutil.copyfile(src, circ)
        meta['circuit_file'] = circ.name
    row = c['manifest']
    if row:
        meta['provenance'] = {k: row.get(k) for k in ('task', 'author', 'source_notebook_url', 'source_notebook_version',
                                                      'notebook_title', 'licence', 'local_task_score',
                                                      'local_score_tier')}
    cands = candidates_index()
    if sha in cands:
        meta['derived_from'] = cands[sha]
    if not is_kmx:
        import qhdl_kmb as K
        k = K.read(str(circ))
        meta['declared_qubits'] = int(k.num_qubits)
        meta['num_ops'] = int(len(k.op_types))
        meta['registers'] = {n: len(cs) for n, cs in k.registers}
        del k
    if meta.get('num_ops', 0) > MAX_OPS_TABLE:
        meta['ops_table'] = f'op tables skipped: {fmt_int(meta["num_ops"])} ops > {fmt_int(MAX_OPS_TABLE)}'
        meta['task'] = (row or {}).get('task') or 'unknown'
        write_json(d / 'meta.json', meta)
        return meta
    kmx = d / 'circuit.kmx'
    if not is_kmx and not kmx.exists():
        to_kmx(circ, kmx)
    ops, header, xq = B.parse_kmx(str(kmx))
    nq_used = 1 + max(max(o.q) for o in ops if o.q)
    nq = max(nq_used, meta.get('declared_qubits', 0))
    meta.setdefault('declared_qubits', nq_used)
    meta['num_ops'] = len(ops)
    meta['kmx_sha256'] = E.file_sha(kmx)
    meta['x_qubits'] = xq
    reg_names = [h.split()[2].strip('"') for h in header if h.startswith('REGISTER')]
    task = (row or {}).get('task')
    if task is None:
        task = 'invsqrt' if (len(xq) == 22 and reg_names == ['x']) else 'unknown'
    meta['task'] = task
    depth, touches, intervals, prof, reg_iv = I.structure(ops, xq, nq, with_regs=True)
    peak = int(prof.max())
    meta['peak_width'] = peak
    meta['peak_ops'] = int((prof >= peak).sum())
    meta['peak_runs'] = len(E.number_runs(prof.tolist(), peak))
    paths, regions = tag_paths(ops)
    stage_src, stages = make_stages(sha, ops, regions)
    meta['stage_source'] = stage_src
    meta['stages'] = len(stages)
    write_json(d / 'stages.json', dict(source=stage_src, stages=stages))
    st_of = [''] * len(ops)
    for s in stages:
        for i in range(s['start'], min(s['end'], len(ops))):
            st_of[i] = s['name']
    rows = []
    for i, o in enumerate(ops):
        kind = B.KINDS[o.code] + ('_IF' if o.cond >= 0 else '')
        rows.append([i, kind, ','.join(f'q{q}' for q in o.q), f'b{o.bit}' if o.bit >= 0 else '',
                     f'b{o.cond}' if o.cond >= 0 else '', int(depth[i]), paths[i], st_of[i]])
    write_tsv(d / 'ops.tsv', ['index', 'kind', 'operands', 'bit', 'cond', 'depth', 'tag', 'stage'], rows)
    allv = [(w, s, e, tl, 'scratch') for w, s, e, tl in intervals] + [(w, s, e, tl, 'register') for w, s, e, tl in reg_iv]
    num = E.number_intervals(allv)
    ivrows = sorted([[f'q{w}#{num[(w, s)]}', w, role, s, e, len(tl)] for w, s, e, tl, role in allv],
                    key=lambda r: (r[1], r[3]))
    write_tsv(d / 'intervals.tsv', ['address', 'wire', 'role', 'start', 'end', 'touches'], ivrows)
    np.save(d / 'profile.npy', prof.astype(np.int16))
    meta['files'] = ['circuit.kmx', 'ops.tsv', 'intervals.tsv', 'profile.npy', 'stages.json']
    write_json(d / 'meta.json', meta)
    return meta


class Loaded:
    """In-memory structure of an opened (enumerable-size) circuit."""

    def __init__(self, meta):
        import qhdl_bitsim as B
        import qhdl_idle_wires as I
        if 'ops_table' in meta:
            raise QhdlError(f"{meta['sha12']}: {meta['ops_table']}",
                            f"on this circuit use peak, exec and evidence (e.g. {cmd('peak', '-c', meta['sha12'])})")
        self.meta = meta
        self.dir = ws_dir(meta['sha256'])
        self.ops, self.header, self.xq = B.parse_kmx(str(self.dir / 'circuit.kmx'))
        self.nq = max(1 + max(max(o.q) for o in self.ops if o.q), meta['declared_qubits'])
        self.depth, self.touches, self.intervals, self.prof, self.reg_iv = I.structure(
            self.ops, self.xq, self.nq, with_regs=True)
        self.peak = int(self.prof.max())
        self.stages = json.loads((self.dir / 'stages.json').read_text())['stages']
        allv = [(w, s, e, tl) for w, s, e, tl in self.intervals] + [(w, s, e, tl) for w, s, e, tl in self.reg_iv]
        self.ivnum = E.number_intervals(allv)

    def level(self, spec):
        if spec in (None, 'peak'):
            return self.peak
        if spec.startswith('peak-'):
            return self.peak - int(spec[5:])
        return int(spec)

    def runs(self, level):
        return E.number_runs(self.prof.tolist(), level)

    def stage(self, i):
        return stage_of(self.stages, i)

    def interval_of(self, w, a):
        for iv in self.intervals + self.reg_iv:
            if iv[0] == w and iv[1] <= a <= iv[2]:
                return f'q{w}#{self.ivnum[(iv[0], iv[1])]}'
        return f'q{w}#?'

    def enumerable(self):
        return self.meta.get('task') == 'invsqrt' and self.nq <= 64


def gaps_for(L, level, registers=False):
    """Idle gaps (qhdl_idle_wires.idle_gaps) closing >= 1 run at `level`, with run ids and interval addresses."""
    import numpy as np
    import qhdl_idle_wires as I
    runs = L.runs(level)
    prof = np.minimum(L.prof, level)  # idle_gaps uses prof.max() as the level
    ivs = list(L.intervals) + (list(L.reg_iv) if registers else [])
    gaps = I.idle_gaps(L.ops, ivs, prof, runs, L.depth) if runs else []
    regs = set(L.xq)
    out = []
    for g in gaps:
        rc = g['runs_closed']
        out.append(dict(wire=f"q{g['wire']}", role='register' if g['wire'] in regs else 'scratch',
                        interval=L.interval_of(g['wire'], g['a']), idle=f"{g['a'] + 1}..{g['b'] - 1}",
                        runs_closed=len(rc), runs=f'R{min(rc)}..R{max(rc)}', peak_ops_closed=g['peak_ops_closed'],
                        next_op=g['next_touch'], dead_tail=g['dead_tail'], _a=g['a'], _b=g['b'], _runs=rc))
    out.sort(key=lambda r: (-r['runs_closed'], -r['peak_ops_closed'], int(r['wire'][1:]), r['_a']))
    return runs, out


def strip_private(rows):
    return [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]


# ---------------------------------------------------------------- verbs
def v_open(a):
    led = E.Ledger()
    if a.circuit and a.circuit_pos and a.circuit != a.circuit_pos:
        raise QhdlError(f'two circuits given ({a.circuit_pos}, -c {a.circuit}); pass one')
    meta = ensure_open(a.circuit or a.circuit_pos)
    led.earn('static', 'workspace built from the circuit file (hash-checked)')
    sha = meta['sha256']
    d = ws_dir(sha)
    nrep = len(reports_for(sha))
    prov = meta.get('provenance') or {}
    # No author in the claim (blind tests read it); credits are printed only with --credits.
    who = f" (derived: {meta['derived_from']['how']})" if meta.get('derived_from') else ''
    if 'ops_table' in meta:
        claim = (f"opened {meta['sha12']} ({meta['task']}{who}): {meta['declared_qubits']} qubits declared, "
                 f"{fmt_int(meta['num_ops'])} ops; too large for op tables: use peak (width) and exec (Toffolis)")
    else:
        claim = (f"opened {meta['sha12']} ({meta['task']}{who}): {meta['declared_qubits']} qubits declared, "
                 f"{fmt_int(meta['num_ops'])} ops; live-width peak {meta['peak_width']} over "
                 f"{fmt_int(meta['peak_ops'])} ops in {meta['peak_runs']} runs; {meta['stages']} stages "
                 f"({meta['stage_source'].split(' (')[0]})")
    facts = dict(workspace=rel(d), files=meta.get('files', []), recorded_verifier_reports=nrep)
    if getattr(a, 'credits', False) and prov:
        facts['credits'] = {k: prov.get(k) for k in ('author', 'source_notebook_url', 'source_notebook_version',
                                                     'licence') if prov.get(k)}
    elif prov.get('author'):
        facts['credits'] = 'public circuit with recorded provenance: pass --credits to print author and source'
    if meta.get('circuit_ref'):
        facts['circuit'] = f"{meta['circuit_ref']} (referenced, sha256 {meta['sha12']}..., not copied)"
    if meta.get('declared_qubits') and meta.get('peak_width') and meta['peak_width'] < meta['declared_qubits']:
        facts['note'] = (f"declared {meta['declared_qubits']} > live-width peak {meta['peak_width']}: "
                         'a lifetime recolouring may remove qubits (see `recolour`)')
    if 'ops_table' in meta:  # only verbs that work on a large circuit
        nxt = [cmd('peak', '-c', meta['sha12']), cmd('exec', '-c', meta['sha12'])]
        if circuit_path(meta).suffix == '.kmb':
            facts['workbench'] = (f"static Toffolis per INFO_TAG region (kickmix-macos fork): {WORKBENCH} summarize "
                                  f"{rel(circuit_path(meta))} --out <report.json>; then {WORKBENCH} query <report.json> "
                                  '--depth 0 --sort toffoli')
    else:
        nxt = [cmd('profile', '-c', meta['sha12']), cmd('runs', '-c', meta['sha12']), cmd('idle', '-c', meta['sha12']),
               cmd('exec', '-c', meta['sha12'])]
    if nrep:
        nxt.append(cmd('evidence', '-c', meta['sha12']))
    return E.Envelope(led, claim, domain='circuit file (static structure)', evidence=rel(d / 'meta.json'),
                      rerun=cmd('open', '-c', meta['sha12']), tool_sha=tool_sha(), circuit_sha=sha, facts=facts,
                      next_cmds=nxt)


def v_profile(a):
    import qhdl_bitsim as B
    led = E.Ledger()
    meta = ensure_open(a.circuit)
    L = Loaded(meta)
    led.earn('static', 'syntactic live-width profile (allocator rule, register gaps) and static gate counts')
    toff = {B.C_CCX, B.C_CCZ, B.C_CSWAP}
    rows = []
    for s in L.stages:
        seg = L.prof[s['start']:s['end']]
        if len(seg) == 0:
            continue
        ops = L.ops[s['start']:s['end']]
        rows.append(dict(stage=s['name'], ops=f"{s['start']}..{s['end'] - 1}", max_width=int(seg.max()),
                         ops_at_peak=int((seg >= L.peak).sum()), static_toffoli=sum(1 for o in ops if o.code in toff),
                         hmr=sum(1 for o in ops if o.code == B.C_HMR)))
    tot_t = sum(r['static_toffoli'] for r in rows)
    main = max(rows, key=lambda r: r['ops_at_peak'])
    claim = (f"{meta['sha12']}: {meta['declared_qubits']} qubits declared, live-width peak {L.peak} on "
             f"{fmt_int(meta['peak_ops'])} of {fmt_int(meta['num_ops'])} ops; stage {main['stage']} holds the most "
             f"peak ops ({fmt_int(main['ops_at_peak'])}); {fmt_int(tot_t)} static Toffoli-like gates")
    facts = dict(declared_qubits=meta['declared_qubits'], peak_width=L.peak, stage_source=meta['stage_source'])
    nxt = [cmd('runs', '-c', meta['sha12']), cmd('idle', '-c', meta['sha12'])]
    if L.peak < meta['declared_qubits']:
        facts['recolour_hint'] = (f"live-width peak {L.peak} < declared {meta['declared_qubits']}: recolouring by "
                                  'lifetime could save qubits (static bound)')
        nxt.insert(0, cmd('recolour', '-c', meta['sha12']))
    out = L.dir / 'profile_stages.tsv'
    write_tsv(out, list(rows[0]), [list(r.values()) for r in rows])
    return E.Envelope(led, claim, domain='op stream (static; widths are syntactic, register gaps honoured)',
                      evidence=rel(out), rerun=cmd('profile', '-c', meta['sha12']), tool_sha=tool_sha(),
                      circuit_sha=meta['sha256'], facts=facts, rows=rows, next_cmds=nxt)


def allocator_profile(meta, register_gaps):
    """qhdl_lifetime_allocator live profile of the circuit's .kmb (register_gaps=False: registers always live, the
    `peak` model), or None when only a .kmx is available."""
    import qhdl_kmb as K
    import qhdl_lifetime_allocator as A
    src = circuit_path(meta)
    if src.suffix != '.kmb':
        return None
    k = K.read(str(src))
    is_reset, _ = A.reset_ops(k)
    ops, qs, order, _ = A.references(k)
    iv = A.build_intervals(k, ops, qs, order, is_reset)
    return A.live_profile(iv, len(k.op_types), A.register_qubits(k), register_gaps=register_gaps)


def _level_spec(spec, peak):
    if spec in (None, 'peak'):
        return peak
    if str(spec).startswith('peak-'):
        return peak - int(spec[5:])
    return int(spec)


def _runs_of(prof, level):
    import numpy as np
    idx = np.flatnonzero(np.asarray(prof) >= level)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return list(zip(idx[np.r_[0, cuts]].tolist(), idx[np.r_[cuts - 1, len(idx) - 1]].tolist()))


def v_runs(a):
    led = E.Ledger()
    meta = ensure_open(a.circuit)
    L = Loaded(meta)
    lmod = getattr(a, 'liveness', None) or 'reset'
    other = None
    alt = allocator_profile(meta, register_gaps=False)
    if lmod == 'allocator':
        if alt is None:
            raise QhdlError('--liveness allocator needs a .kmb circuit', f"use {cmd('peak', '-c', meta['sha12'])}")
        peak_ = int(alt.max())
        lvl = _level_spec(a.level, peak_)
        runs = _runs_of(alt, lvl)
        oruns = L.runs(L.level(a.level))
        other = dict(liveness='reset', peak=L.peak, runs=len(oruns), ops_at_level=sum(e - s + 1 for s, e in oruns))
        peak_shown = peak_
    else:
        lvl = L.level(a.level)
        runs = L.runs(lvl)
        peak_shown = L.peak
        if alt is not None:
            apk = int(alt.max())
            oruns = _runs_of(alt, _level_spec(a.level, apk))
            other = dict(liveness='allocator', peak=apk, runs=len(oruns), ops_at_level=sum(e - s + 1 for s, e in oruns))
    led.earn('static', 'maximal op runs at width >= level on the syntactic live-width profile (' + (
        'reset liveness: a register reset mid-circuit is free until its next touch; `peak` uses allocator liveness, '
        'registers always live' if lmod == 'reset' else 'allocator liveness: registers always live, as `peak`') + ')')
    rows, by = [], {}
    for k, (s, e) in enumerate(runs):
        st = L.stage(s)
        rows.append(dict(run=f'R{k}', start=s, end=e, ops=e - s + 1, stage=st))
        b = by.setdefault(st, [0, 0, k, k])
        b[0] += 1
        b[1] += e - s + 1
        b[3] = k
    top = sorted(by.items(), key=lambda kv: (-kv[1][1], kv[0]))
    by_stage = {st: f'{v[0]} runs R{v[2]}..R{v[3]}, {fmt_int(v[1])} ops' for st, v in top[:6]}
    tot = sum(r['ops'] for r in rows)
    claim = (f"{len(runs)} runs at width >= {lvl} (peak {peak_shown}, {lmod} liveness) covering {fmt_int(tot)} ops; "
             + (f"largest block: stage {top[0][0]} ({by_stage[top[0][0]]})" if top else 'none'))
    if other:
        claim += (f"; under {other['liveness']} liveness {other['runs']} runs, {fmt_int(other['ops_at_level'])} ops "
                  '(run numbers differ between models)')
    out = L.dir / (f'runs_w{lvl}.tsv' if lmod == 'reset' else f'runs_w{lvl}_{lmod}.tsv')
    write_tsv(out, ['run', 'start', 'end', 'ops', 'stage'], [list(r.values()) for r in rows])
    nxt = [cmd('idle', '-c', meta['sha12']) + ('' if a.level is None else f' --level {a.level}')]
    if rows:
        nxt.append(cmd('explain', 'run', top and rows[by[top[0][0]][2]]['run'], '-c', meta['sha12']))
    return E.Envelope(led, claim, domain='op stream (static)', evidence=rel(out),
                      rerun=cmd('runs', '-c', meta['sha12'], *(['--level', a.level] if a.level else []),
                                *(['--liveness', lmod] if lmod != 'reset' else [])),
                      tool_sha=tool_sha(), circuit_sha=meta['sha256'],
                      facts=dict(by_stage=by_stage, liveness=lmod, other_liveness=other),
                      rows=rows, next_cmds=nxt)


def v_idle(a):
    led = E.Ledger()
    meta = ensure_open(a.circuit)
    L = Loaded(meta)
    lvl = L.level(a.level)
    runs, gaps = gaps_for(L, lvl, a.registers)
    led.earn('static', 'idle gaps between consecutive touches inside a live interval (qhdl_idle_wires)')
    gaps = [g for g in gaps if g['runs_closed'] >= a.min_runs]
    rows = strip_private(gaps)
    out = L.dir / f"idle_w{lvl}{'_regs' if a.registers else ''}.tsv"
    write_tsv(out, list(rows[0]) if rows else ['wire'], [list(r.values()) for r in rows])
    covered = set()
    for g in gaps:
        covered.update(g['_runs'])
    if rows:
        t = rows[0]
        claim = (f"{len(rows)} idle gaps span runs at width {lvl}; {len(covered)} of {len(runs)} runs have one. "
                 f"Longest: {t['wire']} ({t['role']}) idle ops {t['idle']} across "
                 f"{t['runs_closed']} runs ({fmt_int(t['peak_ops_closed'])} peak ops). Idle is necessary, not "
                 'sufficient: freeing needs its value recomputable (explain wire)')
    else:
        claim = f'no idle gap spans a run at width {lvl}'
    nxt = [cmd('explain', 'wire', r['wire'], '-c', meta['sha12']) for r in rows[:2]]
    return E.Envelope(led, claim, domain='op stream (static, ' + ('scratch and register wires' if a.registers
                                                                    else 'scratch wires; --registers adds x wires') + ')',
                      evidence=rel(out), rerun=cmd('idle', '-c', meta['sha12'], *(['--level', a.level] if a.level else []),
                                                   '--registers' if a.registers else None),
                      tool_sha=tool_sha(), circuit_sha=meta['sha256'], rows=rows, next_cmds=nxt)


def describe_value(words, n):
    """Exact description of a wire's value as a function of x over all legal x (rows x = 1..n)."""
    import numpy as np
    import qhdl_bitsim as B
    v = B.unpack_bits(words, n).astype(bool)
    ones = int(v.sum())
    if ones == 0:
        return '0 for every x'
    if ones == n:
        return '1 for every x'

    def pw(k):
        return f'2^{k.bit_length() - 1}' if k & (k - 1) == 0 else fmt_int(k)

    def interval(mask):
        idx = np.flatnonzero(mask)
        if len(idx) and idx[-1] - idx[0] + 1 == len(idx):
            lo, hi = int(idx[0]) + 1, int(idx[-1]) + 1
            if lo == 1:
                return f'[x < {pw(hi + 1)}]'
            if hi == n:
                return f'[x >= {pw(lo)}]'
            return f'[{fmt_int(lo)} <= x <= {fmt_int(hi)}]'
        return None
    s = interval(v)
    if s:
        return s
    s = interval(~v)
    if s:
        return 'NOT ' + s
    x = np.arange(1, n + 1, dtype=np.int64)
    for k in range(22):
        b = ((x >> k) & 1).astype(bool)
        if np.array_equal(b, v):
            return f'x bit {k}'
        if np.array_equal(~b, v):
            return f'NOT x bit {k}'
    return f'not a threshold or single bit of x (1 on {fmt_int(ones)} of {fmt_int(n)} x)'


def info_test(snap, w, n, regs):
    """Exact: (conflicting states of the other wires, of the register wires only). 0 means determined."""
    import qhdl_cheap_function as C
    det, conf = C.determined(snap, w, n)
    others = set(range(snap.shape[0])) - set(regs)
    _, conf_reg = C.determined_target(snap, snap[w], n, exclude=others | {w})
    return conf, conf_reg


def cheap_recompute(L, w, a, r, ledger):
    """Affine/quadratic recompute search on a sample (qhdl_cheap_function.analyse_sample), exact re-check on all x."""
    import numpy as np
    import qhdl_cheap_function as C
    import qhdl_width_census as WC
    rows = WC.sample_rows(4000)
    n = len(rows)
    snaps, _ = C.snapshot_values(L.ops, L.xq, L.nq, rows, [a + 1, r], 'zero')
    vals = {q: C.to_int(snaps[r][q], n) for q in range(L.nq)}
    f = C.analyse_sample(C.to_int(snaps[a + 1][w], n), vals, w, n)
    ledger.earn(f'sampled({n})', f'affine/quadratic search on {n} legal x (seeded)')
    if f['cost'] is None:
        return dict(kind=f['kind'], cost=None, terms=None, exact=None, sample=n)
    full, sim = C.snapshot_values(L.ops, L.xq, L.nq, None, [a + 1, r], 'zero')
    got = C.eval_terms([tuple(t) for t in f['terms']], full[r], sim.W) & sim.valid
    ok = bool(np.array_equal(full[a + 1][w] & sim.valid, got))
    if ok:
        ledger.earn('exact-all-x', 'recompute expression re-checked on all legal x (zero branch)')
    return dict(kind=f['kind'], cost=f['cost'], terms=[list(t) for t in f['terms']], exact=ok, sample=n)


def v_explain(a):
    if a.noun == 'wire':
        return explain_wire(a)
    if a.noun == 'run':
        return explain_run(a)
    raise QhdlError(f'explain {a.noun}: P0 supports `explain wire qN` and `explain run Rk`')


def explain_wire(a):
    import qhdl_cheap_function as C
    led = E.Ledger()
    addr = E.parse_address(a.address, 'wire')
    if addr['kind'] not in ('wire', 'interval'):
        raise QhdlError('explain wire takes qN or qN#k')
    w = addr['wire']
    meta = ensure_open(a.circuit)
    L = Loaded(meta)
    if w >= L.nq:
        raise QhdlError(f'q{w} does not exist ({L.nq} wires)')
    lvl = L.level(a.level)
    runs, gaps = gaps_for(L, lvl, registers=True)
    led.earn('static', 'touches, live intervals and idle gaps from the op stream')
    mine = [g for g in gaps if g['wire'] == f'q{w}']
    if addr['kind'] == 'interval':
        mine = [g for g in mine if g['interval'] == E.format_address(addr)]
    if a.at is not None:
        mine = [g for g in mine if g['_a'] < a.at < g['_b']]
    role = 'register x bit %d' % L.xq.index(w) if w in L.xq else 'scratch'
    ntouch = len(L.touches[w])
    nint = sum(1 for iv in L.intervals + L.reg_iv if iv[0] == w)
    key = dict(tool_sha=tool_sha(), verb='explain wire', wire=w, level=lvl, sel=E.format_address(addr), at=a.at)
    ev = L.dir / 'explain' / f"wire_q{w}{'_' + str(addr.get('k')) if addr['kind'] == 'interval' else ''}" \
                             f"{'_at' + str(a.at) if a.at is not None else ''}_w{lvl}.json"
    rr = cmd('explain', 'wire', E.format_address(addr), '-c', meta['sha12'], *(['--at', a.at] if a.at is not None else []),
             *(['--level', a.level] if a.level else []))
    base_facts = dict(role=role, touches=ntouch, intervals=nint, gaps_spanning_runs=len(mine))
    if not mine:
        claim = (f"q{w} ({role}) is not idle across any run at width {lvl}: every such run touches it or it is not "
                 'live there, so parking it cannot lower this peak')
        write_json(ev, dict(cache_key=key, claim=claim, facts=base_facts))
        return E.Envelope(led, claim, domain='op stream (static)', evidence=rel(ev), rerun=rr, tool_sha=tool_sha(),
                          circuit_sha=meta['sha256'], facts=base_facts)
    g = mine[0]
    ga, gb = g['_a'], g['_b']
    first_run, last_run = min(g['_runs']), max(g['_runs'])
    stages = sorted({L.stage(runs[k][0]) for k in g['_runs']}, key=lambda s: [x['name'] for x in L.stages].index(s))
    span = (f"idle over ops {ga + 1}..{gb - 1} ({g['interval']}), spanning {g['runs_closed']} runs at width {lvl} "
            f"(R{first_run}..R{last_run}, {fmt_int(g['peak_ops_closed'])} peak ops; stages {', '.join(stages[:3])})")
    if not L.enumerable():
        claim = f"q{w} ({role}) {span}; value analysis needs an enumerable domain (invsqrt, <= 64 wires) in P0"
        write_json(ev, dict(cache_key=key, claim=claim, gap=strip_private([g])[0]))
        return E.Envelope(led, claim, domain='op stream (static)', evidence=rel(ev), rerun=rr, tool_sha=tool_sha(),
                          circuit_sha=meta['sha256'], facts=base_facts)
    hit = cached(ev, key)
    if hit is None:
        rpts = sorted({gb, runs[last_run][1] + 1})
        snaps, sim = C.snapshot_values(L.ops, L.xq, L.nq, None, sorted({ga + 1, *rpts}), 'zero')
        value = describe_value(snaps[ga + 1][w], sim.n)
        tests = []
        for r in rpts:
            conf, conf_reg = info_test(snaps[r], w, sim.n, L.xq)
            tests.append(dict(at=f'@{r}', op=L.ops[r].text if r < len(L.ops) else 'end', conflicting_states=conf,
                              conflicting_states_register_only=conf_reg))
        del snaps
        res = dict(cache_key=key, wire=f'q{w}', role=role, gap=strip_private([g])[0], stages=stages, value=value,
                   info_tests=tests)
        if all(t['conflicting_states'] > 0 for t in tests):
            res['verdict'] = 'information-bearing'
        else:
            r_ok = next(int(t['at'][1:]) for t in tests if t['conflicting_states'] == 0)
            sub = E.Ledger()
            res['recompute'] = cheap_recompute(L, w, ga, r_ok, sub)
            res['recompute_ledger'] = sub.entries
            res['verdict'] = 'determined'
        write_json(ev, res)
    else:
        res = hit
    led.earn('exact-all-x', f'bitset simulation of all {fmt_int(MAX_X)} legal x, zero measurement branch '
                            '(qhdl_bitsim; information test qhdl_cheap_function.determined)')
    for t, b in res.get('recompute_ledger', []):
        led.earn(t, b)
    tests = res['info_tests']
    t0 = tests[-1] if res['verdict'] == 'information-bearing' else tests[0]
    if res['verdict'] == 'information-bearing':
        claim = (f"q{w} = {res['value']} ({role}), {span}. Information-bearing: at {t0['at']} "
                 f"{fmt_int(t0['conflicting_states'])} states of the other {L.nq - 1} wires occur with both q{w}=0 "
                 f"and q{w}=1, so no recompute of any degree exists; freeing it needs an encoding change")
        tier = 'exact-all-x'
    else:
        rc = res['recompute']
        if rc['cost'] is not None and rc['exact']:
            claim = (f"q{w} = {res['value']} ({role}), {span}. Determined by the other wires; {rc['kind']} recompute "
                     f"costs {rc['cost']} Toffoli (exact on all x): parkable at that price")
            tier = 'exact-all-x'
        elif rc['cost'] is not None:
            claim = (f"q{w} = {res['value']} ({role}), {span}. Determined (exact), but the {rc['kind']} recompute "
                     'found on the sample FAILED the all-x re-check')
            tier = 'exact-all-x'
        else:
            claim = (f"q{w} = {res['value']} ({role}), {span}. Determined by the other wires (exact), but no affine "
                     f"or quadratic recompute found on a {rc['sample']}-x sample (search result: {rc['kind']})")
            tier = 'exact-all-x'
    rows = [dict(at=t['at'], conflicting_states=t['conflicting_states'],
                 register_only=t['conflicting_states_register_only'], op=t['op']) for t in tests]
    facts = dict(base_facts, value=res['value'], verdict=res['verdict'])
    parts = [dict(part=f"q{w}'s value and the information test", tier='exact-all-x')]
    rc = res.get('recompute')
    if rc:
        if rc['cost'] is not None and rc['exact']:
            parts.append(dict(part=f"{rc['kind']} recompute, re-checked on all x", tier='exact-all-x'))
        else:
            parts.append(dict(part=f"affine/quadratic recompute search ({rc['kind']})", tier=f"sampled({rc['sample']})"))
    nxt = [cmd('explain', 'run', f'R{first_run}', '-c', meta['sha12'], *(['--level', a.level] if a.level else []))]
    return E.Envelope(led, claim, tier=tier,
                      domain=f'all {fmt_int(MAX_X)} legal x (1..{fmt_int(MAX_X)}), zero measurement branch',
                      evidence=rel(ev), rerun=rr, tool_sha=tool_sha(), circuit_sha=meta['sha256'], facts=facts,
                      rows=rows, parts=parts, next_cmds=nxt)


def explain_run(a):
    import numpy as np
    import qhdl_bitsim as B
    led = E.Ledger()
    addr = E.parse_address(a.address, 'run')
    if addr['kind'] != 'run':
        raise QhdlError('explain run takes Rk')
    meta = ensure_open(a.circuit)
    L = Loaded(meta)
    lvl = L.level(a.level)
    runs = L.runs(lvl)
    k = addr['run']
    if not 0 <= k < len(runs):
        raise QhdlError(f'R{k} does not exist at width {lvl} ({len(runs)} runs: R0..R{len(runs) - 1})')
    s, e = runs[k]
    _, gaps = gaps_for(L, lvl, registers=True)
    led.earn('static', 'live intervals and idle gaps over the run')
    cands = sorted([g for g in gaps if k in g['_runs']], key=lambda g: (g['role'] != 'scratch', int(g['wire'][1:])))
    live = sorted({iv[0] for iv in L.intervals + L.reg_iv if iv[1] <= s and iv[2] >= e})
    touched = sorted({q for i in range(s, e + 1) for q in L.ops[i].q})
    key = dict(tool_sha=tool_sha(), verb='explain run', run=k, level=lvl)
    ev = L.dir / 'explain' / f'run_R{k}_w{lvl}.json'
    rr = cmd('explain', 'run', f'R{k}', '-c', meta['sha12'], *(['--level', a.level] if a.level else []))
    rows = [dict(wire=g['wire'], role=g['role'], interval=g['interval'], idle=f"{g['_a'] + 1}..{g['_b'] - 1}")
            for g in cands]
    tier_domain = 'op stream (static)'
    if L.enumerable() and cands:
        hit = cached(ev, key)
        if hit is None:
            r = e + 1
            keys = {}

            def cb(i, sim):
                kk = np.zeros(sim.n, np.uint64)
                for q in range(L.nq):
                    kk |= B.unpack_bits(sim.q[q], sim.n).astype(np.uint64) << np.uint64(q)
                keys['K'] = kk
            sim = B.BitSim(L.ops, num_qubits=L.nq, x_qubits=L.xq, outcome='zero')
            sim.run(stop=r, snaps=[r], on_snap=cb)
            K = np.unique(keys.pop('K'))
            verdicts = []
            for g in cands:
                w = int(g['wire'][1:])
                flip = K ^ np.uint64(1 << w)
                pos = np.searchsorted(K, flip)
                pos[pos >= len(K)] = 0
                conf = int((K[pos] == flip).sum()) // 2
                verdicts.append(conf)
            write_json(ev, dict(cache_key=key, at=f'@{r}', conflicting_states=verdicts,
                                wires=[g['wire'] for g in cands], rows=rows))
            hit = json.loads(ev.read_text())
        led.earn('exact-all-x', f'information test at @{e + 1} on all {fmt_int(MAX_X)} legal x, zero branch')
        tier_domain = f'all {fmt_int(MAX_X)} legal x, zero measurement branch (verdicts); op stream (rest)'
        for row, conf in zip(rows, hit['conflicting_states']):
            row['verdict'] = 'information-bearing' if conf else 'determined'
            row['conflicting_states'] = conf
        ninfo = sum(1 for c in hit['conflicting_states'] if c)
        verdict = (f"{ninfo} information-bearing, {len(rows) - ninfo} determined (recompute may exist; "
                   'price with explain wire)')
    else:
        verdict = 'value analysis unavailable (needs invsqrt, <= 64 wires)' if cands else ''
    claim = (f"R{k} ops {s}..{e} ({e - s + 1} ops, stage {L.stage(s)}) at width >= {lvl}: {len(live)} wires live "
             f"throughout, {len(touched)} touched inside; {len(cands)} idle across it"
             + (f': {verdict}' if verdict else ' (none can be parked: the run is blocked by genuine use)'))
    if not cands:
        write_json(ev, dict(cache_key=key, claim=claim))
    return E.Envelope(led, claim, domain=tier_domain, evidence=rel(ev), rerun=rr, tool_sha=tool_sha(),
                      circuit_sha=meta['sha256'], rows=rows,
                      next_cmds=[cmd('explain', 'wire', r_['wire'], '-c', meta['sha12']) for r_ in rows[:2]])


# ---------------------------------------------------------------- evidence (recorded verifier reports)
def report_index():
    """Index of local official-verifier reports under outputs/ and work/: {circuit_sha: [entries]}."""
    idx_path = WORK / '_index' / 'reports.json'
    old = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    files = sorted(list(OUTPUTS.glob('**/report*.json')) + list(WORK.glob('*/verify/**/report.json')))
    new = {}
    for p in files:
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size > 2_000_000:
            continue
        k = rel(p)
        sig = f'{st.st_size}:{int(st.st_mtime)}'
        if k in old and old[k]['sig'] == sig:
            new[k] = old[k]
            continue
        entries = []
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            d = None
        if isinstance(d, dict) and isinstance(d.get('results'), dict) and 'kickmix_module' in d:
            settings = d.get('settings', {})
            overrides = {kk: vv for kk, vv in settings.items() if 'shots' in kk and vv is not None}
            for task, res in sorted(d['results'].items()):
                summ = res.get('summary') if isinstance(res, dict) else None
                if not isinstance(summ, dict) or not summ.get('circuit_sha256'):
                    continue
                shots = summ.get('verified_shots')
                full = (not overrides) and (task != 'invsqrt' or shots == INVSQRT_SHOTS) and \
                       (task != 'dqi' or shots in (None, DQI_SHOTS) or (shots or 0) >= DQI_SHOTS)
                entries.append(dict(sha=summ['circuit_sha256'], task=task, status=summ.get('status'),
                                    score=summ.get('score'), num_qubits=summ.get('num_qubits'),
                                    salt=settings.get('salt'), verified_shots=shots, full_settings=bool(full),
                                    platform=d.get('platform', '')))
        new[k] = dict(sig=sig, entries=entries)
    write_json(idx_path, new)
    by = {}
    for k, v in new.items():
        for en in v['entries']:
            by.setdefault(en['sha'], []).append(dict(en, report=k))
    return by


def reports_for(sha):
    return sorted(report_index().get(sha, []), key=lambda r: r['report'])


def evidence_rows(sha, ledger):
    reps = reports_for(sha)
    rows = []
    for r in reps:
        tier = 'official-local' if r['full_settings'] and 'macOS' in r['platform'] else 'static'
        basis = (f"recorded local official verifier report {r['report']} (its circuit_sha256 is {sha[:12]}; "
                 f"salt {r['salt']!r}, {fmt_int(r['verified_shots'] or 0)} shots; not re-run)")
        ledger.earn(tier, basis)
        rows.append(dict(status=r['status'], score=round(r['score'], 2) if isinstance(r['score'], float) else r['score'],
                         Q=r['num_qubits'], task=r['task'], salt=r['salt'], full_settings=r['full_settings'],
                         tier=tier, report=r['report']))
    return rows


def v_evidence(a):
    led = E.Ledger()
    c = resolve(a.circuit)
    sha = c['sha']
    led.earn('static', 'scan of recorded verifier reports (outputs/**/report*.json, work/*/verify/)')
    rows = evidence_rows(sha, led)
    passes = [r for r in rows if r['status'] == 'PASS' and r['tier'] == 'official-local']
    if passes:
        b = passes[0]
        claim = (f"{sha[:12]}: official verifier PASS locally (recorded), score {b['score']}, {b['Q']} qubits; "
                 f"{len(rows)} recorded report(s)")
        tier = 'official-local'
    elif rows:
        claim = f"{sha[:12]}: {len(rows)} recorded report(s), none a full-settings local PASS (statuses: " + \
                ', '.join(sorted({str(r['status']) for r in rows})) + ')'
        tier = led.best()
    else:
        claim = f'{sha[:12]}: no recorded official verifier report found'
        tier = 'static'
    out = ws_dir(sha) / 'evidence.json'
    write_json(out, dict(circuit_sha=sha, reports=rows))
    return E.Envelope(led, claim, tier=tier, domain='recorded official verifier runs (not re-run by this command)',
                      evidence=rel(out), rerun=cmd('evidence', '-c', sha[:12]), tool_sha=tool_sha(), circuit_sha=sha,
                      rows=rows, next_cmds=[] if passes else [cmd('verify', '-c', sha[:12])])


# ---------------------------------------------------------------- recolour (lifetime allocator)
def v_recolour(a):
    led = E.Ledger()
    meta = ensure_open(a.circuit)
    sha = meta['sha256']
    src = circuit_path(meta)
    if src.suffix != '.kmb':
        raise QhdlError('recolour needs a .kmb circuit')
    tag = 'recolour' + ('' if a.register_gaps else '_noregaps')
    cdir = ws_dir(sha) / 'cand' / tag
    rep_path = cdir / 'allocator_report.json'
    out = cdir / 'circuit.kmb'
    if rep_path.exists() and out.exists():
        rep = json.loads(rep_path.read_text())
        if rep.get('output_sha256') != E.file_sha(out):
            raise QhdlError(f'{rel(out)} does not match its report; refusing to overwrite (move it away first)')
    else:
        cdir.mkdir(parents=True, exist_ok=True)
        tmp = cdir / 'circuit.kmb.tmp'
        args = [sys.executable, '-I', str(HERE / 'qhdl_lifetime_allocator.py'), str(src), str(tmp)]
        if a.register_gaps:
            args.append('--register-gaps')
        p = subprocess.run(args, capture_output=True, text=True)
        if p.returncode != 0:
            write_json(cdir / 'allocator_failure.json', dict(stderr=p.stderr[-4000:]))
            raise QhdlError('allocator failed (renaming equivalence or build error)', rel(cdir / 'allocator_failure.json'))
        rep = json.loads(p.stdout)
        tmp.replace(out)
        rep['output'] = rel(out)
        rep['input'] = rel(src)
        rep['output_sha256'] = E.file_sha(out)
        write_json(rep_path, rep)
    new_sha = rep['output_sha256']
    eq = rep.get('equivalence', {}).get('status')
    if eq != 'EQUIVALENT':
        raise QhdlError(f'renaming check did not pass ({eq}); candidate kept at {rel(out)}')
    register_candidate(new_sha, out, parent=sha, how=f"lifetime recolouring of {sha[:12]}"
                                                     f"{' (register gaps)' if a.register_gaps else ''}")
    led.earn('static', 'lifetime allocator recolouring; qhdl_lifetime_allocator.check_equivalence: output = input '
                       'under a lifetime-respecting qubit renaming (gate sequence unchanged)')
    erows = evidence_rows(new_sha, led)
    passes = [r for r in erows if r['status'] == 'PASS' and r['tier'] == 'official-local']
    qin, qout = rep['num_qubits_in'], rep['num_qubits_out']
    claim = (f"{sha[:12]} recolours {qin} -> {qout} qubits (lower bound {rep['lower_bound']}, "
             f"{'optimal' if rep.get('optimal') else 'not optimal'}), gate sequence unchanged; candidate "
             f"{new_sha[:12]}. ")
    if passes:
        b = passes[0]
        claim += f"Official verifier: PASS locally (recorded report), score {b['score']}, {b['Q']} qubits"
        tier = 'official-local'
    elif erows:
        claim += 'Recorded verifier reports exist but none is a full-settings local PASS'
        tier = 'static'
    else:
        claim += 'Not official-verified yet (no recorded report); run verify'
        tier = 'static'
    facts = dict(candidate=rel(out), candidate_sha=new_sha, wire_switches=rep['equivalence'].get('wire_switches'),
                 renaming_check=eq)
    prow = (meta.get('provenance') or {})
    if prow.get('local_task_score'):
        facts['parent_score_manifest'] = f"{float(prow['local_task_score']):.2f} ({prow.get('local_score_tier')}, " \
                                         'manifest value, not re-checked)'
    return E.Envelope(led, claim, tier=tier,
                      domain='circuit structure (renaming proof); verifier tier from recorded reports for the candidate sha',
                      evidence=rel(rep_path), rerun=cmd('recolour', '-c', sha[:12],
                                                        None if a.register_gaps else '--no-register-gaps'),
                      tool_sha=tool_sha(), circuit_sha=sha, facts=facts, rows=erows,
                      next_cmds=[cmd('evidence', '-c', new_sha[:12])] + ([] if passes else
                                                                          [cmd('verify', '-c', new_sha[:12])]))


# ---------------------------------------------------------------- verify (local official verifier)
def v_verify(a):
    led = E.Ledger()
    c = resolve(a.circuit)
    sha = c['sha']
    task = a.task or (c['manifest'] or {}).get('task')
    if task is None:
        task = ensure_open(a.circuit).get('task')
    if task not in ('invsqrt', 'dqi'):
        raise QhdlError(f'verify supports invsqrt and dqi circuits (task here: {task}); pass --task')
    vdir = ws_dir(sha) / 'verify' / f"{task}_salt-{a.salt or 'empty'}"
    report = vdir / 'report.json'
    fresh = False
    if not report.exists():
        sub = vdir / 'submission'
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c['path'], sub / f'{task}.kmb')
        if E.file_sha(sub / f'{task}.kmb') != sha:
            raise QhdlError('copy hash mismatch')
        args = [KMPY, '-I', str(HERE / 'local_verify_reference.py'), '--root', str(OFFICIAL_ROOT),
                '--submission', str(sub), '--out', str(report), '--tasks', task, '--salt', a.salt]
        if task == 'invsqrt':
            args += ['--invsqrt-workers', str(a.workers)]
        else:
            args += ['--workers', str(min(a.workers, 6))]
        with open(vdir / 'log.txt', 'w') as log:
            p = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0 or not report.exists():
            raise QhdlError(f'verifier run failed (exit {p.returncode}); log kept at {rel(vdir / "log.txt")}')
        fresh = True
    d = json.loads(report.read_text())
    summ = d['results'][task]['summary']
    if summ.get('circuit_sha256') != sha:
        raise QhdlError('report circuit sha does not match')
    led.earn('official-local', ('ran' if fresh else 'recorded run of') + ' the unmodified official verifier with the '
             f"local kickmix build (salt {a.salt!r}); report {rel(report)}")
    claim = (f"{sha[:12]} ({task}): official verifier {summ['status']} locally, score {summ.get('score')}, "
             f"{summ.get('num_qubits')} qubits, Toffoli-eq {summ.get('toffoli_equivalent')}"
             + ('' if fresh else ' (existing run reused)'))
    facts = {k: summ.get(k) for k in ('status', 'score', 'num_qubits', 'toffoli_equivalent', 'verified_shots',
                                      'worst_error_over_tolerance') if k in summ}
    return E.Envelope(led, claim, domain=f"official verifier, salt {a.salt!r}, official shot counts",
                      evidence=rel(report), rerun=cmd('verify', '-c', sha[:12], *(['--salt', a.salt] if a.salt else [])),
                      tool_sha=tool_sha(), circuit_sha=sha, facts=facts)


# ---------------------------------------------------------------- link (viewer deep link string)
def v_link(a):
    led = E.Ledger()
    meta = ensure_open(a.circuit)
    led.earn('static', 'deep-link string assembled from addresses (viewer not modified)')
    op = a.op
    run = None
    stage = a.stage
    if a.run:
        L = Loaded(meta)
        k = E.parse_address(a.run, 'run')['run']
        runs = L.runs(L.peak)
        if not 0 <= k < len(runs):
            raise QhdlError(f'R{k} does not exist')
        run = f'R{k}'
        if op is None:
            op = runs[k][0]
    if stage is None and op is not None and (ws_dir(meta['sha256']) / 'stages.json').exists():
        stage = stage_of(json.loads((ws_dir(meta['sha256']) / 'stages.json').read_text())['stages'], op) or None
    wires = ','.join(str(E.parse_address(w, 'wire')['wire']) for w in a.wires.split(',')) if a.wires else None
    parts = [f"c={meta['sha12']}"] + [f'{k}={v}' for k, v in (('op', op), ('x', a.x), ('wires', wires), ('run', run),
                                                              ('stage', stage)) if v is not None]
    url = VIEWER_URL + '#' + '&'.join(parts)
    is_ref = 'viewer' in meta.get('stage_source', '')
    today = (VIEWER_URL + '#' + stage) if (is_ref and stage) else None
    claim = f'viewer deep link for {meta["sha12"]}' + (f' at op {op}' if op is not None else '')
    facts = dict(url=url, works_today=today or 'no: the viewer currently reads only #<stage-id>, and only for its '
                                                 'bundled circuit 0766a111a76f',
                 note='parameters other than the stage id are the P1 deep-link format (viewer change not in P0)')
    return E.Envelope(led, claim, domain='address formatting only', evidence=None,
                      rerun=cmd('link', '-c', meta['sha12'], *(['--op', a.op] if a.op is not None else []),
                                *(['--x', a.x] if a.x is not None else []), *(['--wires', a.wires] if a.wires else []),
                                *(['--run', a.run] if a.run else []), *(['--stage', a.stage] if a.stage else [])),
                      tool_sha=tool_sha(), circuit_sha=meta['sha256'], facts=facts)


# ---------------------------------------------------------------- peak (any size, INFO_TAG regions)
PEAK_SHOW = ('regions', 'paths', 'runs', 'idle')


def v_peak(a):
    import time
    import qhdl_peak as P
    led = E.Ledger()
    meta = ensure_open(a.circuit)
    sha, s12 = meta['sha256'], meta['sha12']
    src = circuit_path(meta)
    if src.suffix != '.kmb':
        raise QhdlError('peak reads .kmb circuits', f"for a .kmx circuit use {cmd('profile', '-c', s12)} and runs")
    lvl = a.level or 'peak'
    lmod = getattr(a, 'liveness', None) or 'allocator'
    sfx = '' if lmod == 'allocator' else f'_{lmod}'
    d = ws_dir(sha) / 'peak'
    key = dict(tool_sha=tool_sha(), verb='peak', level=lvl, liveness=lmod)
    ev = d / f'peak_{lvl}{sfx}.json'
    res = cached(ev, key)
    if res is None:
        t0 = time.time()
        r = P.analyse(src, a.level, liveness=lmod)
        if r['input_sha256'] != sha:
            raise QhdlError(f'{rel(src)} changed while being read (sha {r["input_sha256"][:12]})')
        tabs = {}
        for name, rows_ in (('regions', r['regions']), ('paths', r['paths']), ('runs', r['run_rows']),
                            ('idle', r['idle'])):
            p = d / f'peak_{lvl}{sfx}_{name}.tsv'
            write_tsv(p, list(rows_[0]) if rows_ else ['none'], [list(x.values()) for x in rows_])
            tabs[name] = rel(p)
        res = dict(cache_key=key, seconds=round(time.time() - t0, 1), tables=tabs,
                   **{k: v for k, v in r.items() if k not in ('paths', 'run_rows', 'idle')},
                   idle_scratch=[x for x in r['idle'] if x['role'] == 'scratch'],
                   idle_register_wires=len({x['wire'] for x in r['idle'] if x['role'] != 'scratch'}),
                   idle_all=r['idle'], run_rows=r['run_rows'], paths=r['paths'])
        write_json(ev, res)
    led.earn('static', 'live intervals as in qhdl_lifetime_allocator (' + (
        'registers always live' if lmod == 'allocator' else 'a register reset mid-circuit is free until its next '
        'touch, as in `qhdl runs`') + '; scratch ends at an unconditional R/HMR or last touch); INFO_TAG regions')
    show = a.show
    if show == 'regions':
        rows = res['regions']
    elif show == 'paths':
        rows = res['paths']
    elif show == 'runs':
        rows = res['run_rows']
    else:
        rows = res['idle_all'] if a.registers else res['idle_scratch']
    rows = [{k: v for k, v in r_.items() if k != 'static_toffolis'} if show == 'regions' else r_ for r_ in rows]
    sc = res['idle_scratch']
    peak, lv = res['peak'], res['level']
    at = f'at width >= {lv} (peak {peak})' if lv != peak else f'at the peak width {peak}'
    rbp = '; '.join(f'{p} {n}' for p, n in res['runs_by_path'][:2])
    oth = res.get('other_liveness') or {}
    wid = (f"{res['registers']} registers + {peak - res['registers']} scratch" if lmod == 'allocator'
           else 'reset liveness: registers free between a reset and their next touch')
    claim = (f"{s12}: peak {peak} live wires ({wid}); "
             f"{fmt_int(res['ops_at_level'])} of {fmt_int(res['num_ops'])} ops in {res['runs']} runs {at}, by tag "
             f"path: {rbp or 'none'}. ")
    if oth:
        claim += (f"Under {oth['liveness']} liveness: peak {oth['peak']}, {oth['runs']} runs, "
                  f"{fmt_int(oth['ops_at_level'])} ops (run numbers differ between models). ")
    if sc:
        claim += (f"{res['runs_with_idle_scratch']} of {res['runs']} runs have an idle scratch wire (longest "
                  f"{sc[0]['wire']}, {sc[0]['runs_closed']} runs {sc[0]['runs']}); {res['idle_register_wires']} "
                  'register wires idle too. Idle is necessary, not sufficient (value must be recomputable)')
    else:
        claim += f"No scratch wire is idle across a run; {res['idle_register_wires']} register wires are"
    an = res.get('anatomy') or {}
    facts = dict(declared_qubits=res['num_qubits'], liveness=lmod, other_liveness=oth or None,
                 first_peak_op=an.get('first_op'), scratch_there_by_start_path=dict(an.get('scratch_by_start', [])[:3]))
    nxt = [cmd('peak', '-c', s12, '--show', x, *(['--level', a.level] if a.level else [])) for x in PEAK_SHOW
           if x != show][:2]
    nxt += [cmd('exec', '-c', s12)]
    if lv == peak:
        nxt.append(cmd('peak', '-c', s12, '--level', 'peak-1'))
    rr = cmd('peak', '-c', s12, *(['--level', a.level] if a.level else []),
             *(['--show', show] if show != 'regions' else []), '--registers' if a.registers else None,
             *(['--liveness', lmod] if lmod != 'allocator' else []))
    return E.Envelope(led, claim, domain='op stream of the .kmb (static; allocator interval model)', evidence=rel(ev),
                      rerun=rr, tool_sha=tool_sha(), circuit_sha=sha, facts=facts, rows=rows,
                      table=res['tables'][show], next_cmds=nxt)


# ---------------------------------------------------------------- exec (executed Toffolis per region)
def recorded_mean_toffolis(sha, task, ledger):
    """Mean Toffolis per shot from a recorded full-settings local official report (not re-run), or None."""
    for r in reports_for(sha):
        if r['task'] == task and r['full_settings'] and 'macOS' in r['platform'] and r['status'] == 'PASS':
            try:
                summ = json.loads((ROOT / r['report']).read_text())['results'][task]['summary']
            except (OSError, KeyError, json.JSONDecodeError):
                continue
            if summ.get('circuit_sha256') == sha and summ.get('toffoli_count') is not None:
                ledger.earn('official-local', f"recorded local official verifier report {r['report']} "
                                              f"(toffoli_count over {fmt_int(summ.get('verified_shots') or 0)} shots; "
                                              'not re-run)')
                return dict(mean=summ['toffoli_count'], report=r['report'], shots=summ.get('verified_shots'),
                            static=summ.get('max_toffoli_count'), rotation_eps=summ.get('rotation_eps'),
                            rotation_count=summ.get('rotation_count'), t_count=summ.get('t_count'),
                            toffoli_equivalent=summ.get('toffoli_equivalent'))
    return None


def exec_small(meta, seed):
    """Per-region static/unconditional/zero-branch/sampled executed Toffolis for an op-table circuit (cached)."""
    import qhdl_exec as X
    L = Loaded(meta)
    names, in_block = X.static_kinds(L.ops)
    ev = L.dir / 'exec' / f'exec_seed{seed}.json'
    key = dict(tool_sha=tool_sha(), verb='exec', seed=seed)
    res = cached(ev, key)
    if res is None:
        res = dict(cache_key=key, enumerable=L.enumerable())
        if L.enumerable():
            z = X.executed(L.ops, L.xq, L.nq, 'zero')
            s = X.executed(L.ops, L.xq, L.nq, seed)
            res.update(zero={str(i): v for i, v in z['per_op'].items()}, sampled={str(i): v for i, v in s['per_op'].items()},
                       rows=s['rows'], mean=s['mean'], sd=s['sd'], zero_mean=z['mean'],
                       zpow_zero={str(i): v for i, v in z['zpow_per_op'].items()},
                       zpow_sampled={str(i): v for i, v in s['zpow_per_op'].items()})
        write_json(ev, res)
    regions = []
    for st in L.stages:
        idx = range(st['start'], min(st['end'], len(L.ops)))
        kinds = {}
        for i in idx:
            kinds[names[i]] = kinds.get(names[i], 0) + 1
        kinds.pop('INFO_TAG', None)
        tof = [i for i in idx if L.ops[i].code in X.B.TOFFOLI_LIKE]
        unc = [i for i in tof if not in_block[i] and L.ops[i].cond < 0]
        r = dict(region=st['name'], ops=f"{st['start']}..{st['end'] - 1}", static_toffolis=len(tof),
                 unconditional=len(unc), kinds=dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))))
        if res['enumerable']:
            r['zero_branch'] = round(sum(res['zero'][str(i)] for i in tof) / res['rows'], 2)
            r['mean_executed'] = round(sum(res['sampled'][str(i)] for i in tof) / res['rows'], 2)
            ex = {}
            for i in tof:
                ex[names[i]] = ex.get(names[i], 0) + res['sampled'][str(i)]
            r['executed_by_kind'] = {k: round(v / res['rows'], 2) for k, v in ex.items()}
        # Z_POW by verifier class (Clifford free, T 1 T, other angles a synthesised rotation)
        zc = {}
        for i in idx:
            if L.ops[i].code == X.B.C_ZPOW:
                zc.setdefault(X.zpow_class(L.ops[i].angle), []).append(i)
        r['rotations'] = len(zc.get('rotation', []))
        r['t_gates'] = len(zc.get('t', []))
        if res['enumerable'] and 'zpow_sampled' in res:
            r['mean_rotations'] = round(sum(res['zpow_sampled'][str(i)] for i in zc.get('rotation', [])) / res['rows'], 4)
            r['mean_t'] = round(sum(res['zpow_sampled'][str(i)] for i in zc.get('t', [])) / res['rows'], 4)
        regions.append(r)
    return res, regions, ev


def exec_large(meta):
    import qhdl_kmb as K
    import qhdl_peak as P
    src = circuit_path(meta)
    if src.suffix != '.kmb':
        raise QhdlError('exec on a large circuit reads .kmb files')
    ev = ws_dir(meta['sha256']) / 'exec' / 'exec_static.json'
    key = dict(tool_sha=tool_sha(), verb='exec', mode='static')
    res = cached(ev, key)
    if res is None:
        k = K.read(str(src))
        regs = P.regions(k)
        kinds = P.region_kinds(k, regs)
        res = dict(cache_key=key, regions=[dict(region=r['name'], ops=f"{r['start']}..{r['end'] - 1}",
                                                static_toffolis=kd['static_toffolis'],
                                                unconditional=kd['unconditional_toffolis'],
                                                z_pow=kd['kinds'].get('Z_POW', 0) + kd['kinds'].get('Z_POW_IF', 0),
                                                kinds=kd['kinds'])
                                           for r, kd in zip(regs, kinds)])
        write_json(ev, res)
    return res, res['regions'], ev


def v_exec(a):
    led = E.Ledger()
    meta = ensure_open(a.circuit)
    sha, s12, task = meta['sha256'], meta['sha12'], meta.get('task')
    large = 'ops_table' in meta
    if large:
        res, regions, ev = exec_large(meta)
    else:
        res, regions, ev = exec_small(meta, a.seed)
    led.earn('static', 'static gate counts per region; a Toffoli-like gate at condition depth 0 without its own '
                       '`if` runs in every shot (unconditional)')
    enum = (not large) and res.get('enumerable')
    if enum:
        led.earn('exact-all-x', f'qhdl_bitsim over all {fmt_int(MAX_X)} legal x, every HMR outcome 0 (zero branch)')
        n = res['rows']
        led.earn(f'sampled({n})', f'qhdl_bitsim over all {fmt_int(n)} legal x with one independent fair outcome per '
                                  f'HMR per x (seed {a.seed}); executed count per shot depends on outcomes only')
    rec = recorded_mean_toffolis(sha, task, led) if task in ('invsqrt', 'dqi') else None
    tot_s = sum(r['static_toffolis'] for r in regions)
    tot_u = sum(r['unconditional'] for r in regions)
    parts = [dict(part='static, unconditional', tier='static')]
    facts = {}
    if enum:
        se = res['sd'] / n ** 0.5
        top = max(regions, key=lambda r: r['mean_executed'])
        claim = (f"{s12}: executed Toffolis per shot: mean {res['mean']:,.4f} +- {se:.4f} (s.e.) over random "
                 f"measurement outcomes, {fmt_int(res['zero_mean'])} on the zero branch; static {fmt_int(tot_s)} "
                 f"({fmt_int(tot_u)} unconditional). Most executed: {top['region']} ({top['mean_executed']:,.2f})")
        parts += [dict(part='zero_branch', tier='exact-all-x'), dict(part='mean_executed', tier=f'sampled({n})')]
        tier = f'sampled({n})'
        domain = (f'all {fmt_int(MAX_X)} legal x: zero measurement branch (exact) and one fair outcome draw per x '
                  f'(seed {a.seed}, sampled)')
    else:
        why = ('control bits depend on the verifier-sampled syndromes and HMR outcomes; a per-region shot counter '
               'needs the kickmix engine (not built)') if large else 'needs an enumerable invsqrt circuit (<= 64 wires)'
        claim = (f"{s12}: {fmt_int(tot_s)} static Toffolis, {fmt_int(tot_u)} unconditional (run in every shot); "
                 f"executed per region unavailable ({why}): per region it lies between the two")
        for r in regions:
            r['mean_executed'] = 'unavailable'
        tier = 'static'
        domain = 'op stream (static)'
    if rec:
        parts.append(dict(part='recorded verifier mean', tier='official-local'))
        facts['recorded_report'] = rec['report']
        if enum:
            claim += (f". Recorded official verifier mean {rec['mean']:,.4f} (not re-run): difference "
                      f"{res['mean'] - rec['mean']:+.4f} = {abs(res['mean'] - rec['mean']) / se:.1f} s.e.")
        else:
            claim += f". Recorded official verifier mean {rec['mean']:,.2f} per shot (whole circuit, not re-run)"
    # Rotations: Z_POW by verifier class, priced in Toffoli-equivalents with a recorded report's eps.
    import qhdl_exec as X
    if large:
        zp = sum(r.get('z_pow', 0) for r in regions)
        facts['rotations'] = dict(z_pow_static=zp, note='DQI rejects Z_POW (Benchmark II)' if task == 'dqi' else
                                  'static count; angle classes need an op table')
    elif any('rotations' in r for r in regions):
        eps = rec.get('rotation_eps') if rec else None
        st_rot = sum(r['rotations'] for r in regions)
        st_t = sum(r['t_gates'] for r in regions)
        rot = dict(static_rotations=st_rot, static_t=st_t, eps=eps,
                   eps_source=(f"recorded report {rec['report']} (rotation_eps; set by the verifier's min phase margin)"
                               if eps else 'unavailable: eps needs the verifier min margin (run qhdl verify)'))
        if eps:
            rot['t_per_rotation'] = round(X.rotation_t_count(eps), 4)
        if enum:
            mr = sum(r.get('mean_rotations', 0) for r in regions)
            mt = sum(r.get('mean_t', 0) for r in regions)
            for r in regions:
                q_ = X.rotation_toffoli_eq(r.get('mean_rotations', 0), r.get('mean_t', 0), eps)
                r['rotation_toffoli_eq'] = round(q_, 2) if q_ is not None else 'unavailable'
            req = X.rotation_toffoli_eq(mr, mt, eps)
            rot.update(mean_rotations=round(mr, 4), mean_t=round(mt, 4))
            if req is not None:
                charge = res['mean'] + req
                rot.update(rotation_toffoli_eq=round(req, 2), charge_toffoli_eq=round(charge, 2),
                           rotation_share=round(req / charge, 4))
                claim += (f". Rotations: {mr:.2f} per shot (+{mt:.2f} T) = {req:,.2f} Toffoli-equivalents at the "
                          f"recorded eps {eps:.3e} ({X.rotation_t_count(eps):.2f} T each), "
                          f"{100 * req / charge:.1f}% of the charge {charge:,.2f}")
                if rec and rec.get('toffoli_equivalent'):
                    rot['recorded_toffoli_equivalent'] = rec['toffoli_equivalent']
                parts.append(dict(part='rotation_toffoli_eq', tier=f'sampled({n})',
                                  note='executed Z_POW counts sampled; eps quoted from the recorded report'))
            else:
                claim += (f". Rotations: {mr:.2f} per shot (+{mt:.2f} T); Toffoli-equivalent unavailable "
                          '(no recorded eps)')
        facts['rotations'] = rot
    if facts:
        facts['facts_file'] = rel(ev.parent / 'exec_facts.json')  # survives the stdout budget (`--all` prints all)
        write_json(ev.parent / 'exec_facts.json', facts)
    if a.region:
        names = [r['region'] for r in regions]
        if a.region not in names:
            raise QhdlError(f'no region {a.region!r}', 'regions: ' + ', '.join(names[:30]))
    if a.by == 'kind':
        full, tot, tex = [], {}, {}
        for r in regions:
            ex = r.get('executed_by_kind', {})
            for kname, c in r['kinds'].items():
                row = dict(region=r['region'], kind=kname, static=c)
                tot[kname] = tot.get(kname, 0) + c
                if kname in ex:
                    row['mean_executed'] = ex[kname]
                    tex[kname] = tex.get(kname, 0) + ex[kname]
                full.append(row)
            for kname, sk, mk in (('ROTATION (Z_POW)', 'rotations', 'mean_rotations'), ('T (Z_POW)', 't_gates', 'mean_t')):
                if r.get(sk):
                    row = dict(region=r['region'], kind=kname, static=r[sk])
                    tot[kname] = tot.get(kname, 0) + r[sk]
                    if mk in r:
                        row['mean_executed'] = r[mk]
                        tex[kname] = tex.get(kname, 0) + r[mk]
                    full.append(row)
        def toff_first(row):
            return (row['kind'].split('_IF')[0] not in ('CCX', 'CCZ', 'CSWAP', 'ROTATION (Z_POW)', 'T (Z_POW)'),
                    -row['static'], row['kind'])
        alls = sorted([dict(region='all', kind=kk, static=c, **({'mean_executed': round(tex[kk], 2)} if kk in tex
                                                                 else {})) for kk, c in tot.items()], key=toff_first)
        order = {r['region']: j for j, r in enumerate(regions)}
        full = sorted(full, key=lambda r: (order[r['region']],) + toff_first(r)) + alls
        rows = [r for r in full if r['region'] == a.region] if a.region else alls
        tname = 'exec_kinds.tsv'
    else:
        full = [{k: v for k, v in r.items() if k not in ('kinds', 'executed_by_kind')} for r in regions]
        rows = [r for r in full if r['region'] == a.region] if a.region else full
        tname = 'exec_regions.tsv'
    table = ev.parent / tname
    keys = []
    for r in full:
        keys += [k for k in r if k not in keys]
    write_tsv(table, keys, [[r.get(k, '') for k in keys] for r in full])
    top = max(regions, key=lambda r: (r['mean_executed'] if enum else r['static_toffolis']))['region']
    nxt = [cmd('exec', '-c', s12, '--by', 'kind', '--region', a.region or top)] if a.by != 'kind' or not a.region \
        else [cmd('exec', '-c', s12, '--region', a.region)]
    if large:
        nxt.append(cmd('peak', '-c', s12))
    elif not rec and task in ('invsqrt', 'dqi'):
        nxt.append(cmd('evidence', '-c', s12))
    return E.Envelope(led, claim, tier=tier, domain=domain, evidence=rel(ev),
                      rerun=cmd('exec', '-c', s12, *(['--by', a.by] if a.by != 'region' else []),
                                *(['--region', a.region] if a.region else []),
                                *(['--seed', a.seed] if a.seed != EXEC_SEED else [])),
                      tool_sha=tool_sha(), circuit_sha=sha, facts=facts, rows=rows, table=rel(table), parts=parts,
                      next_cmds=nxt)


# ---------------------------------------------------------------- circuits (dataset listing)
def v_circuits(a):
    led = E.Ledger()
    led.earn('static', 'public-circuit dataset manifest (hash-pinned files)')
    rows = []
    for r in manifest_rows():
        if a.task and r['task'] != a.task:
            continue
        if a.prefix and not r['sha256'].startswith(a.prefix.lower()):
            continue
        rows.append(dict(sha12=r['sha12'], task=r['task'], Q=int(r['qubits']) if r['qubits'] else None,
                         author=r['author'], recorded_score=r['local_task_score'] or None))
    rows.sort(key=lambda r: (r['task'], float(r['recorded_score']) if r['recorded_score'] else float('inf'), r['sha12']))
    out = WORK / '_index' / 'circuits.tsv'
    write_tsv(out, ['sha12', 'task', 'Q', 'author', 'recorded_score'], [list(r.values()) for r in rows])
    claim = f'{len(rows)} public circuits in the dataset manifest' + (f' (task {a.task})' if a.task else '') + \
            '; recorded_score is the manifest value (local official verifier), not re-checked here'
    return E.Envelope(led, claim, domain='dataset manifest ' + rel(MANIFEST), evidence=rel(out),
                      rerun=cmd('circuits', *(['--task', a.task] if a.task else []),
                                *(['--prefix', a.prefix] if a.prefix else [])),
                      tool_sha=tool_sha(), circuit_sha=None, rows=rows,
                      next_cmds=[cmd('open', '-c', rows[0]['sha12'])] if rows else [])


# ---------------------------------------------------------------- parser
EPILOG = """verbs:
  circuits               list public circuits (sha12, task, qubits, author, recorded score)
  open -c CIRCUIT        build the workspace work/<sha12>/ (other verbs do this on demand)
  peak                   any size, incl. DQI: live-width profile per INFO_TAG region, the peak runs and the wires
                         idle across them (--show regions|paths|runs|idle; --level peak-1)
  exec                   executed (charged) Toffolis per region: invsqrt exact on the zero branch + sampled mean
                         over random outcomes; static/unconditional bounds elsewhere; --by kind for a gate histogram
  profile                per-stage live width, ops at the peak, static Toffolis; flags recolouring headroom
  runs                   peak-width runs R0..Rn with stage; --level peak-1 for the next level down
  idle                   wires idle across peak runs (candidates for parking), longest first
  explain wire qN        what qN holds over its longest idle gap across peak runs, and whether it can be
                         recomputed there (exact information test on all legal x for invsqrt)
  explain run Rk         wires idle across run Rk and, for each, information-bearing vs determined
  recolour               lifetime recolouring (qubit renaming, gates unchanged) -> candidate circuit + its
                         recorded official-verifier status
  evidence               recorded official verifier reports for a circuit sha
  verify                 run the unmodified official verifier locally (invsqrt ~7 min, DQI ~4 min)

profile, runs, idle, explain and recolour need an op table (<= 2,000,000 ops); on DQI use peak, exec, evidence.

examples:
  python3 scripts/qhdl.py circuits --task invsqrt
  python3 scripts/qhdl.py open -c <invsqrt sha>
  python3 scripts/qhdl.py exec -c <invsqrt sha>
  python3 scripts/qhdl.py explain run R1 -c <invsqrt sha>
  python3 scripts/qhdl.py peak -c <dqi sha>
  python3 scripts/qhdl.py peak -c <dqi sha> --show idle
  python3 scripts/qhdl.py exec -c <dqi sha> --by kind
"""


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--all', action='store_true', help='print every row (default: <= 20 rows, <= 2 KB)')
    cc = argparse.ArgumentParser(add_help=False)
    cc.add_argument('-c', '--circuit', help='circuit: .kmb/.kmx path or sha prefix (>= 6 hex)')
    lv = argparse.ArgumentParser(add_help=False)
    lv.add_argument('--level', help="width level: 'peak' (default), 'peak-1', or an integer")
    ap = argparse.ArgumentParser(prog='qhdl', description=__doc__, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='verb', metavar='<verb>')
    p = sub.add_parser('circuits', parents=[common], help='list public circuits in the dataset manifest')
    p.add_argument('--task', choices=['invsqrt', 'dqi', 'dfthcsos'])
    p.add_argument('--prefix', help='sha prefix filter')
    p.set_defaults(fn=v_circuits)
    p = sub.add_parser('open', parents=[common, cc], help='build/refresh the workspace for a circuit')
    p.add_argument('circuit_pos', nargs='?', metavar='CIRCUIT', help='same as -c (kept for old commands)')
    p.add_argument('--credits', action='store_true', help='print the author and source notebook (off by default '
                                                          'so blind tests can use open)')
    p.set_defaults(fn=v_open)
    p = sub.add_parser('peak', parents=[common, cc, lv], help='live-width peak per INFO_TAG region (any size)')
    p.add_argument('--show', choices=PEAK_SHOW, default='regions', help='which table to print as rows')
    p.add_argument('--registers', action='store_true', help='--show idle: include register wires')
    p.add_argument('--liveness', choices=['allocator', 'reset'], default='allocator',
                   help='allocator (default): registers always live; reset: a register reset mid-circuit is free '
                        'until its next touch (the model of runs/idle)')
    p.set_defaults(fn=v_peak)
    p = sub.add_parser('exec', parents=[common, cc], help='executed Toffolis per region and gate-kind histogram')
    p.add_argument('--by', choices=['region', 'kind'], default='region')
    p.add_argument('--seed', type=int, default=EXEC_SEED, help='outcome seed for the sampled mean (invsqrt)')
    p.add_argument('--region', help='only this region (names as in the region rows)')
    p.set_defaults(fn=v_exec)
    p = sub.add_parser('profile', parents=[common, cc], help='per-stage width and gate profile')
    p.set_defaults(fn=v_profile)
    p = sub.add_parser('runs', parents=[common, cc, lv], help='peak-width runs R0..Rn')
    p.add_argument('--liveness', choices=['reset', 'allocator'], default='reset',
                   help='reset (default): a register reset mid-circuit is free until its next touch; allocator: '
                        'registers always live (the model of peak)')
    p.set_defaults(fn=v_runs)
    p = sub.add_parser('idle', parents=[common, cc, lv], help='wires idle across peak runs')
    p.add_argument('--registers', action='store_true', help='include register (x) wires')
    p.add_argument('--min-runs', type=int, default=1)
    p.set_defaults(fn=v_idle)
    p = sub.add_parser('explain', parents=[common, cc, lv], help='explain wire qN | explain run Rk',
                       description='explain wire qN [--at OP] | explain wire qN#k | explain run Rk')
    p.add_argument('noun', choices=['wire', 'run'])
    p.add_argument('address', help='qN, qN#k (interval) or Rk')
    p.add_argument('--at', type=int, help='pick the idle gap containing this op index')
    p.set_defaults(fn=v_explain)
    p = sub.add_parser('recolour', aliases=['recolor'], parents=[common, cc],
                       help='lifetime recolouring (qubit renaming) and its verifier status')
    p.add_argument('--no-register-gaps', dest='register_gaps', action='store_false',
                   help='keep register wires reserved for the whole circuit')
    p.set_defaults(fn=v_recolour)
    p = sub.add_parser('evidence', parents=[common, cc], help='recorded official verifier reports for a circuit')
    p.set_defaults(fn=v_evidence)
    p = sub.add_parser('verify', parents=[common, cc], help='run the official verifier locally (slow)')
    p.add_argument('--task', choices=['invsqrt', 'dqi'])
    p.add_argument('--salt', default='')
    p.add_argument('--workers', type=int, default=6)
    p.set_defaults(fn=v_verify)
    # hidden until the viewer honours its parameters (it reads only #<stage-id>, for one bundled circuit)
    p = sub.add_parser('link', parents=[common, cc])
    p.add_argument('--op', type=int)
    p.add_argument('--x', type=int)
    p.add_argument('--wires', help='comma list, e.g. q7,q0')
    p.add_argument('--run', help='Rk')
    p.add_argument('--stage')
    p.set_defaults(fn=v_link)
    if INV is not None:
        try:
            INV.register(sub, sys.modules[__name__])
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f'qhdl: investigation verbs not registered: {exc}\n')
    for name in PLUGINS:
        if (HERE / f'{name}.py').exists():
            try:
                mod = __import__(name)
            except Exception as exc:  # a broken plugin must not break the core verbs
                sys.stderr.write(f'qhdl: plugin {name} not loaded: {exc}\n')
                continue
            if hasattr(mod, 'qhdl_register'):
                try:
                    mod.qhdl_register(sub, sys.modules[__name__])
                except Exception as exc:
                    sys.stderr.write(f'qhdl: plugin {name} not registered: {exc}\n')
    return ap


def _verb_name(a):
    v = getattr(a, 'verb', None) or ''
    return {'recolor': 'recolour'}.get(v, v)


def _stamp(env, a, verb, addresses):
    """claim_id, provenance, certificate scope and resolved addresses on the envelope (QHDL v1 Phase 0)."""
    if INV is None:
        return
    d = env.d
    st = getattr(env, 'stamps', {})
    env.stamp(claim_id=st.get('claim_id') or INV.claim_id(verb, d.get('rerun'), d.get('tool_sha'),
                                                           d.get('circuit_sha')),
              provenance=st.get('provenance') or INV.default_provenance(verb),
              scope=st.get('scope') or INV.default_scope(verb, a, d),
              addresses=addresses or None)


def main(argv=None):
    import time
    t0 = time.time()
    argv = list(sys.argv[1:] if argv is None else argv)
    given = list(argv)  # the trail keeps what the agent typed; `rerun` and `addresses` show the resolved form
    ap = build_parser()
    addresses = {}
    if INV is not None:
        try:
            argv, addresses = INV.rewrite_argv(argv, sys.modules[__name__])
        except (QhdlError, E.TierError, ValueError) as exc:
            print(E.error_text(str(exc), getattr(exc, 'hint', None)))
            INV.record(given[0] if given else '', given, None, time.time() - t0, 2, error=exc)
            return 2
    a = ap.parse_args(argv)
    if not getattr(a, 'fn', None):
        ap.print_help()
        return 1
    verb = _verb_name(a)
    try:
        env = a.fn(a)
    except (QhdlError, E.TierError, ValueError) as exc:
        print(E.error_text(str(exc), getattr(exc, 'hint', None)))
        if INV is not None:
            INV.record(verb, given, None, time.time() - t0, 2, error=exc, addresses=addresses)
        return 2
    except AssertionError as exc:  # a library precondition failed: still one JSON envelope, never a traceback
        print(E.error_text(f'internal check failed: {exc!r}', 'is the input a valid .kmb/.kmx circuit of the '
                                                              'expected task?'))
        if INV is not None:
            INV.record(verb, given, None, time.time() - t0, 2, error=repr(exc), addresses=addresses)
        return 2
    if INV is not None and hasattr(env, 'stamp'):
        try:
            _stamp(env, a, verb, addresses)
        except Exception as exc:  # noqa: BLE001  (identity is additive; never lose the answer over it)
            sys.stderr.write(f'qhdl: envelope not stamped: {exc}\n')
    print(env.render(all_rows=a.all, max_bytes=(env.max_bytes or E.MAX_BYTES) - 1))
    if INV is not None:
        full = env.as_dict() if hasattr(env, 'as_dict') else {}
        INV.record(verb, given, full, time.time() - t0, 0, addresses=addresses)
    return 0


if __name__ == '__main__':
    sys.exit(main())
