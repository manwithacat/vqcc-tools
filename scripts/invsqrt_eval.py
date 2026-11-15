#!/usr/bin/env python3
"""invsqrt evaluation service for search loops.

Contract: docs/invsqrt-evaluation-service.md. Every answer is *local prescreen (engine verify, parity-proven)*; the
unmodified official verifier (scripts/local_verify_reference.py) stays the promotion gate.

    PY=$QHDL_KICKMIX_PYTHON
    $PY -I scripts/invsqrt_eval.py verify CIRCUIT.kmb [--salt S]
    $PY -I scripts/invsqrt_eval.py seeds  CIRCUIT.kmb --salts 8 [--stop-on-fail]
    $PY -I scripts/invsqrt_eval.py margin CIRCUIT.kmb [--salts 4] [--per-x-out worst.npy]
    $PY -I scripts/invsqrt_eval.py models CIRCUIT.kmb            # native fixed-seed models 7/101 (comparator)
    $PY -I scripts/invsqrt_eval.py verify --candidates DIR       # batch: one JSON line per *.kmb, one engine
    $PY -I scripts/invsqrt_eval.py serve --socket /tmp/ie.sock   # resident engine; then add --via /tmp/ie.sock
    $PY -I scripts/invsqrt_eval.py fixture rare-branch SRC.kmb --k 27 --out OUT.kmb   # coverage-test fixtures

The engine is kickmix-engine (the installed build, or the python/ directory of a checkout given by --engine-python,
default $QHDL_ENGINE_PKG; read only). In-process mode keeps compiled programs (engine resident memory) and the reference tables for the
life of the process, so a batch is one resident engine. `--via SOCKET` sends `verify` requests to a running
`kickmix-engine serve` process instead (verify and seeds only; margin and models need per-shot arrays and always
run in-process). Standard library + NumPy + the engine; written by an AI model (Claude).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vqcc_config as CFG  # noqa: E402

ENGINE_PY = CFG.ENGINE_PKG  # '' = the installed kickmix_engine
VERIFIERS = os.environ.get('KICKMIX_ENGINE_VERIFIERS', str(CFG.VERIFIERS))
SCHEMA = 'vqcc-invsqrt-eval/1'
TIER = 'local prescreen (engine verify, parity-proven)'
TIER_MODELS = ("local prescreen (restatement of native fixed-seed models: one shot per x per seed; "
               "not the official shot schedule)")
GATE = ('promotion gate: the unmodified official verifier on kickmix (scripts/local_verify_reference.py), then '
        'hosted Kaggle')
SUMMARY_FIELDS = ('status', 'score', 'toffoli_equivalent', 'toffoli_count', 'max_toffoli_count', 't_count',
                  'rotation_count', 'rotation_eps', 'num_qubits', 'reaction_depth', 'num_ops', 'verified_shots',
                  'worst_error_over_tolerance', 'worst_x', 'circuit_sha256')
HIST_EDGES = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)  # bins [e_i, e_{i+1}); last two: [0.9999, 1.0] and > 1.0
R_MODEL_SEEDS = (7, 101)
R_CHUNK_ROWS = 16384


# ------------------------------------------------------------------------------------------------ engine plumbing
class Engine:
    """One in-process engine: imports kickmix_engine from ENGINE_PY, keeps programs and reference tables."""

    def __init__(self, engine_py=ENGINE_PY, verifiers=VERIFIERS):
        if engine_py and engine_py not in sys.path:
            sys.path.insert(0, engine_py)
        os.environ.setdefault('KICKMIX_ENGINE_VERIFIERS', verifiers)
        import numpy as np
        import kickmix_engine as ke
        from kickmix_engine import cache as kc
        from kickmix_engine import invsqrt as ki
        if not str(Path(ke.__file__).resolve()).startswith(str(Path(engine_py).resolve())):
            raise SystemExit(f'kickmix_engine imported from {ke.__file__}, not {engine_py}; run with python -I')
        self.np, self.ke, self.kc, self.ki = np, ke, kc, ki
        if kc.memory() is None:
            kc.enable_memory(max_programs=8)
        self.verifier_path = os.path.join(os.environ['KICKMIX_ENGINE_VERIFIERS'], 'invsqrt',
                                          'verify_submission_invsqrt.py')
        self.v = ki.load_verifier(self.verifier_path)
        self.spec = self.v.ProblemSpecInvSqrt.for_n_p(self.v.N_P)
        self.cache = kc.ProgramCache(enabled=True)
        self._ref = None
        self.ref_seconds = 0.0

    def reference(self):
        if self._ref is None:
            t0 = time.perf_counter()
            self._ref = self.v.reference_values(self.spec, self.np.arange(1, self.spec.max_x + 1, dtype=self.np.uint64))
            self.ref_seconds = time.perf_counter() - t0
        return self._ref

    def info(self):
        so = self.ke._engine.__file__ if hasattr(self.ke, '_engine') else None
        return dict(version=self.ke.__version__, package=str(Path(self.ke.__file__).parent),
                    extension_sha256=sha256_file(so) if so else 'unavailable', kernel=self.ke.kernel_info()['kernel'],
                    verifier=self.verifier_path, verifier_sha256=sha256_file(self.verifier_path))

    def load(self, path):
        prog, sha, info = self.kc.load_program(str(path), cache=self.cache)
        if info.get('save_needed'):
            self.cache.save(sha, prog)
        return prog, sha, info

    def task(self, path):
        """An engine InvSqrtTask whose reference tables come from this process's cache (computed once)."""
        t = self.ki.InvSqrtTask(str(path), spec=self.spec, verifier_path=self.verifier_path, cache=self.cache)
        t.v = _CachedReference(self.v, self)
        return t

    def verify(self, path, salt='', threads=4):
        """Engine full verify; returns the official summary dict or {'status': 'FAIL', 'error': ...}."""
        t = self.task(path)
        try:
            return t.verify(salt=salt, threads=threads)
        except (AssertionError, ValueError, RuntimeError, OSError) as ex:  # the official CLI's FAIL exceptions
            return {'status': 'FAIL', 'error': str(ex)}


class _CachedReference:
    """Proxy for the official verifier module whose reference_values() for the full domain is computed once."""

    def __init__(self, v, eng):
        self._v, self._eng = v, eng

    def __getattr__(self, name):
        return getattr(self._v, name)

    def reference_values(self, spec, xs):
        if spec == self._eng.spec and len(xs) == spec.max_x and int(xs[0]) == 1 and int(xs[-1]) == spec.max_x:
            return self._eng.reference()
        return self._v.reference_values(spec, xs)


class Resident:
    """verify through a running `kickmix-engine serve --socket PATH` (the engine CLI, output captured)."""

    def __init__(self, socket_path, engine_py=ENGINE_PY):
        if engine_py and engine_py not in sys.path:
            sys.path.insert(0, engine_py)
        from kickmix_engine.resident import Client
        self.client = Client(socket_path)
        if not self.client.ping():
            raise SystemExit(f'no resident engine answering on {socket_path}')
        self.socket = socket_path

    def verify(self, path, salt='', threads=4):
        r = self.client.call(['verify', 'invsqrt', str(Path(path).resolve()), '--salt', salt, '--threads',
                              str(threads)])
        try:
            out = json.loads(r['stdout'])
        except ValueError:
            out = {'status': 'ERROR', 'error': (r.get('stderr') or '')[-500:]}
        self.last_server_seconds = r.get('seconds')
        return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class Clock:
    def __enter__(self):
        self.w0 = time.perf_counter()
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.c0 = r.ru_utime + r.ru_stime
        return self

    def __exit__(self, *exc):
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.wall = time.perf_counter() - self.w0
        self.cpu = r.ru_utime + r.ru_stime - self.c0

    def as_dict(self, resident=False):
        return dict(wall_s=round(self.wall, 3), cpu_s='unavailable (served by the resident process)' if resident
                    else round(self.cpu, 3), peak_rss_mb=round(peak_rss_mb(), 1))


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == 'darwin' else r / 1024


def salts_from(args):
    if getattr(args, 'salt_list', None) is not None:
        return [s.strip() for s in args.salt_list.split(',')]
    n = getattr(args, 'salts', None) or 1
    fmt = getattr(args, 'salt_format', 'sweep-{}')
    return [''] + [fmt.format(k) for k in range(1, n)]


def candidates(args):
    if getattr(args, 'candidates', None):
        d = Path(args.candidates)
        files = sorted(p for p in d.iterdir() if p.suffix == '.kmb' and p.is_file())
        if not files:
            raise SystemExit(f'no .kmb files in {d}')
        return files
    if not args.circuit:
        raise SystemExit('give CIRCUIT or --candidates DIR')
    return [Path(args.circuit)]


# ---------------------------------------------------------------------------------------------- verify and seeds
def run_verify(backend, path, salt, threads):
    resident = isinstance(backend, Resident)
    with Clock() as c:
        s = backend.verify(path, salt=salt, threads=threads)
    row = dict(salt=salt, **{k: s[k] for k in SUMMARY_FIELDS if k in s})
    if s.get('status') != 'PASS':
        row['error'] = s.get('error')
    row['timing'] = c.as_dict(resident)
    if resident:
        row['timing']['server_s'] = backend.last_server_seconds
    return row


def cmd_verify(args, backend, emit):
    for p in candidates(args):
        row = run_verify(backend, p, args.salt, args.threads)
        emit(envelope('verify', p, backend, args, summary=row))


def cmd_seeds(args, backend, emit):
    salts = salts_from(args)
    for p in candidates(args):
        rows = []
        with Clock() as c:
            for s in salts:
                rows.append(run_verify(backend, p, s, args.threads))
                if args.stop_on_fail and rows[-1]['status'] != 'PASS':
                    break
        emit(envelope('seeds', p, backend, args, **seeds_digest(rows, salts), timing_total=c.as_dict(
            isinstance(backend, Resident))))


def seeds_digest(rows, salts):
    passed = [r for r in rows if r['status'] == 'PASS']
    fails = [i for i, r in enumerate(rows) if r['status'] != 'PASS']
    out = dict(salts_requested=len(salts), salts_run=len(rows), n_pass=len(passed), n_fail=len(fails),
               all_pass=not fails and len(rows) == len(salts),
               first_fail=None if not fails else dict(index=fails[0], salts_to_catch=fails[0] + 1,
                                                      salt=rows[fails[0]]['salt'], error=rows[fails[0]].get('error')),
               per_salt=rows)
    if passed:
        hi = max(passed, key=lambda r: r['score'])
        lo = min(passed, key=lambda r: r['score'])
        wr = max(passed, key=lambda r: (r['worst_error_over_tolerance'], -r['worst_x']))
        out['worst_case'] = dict(max_cost=hi['score'], max_cost_salt=hi['salt'], min_cost=lo['score'],
                                 min_cost_salt=lo['salt'],
                                 max_worst_error_over_tolerance=wr['worst_error_over_tolerance'],
                                 at_x=wr['worst_x'], at_salt=wr['salt'],
                                 min_rotation_eps=min(r['rotation_eps'] for r in passed
                                                      if r.get('rotation_eps') is not None)
                                 if any(r.get('rotation_eps') is not None for r in passed) else None)
    return out


# --------------------------------------------------------------------------------------------------- margin
def margin_one_salt(eng, prog, sha, salt, threads, acc):
    """Engine run of the official schedule without the on-engine check; per-shot phases folded per x into acc.

    Reproduces InvSqrtTask.verify's seeds, chunks and groups; the per-shot check is the verifier's (wrapping int64
    difference, |error| as uint64, x restored > scratch clean > phase). Returns the salt's summary (official fields
    on PASS) plus failure counts; never stops at the first failure."""
    np, v, spec, ke = eng.np, eng.v, eng.spec, eng.ke
    target, tol = eng.reference()
    x_ids, scratch = eng.ki.validate_registers_kmb(prog.registers, prog.num_qubits, spec)
    num_shots, bs, maxx = spec.num_shots, 256, spec.max_x
    base_seed = v._initial_seed(sha, salt)
    chunk_shots = v.BATCHES_PER_CHUNK * bs
    starts = range(0, num_shots, chunk_shots)
    seeds = np.random.SeedSequence(base_seed, pool_size=8).generate_state(4 * len(starts), dtype=np.uint64).reshape(-1, 4)
    tasks = [(s, min(s + chunk_shots, num_shots), int.from_bytes(seeds[i], 'little')) for i, s in enumerate(starts)]
    gsize = max(16, 16 * threads)
    groups, cur = [], []
    for t in tasks:
        nb = (t[1] - t[0]) // bs
        if cur and (len(cur) >= gsize or (cur[0][1] - cur[0][0]) // bs != nb):
            groups.append(cur)
            cur = []
        cur.append(t)
    if cur:
        groups.append(cur)

    def sim(group):
        nb = (group[0][1] - group[0][0]) // bs
        xs = np.stack([v.sample_shots(spec, t[0], t[1]).reshape(nb, bs) for t in group])
        return xs, ke.run(prog, [ke.seed_words(t[2]) for t in group], xs, x_ids, s=1, threads=threads, counting=True)

    worst_used, worst_salt, si = acc['worst_used'], acc['worst_salt'], acc['salt_index']
    min_margin, worst = 1 << 64, (-1.0, 0)
    tof = tcount = nrot = 0
    n_bad = dict(x_not_restored=0, scratch_dirty=0, phase_out_of_tolerance=0)
    first_fail, fail_x = None, set()
    ex = concurrent.futures.ThreadPoolExecutor(1)
    try:
        pending = ex.submit(sim, groups[0])
        for gi, group in enumerate(groups):
            xs, out = pending.result()
            if gi + 1 < len(groups):
                pending = ex.submit(sim, groups[gi + 1])
            flat_x = xs.reshape(-1)
            bad_x = (out['x_out'].reshape(-1) != flat_x)
            dirty = out['scratch_dirty'].reshape(-1).astype(bool) & ~bad_x
            idx = (flat_x - np.uint64(1)).astype(np.int64)
            allowed = tol[idx]
            used = np.abs((out['phase_units'].reshape(-1) - target[idx]).view(np.int64)).view(np.uint64)
            sane = ~(bad_x | dirty)
            bad_ph = sane & (used > allowed)
            n_bad['x_not_restored'] += int(bad_x.sum())
            n_bad['scratch_dirty'] += int(dirty.sum())
            n_bad['phase_out_of_tolerance'] += int(bad_ph.sum())
            anybad = bad_x | dirty | bad_ph
            if anybad.any():
                fail_x.update(int(x) for x in flat_x[anybad][:1000])
                if first_fail is None:
                    first_fail = first_failure(flat_x, bad_x, dirty, bad_ph, used, allowed, group[0][0], bs)
            used_ok = np.where(sane, used, np.uint64(0))
            ok = sane & ~bad_ph
            if ok.any():
                min_margin = min(min_margin, int((allowed[ok] - used_ok[ok]).min()))
                ratio = np.divide(used_ok, allowed, out=np.zeros(len(used_ok)), where=(allowed > 0) & ok)
                rmax = float(ratio.max())
                worst = max(worst, (rmax, -int(flat_x[ratio == rmax].min())))
            # Fold per x: shots are contiguous in shot order and x = shot mod max_x + 1, so split at pass boundaries.
            # Shots that failed the x / scratch checks carry no meaningful phase and are excluded (counted above);
            # phase-failing shots are included (their ratio > 1 is the point).
            fold_per_x(worst_used, worst_salt, si, used_ok, group[0][0], maxx)
            tof += int(out['toffoli_count'].sum())
            zl = out['z_levels']
            tcount += int(zl[:, 3].sum())
            nrot += int(zl[:, 4:].sum())
    finally:
        ex.shutdown(wait=True)
    nfail = sum(n_bad.values())
    row = dict(salt=salt, failing_shots=n_bad, failing_x_sample=sorted(fail_x)[:20], n_failing_x_seen=len(fail_x))
    if nfail:
        row.update(status='FAIL', first_failure=first_fail, error=first_fail['message'])
        return row
    toffoli_count, t_base, n_rot = tof / num_shots, tcount / num_shots, nrot / num_shots
    eps = math.pi * min_margin / 2**64 / n_rot if n_rot else None
    if n_rot and not eps > 0:
        row.update(status='FAIL', error='No tolerance left for rotation synthesis error.')
        return row
    t_count = t_base + (n_rot * v.rotation_t_count(eps) if n_rot else 0)
    toffolis = toffoli_count + t_count / v.T_PER_TOFFOLI
    row.update(status='PASS', score=prog.num_qubits * toffolis, toffoli_equivalent=toffolis,
               toffoli_count=toffoli_count, t_count=t_base, rotation_count=n_rot, rotation_eps=eps,
               num_qubits=prog.num_qubits, verified_shots=num_shots, worst_error_over_tolerance=worst[0],
               worst_x=-worst[1], min_margin_units=min_margin, circuit_sha256=sha)
    return row


def fold_per_x(worst_used, worst_salt, si, used, shot0, maxx):
    """worst_used[x-1] = max(worst_used[x-1], used of every shot of x) for contiguous shots shot0.. (x = shot mod
    max_x + 1); worst_salt records salt index si where the maximum strictly increased."""
    n, pos = len(used), 0
    while pos < n:
        x0 = (shot0 + pos) % maxx
        take = min(n - pos, maxx - x0)
        seg = used[pos:pos + take]
        cur = worst_used[x0:x0 + take]
        upd = seg > cur
        if upd.any():
            cur[upd] = seg[upd]
            worst_salt[x0:x0 + take][upd] = si
        pos += take


def first_failure(flat_x, bad_x, dirty, bad_ph, used, allowed, shot0, bs):
    """First failing 256-shot batch in shot order; within it the verifier's priority x > scratch > phase."""
    np = __import__('numpy')
    anyb = (bad_x | dirty | bad_ph).reshape(-1, bs).any(axis=1)
    b = int(np.flatnonzero(anyb)[0])
    sl = slice(b * bs, (b + 1) * bs)
    for kind, arr in (('x_not_restored', bad_x), ('scratch_dirty', dirty), ('phase_out_of_tolerance', bad_ph)):
        hit = np.flatnonzero(arr[sl])
        if len(hit):
            i = b * bs + int(hit[0])
            x = int(flat_x[i])
            if kind == 'x_not_restored':
                msg = 'x was not restored.'
            elif kind == 'scratch_dirty':
                msg = f'scratch dirty at x={x} (the official message names the qubit; use verify)'
            else:
                err = int(used[i])
                msg = (f'x={x}: phase error |{err / 2**64:.3e}| turns exceeds tolerance '
                       f'{int(allowed[i]) / 2**64:.3e}.')
            return dict(kind=kind, shot=shot0 + i, x=x, message=msg)
    return None


def cmd_margin(args, eng, emit):
    np = eng.np
    salts = salts_from(args)
    target, tol = eng.reference()
    for p in candidates(args):
        with Clock() as c:
            prog, sha, _ = eng.load(p)
            acc = dict(worst_used=np.zeros(eng.spec.max_x, dtype=np.uint64),
                       worst_salt=np.full(eng.spec.max_x, -1, dtype=np.int32), salt_index=0)
            rows = []
            for i, s in enumerate(salts):
                acc['salt_index'] = i
                with Clock() as cs:
                    rows.append(margin_one_salt(eng, prog, sha, s, args.threads, acc))
                rows[-1]['timing'] = cs.as_dict()
            per_x = per_x_digest(np, acc, tol, salts, args.top)
            if args.per_x_out:
                out = Path(args.per_x_out)
                if out.exists():
                    raise SystemExit(f'{out} exists (append-only); choose a new path')
                np.savez_compressed(out, worst_used=acc['worst_used'], worst_salt=acc['worst_salt'], tol=tol)
                per_x['per_x_file'] = str(out)
                per_x['per_x_file_sha256'] = sha256_file(out)
        emit(envelope('margin', p, eng, args, salts=salts, per_salt=rows, per_x=per_x, timing_total=c.as_dict()))
        if args.screen:
            print(screen(rows, per_x), file=sys.stderr)


def per_x_digest(np, acc, tol, salts, top):
    w, ws = acc['worst_used'], acc['worst_salt']
    ratio = w.astype(np.float64) / tol.astype(np.float64)
    # tol < 2^63; |error| can be exactly 2^63 (a half-turn error), which int64 cannot hold: clip it to 2^63 - 1, so
    # the margin is exact for every x within tolerance and off by at most one unit (still negative) for a failing x.
    margin = tol.astype(np.int64) - np.minimum(w, np.uint64(2**63 - 1)).astype(np.int64)
    j = int(np.argmax(ratio))
    jm = int(np.argmin(margin))
    edges = list(HIST_EDGES)
    hist = []
    for a, b in zip(edges[:-2], edges[1:-1]):
        hist.append(dict(bin=f'[{a}, {b})', x_count=int(((ratio >= a) & (ratio < b)).sum())))
    hist.append(dict(bin='[0.9999, 1.0]', x_count=int(((ratio >= 0.9999) & (ratio <= 1.0)).sum())))
    hist.append(dict(bin='> 1.0 (out of tolerance)', x_count=int((ratio > 1.0).sum())))
    order = np.argsort(-ratio, kind='stable')[:top]
    salt_of = lambda k: salts[int(ws[k])] if ws[k] >= 0 else None  # noqa: E731
    return dict(
        definition='per x: largest |phase error| over its 64 shots in each salt and over all salts run, against '
                   'tol(x) = the verifier reference_values tolerance (2^-24 relative, in 2^-64-turn units); '
                   'shots that failed the x or scratch check are excluded',
        x_count=int(len(w)),
        worst_ratio=float(ratio[j]), worst_ratio_x=j + 1, worst_ratio_salt=salt_of(j),
        min_margin_units=int(margin[jm]), min_margin_turns=float(margin[jm]) / 2**64, min_margin_x=jm + 1,
        min_margin_salt=salt_of(jm), histogram=hist,
        near_cap_x=dict(ge_0_99=int((ratio >= 0.99).sum()), ge_0_999=int((ratio >= 0.999).sum()),
                        ge_0_9999=int((ratio >= 0.9999).sum())),
        top=[dict(x=int(k) + 1, ratio=float(ratio[k]), margin_units=int(margin[k]), salt=salt_of(k)) for k in order])


def screen(rows, px):
    lines = [f'{"salt":>12} {"status":>6} {"cost":>14} {"worst/tol":>10} {"worst x":>9} {"min margin (units)":>20}']
    for r in rows:
        lines.append(f'{r["salt"] or "(default)":>12} {r["status"]:>6} {r.get("score", float("nan")):>14.4f} '
                     f'{r.get("worst_error_over_tolerance", float("nan")):>10.7f} {r.get("worst_x", "-"):>9} '
                     f'{r.get("min_margin_units", "-"):>20}')
    lines.append(f'per x over all salts: worst ratio {px["worst_ratio"]:.10f} at x={px["worst_ratio_x"]} '
                 f'(salt {px["worst_ratio_salt"]!r}); min margin {px["min_margin_units"]} units at '
                 f'x={px["min_margin_x"]} (salt {px["min_margin_salt"]!r})')
    lines.append('histogram of per-x worst ratio: ' + ', '.join(f'{h["bin"]}: {h["x_count"]}' for h in px['histogram']))
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- native fixed-seed models (comparator)
def cmd_models(args, eng, emit):
    """Restatement of the native fixed-seed models of an earlier independent stream checker: fixed 16,384-row
    chunks, chunk seed SHA256(seed_le32 || chunk_index_le8), one shot per legal x per seed, the final chunk padded with max_x. The zero-outcome BitSim model is not reproduced here."""
    np = eng.np
    target_all, tol_all = eng.reference()
    seeds = [int(s) for s in args.model_seeds.split(',')]
    for p in candidates(args):
        with Clock() as c:
            prog, sha, _ = eng.load(p)
            x_ids, _ = eng.ki.validate_registers_kmb(prog.registers, prog.num_qubits, eng.spec)
            maxx = eng.spec.max_x
            nchunks = (maxx + R_CHUNK_ROWS - 1) // R_CHUNK_ROWS
            models = []
            for seed in seeds:
                nfail, first, fail_x = 0, None, []
                for lo in range(0, nchunks, 16):
                    hi = min(nchunks, lo + 16)
                    rows = np.minimum(np.arange(lo * R_CHUNK_ROWS + 1, hi * R_CHUNK_ROWS + 1, dtype=np.uint64),
                                      np.uint64(maxx))
                    derived = [int.from_bytes(hashlib.sha256(seed.to_bytes(32, 'little') + i.to_bytes(8, 'little'))
                                              .digest(), 'little') for i in range(lo, hi)]
                    out = eng.ke.run(prog, derived, rows.reshape(hi - lo, 64, 256), x_ids, threads=args.threads,
                                     counting=True)
                    k = (rows - np.uint64(1)).astype(np.int64)
                    used = np.abs((out['phase_units'].reshape(-1) - target_all[k]).view(np.int64)).view(np.uint64)
                    good = ((out['x_out'].reshape(-1) == rows) & ~out['scratch_dirty'].reshape(-1).astype(bool)
                            & (used <= tol_all[k]))
                    if not good.all():
                        bad = np.flatnonzero(~good)
                        nfail += len(bad)
                        fail_x += [int(rows[j]) for j in bad[:20]]
                        if first is None:
                            first = int(rows[bad[0]])
                models.append(dict(seed=seed, valid=nfail == 0, failing_rows=nfail, first_failing_x=first,
                                   failing_x_sample=sorted(set(fail_x))[:20]))
        emit(envelope('models', p, eng, args, tier_override=TIER_MODELS, models=models,
                      valid_on_models=all(m['valid'] for m in models), timing_total=c.as_dict(),
                      note='zero-outcome model not run here: a fault that needs a measured 1 cannot fire under it'))


# --------------------------------------------------------------------------------------------------- fixtures
def cmd_fixture(args):
    """Test fixtures built with scripts/qhdl_edit.py (byte-for-byte copy of everything not edited)."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    import qhdl_edit as E
    import qhdl_kmb as K
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists (append-only); choose a new path')
    k = K.read(str(args.src))
    n = len(k.op_types)
    if args.kind == 'rare-branch':
        # k fresh X-basis measurements of a clean scratch qubit (each outcome uniform, no phase kick on |0>), then a
        # Z on one x qubit inside a condition block over all k outcomes: a half-turn phase error on shots whose k
        # outcomes are all 1 and whose x bit is 1 (probability 2^-(k+1) per shot). Under all-zero outcomes it never
        # fires; equivalent to a phase fix missing only on a rare measurement branch.
        x_ids = [i for _, i, _ in k.registers[0][1]]
        scratch = [q for q in range(int(k.num_qubits)) if q not in set(x_ids)]
        sq, xq, b0 = scratch[0], x_ids[args.x_bit], int(k.num_bits)
        ops = [f'HMR q{sq} b{b0 + i}' for i in range(args.k)]
        ops += [f'PUSH_CONDITION if b{b0 + i}' for i in range(args.k)] + [f'Z q{xq}'] + ['POP_CONDITION'] * args.k
        script = f'insert {n} => ' + '; '.join(ops) + '\n'
        desc = dict(kind='rare-branch', k=args.k, scratch_qubit=sq, x_qubit=xq, x_bit=args.x_bit, new_bits=args.k,
                    fail_probability_per_shot=f'2^-{args.k} x P(x bit {args.x_bit} = 1)')
    else:
        script = f'delete {args.op}\n'
        desc = dict(kind='delete-op', op=args.op, op_type=hex(int(k.op_types[args.op])))
    res = E.edit_file(args.src, script, out)
    print(json.dumps(dict(fixture=desc, edit_script=script.strip(), **res), indent=1))


# --------------------------------------------------------------------------------------------------- compare
def official_summary(path):
    """The invsqrt summary from an official report: local_verify_reference.py report.json, or the official CLI's
    stdout JSON ({'status': ...})."""
    d = json.loads(Path(path).read_text())
    if 'results' in d:
        return d['results']['invsqrt']['summary']
    return d


def compare_summaries(engine_row, official):
    """Field-by-field identity on the official summary fields (status, error text on FAIL)."""
    keys = SUMMARY_FIELDS if official.get('status') == 'PASS' else ('status', 'error')
    keys = tuple(k for k in keys if k in engine_row)  # margin rows omit the static fields (num_ops, depth, max)
    diffs = {k: dict(engine=engine_row.get(k), official=official.get(k)) for k in keys
             if engine_row.get(k) != official.get(k)}
    return dict(match=not diffs, fields_compared=list(keys), differences=diffs)


def cmd_compare(args):
    line = Path(args.engine).read_text().splitlines()[args.line]
    env = json.loads(line)
    row = env.get('summary') or env['per_salt'][args.salt_index]
    res = compare_summaries(row, official_summary(args.official))
    res.update(engine_file=args.engine, engine_file_sha256=sha256_file(args.engine), official_file=args.official,
               official_file_sha256=sha256_file(args.official), salt=row.get('salt'))
    print(json.dumps(res, indent=1))
    return 0 if res['match'] else 3


# --------------------------------------------------------------------------------------------------- output
def envelope(command, path, backend, args, tier_override=None, **body):
    resident = isinstance(backend, Resident)
    eng = None if resident else backend
    return dict(schema=SCHEMA, command=command, tier=tier_override or TIER, gate=GATE, circuit=str(path),
                circuit_sha256=sha256_file(path), threads=args.threads,
                mode=f'resident socket {backend.socket}' if resident else 'in-process (resident for this process)',
                engine=eng.info() if eng else dict(socket=backend.socket), **body)


def make_emitter(out_path):
    fh = None
    if out_path:
        p = Path(out_path)
        if p.exists():
            raise SystemExit(f'{p} exists (append-only); choose a new path')
        fh = open(p, 'x')

    def emit(obj):
        line = json.dumps(obj, default=str)
        print(line, flush=True)
        if fh:
            fh.write(line + '\n')
            fh.flush()
    return emit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(sp, salts=False):
        sp.add_argument('circuit', nargs='?')
        sp.add_argument('--candidates', help='directory: every *.kmb in it, in name order, through one engine')
        sp.add_argument('--threads', type=int, default=4)
        sp.add_argument('--out', help='also write the JSON lines here (new file only)')
        sp.add_argument('--engine-python', default=ENGINE_PY)
        if salts:
            sp.add_argument('--salts', type=int, default=None, help='N salts: the default salt, then sweep-1..N-1')
            sp.add_argument('--salt-list', help='explicit comma-separated salts (an empty item is the default salt)')
            sp.add_argument('--salt-format', default='sweep-{}')

    sp = sub.add_parser('verify')
    common(sp)
    sp.add_argument('--salt', default='')
    sp.add_argument('--via', help='socket of a running resident engine')
    sp = sub.add_parser('seeds')
    common(sp, salts=True)
    sp.add_argument('--stop-on-fail', action='store_true')
    sp.add_argument('--via')
    sp = sub.add_parser('margin')
    common(sp, salts=True)
    sp.add_argument('--top', type=int, default=20)
    sp.add_argument('--per-x-out', help='write per-x worst |error|, salt index and tol as .npz (new file only)')
    sp.add_argument('--screen', action='store_true', help='one-screen summary on stderr')
    sp = sub.add_parser('models')
    common(sp)
    sp.add_argument('--model-seeds', default='7,101')
    sp = sub.add_parser('serve')
    sp.add_argument('--socket', required=True)
    sp.add_argument('--engine-python', default=ENGINE_PY)
    sp.add_argument('--max-programs', type=int, default=8)
    sp = sub.add_parser('fixture')
    sp.add_argument('kind', choices=('rare-branch', 'delete-op'))
    sp.add_argument('src')
    sp.add_argument('--out', required=True)
    sp.add_argument('--k', type=int, default=27)
    sp.add_argument('--x-bit', type=int, default=0)
    sp.add_argument('--op', type=int)
    sp = sub.add_parser('compare', help='engine JSON line vs an official report (field identity)')
    sp.add_argument('--engine', required=True, help='a verify/seeds JSONL file written by this script')
    sp.add_argument('--official', required=True, help='local_verify_reference.py report.json or official CLI JSON')
    sp.add_argument('--line', type=int, default=0)
    sp.add_argument('--salt-index', type=int, default=0, help='for a seeds line: which per_salt row')
    a = ap.parse_args(argv)

    if a.cmd == 'compare':
        return cmd_compare(a)
    if a.cmd == 'fixture':
        return cmd_fixture(a)
    if a.cmd == 'serve':
        if a.engine_python and a.engine_python not in sys.path:
            sys.path.insert(0, a.engine_python)
        os.environ.setdefault('KICKMIX_ENGINE_VERIFIERS', VERIFIERS)
        from kickmix_engine.resident import serve
        return serve(a.socket, max_programs=a.max_programs)
    if a.cmd in ('verify', 'seeds') and a.via:
        backend = Resident(a.via, a.engine_python)
    else:
        backend = Engine(a.engine_python)
    emit = make_emitter(a.out)
    {'verify': cmd_verify, 'seeds': cmd_seeds, 'margin': cmd_margin, 'models': cmd_models}[a.cmd](a, backend, emit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
