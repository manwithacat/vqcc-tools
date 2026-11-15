#!/usr/bin/env python3
"""qhdl plugin: `doctor` — cheap health checks of the tool chain.

    qhdl doctor [--engine-pkg DIR]

Checks, each one row (ok / FAIL / warn, with a fix hint):
  engine    the kickmix engine package that `qhdl_surgery.py gate` and other engine users put on PYTHONPATH
            (QHDL_ENGINE_PKG; VQCC_ENGINE_PKG is an older name): every name `__init__.py` imports from `._engine`
            (run_sequence, run_batches, ...) must be exported by `_engine.abi3.so`. The binary is loaded on its own
            in a subprocess of $QHDL_KICKMIX_PYTHON, so a stale binary is reported instead of raising inside a gate
            run (a stale binary without run_sequence has cost long runs before).
  kickmix   $QHDL_KICKMIX_PYTHON imports kickmix (official verifier runs).
  workbench the fork's workbench launcher exists.
  dataset   the public-circuit manifest is readable.
  work      QHDL_WORK (or work/) is writable.
  plugins   every qhdl plugin module imports.
Never modifies anything (the engine is a separate repository: rebuild there with `maturin develop --release` or
copy target/release/lib_engine.dylib over python/kickmix_engine/_engine.abi3.so).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vqcc_config as CFG  # noqa: E402

DEFAULT_ENGINE_PKG = CFG.ENGINE_PKG or str(CFG.ROOT / 'external' / 'kickmix-engine' / 'python')

_PROBE = r'''
import importlib.machinery, importlib.util, json, sys
so, names = sys.argv[1], sys.argv[2].split(',')
try:
    loader = importlib.machinery.ExtensionFileLoader('kickmix_engine._engine', so)
    spec = importlib.util.spec_from_file_location('kickmix_engine._engine', so, loader=loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    print(json.dumps(dict(ok=True, missing=[n for n in names if not hasattr(m, n)],
                          version=str(getattr(m, 'engine_version', lambda: '?')()))))
except Exception as exc:
    print(json.dumps(dict(ok=False, error=repr(exc))))
'''


def engine_pkg():
    return Path(os.environ.get('QHDL_ENGINE_PKG') or os.environ.get('VQCC_ENGINE_PKG') or DEFAULT_ENGINE_PKG)


def engine_needs(init_text):
    """Names __init__.py imports from ._engine."""
    names = []
    for m in re.finditer(r'from\s+\._engine\s+import\s+\(?([^)\n]+(?:\n[^)\n]+)*?)\)?\s*$', init_text, re.M):
        for part in m.group(1).split(','):
            nm = part.strip().split(' as ')[0].strip()
            if nm and re.fullmatch(r'[A-Za-z_]\w*', nm):
                names.append(nm)
    return sorted(set(names))


def check_engine(pkg, python):
    """(status, detail, hint) for the engine package at pkg."""
    init = pkg / 'kickmix_engine' / '__init__.py'
    so = pkg / 'kickmix_engine' / '_engine.abi3.so'
    hint = (f'rebuild the engine in its own repo (cd {pkg.parent} && maturin develop --release), or copy '
            f'{pkg.parent}/target/release/lib_engine.dylib over {so} if that build is newer')
    if not init.exists() or not so.exists():
        return 'FAIL', f'missing {init if not init.exists() else so}', hint
    needs = engine_needs(init.read_text())
    detail = f'{so} ({len(needs)} names imported by __init__.py)'
    stale = so.stat().st_mtime < init.stat().st_mtime
    if python and Path(python).exists():
        try:
            out = subprocess.run([python, '-I', '-c', _PROBE, str(so), ','.join(needs)], capture_output=True,
                                 text=True, timeout=60)
            import json
            r = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception as exc:  # probe failure: fall back to the byte scan
            r = dict(ok=False, error=repr(exc))
        if r.get('ok'):
            if r['missing']:
                return 'FAIL', f"{detail}: binary lacks {', '.join(r['missing'])}", hint
            return 'ok', f"{detail}: all exported (engine {r['version']})" + \
                ('; binary older than __init__.py but complete' if stale else ''), None
        detail += f" (load probe failed: {r.get('error', '?')[:120]}; byte scan used)"
    blob = so.read_bytes()
    missing = [n for n in needs if n.encode() + b'\0' not in blob and n.encode() not in blob]
    if missing:
        return 'FAIL', f"{detail}: binary lacks {', '.join(missing)}", hint
    if stale:
        return 'warn', f'{detail}: names found, but the binary is older than __init__.py', hint
    return 'ok', f'{detail}: names found (byte scan)', None


def run_checks(api):
    py = getattr(api, 'KMPY', CFG.KMPY)
    rows = []
    st, det, hint = check_engine(engine_pkg(), py)
    rows.append(dict(check='engine', status=st, detail=det, hint=hint or ''))
    if Path(py).exists():
        r = subprocess.run([py, '-I', '-c', 'import kickmix, sys; print(sys.version.split()[0])'], capture_output=True,
                           text=True, timeout=120)
        ok = r.returncode == 0
        rows.append(dict(check='kickmix', status='ok' if ok else 'FAIL',
                         detail=f'{py}: ' + (f'python {r.stdout.strip()}, kickmix imports' if ok else
                                             r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else '?'),
                         hint='' if ok else 'set QHDL_KICKMIX_PYTHON to a Python that can import kickmix'))
    else:
        rows.append(dict(check='kickmix', status='FAIL', detail=f'{py} missing', hint='set QHDL_KICKMIX_PYTHON to a Python that can import kickmix'))
    wb = Path(getattr(api, 'WORKBENCH', ''))
    rows.append(dict(check='workbench', status='ok' if wb.exists() else 'warn', detail=str(wb),
                     hint='' if wb.exists() else 'clone the kickmix-macos fork (docs/agent-circuit-tools.md)'))
    man = Path(getattr(api, 'MANIFEST', ''))
    rows.append(dict(check='dataset', status='ok' if man.exists() else 'warn', detail=str(man),
                     hint='' if man.exists() else 'SHA prefixes will not resolve; pass .kmb paths'))
    work = Path(getattr(api, 'WORK', 'work'))
    try:
        work.mkdir(parents=True, exist_ok=True)
        probe = work / '.doctor-probe'
        probe.write_text('ok')
        probe.unlink()
        rows.append(dict(check='work', status='ok', detail=str(work), hint=''))
    except OSError as exc:
        rows.append(dict(check='work', status='FAIL', detail=f'{work}: {exc}', hint='set QHDL_WORK to a scratch dir'))
    bad = []
    for name in getattr(api, 'PLUGINS', ()):
        if (HERE / f'{name}.py').exists():
            try:
                __import__(name)
            except Exception as exc:
                bad.append(f'{name}: {exc!r}'[:120])
    rows.append(dict(check='plugins', status='FAIL' if bad else 'ok',
                     detail='; '.join(bad) if bad else ', '.join(getattr(api, 'PLUGINS', ())),
                     hint=(f'Rerun with the accepted runtime: {py} {HERE / "qhdl.py"} doctor --all; '
                           f'current interpreter is {sys.executable}' if bad else '')))
    return rows


def v_doctor(api, a):
    import qhdl_envelope as E
    led = E.Ledger()
    if a.engine_pkg:
        os.environ['VQCC_ENGINE_PKG'] = a.engine_pkg
    rows = run_checks(api)
    led.earn('static', 'file, import and export checks only; no circuit is run')
    fails = [r for r in rows if r['status'] == 'FAIL']
    warns = [r for r in rows if r['status'] == 'warn']
    claim = (f"{len(rows) - len(fails) - len(warns)} of {len(rows)} checks ok" +
             (f"; FAIL: {', '.join(r['check'] for r in fails)} ({fails[0]['hint'] or fails[0]['detail']})"
              if fails else '') + (f"; warn: {', '.join(r['check'] for r in warns)}" if warns else ''))
    return E.Envelope(led, claim, domain='this machine: engine package, kickmix venv, workbench, dataset, workspace',
                      evidence=None, rerun=api.cmd('doctor', *(['--engine-pkg', a.engine_pkg] if a.engine_pkg
                                                               else [])),
                      tool_sha=api.tool_sha(), circuit_sha=None, rows=rows, next_cmds=[])


def qhdl_register(subparsers, api):
    import argparse
    p = subparsers.add_parser('doctor', help='health checks: engine binary exports, kickmix venv, workbench, '
                                             'dataset, workspace, plugins',
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all', action='store_true', help='print every row')
    p.add_argument('--engine-pkg', help='engine package dir (default QHDL_ENGINE_PKG or ' + DEFAULT_ENGINE_PKG + ')')
    p.set_defaults(fn=lambda a: v_doctor(api, a))
