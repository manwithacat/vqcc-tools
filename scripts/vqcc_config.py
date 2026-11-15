#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paths and interpreters used by the vqcc-tools scripts, read from environment variables.

Every setting has a default that works in a checkout of this repository; the Kaggle notebook locations are tried
when they exist. `python3 scripts/vqcc_config.py` prints the resolved values.

    QHDL_DATASET          public-circuits dataset directory (manifest.csv, circuits/<task>/<sha12>.kmb, parts/).
                          Default: /kaggle/input/vqcc-apache-public-circuits when present, else
                          <repo>/data/vqcc-apache-public-circuits. Download it from
                          https://www.kaggle.com/datasets/manwithacat/vqcc-apache-public-circuits
    QHDL_WORK             workspace root for qhdl (default <repo>/work)
    QHDL_OUTPUTS          directory scanned for recorded verifier reports and used for new evidence
                          (default <repo>/outputs)
    QHDL_OFFICIAL_ROOT    directory holding the competition's `verifiers/` (and its data). Not shipped here: download
                          the competition data from Kaggle. Default: <repo>/data/official_full, else the Kaggle
                          competition mount when present.
    QHDL_KICKMIX_PYTHON   Python interpreter that can `import kickmix` (default: the running interpreter).
                          VQCC_KMPY is accepted as an older name.
    QHDL_ENGINE_PKG       directory to put on sys.path for `kickmix_engine` (the `python/` directory of a
                          kickmix-engine checkout). Empty (default): use the installed package.
                          VQCC_ENGINE_PKG / VQCC_ENGINE_PY are accepted as older names.
    QHDL_ANALYSIS_PYTHON  interpreter with the `kickmix_analysis` / engine analysis extension (default: running one)
    QHDL_VIEWER_DIR       kickmix-macos `tools/circuit-viewer` directory (optional; default
                          <repo>/external/kickmix-macos/tools/circuit-viewer)
    QHDL_WORKBENCH        the fork's `workbench` launcher (default <QHDL_VIEWER_DIR>/workbench)

Sources: kickmix fork https://github.com/manwithacat/kickmix-macos, engine
https://github.com/manwithacat/kickmix-engine (Kaggle wheels: datasets manwithacat/kickmix-fork-wheels and
manwithacat/kickmix-engine-wheels).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KAGGLE_DATASET = Path('/kaggle/input/vqcc-apache-public-circuits')
KAGGLE_COMPETITION = Path('/kaggle/input/google-verifiable-quantum-circuit-compilation')


def _env(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return paths[-1]


def dataset_dir() -> Path:
    v = _env('QHDL_DATASET')
    if v:
        return Path(v)
    return _first_existing(ROOT / 'data' / 'vqcc-apache-public-circuits', KAGGLE_DATASET,
                           ROOT / 'data' / 'vqcc-apache-public-circuits')


def official_root() -> Path:
    v = _env('QHDL_OFFICIAL_ROOT')
    if v:
        return Path(v)
    return _first_existing(ROOT / 'data' / 'official_full', KAGGLE_COMPETITION, ROOT / 'data' / 'official_full')


DATASET = dataset_dir()
WORK = Path(_env('QHDL_WORK') or ROOT / 'work')
OUTPUTS = Path(_env('QHDL_OUTPUTS') or ROOT / 'outputs')
OFFICIAL_ROOT = official_root()
VERIFIERS = OFFICIAL_ROOT / 'verifiers'
KMPY = _env('QHDL_KICKMIX_PYTHON', 'VQCC_KMPY') or sys.executable
ENGINE_PKG = _env('QHDL_ENGINE_PKG', 'VQCC_ENGINE_PKG', 'VQCC_ENGINE_PY') or ''
ANALYSIS_PYTHON = _env('QHDL_ANALYSIS_PYTHON') or sys.executable
VIEWER_DIR = Path(_env('QHDL_VIEWER_DIR') or ROOT / 'external' / 'kickmix-macos' / 'tools' / 'circuit-viewer')
WORKBENCH = _env('QHDL_WORKBENCH') or str(VIEWER_DIR / 'workbench')


def add_engine_path() -> None:
    """Put QHDL_ENGINE_PKG on sys.path when it is set (otherwise the installed kickmix_engine is used)."""
    if ENGINE_PKG and ENGINE_PKG not in sys.path:
        sys.path.insert(0, ENGINE_PKG)


def add_verifiers_path() -> None:
    """Put the official verifiers directory on sys.path (it is not shipped; see QHDL_OFFICIAL_ROOT)."""
    if str(VERIFIERS) not in sys.path:
        sys.path.insert(0, str(VERIFIERS))


def require_verifiers() -> Path:
    if not VERIFIERS.is_dir():
        raise SystemExit(f'official verifiers not found at {VERIFIERS}: download the competition data from Kaggle '
                         'and set QHDL_OFFICIAL_ROOT to the directory that contains verifiers/')
    return VERIFIERS


if __name__ == '__main__':
    for k in ('ROOT', 'DATASET', 'WORK', 'OUTPUTS', 'OFFICIAL_ROOT', 'VERIFIERS', 'KMPY', 'ENGINE_PKG',
              'ANALYSIS_PYTHON', 'VIEWER_DIR', 'WORKBENCH'):
        print(f'{k:16s} {globals()[k]}')
