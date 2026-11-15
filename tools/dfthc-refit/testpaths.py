# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Where the tests find optional external inputs (not shipped with this tool).

Every location comes from an environment variable. When a variable is unset and there is no public default, the
location is a path that does not exist, so the tests that need it skip with a message naming the variable.

    QHDL_OFFICIAL_ROOT        directory containing the competition `verifiers/` (download the competition data)
    DFTHC_REFIT_VERIFIER_DIR  the verifiers directory itself (default $QHDL_OFFICIAL_ROOT/verifiers, else
                              <repo>/data/official_full/verifiers)
    QHDL_DATASET              the public circuits dataset directory (on Kaggle:
                              /kaggle/input/vqcc-apache-public-circuits; dataset
                              https://www.kaggle.com/datasets/manwithacat/vqcc-apache-public-circuits); its
                              circuits/chemistry/<family>/ directories hold the public families the tests name
    DFTHC_REFIT_LERA_SRC      directory with Lera's tensorqrom.py
    QHDL_KICKMIX_PYTHON       a Python that can import kickmix (run the loader tests with it)
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KAGGLE_DATASET = Path("/kaggle/input/vqcc-apache-public-circuits")


def unset(name: str) -> Path:
    """A path that never exists, standing for the unset variable `name`."""
    return HERE / ".unset" / name


def env_path(name: str, default: Path | None = None) -> Path:
    """$name as a Path, else `default`, else a path that does not exist."""
    v = os.environ.get(name)
    if v:
        return Path(v)
    return default if default is not None else unset(name)


def _verifiers_default() -> Path:
    root = os.environ.get("QHDL_OFFICIAL_ROOT")
    return Path(root) / "verifiers" if root else REPO / "data" / "official_full" / "verifiers"


def _dataset_default() -> Path | None:
    return KAGGLE_DATASET if KAGGLE_DATASET.is_dir() else None


VERIFIERS = env_path("DFTHC_REFIT_VERIFIER_DIR", _verifiers_default())
DATASET = env_path("QHDL_DATASET", _dataset_default())
LERA = env_path("DFTHC_REFIT_LERA_SRC")


def dataset_family(env: str, family: str) -> Path:
    """A public chemistry family: $env, else $QHDL_DATASET/circuits/chemistry/<family>."""
    return env_path(env, DATASET / "circuits" / "chemistry" / family)
