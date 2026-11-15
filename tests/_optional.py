"""Optional-dependency probes shared by the tests: skip cleanly when kickmix, the engine, the official verifiers or the
public circuits dataset are absent. Settings are the environment variables documented in scripts/vqcc_config.py."""
import functools
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import vqcc_config as CFG  # noqa: E402


@functools.lru_cache(maxsize=None)
def has_kickmix() -> bool:
    try:
        r = subprocess.run([CFG.KMPY, "-I", "-c", "import kickmix"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


requires_kickmix = unittest.skipUnless(
    has_kickmix(), "needs kickmix: set QHDL_KICKMIX_PYTHON to a Python that can `import kickmix`")
requires_verifiers = unittest.skipUnless(
    CFG.VERIFIERS.is_dir(), "needs the official verifiers: set QHDL_OFFICIAL_ROOT (competition data, not shipped)")
requires_dataset = unittest.skipUnless(
    (CFG.DATASET / "manifest.csv").is_file(),
    "needs the public circuits dataset: set QHDL_DATASET (Kaggle manwithacat/vqcc-apache-public-circuits)")
