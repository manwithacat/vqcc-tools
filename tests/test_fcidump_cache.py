"""FCIDUMP cache in scripts/local_verify_reference.py: key, option resolution, hit/miss/fallback, updater wiring.

The key and option tests are standard library only. The
hit/miss/fallback tests need NumPy (skipped in the stdlib-only project env; run them with a Python that has NumPy). The
official-parser test also needs the official verifiers ($QHDL_OFFICIAL_ROOT/verifiers) and is skipped without competition data; it uses a
tiny synthetic FCIDUMP, not the 128 MB reference file.
"""

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import local_verify_reference as L  # noqa: E402

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
import vqcc_config as CFG  # noqa: E402
VERIFIERS = CFG.VERIFIERS
HAS_VERIFIERS = (VERIFIERS / "dfthcsos/scoreinfra/reference.py").is_file()

# NORB=3: one-electron, two-electron (Mulliken order) and core-energy lines, values chosen to need float32 rounding.
TINY_FCIDUMP = """&FCI NORB=3,NELEC=2,MS2=0,
 ORBSYM=1,1,1,
 ISYM=1,
&END
  0.7000000000000001   1   1   1   1
  0.1234567890123456   2   1   1   1
  0.3333333333333333   2   2   1   1
  0.0987654321098765   2   1   2   1
  0.5555555555555556   3   3   2   2
  0.0111111111111111   3   2   2   1
  0.4444444444444444   3   3   3   3
 -1.2500000000000002   1   1   0   0
 -0.0500000000000003   2   1   0   0
 -0.9000000000000001   3   3   0   0
  3.1415926535897931   0   0   0   0
"""


class KeyAndOptionTest(unittest.TestCase):
    def test_key_is_full_lowercase_sha(self):
        sha = hashlib.sha256(b"x").hexdigest()
        self.assertEqual(L.fcidump_cache_file(Path("/c"), sha.upper()), Path("/c") / f"{sha}.npz")
        for bad in ("", sha[:16], sha + "0", "g" * 64, "../" + sha[3:]):
            with self.assertRaises(ValueError):
                L.fcidump_cache_file(Path("/c"), bad)

    def test_option_precedence(self):
        env = {L.FCIDUMP_CACHE_ENV: "/from/env"}
        self.assertIsNone(L.resolve_fcidump_cache(None, False, {}))
        self.assertEqual(L.resolve_fcidump_cache(None, False, env), Path("/from/env"))
        self.assertEqual(L.resolve_fcidump_cache(Path("/opt"), False, env), Path("/opt"))
        self.assertIsNone(L.resolve_fcidump_cache(Path("/opt"), True, env))
        self.assertEqual(L.resolve_fcidump_cache(None, False, {L.FCIDUMP_CACHE_ENV: "1"}), L.DEFAULT_FCIDUMP_CACHE)
        self.assertIsNone(L.resolve_fcidump_cache(None, False, {L.FCIDUMP_CACHE_ENV: "off"}))
        self.assertEqual(L.DEFAULT_FCIDUMP_CACHE, ROOT / "outputs/fcidump-cache")


def _h2_full_from_lower(h2_lower, n):
    """Same expansion as the official reference._h2_full_from_lower (used when competition data is absent)."""
    import numpy as np
    rows, cols = np.tril_indices(n)
    sym = np.zeros((n * (n + 1) // 2, n, n))
    sym[:, rows, cols] = np.array(h2_lower)
    sym[:, cols, rows] = np.array(h2_lower)
    full = np.zeros((n, n, n, n))
    full[rows, cols, :, :] = sym
    full[cols, rows, :, :] = sym
    return full


def _fake_official(calls):
    """Deterministic stand-in for load_reference: float32-rounded packed triangle, expanded like the official code."""
    import numpy as np

    def load(path):
        calls.append(str(path))
        seed = int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        n = 4
        pairs = n * (n + 1) // 2
        lower = rng.standard_normal((pairs, pairs)).astype(np.float32)
        lower = (lower + lower.T) / np.float32(2)
        oei = rng.standard_normal((n, n))
        oei = oei + oei.T
        tei = _h2_full_from_lower(lower, n)
        return {"orbitals": n, "electrons": 3, "ecore": float(rng.standard_normal()), "oei": oei, "tei": tei,
                "h1_thc": np.array(oei - 0.5 * np.einsum("pqqs->ps", tei))}
    return load


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed (stdlib-only env); run with the kickmix venv")
class CacheLogicTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.fcidump = self.dir / "x.fcidump"
        self.fcidump.write_text(TINY_FCIDUMP)
        self.cache = self.dir / "cache"
        self.calls, self.events = [], []
        self.load = L.make_cached_load_reference(_fake_official(self.calls), _h2_full_from_lower, self.cache, self.events)

    def tearDown(self):
        self.tmp.cleanup()

    def assertSameRef(self, a, b):
        import numpy as np
        self.assertEqual(set(a), set(b))
        for k in a:
            if isinstance(a[k], np.ndarray):
                self.assertEqual((a[k].dtype, a[k].shape), (b[k].dtype, b[k].shape), k)
                self.assertTrue(np.array_equal(a[k], b[k]), k)
            else:
                self.assertEqual((type(a[k]), a[k]), (type(b[k]), b[k]), k)

    def cache_file(self):
        return L.fcidump_cache_file(self.cache, hashlib.sha256(self.fcidump.read_bytes()).hexdigest())

    def test_miss_then_hit_is_identical_and_skips_parse(self):
        first = self.load(self.fcidump)
        self.assertEqual(self.events[-1]["status"], "miss-written")
        self.assertTrue(self.cache_file().is_file())
        self.assertEqual([p.name for p in self.cache.iterdir()], [self.cache_file().name])  # no temp files left
        second = self.load(self.fcidump)
        self.assertEqual(self.events[-1]["status"], "hit")
        self.assertEqual(len(self.calls), 1)
        self.assertSameRef(first, second)

    def test_changed_fcidump_gets_new_key(self):
        self.load(self.fcidump)
        self.fcidump.write_text(TINY_FCIDUMP + "  0.1   1   1   1   1\n")
        self.load(self.fcidump)
        self.assertEqual(self.events[-1]["status"], "miss-written")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(list(self.cache.glob("*.npz"))), 2)

    def test_corrupt_cache_falls_back(self):
        self.load(self.fcidump)
        self.cache_file().write_bytes(b"not an npz")
        ref = self.load(self.fcidump)
        self.assertEqual(self.events[-1]["status"], "fallback")
        self.assertIn("read:", self.events[-1]["reason"])
        self.assertEqual(len(self.calls), 2)
        self.assertSameRef(ref, _fake_official([])(self.fcidump))

    def test_tampered_payload_or_key_falls_back(self):
        import numpy as np
        self.load(self.fcidump)
        path = self.cache_file()
        with np.load(path, allow_pickle=False) as d:
            data = {k: np.array(d[k]) for k in d.files}
        for field, value in (("h2_lower", data["h2_lower"] + np.float32(1)),
                             ("fcidump_sha256", np.array("0" * 64)),
                             ("format", np.int64(99))):
            with self.subTest(field=field):
                with path.open("wb") as f:
                    np.savez(f, **{**data, field: value})
                self.load(self.fcidump)
                self.assertEqual(self.events[-1]["status"], "fallback")

    def test_unwritable_cache_still_returns_official(self):
        self.cache.write_text("a file, not a directory")
        ref = self.load(self.fcidump)
        self.assertEqual(self.events[-1]["status"], "miss-not-written")
        self.assertSameRef(ref, _fake_official([])(self.fcidump))

    def test_missing_fcidump_raises_like_official(self):
        with self.assertRaises(FileNotFoundError):
            self.load(self.dir / "absent.fcidump")
        self.assertEqual(self.events[-1]["status"], "fallback")


@unittest.skipUnless(HAS_NUMPY and HAS_VERIFIERS, "needs NumPy and the official verifiers (competition data)")
class OfficialParserTest(unittest.TestCase):
    def test_tiny_fcidump_round_trip_through_official_parser(self):
        import numpy as np
        sys.path.insert(0, str(VERIFIERS))
        import dfthcsos.scoreinfra.reference as REF
        with tempfile.TemporaryDirectory() as tmp:
            fcidump = Path(tmp) / "tiny.fcidump"
            fcidump.write_text(TINY_FCIDUMP)
            events = []
            load = L.make_cached_load_reference(REF.load_reference, REF._h2_full_from_lower, Path(tmp) / "c", events)
            official = REF.load_reference(fcidump)
            miss, hit = load(fcidump), load(fcidump)
            self.assertEqual([e["status"] for e in events], ["miss-written", "hit"])
            for ref in (miss, hit):
                self.assertEqual(set(ref), set(official))
                for k, v in official.items():
                    if isinstance(v, np.ndarray):
                        self.assertEqual(ref[k].dtype, v.dtype, k)
                        self.assertTrue(np.array_equal(ref[k], v), k)
                    else:
                        self.assertEqual((type(ref[k]), ref[k]), (type(v), v), k)


if __name__ == "__main__":
    unittest.main()
