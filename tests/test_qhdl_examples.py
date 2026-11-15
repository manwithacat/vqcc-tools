"""qhdl docs and `next` hints stay true.

Runs every example in `qhdl --help` and every `next` string that `open` (and `peak`/`exec`) emits, against
  * the invsqrt fixture: the public frontier circuit bce2948ff24e from the dataset manifest (skipped if the dataset
    or the kickmix interpreter used for .kmb -> .kmx is missing);
  * a large-circuit stand-in: a small synthetic tagged .kmb opened with QHDL_MAX_OPS_TABLE=5, so it takes the DQI
    code path (no op table) in well under a second;
  * the real DQI frontier bdee18489fac (60M ops, about 40 s, 8 GB): only with QHDL_SLOW_TESTS=1.
A command fails the test if its verb is invalid, it prints an error, or a row-producing verb returns zero rows.
Also checks envelope `parts`, `open -c`, QHDL_WORK in rerun strings, and that open references (not copies) large
circuit files. Needs NumPy (system python3); skipped in the stdlib-only project environment.
"""

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import qhdl_envelope as E  # noqa: E402

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
import vqcc_config as CFG  # noqa: E402
KMPY = Path(CFG.KMPY)
MANIFEST = CFG.DATASET / "manifest.csv"
INVSQRT = "bce2948ff24e"
DQI = "bdee18489fac"
ROW_VERBS = {"circuits", "peak", "exec", "profile", "runs", "idle", "explain"}
NOT_RUN = {"verify"}  # runs the official verifier (minutes); parsed, not executed


def help_examples():
    out = subprocess.run([sys.executable, str(SCRIPTS / "qhdl.py"), "--help"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    ex = out.stdout.split("examples:", 1)[1]
    return [ln.strip() for ln in ex.splitlines() if ln.strip().startswith("python3 scripts/qhdl.py")]


def split_cmd(text):
    """'[VAR=v ...] python3 scripts/qhdl.py verb ...' -> (env assignments, argv after the script)."""
    toks = shlex.split(text)
    env = {}
    while toks and re.match(r"^[A-Z_]+=", toks[0]):
        k, v = toks.pop(0).split("=", 1)
        env[k] = v
    assert toks[0] in ("python3", sys.executable) and Path(toks[1]).resolve() == (SCRIPTS / "qhdl.py").resolve(), text
    return env, toks[2:]


def write_large_standin(path):
    """Tagged .kmb with registers, scratch, resets, an MBU-style HMR + PUSH_CONDITION block and Toffolis."""
    import qhdl_kmb as K
    ops, st = [], {n: [] for n in ("qqq0", "qqq1", "qqq2", "qq0", "qq1", "q0", "b0", "bc")}
    tags = []

    def tag(text):
        ops.append(K.INFO_TAG)
        tags.append(text.encode())

    def g3(code, a, b, c):
        ops.append(code)
        st["qqq0"].append(a), st["qqq1"].append(b), st["qqq2"].append(c)

    def g2(code, a, b):
        ops.append(code)
        st["qq0"].append(a), st["qq1"].append(b)

    def g1(code, a):
        ops.append(code)
        st["q0"].append(a)
    tag("@enter:compute")
    g3(0x31, 0, 1, 4)        # CCX r0 r1 -> s4
    g3(0x31, 2, 3, 5)        # CCX r2 r3 -> s5
    tag("@enter:inner")
    g3(0x31, 4, 5, 6)        # peak: 4 regs + s4 s5 s6
    g2(0x21, 6, 7)           # CX s6 -> s7 (s4 idle across this)
    g2(0x21, 6, 7)
    g3(0x31, 4, 5, 6)
    tag("@exit:inner")
    g3(0x31, 2, 3, 5)
    g1(K.R, 5)
    tag("@exit:compute")
    tag("@enter:erase")
    ops.append(K.HMR)        # HMR s4 b0
    st["q0"].append(4), st["b0"].append(0)
    ops.append(K.PUSH_CONDITION)
    st["bc"].append(0)
    g3(0x30, 0, 1, 2)        # CCZ inside the block: conditional Toffoli
    ops.append(K.POP_CONDITION)
    tag("@exit:erase")
    K.write_kmb(str(path), ops, {k: v for k, v in st.items()}, tags=tags,
                registers=[("r", [(True, 0), (True, 1), (True, 2), (True, 3)])], num_qubits=8, num_bits=1)


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed (stdlib-only env); run with system python3")
class QhdlCommandHarness(unittest.TestCase):
    base_env = {}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name) / "work"

    def tearDown(self):
        self._tmp.cleanup()

    def run_text(self, text, subs=None):
        for k, v in (subs or {}).items():
            text = text.replace(k, v)
        env_add, argv = split_cmd(text)
        if argv and argv[0] in NOT_RUN:
            h = subprocess.run([sys.executable, str(SCRIPTS / "qhdl.py"), argv[0], "--help"], capture_output=True,
                               text=True)
            self.assertEqual(h.returncode, 0, f"invalid verb in {text}")
            return None
        env = dict(os.environ, QHDL_WORK=str(self.work), **self.base_env)
        env.update(env_add)
        out = subprocess.run([sys.executable, str(SCRIPTS / "qhdl.py"), *argv], capture_output=True, text=True,
                             env=env, cwd=ROOT)
        self.assertEqual(out.returncode, 0, f"{text}\n{out.stdout}\n{out.stderr}")
        if "--all" not in argv:
            self.assertLessEqual(len(out.stdout.encode()), E.MAX_BYTES, text)
        d = json.loads(out.stdout)
        self.assertNotIn("error", d, text)
        if argv[0] in ROW_VERBS:
            self.assertGreater(d.get("rows_total", 0), 0, f"zero rows: {text}\n{out.stdout}")
        else:
            self.assertTrue(d.get("rows_total") or d.get("facts"), f"empty result: {text}")
        self.assertIn(f"QHDL_WORK={self.work}", d["rerun"])
        return d

    def run_next(self, d, subs=None, depth=1):
        for n in d.get("next", []):
            d2 = self.run_text(n, subs)
            if d2 is not None and depth > 1:
                self.run_next(d2, subs, depth - 1)


class LargeStandinTest(QhdlCommandHarness):
    base_env = {"QHDL_MAX_OPS_TABLE": "5"}

    def setUp(self):
        super().setUp()
        self.src = Path(self._tmp.name) / "standin.kmb"
        write_large_standin(self.src)
        self.sha12 = E.file_sha(self.src, 12)

    def test_open_next_and_help_examples(self):
        d = self.run_text(f"python3 scripts/qhdl.py open -c {self.src}")
        self.assertIn("peak", " ".join(d["next"]))
        for bad in ("profile", "runs", "idle", "explain", "link"):
            self.assertFalse(any(f"qhdl.py {bad} " in n for n in d["next"]), d["next"])
        self.run_next(d, depth=2)
        for ex in help_examples():
            if "<dqi sha>" in ex:
                self.run_text(ex, {"<dqi sha>": self.sha12})

    def test_peak_and_exec_numbers(self):
        p = self.run_text(f"python3 scripts/qhdl.py peak -c {self.src} --show regions --all")
        self.assertEqual(p["facts"]["declared_qubits"], 8)
        self.assertIn("peak 8 live wires (4 registers + 4 scratch)", p["claim"])
        idle = self.run_text(f"python3 scripts/qhdl.py peak -c {self.src} --show idle")
        self.assertEqual(idle["rows"][0]["wire"], "q4")  # s4 idle across the CX pair at the peak
        x = self.run_text(f"python3 scripts/qhdl.py exec -c {self.src} --all")
        self.assertIn("6 static Toffolis, 5 unconditional", x["claim"])
        self.assertEqual(x["tier"], "static")
        self.assertTrue(all(r["mean_executed"] == "unavailable" for r in x["rows"]))

    def test_open_positional_and_dash_c_agree(self):
        a = self.run_text(f"python3 scripts/qhdl.py open {self.src}")
        b = self.run_text(f"python3 scripts/qhdl.py open -c {self.src}")
        self.assertEqual(a["claim"], b["claim"])
        self.assertTrue(a["rerun"].endswith(f"open -c {self.sha12}"))

    def test_large_file_referenced_not_copied(self):
        self.base_env = dict(self.base_env, QHDL_REF_BYTES="0")
        d = self.run_text(f"python3 scripts/qhdl.py open -c {self.src}")
        ws = self.work / self.sha12
        self.assertFalse((ws / "circuit.kmb").exists())
        meta = json.loads((ws / "meta.json").read_text())
        self.assertTrue(meta["circuit_ref"].endswith("standin.kmb"))
        self.assertIn("not copied", d["facts"]["circuit"])
        self.run_text(f"python3 scripts/qhdl.py peak -c {self.sha12}")  # resolves through the reference


@unittest.skipUnless(HAS_NUMPY and KMPY.exists() and MANIFEST.exists() and INVSQRT in MANIFEST.read_text(),
                     "needs NumPy, the kickmix venv and the public-circuit dataset")
class InvsqrtFixtureTest(QhdlCommandHarness):
    # one workspace for the class, so the bitsim passes are cached across its tests
    @classmethod
    def setUpClass(cls):
        cls._ctmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._ctmp.cleanup()

    def setUp(self):
        self.work = Path(self._ctmp.name) / "work"

    def tearDown(self):
        pass

    def test_open_next_and_help_examples(self):
        d = self.run_text(f"python3 scripts/qhdl.py open -c {INVSQRT}")
        self.run_next(d)
        for ex in help_examples():
            if "<dqi sha>" not in ex:
                self.run_text(ex, {"<invsqrt sha>": INVSQRT})

    def test_executed_toffolis_match_recorded_verifier(self):
        d = self.run_text(f"python3 scripts/qhdl.py exec -c {INVSQRT} --all")
        n = re.search(r"mean ([\d,]+\.\d+) \+- ([\d.]+)", d["claim"])
        mean, se = float(n.group(1).replace(",", "")), float(n.group(2))
        self.assertAlmostEqual(mean, 1325.48, delta=0.01)  # manifest / verifier mean Toffolis per shot
        self.assertIn("1,274 on the zero branch", d["claim"])
        self.assertIn("static 1,495", d["claim"])
        self.assertEqual(sum(r["static_toffolis"] for r in d["rows"]), 1495)
        self.assertAlmostEqual(sum(r["mean_executed"] for r in d["rows"]), mean, delta=0.05)
        tiers = {p["part"]: p["tier"] for p in d["parts"]}
        self.assertEqual(tiers["zero_branch"], "exact-all-x")
        self.assertTrue(tiers["mean_executed"].startswith("sampled("))
        if "recorded verifier mean" in tiers:
            self.assertLess(float(re.search(r"= ([\d.]+) s\.e\.", d["claim"]).group(1)), 3.0)
        self.assertLess(se, 0.01)


@unittest.skipUnless(os.environ.get("QHDL_SLOW_TESTS") == "1" and HAS_NUMPY and MANIFEST.exists()
                     and DQI in MANIFEST.read_text(), "slow: set QHDL_SLOW_TESTS=1 (real 60M-op DQI, ~40 s, 8 GB)")
class DqiFixtureSlowTest(QhdlCommandHarness):
    def test_open_next_and_help_examples(self):
        d = self.run_text(f"python3 scripts/qhdl.py open -c {DQI}")
        self.assertFalse((self.work / DQI / "circuit.kmb").exists())  # referenced, not copied
        self.run_next(d)
        for ex in help_examples():
            if "<dqi sha>" in ex:
                self.run_text(ex, {"<dqi sha>": DQI})
        p = self.run_text(f"python3 scripts/qhdl.py peak -c {DQI}")
        self.assertIn("peak 892 live wires", p["claim"])


class PartsEnvelopeTest(unittest.TestCase):
    def env(self, led, parts):
        return E.Envelope(led, claim="c", domain="d", evidence=None, rerun="r", tool_sha="t", circuit_sha=None,
                          parts=parts)

    def test_parts_need_an_earned_tier(self):
        led = E.Ledger()
        led.earn("static", "counted")
        led.earn("exact-all-x", "all x")
        with self.assertRaises(E.TierError):
            self.env(led, [dict(part="recompute search", tier="sampled(4000)")])  # below best, but not earned
        with self.assertRaises(E.TierError):
            self.env(led, [dict(part="no tier")])
        d = json.loads(self.env(led, [dict(part="value", tier="exact-all-x"), dict(part="n", tier="static")]).render())
        self.assertEqual([p["tier"] for p in d["parts"]], ["exact-all-x", "static"])
        self.assertEqual(d["parts"][0]["basis"], "all x")

    def test_part_bases_dropped_before_rows(self):
        led = E.Ledger()
        led.earn("static", "b" * 900)
        rows = [{"i": i, "v": "x" * 40} for i in range(20)]
        env = E.Envelope(led, claim="c", domain="d", evidence=None, rerun="r", tool_sha="t", circuit_sha=None,
                         rows=rows, parts=[dict(part="p", tier="static")])
        d = json.loads(env.render())
        self.assertNotIn("basis", d["parts"][0])
        self.assertGreaterEqual(d["rows_shown"], 3)


if __name__ == "__main__":
    unittest.main()
