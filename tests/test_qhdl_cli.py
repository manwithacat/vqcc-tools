"""qhdl CLI: evidence envelope, tier discipline, stdout limits, address stability.

The envelope/address tests are standard library only. The end-to-end workspace test runs the CLI on a tiny
synthetic .kmx and needs NumPy (system python3); it is skipped in the stdlib-only project environment.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import qhdl_envelope as E  # noqa: E402

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def envelope(ledger, **kw):
    args = dict(claim="c", domain="d", evidence="work/x/e.json", rerun="python3 scripts/qhdl.py v", tool_sha="t" * 12,
                circuit_sha="a" * 64)
    args.update(kw)
    return E.Envelope(ledger, **args)


class EnvelopeTests(unittest.TestCase):
    def test_mandatory_fields_in_fixed_order(self):
        led = E.Ledger()
        led.earn("static", "counted ops")
        d = json.loads(envelope(led).render())
        self.assertEqual(list(d)[:7], ["claim", "tier", "domain", "evidence", "rerun", "tool_sha", "circuit_sha"])
        self.assertEqual(d["tier"], "static")
        self.assertEqual(d["basis"], "counted ops")

    def test_default_tier_is_best_earned(self):
        led = E.Ledger()
        led.earn("static", "a")
        led.earn("sampled(4000)", "b")
        led.earn("exact-all-x", "c")
        self.assertEqual(json.loads(envelope(led).render())["tier"], "exact-all-x")

    def test_render_is_deterministic(self):
        led = E.Ledger()
        led.earn("static", "a")
        rows = [{"k": i, "v": f"x{i}"} for i in range(50)]
        self.assertEqual(envelope(led, rows=rows).render(), envelope(led, rows=rows).render())


class TierDisciplineTests(unittest.TestCase):
    def test_order(self):
        order = ["static", "sampled(10)", "exact-all-x", "engine-parity", "official-local", "official-hosted"]
        self.assertEqual([E.tier_rank(t) for t in order], list(range(6)))

    def test_refuses_higher_claim_tier(self):
        led = E.Ledger()
        led.earn("sampled(4000)", "sample search")
        with self.assertRaises(E.TierError):
            envelope(led, tier="exact-all-x")
        envelope(led, tier="static")  # lower is allowed

    def test_refuses_row_or_fact_above_computed(self):
        led = E.Ledger()
        led.earn("static", "a")
        with self.assertRaises(E.TierError):
            envelope(led, rows=[{"status": "PASS", "tier": "official-local"}])
        with self.assertRaises(E.TierError):
            envelope(led, facts={"tier": "engine-parity"})

    def test_nothing_computed_no_tier(self):
        with self.assertRaises(E.TierError):
            envelope(E.Ledger())

    def test_unknown_and_unsized_tiers(self):
        led = E.Ledger()
        for bad in ("sampled", "proof", "official", "sampled(n)"):
            with self.assertRaises(E.TierError):
                led.earn(bad, "x")


class OutputLimitTests(unittest.TestCase):
    def setUp(self):
        self.led = E.Ledger()
        self.led.earn("static", "a")
        self.rows = [{"wire": f"q{i}", "interval": f"q{i}#0", "idle": f"{i}..{i + 1000}", "runs": 63}
                     for i in range(1000)]

    def test_default_caps(self):
        text = envelope(self.led, rows=self.rows, table="work/x/full.tsv").render()
        self.assertLessEqual(len(text.encode()), E.MAX_BYTES)
        d = json.loads(text)
        self.assertLessEqual(d["rows_shown"], E.MAX_ROWS)
        self.assertEqual(d["rows_total"], 1000)
        self.assertEqual(d["table"], "work/x/full.tsv")
        self.assertEqual(d["rows"], self.rows[:d["rows_shown"]])

    def test_small_rows_capped_at_twenty(self):
        rows = [{"i": i} for i in range(30)]
        d = json.loads(envelope(self.led, rows=rows).render())
        self.assertEqual(d["rows_shown"], 20)

    def test_all_prints_everything(self):
        d = json.loads(envelope(self.led, rows=self.rows).render(all_rows=True))
        self.assertEqual(d["rows_shown"], 1000)
        self.assertNotIn("table", d)

    def test_oversized_header_refused(self):
        with self.assertRaises(ValueError):
            envelope(self.led, claim="x" * 3000).render()

    def test_next_and_facts_dropped_before_overflow(self):
        env = envelope(self.led, claim="y" * 1200, facts={"f": "z" * 500}, next_cmds=["n" * 400])
        self.assertLessEqual(len(env.render().encode()), E.MAX_BYTES)


class AddressTests(unittest.TestCase):
    def test_round_trip(self):
        for text in ("op@0766a111a76f:17552", "@17552", "q23", "q23#2", "R17"):
            self.assertEqual(E.format_address(E.parse_address(text)), text)
        self.assertEqual(E.format_address(E.parse_address("coeff_lookup", "stage")), "stage coeff_lookup")
        self.assertEqual(E.format_address(E.parse_address("extract/epoch3", "tag")), "tag extract/epoch3")

    def test_bare_forms_need_noun(self):
        self.assertEqual(E.parse_address("23", "wire"), {"kind": "wire", "wire": 23})
        self.assertEqual(E.parse_address("17", "run"), {"kind": "run", "run": 17})
        with self.assertRaises(ValueError):
            E.parse_address("23")
        with self.assertRaises(ValueError):
            E.parse_address("op@0766a111:5")  # sha part must be 12 hex digits

    def test_run_numbering_op_order_and_stable(self):
        prof = [1, 3, 3, 2, 3, 1, 3, 3, 3]
        runs = E.number_runs(prof, 3)
        self.assertEqual(runs, [(1, 2), (4, 4), (6, 8)])
        self.assertEqual(runs, E.number_runs(list(prof), 3))
        self.assertEqual(E.number_runs(prof, 2), [(1, 4), (6, 8)])

    def test_interval_numbering_independent_of_input_order(self):
        ivs = [(5, 30, 40), (2, 0, 9), (5, 0, 10), (5, 12, 20)]
        a = E.number_intervals(ivs)
        b = E.number_intervals(list(reversed(ivs)))
        self.assertEqual(a, b)
        self.assertEqual([a[(5, 0)], a[(5, 12)], a[(5, 30)]], [0, 1, 2])


SYNTH = """APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
REGISTER r0 "y"
CX q0 q2
CCX q0 q1 q3
CX q1 q4
CX q3 q5
CX q3 q5
CX q1 q4
CCX q0 q1 q3
CX q0 q2
"""


class CliHelpTest(unittest.TestCase):
    def test_help_runs_without_numpy_and_lists_verbs(self):
        out = subprocess.run([sys.executable, str(SCRIPTS / "qhdl.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        for verb in ("open", "peak", "exec", "profile", "runs", "idle", "explain", "recolour", "evidence", "verify"):
            self.assertIn(verb, out.stdout)
        self.assertNotIn("link", out.stdout)  # hidden until the viewer honours its parameters


@unittest.skipUnless(HAS_NUMPY, "NumPy not installed (stdlib-only env); run with system python3")
class WorkspaceStabilityTest(unittest.TestCase):
    def run_cli(self, work, *args):
        env = dict(os.environ, QHDL_WORK=str(work))
        out = subprocess.run([sys.executable, str(SCRIPTS / "qhdl.py"), *args], capture_output=True, text=True,
                             env=env)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertLessEqual(len(out.stdout.encode()), E.MAX_BYTES)
        return out.stdout.replace(str(work), "WORK")

    def test_same_circuit_same_addresses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "synth.kmx"
            src.write_text(SYNTH)
            outs, tables = [], []
            for k in range(2):
                work = tmp / f"work{k}"
                self.run_cli(work, "open", str(src))
                outs.append(self.run_cli(work, "runs", "-c", str(src)))
                outs.append(self.run_cli(work, "idle", "-c", str(src)))
                sha12 = E.file_sha(src, 12)
                tables.append([(work / sha12 / f).read_bytes() for f in ("ops.tsv", "intervals.tsv")])
            self.assertEqual(outs[0], outs[2])
            self.assertEqual(outs[1], outs[3])
            self.assertEqual(tables[0], tables[1])
            d = json.loads(outs[0])
            self.assertEqual(d["tier"], "static")
            self.assertTrue(all(r["run"] == f"R{i}" for i, r in enumerate(d["rows"])))


if __name__ == "__main__":
    unittest.main()
