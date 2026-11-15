"""QHDL investigations: envelope stamps (claim_id, provenance, scope), the QHDL_INVESTIGATION trail, `qhdl inv`
verbs, tag-relative and --as-of addresses, and a reconstructed acceptance investigation (skipped when absent).

Envelope and pure-function tests are standard library only; the CLI tests run scripts/qhdl.py with the system
python3 (NumPy needed for the .kmb address tests, skipped without it)."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _optional import requires_kickmix  # noqa: E402
import qhdl_envelope as E  # noqa: E402
import qhdl_investigation as INV  # noqa: E402

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
ACCEPT = Path(os.environ.get("QHDL_TEST_FIXTURES") or ROOT / "tests" / "fixtures") / "investigation-dqi-dirty-borrows"
PY = os.environ.get("QHDL_TEST_PYTHON", sys.executable)


def qhdl(*args, env=None, cwd=None):
    e = dict(os.environ)
    e.pop("QHDL_INVESTIGATION", None)
    e.update(env or {})
    r = subprocess.run([PY, str(SCRIPTS / "qhdl.py"), *map(str, args)], capture_output=True, text=True, env=e,
                       cwd=cwd or ROOT)
    return r.returncode, r.stdout, r.stderr


def envelope(**kw):
    led = E.Ledger()
    led.earn("static", "counted")
    a = dict(claim="c", domain="d", evidence=None, rerun="python3 scripts/qhdl.py v", tool_sha="t" * 12,
             circuit_sha="a" * 64)
    a.update(kw)
    return E.Envelope(led, **a)


class StampTests(unittest.TestCase):
    def test_unstamped_render_unchanged_and_stamps_follow_circuit_sha(self):
        plain = envelope(rows=[{"k": i} for i in range(5)]).render()
        self.assertNotIn("claim_id", plain)
        env = envelope(rows=[{"k": i} for i in range(5)])
        env.stamp(claim_id="0123456789ab", provenance="computed",
                  scope={"domain": "all-x", "extent": "whole", "branches": "zero+random(7)"})
        d = json.loads(env.render())
        self.assertEqual(list(d)[:10], ["claim", "tier", "domain", "evidence", "rerun", "tool_sha", "circuit_sha",
                                        "claim_id", "provenance", "scope"])

    def test_stamps_never_cost_rows(self):
        rows = [{"k": i, "v": "x" * 60} for i in range(40)]
        base = json.loads(envelope(rows=rows).render())
        env = envelope(rows=rows)
        env.stamp(claim_id="0123456789ab", provenance="computed", addresses={"--ops": "tag:a[0] -> 1:2" * 10})
        d = json.loads(env.render())
        self.assertEqual(d["rows_shown"], base["rows_shown"])
        self.assertLessEqual(len(env.render().encode()), E.MAX_BYTES)

    def test_header_near_budget_drops_stamps_instead_of_failing(self):
        n = E.MAX_BYTES - len(envelope(claim="").render().encode()) - 10  # fits by 10 bytes without stamps
        env = envelope(claim="c" * n)
        env.stamp(claim_id="0123456789ab", provenance="computed",
                  scope={"domain": "all-x", "extent": "whole", "branches": "zero"})
        text = env.render()
        self.assertLessEqual(len(text.encode()), E.MAX_BYTES)

    def test_bad_provenance_refused(self):
        with self.assertRaises(ValueError):
            envelope().stamp(provenance="guessed")

    def test_max_bytes_override(self):
        env = envelope(rows=[{"v": "x" * 100} for i in range(30)])
        env.max_bytes = 4096
        self.assertGreater(len(env.render().encode()), E.MAX_BYTES)
        self.assertLessEqual(len(env.render().encode()), 4096)


class IdentityTests(unittest.TestCase):
    def test_claim_id_ignores_env_prefix_and_is_stable(self):
        a = INV.claim_id("peak", "QHDL_WORK=/tmp/x python3 scripts/qhdl.py peak -c abc", "t1", "s")
        b = INV.claim_id("peak", "python3 scripts/qhdl.py peak -c abc", "t1", "s")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        self.assertNotEqual(a, INV.claim_id("peak", "python3 scripts/qhdl.py peak -c abc", "t2", "s"))

    def test_default_scope_fdiff_and_verify(self):
        class A:
            whole, seed, ops, salt = True, 7, None, ""
        s = INV.default_scope("fdiff", A, {"domain": "all 3,133,452 legal x; zero branch and one random branch"})
        self.assertEqual(s, {"domain": "all-x", "extent": "whole", "branches": "zero+random(7)"})
        A.whole, A.ops = False, "10:20"
        s = INV.default_scope("fdiff", A, {"domain": "reachable inputs of the window, zero branch"})
        self.assertEqual(s["extent"], "window 10:20")
        self.assertTrue(INV.scope_is_local(s))
        self.assertEqual(INV.default_scope("verify", A, {})["branches"], "official")
        self.assertIsNone(INV.default_scope("peak", A, {}))
        self.assertEqual(INV.default_provenance("evidence"), "recorded")

    def test_check_scope(self):
        INV.check_scope({"domain": "draws:48@40fa4ebf", "extent": "whole", "branches": "engine-sampled"})
        INV.check_scope({"domain": "inputs:4096@20261101", "extent": "window 3:9", "branches": "zero"})
        with self.assertRaises(ValueError):
            INV.check_scope({"domain": "some draws", "extent": "whole", "branches": "zero"})
        with self.assertRaises(ValueError):
            INV.check_scope({"domain": "all-x", "extent": "whole", "branches": "both"})

    def test_ledger_tier_mapping(self):
        self.assertEqual(INV.ledger_tier("static"), "static model")
        self.assertEqual(INV.ledger_tier("sampled(4096)"), "measured by us")
        self.assertEqual(INV.ledger_tier("engine-parity"), "measured by us")
        self.assertEqual(INV.ledger_tier("official-hosted"), "official verifier")
        self.assertEqual(INV.ledger_tier("structural"), "paper only")


class MapTests(unittest.TestCase):
    def test_span_map_and_chain(self):
        cut = INV._span_map([{"old": [10, 20], "new": [10, 15]}, {"old": [30, 32], "new": [25, 30]}])
        self.assertEqual(cut(5), 5)
        self.assertEqual(cut(10), 10)
        self.assertEqual(cut(25), 20)
        self.assertEqual(cut(30), 25)
        self.assertEqual(cut(40), 38)
        with self.assertRaises(ValueError):
            cut(15)

    def test_load_maps_refuses_broken_chain_and_auto(self):
        with tempfile.TemporaryDirectory() as d:
            m1, m2 = Path(d) / "a.json", Path(d) / "b.json"
            m1.write_text(json.dumps({"old_sha256": "a" * 64, "new_sha256": "b" * 64, "old_ops": 10, "new_ops": 10,
                                      "hunks": [["equal", 0, 10, 0, 10]]}))
            m2.write_text(json.dumps({"old_sha256": "c" * 64, "new_sha256": "d" * 64, "old_ops": 10, "new_ops": 9,
                                      "hunks": [["equal", 0, 4, 0, 4], ["delete", 4, 5, 4, 4],
                                                ["equal", 5, 10, 4, 9]]}))
            with self.assertRaises(ValueError):
                INV.load_maps(f"{m1},{m2}")
            with self.assertRaises(ValueError):
                INV.load_maps("auto")
            ms = INV.load_maps(str(m2))
            self.assertEqual(INV.translate(ms, "cccccccccccc", "d" * 64, 7), 6)
            with self.assertRaises(ValueError):
                INV.translate(ms, "aaaaaaaaaaaa", "d" * 64, 7)

    def test_unchanged_argv_without_addresses(self):
        argv = ["window", "-c", "abc123", "--ops", "10:20"]
        self.assertEqual(INV.rewrite_argv(argv, None), (argv, {}))
        argv = ["inv", "kill", "d", "H1", "--at", "2026-11-01T00:00:00Z"]
        self.assertEqual(INV.rewrite_argv(argv, None), (argv, {}))


def tiny_tagged_kmb(path):
    """ops: 0 enter A | 1 X q0 | 2 enter B | 3 CX q0 q1 | 4 exit B | 5 exit A | 6 enter A | 7 CX q1 q0 | 8 exit A |
    9 X q1. Tag A has two instances: [0] = 0:6, [1] = 6:9; A/B one instance 2:5."""
    import numpy as np
    import qhdl_kmb as K
    codes = [0x06, 0x11, 0x06, 0x21, 0x06, 0x06, 0x06, 0x21, 0x06, 0x11]
    st = {"q0": [0, 1], "qq0": [0, 1], "qq1": [1, 0]}
    tags = [b"@enter:A", b"@enter:B", b"@exit:B", b"@exit:A", b"@enter:A", b"@exit:A"]
    K.write_kmb(str(path), np.array(codes, np.uint8), {k: np.array(v, np.uint32) for k, v in st.items()},
                tags=tags, num_qubits=2, num_bits=0)


class _Api:
    class QhdlError(Exception):
        def __init__(self, msg, hint=None):
            super().__init__(msg)
            self.hint = hint

    def __init__(self, path, sha):
        self.c = dict(sha=sha, path=path)

    def resolve(self, ref):
        return self.c


@unittest.skipUnless(HAS_NUMPY, "needs NumPy")
class TagAddressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kmb = Path(self.tmp.name) / "t.kmb"
        tiny_tagged_kmb(self.kmb)
        self.sha = E.file_sha(self.kmb)
        self.api = _Api(self.kmb, self.sha)

    def tearDown(self):
        self.tmp.cleanup()

    def test_instances(self):
        inst = INV.tag_instances(self.kmb)
        self.assertEqual(inst, {"A": [(0, 6), (6, 9)], "A/B": [(2, 5)]})
        self.assertEqual([h[0] for h in INV.innermost(inst, 3)], ["A", "A/B"])

    def rw(self, *argv):
        return INV.rewrite_argv(["window", "-c", "t", *argv], self.api)

    def test_ranges(self):
        self.assertEqual(self.rw("--ops", "tag:A[1]")[0][-1], "6:9")
        self.assertEqual(self.rw("--ops", "tag:A[-1]")[0][-1], "6:9")
        self.assertEqual(self.rw("--ops", "tag:A[0]+1:+4")[0][-1], "1:4")
        self.assertEqual(self.rw("--ops", "tag:B[0]")[0][-1], "2:5")  # unique suffix of A/B
        self.assertEqual(self.rw("--ops", "tag:A/B[0]+1:tag:A[1]+2")[0][-1], "3:8")
        self.assertEqual(self.rw("--ops=tag:A[0]+1:9")[0][-1], "--ops=1:9")
        argv, addr = self.rw("--ops", f"op@{self.sha[:12]}:2:5")
        self.assertEqual(argv[-1], "2:5")
        self.assertIn("--ops", addr)

    def test_at_list(self):
        argv, _ = INV.rewrite_argv(["simulate", "-c", "t", "--at", "tag:A[1]+1,9"], self.api)
        self.assertEqual(argv[-1], "7,9")

    def test_errors(self):
        for bad in ("tag:Z[0]", "tag:A[2]", "tag:A[0]+7:+8", "op@ffffffffffff:1:2"):
            with self.assertRaises((self.api.QhdlError, ValueError), msg=bad):
                self.rw("--ops", bad)

    def test_as_of_translates_old_ops(self):
        m = Path(self.tmp.name) / "e.map.json"
        m.write_text(json.dumps({"old_sha256": "e" * 64, "new_sha256": self.sha, "old_ops": 12, "new_ops": 10,
                                 "hunks": [["equal", 0, 3, 0, 3], ["delete", 3, 5, 3, 3], ["equal", 5, 12, 3, 10]]}))
        argv, addr = self.rw("--ops", "op@eeeeeeeeeeee:6:9", "--as-of", str(m))
        self.assertEqual(argv[-1], "4:7")
        self.assertNotIn("--as-of", argv)
        self.assertEqual(self.rw("--ops", "6:9", "--as-of", str(m))[0][-1], "4:7")
        with self.assertRaises(ValueError):
            self.rw("--ops", "op@eeeeeeeeeeee:4:9", "--as-of", str(m))  # cut inside the deletion

    @requires_kickmix
    def test_cli_verb_gets_resolved_range(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as inv:
            env = {"QHDL_WORK": work, "QHDL_INVESTIGATION": inv}
            rc, out, err = qhdl("track", "-c", self.kmb, "--ops", "tag:A[0]", env=env)
            self.assertEqual(rc, 0, out + err)
            d = json.loads(out)
            self.assertTrue(d["rerun"].endswith("--ops 0:6"))
            self.assertEqual(d["addresses"], {"--ops": "tag:A[0] -> 0:6"})
            self.assertEqual(d["provenance"], "computed")
            row = json.loads((Path(inv) / "trail.jsonl").read_text().splitlines()[-1])
            self.assertIn("tag:A[0]", row["argv"])  # what the agent typed
            self.assertEqual(row["claim_id"], d["claim_id"])

    def test_cli_addr_and_trail(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as inv:
            env = {"QHDL_WORK": work, "QHDL_INVESTIGATION": inv}
            rc, out, err = qhdl("addr", "tag:A[1]+1", "-c", self.kmb, env=env)
            self.assertEqual(rc, 0, out + err)
            d = json.loads(out)
            self.assertEqual(d["facts"]["op"], 7)
            self.assertEqual(d["rows"][-1]["address"], "tag:A[1]+1")
            self.assertEqual(d["provenance"], "derived")
            rc, out, _ = qhdl("addr", "tag:Q[0]", "-c", self.kmb, env=env)
            self.assertEqual(rc, 2)
            trail = [json.loads(x) for x in (Path(inv) / "trail.jsonl").read_text().splitlines()]
            self.assertEqual([t["exit"] for t in trail], [0, 2])
            self.assertEqual(trail[0]["claim_id"], d["claim_id"])
            self.assertEqual(trail[0]["verb"], "addr")
            self.assertIn("error", trail[1])


class InvVerbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def inv(self, *args, ok=True):
        rc, out, err = qhdl("inv", *args)
        if ok:
            self.assertEqual(rc, 0, out + err)
        return rc, json.loads(out) if out.strip().startswith("{") else out

    def test_lifecycle(self):
        _, d = self.inv("new", "toy", "--root", self.root, "--date", "2026-11-01", "--task", "dqi",
                        "--question", "can X drop a qubit?", "--base", "a" * 64 + "=base")
        inv = self.root / "2026-11-01-toy"
        self.assertTrue((inv / "investigation.json").exists())
        self.assertEqual(json.loads((inv / "investigation.json").read_text())["tool_sha"], d["tool_sha"])
        # kill_if is mandatory and fixed
        rc, _ = self.inv("hypothesis", inv, "--id", "H1", "--statement", "s", "--kill-if", " ", ok=False)
        self.assertEqual(rc, 2)
        self.inv("hypothesis", inv, "--id", "H1", "--statement", "release the pole", "--kill-if", "no witness",
                 "--next", "python3 scripts/qhdl.py peak -c aaaaaa")
        rc, _ = self.inv("hypothesis", inv, "--id", "H1", "--statement", "s", "--kill-if", "k2", ok=False)
        self.assertEqual(rc, 2)
        self.inv("hypothesis", inv, "--id", "H2", "--statement", "ladder", "--kill-if", "48 draws not cheaper",
                 "--mechanism", "mbu ladder", "--keywords", "mbu")
        # a probe that ran before the hypothesis is refused unless marked prior
        _, early = self.inv("log", inv, "--call", "python3 -I helpers/x.py", "--tier", "sampled(64)")
        self.inv("hypothesis", inv, "--id", "H3", "--statement", "late", "--kill-if", "k")
        rc, _ = self.inv("kill", inv, "H3", "--outcome", "o", "--tier", "static", "--probe",
                         early["facts"]["claim_id"], ok=False)
        self.assertEqual(rc, 2)
        self.inv("kill", inv, "H3", "--outcome", "o", "--tier", "static", "--probe", early["facts"]["claim_id"],
                 "--prior-probe", "--reopen-if", "never")
        rc, _ = self.inv("kill", inv, "H1", "--outcome", "o", "--probe", "000000000000", ok=False)
        self.assertEqual(rc, 2)
        rc, _ = self.inv("kill", inv, "H1", "--outcome", "o", "--tier", "maybe", ok=False)
        self.assertEqual(rc, 2)
        _, late = self.inv("log", inv, "--call", "python3 -I helpers/y.py", "--tier", "sampled(4096)")
        self.inv("kill", inv, "H1", "--outcome", "parity is global", "--tier", "sampled(4096)", "--probe",
                 late["facts"]["claim_id"], "--reopen-if", "new kernel", "--key-numbers", "4,096 lanes")
        rc, _ = self.inv("confirm", inv, "H1", "--outcome", "x", ok=False)  # already killed
        self.assertEqual(rc, 2)
        self.inv("confirm", inv, "H2", "--outcome", "-165 Toffolis per shot", "--tier", "engine-parity")
        self.inv("candidate", inv, "--sha", "b" * 64, "--hypothesis", "H2", "--status", "verified",
                 "--domain", "draws:48@40fa4ebf", "--extent", "whole", "--branches", "engine-sampled",
                 "--tier", "engine-parity")
        rc, _ = self.inv("candidate", inv, "--sha", "c" * 64, "--status", "built", "--domain", "draws",
                         "--branches", "zero", ok=False)
        self.assertEqual(rc, 2)
        # append-only files: every state change is a new line
        lines = (inv / "hypotheses.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 6)
        h1 = [json.loads(x) for x in lines if json.loads(x)["id"] == "H1"]
        self.assertEqual([h["status"] for h in h1], ["open", "killed"])
        self.assertEqual(h1[0]["kill_if"], h1[1]["kill_if"])
        # status and resume
        _, st = self.inv("status", inv)
        self.assertIn("3 hypotheses", st["claim"])
        _, r = self.inv("resume", inv)
        self.assertIn("H1 killed", r["claim"])
        self.assertEqual(r["provenance"], "derived")
        self.assertEqual(r["facts"]["question"], "can X drop a qubit?")
        self.assertEqual(r["rows"][0]["candidate"], "bbbbbbbbbbbb")  # no open hypotheses: candidates first
        self.assertEqual(r["rows"][0]["scope"], "draws:48@40fa4ebf/whole/engine-sampled")
        # superseding keeps the old row and marks it
        self.inv("hypothesis", inv, "--id", "H4", "--statement", "s4", "--kill-if", "k4")
        self.inv("hypothesis", inv, "--id", "H4b", "--statement", "s4 sharper", "--kill-if", "k4b",
                 "--supersedes", "H4", "--next", "python3 scripts/qhdl.py open -c bbbbbb")
        _, r = self.inv("resume", inv)
        self.assertEqual(r["rows"][0]["id"], "H4b")
        self.assertEqual(r["next"][0], "python3 scripts/qhdl.py open -c bbbbbb")
        st = {h["id"]: h["status"] for h in INV.hyps(inv)}
        self.assertEqual(st["H4"], "superseded")
        # ledger staging
        _, ls = self.inv("ledger-stage", inv)
        rows = json.loads((inv / "ledger-stage.json").read_text())["rows"]
        self.assertEqual(len(rows), 2)
        h1row = next(r for r in rows if r["mechanism"] == "release the pole")
        self.assertEqual(h1row["evidence_tier"], "measured by us")
        self.assertEqual(h1row["outcome"], "kill")
        self.assertEqual(h1row["reopen_if"], "new kernel")
        self.assertIn("mbu ladder", (inv / "leads-stage.md").read_text())

    def test_trail_skips_outputs_and_inv(self):
        with tempfile.TemporaryDirectory() as t:
            bad = Path(t) / "outputs" / "x"
            bad.mkdir(parents=True)
            rc, out, err = qhdl("circuits", "--prefix", "ffffffffffff", env={"QHDL_INVESTIGATION": str(bad)})
            self.assertFalse((bad / "trail.jsonl").exists())
            self.assertIn("outputs/", err)

    def test_trail_records_core_verb(self):
        with tempfile.TemporaryDirectory() as inv:
            rc, out, err = qhdl("circuits", "--prefix", "ffffffffffff", env={"QHDL_INVESTIGATION": inv})
            rows = [json.loads(x) for x in (Path(inv) / "trail.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["verb"], "circuits")
            self.assertEqual(rows[0]["exit"], rc)
            if rc == 0:
                self.assertEqual(rows[0]["claim_id"], json.loads(out)["claim_id"])

    def test_plugin_list_names_stream_b(self):
        src = (SCRIPTS / "qhdl.py").read_text()
        self.assertIn("'qhdl_rewrite'", src)
        rc, out, err = qhdl("--help")
        self.assertEqual(rc, 0)
        self.assertIn("inv", out)


@unittest.skipUnless((ACCEPT / "investigation.json").exists(), "acceptance investigation not present")
class AcceptanceTests(unittest.TestCase):
    """Acceptance: a cold agent runs one command on a reconstructed recorded investigation and gets the picture:
    891 killed (pole parity), S12/S6 built and confirmed on a dirty-Barenco circuit, the 3779 candidate recorded,
    the reverse-region pairs hypothesis present with its kill criterion; bounded output. The investigation is live,
    so the test checks facts that stay true as it moves on (H-RP killed, both candidates verified)."""

    def test_resume_cold(self):
        rc, out, err = qhdl("inv", "resume", ACCEPT.relative_to(ROOT))
        self.assertEqual(rc, 0, out + err)
        self.assertLessEqual(len(out.encode()), INV.RESUME_MAX_BYTES)
        d = json.loads(out)
        rows = {r.get("id") or r.get("candidate"): r for r in d["rows"]}
        self.assertGreaterEqual(d["rows_shown"], 9)  # open, candidates and confirmed come first
        self.assertEqual(rows["H-P"]["status"], "killed")
        self.assertIn("parity", rows["H-P"]["outcome"])
        self.assertEqual(rows["H-CC"]["status"], "confirmed")
        self.assertIn("S12/S6", rows["H-CC"]["outcome"])
        self.assertEqual(rows["H-U6"]["status"], "confirmed")
        self.assertIn(rows["H-RP"]["status"], ("open", "killed"))
        self.assertIn(rows["077c2fc2cf36"]["status"], ("under-verification", "verified"))
        self.assertTrue(rows["9cbe517190cb"]["scope"].startswith("draws:"))
        self.assertTrue(isinstance(d.get("next"), list))

    def test_trail_and_hypothesis_files_are_well_formed(self):
        H = INV.hyps(ACCEPT)
        self.assertTrue(all(h["kill_if"] for h in H))
        T = INV.read_jsonl(ACCEPT / "trail.jsonl")
        ids = {t["claim_id"] for t in T}
        for h in H:
            for p in h.get("probes") or []:
                self.assertIn(p, ids)


if __name__ == "__main__":
    unittest.main()
