import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qhdl_identities as I  # noqa: E402


class IdentityRegistryTests(unittest.TestCase):
    def test_registry_validates(self):
        self.assertEqual(I.cmd_validate(I.load()), 0)

    def test_measure_first_missing_fixup_fails(self):
        ok, _ = I.check_fixture(["CX q0 q2", "CCX q0 q1 q2", "HMR q2 b0"], ["HMR q2 b0", "CZ q0 q1 if b0"])
        self.assertFalse(ok)   # the Z q0 fix-up for the CX is missing

    def test_mbu_needs_clean_temporary(self):
        lhs = ["CCX q0 q1 q2", "CX q2 q3", "CCX q0 q1 q2"]
        rhs = ["CCX q0 q1 q2", "CX q2 q3", "HMR q2 b0", "CZ q0 q1 if b0"]
        self.assertTrue(I.check_fixture(lhs, rhs, ["q2"])[0])
        self.assertFalse(I.check_fixture(lhs, rhs)[0])   # without the clean-entry precondition it is false

    def test_wrong_merge_fails(self):
        self.assertFalse(I.check_fixture(["CCX q0 q1 q3", "CCX q0 q2 q3"], ["CCX q0 q1 q3"])[0])

    def test_ids_unique(self):
        ids = [e["id"] for e in I.load()["identities"]]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
