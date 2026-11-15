"""A successful health probe must emit a command using that same runtime."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RerunRuntime(unittest.TestCase):
    def test_doctor_reruns_from_another_directory(self):
        with tempfile.TemporaryDirectory(prefix='qhdl runtime ') as td:
            env = dict(os.environ, QHDL_WORK=str(Path(td) / 'work with spaces'))
            first = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/qhdl.py'), 'doctor', '--all'],
                env=env, cwd=td, capture_output=True, text=True, check=True, timeout=120)
            report = json.loads(first.stdout)
            args = shlex.split(report['rerun'])
            self.assertEqual(args.pop(0), 'QHDL_WORK=' + env['QHDL_WORK'])
            self.assertEqual(args[0], sys.executable)
            self.assertEqual(args[1], str(ROOT / 'scripts/qhdl.py'))
            repeat = subprocess.run(args + ['--all'], env=env, cwd=td,
                                    capture_output=True, text=True, check=True, timeout=120)
            self.assertEqual(json.loads(repeat.stdout)['claim'], report['claim'])


if __name__ == '__main__':
    unittest.main()
