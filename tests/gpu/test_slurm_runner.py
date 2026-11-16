"""Exercise environment setup and automatic reporting without Slurm or a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class SlurmRunnerTests(unittest.TestCase):
    def run_mock_job(self, directory, mode, failure=""):
        root = Path(directory)
        scripts = root / "tests/gpu"
        scripts.mkdir(parents=True)
        for name in ("run.sh", "setup_env.sh", "report.py", "run_suite.py"):
            shutil.copy(Path(__file__).parent / name, scripts / name)
        # Substitute only the costly GPU executable runner; keep reporting real.
        (scripts / "run_suite.py").write_text(
            "import json, os, pathlib, sys\n"
            "passed = os.environ.get('MOCK_FAIL') != 'suite'\n"
            "result = {'sha': os.environ['CAESAR_GPU_SHA'], 'backend': 'nvidia', "
            "'mode': sys.argv[1], 'passed': passed, 'tests': [{'passed': passed}], "
            "'shape': '256,256,256,256', 'error_bound': 0.1, 'performance': "
            "{op: {'runs': [{'seconds': 1}] * 5} for op in ('compression', 'decompression')}}\n"
            "pathlib.Path(os.environ['CAESAR_GPU_RESULT']).write_text(json.dumps(result))\n"
            "sys.exit(0 if passed else 1)\n"
        )
        # Importing a substitute runner must not execute it: use the real
        # module under a separate path when report.py validates performance.
        if mode == "performance":
            shutil.copy(Path(__file__).parent / "run_suite.py", scripts / "gate.py")
            report = scripts / "report.py"
            report.write_text(
                report.read_text().replace("from run_suite import", "from gate import")
            )
        (root / ".bashrc").write_text('export SETUP_ORDER="bashrc"\n')
        site = root / "site.sh"
        site.write_text('export SETUP_ORDER="$SETUP_ORDER,site"\n')
        venv = root / "venv/bin"
        venv.mkdir(parents=True)
        (venv / "activate").write_text('export SETUP_ORDER="$SETUP_ORDER,venv"\n')
        commands = root / "commands"
        commands.mkdir()
        mocks = {
            "git": '#!/bin/bash\nif [[ "$*" == *show-toplevel* ]]; then echo "$MOCK_ROOT"; else printf "%040d\\n" 0; fi\n',
            "python3": '#!/bin/bash\n[[ "$SETUP_ORDER" == "bashrc,site,venv" && "$PYTHONNOUSERSITE" == 1 ]] || exit 91\nif [[ "${1:-}" == -c ]]; then\n  if [[ "$2" == *cmake_prefix_path* ]]; then echo /mock/torch; else echo nvidia; fi\nelse\n  exec "$REAL_PYTHON" "$@"\nfi\n',
            "cmake": '#!/bin/bash\n[[ "$MOCK_FAIL" != build ]] || exit 7\n',
            "make": '#!/bin/bash\n[[ "$*" == *-j6* ]] || exit 92\n',
            "gh": '#!/bin/bash\ncat > "$MOCK_PUBLISHED"\n',
        }
        for name, source in mocks.items():
            path = commands / name
            path.write_text(source)
            path.chmod(0o755)
        env = dict(os.environ)
        env.update(
            HOME=str(root),
            PATH=f"{commands}:{env['PATH']}",
            REAL_PYTHON=sys.executable,
            MOCK_ROOT=str(root),
            MOCK_FAIL=failure,
            MOCK_PUBLISHED=str(root / "published.json"),
            CAESAR_ENV_FILE=str(site),
            CAESAR_VENV=str(venv.parent),
            CAESAR_BUILD_DIR=str(root / "build"),
            SLURM_SUBMIT_DIR=str(root),
        )
        process = subprocess.run(
            ["bash", str(scripts / "run.sh"), mode],
            env=env,
            text=True,
            capture_output=True,
        )
        published = root / "published.json"
        self.assertTrue(published.exists(), process.stdout + process.stderr)
        return process, json.loads(published.read_text())

    def test_each_suite_reports_success_after_sourcing_all_three(self):
        for mode in ("correctness", "performance"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                process, payload = self.run_mock_job(directory, mode)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(payload["state"], "success")
                self.assertEqual(payload["context"], f"gpu-tests/nvidia/{mode}")

    def test_build_and_suite_failures_are_reported_and_remain_failures(self):
        for failure in ("build", "suite"):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                process, payload = self.run_mock_job(directory, "correctness", failure)
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(payload["state"], "failure")


if __name__ == "__main__":
    unittest.main()
