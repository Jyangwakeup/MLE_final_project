"""Run the smoke checkpoint without repository shared packages."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from experiments.package_agent import build_submission
from tests import test_submission_package as package_tests


class CratePackageTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CRATE_SMOKE_CHECKPOINT"),
                         "requires trained smoke checkpoint")
    def test_official_style_reload(self):
        agent = "optimized_double_q_lambda_crate_agent"
        checkpoint = Path(os.environ["CRATE_SMOKE_CHECKPOINT"]).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = build_submission(agent, checkpoint, root / "agent.zip")
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(root / "unpacked")
            helper = package_tests.SubmissionPackageTest()
            isolated, env = helper._prepare_isolated_root(
                root, root / "unpacked", agent, "continuous-v2",
                "r7_safe_credit_potential")
            result = subprocess.run(
                [sys.executable, "main.py", "play", "--agents", agent,
                 "--scenario", "coin-heaven", "--n-rounds", "1", "--no-gui"],
                cwd=isolated, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
