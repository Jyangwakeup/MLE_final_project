"""Run a trained history-agent package without repository shared modules."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from experiments.package_agent import build_submission
from tests import test_submission_package as package_tests


class HistoryAgentPackageTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("HISTORY_SMOKE_CHECKPOINT"),
        "requires a trained history-agent smoke checkpoint",
    )
    def test_official_style_reload(self):
        agent = "optimized_double_q_lambda_history_agent"
        checkpoint = Path(os.environ["HISTORY_SMOKE_CHECKPOINT"]).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = build_submission(agent, checkpoint, root / "agent.zip")
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(root / "unpacked")
            helper = package_tests.SubmissionPackageTest()
            isolated, environment = helper._prepare_isolated_root(
                root, root / "unpacked", agent, "continuous-v2",
                "r20_safe_credit_targeted_wait")
            result = subprocess.run(
                [sys.executable, "main.py", "play", "--agents", agent,
                 "--scenario", "coin-heaven", "--n-rounds", "1", "--no-gui"],
                cwd=isolated, env=environment, capture_output=True, text=True,
                timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])


if __name__ == "__main__":
    unittest.main()
