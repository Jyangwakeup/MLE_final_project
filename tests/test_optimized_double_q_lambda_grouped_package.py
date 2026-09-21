"""Official-style package smoke test for a trained grouped agent."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from experiments.package_agent import build_submission
from tests import test_submission_package as package_tests


class GroupedAgentPackageTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("GROUPED_SMOKE_CHECKPOINT"), "requires smoke checkpoint")
    def test_official_style_reload(self):
        agent = "optimized_double_q_lambda_grouped_agent"
        checkpoint = Path(os.environ["GROUPED_SMOKE_CHECKPOINT"]).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "restore"):
                build_submission(agent, checkpoint, root / "agent.zip")


if __name__ == "__main__":
    unittest.main()
