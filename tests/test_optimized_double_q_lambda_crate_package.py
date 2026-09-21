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
            with self.assertRaisesRegex(ValueError, "restore"):
                build_submission(agent, checkpoint, root / "agent.zip")
