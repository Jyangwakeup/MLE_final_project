"""Submission packaging regression for the demonstration agent."""

from pathlib import Path
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile

from agent_code.learning_common.runtime import CHECKPOINT_SCHEMA
from experiments.agent_variants.optimized_double_q_lambda_demo_agent import callbacks
from agent_code.team_agent.rewards import resolve_reward_spec
from agent_code.team_agent.safety import resolve_safety_spec
from experiments.package_agent import build_submission
from tests import test_submission_package as package_tests


class DemoPackageTests(unittest.TestCase):
    def test_package_does_not_require_teacher_or_dataset(self):
        model = callbacks.make_model(11)
        payload = model.checkpoint()
        payload.update({
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "algorithm": callbacks.ALGORITHM,
            "actions": list(callbacks.ACTIONS),
            "feature_id": callbacks.FEATURE_ID,
            "feature_schema": callbacks.FEATURE_SCHEMA,
            "reward_id": "r20_safe_credit_targeted_wait",
            "reward_version": "r20_safe_credit_targeted_wait",
            "reward_spec": resolve_reward_spec("r20_safe_credit_targeted_wait"),
            "hyperparameters": callbacks.HYPERPARAMETERS,
            "network_spec": None,
            "safety_spec": resolve_safety_spec({
                "version": "survival-mask-v1", "mode": "all",
                "horizon": 7, "fallback": "physical_q"}),
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, archive = root / "model.pkl", root / "agent.zip"
            with checkpoint.open("wb") as file:
                pickle.dump(payload, file)
            with self.assertRaisesRegex(ValueError, "restore"):
                build_submission(
                    "optimized_double_q_lambda_demo_agent", checkpoint, archive)

    @unittest.skipUnless(
        os.environ.get("DEMO_SMOKE_CHECKPOINT"),
        "requires a trained demonstration-agent smoke checkpoint",
    )
    def test_official_style_reload(self):
        agent = "optimized_double_q_lambda_demo_agent"
        checkpoint = Path(os.environ["DEMO_SMOKE_CHECKPOINT"]).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "restore"):
                build_submission(agent, checkpoint, root / "agent.zip")


if __name__ == "__main__":
    unittest.main()
