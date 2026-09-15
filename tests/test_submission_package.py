import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from agent_code.double_q_compact_agent.callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS,
)
from experiments.package_agent import build_submission


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SubmissionPackageTest(unittest.TestCase):
    def test_generated_agent_runs_without_repository_shared_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "smoke.pkl"
            payload = {
                "checkpoint_schema": "training-resume-v1", "algorithm": ALGORITHM,
                "actions": list(ACTIONS), "feature_id": FEATURE_ID,
                "feature_schema": FEATURE_SCHEMA, "reward_id": "r1",
                "reward_version": "r1", "reward_spec": {},
                "hyperparameters": HYPERPARAMETERS, "network_spec": None,
                "q_table_a": {}, "q_table_b": {}, "training_steps": 0,
                "action_steps": 0, "training_task": "coin_navigation",
                "rng_state": __import__("random").Random(0).getstate(),
                "agent_rng_state": __import__("random").Random(0).getstate(),
            }
            with checkpoint.open("wb") as file:
                pickle.dump(payload, file)
            archive = build_submission(
                "double_q_compact_agent", checkpoint, root / "submission.zip")
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertIn("double_q_compact_agent/final.pkl", names)
                self.assertIn(
                    "double_q_compact_agent/_vendor/team_agent/opponent_transitions.py",
                    names,
                )
                self.assertIn(
                    "double_q_compact_agent/_vendor/dqn_model.py", names,
                )
                self.assertTrue(any("/_vendor/team_agent/feature_system/" in n for n in names))
                self.assertFalse(any("/logs/" in n for n in names))
                zipped.extractall(root / "unpacked")

            isolated = root / "official"
            isolated.mkdir()
            (isolated / "agent_code").mkdir()
            (isolated / "agent_code" / "__init__.py").touch()
            for source in PROJECT_ROOT.glob("*.py"):
                shutil.copy2(source, isolated / source.name)
            shutil.move(
                str(root / "unpacked" / "double_q_compact_agent"),
                str(isolated / "agent_code" / "double_q_compact_agent"),
            )
            (isolated / "logs").mkdir()
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(isolated)
            environment["BOMBERMAN_FEATURE_ID"] = FEATURE_ID
            environment["BOMBERMAN_REWARD_ID"] = "r1"
            result = subprocess.run(
                [sys.executable, "main.py", "play", "--agents",
                 "double_q_compact_agent", "--scenario", "coin-heaven",
                 "--n-rounds", "1", "--no-gui"],
                cwd=isolated, env=environment, capture_output=True, text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
