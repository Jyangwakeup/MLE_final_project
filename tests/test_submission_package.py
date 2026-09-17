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
    def _prepare_isolated_root(self, root, unpacked, agent, feature_id, reward_id):
        isolated = root / "official"
        isolated.mkdir()
        (isolated / "agent_code").mkdir()
        (isolated / "agent_code" / "__init__.py").touch()
        for source in PROJECT_ROOT.glob("*.py"):
            shutil.copy2(source, isolated / source.name)
        shutil.copytree(PROJECT_ROOT / "assets", isolated / "assets")
        shutil.copytree(
            PROJECT_ROOT / "agent_code" / "random_agent",
            isolated / "agent_code" / "random_agent",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.move(
            str(unpacked / agent), str(isolated / "agent_code" / agent))
        (isolated / "logs").mkdir()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(isolated)
        environment["BOMBERMAN_FEATURE_ID"] = feature_id
        environment["BOMBERMAN_REWARD_ID"] = reward_id
        return isolated, environment

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
                    "double_q_compact_agent/_vendor/team_agent/controllable_survival.py",
                    names,
                )
                self.assertIn(
                    "double_q_compact_agent/_vendor/dqn_model.py", names,
                )
                self.assertTrue(any("/_vendor/team_agent/feature_system/" in n for n in names))
                self.assertFalse(any("/logs/" in n for n in names))
                zipped.extractall(root / "unpacked")

            isolated, environment = self._prepare_isolated_root(
                root, root / "unpacked", "double_q_compact_agent", FEATURE_ID, "r1")
            result = subprocess.run(
                [sys.executable, "main.py", "play", "--agents",
                 "double_q_compact_agent", "--scenario", "coin-heaven",
                 "--n-rounds", "1", "--no-gui"],
                cwd=isolated, env=environment, capture_output=True, text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_task3_archive_contains_only_selected_weights_and_dependencies(self):
        import hashlib
        import json
        agent = "double_dqn_continuous_v2_agent"
        checkpoint = PROJECT_ROOT / "agent_code" / agent / "task3_validated.pt"
        with tempfile.TemporaryDirectory() as directory:
            archive = build_submission(agent, checkpoint, Path(directory) / "submission.zip")
            with zipfile.ZipFile(archive) as zipped:
                weights = [name for name in zipped.namelist()
                           if name.endswith((".pt", ".pkl"))]
                self.assertEqual(weights, [f"{agent}/final.pt"])
                expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                self.assertEqual(hashlib.sha256(zipped.read(weights[0])).hexdigest(), expected)
                manifest = json.loads(zipped.read(f"{agent}/SUBMISSION_MANIFEST.json"))
                self.assertEqual(manifest["checkpoint_sha256"], expected)
                dependencies = zipped.read(f"{agent}/requirements.txt").decode()
                self.assertIn("numpy==", dependencies)
                self.assertIn("torch==", dependencies)

    def test_cnn_submission_runs_without_teacher_dataset_or_repository_packages(self):
        agent = "cnn_distilled_double_dqn_agent"
        checkpoint = PROJECT_ROOT / "agent_code" / agent / "final.pt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = build_submission(agent, checkpoint, root / "submission.zip")
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertIn(f"{agent}/final.pt", names)
                self.assertIn(f"{agent}/requirements.txt", names)
                self.assertFalse(any("teacher" in name.lower() for name in names))
                self.assertFalse(any("dataset" in name.lower() for name in names))
                zipped.extractall(root / "unpacked")

            isolated, environment = self._prepare_isolated_root(
                root, root / "unpacked", agent, "board-path-history-v2",
                "r5_conditional_loop")
            result = subprocess.run(
                [sys.executable, "main.py", "play", "--agents", agent,
                 "random_agent", "random_agent", "random_agent", "--scenario",
                 "classic", "--n-rounds", "1", "--no-gui"],
                cwd=isolated, env=environment, capture_output=True, text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
