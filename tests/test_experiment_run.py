import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "experiments" / "run.py"
BASE_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "base.json"
RUNS_ROOT = PROJECT_ROOT / "runs"


class ExperimentRunCliTest(unittest.TestCase):
    def setUp(self):
        self.run_ids = []
        self.runs_root_existed = RUNS_ROOT.exists()

    def tearDown(self):
        for run_id in self.run_ids:
            run_directory = RUNS_ROOT / run_id
            if run_directory.exists():
                shutil.rmtree(run_directory)
        if not self.runs_root_existed and RUNS_ROOT.exists() and not any(RUNS_ROOT.iterdir()):
            RUNS_ROOT.rmdir()

    def _run_id(self):
        run_id = f"test-baseline-{uuid.uuid4().hex}"
        self.run_ids.append(run_id)
        return run_id

    def _baseline_config(self, directory):
        config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        config["baseline"] = {
            "agents": ["random_agent"],
            "n_rounds": 1,
            "scenario": "classic",
        }
        path = Path(directory) / "baseline.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return path

    def _invoke(self, config, run_id, mode="baseline"):
        return subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--config",
                str(config),
                "--mode",
                mode,
                "--seed",
                "10001",
                "--run-id",
                run_id,
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=60,
        )

    def test_baseline_reads_config_preserves_it_and_writes_official_stats(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = self._baseline_config(temporary_directory)
            original_config = config.read_bytes()
            run_id = self._run_id()

            result = self._invoke(config, run_id)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(config.read_bytes(), original_config)

            output = RUNS_ROOT / run_id
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["mode"], "baseline")
            self.assertEqual(metadata["seed"], 10001)
            self.assertEqual(metadata["expanded_config"]["baseline"]["n_rounds"], 1)

            official_stats = json.loads((output / "official_stats.json").read_text(encoding="utf-8"))
            self.assertIn("by_agent", official_stats)
            self.assertIn("by_round", official_stats)

    def test_existing_run_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = self._baseline_config(temporary_directory)
            run_id = self._run_id()
            output = RUNS_ROOT / run_id
            output.mkdir(parents=True)
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            result = self._invoke(config, run_id)

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertFalse((output / "metadata.json").exists())

    def test_unsupported_mode_records_failed_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = self._baseline_config(temporary_directory)
            run_id = self._run_id()

            result = self._invoke(config, run_id, mode="train")

            self.assertNotEqual(result.returncode, 0)
            metadata = json.loads(
                (RUNS_ROOT / run_id / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["mode"], "train")
            self.assertEqual(metadata["error"]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
