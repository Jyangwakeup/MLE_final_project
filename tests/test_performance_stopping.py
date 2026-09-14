import json
from pathlib import Path
import tempfile
import unittest

from experiments.performance_stopping import (
    DEFAULT_TASK1_PERFORMANCE_STOPPING,
    Task1PerformanceStopping,
    load_committed_history,
    resolve_performance_stopping,
)


class Task1PerformanceStoppingTestCase(unittest.TestCase):
    def setUp(self):
        self.spec = resolve_performance_stopping(
            DEFAULT_TASK1_PERFORMANCE_STOPPING, task="coin_navigation")

    def _latest(self, run: Path, round_index: int):
        root = run / "resume"
        root.mkdir(exist_ok=True)
        (root / "latest.json").write_text(json.dumps({
            "generations": [f"generation-{round_index:08d}"],
            "generation_hash": f"hash-{round_index}",
        }))

    def _assessor(self, root: Path, scores):
        calls = []

        def assess(round_index, generation, generation_hash):
            calls.append((round_index, generation, generation_hash))
            evaluation = root / f"evaluation-{round_index}"
            evaluation.mkdir()
            return {"scores": list(scores.pop(0)), "evaluation_root": evaluation}

        return assess, calls

    def test_fresh_run_checks_200_250_300_and_stops_after_three_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; run.mkdir()
            assessor, calls = self._assessor(
                Path(directory), [[48.0] * 20, [49.0] * 20, [50.0] * 20])
            stopper = Task1PerformanceStopping(
                self.spec, run_directory=run, base_cumulative_rounds=0,
                assessor=assessor)
            for round_index in range(1, 301):
                self._latest(run, round_index)
                stopped = stopper(round_index)
                if round_index < 300:
                    self.assertFalse(stopped)
            self.assertTrue(stopped)
            self.assertEqual([item[0] for item in calls], [200, 250, 300])
            self.assertEqual(stopper.result["reason"], "task1_score_converged")
            self.assertEqual(stopper.consecutive_passes, 3)

    def test_failed_assessment_resets_consecutive_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; run.mkdir()
            assessor, _ = self._assessor(Path(directory), [
                [48.0] * 20, [47.0] * 20, [48.0] * 20,
            ])
            stopper = Task1PerformanceStopping(
                self.spec, run_directory=run, base_cumulative_rounds=0,
                assessor=assessor)
            for round_index in (200, 250, 300):
                self._latest(run, round_index)
                self.assertFalse(stopper(round_index))
            self.assertEqual(
                [item["consecutive_passes"] for item in stopper.history], [1, 0, 1])

    def test_migration_assesses_parent_then_waits_fifty_new_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; run.mkdir()
            assessor, calls = self._assessor(
                Path(directory), [[50.0] * 20, [50.0] * 20])
            stopper = Task1PerformanceStopping(
                self.spec, run_directory=run, base_cumulative_rounds=750,
                assessor=assessor)
            stopper.assess(750, "generation-00000750", "parent-hash")
            for local_round in range(1, 50):
                self._latest(run, 750 + local_round)
                self.assertFalse(stopper(local_round))
            self._latest(run, 800)
            self.assertFalse(stopper(50))
            self.assertEqual([item[0] for item in calls], [750, 800])

    def test_maximum_round_is_not_converged_without_three_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; run.mkdir()
            assessor, _ = self._assessor(Path(directory), [[47.0] * 20])
            stopper = Task1PerformanceStopping(
                self.spec, run_directory=run, base_cumulative_rounds=950,
                assessor=assessor)
            self._latest(run, 1000)
            self.assertTrue(stopper(50))
            self.assertEqual(stopper.result["reason"], "task1_score_not_converged")

    def test_committed_assessment_after_snapshot_is_recovered_once(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; run.mkdir()
            assessor, _ = self._assessor(Path(directory), [[49.0] * 20])
            stopper = Task1PerformanceStopping(
                self.spec, run_directory=run, base_cumulative_rounds=0,
                assessor=assessor)
            stopper.assess(200, "generation-00000200", "hash-200")
            recovered = load_committed_history(run, [], self.spec)
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["cumulative_round"], 200)

    def test_contract_is_task1_only_and_uses_reserved_monitor_seeds(self):
        with self.assertRaisesRegex(ValueError, "only for Task 1"):
            resolve_performance_stopping(
                DEFAULT_TASK1_PERFORMANCE_STOPPING, task="crate_navigation")
        changed = dict(DEFAULT_TASK1_PERFORMANCE_STOPPING)
        changed["evaluation_seeds"] = list(range(10000, 10020))
        with self.assertRaisesRegex(ValueError, "9000..9019"):
            resolve_performance_stopping(changed, task="coin_navigation")


if __name__ == "__main__":
    unittest.main()
