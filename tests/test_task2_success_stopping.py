"""Unit tests for Task 2's success-only frozen-quality stop."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.task2_success_stopping import (
    DEFAULT_TASK2_SUCCESS_STOPPING, Task2SuccessStopping,
    resolve_task2_success_stopping,
)


def _write_steps(root: Path, steps: int) -> None:
    with (root / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage_action_steps"])
        writer.writeheader()
        writer.writerow({"stage_action_steps": steps})


def _gates(passed: bool) -> dict[str, bool]:
    return {f"gate_{index}": passed for index in range(19)}


class Task2SuccessStoppingTests(unittest.TestCase):
    def test_task2_only_fixed_contract(self):
        self.assertIsNotNone(resolve_task2_success_stopping(
            DEFAULT_TASK2_SUCCESS_STOPPING, task="crate_navigation"))
        with self.assertRaisesRegex(ValueError, "only for Task 2"):
            resolve_task2_success_stopping(
                DEFAULT_TASK2_SUCCESS_STOPPING, task="coin_navigation")
        with self.assertRaisesRegex(ValueError, "25000"):
            resolve_task2_success_stopping(
                dict(DEFAULT_TASK2_SUCCESS_STOPPING,
                     interval_stage_action_steps=50000),
                task="crate_navigation")

    def test_two_passes_stop_and_failure_resets(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshots = root / "checkpoints" / "snapshots"
            snapshots.mkdir(parents=True)
            outcomes = iter((True, False, True, True))

            def assess(_checkpoint: Path):
                return {"gates": _gates(next(outcomes))}

            stop = Task2SuccessStopping(
                DEFAULT_TASK2_SUCCESS_STOPPING, run_directory=root,
                training_path=root / "training.csv", assessor=assess)
            for steps, expected in (
                (100000, False), (125000, False),
                (150000, False), (175000, True),
            ):
                (snapshots / f"step_{steps:07d}.pkl").write_bytes(b"checkpoint")
                _write_steps(root, steps)
                self.assertEqual(stop(1), expected)
            self.assertEqual(
                [item["consecutive_passes"] for item in stop.history],
                [1, 0, 1, 2])
            self.assertEqual(stop.result["reason"], "task2_quality_converged")

    def test_resume_history_does_not_repeat_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshots = root / "checkpoints" / "snapshots"
            snapshots.mkdir(parents=True)
            checkpoint = snapshots / "step_0100000.pkl"
            checkpoint.write_bytes(b"checkpoint")
            _write_steps(root, 100000)
            first = Task2SuccessStopping(
                DEFAULT_TASK2_SUCCESS_STOPPING, run_directory=root,
                training_path=root / "training.csv",
                assessor=lambda _: {"gates": _gates(True)})
            self.assertFalse(first(1))

            def duplicate(_checkpoint: Path):
                self.fail("same checkpoint was assessed twice")

            repeated = Task2SuccessStopping(
                DEFAULT_TASK2_SUCCESS_STOPPING, run_directory=root,
                training_path=root / "training.csv", assessor=duplicate,
                history=first.history)
            self.assertFalse(repeated(2))


if __name__ == "__main__":
    unittest.main()
