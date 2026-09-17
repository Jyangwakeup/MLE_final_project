"""Frozen-quality, success-only stopping for Task 2 extensions.

This deliberately has no plateau/reward criterion: a run can stop early only
after two independently frozen evaluations satisfy every registered Task 2
gate.  It is used after the fair 100k ablation screen, never by that screen.
"""

from __future__ import annotations

import json
import math
import csv
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.select_task2_snapshots import _sha256, _summary


TASK2_SUCCESS_STOPPING_VERSION = "task2-frozen-quality-v1"
DEFAULT_TASK2_SUCCESS_STOPPING = {
    "enabled": True,
    "version": TASK2_SUCCESS_STOPPING_VERSION,
    "start_stage_action_steps": 100000,
    "interval_stage_action_steps": 25000,
    "consecutive_passes": 2,
    "evaluation_seeds": list(range(10000, 10020)),
    "rounds_per_seed": 1,
}


def resolve_task2_success_stopping(
    value: Mapping[str, Any] | None, *, task: str,
) -> dict[str, Any] | None:
    """Validate the fixed Task 2 success-stop protocol."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("config.training.task2_success_stopping must be an object")
    if value.get("enabled", True) is False:
        return None
    if value.get("enabled", True) is not True:
        raise ValueError("task2_success_stopping.enabled must be a boolean")
    if task != "crate_navigation":
        raise ValueError("task2_success_stopping is supported only for Task 2")
    if set(value) != set(DEFAULT_TASK2_SUCCESS_STOPPING):
        raise ValueError("task2_success_stopping must contain exactly: " + ", ".join(sorted(DEFAULT_TASK2_SUCCESS_STOPPING)))
    result = dict(value)
    if result["version"] != TASK2_SUCCESS_STOPPING_VERSION:
        raise ValueError("unsupported task2_success_stopping version")
    for name, expected in (("start_stage_action_steps", 100000), ("interval_stage_action_steps", 25000), ("consecutive_passes", 2), ("rounds_per_seed", 1)):
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, int) or item != expected:
            raise ValueError(f"task2_success_stopping.{name} must be {expected}")
    if result["evaluation_seeds"] != list(range(10000, 10020)):
        raise ValueError("Task 2 success stopping seeds must be exactly 10000..10019")
    return result


def task2_quality_assessment(
    *, checkpoint: Path, agent: str, gate_path: Path,
    parent_task1: Path, parent_task2: Path, task1_evaluation: Path,
    task2_evaluation: Path,
) -> dict[str, Any]:
    """Apply the exact 19-gate selector contract to one frozen checkpoint."""
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))["gates"]
    base_t1, base_t2 = _summary(parent_task1, agent), _summary(parent_task2, agent)
    task1, task2 = _summary(task1_evaluation, agent), _summary(task2_evaluation, agent)
    retention = task1["mean_score"] / base_t1["mean_score"] if base_t1["mean_score"] else 0.0
    crate_gain = task2["mean_crates"] - base_t2["mean_crates"]
    checks = {
        "episode_count": task2["episode_count"] == 20,
        "mean_coins": task2["mean_coins"] >= gate["task2_mean_coins_min"],
        "zero_coin_round_rate": task2["zero_coin_round_rate"] <= gate["task2_zero_coin_round_rate_max"],
        "mean_crates": task2["mean_crates"] >= gate["task2_mean_crates_min"],
        "all_coins_rate": task2["all_coins_rate"] >= gate["task2_all_coins_rate_min"],
        "coins_per_100_steps": task2["coins_per_100_steps"] >= gate["task2_coins_per_100_steps_min"],
        "max_steps_rate": task2["max_steps_rate"] <= gate["task2_max_steps_rate_max"],
        "long_wait_loop_rate": task2["long_wait_loop_rate"] <= gate["task2_long_wait_loop_rate_max"],
        "long_ping_pong_loop_rate": task2["long_ping_pong_loop_rate"] <= gate["task2_long_ping_pong_loop_rate_max"],
        "suicide_rate": task2["suicide_rate"] <= gate["task2_suicide_rate_max"],
        "zero_bomb_round_rate": task2["zero_bomb_round_rate"] <= gate["zero_bomb_round_rate_max"],
        "zero_utility_bomb_rate": task2["zero_utility_bomb_rate"] <= gate["zero_utility_bomb_rate_max"],
        "crates_per_bomb": task2["crates_per_bomb"] >= gate["crates_per_bomb_min"],
        "survived_bomb_rate": task2["survived_bomb_rate"] >= gate["survived_bomb_rate_min"],
        "invalid_action_rate": task2["invalid_action_rate"] <= gate["invalid_action_rate_max"],
        "act_p95": task2["act_p95_seconds"] < gate["act_p95_seconds_max"],
        "act_max": task2["act_max_seconds"] < gate["act_max_seconds_max"],
        "task1_retention": retention >= gate["task1_retention_min"],
        "crate_gain": crate_gain >= 0.5,
    }
    return {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": _sha256(Path(checkpoint)), "task1": task1, "task2": task2, "task1_retention": retention, "crate_gain": crate_gain, "gates": checks, "passed": all(checks.values())}


def validate_history(history: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = -1
    seen: set[str] = set()
    streak = 0
    for raw in history:
        item = dict(raw)
        required = {"assessment_id", "stage_action_steps", "checkpoint", "checkpoint_sha256", "passed", "consecutive_passes", "gates"}
        if set(item) != required:
            raise ValueError("Task 2 success history has an invalid shape")
        steps = int(item["stage_action_steps"])
        if steps < int(spec["start_stage_action_steps"]) or steps <= previous:
            raise ValueError("Task 2 success assessment steps must increase from 100k")
        checkpoint = str(item["checkpoint"])
        if checkpoint in seen or len(item["gates"]) != 19:
            raise ValueError("Task 2 success assessment checkpoint/gates are invalid")
        passed = all(bool(value) for value in item["gates"].values())
        if bool(item["passed"]) != passed:
            raise ValueError("Task 2 success assessment pass flag is inconsistent")
        streak = streak + 1 if passed else 0
        if int(item["consecutive_passes"]) != streak:
            raise ValueError("Task 2 success assessment streak is inconsistent")
        item.update(stage_action_steps=steps, passed=passed, consecutive_passes=streak)
        result.append(item); seen.add(checkpoint); previous = steps
    return result


class Task2SuccessStopping:
    """Assess each immutable 25k snapshot and stop only after two passes."""
    def __init__(self, spec: Mapping[str, Any], *, run_directory: Path, training_path: Path,
                 assessor: Callable[[Path], Mapping[str, Any]], history: Sequence[Mapping[str, Any]] = ()):
        self.spec, self.run_directory, self.training_path, self.assessor = dict(spec), Path(run_directory), Path(training_path), assessor
        self.history = validate_history(history, self.spec)
        self.result: dict[str, Any] | None = None

    def _snapshot(self, stage_steps: int) -> Path | None:
        snapshots = self.run_directory / "checkpoints" / "snapshots"
        choices = []
        for path in snapshots.glob("step_*.pkl"):
            try: choices.append((int(path.stem.split("_")[1]), path))
            except (IndexError, ValueError): continue
        choices = [item for item in choices if item[0] <= stage_steps]
        return max(choices, default=(None, None))[1]

    def __call__(self, _completed_rounds: int) -> bool:
        with self.training_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if not rows:
            return False
        stage_steps = int(rows[-1]["stage_action_steps"])
        if stage_steps < int(self.spec["start_stage_action_steps"]): return False
        last = self.history[-1]["stage_action_steps"] if self.history else None
        if last is not None and stage_steps < int(last) + int(self.spec["interval_stage_action_steps"]): return False
        checkpoint = self._snapshot(stage_steps)
        if checkpoint is None or any(item["checkpoint"] == str(checkpoint.resolve()) for item in self.history): return False
        raw = dict(self.assessor(checkpoint))
        gates = dict(raw["gates"]); passed = all(gates.values())
        streak = (self.history[-1]["consecutive_passes"] if self.history else 0) + 1 if passed else 0
        item = {"assessment_id": f"assessment-{stage_steps:09d}", "stage_action_steps": int(stage_steps), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint), "passed": passed, "consecutive_passes": streak, "gates": gates}
        root = self.run_directory / "task2_success"
        root.mkdir(exist_ok=True)
        final = root / item["assessment_id"]
        if final.exists():
            raise FileExistsError(f"Task 2 success assessment already exists: {final}")
        temporary = root / ("." + item["assessment_id"] + "-" + uuid.uuid4().hex + ".tmp")
        temporary.mkdir()
        try:
            (temporary / "assessment.json").write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (temporary / "quality.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.history.append(item)
        if streak >= int(self.spec["consecutive_passes"]):
            self.result = {"reason": "task2_quality_converged", "stage_action_steps": int(stage_steps), "assessment": item}
            return True
        return False


def load_committed_history(run_directory: Path, snapshot_history: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Merge snapshot state with assessments committed after the last snapshot."""
    entries = {str(item["assessment_id"]): dict(item) for item in snapshot_history}
    for path in (Path(run_directory) / "task2_success").glob("assessment-*/assessment.json"):
        item = json.loads(path.read_text(encoding="utf-8")); identifier = str(item["assessment_id"])
        if identifier in entries and entries[identifier] != item:
            raise ValueError("committed Task 2 assessment conflicts with snapshot")
        entries[identifier] = item
    return validate_history(sorted(entries.values(), key=lambda item: int(item["stage_action_steps"])), spec)


def frozen_quality_assessor(*, project_root: Path, run_directory: Path, agent: str,
                            task1_config: Path, task2_config: Path, gate_path: Path,
                            parent_task1: Path, parent_task2: Path,
                            seeds: Sequence[int], rounds_per_seed: int) -> Callable[[Path], Mapping[str, Any]]:
    """Create a paired, exploration-disabled Task 1/2 frozen evaluator."""
    project_root = Path(project_root).resolve()
    def assess(checkpoint: Path) -> Mapping[str, Any]:
        token = uuid.uuid4().hex
        roots: list[Path] = []
        try:
            for task, config in (("1", task1_config), ("2", task2_config)):
                run_id = f"_task2_quality_{run_directory.name}_{checkpoint.stem}_{task}_{token}"
                root = project_root / "runs" / run_id
                roots.append(root)
                command = [sys.executable, str(project_root / "experiments" / "run.py"), "--config", str(config), "--mode", "evaluate", "--device", "cpu", "--task", task, "--agent", agent, "--checkpoint", str(checkpoint), "--n-rounds", str(rounds_per_seed), "--run-id", run_id, "--seeds", *(str(seed) for seed in seeds), "--replay-policy", "none"]
                completed = subprocess.run(command, cwd=project_root, text=True, capture_output=True)
                if completed.returncode:
                    raise RuntimeError("Task 2 frozen quality assessment failed: " + (completed.stderr.strip() or completed.stdout.strip()))
            return task2_quality_assessment(checkpoint=checkpoint, agent=agent, gate_path=gate_path, parent_task1=parent_task1, parent_task2=parent_task2, task1_evaluation=roots[0], task2_evaluation=roots[1])
        except BaseException:
            for root in roots: shutil.rmtree(root, ignore_errors=True)
            raise
    return assess
