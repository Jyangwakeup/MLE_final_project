"""Frozen-score convergence checks for Task 1 training."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PERFORMANCE_STOPPING_VERSION = "task1-frozen-score-v1"
DEFAULT_TASK1_PERFORMANCE_STOPPING = {
    "enabled": True,
    "version": PERFORMANCE_STOPPING_VERSION,
    "metric": "mean_score",
    "threshold": 48.0,
    "start_round": 200,
    "interval_rounds": 50,
    "evaluation_seeds": list(range(9000, 9020)),
    "rounds_per_seed": 1,
    "consecutive_passes": 3,
    "max_total_rounds": 1000,
}


def resolve_performance_stopping(
    value: Mapping[str, Any] | None, *, task: str,
) -> dict[str, Any] | None:
    """Validate the complete Task 1 convergence contract."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("config.training.performance_stopping must be an object")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("performance_stopping.enabled must be a boolean")
    if not enabled:
        return None
    if task != "coin_navigation":
        raise ValueError("performance_stopping is currently supported only for Task 1")
    required = set(DEFAULT_TASK1_PERFORMANCE_STOPPING)
    if set(value) != required:
        raise ValueError(
            "performance_stopping must contain exactly: "
            + ", ".join(sorted(required)))
    result = dict(value)
    if result["version"] != PERFORMANCE_STOPPING_VERSION:
        raise ValueError("unsupported performance_stopping version")
    if result["metric"] != "mean_score":
        raise ValueError("Task 1 performance_stopping requires metric='mean_score'")
    threshold = result["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("performance_stopping.threshold must be numeric")
    if not math.isfinite(threshold) or not 0 <= float(threshold) <= 50:
        raise ValueError("performance_stopping.threshold must be between 0 and 50")
    result["threshold"] = float(threshold)
    for name in (
        "start_round", "interval_rounds", "rounds_per_seed",
        "consecutive_passes", "max_total_rounds",
    ):
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"performance_stopping.{name} must be a positive integer")
    if result["start_round"] > result["max_total_rounds"]:
        raise ValueError("performance_stopping.start_round exceeds max_total_rounds")
    seeds = result["evaluation_seeds"]
    if (
        not isinstance(seeds, list) or len(seeds) != 20
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("performance_stopping.evaluation_seeds must contain 20 unique integers")
    if seeds != list(range(9000, 9020)):
        raise ValueError("Task 1 convergence seeds must be exactly 9000..9019")
    if result["rounds_per_seed"] != 1:
        raise ValueError("Task 1 convergence requires one round per seed")
    return result


def validate_history(
    history: Sequence[Mapping[str, Any]], spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate committed assessment history loaded from a snapshot."""
    result: list[dict[str, Any]] = []
    previous_round = -1
    consecutive = 0
    generations: set[str] = set()
    for raw in history:
        item = dict(raw)
        required = {
            "assessment_id", "checkpoint_generation", "checkpoint_generation_hash",
            "cumulative_round", "scores", "mean_score", "passed",
            "consecutive_passes",
        }
        if set(item) != required:
            raise ValueError("performance convergence history has an invalid shape")
        round_index = int(item["cumulative_round"])
        if round_index <= previous_round:
            raise ValueError("performance convergence rounds must increase")
        generation = str(item["checkpoint_generation"])
        if generation in generations:
            raise ValueError("one checkpoint generation cannot be counted twice")
        scores = [float(score) for score in item["scores"]]
        if len(scores) != len(spec["evaluation_seeds"]):
            raise ValueError("performance assessment has the wrong score count")
        if any(not 0 <= score <= 50 for score in scores):
            raise ValueError("Task 1 evaluation score must be between 0 and 50")
        mean_score = sum(scores) / len(scores)
        if not math.isclose(mean_score, float(item["mean_score"]), abs_tol=1e-12):
            raise ValueError("performance assessment mean does not match scores")
        passed = mean_score >= float(spec["threshold"])
        if bool(item["passed"]) != passed:
            raise ValueError("performance assessment pass flag is inconsistent")
        consecutive = consecutive + 1 if passed else 0
        if int(item["consecutive_passes"]) != consecutive:
            raise ValueError("performance assessment consecutive count is inconsistent")
        item.update({
            "cumulative_round": round_index, "scores": scores,
            "mean_score": mean_score, "passed": passed,
            "consecutive_passes": consecutive,
        })
        result.append(item)
        generations.add(generation)
        previous_round = round_index
    return result


class Task1PerformanceStopping:
    """Synchronously assess immutable round-boundary checkpoints."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        run_directory: Path,
        base_cumulative_rounds: int,
        assessor: Callable[[int, str, str], Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]] = (),
    ):
        self.spec = dict(spec)
        self.run_directory = Path(run_directory)
        self.base_cumulative_rounds = int(base_cumulative_rounds)
        self.assessor = assessor
        self.history = validate_history(history, self.spec)
        self.result: dict[str, Any] | None = None

    @property
    def consecutive_passes(self) -> int:
        return 0 if not self.history else int(self.history[-1]["consecutive_passes"])

    def _due(self, cumulative_round: int) -> bool:
        if cumulative_round < int(self.spec["start_round"]):
            return False
        if self.history:
            return cumulative_round >= (
                int(self.history[-1]["cumulative_round"])
                + int(self.spec["interval_rounds"])
            )
        return (
            cumulative_round == self.base_cumulative_rounds
            or (cumulative_round - int(self.spec["start_round"]))
            % int(self.spec["interval_rounds"]) == 0
        )

    def assess(self, cumulative_round: int, generation: str, generation_hash: str) -> dict[str, Any]:
        if any(item["checkpoint_generation"] == generation for item in self.history):
            raise ValueError("one checkpoint generation cannot be assessed twice")
        raw = dict(self.assessor(cumulative_round, generation, generation_hash))
        scores = [float(score) for score in raw["scores"]]
        mean_score = sum(scores) / len(scores)
        passed = mean_score >= float(self.spec["threshold"])
        consecutive = self.consecutive_passes + 1 if passed else 0
        assessment = {
            "assessment_id": f"assessment-{cumulative_round:08d}",
            "checkpoint_generation": generation,
            "checkpoint_generation_hash": generation_hash,
            "cumulative_round": int(cumulative_round),
            "scores": scores,
            "mean_score": mean_score,
            "passed": passed,
            "consecutive_passes": consecutive,
        }
        performance_root = self.run_directory / "performance"
        performance_root.mkdir(exist_ok=True)
        final = performance_root / assessment["assessment_id"]
        if final.exists():
            raise FileExistsError(f"Performance assessment already exists: {final}")
        temporary = performance_root / (
            "." + assessment["assessment_id"] + "-" + uuid.uuid4().hex + ".tmp")
        temporary.mkdir()
        try:
            evaluation_root = Path(raw["evaluation_root"])
            shutil.move(str(evaluation_root), temporary / "evaluation")
            (temporary / "assessment.json").write_text(
                json.dumps(assessment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.history.append(assessment)
        return assessment

    def __call__(self, local_completed_rounds: int) -> bool:
        cumulative = self.base_cumulative_rounds + int(local_completed_rounds)
        if self._due(cumulative):
            latest = json.loads(
                (self.run_directory / "resume" / "latest.json").read_text(encoding="utf-8"))
            generation = latest["generations"][0]
            self.assess(cumulative, generation, latest["generation_hash"])
            if self.consecutive_passes >= int(self.spec["consecutive_passes"]):
                self.result = {
                    "reason": "task1_score_converged",
                    "cumulative_round": cumulative,
                    "assessment": self.history[-1],
                }
                return True
        if cumulative >= int(self.spec["max_total_rounds"]):
            self.result = {
                "reason": "task1_score_not_converged",
                "cumulative_round": cumulative,
                "last_assessment": None if not self.history else self.history[-1],
            }
            return True
        return False


def load_committed_history(
    run_directory: Path, snapshot_history: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Merge snapshot history with later atomically committed assessments."""
    by_id = {str(item["assessment_id"]): dict(item) for item in snapshot_history}
    root = Path(run_directory) / "performance"
    if root.is_dir():
        for path in root.glob("assessment-*/assessment.json"):
            item = json.loads(path.read_text(encoding="utf-8"))
            identifier = str(item["assessment_id"])
            if identifier in by_id and by_id[identifier] != item:
                raise ValueError("committed performance assessment conflicts with snapshot")
            by_id[identifier] = item
    ordered = sorted(by_id.values(), key=lambda item: int(item["cumulative_round"]))
    return validate_history(ordered, spec)


def frozen_score_assessor(
    *, project_root: Path, config_path: Path, run_directory: Path,
    agent: str, checkpoint: Path, seeds: Sequence[int], rounds_per_seed: int,
) -> Callable[[int, str, str], Mapping[str, Any]]:
    """Build a subprocess assessor isolated from every training RNG."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    runs_root = project_root / "runs"

    def assess(cumulative_round: int, generation: str, generation_hash: str):
        run_id = (
            f"_performance_{run_directory.name}_{cumulative_round:08d}_"
            f"{uuid.uuid4().hex}"
        )
        evaluation_root = runs_root / run_id
        command = [
            sys.executable, str(project_root / "experiments" / "run.py"),
            "--config", str(config_path), "--mode", "evaluate",
            "--device", "cpu", "--task", "1", "--agent", agent,
            "--checkpoint", str(checkpoint), "--n-rounds", str(rounds_per_seed),
            "--run-id", run_id, "--seeds", *(str(seed) for seed in seeds),
            "--replay-policy", "none",
        ]
        try:
            completed = subprocess.run(
                command, cwd=project_root, text=True, capture_output=True)
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError("Frozen score assessment failed: " + message)
            scores: list[float] = []
            for seed in seeds:
                episode_path = evaluation_root / f"{run_id}_s{seed}" / "episodes.jsonl"
                lines = episode_path.read_text(encoding="utf-8").splitlines()
                if len(lines) != rounds_per_seed:
                    raise ValueError("Frozen score assessment has the wrong episode count")
                for line in lines:
                    episode = json.loads(line)
                    if not episode.get("exploration_disabled"):
                        raise ValueError("Performance assessment must disable exploration")
                    target = next(
                        item for item in episode["agents"] if item["name"] == agent)
                    if float(target["score"]) != float(target["coins"]):
                        raise ValueError("Task 1 performance requires score == coins")
                    scores.append(float(target["score"]))
            return {"scores": scores, "evaluation_root": str(evaluation_root)}
        except BaseException:
            shutil.rmtree(evaluation_root, ignore_errors=True)
            raise

    return assess
