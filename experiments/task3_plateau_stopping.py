"""Run the preregistered Task 3 frozen-capability plateau experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.compare_evaluations import compare_evaluations
from experiments.resume import CHECKPOINT_SCHEMA_VERSION, _load_generation
from experiments.task3_retention_prefix import (
    _evaluation_command,
    _evaluation_directories,
    evaluate_gates,
    summarize_evaluation,
)


PASS = 0
INFRASTRUCTURE_ERROR = 1
GATE_FAILURE = 2
PLATEAU_VERSION = "task3-plateau-v1"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=project_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_is_ancestor(project_root: Path, ancestor: str, descendant: str) -> bool:
    """Return whether descendant preserves the preregistered source history."""
    return subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=project_root, check=False, capture_output=True, text=True,
    ).returncode == 0


def _resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(command: Sequence[str], project_root: Path) -> None:
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    completed = subprocess.run(command, cwd=project_root, env=environment)
    if completed.returncode:
        raise RuntimeError(
            f"Subprocess exited with {completed.returncode}: {' '.join(command)}")


def resolve_plateau_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "version", "interval_rounds", "max_total_rounds",
        "minimum_score_improvement", "qualified_patience",
        "unqualified_patience",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            "plateau_stopping must contain exactly: " + ", ".join(sorted(required)))
    result = dict(value)
    if result["version"] != PLATEAU_VERSION:
        raise ValueError("unsupported Task 3 plateau version")
    for name in (
        "interval_rounds", "max_total_rounds", "qualified_patience",
        "unqualified_patience",
    ):
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"plateau_stopping.{name} must be a positive integer")
    improvement = result["minimum_score_improvement"]
    if (
        isinstance(improvement, bool)
        or not isinstance(improvement, (int, float))
        or not math.isfinite(float(improvement))
        or float(improvement) <= 0
    ):
        raise ValueError("minimum_score_improvement must be positive and finite")
    result["minimum_score_improvement"] = float(improvement)
    if result["max_total_rounds"] % result["interval_rounds"]:
        raise ValueError("max_total_rounds must be divisible by interval_rounds")
    return result


def _candidate_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Sort qualified checkpoints by the preregistered official-metric order."""
    return (
        -float(item["task3_score"]),
        -float(item["task3_kills"]),
        -float(item["task3_first_place_rate"]),
        float(item["task3_suicide_rate"]),
        -float(item["task3_bomb_survival_rate"]),
        -float(item["minimum_retention"]),
        float(item["act_p95_seconds"]),
        str(item["checkpoint_run"]),
    )


class PlateauTracker:
    """Pure decision engine for Task 3 checkpoint selection and stopping."""

    def __init__(
        self, spec: Mapping[str, Any], *,
        history: Sequence[Mapping[str, Any]] = (),
    ):
        self.spec = resolve_plateau_spec(spec)
        self.history: list[dict[str, Any]] = []
        self._best: dict[str, Any] | None = None
        self._eligible_anchor: float | None = None
        self._ineligible_anchor: float | None = None
        self._best_failed_count: int | None = None
        self._qualified_patience = 0
        self._unqualified_patience = 0
        self.state: dict[str, Any] = self._state(False, None)
        for item in history:
            if self.state["stop"]:
                raise ValueError("plateau history contains assessments after stopping")
            self.record(item)

    def _state(self, stop: bool, reason: str | None) -> dict[str, Any]:
        return {
            "stop": stop,
            "reason": reason,
            "plateau_converged": reason == "qualified_plateau",
            "qualified_patience": self._qualified_patience,
            "unqualified_patience": self._unqualified_patience,
            "plateau_anchor_score": (
                self._eligible_anchor
                if self._eligible_anchor is not None else self._ineligible_anchor),
            "selected_checkpoint_run": (
                None if self._best is None else self._best["checkpoint_run"]),
            "selected_checkpoint_sha256": (
                None if self._best is None else self._best["checkpoint_sha256"]),
            "selected_round": (
                None if self._best is None else self._best["cumulative_round"]),
        }

    def record(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(raw)
        required = {
            "cumulative_round", "eligible", "failed_gate_count",
            "checkpoint_run", "checkpoint_sha256", "task3_score",
            "task3_kills", "task3_first_place_rate", "task3_suicide_rate",
            "task3_bomb_survival_rate", "minimum_retention",
            "act_p95_seconds",
        }
        if not required.issubset(item):
            raise ValueError("plateau assessment is missing required fields")
        round_index = int(item["cumulative_round"])
        if self.history and round_index <= int(self.history[-1]["cumulative_round"]):
            raise ValueError("plateau assessment rounds must strictly increase")
        interval = int(self.spec["interval_rounds"])
        if round_index < interval or round_index % interval:
            raise ValueError("plateau assessment is not on a scheduled boundary")
        if round_index > int(self.spec["max_total_rounds"]):
            raise ValueError("plateau assessment exceeds the round cap")
        for name in (
            "task3_score", "task3_kills", "task3_first_place_rate",
            "task3_suicide_rate", "task3_bomb_survival_rate",
            "minimum_retention", "act_p95_seconds",
        ):
            value = float(item[name])
            if not math.isfinite(value):
                raise ValueError(f"plateau assessment {name} must be finite")
            item[name] = value
        item["cumulative_round"] = round_index
        item["failed_gate_count"] = int(item["failed_gate_count"])
        item["eligible"] = bool(item["eligible"])

        threshold = float(self.spec["minimum_score_improvement"])
        if item["eligible"]:
            if self._best is None or _candidate_key(item) < _candidate_key(self._best):
                self._best = item
            score = float(item["task3_score"])
            if self._eligible_anchor is None:
                self._eligible_anchor = score
                self._qualified_patience = 0
            elif score >= self._eligible_anchor + threshold:
                self._eligible_anchor = score
                self._qualified_patience = 0
            else:
                self._qualified_patience += 1
        elif self._eligible_anchor is not None:
            self._qualified_patience += 1
        else:
            score = float(item["task3_score"])
            failures = int(item["failed_gate_count"])
            progressed = (
                self._ineligible_anchor is None
                or score >= self._ineligible_anchor + threshold
                or self._best_failed_count is None
                or failures < self._best_failed_count
            )
            if progressed:
                self._ineligible_anchor = score
                self._best_failed_count = failures
                self._unqualified_patience = 0
            else:
                self._unqualified_patience += 1

        self.history.append(item)
        reason = None
        if (
            self._best is not None
            and self._qualified_patience >= int(self.spec["qualified_patience"])
        ):
            reason = "qualified_plateau"
        elif (
            self._best is None
            and self._unqualified_patience >= int(self.spec["unqualified_patience"])
        ):
            reason = "unqualified_plateau"
        elif round_index >= int(self.spec["max_total_rounds"]):
            reason = (
                "budget_truncated_qualified"
                if self._best is not None else "round_cap_unqualified")
        self.state = self._state(reason is not None, reason)
        return dict(self.state)


def _minimum_retention(
    parent: Mapping[str, Mapping[str, float]],
    child: Mapping[str, Mapping[str, float]],
) -> float:
    values = []
    for task, metric in (
        ("task1", "mean_score"), ("task2", "mean_coins"),
        ("task2", "mean_crates"), ("task3", "mean_coins"),
        ("task3", "mean_crates"),
    ):
        denominator = float(parent[task][metric])
        values.append(
            float(child[task][metric]) / denominator if denominator else 1.0)
    return min(values)


def build_assessment(
    *, cumulative_round: int, checkpoint_run: str, checkpoint: Path,
    parent: Mapping[str, Mapping[str, float]],
    child: Mapping[str, Mapping[str, float]], gates: Mapping[str, float],
    gate_checks: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if gate_checks is None:
        eligible, resolved_checks = evaluate_gates(parent, child, gates)
    else:
        resolved_checks = dict(gate_checks)
        eligible = all(bool(item["passed"]) for item in resolved_checks.values())
    task3 = child["task3"]
    return {
        "cumulative_round": int(cumulative_round),
        "eligible": eligible,
        "failed_gate_count": sum(
            not bool(item["passed"]) for item in resolved_checks.values()),
        "checkpoint_run": checkpoint_run,
        "checkpoint_sha256": _sha256(checkpoint),
        "task3_score": float(task3["mean_score"]),
        "task3_kills": float(task3["mean_kills"]),
        "task3_first_place_rate": float(task3["first_place_rate"]),
        "task3_suicide_rate": float(task3["suicide_rate"]),
        "task3_bomb_survival_rate": float(task3["bomb_survival_rate"]),
        "minimum_retention": _minimum_retention(parent, child),
        "act_p95_seconds": max(
            float(child[task]["act_p95_seconds"])
            for task in ("task1", "task2", "task3")),
        "gate_checks": resolved_checks,
        "summaries": {"parent": parent, "child": child},
    }


def _unique_target(path: Path) -> tuple[Path, str]:
    if not path.exists():
        return path, ""
    attempt = 1
    while path.with_name(path.name + f"_retry{attempt}").exists():
        attempt += 1
    suffix = f"_retry{attempt}"
    return path.with_name(path.name + suffix), suffix


def _run_evaluation_pair(
    *, project_root: Path, manifest: Mapping[str, Any], training_seed: int,
    checkpoint: Path, role: str, phase: str, evaluation_seeds: Sequence[int],
    config_path: str, cpu_offset: int,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    short_commit = _git(project_root, "rev-parse", "HEAD")[:7]
    prefixes: dict[str, str] = {}
    processes = []
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    cpu_pool = [int(item) for item in manifest["cpus"]]
    evaluation_manifest = {
        "evaluation_seeds": list(evaluation_seeds),
        "evaluation_config": config_path,
        "agent": manifest["agent"],
    }
    for index, task in enumerate((1, 2, 3)):
        base = project_root / "runs" / (
            f"task3_plateau_{phase}_s{training_seed}_{role}_t{task}_"
            f"{evaluation_seeds[0]}_{evaluation_seeds[-1]}_{short_commit}")
        target, suffix = _unique_target(base)
        prefix = target.name
        prefixes[str(task)] = prefix
        command = _evaluation_command(
            project_root, evaluation_manifest, role=role, task=task,
            checkpoint=checkpoint, prefix=prefix,
            cpu=cpu_pool[(cpu_offset + index) % len(cpu_pool)],
        )
        processes.append((task, command, subprocess.Popen(
            command, cwd=project_root, env=environment)))
    failures = []
    for task, command, process in processes:
        code = process.wait()
        if code:
            failures.append({"task": task, "exit_code": code, "command": command})
    if failures:
        raise RuntimeError(f"Frozen evaluation subprocesses failed: {failures}")
    summaries = {
        f"task{task}": summarize_evaluation(
            project_root / "runs" / prefixes[str(task)], manifest["agent"],
            tuple(int(seed) for seed in evaluation_seeds))
        for task in (1, 2, 3)
    }
    return summaries, prefixes


def _compare_pair(
    *, project_root: Path, evidence_root: Path, phase: str,
    evaluation_seeds: Sequence[int], parent_prefixes: Mapping[str, str],
    child_prefixes: Mapping[str, str], bootstrap_samples: int,
) -> dict[str, Any]:
    result = {}
    seed_tuple = tuple(int(seed) for seed in evaluation_seeds)
    for task in (1, 2, 3):
        name = f"task{task}"
        result[name] = compare_evaluations(
            _evaluation_directories(
                project_root / "runs", child_prefixes[str(task)], seed_tuple),
            _evaluation_directories(
                project_root / "runs", parent_prefixes[str(task)], seed_tuple),
            evidence_root / "comparisons" / phase / name,
            bootstrap_samples=bootstrap_samples,
        )
    return result


def _validate_parent(parent: Path, expected_sha256: str) -> None:
    checkpoint = parent / "checkpoints" / "final.pt"
    if _sha256(checkpoint) != expected_sha256:
        raise ValueError(f"Task 2 parent checkpoint hash mismatch: {parent}")
    latest = _json(parent / "resume" / "latest.json")
    generations = latest.get("generations", [])
    if len(generations) != 2:
        raise ValueError("Task 2 parent does not retain two generations")
    for index, generation in enumerate(generations):
        _load_generation(
            parent, generation,
            latest.get("generation_hash") if index == 0 else None,
            expected_schema="training-resume-v7",
        )


def validate_plateau_training_run(
    run: Path, *, local_rounds: int, cumulative_round: int,
) -> dict[str, Any]:
    """Validate one immutable segment and its Task 3-local generations."""
    metadata = _json(run / "metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError("Plateau training segment is not completed")
    termination = metadata.get("termination", {})
    if int(termination.get("local_completed_rounds", -1)) != local_rounds:
        raise ValueError("Plateau segment has an unexpected local round count")
    lineage = metadata.get("lineage", {})
    parent_value = lineage.get("parent_run")
    if not parent_value:
        raise ValueError("Plateau segment has no parent run")
    parent = Path(parent_value)
    if not parent.is_absolute():
        parent = (run.parent.parent / parent).resolve()
    parent_metadata = _json(parent / "metadata.json")
    parent_cumulative = int(parent_metadata.get(
        "termination", {}).get("cumulative_completed_rounds", -1))
    segment_cumulative = int(termination.get("cumulative_completed_rounds", -1))
    if segment_cumulative != parent_cumulative + local_rounds:
        raise ValueError("Plateau segment breaks parent cumulative round continuity")
    latest = _json(run / "resume" / "latest.json")
    generations = latest.get("generations", [])
    expected = [
        f"generation-{cumulative_round:08d}",
        f"generation-{cumulative_round - 1:08d}",
    ]
    if generations != expected:
        raise ValueError("Plateau segment does not retain expected generations")
    loaded = []
    for index, generation in enumerate(generations):
        snapshot = _load_generation(
            run, generation,
            latest.get("generation_hash") if index == 0 else None,
            expected_schema=CHECKPOINT_SCHEMA_VERSION,
        )
        loaded.append({
            "generation": generation,
            "generation_hash": snapshot.generation_hash,
            "round": cumulative_round - index,
        })
    with (run / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != local_rounds:
        raise ValueError("Plateau segment training.csv has an unexpected row count")
    loss = float(rows[-1]["loss"])
    updates = int(rows[-1]["updates"])
    if not math.isfinite(loss) or updates <= 0:
        raise ValueError("Plateau segment has invalid loss or update count")
    return {"generations": loaded, "final_loss": loss, "final_updates": updates}


def _train_segment(
    *, project_root: Path, manifest: Mapping[str, Any], seed: int,
    cumulative_round: int, parent_run: Path, previous_run: Path | None,
) -> Path:
    short_commit = _git(project_root, "rev-parse", "HEAD")[:7]
    base = project_root / "runs" / (
        f"task3_plateau_ddqn_cv2_r7_s{seed}_c{cumulative_round:04d}_{short_commit}")
    run, suffix = _unique_target(base)
    command = [
        "taskset", "-c", str(manifest["training_cpus"][str(seed)]),
        sys.executable, "experiments/run.py", "--config",
        str(_resolve(project_root, manifest["configs"]["training"])),
        "--mode", "train", "--device", "cpu", "--task", "3",
        "--agent", manifest["agent"], "--seed", str(seed),
        "--n-rounds", str(manifest["plateau_stopping"]["interval_rounds"]),
        "--replay-policy", "sampled", "--replay-interval", "50",
        "--run-id", run.name,
    ]
    if previous_run is None:
        command.extend(("--transfer-task3-safety-from", str(parent_run)))
    else:
        command.extend(("--resume-from", str(previous_run)))
    _run_checked(command, project_root)
    validate_plateau_training_run(
        run,
        local_rounds=int(manifest["plateau_stopping"]["interval_rounds"]),
        cumulative_round=cumulative_round,
    )
    return run


def _seed_evidence_path(project_root: Path, seed: int, short_commit: str) -> Path:
    return project_root / "runs" / (
        f"task3_plateau_evidence_s{seed}_{short_commit}") / "result.json"


def run_training_seed(
    project_root: Path, manifest: Mapping[str, Any], *, seed: int,
    resume: bool,
) -> dict[str, Any]:
    commit = _git(project_root, "rev-parse", "HEAD")
    short_commit = commit[:7]
    result_path = _seed_evidence_path(project_root, seed, short_commit)
    parent_spec = manifest["task2_parents"][str(seed)]
    parent_run = _resolve(project_root, parent_spec["run"])
    _validate_parent(parent_run, parent_spec["checkpoint_sha256"])
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Seed evidence already exists: {result_path}")
        result = _json(result_path)
        if result.get("source_commit") != commit:
            raise ValueError("Cannot resume plateau evidence across source commits")
    else:
        result = {
            "schema_version": "task3-plateau-seed-result-v1",
            "status": "running", "source_commit": commit,
            "training_seed": seed, "parent_run": str(parent_run),
            "parent": None, "history": [], "decision": None,
        }
        _write_json(result_path, result)

    development_seeds = tuple(int(item) for item in manifest["development_seeds"])
    if result.get("parent") is None:
        summaries, prefixes = _run_evaluation_pair(
            project_root=project_root, manifest=manifest, training_seed=seed,
            checkpoint=parent_run / "checkpoints" / "final.pt", role="parent",
            phase="development", evaluation_seeds=development_seeds,
            config_path=manifest["configs"]["development"], cpu_offset=seed,
        )
        result["parent"] = {"summaries": summaries, "evaluation_runs": prefixes}
        _write_json(result_path, result)

    tracker = PlateauTracker(
        manifest["plateau_stopping"], history=result.get("history", []))
    if tracker.state["stop"]:
        result["decision"] = tracker.state
        result["status"] = (
            "qualified" if tracker.state["selected_checkpoint_run"] else "gate_failed")
        _write_json(result_path, result)
        return result
    previous_run = (
        None if not result["history"] else
        project_root / "runs" / result["history"][-1]["checkpoint_run"])
    cumulative_round = (
        int(result["history"][-1]["cumulative_round"])
        if result["history"] else 0)
    while not tracker.state["stop"]:
        cumulative_round += int(manifest["plateau_stopping"]["interval_rounds"])
        segment = _train_segment(
            project_root=project_root, manifest=manifest, seed=seed,
            cumulative_round=cumulative_round, parent_run=parent_run,
            previous_run=previous_run,
        )
        child_summaries, child_prefixes = _run_evaluation_pair(
            project_root=project_root, manifest=manifest, training_seed=seed,
            checkpoint=segment / "checkpoints" / "final.pt", role="child",
            phase=f"development_c{cumulative_round:04d}",
            evaluation_seeds=development_seeds,
            config_path=manifest["configs"]["development"], cpu_offset=seed + 3,
        )
        parent_summaries = result["parent"]["summaries"]
        eligible, checks = evaluate_gates(
            parent_summaries, child_summaries, manifest["gates"])
        assessment = build_assessment(
            cumulative_round=cumulative_round, checkpoint_run=segment.name,
            checkpoint=segment / "checkpoints" / "final.pt",
            parent=parent_summaries, child=child_summaries,
            gates=manifest["gates"], gate_checks=checks,
        )
        assessment["evaluation_runs"] = child_prefixes
        assessment["bootstrap"] = _compare_pair(
            project_root=project_root, evidence_root=result_path.parent,
            phase=f"c{cumulative_round:04d}",
            evaluation_seeds=development_seeds,
            parent_prefixes=result["parent"]["evaluation_runs"],
            child_prefixes=child_prefixes,
            bootstrap_samples=int(manifest["bootstrap_samples"]),
        )
        decision = tracker.record(assessment)
        result["history"].append(assessment)
        result["decision"] = decision
        _write_json(result_path, result)
        previous_run = segment
    result["status"] = (
        "qualified" if tracker.state["selected_checkpoint_run"] else "gate_failed")
    result["decision"] = tracker.state
    _write_json(result_path, result)
    return result


def _selected_checkpoint(project_root: Path, seed_result: Mapping[str, Any]) -> Path:
    run = seed_result["decision"]["selected_checkpoint_run"]
    if not run:
        raise ValueError("Seed result has no qualified checkpoint")
    checkpoint = project_root / "runs" / run / "checkpoints" / "final.pt"
    if _sha256(checkpoint) != seed_result["decision"]["selected_checkpoint_sha256"]:
        raise ValueError("Selected plateau checkpoint hash mismatch")
    return checkpoint


def _evaluate_stage(
    project_root: Path, manifest: Mapping[str, Any], *, stage: str,
    seed_results: Mapping[int, Mapping[str, Any]], evaluation_seeds: Sequence[int],
    config_path: str,
) -> tuple[bool, dict[str, Any]]:
    commit = _git(project_root, "rev-parse", "HEAD")
    evidence = project_root / "runs" / f"task3_plateau_{stage}_{commit[:7]}"
    result_path = evidence / "result.json"
    if result_path.exists():
        result = _json(result_path)
        if result["status"] in ("passed", "gate_failed"):
            return result["status"] == "passed", result
        if result.get("source_commit") != commit:
            raise ValueError("Cannot resume stage evaluation across source commits")
    else:
        result = {
            "schema_version": "task3-plateau-stage-result-v1",
            "status": "running", "stage": stage, "source_commit": commit,
            "evaluation_seeds": list(evaluation_seeds), "per_seed": {},
        }
        _write_json(result_path, result)
    for offset, (seed, seed_result) in enumerate(sorted(seed_results.items())):
        if str(seed) in result["per_seed"]:
            continue
        parent_spec = manifest["task2_parents"][str(seed)]
        parent_run = _resolve(project_root, parent_spec["run"])
        parent_summaries, parent_prefixes = _run_evaluation_pair(
            project_root=project_root, manifest=manifest, training_seed=seed,
            checkpoint=parent_run / "checkpoints" / "final.pt", role="parent",
            phase=stage, evaluation_seeds=evaluation_seeds,
            config_path=config_path, cpu_offset=offset * 6,
        )
        child_checkpoint = _selected_checkpoint(project_root, seed_result)
        child_summaries, child_prefixes = _run_evaluation_pair(
            project_root=project_root, manifest=manifest, training_seed=seed,
            checkpoint=child_checkpoint, role="child", phase=stage,
            evaluation_seeds=evaluation_seeds, config_path=config_path,
            cpu_offset=offset * 6 + 3,
        )
        passed, checks = evaluate_gates(
            parent_summaries, child_summaries, manifest["gates"])
        result["per_seed"][str(seed)] = {
            "status": "passed" if passed else "gate_failed",
            "summaries": {"parent": parent_summaries, "child": child_summaries},
            "gate_checks": checks,
            "evaluation_runs": {"parent": parent_prefixes, "child": child_prefixes},
            "bootstrap": _compare_pair(
                project_root=project_root, evidence_root=evidence,
                phase=f"s{seed}", evaluation_seeds=evaluation_seeds,
                parent_prefixes=parent_prefixes, child_prefixes=child_prefixes,
                bootstrap_samples=int(manifest["bootstrap_samples"]),
            ),
            "checkpoint": str(child_checkpoint),
            "checkpoint_sha256": _sha256(child_checkpoint),
        }
        _write_json(result_path, result)
    stage_passed = all(
        item["status"] == "passed" for item in result["per_seed"].values())
    result["status"] = "passed" if stage_passed else "gate_failed"
    _write_json(result_path, result)
    return stage_passed, result


def _confirmation_key(item: Mapping[str, Any], seed: int) -> tuple[Any, ...]:
    parent = item["summaries"]["parent"]
    child = item["summaries"]["child"]
    task3 = child["task3"]
    return (
        -float(task3["mean_score"]), -float(task3["mean_kills"]),
        -float(task3["first_place_rate"]), float(task3["suicide_rate"]),
        -float(task3["bomb_survival_rate"]),
        -_minimum_retention(parent, child), float(task3["act_p95_seconds"]), seed,
    )


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != PLATEAU_VERSION:
        raise ValueError("unsupported Task 3 plateau manifest")
    result = dict(manifest)
    result["plateau_stopping"] = resolve_plateau_spec(manifest["plateau_stopping"])
    interval = result["plateau_stopping"]["interval_rounds"]
    maximum = result["plateau_stopping"]["max_total_rounds"]
    if result["training_checkpoints"] != list(range(interval, maximum + 1, interval)):
        raise ValueError("training_checkpoints do not match the plateau schedule")
    expected_sets = {
        "development_seeds": list(range(19200, 19220)),
        "confirmation_seeds": list(range(19300, 19320)),
        "main_validation_seeds": list(range(19400, 19500)),
        "reserved_final_test_seeds": list(range(20000, 20100)),
    }
    for name, expected in expected_sets.items():
        if result.get(name) != expected:
            raise ValueError(f"{name} does not match its preregistered range")
    return result


def run_pipeline(
    manifest_path: Path, project_root: Path, *, resume: bool = False,
) -> tuple[int, dict[str, Any]]:
    manifest = _validate_manifest(_json(manifest_path))
    commit = _git(project_root, "rev-parse", "HEAD")
    if _git(project_root, "status", "--porcelain"):
        raise ValueError("Task 3 plateau experiment requires a clean worktree")
    base = str(manifest["source_base"])
    if not _git_is_ancestor(project_root, base, commit):
        raise ValueError("Plateau implementation must descend from source_base")
    short_commit = commit[:7]
    pipeline_path = (
        project_root / "runs" / f"task3_plateau_pipeline_{short_commit}" / "result.json")
    if pipeline_path.exists() and not resume:
        raise FileExistsError(f"Pipeline evidence already exists: {pipeline_path}")
    state = _json(pipeline_path) if pipeline_path.exists() else {
        "schema_version": "task3-plateau-pipeline-result-v1",
        "status": "running_seed_33", "source_commit": commit,
        "seed_results": {}, "confirmation": None,
        "selected_training_seed": None, "main_validation": None,
        "qualified_for_task4": False,
    }
    _write_json(pipeline_path, state)
    try:
        seed33 = run_training_seed(
            project_root, manifest, seed=33, resume=resume)
        state["seed_results"]["33"] = str(
            _seed_evidence_path(project_root, 33, short_commit).relative_to(project_root))
        if seed33["status"] != "qualified":
            state["status"] = "stopped_seed_33_unqualified"
            _write_json(pipeline_path, state)
            return GATE_FAILURE, state
        state["status"] = "running_replication_11_22"
        _write_json(pipeline_path, state)
        replicated: dict[int, dict[str, Any]] = {33: seed33}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_training_seed, project_root, manifest,
                    seed=seed, resume=resume): seed
                for seed in (11, 22)
            }
            for future in as_completed(futures):
                seed = futures[future]
                replicated[seed] = future.result()
                state["seed_results"][str(seed)] = str(
                    _seed_evidence_path(project_root, seed, short_commit).relative_to(project_root))
                _write_json(pipeline_path, state)
        if any(item["status"] != "qualified" for item in replicated.values()):
            state["status"] = "stopped_replication_unqualified"
            _write_json(pipeline_path, state)
            return GATE_FAILURE, state
        state["status"] = "running_confirmation"
        _write_json(pipeline_path, state)
        confirmation_passed, confirmation = _evaluate_stage(
            project_root, manifest, stage="confirmation",
            seed_results=replicated,
            evaluation_seeds=manifest["confirmation_seeds"],
            config_path=manifest["configs"]["confirmation"],
        )
        state["confirmation"] = str(
            (project_root / "runs" / f"task3_plateau_confirmation_{short_commit}" /
             "result.json").relative_to(project_root))
        if not confirmation_passed:
            state["status"] = "stopped_confirmation_failure"
            _write_json(pipeline_path, state)
            return GATE_FAILURE, state
        selected_seed = min(
            replicated,
            key=lambda seed: _confirmation_key(
                confirmation["per_seed"][str(seed)], seed),
        )
        state["selected_training_seed"] = selected_seed
        state["status"] = "running_main_validation"
        _write_json(pipeline_path, state)
        main_passed, main_result = _evaluate_stage(
            project_root, manifest, stage=f"main_s{selected_seed}",
            seed_results={selected_seed: replicated[selected_seed]},
            evaluation_seeds=manifest["main_validation_seeds"],
            config_path=manifest["configs"]["main_validation"],
        )
        state["main_validation"] = str(
            (project_root / "runs" /
             f"task3_plateau_main_s{selected_seed}_{short_commit}" /
             "result.json").relative_to(project_root))
        state["status"] = "passed" if main_passed else "stopped_main_validation_failure"
        state["qualified_for_task4"] = bool(main_passed)
        _write_json(pipeline_path, state)
        return (PASS if main_passed else GATE_FAILURE), state
    except BaseException as exception:
        state.update({
            "status": "infrastructure_error",
            "error": {"type": type(exception).__name__, "message": str(exception)},
        })
        _write_json(pipeline_path, state)
        return INFRASTRUCTURE_ERROR, state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        code, _ = run_pipeline(
            args.manifest.resolve(), args.project_root.resolve(), resume=args.resume)
        return code
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return INFRASTRUCTURE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
