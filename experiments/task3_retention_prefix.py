"""Run and judge the preregistered Task 3 retention-prefix experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

# ``python experiments/task3_retention_prefix.py`` puts only the experiments
# directory on sys.path.  Add the repository root before importing sibling
# packages so the documented direct invocation works outside ``python -m``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.compare_evaluations import compare_evaluations
from experiments.resume import CHECKPOINT_SCHEMA_VERSION, _load_generation


PASS = 0
INFRASTRUCTURE_ERROR = 1
GATE_FAILURE = 2
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


def _resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _run_checked(command: Sequence[str], project_root: Path) -> None:
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    completed = subprocess.run(command, cwd=project_root, env=environment)
    if completed.returncode:
        raise RuntimeError(
            f"Subprocess exited with {completed.returncode}: {' '.join(command)}")


def _canonical_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(
            right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        return _canonical_equal(
            left.item() if isinstance(left, np.generic) else left,
            right.item() if isinstance(right, np.generic) else right,
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and (
            left.keys() == right.keys()
            and all(_canonical_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _canonical_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def verify_deterministic_prefix(
    candidate_run: Path, reference_run: Path, *, rounds: int,
) -> dict[str, Any]:
    """Verify the new run is the deterministic prefix of the prior longer run."""
    candidate_episodes = [
        json.loads(line) for line in (candidate_run / "episodes.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    reference_episodes = [
        json.loads(line) for line in (reference_run / "episodes.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ][:rounds]
    if len(candidate_episodes) != rounds or len(reference_episodes) != rounds:
        raise ValueError("Prefix comparison does not contain the requested rounds")
    for episode in (*candidate_episodes, *reference_episodes):
        episode.pop("run_id", None)
    if candidate_episodes != reference_episodes:
        raise ValueError("Episode records do not reproduce the reference prefix")

    def training_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            row.pop("checkpoint", None)
        return rows

    candidate_training = training_rows(candidate_run / "training.csv")
    reference_training = training_rows(reference_run / "training.csv", rounds)
    if len(candidate_training) != rounds or candidate_training != reference_training:
        raise ValueError("Training metrics do not reproduce the reference prefix")

    replay_name = f"round_{rounds:05d}.pt"
    with (candidate_run / "replays" / replay_name).open("rb") as handle:
        candidate_replay = pickle.load(handle)
    with (reference_run / "replays" / replay_name).open("rb") as handle:
        reference_replay = pickle.load(handle)
    if not _canonical_equal(candidate_replay, reference_replay):
        raise ValueError(f"{replay_name} does not reproduce the reference replay")
    return {
        "rounds": rounds,
        "episodes_equal": True,
        "training_metrics_equal": True,
        "sampled_replay_equal": True,
    }


def validate_training_run(run: Path, *, rounds: int) -> dict[str, Any]:
    metadata = _json(run / "metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError("Training run is not completed")
    termination = metadata.get("termination", {})
    if int(termination.get("local_completed_rounds", -1)) != rounds:
        raise ValueError("Training run completed an unexpected number of rounds")
    latest = _json(run / "resume" / "latest.json")
    generations = latest.get("generations", [])
    expected = [f"generation-{rounds:08d}", f"generation-{rounds - 1:08d}"]
    if generations != expected:
        raise ValueError("Training run does not retain the expected two generations")
    loaded = []
    for index, generation in enumerate(generations):
        snapshot = _load_generation(
            run, generation,
            latest.get("generation_hash") if index == 0 else None,
            expected_schema=CHECKPOINT_SCHEMA_VERSION,
        )
        loaded.append({
            "generation": snapshot.generation,
            "generation_hash": snapshot.generation_hash,
            "round": snapshot.round_index,
        })
    with (run / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != rounds:
        raise ValueError("training.csv has an unexpected number of rows")
    loss = float(rows[-1]["loss"])
    updates = int(rows[-1]["updates"])
    if not math.isfinite(loss) or updates <= 0:
        raise ValueError("Training checkpoint has invalid loss or update count")
    return {"generations": loaded, "final_loss": loss, "final_updates": updates}


def _summary_rows(evaluation_root: Path, agent: str, seeds: Sequence[int]) -> tuple[
    list[dict[str, str]], dict[str, str]
]:
    summary = evaluation_root / f"{evaluation_root.name}_summary" / "summary.csv"
    if not summary.is_file():
        raise ValueError(f"Evaluation summary is missing: {summary}")
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    agent_rows = [
        row for row in rows
        if row["agent_name"] == agent and row["run_id"] != "AVERAGE"
    ]
    average = next((
        row for row in rows
        if row["agent_name"] == agent and row["run_id"] == "AVERAGE"
    ), None)
    expected_ids = {f"{evaluation_root.name}_s{seed}" for seed in seeds}
    if average is None or {row["run_id"] for row in agent_rows} != expected_ids:
        raise ValueError("Evaluation is incomplete or uses unexpected seeds")
    if any(row["exploration_disabled"] != "True" for row in agent_rows):
        raise ValueError("Evaluation did not disable exploration")
    if any(int(row["episode_count"]) != 1 for row in agent_rows):
        raise ValueError("Evaluation must contain exactly one round per seed")
    return agent_rows, average


def summarize_evaluation(
    evaluation_root: Path, agent: str, seeds: Sequence[int],
) -> dict[str, float]:
    rows, average = _summary_rows(evaluation_root, agent, seeds)
    episodes = sum(int(row["episode_count"]) for row in rows)
    resolved = sum(float(row["bombs_resolved"] or 0) for row in rows)
    survived = sum(float(row["bombs_survived"] or 0) for row in rows)
    act_count = sum(int(row["act_count"] or 0) for row in rows)
    invalid = sum(float(row["invalid_actions"] or 0) for row in rows)
    total = lambda name: sum(float(row[name] or 0) for row in rows)
    return {
        "episodes": float(episodes),
        "mean_score": float(average["mean_score"]),
        "mean_coins": float(average["mean_coins"]),
        "mean_crates": float(average["mean_crates"]),
        "mean_kills": float(average["mean_kills"]),
        "first_place_rate": (
            total("exclusive_wins") + total("tied_first")) / episodes,
        "suicide_rate": total("suicides") / episodes,
        "bomb_survival_rate": survived / resolved if resolved else 0.0,
        "zero_bomb_round_rate": sum(
            float(row["mean_bombs"] or 0) == 0 for row in rows) / episodes,
        "invalid_action_rate": invalid / act_count if act_count else 0.0,
        "act_p95_seconds": float(average["act_p95_time"]),
        "act_max_seconds": float(average["act_max_time"]),
        "act_timeouts": total("act_timeout_count"),
        "act_skipped": total("act_skipped_count"),
        "avoidable_escape_collapses": total("avoidable_escape_collapses"),
        "robust_guarantee_losses": total("robust_guarantee_losses"),
        "robust_search_timeouts": total("robust_search_timeouts"),
    }


def evaluate_gates(
    parent: Mapping[str, Mapping[str, float]],
    child: Mapping[str, Mapping[str, float]],
    gates: Mapping[str, float],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, actual: float, threshold: float, operator: str) -> None:
        passed = actual >= threshold if operator == ">=" else actual <= threshold
        checks[name] = {
            "actual": actual, "operator": operator,
            "threshold": threshold, "passed": passed,
        }

    minimum_retention = float(gates["minimum_retention"])
    for task, metric in (
        ("task1", "mean_score"), ("task2", "mean_coins"),
        ("task2", "mean_crates"), ("task3", "mean_coins"),
        ("task3", "mean_crates"),
    ):
        denominator = float(parent[task][metric])
        retention = float(child[task][metric]) / denominator if denominator else 1.0
        check(f"{task}_{metric}_retention", retention, minimum_retention, ">=")
    check(
        "task3_score_gain",
        float(child["task3"]["mean_score"] - parent["task3"]["mean_score"]),
        float(gates["task3_score_gain"]), ">=",
    )
    kill_gain = float(child["task3"]["mean_kills"] - parent["task3"]["mean_kills"])
    first_gain = float(
        child["task3"]["first_place_rate"] - parent["task3"]["first_place_rate"])
    combat_passed = (
        kill_gain >= float(gates["task3_kill_gain"])
        or first_gain >= float(gates["task3_first_place_gain"])
    )
    checks["task3_combat_gain"] = {
        "kill_gain": kill_gain,
        "kill_threshold": float(gates["task3_kill_gain"]),
        "first_place_gain": first_gain,
        "first_place_threshold": float(gates["task3_first_place_gain"]),
        "operator": "OR", "passed": combat_passed,
    }
    for task in ("task2", "task3"):
        check(f"{task}_suicide_rate", float(child[task]["suicide_rate"]),
              float(gates["maximum_suicide_rate"]), "<=")
        check(f"{task}_bomb_survival_rate", float(child[task]["bomb_survival_rate"]),
              float(gates["minimum_bomb_survival_rate"]), ">=")
    check("task3_zero_bomb_round_rate", float(child["task3"]["zero_bomb_round_rate"]),
          float(gates["maximum_zero_bomb_round_rate"]), "<=")
    for task in ("task1", "task2", "task3"):
        check(f"{task}_invalid_action_rate", float(child[task]["invalid_action_rate"]),
              float(gates["maximum_invalid_rate"]), "<=")
        check(f"{task}_act_p95", float(child[task]["act_p95_seconds"]),
              float(gates["maximum_act_p95_seconds"]), "<=")
        check(f"{task}_act_max", float(child[task]["act_max_seconds"]),
              float(gates["maximum_act_seconds"]), "<=")
        for metric in (
            "act_timeouts", "act_skipped", "avoidable_escape_collapses",
            "robust_guarantee_losses", "robust_search_timeouts",
        ):
            check(f"{task}_{metric}", float(child[task][metric]), 0.0, "<=")
    return all(item["passed"] for item in checks.values()), checks


def _evaluation_directories(root: Path, prefix: str, seeds: Sequence[int]) -> list[Path]:
    return [root / prefix / f"{prefix}_s{seed}" for seed in seeds]


def _evaluation_command(
    project_root: Path, manifest: Mapping[str, Any], *, role: str, task: int,
    checkpoint: Path, prefix: str, cpu: int,
) -> list[str]:
    seeds = [str(seed) for seed in manifest["evaluation_seeds"]]
    return [
        "taskset", "-c", str(cpu), sys.executable, "experiments/run.py",
        "--config", str(_resolve(project_root, manifest["evaluation_config"])),
        "--mode", "evaluate", "--device", "cpu", "--task", str(task),
        "--agent", manifest["agent"], "--seeds", *seeds, "--n-rounds", "1",
        "--checkpoint", str(checkpoint), "--run-id", prefix,
    ]


def run_pipeline(
    manifest_path: Path, project_root: Path, *, attempt_suffix: str = "",
) -> tuple[int, dict[str, Any]]:
    manifest = _json(manifest_path)
    if attempt_suffix and not (
        attempt_suffix.startswith("_retry") and attempt_suffix[6:].isdigit()
    ):
        raise ValueError("attempt suffix must be empty or match _retryN")
    commit = _git(project_root, "rev-parse", "HEAD")
    short_commit = commit[:7]
    if _git(project_root, "status", "--porcelain"):
        raise ValueError("Task 3 retention experiment requires a clean worktree")
    expected_parent = str(manifest["expected_parent_commit"])
    actual_parent = _git(project_root, "rev-parse", "HEAD^")
    if actual_parent != expected_parent:
        raise ValueError(
            "Experiment commit is not the direct child of the preregistered "
            f"source commit: expected {expected_parent}, got {actual_parent}")
    parent_run = _resolve(project_root, manifest["parent_run"])
    reference_run = _resolve(project_root, manifest["reference_500_run"])
    parent_checkpoint = parent_run / "checkpoints" / "final.pt"
    run_id = manifest["run_id_template"].format(commit=short_commit) + attempt_suffix
    candidate_run = project_root / "runs" / run_id
    evidence_id = (
        manifest["evidence_id_template"].format(commit=short_commit) + attempt_suffix)
    evidence_root = project_root / "runs" / evidence_id
    prefixes = {
        role: {
            task: manifest["evaluation_id_template"].format(
                role=role, task=task, commit=short_commit) + attempt_suffix
            for task in (1, 2, 3)
        } for role in ("parent", "child")
    }
    targets = [candidate_run, evidence_root, *(
        project_root / "runs" / prefixes[role][task]
        for role in ("parent", "child") for task in (1, 2, 3)
    )]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite experiment outputs: " + ", ".join(existing))
    evidence_root.mkdir(parents=True)
    result_path = evidence_root / "result.json"
    result: dict[str, Any] = {
        "schema_version": "task3-retention-prefix-result-v1",
        "status": "running", "source_commit": commit,
        "manifest": str(manifest_path.relative_to(project_root)),
        "training_run": str(candidate_run.relative_to(project_root)),
        "evaluation_runs": prefixes,
    }
    _write_json(result_path, result)
    try:
        training_command = [
            "taskset", "-c", str(manifest["cpus"]["training"]),
            sys.executable, "experiments/run.py", "--config",
            str(_resolve(project_root, manifest["training_config"])),
            "--mode", "train", "--device", "cpu", "--task", "3",
            "--agent", manifest["agent"], "--seed", str(manifest["training_seed"]),
            "--n-rounds", str(manifest["training_rounds"]),
            "--transfer-task3-safety-from", str(parent_run),
            "--replay-policy", "sampled", "--replay-interval", "50",
            "--run-id", run_id,
        ]
        _run_checked(training_command, project_root)
        result["training_validation"] = validate_training_run(
            candidate_run, rounds=int(manifest["training_rounds"]))
        result["prefix_validation"] = verify_deterministic_prefix(
            candidate_run, reference_run, rounds=int(manifest["training_rounds"]))

        processes = []
        environment = dict(os.environ); environment.update(THREAD_ENVIRONMENT)
        child_checkpoint = candidate_run / "checkpoints" / "final.pt"
        for role, checkpoint in (("parent", parent_checkpoint), ("child", child_checkpoint)):
            for task in (1, 2, 3):
                command = _evaluation_command(
                    project_root, manifest, role=role, task=task,
                    checkpoint=checkpoint, prefix=prefixes[role][task],
                    cpu=int(manifest["cpus"]["evaluation"][role][str(task)]),
                )
                processes.append((role, task, command, subprocess.Popen(
                    command, cwd=project_root, env=environment)))
        failures = []
        for role, task, command, process in processes:
            code = process.wait()
            if code:
                failures.append({
                    "role": role, "task": task, "exit_code": code,
                    "command": command,
                })
        if failures:
            raise RuntimeError(f"Frozen evaluation subprocesses failed: {failures}")

        summaries: dict[str, dict[str, dict[str, float]]] = {
            "parent": {}, "child": {},
        }
        comparisons = {}
        seeds = tuple(int(seed) for seed in manifest["evaluation_seeds"])
        for task_number, task_name in ((1, "task1"), (2, "task2"), (3, "task3")):
            for role in ("parent", "child"):
                evaluation_root = project_root / "runs" / prefixes[role][task_number]
                summaries[role][task_name] = summarize_evaluation(
                    evaluation_root, manifest["agent"], seeds)
            comparison_root = evidence_root / "comparisons" / task_name
            comparisons[task_name] = compare_evaluations(
                _evaluation_directories(project_root / "runs", prefixes["child"][task_number], seeds),
                _evaluation_directories(project_root / "runs", prefixes["parent"][task_number], seeds),
                comparison_root, bootstrap_samples=int(manifest["bootstrap"]["samples"]),
            )
        passed, checks = evaluate_gates(
            summaries["parent"], summaries["child"], manifest["gates"])
        result.update({
            "status": "passed" if passed else "gate_failed",
            "summaries": summaries, "gate_checks": checks,
            "bootstrap": comparisons,
        })
        _write_json(result_path, result)
        return (PASS if passed else GATE_FAILURE), result
    except BaseException as exception:
        result.update({
            "status": "infrastructure_error",
            "error": {"type": type(exception).__name__, "message": str(exception)},
        })
        _write_json(result_path, result)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--attempt-suffix", default="")
    args = parser.parse_args(argv)
    try:
        code, _ = run_pipeline(
            args.manifest.resolve(), args.project_root.resolve(),
            attempt_suffix=args.attempt_suffix,
        )
        return code
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return INFRASTRUCTURE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
