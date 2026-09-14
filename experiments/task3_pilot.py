"""Validate and score the preregistered Task 3 pilot experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys
from typing import Any, Iterable, Sequence

from agent_code.team_agent.feature_system import get_feature_schema
from agent_code.team_agent.rewards import resolve_reward_spec


ROLE_TASKS = {
    "parent_task1": "coin_navigation",
    "parent_task2": "crate_navigation",
    "parent_task3": "weak_opponents",
    "child_task1": "coin_navigation",
    "child_task2": "crate_navigation",
    "child_task3": "weak_opponents",
}
EVIDENCE_FIELDS = (
    "training_seed", "cumulative_task3_rounds", "role", "task",
    "environment_seed", "run_id", "checkpoint", "source_commit", "source_hash",
    "status", "score", "coins", "crates", "kills", "suicides", "bombs",
    "bombs_resolved", "bombs_survived", "invalid_actions", "survived",
    "exclusive_first", "tied_first", "act_count", "act_p95_seconds",
    "act_max_seconds", "timeouts", "skipped_actions",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = manifest_path.parent.parent / manifest["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = manifest["contracts"]
    training = config["training"]
    if config["algorithm"] != contract["algorithm"]:
        raise ValueError("Task 3 config algorithm differs from the preregistration")
    if config["feature_id"] != contract["feature_id"]:
        raise ValueError("Task 3 config feature differs from the preregistration")
    if get_feature_schema(config["feature_id"]).vector_shape != (
            contract["feature_dimensions"],):
        raise ValueError("Task 3 feature dimensions differ from the preregistration")
    if config["reward_id"] != contract["reward_id"]:
        raise ValueError("Task 3 config reward differs from the preregistration")
    if resolve_reward_spec(config["reward_id"]) != resolve_reward_spec(
            "r7_safe_credit_sparse"):
        raise ValueError("Task 3 pilot must keep the complete r7 reward spec")
    if config["safety"] != contract["safety"]:
        raise ValueError("Task 3 config safety contract differs from preregistration")
    if training.get("target_stage_action_steps") is not None:
        raise ValueError("Task 3 pilot cannot stop on an action count")
    if training.get("performance_stopping") is not None:
        raise ValueError("Task 3 pilot cannot use Task 1 performance stopping")
    if training.get("early_stopping", {}).get("enabled") is not False:
        raise ValueError("Task 3 pilot rolling early stopping must be disabled")
    expected_seeds = list(range(
        manifest["development"]["seeds"]["start"],
        manifest["development"]["seeds"]["end"] + 1,
    ))
    if config["evaluation"]["seeds"] != expected_seeds:
        raise ValueError("Task 3 evaluation seeds differ from the preregistration")
    manifest["resolved_config"] = {
        "path": str(config_path), "sha256": _sha256(config_path),
    }
    return manifest, config


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Evaluation timing data is empty")
    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _evaluation_directories(root: Path) -> list[Path]:
    directories = sorted(
        path for path in Path(root).iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
        and (path / "episodes.jsonl").is_file()
    )
    if not directories:
        raise ValueError(f"No evaluation runs found under {root}")
    return directories


def load_evaluation(
    root: Path, *, role: str, training_seed: int, cumulative_rounds: int,
    agent: str, expected_seeds: set[int], expected_source: dict[str, str],
) -> list[dict[str, Any]]:
    if role not in ROLE_TASKS:
        raise ValueError(f"Unknown Task 3 evaluation role: {role}")
    rows = []
    for directory in _evaluation_directories(root):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        episodes = [json.loads(line) for line in (
            directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        if metadata.get("status") != "completed" or len(episodes) != 1:
            raise ValueError(f"Evaluation {directory} is not one completed round")
        if metadata.get("task") != ROLE_TASKS[role]:
            raise ValueError(f"Evaluation {directory} has the wrong Task")
        if metadata.get("exploration_disabled") is not True:
            raise ValueError(f"Evaluation {directory} did not disable exploration")
        if metadata.get("source_commit") != expected_source["commit"] or metadata.get(
                "source_hash") != expected_source["hash"]:
            raise ValueError(f"Evaluation {directory} has the wrong source identity")
        episode = episodes[0]
        environment_seed = int(episode["environment_seed"])
        targets = [item for item in episode["agents"] if item["name"] == agent]
        if len(targets) != 1:
            raise ValueError(f"Evaluation {directory} does not contain one target agent")
        target = targets[0]
        scores = [float(item["score"]) for item in episode["agents"]]
        top = max(scores)
        leaders = sum(float(item["score"]) == top for item in episode["agents"])
        zero_tie = top == 0 and all(score == 0 for score in scores)
        timing = [json.loads(line) for line in (
            directory / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
        timing = [item for item in timing if item.get("agent_name") == agent]
        think_times = [float(item["think_time"]) for item in timing]
        row = {
            "training_seed": training_seed,
            "cumulative_task3_rounds": cumulative_rounds,
            "role": role,
            "task": ROLE_TASKS[role],
            "environment_seed": environment_seed,
            "run_id": directory.name,
            "checkpoint": metadata.get("checkpoint"),
            "source_commit": metadata["source_commit"],
            "source_hash": metadata["source_hash"],
            "status": metadata["status"],
            "score": float(target["score"]),
            "coins": float(target.get("coins", 0)),
            "crates": float(target.get("crates", 0)),
            "kills": float(target.get("kills", 0)),
            "suicides": float(target.get("suicides", 0)),
            "bombs": float(target.get("bombs", 0)),
            "bombs_resolved": float(target.get("bombs_resolved", 0)),
            "bombs_survived": float(target.get("bombs_survived", 0)),
            "invalid_actions": float(target.get("invalid", 0)),
            "survived": float(bool(target.get("survived", False))),
            "exclusive_first": float(not zero_tie and target["score"] == top and leaders == 1),
            "tied_first": float(not zero_tie and target["score"] == top and leaders > 1),
            "act_count": len(timing),
            "act_p95_seconds": _percentile(think_times, 95),
            "act_max_seconds": max(think_times),
            "timeouts": sum(bool(item.get("timed_out")) for item in timing),
            "skipped_actions": sum(bool(item.get("skipped")) for item in timing),
        }
        if role.endswith("task1") and row["score"] != row["coins"]:
            raise ValueError(f"Task 1 score and coins differ in {directory}")
        rows.append(row)
    seeds = [int(row["environment_seed"]) for row in rows]
    if len(rows) != len(expected_seeds) or set(seeds) != expected_seeds:
        raise ValueError(f"Evaluation {root} does not contain the exact development seeds")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Evaluation {root} contains duplicate environment seeds")
    return sorted(rows, key=lambda item: int(item["environment_seed"]))


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot summarize empty evidence")
    bombs_resolved = sum(float(row["bombs_resolved"]) for row in rows)
    actions = sum(int(row["act_count"]) for row in rows)
    return {
        "mean_score": mean(float(row["score"]) for row in rows),
        "mean_coins": mean(float(row["coins"]) for row in rows),
        "mean_crates": mean(float(row["crates"]) for row in rows),
        "mean_kills": mean(float(row["kills"]) for row in rows),
        "first_place_rate": mean(
            float(row["exclusive_first"]) + float(row["tied_first"]) for row in rows),
        "suicide_rate": mean(float(row["suicides"]) for row in rows),
        "zero_bomb_round_rate": mean(float(row["bombs"]) == 0 for row in rows),
        "survived_bomb_rate": (
            sum(float(row["bombs_survived"]) for row in rows) / bombs_resolved
            if bombs_resolved else 0.0
        ),
        "invalid_action_rate": (
            sum(float(row["invalid_actions"]) for row in rows) / actions
            if actions else 1.0
        ),
        "act_p95_seconds": max(float(row["act_p95_seconds"]) for row in rows),
        "act_max_seconds": max(float(row["act_max_seconds"]) for row in rows),
        "timeouts": sum(int(row["timeouts"]) for row in rows),
        "skipped_actions": sum(int(row["skipped_actions"]) for row in rows),
    }


def _paired_differences(
    child: Sequence[dict[str, Any]], parent: Sequence[dict[str, Any]], field: str,
) -> list[float]:
    parent_by_seed = {int(row["environment_seed"]): row for row in parent}
    if {int(row["environment_seed"]) for row in child} != set(parent_by_seed):
        raise ValueError("Parent and child evaluation seeds do not match")
    if field == "first_place":
        value = lambda row: float(row["exclusive_first"]) + float(row["tied_first"])
    else:
        value = lambda row: float(row[field])
    return [
        value(row) - value(parent_by_seed[int(row["environment_seed"])])
        for row in child
    ]


def _bootstrap(differences: Sequence[float], *, samples: int, seed: int) -> list[float]:
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return [estimates[int(.025 * samples)], estimates[min(samples - 1, int(.975 * samples))]]


def _retained(child: float, parent: float, threshold: float) -> bool:
    return child >= threshold * parent


def _ratio(child: float, parent: float) -> float:
    if parent == 0:
        return 1.0
    return child / parent


def validate_training_run(
    run: Path, *, expected_source: dict[str, str], expected_seed: int,
) -> dict[str, Any]:
    run = Path(run)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(f"Training run {run} is not completed")
    if metadata.get("task") != "weak_opponents" or metadata.get("seed") != expected_seed:
        raise ValueError(f"Training run {run} has the wrong Task or seed")
    for field in ("commit", "hash"):
        if metadata.get("source_" + field) != expected_source[field]:
            raise ValueError(f"Training run {run} has the wrong source {field}")
    checkpoint = run / "checkpoints" / "final.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Training run {run} has no final checkpoint")
    with (run / "training.csv").open(encoding="utf-8", newline="") as file:
        training_rows = list(csv.DictReader(file))
    if not training_rows:
        raise ValueError(f"Training run {run} has no training rows")
    last = training_rows[-1]
    if int(last["updates"]) <= 0 or not math.isfinite(float(last["loss"])):
        raise ValueError(f"Training run {run} has no finite learned state")
    latest = json.loads((run / "resume" / "latest.json").read_text(encoding="utf-8"))
    if latest.get("checkpoint_schema") != "training-resume-v7":
        raise ValueError(f"Training run {run} does not use v7 resume")
    generations = latest.get("generations", [])
    if len(generations) != 2:
        raise ValueError(f"Training run {run} does not retain two generations")
    for generation in generations:
        directory = run / "resume" / generation
        manifest_path = directory / "manifest.json"
        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected_hash in snapshot["files"].items():
            if _sha256(directory / name) != expected_hash:
                raise ValueError(f"Training run {run} has a corrupt {generation}/{name}")
    newest_manifest = run / "resume" / generations[0] / "manifest.json"
    if _sha256(newest_manifest) != latest["generation_hash"]:
        raise ValueError(f"Training run {run} latest generation hash is invalid")
    return {
        "run": str(run), "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "latest_generation": generations[0],
        "latest_generation_hash": latest["generation_hash"],
        "updates": int(last["updates"]), "loss": float(last["loss"]),
        "local_rounds": int(metadata["termination"]["local_completed_rounds"]),
    }


def assess(
    manifest: dict[str, Any], evidence: dict[str, list[dict[str, Any]]],
    training_run: dict[str, Any], *, training_seed: int, cumulative_rounds: int,
) -> dict[str, Any]:
    summaries = {role: summarize(rows) for role, rows in evidence.items()}
    gates = manifest["gates"]
    p1, p2, p3 = (summaries[f"parent_task{task}"] for task in (1, 2, 3))
    c1, c2, c3 = (summaries[f"child_task{task}"] for task in (1, 2, 3))
    checks = {
        "task3_score_gain": c3["mean_score"] - p3["mean_score"] >= gates["task3_mean_score_gain"],
        "task3_combat_gain": (
            c3["mean_kills"] - p3["mean_kills"] >= gates["task3_mean_kill_gain"]
            or c3["first_place_rate"] - p3["first_place_rate"]
            >= gates["task3_first_place_rate_gain"]
        ),
        "task3_coin_retention": _retained(c3["mean_coins"], p3["mean_coins"], gates["task3_coin_retention"]),
        "task3_crate_retention": _retained(c3["mean_crates"], p3["mean_crates"], gates["task3_crate_retention"]),
        "task1_score_retention": _retained(c1["mean_score"], p1["mean_score"], gates["task1_score_retention"]),
        "task2_coin_retention": _retained(c2["mean_coins"], p2["mean_coins"], gates["task2_coin_retention"]),
        "task2_crate_retention": _retained(c2["mean_crates"], p2["mean_crates"], gates["task2_crate_retention"]),
        "task2_suicide": c2["suicide_rate"] <= gates["suicide_rate"],
        "task3_suicide": c3["suicide_rate"] <= gates["suicide_rate"],
        "task2_bomb_survival": c2["survived_bomb_rate"] >= gates["survived_bomb_rate"],
        "task3_bomb_survival": c3["survived_bomb_rate"] >= gates["survived_bomb_rate"],
        "task3_bomb_activity": c3["zero_bomb_round_rate"] <= gates["zero_bomb_round_rate_task3"],
        "invalid_actions": all(item["invalid_action_rate"] <= gates["invalid_action_rate"] for item in (c1, c2, c3)),
        "act_p95": all(item["act_p95_seconds"] < gates["act_p95_seconds"] for item in (c1, c2, c3)),
        "act_max": all(item["act_max_seconds"] < gates["act_max_seconds"] for item in (c1, c2, c3)),
        "no_timeouts": all(item["timeouts"] == 0 for item in (c1, c2, c3)),
        "no_skipped_actions": all(item["skipped_actions"] == 0 for item in (c1, c2, c3)),
    }
    bootstrap_spec = manifest["development"]["bootstrap"]
    paired = {}
    fields = {
        "task1_score": ("child_task1", "parent_task1", "score"),
        "task2_coins": ("child_task2", "parent_task2", "coins"),
        "task2_crates": ("child_task2", "parent_task2", "crates"),
        "task3_score": ("child_task3", "parent_task3", "score"),
        "task3_coins": ("child_task3", "parent_task3", "coins"),
        "task3_crates": ("child_task3", "parent_task3", "crates"),
        "task3_kills": ("child_task3", "parent_task3", "kills"),
        "task3_first_place": ("child_task3", "parent_task3", "first_place"),
    }
    for name, (child_role, parent_role, field) in fields.items():
        differences = _paired_differences(
            evidence[child_role], evidence[parent_role], field)
        paired[name] = {
            "mean_difference": mean(differences),
            "bootstrap_ci95": _bootstrap(
                differences, samples=bootstrap_spec["samples"],
                seed=bootstrap_spec["rng_seed"]),
        }
    ratios = {
        "task1_score": _ratio(c1["mean_score"], p1["mean_score"]),
        "task2_coins": _ratio(c2["mean_coins"], p2["mean_coins"]),
        "task2_crates": _ratio(c2["mean_crates"], p2["mean_crates"]),
        "task3_coins": _ratio(c3["mean_coins"], p3["mean_coins"]),
        "task3_crates": _ratio(c3["mean_crates"], p3["mean_crates"]),
    }
    return {
        "schema_version": "task3-pilot-assessment-v1",
        "training_seed": training_seed,
        "cumulative_task3_rounds": cumulative_rounds,
        "training_run": training_run,
        "summaries": summaries,
        "paired": paired,
        "retention_ratios": ratios,
        "minimum_resource_and_retention_ratio": min(ratios.values()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_candidate(
    manifest: dict[str, Any], assessments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    expected_seeds = set(manifest["training"]["seeds"])
    actual_seeds = [int(item["training_seed"]) for item in assessments]
    if len(actual_seeds) != len(set(actual_seeds)) or set(actual_seeds) != expected_seeds:
        raise ValueError("Task 3 selection requires one assessment for each training seed")
    allowed_rounds = set(manifest["training"]["assessment_rounds"])
    if any(int(item["cumulative_task3_rounds"]) not in allowed_rounds for item in assessments):
        raise ValueError("Task 3 selection received a non-preregistered assessment round")

    def ranking(item: dict[str, Any]) -> tuple[Any, ...]:
        task3 = item["summaries"]["child_task3"]
        return (
            -float(task3["mean_score"]),
            float(task3["suicide_rate"]),
            -float(task3["mean_kills"]),
            -float(task3["first_place_rate"]),
            -float(item["minimum_resource_and_retention_ratio"]),
            float(task3["act_p95_seconds"]),
            str(item["training_run"]["run"]),
        )

    passed = all(bool(item["passed"]) for item in assessments)
    ranked = sorted(assessments, key=ranking) if passed else []
    selected = ranked[0] if ranked else None
    return {
        "schema_version": "task3-pilot-result-v1",
        "designation": (
            manifest["outcome"]["successful_designation"] if passed
            else "task3_pilot_failure"
        ),
        "all_training_seeds_passed": passed,
        "qualified_for_task4": False,
        "main_validation_performed": False,
        "training_seeds": sorted(actual_seeds),
        "assessments": list(assessments),
        "checkpoint_rank": [
            int(item["training_seed"]) for item in ranked
        ],
        "selected": None if selected is None else {
            "training_seed": int(selected["training_seed"]),
            "cumulative_task3_rounds": int(selected["cumulative_task3_rounds"]),
            **selected["training_run"],
        },
    }


def _write_evidence(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _assessment_command(args: argparse.Namespace) -> int:
    manifest, _ = load_preregistration(args.manifest)
    expected_seeds = set(range(
        manifest["development"]["seeds"]["start"],
        manifest["development"]["seeds"]["end"] + 1,
    ))
    roots = {
        role: getattr(args, role) for role in ROLE_TASKS
    }
    evidence = {
        role: load_evaluation(
            root, role=role, training_seed=args.training_seed,
            cumulative_rounds=args.cumulative_rounds,
            agent=manifest["contracts"]["agent"], expected_seeds=expected_seeds,
            expected_source=manifest["training_source"],
        ) for role, root in roots.items()
    }
    training = validate_training_run(
        args.training_run, expected_source=manifest["training_source"],
        expected_seed=args.training_seed)
    result = assess(
        manifest, evidence, training, training_seed=args.training_seed,
        cumulative_rounds=args.cumulative_rounds)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write_evidence(
        output / "evaluations.csv",
        (row for role in ROLE_TASKS for row in evidence[role]),
    )
    (output / "assessment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _selection_command(args: argparse.Namespace) -> int:
    manifest, _ = load_preregistration(args.manifest)
    assessments = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.assessments
    ]
    result = select_candidate(manifest, assessments)
    result["protocol"] = {
        "commit": args.protocol_commit,
        "config_sha256": manifest["resolved_config"]["sha256"],
        "training_source": manifest["training_source"],
    }
    evidence_rows = []
    for path in args.assessments:
        evidence_path = Path(path).with_name("evaluations.csv")
        with evidence_path.open(encoding="utf-8", newline="") as file:
            evidence_rows.extend(csv.DictReader(file))
    results_path = Path(args.results)
    evidence_path = Path(args.evidence)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_evidence(evidence_path, evidence_rows)
    if result["selected"] is not None:
        if args.candidate is None:
            raise ValueError("A passing family requires --candidate")
        candidate = {
            "schema_version": "task3-pilot-candidate-v1",
            "designation": "task3_pilot_candidate",
            "development_only": True,
            "qualified_for_task4": False,
            "main_validation_performed": False,
            "protocol": result["protocol"],
            "contracts": manifest["contracts"],
            "selected": result["selected"],
            "results": str(results_path),
            "evaluation_evidence": str(evidence_path),
        }
        candidate_path = Path(args.candidate)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    elif args.candidate is not None and Path(args.candidate).exists():
        raise ValueError("A failed family cannot retain a Task 3 candidate file")
    return 0


def _record_protocol_command(args: argparse.Namespace) -> int:
    manifest, _ = load_preregistration(args.manifest)
    run = Path(args.run)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    source = manifest["training_source"]
    if metadata.get("source_commit") != source["commit"] or metadata.get(
            "source_hash") != source["hash"]:
        raise ValueError("Task 3 run has the wrong training source identity")
    if metadata.get("config_source_sha256") != manifest["resolved_config"]["sha256"]:
        raise ValueError("Task 3 run did not use the preregistered config bytes")
    record = {
        "schema_version": "task3-pilot-run-protocol-v1",
        "protocol_commit": args.protocol_commit,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "config": manifest["resolved_config"],
        "training_source": source,
    }
    destination = run / "task3_protocol.json"
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise ValueError("Task 3 run already has a different protocol record")
        return 0
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-preregistration")
    validate.add_argument("--manifest", type=Path, default=Path(
        "experiments/task3_pilot.json"))
    assessment = subparsers.add_parser("assess")
    assessment.add_argument("--manifest", required=True, type=Path)
    assessment.add_argument("--training-seed", required=True, type=int)
    assessment.add_argument("--cumulative-rounds", required=True, type=int)
    assessment.add_argument("--training-run", required=True, type=Path)
    for role in ROLE_TASKS:
        assessment.add_argument("--" + role.replace("_", "-"), required=True, type=Path)
    assessment.add_argument("--output", required=True, type=Path)
    selection = subparsers.add_parser("select")
    selection.add_argument("--manifest", required=True, type=Path)
    selection.add_argument("--assessments", required=True, nargs="+", type=Path)
    selection.add_argument("--protocol-commit", required=True)
    selection.add_argument("--results", required=True, type=Path)
    selection.add_argument("--evidence", required=True, type=Path)
    selection.add_argument("--candidate", type=Path)
    record = subparsers.add_parser("record-protocol")
    record.add_argument("--manifest", required=True, type=Path)
    record.add_argument("--protocol-commit", required=True)
    record.add_argument("--run", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-preregistration":
            load_preregistration(args.manifest)
            return 0
        if args.command == "assess":
            return _assessment_command(args)
        if args.command == "select":
            return _selection_command(args)
        return _record_protocol_command(args)
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
