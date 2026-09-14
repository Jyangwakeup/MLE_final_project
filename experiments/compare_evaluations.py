"""Compare two frozen evaluations with matched environment seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence


TASK_METRICS = {
    "coin_navigation": ("score", "coins", "round_steps"),
    "crate_navigation": (
        "score", "coins", "crates", "suicides", "survived", "survival_steps"),
    "weak_opponents": (
        "score", "coins", "crates", "kills", "suicides", "survived",
        "survival_steps", "exclusive_win", "tied_first"),
    "full_match": (
        "score", "coins", "crates", "kills", "suicides", "survived",
        "survival_steps", "exclusive_win", "tied_first"),
}
FIELDS = (
    "metric", "paired_count", "candidate_mean", "reference_mean",
    "mean_difference", "median_difference", "candidate_wins", "ties",
    "candidate_losses", "bootstrap_ci95_low", "bootstrap_ci95_high",
)


def _load_group(directories: Iterable[Path]) -> tuple[str, str, dict[tuple[int, int], dict[str, float]]]:
    samples: dict[tuple[int, int], dict[str, float]] = {}
    task = agent_name = None
    for raw_directory in directories:
        directory = Path(raw_directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        current_task = str(metadata["task"])
        current_agent = str(metadata["agent"])
        if task is not None and current_task != task:
            raise ValueError("Evaluation directories use different tasks")
        if agent_name is not None and current_agent != agent_name:
            raise ValueError("Evaluation directories use different target agents")
        task, agent_name = current_task, current_agent
        for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            agents = episode["agents"]
            target = next((item for item in agents if item["name"] == agent_name), None)
            if target is None:
                raise ValueError(f"Target agent {agent_name!r} is absent from {directory}")
            scores = [float(item["score"]) for item in agents]
            top = max(scores)
            leaders = sum(float(item["score"]) == top for item in agents)
            survived = bool(target["survived"])
            row = {
                name: float(target.get(name, 0))
                for name in ("score", "coins", "crates", "kills", "suicides", "invalid")
            }
            row.update({
                "round_steps": float(episode["round_steps"]),
                "survived": float(survived),
                "survival_steps": float(
                    episode["round_steps"] if survived
                    else target.get("death_step") or episode["round_steps"]),
                "exclusive_win": float(float(target["score"]) == top and leaders == 1),
                "tied_first": float(float(target["score"]) == top and leaders > 1),
            })
            key = (int(episode["environment_seed"]), int(episode["round_index"]))
            if key in samples:
                raise ValueError(f"Duplicate environment seed/round pair: {key}")
            samples[key] = row
    if task is None or agent_name is None or not samples:
        raise ValueError("At least one non-empty evaluation directory is required")
    return task, agent_name, samples


def _bootstrap_ci(differences: list[float], metric: str, samples: int) -> tuple[float | None, float | None]:
    if len(differences) < 2:
        return None, None
    seed = int.from_bytes(hashlib.sha256(metric.encode("utf-8")).digest()[:8], "big")
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return estimates[int(.025 * samples)], estimates[min(samples - 1, int(.975 * samples))]


def compare_evaluations(
    candidate_directories: Iterable[Path], reference_directories: Iterable[Path],
    output_directory: Path, *, bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    task, candidate_agent, candidate = _load_group(candidate_directories)
    reference_task, reference_agent, reference = _load_group(reference_directories)
    if reference_task != task:
        raise ValueError("Candidate and reference tasks differ")
    if set(candidate) != set(reference):
        raise ValueError("Candidate and reference seed/round pairs differ")
    seeds = sorted({key[0] for key in candidate})
    rows = []
    for metric in TASK_METRICS.get(task, TASK_METRICS["full_match"]):
        candidate_values = [
            mean(candidate[key][metric] for key in sorted(candidate) if key[0] == seed)
            for seed in seeds
        ]
        reference_values = [
            mean(reference[key][metric] for key in sorted(reference) if key[0] == seed)
            for seed in seeds
        ]
        differences = [a - b for a, b in zip(candidate_values, reference_values)]
        low, high = _bootstrap_ci(differences, metric, bootstrap_samples)
        rows.append({
            "metric": metric, "paired_count": len(differences),
            "candidate_mean": mean(candidate_values),
            "reference_mean": mean(reference_values),
            "mean_difference": mean(differences),
            "median_difference": median(differences),
            "candidate_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "candidate_losses": sum(value < 0 for value in differences),
            "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
        })
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    report = {
        "schema_version": "paired-evaluation-v1", "task": task,
        "candidate_agent": candidate_agent, "reference_agent": reference_agent,
        "matched_episode_key": ["environment_seed", "round_index"],
        "replication_unit": "environment_seed",
        "bootstrap_samples": bootstrap_samples, "metrics": rows,
    }
    (output / "paired_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-runs", nargs="+", required=True, type=Path)
    parser.add_argument("--reference-runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        compare_evaluations(args.candidate_runs, args.reference_runs, args.output,
                            bootstrap_samples=args.bootstrap_samples)
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
