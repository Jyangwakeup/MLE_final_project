"""Aggregate raw experiment episodes into a CSV summary and score chart."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt


EPISODE_SCHEMA_VERSION = "episode-v1"
AGENT_METRICS = ("coins", "kills", "suicides", "crates", "invalid")
SUMMARY_FIELDS = (
    "run_id",
    "agent_name",
    "episode_count",
    "mean_score",
    "score_std",
    "coins",
    "kills",
    "suicides",
    "crates",
    "invalid_actions",
    "survival_rate",
    "mean_survival_steps",
    "exclusive_wins",
    "tied_first",
    "zero_score_ties",
)


def _invalid(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}: line {line_number}: {message}")


def _number(value: Any, field: str, path: Path, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _invalid(path, line_number, f"{field} must be a finite number")
    return float(value)


def _validate_episode(value: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(path, line_number, "episode must be a JSON object")
    if value.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise _invalid(path, line_number, f"schema_version must be {EPISODE_SCHEMA_VERSION!r}")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise _invalid(path, line_number, "run_id must be a non-empty string")
    if not isinstance(value.get("round_index"), int) or isinstance(value["round_index"], bool):
        raise _invalid(path, line_number, "round_index must be an integer")
    if not isinstance(value.get("round_steps"), int) or isinstance(value["round_steps"], bool):
        raise _invalid(path, line_number, "round_steps must be an integer")

    agents = value.get("agents")
    if not isinstance(agents, list) or not agents:
        raise _invalid(path, line_number, "agents must be a non-empty list")

    names: set[str] = set()
    for agent in agents:
        if not isinstance(agent, dict):
            raise _invalid(path, line_number, "each agent must be an object")
        name = agent.get("name")
        if not isinstance(name, str) or not name:
            raise _invalid(path, line_number, "agent.name must be a non-empty string")
        if name in names:
            raise _invalid(path, line_number, f"duplicate agent name: {name}")
        names.add(name)
        _number(agent.get("score"), "agent.score", path, line_number)
        for metric in AGENT_METRICS:
            _number(agent.get(metric), f"agent.{metric}", path, line_number)
        survived = agent.get("survived")
        dead = agent.get("dead")
        if not isinstance(survived, bool) or not isinstance(dead, bool) or survived == dead:
            raise _invalid(path, line_number, "agent.survived and agent.dead must be opposite booleans")
    return value


def _read_episodes(run_directory: Path) -> list[dict[str, Any]]:
    episodes_path = run_directory / "episodes.jsonl"
    if not episodes_path.is_file():
        raise ValueError(f"{run_directory}: missing episodes.jsonl")

    episodes: list[dict[str, Any]] = []
    for line_number, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise _invalid(episodes_path, line_number, "blank lines are not valid episode records")
        try:
            raw_episode = json.loads(line)
        except json.JSONDecodeError as exception:
            raise _invalid(episodes_path, line_number, f"invalid JSON: {exception.msg}") from exception
        episode = _validate_episode(raw_episode, episodes_path, line_number)
        if episode["run_id"] != run_directory.name:
            raise _invalid(
                episodes_path,
                line_number,
                f"run_id {episode['run_id']!r} does not match directory name {run_directory.name!r}",
            )
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"{episodes_path}: contains no episode records")
    return episodes


def _ranking(agents: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    scores = {agent["name"]: float(agent["score"]) for agent in agents}
    highest_score = max(scores.values())
    first = {name for name, score in scores.items() if score == highest_score}
    if highest_score == 0 and all(score == 0 for score in scores.values()):
        return set(), set(), set(scores)
    if len(first) == 1:
        return first, set(), set()
    return set(), first, set()


def _summary_row(run_id: str, agent_name: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [sample["score"] for sample in samples]
    return {
        "run_id": run_id,
        "agent_name": agent_name,
        "episode_count": len(samples),
        "mean_score": mean(scores),
        "score_std": pstdev(scores),
        "coins": sum(sample["coins"] for sample in samples),
        "kills": sum(sample["kills"] for sample in samples),
        "suicides": sum(sample["suicides"] for sample in samples),
        "crates": sum(sample["crates"] for sample in samples),
        "invalid_actions": sum(sample["invalid"] for sample in samples),
        "survival_rate": mean(sample["survived"] for sample in samples),
        # C3 records complete-round steps, but not a dead agent's final active step.
        "mean_survival_steps": None,
        "exclusive_wins": sum(sample["exclusive_win"] for sample in samples),
        "tied_first": sum(sample["tied_first"] for sample in samples),
        "zero_score_ties": sum(sample["zero_score_tie"] for sample in samples),
    }


def summarize_runs(run_directories: Iterable[Path]) -> list[dict[str, Any]]:
    """Return one provenance-preserving summary row for every run and agent."""
    samples_by_agent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    processed_directories: set[Path] = set()
    for candidate in run_directories:
        run_directory = Path(candidate).resolve()
        if not run_directory.is_dir():
            raise ValueError(f"{run_directory}: run directory does not exist")
        if run_directory in processed_directories:
            raise ValueError(f"{run_directory}: duplicate run directory")
        processed_directories.add(run_directory)

        for episode in _read_episodes(run_directory):
            exclusive, tied, zero_score = _ranking(episode["agents"])
            for agent in episode["agents"]:
                samples_by_agent[(episode["run_id"], agent["name"])].append(
                    {
                        "score": float(agent["score"]),
                        "coins": float(agent["coins"]),
                        "kills": float(agent["kills"]),
                        "suicides": float(agent["suicides"]),
                        "crates": float(agent["crates"]),
                        "invalid": float(agent["invalid"]),
                        "survived": float(agent["survived"]),
                        "exclusive_win": float(agent["name"] in exclusive),
                        "tied_first": float(agent["name"] in tied),
                        "zero_score_tie": float(agent["name"] in zero_score),
                    }
                )

    return [
        _summary_row(run_id, agent_name, samples)
        for (run_id, agent_name), samples in sorted(samples_by_agent.items())
    ]


def _write_summary(rows: list[dict[str, Any]], output_directory: Path) -> Path:
    summary_path = output_directory / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _write_mean_score_chart(rows: list[dict[str, Any]], output_directory: Path) -> Path:
    labels = [f"{row['run_id']}\n{row['agent_name']}" for row in rows]
    scores = [row["mean_score"] for row in rows]
    figure, axis = plt.subplots(figsize=(max(6, len(rows) * 1.3), 4.5))
    axis.bar(range(len(rows)), scores)
    axis.set_ylabel("Mean score per round")
    axis.set_title("Experiment mean scores")
    axis.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
    figure.tight_layout()
    chart_path = output_directory / "mean_score.png"
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return chart_path


def analyze_runs(run_directories: Iterable[Path], output_directory: Path) -> list[dict[str, Any]]:
    """Aggregate raw episodes and write ``summary.csv`` plus ``mean_score.png``."""
    rows = summarize_runs(run_directories)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_summary(rows, output)
    _write_mean_score_chart(rows, output)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analyze_runs(args.runs, args.output)
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
