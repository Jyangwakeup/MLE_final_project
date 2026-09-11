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
TIMING_SCHEMA_VERSION = "timing-v1"
AGENT_METRICS = ("coins", "kills", "suicides", "crates", "invalid")
SUMMARY_FIELDS = (
    "run_id",
    "agent_name",
    "episode_count",
    "total_score",
    "rank_by_total_score",
    "mean_score",
    "score_std",
    "coins",
    "mean_coins",
    "kills",
    "mean_kills",
    "suicides",
    "suicide_rate",
    "killed_by_opponent",
    "killed_by_opponent_rate",
    "crates",
    "mean_crates",
    "invalid_actions",
    "invalid_action_rate",
    "survival_rate",
    "mean_survival_steps",
    "exclusive_wins",
    "exclusive_win_rate",
    "tied_first",
    "tied_first_rate",
    "zero_score_ties",
    "zero_score_tie_rate",
    "act_count",
    "act_mean_time",
    "act_p95_time",
    "act_max_time",
    "act_timeout_count",
    "act_skipped_count",
    "unseen_q_states",
    "q_decisions",
    "unseen_q_state_rate",
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
        death_step = agent.get("death_step")
        if death_step is not None and (
            not isinstance(death_step, int) or isinstance(death_step, bool) or death_step < 1
        ):
            raise _invalid(path, line_number, "agent.death_step must be null or a positive integer")
        killed_by_opponent = agent.get("killed_by_opponent", False)
        if not isinstance(killed_by_opponent, bool):
            raise _invalid(path, line_number, "agent.killed_by_opponent must be a boolean")
        killed_by_self = agent.get("killed_by_self", bool(agent.get("suicides", 0)))
        if not isinstance(killed_by_self, bool):
            raise _invalid(path, line_number, "agent.killed_by_self must be a boolean")
        death_causes = agent.get("death_causes", [])
        if not isinstance(death_causes, list):
            raise _invalid(path, line_number, "agent.death_causes must be a list")
        for cause in death_causes:
            if not isinstance(cause, dict):
                raise _invalid(path, line_number, "each death cause must be an object")
            if cause.get("type") not in {"self_bomb", "opponent_bomb"}:
                raise _invalid(path, line_number, "death cause type must be self_bomb or opponent_bomb")
            if not isinstance(cause.get("owner"), str) or not cause["owner"]:
                raise _invalid(path, line_number, "death cause owner must be a non-empty string")
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


def _validate_timing(value: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(path, line_number, "timing record must be a JSON object")
    if value.get("schema_version") != TIMING_SCHEMA_VERSION:
        raise _invalid(path, line_number, f"schema_version must be {TIMING_SCHEMA_VERSION!r}")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise _invalid(path, line_number, "run_id must be a non-empty string")
    if not isinstance(value.get("agent_name"), str) or not value["agent_name"]:
        raise _invalid(path, line_number, "agent_name must be a non-empty string")
    for field in ("skipped", "timed_out"):
        if not isinstance(value.get(field), bool):
            raise _invalid(path, line_number, f"{field} must be a boolean")
    think_time = value.get("think_time")
    if think_time is not None:
        _number(think_time, "think_time", path, line_number)
    return value


def _read_timing(run_directory: Path) -> list[dict[str, Any]]:
    timing_path = run_directory / "timing.jsonl"
    if not timing_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(timing_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise _invalid(timing_path, line_number, "blank lines are not valid timing records")
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exception:
            raise _invalid(timing_path, line_number, f"invalid JSON: {exception.msg}") from exception
        record = _validate_timing(raw_record, timing_path, line_number)
        if record["run_id"] != run_directory.name:
            raise _invalid(
                timing_path,
                line_number,
                f"run_id {record['run_id']!r} does not match directory name {run_directory.name!r}",
            )
        records.append(record)
    return records


def _read_q_diagnostics(run_directory: Path) -> dict[str, dict[str, float]]:
    diagnostics_path = run_directory / "q_diagnostics.jsonl"
    if not diagnostics_path.exists():
        return {}
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"unseen_q_states": 0.0, "q_decisions": 0.0}
    )
    for line_number, line in enumerate(diagnostics_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise _invalid(diagnostics_path, line_number, "blank lines are not valid Q diagnostic records")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exception:
            raise _invalid(diagnostics_path, line_number, f"invalid JSON: {exception.msg}") from exception
        if not isinstance(record, dict):
            raise _invalid(diagnostics_path, line_number, "Q diagnostic record must be an object")
        if record.get("run_id") != run_directory.name:
            raise _invalid(diagnostics_path, line_number, "run_id does not match directory name")
        agent_name = record.get("agent_name")
        if not isinstance(agent_name, str) or not agent_name:
            raise _invalid(diagnostics_path, line_number, "agent_name must be a non-empty string")
        totals[agent_name]["unseen_q_states"] += _number(
            record.get("unseen_q_states", 0), "unseen_q_states", diagnostics_path, line_number
        )
        totals[agent_name]["q_decisions"] += _number(
            record.get("q_decisions", 0), "q_decisions", diagnostics_path, line_number
        )
    return totals


def _ranking(agents: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    scores = {agent["name"]: float(agent["score"]) for agent in agents}
    highest_score = max(scores.values())
    first = {name for name, score in scores.items() if score == highest_score}
    if highest_score == 0 and all(score == 0 for score in scores.values()):
        return set(), set(), set(scores)
    if len(first) == 1:
        return first, set(), set()
    return set(), first, set()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _summary_row(
    run_id: str,
    agent_name: str,
    samples: list[dict[str, Any]],
    timing: dict[str, Any],
    q_diagnostics: dict[str, float],
) -> dict[str, Any]:
    scores = [sample["score"] for sample in samples]
    episode_count = len(samples)
    act_times = timing.get("think_times", [])
    q_decisions = q_diagnostics.get("q_decisions", 0.0)
    unseen_q_states = q_diagnostics.get("unseen_q_states", 0.0)
    return {
        "run_id": run_id,
        "agent_name": agent_name,
        "episode_count": episode_count,
        "total_score": sum(scores),
        "rank_by_total_score": None,
        "mean_score": mean(scores),
        "score_std": pstdev(scores),
        "coins": sum(sample["coins"] for sample in samples),
        "mean_coins": sum(sample["coins"] for sample in samples) / episode_count,
        "kills": sum(sample["kills"] for sample in samples),
        "mean_kills": sum(sample["kills"] for sample in samples) / episode_count,
        "suicides": sum(sample["suicides"] for sample in samples),
        "suicide_rate": sum(sample["suicides"] for sample in samples) / episode_count,
        "killed_by_opponent": sum(sample["killed_by_opponent"] for sample in samples),
        "killed_by_opponent_rate": sum(sample["killed_by_opponent"] for sample in samples) / episode_count,
        "crates": sum(sample["crates"] for sample in samples),
        "mean_crates": sum(sample["crates"] for sample in samples) / episode_count,
        "invalid_actions": sum(sample["invalid"] for sample in samples),
        "invalid_action_rate": (
            sum(sample["invalid"] for sample in samples)
            / max(1.0, sum(sample["round_steps"] for sample in samples))
        ),
        "survival_rate": mean(sample["survived"] for sample in samples),
        "mean_survival_steps": mean(sample["survival_steps"] for sample in samples),
        "exclusive_wins": sum(sample["exclusive_win"] for sample in samples),
        "exclusive_win_rate": sum(sample["exclusive_win"] for sample in samples) / episode_count,
        "tied_first": sum(sample["tied_first"] for sample in samples),
        "tied_first_rate": sum(sample["tied_first"] for sample in samples) / episode_count,
        "zero_score_ties": sum(sample["zero_score_tie"] for sample in samples),
        "zero_score_tie_rate": sum(sample["zero_score_tie"] for sample in samples) / episode_count,
        "act_count": timing.get("act_count", 0),
        "act_mean_time": mean(act_times) if act_times else None,
        "act_p95_time": _percentile(act_times, 95),
        "act_max_time": max(act_times) if act_times else None,
        "act_timeout_count": timing.get("timeout_count", 0),
        "act_skipped_count": timing.get("skipped_count", 0),
        "unseen_q_states": unseen_q_states,
        "q_decisions": q_decisions,
        "unseen_q_state_rate": unseen_q_states / q_decisions if q_decisions else None,
    }


def summarize_runs(run_directories: Iterable[Path]) -> list[dict[str, Any]]:
    """Return one provenance-preserving summary row for every run and agent."""
    samples_by_agent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    timing_by_agent: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"think_times": [], "act_count": 0, "timeout_count": 0, "skipped_count": 0}
    )
    q_by_agent: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"unseen_q_states": 0.0, "q_decisions": 0.0}
    )
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
                        "killed_by_opponent": float(agent.get("killed_by_opponent", False)),
                        "round_steps": float(episode["round_steps"]),
                        "survival_steps": float(
                            episode["round_steps"]
                            if agent["survived"]
                            else agent.get("death_step") or episode["round_steps"]
                        ),
                        "exclusive_win": float(agent["name"] in exclusive),
                        "tied_first": float(agent["name"] in tied),
                        "zero_score_tie": float(agent["name"] in zero_score),
                    }
                )

        for record in _read_timing(run_directory):
            bucket = timing_by_agent[(record["run_id"], record["agent_name"])]
            if record["think_time"] is not None:
                bucket["think_times"].append(float(record["think_time"]))
                bucket["act_count"] += 1
            if record["timed_out"]:
                bucket["timeout_count"] += 1
            if record["skipped"]:
                bucket["skipped_count"] += 1

        for agent_name, diagnostics in _read_q_diagnostics(run_directory).items():
            bucket = q_by_agent[(run_directory.name, agent_name)]
            bucket["unseen_q_states"] += diagnostics["unseen_q_states"]
            bucket["q_decisions"] += diagnostics["q_decisions"]

    rows = [
        _summary_row(
            run_id,
            agent_name,
            samples,
            timing_by_agent[(run_id, agent_name)],
            q_by_agent[(run_id, agent_name)],
        )
        for (run_id, agent_name), samples in sorted(samples_by_agent.items())
    ]
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_run[row["run_id"]].append(row)
    for run_rows in rows_by_run.values():
        ordered_scores = sorted({row["total_score"] for row in run_rows}, reverse=True)
        for row in run_rows:
            row["rank_by_total_score"] = ordered_scores.index(row["total_score"]) + 1
    return rows


def _weighted_mean(rows: list[dict[str, Any]], field: str, weight_field: str = "episode_count") -> float | None:
    weighted_values = [
        (float(row[field]), float(row[weight_field]))
        for row in rows
        if row.get(field) is not None and float(row[weight_field]) > 0
    ]
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_agent[row["agent_name"]].append(row)

    averages: list[dict[str, Any]] = []
    for agent_name, agent_rows in sorted(rows_by_agent.items()):
        mean_scores = [float(row["mean_score"]) for row in agent_rows]
        q_decisions = sum(float(row["q_decisions"]) for row in agent_rows)
        unseen_q_states = sum(float(row["unseen_q_states"]) for row in agent_rows)
        act_times = [row["act_mean_time"] for row in agent_rows if row["act_mean_time"] is not None]
        p95_times = [row["act_p95_time"] for row in agent_rows if row["act_p95_time"] is not None]
        max_times = [row["act_max_time"] for row in agent_rows if row["act_max_time"] is not None]
        averages.append(
            {
                "run_id": "AVERAGE",
                "agent_name": agent_name,
                "episode_count": sum(int(row["episode_count"]) for row in agent_rows),
                "total_score": sum(float(row["total_score"]) for row in agent_rows),
                "rank_by_total_score": mean(float(row["rank_by_total_score"]) for row in agent_rows),
                "mean_score": _weighted_mean(agent_rows, "mean_score"),
                "score_std": pstdev(mean_scores) if len(mean_scores) > 1 else 0.0,
                "coins": sum(float(row["coins"]) for row in agent_rows),
                "mean_coins": _weighted_mean(agent_rows, "mean_coins"),
                "kills": sum(float(row["kills"]) for row in agent_rows),
                "mean_kills": _weighted_mean(agent_rows, "mean_kills"),
                "suicides": sum(float(row["suicides"]) for row in agent_rows),
                "suicide_rate": _weighted_mean(agent_rows, "suicide_rate"),
                "killed_by_opponent": sum(float(row["killed_by_opponent"]) for row in agent_rows),
                "killed_by_opponent_rate": _weighted_mean(agent_rows, "killed_by_opponent_rate"),
                "crates": sum(float(row["crates"]) for row in agent_rows),
                "mean_crates": _weighted_mean(agent_rows, "mean_crates"),
                "invalid_actions": sum(float(row["invalid_actions"]) for row in agent_rows),
                "invalid_action_rate": _weighted_mean(agent_rows, "invalid_action_rate"),
                "survival_rate": _weighted_mean(agent_rows, "survival_rate"),
                "mean_survival_steps": _weighted_mean(agent_rows, "mean_survival_steps"),
                "exclusive_wins": sum(float(row["exclusive_wins"]) for row in agent_rows),
                "exclusive_win_rate": _weighted_mean(agent_rows, "exclusive_win_rate"),
                "tied_first": sum(float(row["tied_first"]) for row in agent_rows),
                "tied_first_rate": _weighted_mean(agent_rows, "tied_first_rate"),
                "zero_score_ties": sum(float(row["zero_score_ties"]) for row in agent_rows),
                "zero_score_tie_rate": _weighted_mean(agent_rows, "zero_score_tie_rate"),
                "act_count": sum(int(row["act_count"]) for row in agent_rows),
                "act_mean_time": mean(act_times) if act_times else None,
                "act_p95_time": mean(p95_times) if p95_times else None,
                "act_max_time": max(max_times) if max_times else None,
                "act_timeout_count": sum(int(row["act_timeout_count"]) for row in agent_rows),
                "act_skipped_count": sum(int(row["act_skipped_count"]) for row in agent_rows),
                "unseen_q_states": unseen_q_states,
                "q_decisions": q_decisions,
                "unseen_q_state_rate": unseen_q_states / q_decisions if q_decisions else None,
            }
        )
    return averages


def _write_summary(rows: list[dict[str, Any]], output_directory: Path) -> Path:
    summary_path = output_directory / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows([*rows, *_average_rows(rows)])
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
