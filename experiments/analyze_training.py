"""Summarise per-round training metrics without changing raw experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


_TRAINING_COLUMNS = (
    "schema_version", "algorithm", "round", "reward", "action_steps", "epsilon",
    "q_states", "loss", "updates", "checkpoint",
)
_ROUND_COLUMNS = (
    "run_id", "algorithm", "seed", "round", "coins", "reward", "epsilon",
    "q_states", "dqn_last_update_loss", "updates", "coins_rolling_mean",
    "reward_rolling_mean", "cumulative_coins",
)


def analyze_training(run_directory: Path) -> dict[str, object]:
    """Create the established one-run training summary and plot."""
    training_path = run_directory / "training.csv"
    if not training_path.is_file():
        raise FileNotFoundError(f"Training metrics do not exist: {training_path}")
    with training_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Training metrics are empty: {training_path}")
    rounds = [int(row["round"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    steps = [int(row["action_steps"]) for row in rows]
    epsilons = [float(row["epsilon"]) for row in rows]
    losses = [float(row["loss"]) for row in rows if row.get("loss")]
    if not all(math.isfinite(value) for value in [*rewards, *epsilons, *losses]):
        raise ValueError(f"Training metrics contain non-finite values: {training_path}")
    tail_size = min(100, len(rewards))
    summary = {
        "schema_version": "training-summary-v1", "algorithm": rows[-1]["algorithm"],
        "rounds": len(rows), "final_action_steps": steps[-1], "final_epsilon": epsilons[-1],
        "mean_reward": mean(rewards), "mean_reward_last_100": mean(rewards[-tail_size:]),
        "best_round_reward": max(rewards), "best_round": rounds[rewards.index(max(rewards))],
        "final_q_states": int(rows[-1]["q_states"]) if rows[-1].get("q_states") else None,
        "final_loss": losses[-1] if losses else None,
        "final_updates": int(rows[-1]["updates"]) if rows[-1].get("updates") else None,
        "checkpoint": rows[-1]["checkpoint"],
    }
    (run_directory / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure, reward_axis = plt.subplots(figsize=(8, 4.5))
    reward_axis.plot(rounds, rewards, color="#176B87", linewidth=1)
    reward_axis.set(xlabel="Training round", ylabel="Reward")
    reward_axis.grid(alpha=.25)
    epsilon_axis = reward_axis.twinx()
    epsilon_axis.plot(rounds, epsilons, color="#D95F59", linewidth=1)
    epsilon_axis.set_ylabel("Epsilon")
    figure.tight_layout(); figure.savefig(run_directory / "training_progress.png", dpi=150); plt.close(figure)
    return summary


def analyze_training_runs(
    run_directories: Iterable[Path], output_directory: Path, *, rolling_window: int = 25,
    evaluation_run_directories: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Aggregate complete comparable training runs, treating each seed as one replicate."""
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")
    directories = [Path(directory) for directory in run_directories]
    if not directories:
        raise ValueError("At least one training run is required")
    if len(set(directories)) != len(directories):
        raise ValueError("A training run was supplied more than once")
    loaded = [_load_training_run(directory) for directory in directories]
    _validate_comparable_runs(loaded)
    if not _schedules_identical(loaded):
        raise ValueError("Exploration schedules differ between comparison runs")
    round_metrics = _round_metrics(loaded, rolling_window)
    run_summary = [_summarize_run(run["rows"]) for run in loaded]
    algorithm_summary = _summarize_algorithms(run_summary)
    metadata: dict[str, Any] = {
        "schema_version": "training-analysis-v1", "input_runs": [str(path) for path in directories],
        "task": loaded[0]["task"], "feature_version": loaded[0]["feature_version"],
        "reward_version": loaded[0]["reward_version"], "round_budget": loaded[0]["round_budget"],
        "rolling_window": rolling_window, "replication_unit": "training seed",
        "statistics": {
            "algorithm_mean": "mean of per-seed run means",
            "algorithm_sample_sd": "sample standard deviation across per-seed run means",
            "rolling_mean": "trailing window including the current round",
            "first_last_window": "first/last min(100, completed rounds) rounds per seed",
        },
        "exploration_schedules_identical": _schedules_identical(loaded), "evaluation": None,
    }
    # Validation is complete before this point, so invalid inputs leave no report.
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "round_metrics.csv", _ROUND_COLUMNS, round_metrics)
    _write_csv(output_directory / "run_summary.csv", tuple(run_summary[0]), run_summary)
    _write_csv(output_directory / "algorithm_summary.csv", tuple(algorithm_summary[0]), algorithm_summary)
    _plot_reports(round_metrics, run_summary, output_directory, rolling_window)
    evaluation_directories = [Path(path) for path in evaluation_run_directories or ()]
    if evaluation_directories:
        from experiments.analyze import analyze_runs
        evaluation_output = output_directory / "evaluation"
        analyze_runs(evaluation_directories, evaluation_output)
        metadata["evaluation"] = {
            "input_runs": [str(path) for path in evaluation_directories],
            "output_directory": str(evaluation_output),
            "purpose": "frozen evaluation; not training-curve promotion evidence",
        }
    (output_directory / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"round_metrics": round_metrics, "run_summary": run_summary, "algorithm_summary": algorithm_summary, "metadata": metadata, "output_directory": output_directory}


def _load_training_run(run_directory: Path) -> dict[str, Any]:
    required = {name: run_directory / name for name in ("metadata.json", "episodes.jsonl", "training.csv")}
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Run is missing required {name}: {path}")
    try:
        metadata = json.loads(required["metadata.json"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid metadata JSON: {required['metadata.json']}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be an object: {required['metadata.json']}")
    for key in ("algorithm", "seed", "task", "feature_version", "reward_version", "termination"):
        if key not in metadata:
            raise ValueError(f"Metadata is missing {key}: {required['metadata.json']}")
    termination = metadata["termination"]
    if not isinstance(termination, dict) or not isinstance(termination.get("requested_rounds"), int):
        raise ValueError(f"Metadata has no integer requested_rounds: {required['metadata.json']}")
    algorithm, seed = str(metadata["algorithm"]), metadata["seed"]
    if not isinstance(seed, int):
        raise ValueError(f"Metadata seed must be an integer: {required['metadata.json']}")
    training_rows = _read_training_rows(required["training.csv"], algorithm)
    episodes = _read_episodes(required["episodes.jsonl"])
    if [row["round"] for row in training_rows] != [row["round"] for row in episodes]:
        raise ValueError(f"Training and episode round identifiers differ: {run_directory}")
    if len(training_rows) != termination["requested_rounds"]:
        raise ValueError(f"Completed rounds do not match requested round budget: {run_directory}")
    rows = []
    for training, episode in zip(training_rows, episodes, strict=True):
        rows.append({"run_id": str(metadata.get("run_id", run_directory.name)), "algorithm": algorithm, "seed": seed,
                     "round": training["round"], "coins": episode["coins"], "reward": training["reward"],
                     "epsilon": training["epsilon"], "q_states": training["q_states"],
                     "dqn_last_update_loss": training["loss"], "updates": training["updates"],
                     "action_steps": training["action_steps"]})
    return {"directory": run_directory, "algorithm": algorithm, "seed": seed, "task": str(metadata["task"]),
            "feature_version": str(metadata["feature_version"]), "reward_version": str(metadata["reward_version"]),
            "round_budget": termination["requested_rounds"], "rows": rows}


def _read_training_rows(path: Path, expected_algorithm: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(_TRAINING_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Training metrics have an invalid header: {path}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError(f"Training metrics are empty: {path}")
    rows = []
    for expected_round, source in enumerate(source_rows, start=1):
        try:
            row = {"round": int(source["round"]), "reward": float(source["reward"]), "action_steps": int(source["action_steps"]),
                   "epsilon": float(source["epsilon"]), "q_states": _optional_int(source.get("q_states")),
                   "loss": _optional_float(source.get("loss")), "updates": _optional_int(source.get("updates"))}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Training metrics have an invalid numeric value: {path}") from error
        if source["algorithm"] != expected_algorithm:
            raise ValueError(f"Training algorithm disagrees with metadata: {path}")
        if row["round"] != expected_round:
            raise ValueError(f"Training round identifiers must be consecutive from 1: {path}")
        if not all(value is None or math.isfinite(value) for value in (row["reward"], row["epsilon"], row["loss"])):
            raise ValueError(f"Training metrics contain non-finite values: {path}")
        rows.append(row)
    return rows


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    rows = []
    for expected_round, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            episode = json.loads(line); agents = episode["agents"]
            if not isinstance(agents, list) or len(agents) != 1:
                raise ValueError("requires one agent")
            round_index, coins = int(episode["round_index"]), float(agents[0]["coins"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid episode record in {path}, line {expected_round}") from error
        if round_index != expected_round:
            raise ValueError(f"Episode round identifiers must be consecutive from 1: {path}")
        if not math.isfinite(coins):
            raise ValueError(f"Episode coins are non-finite: {path}")
        rows.append({"round": round_index, "coins": coins})
    if not rows:
        raise ValueError(f"Episodes are empty: {path}")
    return rows


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _optional_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def _validate_comparable_runs(runs: list[dict[str, Any]]) -> None:
    reference = runs[0]
    for run in runs[1:]:
        for key in ("task", "feature_version", "reward_version", "round_budget"):
            if run[key] != reference[key]:
                raise ValueError(f"{'Task' if key == 'task' else key} differs between comparison runs")
    seen: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    for run in runs:
        identifier = (run["algorithm"], run["seed"])
        if identifier in seen:
            raise ValueError(f"A duplicate algorithm/seed pair was supplied: {identifier}")
        run_id = run["rows"][0]["run_id"]
        if run_id in run_ids:
            raise ValueError(f"A duplicate run_id was supplied: {run_id}")
        seen.add(identifier)
        run_ids.add(run_id)


def _round_metrics(runs: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        cumulative = 0.0
        for index, row in enumerate(run["rows"]):
            cumulative += row["coins"]
            history = run["rows"][max(0, index - window + 1):index + 1]
            result = dict(row, coins_rolling_mean=mean(item["coins"] for item in history),
                          reward_rolling_mean=mean(item["reward"] for item in history), cumulative_coins=cumulative)
            output.append({key: result[key] for key in _ROUND_COLUMNS})
    return output


def _summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    window = min(100, len(rows)); first, last, final = rows[:window], rows[-window:], rows[-1]
    first_coins, last_coins = mean(row["coins"] for row in first), mean(row["coins"] for row in last)
    return {"run_id": final["run_id"], "algorithm": final["algorithm"], "seed": final["seed"], "rounds": len(rows),
            "action_steps": final["action_steps"], "mean_coins": mean(row["coins"] for row in rows),
            "first_100_mean_coins": first_coins, "last_100_mean_coins": last_coins,
            "coins_change_last_minus_first_100": last_coins - first_coins, "mean_reward": mean(row["reward"] for row in rows),
            "first_100_mean_reward": mean(row["reward"] for row in first), "last_100_mean_reward": mean(row["reward"] for row in last),
            "final_epsilon": final["epsilon"], "final_q_states": final["q_states"],
            "final_dqn_last_update_loss": final["dqn_last_update_loss"], "final_updates": final["updates"]}


def _summarize_algorithms(run_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_summary: grouped[row["algorithm"]].append(row)
    result = []
    metrics = ("mean_coins", "first_100_mean_coins", "last_100_mean_coins", "coins_change_last_minus_first_100", "mean_reward")
    for algorithm in sorted(grouped):
        rows = grouped[algorithm]; summary: dict[str, Any] = {"algorithm": algorithm, "n_seeds": len(rows)}
        for metric in metrics:
            values = [row[metric] for row in rows]
            summary[metric], summary[f"{metric}_sample_sd"] = mean(values), stdev(values) if len(values) > 1 else None
        result.append(summary)
    return result


def _schedules_identical(runs: list[dict[str, Any]]) -> bool:
    schedules = [tuple(row["epsilon"] for row in run["rows"]) for run in runs]
    return all(schedule == schedules[0] for schedule in schedules[1:])


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _plot_reports(metrics: list[dict[str, Any]], summaries: list[dict[str, Any]], output: Path, window: int) -> None:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics: runs[row["run_id"]].append(row)
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for run in runs.values(): grouped[run[0]["algorithm"]].append(run)
    colors = {"q_learning": "#176B87", "dqn": "#D95F59"}
    for algorithm in ("q_learning", "dqn"):
        figure, axis = plt.subplots(figsize=(8, 4.5)); color = colors[algorithm]
        for run in grouped.get(algorithm, []):
            rounds = [row["round"] for row in run]
            axis.plot(rounds, [row["coins"] for row in run], color=color, alpha=.18, linewidth=.8)
            axis.plot(rounds, [row["coins_rolling_mean"] for row in run], color=color, linewidth=2, label=f"seed {run[0]['seed']} rolling mean")
        axis.set(title=f"Coins per training round: {algorithm}", xlabel="Training round", ylabel="Coins"); axis.grid(alpha=.25)
        if grouped.get(algorithm): axis.legend(fontsize=8)
        figure.tight_layout(); figure.savefig(output / f"coins_per_round_{algorithm}.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for algorithm, algorithm_runs in sorted(grouped.items()):
        values = [[row["coins_rolling_mean"] for row in run] for run in algorithm_runs]; rounds = [row["round"] for row in algorithm_runs[0]]
        averages = [mean(seed_values) for seed_values in zip(*values)]
        deviations = [stdev(seed_values) if len(seed_values) > 1 else 0.0 for seed_values in zip(*values)]
        axis.plot(rounds, averages, color=colors.get(algorithm, "#444"), linewidth=2, label=f"{algorithm} mean")
        axis.fill_between(rounds, [a-d for a, d in zip(averages, deviations)], [a+d for a, d in zip(averages, deviations)], color=colors.get(algorithm, "#444"), alpha=.18, label=f"{algorithm} sample SD")
    axis.set(title=f"Coins comparison (trailing {window}-round mean)", xlabel="Training round", ylabel="Coins"); axis.grid(alpha=.25); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(output / "coins_algorithm_comparison.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, row in enumerate(sorted(summaries, key=lambda value: (value["algorithm"], value["seed"]))):
        color = colors.get(row["algorithm"], "#444")
        axis.plot([index-.14, index+.14], [row["first_100_mean_coins"], row["last_100_mean_coins"]], color=color, marker="o")
        axis.text(index, min(row["first_100_mean_coins"], row["last_100_mean_coins"])-.4, f"{row['algorithm']}\nseed {row['seed']}", ha="center", va="top", fontsize=7)
    axis.set(title="First vs last 100 rounds", ylabel="Mean coins"); axis.set_xticks([]); axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(output / "coins_first_vs_last100.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for algorithm, algorithm_runs in sorted(grouped.items()):
        cumulative = [[row["cumulative_coins"] for row in run] for run in algorithm_runs]; rounds = [row["round"] for row in algorithm_runs[0]]
        for run in algorithm_runs: axis.plot(rounds, [row["cumulative_coins"] for row in run], color=colors.get(algorithm, "#444"), alpha=.18)
        axis.plot(rounds, [mean(values) for values in zip(*cumulative)], color=colors.get(algorithm, "#444"), linewidth=2, label=f"{algorithm} mean")
    axis.set(title="Cumulative coins", xlabel="Training round", ylabel="Cumulative coins"); axis.grid(alpha=.25); axis.legend()
    figure.tight_layout(); figure.savefig(output / "cumulative_coins.png", dpi=150); plt.close(figure)
    figure, (reward_axis, epsilon_axis) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for algorithm, algorithm_runs in sorted(grouped.items()):
        rounds = [row["round"] for row in algorithm_runs[0]]; color = colors.get(algorithm, "#444")
        reward_axis.plot(rounds, [mean(values) for values in zip(*[[row["reward_rolling_mean"] for row in run] for run in algorithm_runs])], color=color, label=algorithm)
        epsilon_axis.plot(rounds, [row["epsilon"] for row in algorithm_runs[0]], color=color, label=algorithm)
    reward_axis.set(title=f"Reward and epsilon progress (trailing {window}-round reward mean)", ylabel="Reward"); epsilon_axis.set(xlabel="Training round", ylabel="Epsilon")
    reward_axis.legend(); epsilon_axis.legend(); reward_axis.grid(alpha=.25); epsilon_axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(output / "reward_epsilon_progress.png", dpi=150); plt.close(figure)
    figure, (q_axis, dqn_axis) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for run in grouped.get("q_learning", []): q_axis.plot([row["round"] for row in run], [row["q_states"] for row in run], label=f"seed {run[0]['seed']}")
    q_axis.set(title="Learner diagnostics", ylabel="Q states"); q_axis.grid(alpha=.25)
    if grouped.get("q_learning"): q_axis.legend(fontsize=8)
    updates_axis = dqn_axis.twinx()
    for run in grouped.get("dqn", []):
        dqn_axis.plot([row["round"] for row in run], _rolling_optional([row["dqn_last_update_loss"] for row in run], window), label=f"loss seed {run[0]['seed']}")
        updates_axis.plot([row["round"] for row in run], [row["updates"] for row in run], linestyle="--", alpha=.7, label=f"updates seed {run[0]['seed']}")
    dqn_axis.set(xlabel="Training round", ylabel="Last-update loss rolling mean"); updates_axis.set_ylabel("Cumulative updates"); dqn_axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(output / "learner_diagnostics.png", dpi=150); plt.close(figure)


def _rolling_optional(values: list[float | None], window: int) -> list[float]:
    output, previous = [], math.nan
    for index, value in enumerate(values):
        selected = [item for item in values[max(0, index-window+1):index+1] if item is not None]
        previous = mean(selected) if selected else previous; output.append(previous)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate comparable Bomberman training runs.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument("--evaluation-runs", nargs="*", type=Path)
    arguments = parser.parse_args()
    analyze_training_runs(arguments.runs, arguments.output, rolling_window=arguments.rolling_window, evaluation_run_directories=arguments.evaluation_runs)


if __name__ == "__main__":
    main()
