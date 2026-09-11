"""Create a compact report from per-round structured training metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt


def analyze_training(run_directory: Path) -> dict[str, object]:
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
        "schema_version": "training-summary-v1",
        "algorithm": rows[-1]["algorithm"],
        "rounds": len(rows),
        "final_action_steps": steps[-1],
        "final_epsilon": epsilons[-1],
        "mean_reward": mean(rewards),
        "mean_reward_last_100": mean(rewards[-tail_size:]),
        "best_round_reward": max(rewards),
        "best_round": rounds[rewards.index(max(rewards))],
        "final_q_states": int(rows[-1]["q_states"]) if rows[-1].get("q_states") else None,
        "final_loss": losses[-1] if losses else None,
        "final_updates": int(rows[-1]["updates"]) if rows[-1].get("updates") else None,
        "checkpoint": rows[-1]["checkpoint"],
    }
    (run_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figure, reward_axis = plt.subplots(figsize=(8, 4.5))
    reward_axis.plot(rounds, rewards, color="#176B87", linewidth=1, label="round reward")
    reward_axis.set_xlabel("Training round")
    reward_axis.set_ylabel("Reward")
    reward_axis.grid(alpha=0.25)
    epsilon_axis = reward_axis.twinx()
    epsilon_axis.plot(rounds, epsilons, color="#D95F59", linewidth=1, label="epsilon")
    epsilon_axis.set_ylabel("Epsilon")
    figure.tight_layout()
    figure.savefig(run_directory / "training_progress.png", dpi=150)
    plt.close(figure)
    return summary
