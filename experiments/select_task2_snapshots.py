"""Apply the prospective Task 2 quality gate and rank paired frozen evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leaves(root: Path) -> list[Path]:
    if (root / "episodes.jsonl").is_file():
        return [root]
    leaves = sorted(path.parent for path in root.glob("*_s*/episodes.jsonl"))
    if not leaves:
        raise ValueError(f"{root} contains no evaluation episodes")
    return leaves


def _summary(root: Path, agent: str) -> dict:
    episodes = []
    summaries = []
    for leaf in _leaves(root):
        for line in (leaf / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            target = next(item for item in episode["agents"] if item["name"] == agent)
            episodes.append((episode, target))
        summary = leaf / "summary" / "summary.csv"
        if summary.is_file():
            summaries.extend(csv.DictReader(summary.open(encoding="utf-8")))
    count = len(episodes)
    steps = sum(int(episode["round_steps"]) for episode, _ in episodes)
    bombs = sum(float(target["bombs"]) for _, target in episodes)
    resolved = sum(float(target["bombs_resolved"]) for _, target in episodes)
    return {
        "episode_count": count,
        "mean_score": sum(float(target["score"]) for _, target in episodes) / count,
        "mean_coins": sum(float(target["coins"]) for _, target in episodes) / count,
        "zero_coin_round_rate": sum(float(target["coins"]) == 0 for _, target in episodes) / count,
        "all_coins_rate": sum(bool(target["all_coins"]) for _, target in episodes) / count,
        "mean_crates": sum(float(target["crates"]) for _, target in episodes) / count,
        "coins_per_100_steps": 100 * sum(float(target["coins"]) for _, target in episodes) / steps,
        "max_steps_rate": sum(bool(target["max_steps"]) for _, target in episodes) / count,
        "long_wait_loop_rate": sum(bool(target["long_wait_loop"]) for _, target in episodes) / count,
        "long_ping_pong_loop_rate": sum(bool(target["long_ping_pong_loop"]) for _, target in episodes) / count,
        "suicide_rate": sum(float(target["suicides"]) for _, target in episodes) / count,
        "zero_bomb_round_rate": sum(float(target["bombs"]) == 0 for _, target in episodes) / count,
        "zero_utility_bomb_rate": (
            sum(float(target.get("zero_utility_bombs", 0)) for _, target in episodes) / bombs
            if bombs else 1.0
        ),
        "crates_per_bomb": (
            sum(float(target["crates"]) for _, target in episodes) / bombs if bombs else 0.0
        ),
        "survived_bomb_rate": (
            sum(float(target["bombs_survived"]) for _, target in episodes) / resolved
            if resolved else 0.0
        ),
        "invalid_action_rate": sum(float(target["invalid"]) for _, target in episodes) / steps,
        "act_p95_seconds": max(float(row["act_p95_time"]) for row in summaries),
        "act_max_seconds": max(float(row["act_max_time"]) for row in summaries),
    }


def _ranking(item: dict) -> tuple:
    metrics = item["task2"]
    return (
        not item["passed"], -sum(item["gates"].values()), metrics["suicide_rate"],
        metrics["zero_bomb_round_rate"], -metrics["survived_bomb_rate"],
        -item["balanced_capability"], -metrics["mean_coins"],
        -item["task1_retention"], metrics["act_p95_seconds"], item["checkpoint"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--parent-task1", required=True, type=Path)
    parser.add_argument("--parent-task2", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument(
        "--candidate", action="append", nargs=3, required=True,
        metavar=("CHECKPOINT", "TASK1_EVAL", "TASK2_EVAL"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))["gates"]
    parent_task1 = _summary(args.parent_task1.resolve(), args.agent)
    parent_task2 = _summary(args.parent_task2.resolve(), args.agent)
    candidates = []
    for checkpoint_raw, task1_raw, task2_raw in args.candidate:
        checkpoint = Path(checkpoint_raw).resolve()
        task1 = _summary(Path(task1_raw).resolve(), args.agent)
        task2 = _summary(Path(task2_raw).resolve(), args.agent)
        retention = task1["mean_score"] / parent_task1["mean_score"]
        crate_gain = task2["mean_crates"] - parent_task2["mean_crates"]
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
        candidates.append({
            "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
            "task1": task1, "task2": task2, "task1_retention": retention,
            "crate_gain": crate_gain,
            "balanced_capability": min(
                task2["mean_coins"] / gate["task2_mean_coins_min"],
                task2["mean_crates"] / gate["task2_mean_crates_min"],
            ),
            "gates": checks, "passed": all(checks.values()),
        })
    candidates.sort(key=_ranking)
    payload = {
        "schema_version": "optimized-q-task2-selection-v1",
        "gate": str(args.gate.resolve()), "parent_task1": parent_task1,
        "parent_task2": parent_task2, "candidates": candidates,
        "best": candidates[0], "any_passed": any(item["passed"] for item in candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
