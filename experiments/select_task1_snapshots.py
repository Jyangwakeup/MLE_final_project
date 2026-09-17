"""Rank frozen Task 1 checkpoints from their immutable evaluation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_directories(path: Path) -> list[Path]:
    """Return leaf evaluation runs from a single- or multi-seed invocation."""
    if (path / "episodes.jsonl").is_file():
        return [path]
    runs = sorted(child for child in path.iterdir()
                  if child.is_dir() and (child / "episodes.jsonl").is_file())
    if not runs:
        raise ValueError(f"{path} has no evaluation episode files")
    return runs


def summarize(path: Path, agent: str) -> dict:
    rows = []
    metadata = None
    for run in _episode_directories(path):
        run_metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        if metadata is None:
            metadata = run_metadata
        elif run_metadata["checkpoint"] != metadata["checkpoint"]:
            raise ValueError(f"{path} mixes checkpoints")
        for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            target = next(item for item in episode["agents"] if item["name"] == agent)
            rows.append((target, int(episode["round_steps"])))
    if not rows:
        raise ValueError(f"{path} contains no {agent} episodes")
    coins = [item[0]["coins"] for item in rows]
    completed = [steps for target, steps in rows if target["all_coins"]]
    total_steps = sum(steps for _, steps in rows)
    loop_rate = sum(
        bool(target["long_wait_loop"] or target["long_ping_pong_loop"])
        for target, _ in rows
    ) / len(rows)
    assert metadata is not None
    checkpoint = Path(metadata["checkpoint"])
    return {
        "evaluation_run": str(path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "episode_count": len(rows),
        "all_coins_rate": sum(target["all_coins"] for target, _ in rows) / len(rows),
        "mean_coins": sum(coins) / len(coins),
        "coins_per_100_steps": 100 * sum(coins) / total_steps,
        "completion_steps": None if not completed else sum(completed) / len(completed),
        "loop_rate": loop_rate,
    }


def ranking_key(item: dict):
    return (
        -item["all_coins_rate"], -item["mean_coins"],
        -item["coins_per_100_steps"],
        float("inf") if item["completion_steps"] is None else item["completion_steps"],
        item["loop_rate"], item["checkpoint"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--evaluation-run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    candidates = [summarize(path.resolve(), args.agent) for path in args.evaluation_run]
    candidates.sort(key=ranking_key)
    payload = {
        "schema_version": "task1-snapshot-selection-v1",
        "selection_order": [
            "all_coins_rate", "mean_coins", "coins_per_100_steps",
            "completion_steps", "loop_rate", "checkpoint",
        ],
        "candidates": candidates,
        "best": candidates[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
