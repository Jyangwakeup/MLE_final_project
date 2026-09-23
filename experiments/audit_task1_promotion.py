"""Audit one formal Task 1 run and its independent frozen stage gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _leaves(root: Path) -> list[Path]:
    if (root / "episodes.jsonl").is_file():
        return [root]
    leaves = sorted(path.parent for path in root.glob("*_s*/episodes.jsonl"))
    if not leaves:
        raise ValueError(f"{root} contains no evaluation episodes")
    return leaves


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary_rows(root: Path, leaves: list[Path]) -> list[dict[str, str]]:
    """Read the aggregate multi-seed summary, or leaf summaries for one run."""
    aggregate = root / f"{root.name}_summary" / "summary.csv"
    if aggregate.is_file():
        return list(csv.DictReader(aggregate.open(encoding="utf-8")))
    rows: list[dict[str, str]] = []
    for leaf in leaves:
        summary = leaf / "summary" / "summary.csv"
        if summary.is_file():
            rows.extend(csv.DictReader(summary.open(encoding="utf-8")))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--stage-gate", required=True, type=Path)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    training = args.training_run.resolve()
    metadata = json.loads((training / "metadata.json").read_text(encoding="utf-8"))
    performance = metadata.get("termination", {}).get("performance_stopping") or {}
    history = performance.get("history") or []
    final_three = history[-3:]
    separated = len(final_three) == 3 and all(
        final_three[index]["cumulative_round"]
        - final_three[index - 1]["cumulative_round"] >= 50
        for index in range(1, 3)
    )
    checkpoint = Path(metadata["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (training / checkpoint).resolve()

    episodes = []
    stage_gate = args.stage_gate.resolve()
    leaves = _leaves(stage_gate)
    for leaf in leaves:
        for line in (leaf / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            target = next(item for item in episode["agents"] if item["name"] == args.agent)
            episodes.append((episode, target))
    summary_rows = _summary_rows(stage_gate, leaves)
    if not summary_rows:
        raise ValueError(f"{stage_gate} contains no timing summary")

    steps = sum(int(episode["round_steps"]) for episode, _ in episodes)
    mean_score = sum(float(target["score"]) for _, target in episodes) / len(episodes)
    invalid_rate = sum(float(target["invalid"]) for _, target in episodes) / steps
    p95 = max(float(row["act_p95_time"]) for row in summary_rows)
    maximum = max(float(row["act_max_time"]) for row in summary_rows)
    checks = {
        "training_status": metadata.get("status") in {"completed", "early_stopped"},
        "score_converged": bool(performance.get("converged")),
        "three_consecutive_passes": (
            separated and all(bool(item["passed"]) for item in final_three)
            and final_three[-1]["consecutive_passes"] >= 3
        ),
        "stage_gate_episode_count": len(episodes) == 20,
        "stage_gate_mean_score": mean_score >= 48.0,
        "invalid_action_rate": invalid_rate <= 0.01,
        "act_p95": p95 < 0.05,
        "act_max": maximum < 0.5,
        "checkpoint_exists": checkpoint.is_file(),
        "package_exists": args.package.is_file(),
    }
    payload = {
        "schema_version": "optimized-q-task1-promotion-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "training_run": str(training),
        "stage_gate": str(args.stage_gate.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        "assessment_history": final_three,
        "metrics": {
            "episode_count": len(episodes),
            "mean_score": mean_score,
            "invalid_action_rate": invalid_rate,
            "act_p95_seconds": p95,
            "act_max_seconds": maximum,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
