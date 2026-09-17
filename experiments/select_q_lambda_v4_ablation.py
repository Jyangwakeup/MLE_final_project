"""Select one safe 100k v4 candidate from completed snapshot selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _eligible(item: dict) -> bool:
    task2 = item["task2"]
    return (
        task2["suicide_rate"] <= 0.05
        and task2["survived_bomb_rate"] >= 0.95
        and item["task1_retention"] >= 0.90
    )


def _rank(item: dict) -> tuple:
    task2 = item["task2"]
    gate_count = sum(bool(value) for value in item["gates"].values())
    balanced = min(task2["mean_coins"] / 6.0, task2["mean_crates"] / 60.0)
    loops = task2["long_wait_loop_rate"] + task2["long_ping_pong_loop_rate"]
    return (-gate_count, -balanced, loops, item["checkpoint"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", action="append", nargs=2, required=True,
        metavar=("LABEL", "BEST_TASK2_SELECTION_JSON"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    candidates = []
    for label, source_raw in args.selection:
        source = Path(source_raw).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        for candidate in payload["candidates"]:
            candidates.append({**candidate, "label": label, "selection": str(source)})
    safe = sorted((item for item in candidates if _eligible(item)), key=_rank)
    payload = {
        "schema_version": "optimized-q-v4-ablation-selection-v1",
        "safety_floor": {
            "suicide_rate_max": 0.05,
            "survived_bomb_rate_min": 0.95,
            "task1_retention_min": 0.90,
        },
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe),
        "winner": safe[0] if safe else None,
        "safe_ranking": safe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
