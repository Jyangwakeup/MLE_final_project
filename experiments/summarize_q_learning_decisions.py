"""Summarize read-only Double Q(lambda) decision traces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    counts = Counter()
    tie_sizes: list[int] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                counts["decisions"] += 1
                counts[f"action_{row['action']}"] += 1
                counts["wait_in_danger"] += int(
                    row["action"] == "WAIT" and row["in_current_danger"])
                counts["wait_with_visible_coin"] += int(
                    row["action"] == "WAIT" and row["visible_coin_count"] > 0)
                counts["wait_with_reachable_coin"] += int(
                    row["action"] == "WAIT" and row["reachable_coin_exists"])
                counts["wait_with_reachable_crate"] += int(
                    row["action"] == "WAIT" and row["reachable_crate_frontier_exists"])
                counts["raw_argmax_vetoed"] += int(row["raw_argmax_vetoed"])
                counts["raw_bomb_vetoed"] += int(
                    5 in row["raw_argmax_ties"] and not row["legal_mask"][5])
                counts["physical_bomb"] += int(row["physical_mask"][5])
                counts["physical_useful_bomb"] += int(row["physical_useful_bomb"])
                counts["safe_bomb"] += int(row["legal_mask"][5])
                tie_sizes.append(len(row["legal_argmax_ties"]))
    decisions = counts["decisions"]
    waits = counts["action_WAIT"]
    return {
        "schema_version": "q-learning-decision-summary-v1",
        "inputs": [str(path) for path in paths],
        "decisions": decisions,
        "action_counts": {name: counts[f"action_{name}"] for name in
                          ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")},
        "wait_rate": None if not decisions else waits / decisions,
        "wait_reasons": {
            "in_current_danger": None if not waits else counts["wait_in_danger"] / waits,
            "with_visible_coin": None if not waits else counts["wait_with_visible_coin"] / waits,
            "with_reachable_coin": None if not waits else counts["wait_with_reachable_coin"] / waits,
            "with_reachable_crate": None if not waits else counts["wait_with_reachable_crate"] / waits,
        },
        "legal_argmax_tie_rate": None if not decisions else sum(size > 1 for size in tie_sizes) / decisions,
        "mean_legal_argmax_tie_size": None if not tie_sizes else sum(tie_sizes) / len(tie_sizes),
        "raw_argmax_veto_rate": None if not decisions else counts["raw_argmax_vetoed"] / decisions,
        "raw_bomb_veto_count": counts["raw_bomb_vetoed"],
        "physical_bomb_count": counts["physical_bomb"],
        "safe_bomb_count": counts["safe_bomb"],
        "physical_useful_bomb_count": counts["physical_useful_bomb"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.paths)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
