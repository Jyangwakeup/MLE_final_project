"""Summarize opt-in CNN Q/mask decision diagnostics without replay heuristics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(paths):
    counts = Counter()
    q_margins = defaultdict(list)
    per_round = defaultdict(Counter)
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                counts["decisions"] += 1
                action = row["action"]
                counts[f"action_{action}"] += 1
                round_counts = per_round[(str(path), int(row["round"]))]
                round_counts["steps"] += 1
                round_counts[f"action_{action}"] += 1
                if action == "WAIT" and row.get("avoidable"):
                    counts["avoidable_wait_selected"] += 1
                    round_counts["avoidable_wait_selected"] += 1
                if row.get("safe_progress_move"):
                    counts["states_with_safe_progress_move"] += 1
                if row.get("safe_useful_bomb"):
                    counts["states_with_safe_useful_bomb"] += 1
                if row.get("legal_mask", [False] * 6)[5]:
                    counts["states_with_safe_bomb"] += 1
                physical = row.get("physical_mask", [False] * 6)
                legal = row.get("legal_mask", [False] * 6)
                if physical[5]:
                    counts["physical_bomb_available"] += 1
                    if legal[5]:
                        counts["safe_bomb_available"] += 1
                    else:
                        counts["bomb_veto"] += 1
                q_values = [float(value) for value in row["q_values"]]
                physical_indices = [
                    index for index, allowed in enumerate(physical) if allowed]
                raw_index = max(
                    physical_indices, key=lambda index: (q_values[index], -index))
                raw_action = (
                    "UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")[raw_index]
                counts[f"raw_argmax_{raw_action}"] += 1
                if not legal[raw_index]:
                    counts["raw_argmax_veto"] += 1
                    counts[f"raw_argmax_veto_{raw_action}"] += 1
                if row.get("safety_fallback"):
                    counts["safety_fallback"] += 1
                margin = row.get("wait_q_margin")
                if margin is not None:
                    q_margins["wait"].append(float(margin))
    decisions = counts["decisions"]
    wait = counts["action_WAIT"]
    physical_bomb = counts["physical_bomb_available"]
    raw_bomb = counts["raw_argmax_BOMB"]
    margins = sorted(q_margins["wait"])
    quantile = lambda fraction: (
        None if not margins else margins[round((len(margins) - 1) * fraction)])
    stuck_rounds = sum(
        value["steps"] > 0 and value["action_WAIT"] / value["steps"] >= 0.95
        for value in per_round.values())
    return {
        "inputs": [str(Path(path)) for path in paths],
        "decisions": decisions,
        "rounds": len(per_round),
        "action_counts": {action: counts[f"action_{action}"] for action in (
            "UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")},
        "wait_rate": wait / decisions if decisions else None,
        "avoidable_wait_selected_rate": (
            counts["avoidable_wait_selected"] / wait if wait else None),
        "states_with_safe_progress_move_rate": (
            counts["states_with_safe_progress_move"] / decisions
            if decisions else None),
        "states_with_safe_useful_bomb_rate": (
            counts["states_with_safe_useful_bomb"] / decisions
            if decisions else None),
        "states_with_safe_bomb_rate": (
            counts["states_with_safe_bomb"] / decisions if decisions else None),
        "physical_bomb_available_count": physical_bomb,
        "safe_bomb_available_count": counts["safe_bomb_available"],
        "bomb_veto_count": counts["bomb_veto"],
        "bomb_veto_rate": (
            counts["bomb_veto"] / physical_bomb if physical_bomb else None),
        "raw_argmax_veto_count": counts["raw_argmax_veto"],
        "raw_argmax_veto_rate": (
            counts["raw_argmax_veto"] / decisions if decisions else None),
        "raw_argmax_veto_by_action": {
            action: counts[f"raw_argmax_veto_{action}"] for action in (
                "UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")},
        "raw_bomb_argmax_count": raw_bomb,
        "raw_bomb_argmax_veto_count": counts["raw_argmax_veto_BOMB"],
        "raw_bomb_argmax_veto_rate": (
            counts["raw_argmax_veto_BOMB"] / raw_bomb if raw_bomb else None),
        "safety_fallback_count": counts["safety_fallback"],
        "wait_q_margin": {
            "count": len(margins), "p10": quantile(0.10),
            "median": quantile(0.50), "p90": quantile(0.90),
        },
        "rounds_with_wait_rate_at_least_95pct": stuck_rounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.paths)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
