"""Compare demonstration and online-only arms using preregistered ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rank(item):
    best = item["selection"]["best"]
    gates = sum(bool(value) for value in best["gates"].values())
    task2 = best["task2"]
    return (
        bool(best["passed"]), gates, float(best["balanced_capability"]),
        -(float(task2["long_wait_loop_rate"])
          + float(task2["long_ping_pong_loop_rate"])),
    )


def compare(demo: Path, control: Path, output: Path):
    arms = []
    for name, directory in (("demonstration", demo), ("online_control", control)):
        selection_path = Path(directory) / "best_task2_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        arms.append({"name": name, "run": str(directory), "selection": selection})
    ordered = sorted(arms, key=_rank, reverse=True)
    payload = {
        "schema_version": "q-learning-demonstration-comparison-v1",
        "ordering": [
            "passed", "task2_gate_count", "balanced_capability",
            "negative_loop_rate_sum",
        ],
        "winner": ordered[0]["name"],
        "arms": [{
            "name": item["name"], "run": item["run"],
            "best": item["selection"]["best"],
            "gate_count": sum(bool(value) for value in item["selection"]["best"]["gates"].values()),
        } for item in arms],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.demo, args.control, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
