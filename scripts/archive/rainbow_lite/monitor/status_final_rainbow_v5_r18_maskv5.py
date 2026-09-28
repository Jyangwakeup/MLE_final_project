#!/usr/bin/env python3
"""Show training and evaluation progress for the final V5 chain."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
runs = root / "runs"
prefix = "final_rainbow_v5_r18_maskv5_s11"
totals = {1: 500, 2: 1000, 3: 1500, 4: 3000}

for task, total in totals.items():
    print(f"Task {task} ({total} training rounds):")
    for cumulative in range(100, total + 1, 100):
        if task == 3:
            name = f"{prefix}_t3_from_t2c0800_c{cumulative:04d}"
        elif task == 4:
            continue  # Task 4 starts only after selecting the Task 3 checkpoint.
        else:
            name = f"{prefix}_t{task}_c{cumulative:04d}"
        train = runs / name
        if not train.exists():
            continue
        try:
            status = json.loads((train / "metadata.json").read_text())["status"]
        except (OSError, KeyError, json.JSONDecodeError):
            status = "running"
        episodes = train / "episodes.jsonl"
        count = sum(1 for _ in episodes.open()) if episodes.exists() else 0
        eid = f"{name}_eval20"
        summary = runs / eid / f"{eid}_summary" / "summary.csv"
        if summary.exists():
            with summary.open() as handle:
                row = next((r for r in csv.DictReader(handle)
                            if r["run_id"] == "AVERAGE"
                            and r["agent_name"] == "rainbow_lite_v5_agent"), None)
            evaluation = (f"score={float(row['mean_score']):.2f} "
                          f"coins={float(row['mean_coins']):.2f} "
                          f"kills={float(row['mean_kills']):.2f}") if row else "summary missing agent"
        elif (runs / eid).exists():
            evaluation = f"evaluating ({len(list((runs / eid).glob('*_s[0-9]*')))}/20 seeds started)"
        else:
            evaluation = "evaluation pending"
        print(f"  c{cumulative:04d}: train {status} {count}/100 | {evaluation}")
