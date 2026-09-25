#!/usr/bin/env python3
"""Show Task 4 training and asynchronous 20-seed evaluation progress."""
import csv
import json
import statistics
from pathlib import Path

runs = Path(__file__).resolve().parents[1] / "runs"
prefix = "final_rainbow_v5_r18_maskv5_s11_t4_from_t3c0300"
print("Rainbow Lite V5 / R18 / mask-v5 | Task 4 from Task 3 c0300 | cap 1500")
for cumulative in range(100, 1501, 100):
    name = f"{prefix}_c{cumulative:04d}"
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
    detail = ""
    if summary.exists():
        with summary.open() as handle:
            row = next((r for r in csv.DictReader(handle)
                        if r["run_id"] == "AVERAGE"
                        and r["agent_name"] == "rainbow_lite_v5_agent"), None)
        evaluation = (f"EVAL 20/20 | score {float(row['mean_score']):.2f} "
                      f"coins {float(row['mean_coins']):.2f} | "
                      f"kills {float(row['mean_kills']):.2f} | "
                      f"first {100*float(row['first_place_rate']):.0f}% | "
                      f"exclusive {100*float(row['exclusive_win_rate']):.0f}%") if row else "summary missing agent"
        if row:
            detail = (f"    median/min/max {float(row['median_score']):.1f}/"
                      f"{float(row['min_score']):.0f}/{float(row['max_score']):.0f} | "
                      f"survive {100*float(row['survival_rate']):.0f}% "
                      f"suicide {100*float(row['suicide_rate']):.0f}% "
                      f"opp-death {100*float(row['killed_by_opponent_rate']):.0f}% | "
                      f"loops W/P {100*float(row['long_wait_loop_rate']):.0f}/"
                      f"{100*float(row['long_ping_pong_loop_rate']):.0f}% | "
                      f"p95/max {1000*float(row['act_p95_time']):.1f}/"
                      f"{1000*float(row['act_max_time']):.1f} ms")
    elif (runs / eid).exists():
        episodes = []
        for path in sorted((runs / eid).glob("*_s[0-9]*/episodes.jsonl")):
            for line in path.read_text().splitlines():
                if line:
                    episodes.append(json.loads(line))
        targets = [next(a for a in e["agents"] if a["name"] == "rainbow_lite_v5_agent")
                   for e in episodes]
        if targets:
            n = len(targets)
            first = sum(a["score"] == max(x["score"] for x in e["agents"])
                        for e, a in zip(episodes, targets)) / n
            exclusive = sum(a["score"] == max(x["score"] for x in e["agents"])
                            and sum(x["score"] == a["score"] for x in e["agents"]) == 1
                            for e, a in zip(episodes, targets)) / n
            evaluation = (f"EVAL LIVE {n}/20 | score {statistics.fmean(a['score'] for a in targets):.2f} "
                          f"coins {statistics.fmean(a['coins'] for a in targets):.2f} | "
                          f"kills {statistics.fmean(a['kills'] for a in targets):.2f} | "
                          f"first {100*first:.0f}% | exclusive {100*exclusive:.0f}%")
            detail = (f"    survive {100*statistics.fmean(bool(a['survived']) for a in targets):.0f}% "
                      f"suicide {100*statistics.fmean(bool(a['suicides']) for a in targets):.0f}% "
                      f"opp-death {100*statistics.fmean(bool(a['killed_by_opponent']) for a in targets):.0f}% | "
                      f"loops W/P {100*statistics.fmean(bool(a['long_wait_loop']) for a in targets):.0f}/"
                      f"{100*statistics.fmean(bool(a['long_ping_pong_loop']) for a in targets):.0f}%")
        else:
            evaluation = "evaluation starting"
    else:
        evaluation = "evaluation pending"
    print(f"c{cumulative:04d} TRAIN {status} {count}/100 | {evaluation}")
    if detail:
        print(detail)
