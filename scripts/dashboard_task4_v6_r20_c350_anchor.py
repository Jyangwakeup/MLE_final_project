#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PREFIX = "rainbow_lite_v6_r20_maskv5_s11_task4_c350_anchor"
AGENT = "rainbow_lite_v6_agent"


def average_row(path: Path):
    if not path.exists():
        return None
    with path.open() as handle:
        return next((row for row in csv.DictReader(handle)
                     if row["run_id"] == "AVERAGE" and row["agent_name"] == AGENT), None)


def num(row, key, scale=1.0):
    if not row or row.get(key, "") == "":
        return "-"
    return f"{float(row[key]) * scale:.2f}"


def render():
    lines = ["V6/R20/mask-v5 | original c0350 anchored Task 4", ""]
    for cumulative in range(50, 401, 50):
        run = RUNS / f"{PREFIX}_c{cumulative:04d}"
        status = "waiting"
        try:
            status = json.loads((run / "metadata.json").read_text()).get("status", status)
        except (OSError, json.JSONDecodeError):
            pass
        episodes = []
        episode_path = run / "episodes.jsonl"
        if episode_path.exists():
            episodes = [json.loads(line) for line in episode_path.read_text().splitlines() if line]
        agents = [next(agent for agent in episode["agents"] if agent["name"] == AGENT)
                  for episode in episodes]
        count = len(agents)
        if count:
            first = sum(agent["score"] == max(item["score"] for item in episode["agents"])
                        for episode, agent in zip(episodes, agents)) / count
            exclusive = sum(
                sum(item["score"] == agent["score"] for item in episode["agents"]) == 1
                and agent["score"] == max(item["score"] for item in episode["agents"])
                for episode, agent in zip(episodes, agents)
            ) / count
            lines += [
                f"c{cumulative:04d} TRAIN {status.upper()} {count}/50 | "
                f"score {statistics.fmean(agent['score'] for agent in agents):.2f} "
                f"coins {statistics.fmean(agent['coins'] for agent in agents):.2f} "
                f"kills {statistics.fmean(agent['kills'] for agent in agents):.2f}",
                f"  first {100 * first:.1f}% exclusive {100 * exclusive:.1f}% | "
                f"survival {100 * statistics.fmean(bool(agent['survived']) for agent in agents):.1f}% "
                f"suicide {100 * statistics.fmean(agent['suicides'] for agent in agents):.1f}% | "
                f"wait-loop {100 * statistics.fmean(bool(agent.get('long_wait_loop')) for agent in agents):.1f}% "
                f"ping-pong {100 * statistics.fmean(bool(agent.get('long_ping_pong_loop')) for agent in agents):.1f}%",
            ]
        else:
            lines.append(f"c{cumulative:04d} TRAIN {status.upper()} 0/50")

        for task in (4, 3):
            evaluation_id = f"{PREFIX}_c{cumulative:04d}_task{task}_eval20"
            summary = RUNS / evaluation_id / f"{evaluation_id}_summary" / "summary.csv"
            row = average_row(summary)
            if row:
                lines.append(
                    f"  T{task} EVAL score {num(row, 'mean_score')} "
                    f"coins {num(row, 'mean_coins')} kills {num(row, 'mean_kills')} | "
                    f"first {num(row, 'first_place_rate', 100)}% "
                    f"exclusive {num(row, 'exclusive_win_rate', 100)}% | "
                    f"survive {num(row, 'survival_rate', 100)}% "
                    f"suicide {num(row, 'suicide_rate', 100)}% | "
                    f"loops W/P {num(row, 'long_wait_loop_rate', 100)}/"
                    f"{num(row, 'long_ping_pong_loop_rate', 100)}% | "
                    f"p95/max {num(row, 'act_p95_time', 1000)}/"
                    f"{num(row, 'act_max_time', 1000)} ms"
                )
            elif (RUNS / evaluation_id).exists():
                completed_seeds = len(list((RUNS / evaluation_id).glob("*_s[0-9]*")))
                lines.append(f"  T{task} EVAL RUNNING {completed_seeds}/20")
            else:
                lines.append(f"  T{task} EVAL waiting")
        lines.append("")
    lines.append("Refresh 3s | Ctrl+C exits dashboard only")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        while True:
            os.system("clear")
            print(render(), flush=True)
            time.sleep(3)
    except KeyboardInterrupt:
        pass
