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
PREFIX = "rainbow_lite_v5_r18_maskv5_s11_task4_c350_anchor"
AGENT = "rainbow_lite_v5_agent"


def completed(run: Path) -> bool:
    try:
        return json.loads((run / "metadata.json").read_text()).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def average_row(path: Path):
    if not path.exists():
        return None
    return next((r for r in csv.DictReader(path.open())
                 if r["run_id"] == "AVERAGE" and r["agent_name"] == AGENT), None)


def num(row, key, scale=1.0):
    if not row or row.get(key, "") == "":
        return "-"
    return f"{float(row[key]) * scale:.2f}"


def render():
    lines = ["V5/R18/mask-v5 | c0350 anchored Task 4", ""]
    for cumulative in (50, 100):
        run = RUNS / f"{PREFIX}_c{cumulative:04d}"
        status = "waiting"
        try:
            status = json.loads((run / "metadata.json").read_text()).get("status", status)
        except (OSError, json.JSONDecodeError):
            pass
        episodes = []
        ep = run / "episodes.jsonl"
        if ep.exists():
            episodes = [json.loads(x) for x in ep.read_text().splitlines() if x]
        agents = [next(a for a in e["agents"] if a["name"] == AGENT) for e in episodes]
        n = len(agents)
        if n:
            first = sum(a["score"] == max(x["score"] for x in e["agents"])
                        for e, a in zip(episodes, agents)) / n
            exclusive = sum(sum(x["score"] == a["score"] for x in e["agents"]) == 1
                            and a["score"] == max(x["score"] for x in e["agents"])
                            for e, a in zip(episodes, agents)) / n
            lines += [
                f"c{cumulative:04d} TRAIN {status.upper()} {n}/50 | score {statistics.fmean(a['score'] for a in agents):.2f} "
                f"coins {statistics.fmean(a['coins'] for a in agents):.2f} kills {statistics.fmean(a['kills'] for a in agents):.2f}",
                f"  first {100*first:.1f}% exclusive {100*exclusive:.1f}% | survival {100*statistics.fmean(bool(a['survived']) for a in agents):.1f}% "
                f"suicide {100*statistics.fmean(a['suicides'] for a in agents):.1f}% | wait-loop {100*statistics.fmean(bool(a.get('long_wait_loop')) for a in agents):.1f}% "
                f"ping-pong {100*statistics.fmean(bool(a.get('long_ping_pong_loop')) for a in agents):.1f}%",
            ]
        else:
            lines.append(f"c{cumulative:04d} TRAIN {status.upper()} 0/50")

        for task in (4, 3):
            eid = f"{PREFIX}_c{cumulative:04d}_task{task}_eval20"
            summary = RUNS / eid / f"{eid}_summary" / "summary.csv"
            row = average_row(summary)
            if row:
                lines.append(
                    f"  T{task} EVAL score {num(row,'mean_score')} coins {num(row,'mean_coins')} kills {num(row,'mean_kills')} | "
                    f"first {num(row,'first_place_rate',100)}% exclusive {num(row,'exclusive_win_rate',100)}% | "
                    f"survive {num(row,'survival_rate',100)}% suicide {num(row,'suicide_rate',100)}% | "
                    f"loops W/P {num(row,'long_wait_loop_rate',100)}/{num(row,'long_ping_pong_loop_rate',100)}% | "
                    f"p95/max {num(row,'act_p95_time',1000)}/{num(row,'act_max_time',1000)} ms")
            elif (RUNS / eid).exists():
                done = len(list((RUNS / eid).glob("*_s[0-9]*")))
                lines.append(f"  T{task} EVAL RUNNING {done}/20")
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
