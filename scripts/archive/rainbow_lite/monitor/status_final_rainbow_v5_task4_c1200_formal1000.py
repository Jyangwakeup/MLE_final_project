#!/usr/bin/env python3
"""Live progress and metrics for the frozen c1200 1000-round evaluation."""
import json
import statistics
from pathlib import Path

runs = Path(__file__).resolve().parents[4] / "runs"
eid = "final_rainbow_v5_r18_maskv5_s11_t4_c1200_formal1000"
root = runs / eid
episodes = []
for path in sorted(root.glob("*_s[0-9]*/episodes.jsonl")):
    for line in path.read_text().splitlines():
        if line:
            episodes.append(json.loads(line))

print("Rainbow Lite V5 c1200 | Task 4 frozen CPU evaluation | 50 seeds x 20 = 1000")
if not episodes:
    print("Evaluation starting: 0/1000")
    raise SystemExit

targets = [next(a for a in e["agents"] if a["name"] == "rainbow_lite_v5_agent")
           for e in episodes]
n = len(targets)
first = sum(a["score"] == max(x["score"] for x in e["agents"])
            for e, a in zip(episodes, targets)) / n
exclusive = sum(a["score"] == max(x["score"] for x in e["agents"])
                and sum(x["score"] == a["score"] for x in e["agents"]) == 1
                for e, a in zip(episodes, targets)) / n
scores = [a["score"] for a in targets]
print(f"Progress {n}/1000 ({100*n/1000:.1f}%) | seeds started "
      f"{len(list(root.glob('*_s[0-9]*')))}/50")
print(f"score mean/median/min/max {statistics.fmean(scores):.3f}/"
      f"{statistics.median(scores):.1f}/{min(scores)}/{max(scores)} | "
      f"coins {statistics.fmean(a['coins'] for a in targets):.3f} | "
      f"kills {statistics.fmean(a['kills'] for a in targets):.3f}")
print(f"first {100*first:.1f}% | exclusive {100*exclusive:.1f}% | "
      f"survive {100*statistics.fmean(bool(a['survived']) for a in targets):.1f}% | "
      f"suicide {100*statistics.fmean(bool(a['suicides']) for a in targets):.1f}% | "
      f"opponent-death {100*statistics.fmean(bool(a['killed_by_opponent']) for a in targets):.1f}%")
print(f"long WAIT/ping-pong "
      f"{100*statistics.fmean(bool(a['long_wait_loop']) for a in targets):.1f}%/"
      f"{100*statistics.fmean(bool(a['long_ping_pong_loop']) for a in targets):.1f}% | "
      f"max-steps {100*statistics.fmean(bool(a['max_steps']) for a in targets):.1f}% | "
      f"invalid/round {statistics.fmean(a['invalid'] for a in targets):.3f}")
