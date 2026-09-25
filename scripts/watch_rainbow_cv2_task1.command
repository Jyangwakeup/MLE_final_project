#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

.venv/bin/python - <<'PY'
import csv
import glob
import json
import os
import statistics
import time
from pathlib import Path

PREFIX = "rainbow_cv2_r7_maskv5_s11_task1"
TOTAL_ROUNDS = 1000


def number(value, digits=3):
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def read_csv(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (FileNotFoundError, OSError, csv.Error):
        return []


def read_episodes(path):
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("agents"):
                    rows.append({**payload, **payload["agents"][0]})
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return rows


while True:
    run_dirs = sorted(Path("runs").glob(f"{PREFIX}_c[0-9][0-9][0-9][0-9]"))
    training = []
    episodes = []
    for run_dir in run_dirs:
        training.extend(read_csv(run_dir / "training.csv"))
        episodes.extend(read_episodes(run_dir / "episodes.jsonl"))

    rounds = len(training)
    recent = episodes[-20:]
    latest = training[-1] if training else {}
    active_chunk = run_dirs[-1].name if run_dirs else "等待启动"
    scores = [float(row.get("score", 0)) for row in recent]
    coins = [float(row.get("coins", 0)) for row in recent]
    steps = [float(row.get("round_steps", 0)) for row in recent]

    print("\033[2J\033[H", end="")
    print("Rainbow Lite / continuous-v2 / R7 / survival-mask-v5 — Task 1")
    print("=" * 74)
    print(f"总进度       {rounds:4d}/{TOTAL_ROUNDS}  ({rounds / TOTAL_ROUNDS:6.1%})")
    print(f"当前训练块   {active_chunk}")
    print(f"最近局数     {len(recent)}")
    if recent:
        print(
            f"最近20局     平均得分 {statistics.fmean(scores):6.2f} | "
            f"平均金币 {statistics.fmean(coins):6.2f} | "
            f"平均步数 {statistics.fmean(steps):7.1f}"
        )
        print(
            f"             全金币率 {sum(bool(x.get('all_coins')) for x in recent)/len(recent):6.1%} | "
            f"存活率 {sum(bool(x.get('survived')) for x in recent)/len(recent):6.1%} | "
            f"到达400步 {sum(bool(x.get('max_steps')) for x in recent)/len(recent):6.1%}"
        )
        print(
            f"             WAIT循环 {sum(bool(x.get('long_wait_loop')) for x in recent):2d} | "
            f"往返循环 {sum(bool(x.get('long_ping_pong_loop')) for x in recent):2d} | "
            f"非法动作 {sum(int(x.get('invalid', 0)) for x in recent):3d}"
        )

    print("-" * 74)
    print(
        f"最新训练     reward {number(latest.get('reward')):>9} | "
        f"epsilon {number(latest.get('epsilon'), 4):>7} | "
        f"loss {number(latest.get('loss'), 6):>10}"
    )
    print(
        f"             action_steps {latest.get('action_steps', '-'):>8} | "
        f"updates {latest.get('updates', '-'):>7} | "
        f"replay {latest.get('replay_size', '-'):>7}"
    )
    print(
        f"安全统计     decisions {latest.get('safety_decisions', '-'):>8} | "
        f"interventions {latest.get('safety_interventions', '-'):>7} | "
        f"fallbacks {latest.get('safety_fallbacks', '-'):>5}"
    )

    summaries = sorted(Path("runs").glob(
        f"{PREFIX}_c*_eval5/{PREFIX}_c*_eval5_summary/summary.csv"))
    print("-" * 74)
    print(f"冻结评估     {len(summaries)}/10 已完成")
    for summary in summaries[-10:]:
        rows = read_csv(summary)
        if not rows:
            continue
        row = rows[-1]
        checkpoint = summary.parts[-3].split("_eval5", 1)[0].rsplit("_c", 1)[-1]
        print(
            f"  c{checkpoint}: score {number(row.get('mean_score'), 2):>6} | "
            f"coins {number(row.get('mean_coins'), 2):>6} | "
            f"all-coins {number(100 * float(row.get('all_coins_rate') or 0), 1):>5}% | "
            f"steps/coin {number(row.get('steps_per_coin'), 2):>6} | "
            f"WAIT {number(100 * float(row.get('wait_action_rate_all_actions') or 0), 1):>5}% | "
            f"p95 {number(1000 * float(row.get('act_p95_time') or 0), 2):>6} ms"
        )
    print("=" * 74)
    print(f"更新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}  Ctrl+C 仅退出监控")
    time.sleep(5)
PY
