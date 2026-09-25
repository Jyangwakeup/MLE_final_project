#!/usr/bin/env python3
"""Live terminal dashboard for the V5/R18/mask-v5 Task 3 experiment."""

from __future__ import annotations

import csv
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PREFIX = "rainbow_lite_v5_r18_maskv5_s11_task3_from_full500"
AGENT = "rainbow_lite_v5_agent"
TOTAL = 1500
BASE = 0


def completed_metadata(path: Path) -> bool:
    try:
        return json.loads(path.read_text())["status"] == "completed"
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def latest_training():
    completed = BASE
    active = None
    local = 0
    for cumulative in range(BASE + 50, TOTAL + 1, 50):
        run = RUNS / f"{PREFIX}_c{cumulative:04d}"
        if completed_metadata(run / "metadata.json"):
            completed = cumulative
        elif run.exists():
            active = run
            training = run / "training.csv"
            if training.exists():
                local = max(0, sum(1 for _ in training.open()) - 1)
            break
        else:
            break
    current = min(TOTAL, completed + local)
    run = active or (RUNS / f"{PREFIX}_c{completed:04d}" if completed else None)
    return current, completed, run


def recent_training(run: Path | None):
    if run is None or not (run / "episodes.jsonl").exists():
        return None
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()[-50:]]
    agents = [next(a for a in e["agents"] if a["name"] == AGENT) for e in episodes]
    if not agents:
        return None
    n = len(agents)
    steps = sum(
        int(a.get("death_step") or e.get("round_steps", 0))
        for e, a in zip(episodes, agents)
    )
    bombs = sum(int(a.get("bombs_resolved", a.get("bombs", 0))) for a in agents)
    bombs_survived = sum(int(a.get("bombs_survived", 0)) for a in agents)
    zero_utility = sum(int(a.get("zero_utility_bombs", 0)) for a in agents)
    return {
        "window": n,
        "score": sum(a["score"] for a in agents) / n,
        "coins": sum(a["coins"] for a in agents) / n,
        "crates": sum(a["crates"] for a in agents) / n,
        "kills": sum(a["kills"] for a in agents) / n,
        "suicide": sum(a["suicides"] for a in agents) / n,
        "survival": sum(bool(a["survived"]) for a in agents) / n,
        "opponent_death": sum(bool(a.get("killed_by_opponent")) for a in agents) / n,
        "bomb_survival": bombs_survived / bombs if bombs else 1.0,
        "zero_utility": zero_utility / bombs if bombs else 0.0,
        "crates_per_bomb": sum(a["crates"] for a in agents) / bombs if bombs else 0.0,
        "invalid_rate": sum(int(a.get("invalid", 0)) for a in agents) / steps if steps else 0.0,
        "long_wait": sum(bool(a.get("long_wait_loop")) for a in agents) / n,
        "ping_pong": sum(bool(a.get("long_ping_pong_loop")) for a in agents) / n,
    }


def live_health(run: Path | None):
    if run is None:
        return None
    metadata_path = run / "metadata.json"
    timing_path = run / "timing.jsonl"
    training_path = run / "training.csv"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        metadata = {}
    rows = []
    if timing_path.exists():
        for line in timing_path.read_text(errors="replace").splitlines()[-8000:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("agent_name") == AGENT:
                rows.append(item)
    cache_rows = [r for r in rows if "training_decision_cache_hit" in (r.get("safety") or {})]
    times = sorted(float(r.get("think_time", 0.0)) * 1000 for r in rows)
    started = metadata.get("started_at")
    rounds = max(0, sum(1 for _ in training_path.open()) - 1) if training_path.exists() else 0
    rounds_per_minute = None
    if started and rounds:
        try:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
            rounds_per_minute = 60 * rounds / elapsed if elapsed > 0 else None
        except ValueError:
            pass
    percentile_index = max(0, int(0.95 * len(times)) - 1)
    freshness = time.time() - max(
        (p.stat().st_mtime for p in (timing_path, training_path) if p.exists()),
        default=0,
    )
    return {
        "status": metadata.get("status", "starting"),
        "freshness": freshness,
        "rpm": rounds_per_minute,
        "act_mean": statistics.fmean(times) if times else None,
        "act_p95": times[percentile_index] if times else None,
        "act_max": max(times) if times else None,
        "timeouts": sum(bool(r.get("timed_out")) for r in rows),
        "cache_rate": (
            sum(bool(r["safety"]["training_decision_cache_hit"]) for r in cache_rows)
            / len(cache_rows) if cache_rows else None
        ),
        "search_timeouts": sum(
            bool((r.get("safety") or {}).get("robust_search_timed_out")) for r in rows
        ),
        "wait_rate": sum(r.get("action") == "WAIT" for r in rows) / len(rows) if rows else None,
        "intervention_rate": sum(
            bool((r.get("safety") or {}).get("intervened")) for r in rows
        ) / len(rows) if rows else None,
        "robust_intervention_rate": sum(
            bool((r.get("safety") or {}).get("robust_intervened")) for r in rows
        ) / len(rows) if rows else None,
        "robust_losses": sum(
            bool((r.get("safety") or {}).get("robust_guarantee_loss")) for r in rows
        ),
        "v1_fallbacks": sum(
            bool((r.get("safety") or {}).get("robust_to_v1_fallback")) for r in rows
        ),
        "opponent_fallbacks": sum(
            bool((r.get("safety") or {}).get("opponent_to_v3_fallback")) for r in rows
        ),
    }


def evaluations():
    records = []
    active = 0
    failed = 0
    for cumulative in range(BASE + 50, TOTAL + 1, 50):
        eid = f"{PREFIX}_c{cumulative:04d}_eval10"
        summary = RUNS / eid / f"{eid}_summary" / "summary.csv"
        log = RUNS / f"{eid}.console.log"
        if summary.exists():
            rows = list(csv.DictReader(summary.open()))
            row = next((r for r in rows if r["run_id"] == "AVERAGE" and r["agent_name"] == AGENT), None)
            if row:
                records.append((cumulative, row))
        elif log.exists():
            text = log.read_text(errors="replace")
            if any(token in text for token in ("Traceback", "error:", "Exception")):
                failed += 1
            else:
                active += 1
    return records, active, failed


def number(row, key, scale=1.0):
    value = row.get(key, "")
    return "-" if value == "" else f"{float(value) * scale:.2f}"


def render():
    current, completed, run = latest_training()
    recent = recent_training(run)
    health = live_health(run)
    evals, active, failed = evaluations()
    width = 30
    filled = int(width * current / TOTAL)
    lines = [
        "V5 + R18 + survival-mask-v5 | Task 3 control panel",
        f"Training  [{'#' * filled}{'-' * (width-filled)}] {current:4d}/{TOTAL} ({100*current/TOTAL:5.1f}%)",
        f"Checkpoint c{completed:04d} | evaluations {len(evals):2d}/30 | queued/running {active:2d} | failed {failed}",
    ]
    if health:
        live = "LIVE" if health["status"] == "running" and health["freshness"] < 20 else health["status"].upper()
        rpm = "-" if health["rpm"] is None else f"{health['rpm']:.2f} rounds/min"
        cache = "-" if health["cache_rate"] is None else f"{100*health['cache_rate']:.1f}%"
        timing = "-" if health["act_mean"] is None else (
            f"{health['act_mean']:.2f}/{health['act_p95']:.2f}/{health['act_max']:.2f} ms")
        wait = "-" if health["wait_rate"] is None else f"{100*health['wait_rate']:.1f}%"
        intervene = "-" if health["intervention_rate"] is None else f"{100*health['intervention_rate']:.1f}%"
        robust_intervene = "-" if health["robust_intervention_rate"] is None else f"{100*health['robust_intervention_rate']:.1f}%"
        lines += [
            f"Status {live} | speed {rpm} | next frozen eval c{min(TOTAL, completed + 50):04d}",
            f"Act mean/p95/max {timing} | framework timeouts {health['timeouts']} | safety-search timeouts {health['search_timeouts']} | cache hits {cache}",
            f"Actions WAIT {wait} | mask intervention {intervene} | robust intervention {robust_intervene}",
            f"Safety robust loss {health['robust_losses']} | robust->v1 fallback {health['v1_fallbacks']} | opponent->v3 fallback {health['opponent_fallbacks']}",
        ]
    if recent:
        lines += [
            "",
            f"Recent train ({recent['window']:2d} rounds): score {recent['score']:.2f} | coins {recent['coins']:.2f} | crates {recent['crates']:.2f} | kills {recent['kills']:.2f}",
            f"Outcome: suicide {100*recent['suicide']:.1f}% | opponent death {100*recent['opponent_death']:.1f}% | survival {100*recent['survival']:.1f}%",
            f"Bombs: survival {100*recent['bomb_survival']:.1f}% | zero utility {100*recent['zero_utility']:.1f}% | crates/bomb {recent['crates_per_bomb']:.2f}",
            f"Behaviour: invalid {100*recent['invalid_rate']:.2f}% | long WAIT loop {100*recent['long_wait']:.1f}% | ping-pong loop {100*recent['ping_pong']:.1f}%",
        ]
    lines += ["", "Frozen 10-seed checkpoints:", " round | score coins crates kills | suicide survive | robust loss timeout | wait loop | p95/max ms"]
    for cumulative, row in evals[-12:]:
        lines.append(
            f" {cumulative:4d} | {number(row,'mean_score'):>5} {number(row,'mean_coins'):>5} {number(row,'mean_crates'):>6} {number(row,'mean_kills'):>5} |"
            f" {number(row,'suicide_rate',100):>6}% {number(row,'survival_rate',100):>6}% |"
            f" {number(row,'robust_guarantee_losses'):>11} {number(row,'robust_search_timeouts'):>7} |"
            f" {number(row,'wait_action_rate',100):>5}% {number(row,'long_wait_loop_rate',100):>5}% |"
            f" {number(row,'act_p95_time',1000):>7}/{number(row,'act_max_time',1000):>6}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--agent", default=AGENT)
    parser.add_argument("--total", type=int, default=TOTAL)
    parser.add_argument("--base", type=int, default=BASE)
    args = parser.parse_args()
    PREFIX, AGENT, TOTAL, BASE = args.prefix, args.agent, args.total, args.base
    try:
        while True:
            os.system("clear")
            print(render(), flush=True)
            time.sleep(3)
    except KeyboardInterrupt:
        pass
