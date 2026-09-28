#!/bin/zsh

cd "${0:A:h}/../../../.." || exit 1

while true; do
  clear
  .venv/bin/python - <<'PY'
import csv
import datetime as dt
from pathlib import Path
import subprocess

RUNS = (
    ("Rainbow · Safety", "rainbow_lite_v4_r10_s11_task2_safety_resume_v2"),
    ("Rainbow · No Safety", "rainbow_lite_v4_r13_s11_task2_no_safety_v2"),
)
MIN_ROUNDS = 500
MAX_ROUNDS = 2000
TARGET_STEPS = 150_000
WIDTH = 40

def alive(run_id):
    result = subprocess.run(
        ["pgrep", "-f", f"experiments/run.py.*--run-id {run_id}"],
        capture_output=True, text=True,
    )
    return result.returncode == 0

def bar(fraction):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * WIDTH)
    return "█" * filled + "░" * (WIDTH - filled)

print("Rainbow Task 2 · Live Training")
print(dt.datetime.now().strftime("Updated %Y-%m-%d %H:%M:%S"))
print("Stop condition: ≥500 rounds AND ≥150,000 stage action steps (max 2,000 rounds)\n")

for label, run_id in RUNS:
    path = Path("runs") / run_id / "training.csv"
    rounds = steps = 0
    reward = loss = epsilon = None
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            last = rows[-1]
            rounds = int(last["round"])
            steps = int(last["stage_action_steps"])
            reward = float(last["reward"])
            epsilon = float(last["epsilon"])
            loss = float(last["loss"]) if last.get("loss") else None
    round_fraction = rounds / MIN_ROUNDS
    step_fraction = steps / TARGET_STEPS
    completion = min(round_fraction, step_fraction)
    state = "RUNNING" if alive(run_id) else ("DONE" if completion >= 1 else "STOPPED")
    print(f"{label:<27} {state}")
    print(f"[{bar(completion)}] {completion * 100:6.2f}%")
    print(f"rounds {rounds:4d}/{MIN_ROUNDS}+ (cap {MAX_ROUNDS})  actions {steps:7d}/{TARGET_STEPS}")
    if reward is not None:
        loss_text = "warming up" if loss is None else f"{loss:.4f}"
        print(f"reward {reward:8.2f}  epsilon {epsilon:.4f}  loss {loss_text}")
    print()

print("Press Ctrl-C to close this monitor. Training continues in the background.")
PY
  sleep 2
done
