#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-t2-maskpair
#SBATCH --output=%x_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"

all_job_id="${ALL_DIAGNOSTIC_JOB_ID:?Set ALL_DIAGNOSTIC_JOB_ID}"
all_run="runs/cnn_t2_d01_maskdiag_j${all_job_id}"
summary="$all_run/decision_summary.json"
pair_dir="runs/cnn_t2_d01_maskpair_j${SLURM_JOB_ID}"
mkdir -p "$pair_dir"

if ! .venv/bin/python - "$summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
bomb = summary.get("bomb_veto_rate")
raw_bomb = summary.get("raw_bomb_argmax_veto_rate")
trigger = any(value is not None and float(value) > 0.05
              for value in (bomb, raw_bomb))
raise SystemExit(0 if trigger else 1)
PY
then
  .venv/bin/python - "$summary" "$pair_dir/status.json" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
summary = json.load(open(source, encoding="utf-8"))
result = {
    "status": "skipped",
    "reason": "both BOMB veto rates are at most 5%",
    "all_decision_summary": source,
    "bomb_veto_rate": summary.get("bomb_veto_rate"),
    "raw_bomb_argmax_veto_rate": summary.get("raw_bomb_argmax_veto_rate"),
}
open(destination, "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
PY
  exit 0
fi

run_id="cnn_t2_d01_maskoff_j${SLURM_JOB_ID}"
.venv/bin/python -u -m experiments.run \
  --config experiments/configs/task2_cnn_decision_diagnostics_off.json \
  --mode evaluate --task 2 --agent cnn_distilled_double_dqn_agent \
  --reward-id r7_safe_credit_sparse \
  --checkpoint runs/cnn_distilled_t2_j471458_transfer_s11/checkpoints/snapshots/policy_200000.pt \
  --device cpu --n-rounds 1 \
  --seeds 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 \
          10010 10011 10012 10013 10014 10015 10016 10017 10018 10019 \
  --run-id "$run_id" --replay-policy none

mapfile -t diagnostics < <(find "runs/$run_id" -name cnn_decisions.jsonl -type f | sort)
.venv/bin/python -m experiments.summarize_cnn_decisions \
  "${diagnostics[@]}" --output "runs/$run_id/decision_summary.json"
all_metrics="$(find "$all_run" -path '*_summary/summary.csv' -type f -print -quit)"
off_metrics="$(find "runs/$run_id" -path '*_summary/summary.csv' -type f -print -quit)"
.venv/bin/python - \
  "$summary" "runs/$run_id/decision_summary.json" \
  "$all_metrics" "$off_metrics" "$pair_dir/status.json" <<'PY'
import csv
import json
import sys

all_path, off_path, all_metrics_path, off_metrics_path, destination = sys.argv[1:]

def average(path):
    with open(path, newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["run_id"] == "AVERAGE"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one AVERAGE row in {path}")
    return rows[0]

all_metrics = average(all_metrics_path)
off_metrics = average(off_metrics_path)
all_crates = float(all_metrics["mean_crates"])
off_crates = float(off_metrics["mean_crates"])
crate_gain = ((off_crates / all_crates) - 1.0) if all_crates else None
off_suicide = float(off_metrics["suicide_rate"])
result = {
    "status": "completed",
    "all_decision_summary": all_path,
    "off_decision_summary": off_path,
    "all": json.load(open(all_path, encoding="utf-8")),
    "off": json.load(open(off_path, encoding="utf-8")),
    "all_metrics": all_metrics,
    "off_metrics": off_metrics,
    "off_crate_gain_fraction": crate_gain,
    "off_meets_followup_threshold": (
        crate_gain is not None and crate_gain >= 0.10 and off_suicide <= 0.05),
}
open(destination, "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
PY
