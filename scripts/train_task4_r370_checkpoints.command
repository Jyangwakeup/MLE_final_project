#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v5_r18_task2.json"
agent="rainbow_lite_v5_agent"
seed=11
rounds_per_chunk=100
chunk_count=30
initial_checkpoint="runs/rainbow_lite_v5_r18_s11_task3_diag10_snapshot/checkpoint.pt"
prefix="rainbow_lite_v5_r18_s11_task4_r370"
evaluation_seeds=(12020 12021 12022 12023 12024 12025 12026 12027 12028 12029)

if [[ ! -x "$python_bin" ]]; then
  print -u2 "Missing Python environment: $python_bin"
  exit 1
fi
if [[ ! -f "$initial_checkpoint" ]]; then
  print -u2 "Missing initial checkpoint: $initial_checkpoint"
  exit 1
fi

run_completed() {
  local metadata="$1/metadata.json"
  [[ -f "$metadata" ]] || return 1
  "$python_bin" - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if metadata.get("status") == "completed" else 1)
PY
}

for chunk in $(seq 1 "$chunk_count"); do
  cumulative=$((chunk * rounds_per_chunk))
  suffix=$(printf "%04d" "$cumulative")
  train_id="${prefix}_c${suffix}"
  train_dir="runs/${train_id}"

  if run_completed "$train_dir"; then
    print "[Task 4] training chunk ${cumulative} already complete"
  else
    if [[ -e "$train_dir" ]]; then
      print -u2 "Incomplete run directory exists: $train_dir"
      print -u2 "Resume or move it explicitly before restarting this script."
      exit 1
    fi

    if (( chunk == 1 )); then
      source_args=(--init-from-checkpoint "$initial_checkpoint")
    else
      previous=$((cumulative - rounds_per_chunk))
      previous_suffix=$(printf "%04d" "$previous")
      parent_run="runs/${prefix}_c${previous_suffix}"
      if (( chunk == 3 )) && [[ -d "${parent_run}_evalmetrics_migrated" ]]; then
        parent_run="${parent_run}_evalmetrics_migrated"
      fi
      source_args=(--resume-from "$parent_run")
    fi

    print "[Task 4] training rounds $((cumulative - 99))-${cumulative}"
    "$python_bin" -m experiments.run \
      --config "$config" \
      --mode train \
      --device cpu \
      --task 4 \
      --agent "$agent" \
      --seed "$seed" \
      --n-rounds "$rounds_per_chunk" \
      --target-stage-action-steps 999999999 \
      --min-rounds 100 \
      "${source_args[@]}" \
      --run-id "$train_id"
  fi

  checkpoint="${train_dir}/checkpoints/final.pt"
  if [[ ! -f "$checkpoint" ]]; then
    print -u2 "Completed chunk has no checkpoint: $checkpoint"
    exit 1
  fi

  eval_id="${train_id}_diag10"
  eval_summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
  if [[ -f "$eval_summary" ]]; then
    print "[Task 4] evaluation ${eval_id} already complete"
  else
    if [[ -e "runs/${eval_id}" ]]; then
      print -u2 "Incomplete evaluation directory exists: runs/${eval_id}"
      exit 1
    fi
    print "[Task 4] evaluating checkpoint after ${cumulative} rounds"
    "$python_bin" -m experiments.run \
      --config "$config" \
      --mode evaluate \
      --device cpu \
      --task 4 \
      --agent "$agent" \
      --checkpoint "$checkpoint" \
      --seeds "${evaluation_seeds[@]}" \
      --n-rounds 1 \
      --replay-policy failures \
      --run-id "$eval_id"
  fi
done

print "Task 4 checkpoint training and evaluation completed through 3000 rounds."
