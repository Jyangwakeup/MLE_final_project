#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v6_r19_task4.json"
agent="rainbow_lite_v6_agent"
seed=11
rounds_per_chunk=100
chunk_count=30
task3_checkpoint="runs/rainbow_lite_v5_r18_s11_task3_diag10_snapshot/checkpoint.pt"
initial_checkpoint="runs/rainbow_lite_v6_r19_task4_r370_warm_init.pt"
prefix="rainbow_lite_v6_r19_s11_task4_r370"
evaluation_seeds=(12120 12121 12122 12123 12124 12125 12126 12127 12128 12129)

if [[ ! -x "$python_bin" || ! -f "$task3_checkpoint" ]]; then
  print -u2 "Missing Python environment or Task 3 r370 checkpoint."
  exit 1
fi

if [[ ! -f "$initial_checkpoint" ]]; then
  "$python_bin" -m experiments.migrate_rainbow_v5_to_v6 \
    "$task3_checkpoint" "$initial_checkpoint"
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
    print "[Task 4 V6] training chunk ${cumulative} already complete"
  else
    if [[ -e "$train_dir" ]]; then
      print -u2 "Incomplete run directory exists: $train_dir"
      exit 1
    fi
    if (( chunk == 1 )); then
      source_args=(--init-from-checkpoint "$initial_checkpoint")
    else
      previous_suffix=$(printf "%04d" "$((cumulative - rounds_per_chunk))")
      source_args=(--resume-from "runs/${prefix}_c${previous_suffix}")
    fi
    "$python_bin" -m experiments.run --config "$config" --mode train \
      --device cpu --task 4 --agent "$agent" --seed "$seed" \
      --n-rounds "$rounds_per_chunk" --target-stage-action-steps 999999999 \
      --min-rounds 100 "${source_args[@]}" --run-id "$train_id"
  fi
  checkpoint="${train_dir}/checkpoints/final.pt"
  eval_id="${train_id}_diag10"
  eval_summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
  if [[ ! -f "$eval_summary" ]]; then
    "$python_bin" -m experiments.run --config "$config" --mode evaluate \
      --device cpu --task 4 --agent "$agent" --checkpoint "$checkpoint" \
      --seeds "${evaluation_seeds[@]}" --n-rounds 1 --replay-policy failures \
      --run-id "$eval_id"
  fi
done
