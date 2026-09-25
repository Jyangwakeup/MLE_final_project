#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v7_r21_task4.json"
agent="rainbow_lite_v7_agent"
prefix="rainbow_lite_v7_r21_s11_task4_r370"
initial_checkpoint="runs/rainbow_lite_v7_r21_task4_r370_warm_init.pt"
evaluation_seeds=(12120 12121 12122 12123 12124 12125 12126 12127 12128 12129)

run_completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$1/metadata.json"
}

for cumulative in $(seq 100 100 3000); do
  suffix=$(printf "%04d" "$cumulative")
  run_id="${prefix}_c${suffix}"
  run_dir="runs/${run_id}"
  if ! run_completed "$run_dir"; then
    [[ ! -e "$run_dir" ]] || { print -u2 "Incomplete run: $run_dir"; exit 1; }
    if (( cumulative == 100 )); then
      source_args=(--init-from-checkpoint "$initial_checkpoint")
    else
      previous=$(printf "%04d" "$((cumulative-100))")
      source_args=(--resume-from "runs/${prefix}_c${previous}")
    fi
    "$python_bin" -m experiments.run --config "$config" --mode train --device cpu \
      --task 4 --agent "$agent" --seed 11 --n-rounds 100 \
      --target-stage-action-steps 999999999 --min-rounds 100 \
      "${source_args[@]}" --run-id "$run_id"
  fi
  eval_id="${run_id}_diag10"
  summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
  if [[ ! -f "$summary" ]]; then
    [[ ! -e "runs/${eval_id}" ]] || { print -u2 "Incomplete evaluation: runs/${eval_id}"; exit 1; }
    "$python_bin" -m experiments.run --config "$config" --mode evaluate --device cpu \
      --task 4 --agent "$agent" --checkpoint "$run_dir/checkpoints/final.pt" \
      --seeds "${evaluation_seeds[@]}" --n-rounds 1 --replay-policy failures \
      --run-id "$eval_id"
  fi
done
