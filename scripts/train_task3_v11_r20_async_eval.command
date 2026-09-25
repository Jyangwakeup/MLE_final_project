#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v11_r20_task3.json"
agent="rainbow_lite_v11_agent"
seed=11
rounds_per_chunk=50
total_rounds=1500
initial_checkpoint="runs/rainbow_lite_v11_r20_task3_task2_warm_init.pt"
prefix="rainbow_lite_v11_r20_s11_task3_from_task2"
evaluation_seeds=(12120 12121 12122 12123 12124)
typeset -a evaluation_pids

run_completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$1/metadata.json"
}

launch_evaluation() {
  local cumulative="$1" suffix train_id eval_id summary log
  suffix=$(printf "%04d" "$cumulative")
  train_id="${prefix}_c${suffix}"
  eval_id="${train_id}_eval5"
  summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
  log="runs/${eval_id}.console.log"
  [[ -f "$summary" ]] && return 0
  [[ ! -e "runs/${eval_id}" ]] || { print -u2 "Incomplete evaluation exists: runs/${eval_id}"; return 1; }
  (
    nice -n 10 "$python_bin" -m experiments.run \
      --config "$config" --mode evaluate --device cpu --task 3 --agent "$agent" \
      --checkpoint "runs/${train_id}/checkpoints/final.pt" \
      --seeds "${evaluation_seeds[@]}" --n-rounds 1 \
      --replay-policy failures --run-id "$eval_id"
  ) >"$log" 2>&1 &
  evaluation_pids+=("$!")
  print "[Task 3 V11/R20] c${suffix} 5-seed evaluation started (pid $!)"
}

for cumulative in $(seq 50 50 "$total_rounds"); do
  suffix=$(printf "%04d" "$cumulative")
  train_id="${prefix}_c${suffix}"
  train_dir="runs/${train_id}"
  if ! run_completed "$train_dir"; then
    [[ ! -e "$train_dir" ]] || { print -u2 "Incomplete training exists: $train_dir"; exit 1; }
    if (( cumulative == rounds_per_chunk )); then
      source_args=(--init-from-checkpoint "$initial_checkpoint")
    else
      previous=$(printf "%04d" "$((cumulative-rounds_per_chunk))")
      source_args=(--resume-from "runs/${prefix}_c${previous}")
    fi
    print "[Task 3 V11/R20] training $((cumulative-rounds_per_chunk+1))-${cumulative}/${total_rounds}"
    "$python_bin" -m experiments.run \
      --config "$config" --mode train --device cpu --task 3 --agent "$agent" --seed "$seed" \
      --n-rounds "$rounds_per_chunk" --target-stage-action-steps 999999999 \
      --min-rounds "$rounds_per_chunk" --replay-interval 5 \
      "${source_args[@]}" --run-id "$train_id"
  fi
  launch_evaluation "$cumulative"
done

print "[Task 3 V11/R20] 1500-round training complete; waiting for evaluations"
for pid in "${evaluation_pids[@]}"; do wait "$pid"; done
print "[Task 3 V11/R20] all 1500 rounds and 30 evaluations complete"
