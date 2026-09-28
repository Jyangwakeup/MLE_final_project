#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../../../.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v5_r18_maskv5_task3.json"
agent="rainbow_lite_v5_agent"
seed=11
rounds_per_chunk=50
total_rounds=1500
task2_parent="runs/1.3/rainbow_lite_v5_r18_s11_task2_full500"
prefix="rainbow_lite_v5_r18_maskv5_s11_task3_from_full500"
evaluation_seeds=(12000 12001 12002 12003 12004 12005 12006 12007 12008 12009)

run_completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$1/metadata.json"
}

evaluate_queue() {
  for cumulative in $(seq 50 50 "$total_rounds"); do
    suffix=$(printf "%04d" "$cumulative")
    train_id="${prefix}_c${suffix}"
    checkpoint="runs/${train_id}/checkpoints/final.pt"
    eval_id="${train_id}_eval10"
    summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
    log="runs/${eval_id}.console.log"

    while ! run_completed "runs/${train_id}"; do sleep 5; done
    [[ -f "$summary" ]] && continue
    if [[ -e "runs/${eval_id}" ]]; then
      print -u2 "Incomplete evaluation exists: runs/${eval_id}"
      return 1
    fi
    print "[Task 3 V5/R18/M5] evaluating c${suffix} on 10 seeds"
    nice -n 15 "$python_bin" -u -m experiments.run \
      --config "$config" --mode evaluate --device cpu \
      --task 3 --agent "$agent" --checkpoint "$checkpoint" \
      --seeds "${evaluation_seeds[@]}" --n-rounds 1 \
      --replay-policy failures --run-id "$eval_id" >"$log" 2>&1
  done
}

[[ -x "$python_bin" ]] || { print -u2 "Missing Python environment: $python_bin"; exit 1; }
[[ -f "$task2_parent/checkpoints/final.pt" ]] || { print -u2 "Missing Task 2 parent: $task2_parent"; exit 1; }

evaluate_queue &
evaluation_worker=$!
trap 'kill "$evaluation_worker" 2>/dev/null || true' INT TERM EXIT

for cumulative in $(seq 50 50 "$total_rounds"); do
  suffix=$(printf "%04d" "$cumulative")
  train_id="${prefix}_c${suffix}"
  train_dir="runs/${train_id}"
  if run_completed "$train_dir"; then
    print "[Task 3 V5/R18/M5] c${suffix} already complete"
    continue
  fi
  [[ ! -e "$train_dir" ]] || {
    print -u2 "Incomplete training exists: $train_dir"
    exit 1
  }
  if (( cumulative == rounds_per_chunk )); then
    source_args=(--transfer-task3-safety-from "$task2_parent")
  else
    previous=$(printf "%04d" "$((cumulative-rounds_per_chunk))")
    source_args=(--resume-from "runs/${prefix}_c${previous}")
  fi
  print "[Task 3 V5/R18/M5] training $((cumulative-49))-${cumulative}/${total_rounds}"
  nice -n 5 "$python_bin" -u -m experiments.run \
    --config "$config" --mode train --device cpu \
    --task 3 --agent "$agent" --seed "$seed" \
    --n-rounds "$rounds_per_chunk" --min-rounds "$rounds_per_chunk" \
    --target-stage-action-steps 999999999 --replay-policy none \
    "${source_args[@]}" --run-id "$train_id"
done

print "[Task 3 V5/R18/M5] training complete; evaluation queue is draining"
wait "$evaluation_worker"
trap - INT TERM EXIT
print "[Task 3 V5/R18/M5] all training and evaluations complete"
