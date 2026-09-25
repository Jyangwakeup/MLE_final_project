#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v5_r18_maskv5_task4_c350_anchor.json"
agent="rainbow_lite_v5_agent"
seed=11
prefix="rainbow_lite_v5_r18_maskv5_s11_task4_c350_anchor"
initial_checkpoint="runs/rainbow_lite_v5_r18_maskv5_s11_task3_c100_anchor_c0350/checkpoints/final.pt"
task4_seeds=($(seq 12020 12039))
task3_seeds=($(seq 12000 12019))

run_completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$1/metadata.json"
}

evaluate_queue() {
  for cumulative in 50 100; do
    suffix=$(printf "%04d" "$cumulative")
    train_id="${prefix}_c${suffix}"
    train_dir="runs/${train_id}"
    checkpoint="${train_dir}/checkpoints/final.pt"
    while ! run_completed "$train_dir"; do sleep 5; done
    for task in 4 3; do
      if (( task == 4 )); then seeds=("${task4_seeds[@]}"); else seeds=("${task3_seeds[@]}"); fi
      eval_id="${train_id}_task${task}_eval20"
      summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
      [[ -f "$summary" ]] && continue
      [[ ! -e "runs/${eval_id}" ]] || { print -u2 "Incomplete eval exists: runs/${eval_id}"; return 1; }
      print "[Task 4 anchor] evaluating c${suffix} task${task} on 20 seeds"
      nice -n 15 "$python_bin" -u -m experiments.run \
        --config "$config" --mode evaluate --device cpu --task "$task" \
        --agent "$agent" --checkpoint "$checkpoint" \
        --seeds "${seeds[@]}" --n-rounds 1 --replay-policy failures \
        --run-id "$eval_id"
    done
  done
}

[[ -x "$python_bin" ]] || { print -u2 "Missing Python environment"; exit 1; }
[[ -f "$initial_checkpoint" ]] || { print -u2 "Missing c0350 checkpoint"; exit 1; }

evaluate_queue &
evaluation_worker=$!
trap 'kill "$evaluation_worker" 2>/dev/null || true' INT TERM EXIT

for cumulative in 50 100; do
  suffix=$(printf "%04d" "$cumulative")
  train_id="${prefix}_c${suffix}"
  train_dir="runs/${train_id}"
  if run_completed "$train_dir"; then continue; fi
  [[ ! -e "$train_dir" ]] || { print -u2 "Incomplete run exists: $train_dir"; exit 1; }
  if (( cumulative == 50 )); then
    source_args=(--init-from-checkpoint "$initial_checkpoint")
  else
    source_args=(--resume-from "runs/${prefix}_c0050")
  fi
  print "[Task 4 anchor] training $((cumulative-49))-${cumulative}/100"
  nice -n 5 "$python_bin" -u -m experiments.run \
    --config "$config" --mode train --device cpu --task 4 \
    --agent "$agent" --seed "$seed" --n-rounds 50 --min-rounds 50 \
    --target-stage-action-steps 999999999 --replay-policy none \
    "${source_args[@]}" --run-id "$train_id"
done

print "[Task 4 anchor] training complete; evaluations draining"
wait "$evaluation_worker"
trap - INT TERM EXIT
