#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../../../.."

python_bin=.venv/bin/python
config=experiments/configs/final_rainbow_v5_r18_maskv5.json
agent=rainbow_lite_v5_agent
prefix=final_rainbow_v5_r18_maskv5_s11_t4_from_t3c0300
parent=runs/final_rainbow_v5_r18_maskv5_s11_t3_from_t2c0800_c0300
typeset -a seeds=({12020..12039})

completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(json.load(open(sys.argv[1])).get("status") != "completed")' "$1/metadata.json"
}

evaluate_queue() {
  local cumulative suffix train_id eval_id summary
  for cumulative in {100..1500..100}; do
    suffix=$(printf '%04d' "$cumulative")
    train_id="${prefix}_c${suffix}"
    eval_id="${train_id}_eval20"
    summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
    while ! completed "runs/${train_id}"; do sleep 10; done
    [[ -f "$summary" ]] && continue
    [[ ! -e "runs/${eval_id}" ]] || { print -u2 "Incomplete evaluation: ${eval_id}"; return 1; }
    print "[eval] Task 4 c${suffix}: 20 seeds"
    nice -n 15 "$python_bin" -u -m experiments.run \
      --config "$config" --mode evaluate --device cpu --task 4 \
      --agent "$agent" --checkpoint "runs/${train_id}/checkpoints/final.pt" \
      --seeds "${seeds[@]}" --n-rounds 1 --replay-policy none \
      --run-id "$eval_id" >"runs/${eval_id}.console.log" 2>&1
  done
}

completed "$parent" || { print -u2 "Task 3 c0300 parent is incomplete"; exit 1; }
evaluate_queue >"runs/${prefix}.eval-worker.log" 2>&1 &
eval_worker=$!
trap 'kill "$eval_worker" 2>/dev/null || true' INT TERM EXIT

previous="$parent"
for cumulative in {100..1500..100}; do
  suffix=$(printf '%04d' "$cumulative")
  train_id="${prefix}_c${suffix}"
  train_dir="runs/${train_id}"
  if completed "$train_dir"; then previous="$train_dir"; continue; fi
  [[ ! -e "$train_dir" ]] || { print -u2 "Incomplete training: $train_dir"; exit 1; }
  print "[train] Task 4 $((cumulative-99))-${cumulative}/1500 from Task 3 c0300"
  nice -n 5 "$python_bin" -u -m experiments.run \
    --config "$config" --mode train --device cpu --task 4 \
    --agent "$agent" --seed 11 --n-rounds 100 --min-rounds 100 \
    --target-stage-action-steps 999999999 --replay-policy none \
    --resume-from "$previous" --run-id "$train_id"
  previous="$train_dir"
done

print '[train] Task 4 reached the 1500-round cap; evaluations continue'
wait "$eval_worker"
trap - INT TERM EXIT
