#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../../../.."

python_bin=.venv/bin/python
config=experiments/configs/final_rainbow_v5_r18_maskv5.json
agent=rainbow_lite_v5_agent
prefix=final_rainbow_v5_r18_maskv5_s11
typeset -A totals=(1 500 2 1000 3 1500 4 3000)
typeset -A seed_starts=(1 12100 2 12120 3 12000 4 12020)

completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(json.load(open(sys.argv[1])).get("status") != "completed")' "$1/metadata.json"
}

evaluate_queue() {
  local task cumulative suffix train_id eval_id checkpoint summary log
  for task in 1 2; do
    for cumulative in $(seq 100 100 "${totals[$task]}"); do
      suffix=$(printf '%04d' "$cumulative")
      train_id="${prefix}_t${task}_c${suffix}"
      eval_id="${train_id}_eval20"
      checkpoint="runs/${train_id}/checkpoints/final.pt"
      summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
      log="runs/${eval_id}.console.log"
      while ! completed "runs/${train_id}"; do sleep 10; done
      [[ -f "$summary" ]] && continue
      [[ ! -e "runs/${eval_id}" ]] || { print -u2 "Incomplete evaluation: runs/${eval_id}"; return 1; }
      local -a seeds=()
      for (( i=0; i<20; i++ )); do seeds+=("$((seed_starts[$task]+i))"); done
      print "[eval] Task ${task} c${suffix}: 20 seeds"
      nice -n 15 "$python_bin" -u -m experiments.run \
        --config "$config" --mode evaluate --device cpu --task "$task" \
        --agent "$agent" --checkpoint "$checkpoint" --seeds "${seeds[@]}" \
        --n-rounds 1 --replay-policy none --run-id "$eval_id" >"$log" 2>&1
    done
  done
}

[[ -x "$python_bin" ]] || { print -u2 "Missing $python_bin"; exit 1; }
evaluate_queue >"runs/${prefix}.eval-worker.log" 2>&1 &
eval_worker=$!
trap 'kill "$eval_worker" 2>/dev/null || true' INT TERM EXIT

previous_dir=""
for task in 1 2; do
  for cumulative in $(seq 100 100 "${totals[$task]}"); do
    suffix=$(printf '%04d' "$cumulative")
    train_id="${prefix}_t${task}_c${suffix}"
    train_dir="runs/${train_id}"
    if completed "$train_dir"; then
      previous_dir="$train_dir"
      continue
    fi
    [[ ! -e "$train_dir" ]] || { print -u2 "Incomplete training run: $train_dir"; exit 1; }
    local_source=()
    [[ -z "$previous_dir" ]] || local_source=(--resume-from "$previous_dir")
    print "[train] Task ${task} $((cumulative-99))-${cumulative}/${totals[$task]}"
    nice -n 5 "$python_bin" -u -m experiments.run \
      --config "$config" --mode train --device cpu --task "$task" \
      --agent "$agent" --seed 11 --n-rounds 100 --min-rounds 100 \
      --target-stage-action-steps 999999999 --replay-policy none \
      "${local_source[@]}" --run-id "$train_id"
    previous_dir="$train_dir"
  done
done
print '[train] Task 2 complete; select a checkpoint before Task 3'
wait "$eval_worker"
trap - INT TERM EXIT
print '[eval] All evaluations complete'
