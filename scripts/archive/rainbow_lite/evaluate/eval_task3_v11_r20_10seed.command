#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../../../.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v11_r20_task3.json"
agent="rainbow_lite_v11_agent"
prefix="rainbow_lite_v11_r20_s11_task3_from_task2"
evaluation_seeds=(12120 12121 12122 12123 12124 12125 12126 12127 12128 12129)

for cumulative in $(seq 50 50 1500); do
  suffix=$(printf "%04d" "$cumulative")
  train_id="${prefix}_c${suffix}"
  metadata="runs/${train_id}/metadata.json"
  checkpoint="runs/${train_id}/checkpoints/final.pt"
  eval_id="${train_id}_eval10"
  summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
  log="runs/${eval_id}.console.log"
  while true; do
    if [[ -f "$metadata" ]] && "$python_bin" -c \
      'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' \
      "$metadata"; then
      break
    fi
    sleep 10
  done
  [[ -f "$checkpoint" ]] || { print -u2 "Completed run lacks checkpoint: $checkpoint"; exit 1; }
  [[ -f "$summary" ]] && continue
  [[ ! -e "runs/${eval_id}" ]] || {
    print -u2 "Incomplete 10-seed evaluation exists: runs/${eval_id}"
    exit 1
  }
  print "[Task 3 V11/R20] c${suffix} 10-seed evaluation"
  nice -n 10 "$python_bin" -m experiments.run \
    --config "$config" --mode evaluate --device cpu --task 3 --agent "$agent" \
    --checkpoint "$checkpoint" --seeds "${evaluation_seeds[@]}" --n-rounds 1 \
    --replay-policy failures --run-id "$eval_id" >"$log" 2>&1
done

print "[Task 3 V11/R20] all 30 10-seed evaluations complete"
