#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v5_r18_maskv5_task3_c100_anchor.json"
agent="rainbow_lite_v5_agent"
anchor_prefix="rainbow_lite_v5_r18_maskv5_s11_task3_c100_anchor"
base_prefix="rainbow_lite_v5_r18_maskv5_s11_task3_from_full500"
evaluation_seeds=($(seq 12000 12019))

run_completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$1/metadata.json"
}

for cumulative in $(seq 100 50 400); do
  suffix=$(printf "%04d" "$cumulative")
  if (( cumulative == 100 )); then
    train_id="${base_prefix}_c${suffix}"
  else
    train_id="${anchor_prefix}_c${suffix}"
  fi
  train_dir="runs/${train_id}"
  eval_id="${train_id}_eval20"
  summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
  checkpoint="${train_dir}/checkpoints/final.pt"

  while ! run_completed "$train_dir"; do sleep 5; done
  [[ -f "$checkpoint" ]] || { print -u2 "Missing checkpoint: $checkpoint"; exit 1; }
  if [[ -f "$summary" ]]; then
    print "[20-seed eval] c${suffix} already complete"
    continue
  fi
  [[ ! -e "runs/${eval_id}" ]] || {
    print -u2 "Incomplete evaluation exists: runs/${eval_id}"
    exit 1
  }
  print "[20-seed eval] evaluating c${suffix}"
  nice -n 15 "$python_bin" -u -m experiments.run \
    --config "$config" --mode evaluate --device cpu \
    --task 3 --agent "$agent" --checkpoint "$checkpoint" \
    --seeds "${evaluation_seeds[@]}" --n-rounds 1 \
    --replay-policy failures --run-id "$eval_id"
done
