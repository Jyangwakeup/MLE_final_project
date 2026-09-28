#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../../../.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v5_r18_maskv5_task3_c100_anchor.json"
agent="rainbow_lite_v5_agent"
seed=11
rounds_per_chunk=50
base_rounds=100
total_rounds=400
initial_checkpoint="runs/rainbow_lite_v5_r18_maskv5_s11_task3_from_full500_c0100/checkpoints/final.pt"
prefix="rainbow_lite_v5_r18_maskv5_s11_task3_c100_anchor"

[[ -x "$python_bin" ]] || { print -u2 "Missing Python environment: $python_bin"; exit 1; }
[[ -f "$initial_checkpoint" ]] || { print -u2 "Missing c0100 checkpoint: $initial_checkpoint"; exit 1; }

run_completed() {
  [[ -f "$1/metadata.json" ]] || return 1
  "$python_bin" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$1/metadata.json"
}

for cumulative in $(seq "$((base_rounds+rounds_per_chunk))" 50 "$total_rounds"); do
  suffix=$(printf "%04d" "$cumulative")
  train_id="${prefix}_c${suffix}"
  train_dir="runs/${train_id}"
  if run_completed "$train_dir"; then
    print "[C100 anchor] c${suffix} already complete"
    continue
  fi
  [[ ! -e "$train_dir" ]] || { print -u2 "Incomplete run exists: $train_dir"; exit 1; }
  if (( cumulative == base_rounds + rounds_per_chunk )); then
    source_args=(--init-from-checkpoint "$initial_checkpoint")
  else
    previous=$(printf "%04d" "$((cumulative-rounds_per_chunk))")
    source_args=(--resume-from "runs/${prefix}_c${previous}")
  fi
  print "[C100 anchor] training $((cumulative-49))-${cumulative}/${total_rounds}"
  BOMBERMAN_INHERIT_WARM_START_PROGRESS=1 nice -n 5 "$python_bin" -u -m experiments.run \
    --config "$config" --mode train --device cpu \
    --task 3 --agent "$agent" --seed "$seed" \
    --n-rounds "$rounds_per_chunk" --min-rounds "$rounds_per_chunk" \
    --target-stage-action-steps 999999999 --replay-policy none \
    "${source_args[@]}" --run-id "$train_id"
done
