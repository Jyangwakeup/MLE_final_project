#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

python_bin=".venv/bin/python"
config="experiments/configs/rainbow_lite_v10_r21_task3.json"
agent="rainbow_lite_v10_agent"
initial_checkpoint="runs/rainbow_lite_v10_r21_task3_c0400_quadrant_warm_init.pt"
run_id="rainbow_lite_v10_r21_s11_task3_from_c0400_c0050"
eval_id="${run_id}_diag10"
evaluation_seeds=(12120 12121 12122 12123 12124 12125 12126 12127 12128 12129)

[[ -f "$initial_checkpoint" ]] || {
  print -u2 "Missing V10 warm-start checkpoint: $initial_checkpoint"
  exit 1
}

if [[ ! -f "runs/${run_id}/checkpoints/final.pt" ]]; then
  [[ ! -e "runs/${run_id}" ]] || { print -u2 "Incomplete run: runs/${run_id}"; exit 1; }
  "$python_bin" -m experiments.run --config "$config" --mode train \
    --device cpu --task 3 --agent "$agent" --seed 11 --n-rounds 50 \
    --target-stage-action-steps 999999999 --min-rounds 50 \
    --init-from-checkpoint "$initial_checkpoint" --run-id "$run_id"
fi

if [[ ! -f "runs/${eval_id}/${eval_id}_summary/summary.csv" ]]; then
  [[ ! -e "runs/${eval_id}" ]] || { print -u2 "Incomplete evaluation: runs/${eval_id}"; exit 1; }
  "$python_bin" -m experiments.run --config "$config" --mode evaluate \
    --device cpu --task 3 --agent "$agent" \
    --checkpoint "runs/${run_id}/checkpoints/final.pt" \
    --seeds "${evaluation_seeds[@]}" --n-rounds 1 \
    --replay-policy failures --run-id "$eval_id"
fi

"$python_bin" - <<'PY'
import torch

path = "runs/rainbow_lite_v10_r21_s11_task3_from_c0400_c0050/checkpoints/final.pt"
payload = torch.load(path, map_location="cpu", weights_only=True)
weight = payload["policy"]["trunk.0.weight"][:, 162:170]
names = ("crate_nw", "crate_ne", "crate_sw", "crate_se",
         "opponent_nw", "opponent_ne", "opponent_sw", "opponent_se")
for name, column in zip(names, weight.T):
    print(f"{name}: l2={column.norm().item():.8f} max_abs={column.abs().max().item():.8f}")
PY
