#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/../../../.."

python_bin=".venv/bin/python"
agent="rainbow_lite_v6_stable_agent"
source_checkpoint="runs/rainbow_lite_v6_r19_s11_task4_r370_c0100/checkpoints/final.pt"
evaluation_seeds=(12120 12121 12122 12123 12124 12125 12126 12127 12128 12129 12130 12131 12132 12133 12134 12135 12136 12137 12138 12139)

[[ -x "$python_bin" && -f "$source_checkpoint" ]] || {
  print -u2 "Missing Python environment or V6/100 source checkpoint."
  exit 1
}

run_completed() {
  local metadata="$1/metadata.json"
  [[ -f "$metadata" ]] || return 1
  "$python_bin" - "$metadata" <<'PY'
import json
import sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if metadata.get("status") == "completed" else 1)
PY
}

evaluate() {
  local config="$1" checkpoint="$2" eval_id="$3"
  local summary="runs/${eval_id}/${eval_id}_summary/summary.csv"
  [[ -f "$summary" ]] && return
  [[ ! -e "runs/${eval_id}" ]] || {
    print -u2 "Incomplete evaluation exists: runs/${eval_id}"
    exit 1
  }
  "$python_bin" -m experiments.run --config "$config" --mode evaluate \
    --device cpu --task 4 --agent "$agent" --checkpoint "$checkpoint" \
    --seeds "${evaluation_seeds[@]}" --n-rounds 1 \
    --replay-policy failures --run-id "$eval_id"
}

baseline_eval_id="rainbow_lite_v6_r19_s11_task4_r370_c0100_diag20_stability"
baseline_summary="runs/${baseline_eval_id}/${baseline_eval_id}_summary/summary.csv"
if [[ ! -f "$baseline_summary" ]]; then
  [[ ! -e "runs/${baseline_eval_id}" ]] || {
    print -u2 "Incomplete baseline evaluation exists: runs/${baseline_eval_id}"
    exit 1
  }
  "$python_bin" -m experiments.run \
    --config experiments/configs/rainbow_lite_v6_r19_task4.json \
    --mode evaluate --device cpu --task 4 --agent rainbow_lite_v6_agent \
    --checkpoint "$source_checkpoint" --seeds "${evaluation_seeds[@]}" \
    --n-rounds 1 --replay-policy failures --run-id "$baseline_eval_id"
fi

for variant in r19 r19a r19b; do
  config="experiments/configs/rainbow_lite_v6_stable_${variant}_task4.json"
  run_id="rainbow_lite_v6_stable_${variant}_s11_task4_from_c0100_c0050_v2"
  run_dir="runs/${run_id}"
  if ! run_completed "$run_dir"; then
    [[ ! -e "$run_dir" ]] || {
      print -u2 "Incomplete training run exists: $run_dir"
      exit 1
    }
    "$python_bin" -m experiments.run --config "$config" --mode train \
      --device cpu --task 4 --agent "$agent" --seed 11 --n-rounds 50 \
      --target-stage-action-steps 999999999 --min-rounds 50 \
      --init-from-checkpoint "$source_checkpoint" --run-id "$run_id"
  fi
  evaluate "$config" "$run_dir/checkpoints/final.pt" "${run_id}_diag20"
done

print "V6 stability ablation completed."
