#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda-r20-t2
#SBATCH --output=qlambda_r20_t2_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent_run="${1:?usage: sbatch $0 runs/<qualified-r20-task1-run>}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_r20_t2_100k_s11_j${job_id}"
py=.venv-compute/bin/python
t1=experiments/configs/optimized_double_q_lambda_r20_task1.json
t2=experiments/configs/optimized_double_q_lambda_r20_task2_100k.json
"$py" - "$parent_run/promotion_audit.json" <<'PY'
import json, sys
if not json.load(open(sys.argv[1], encoding="utf-8")).get("passed"):
    raise SystemExit("Task 1 parent did not pass promotion audit")
PY
"$py" -m experiments.run --config "$t2" --mode evaluate --device cpu --task 2 --agent optimized_double_q_lambda_agent --checkpoint "$parent_run/checkpoints/final.pkl" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "${run_id}_parent_task2"
"$py" -m experiments.run --config "$t2" --mode train --device cpu --task 2 --agent optimized_double_q_lambda_agent --seed 11 --run-id "$run_id" --resume-from "$parent_run"
parent_t1="$($py - "$parent_run/promotion_audit.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["stage_gate"])
PY
)"
candidate=()
for checkpoint in "runs/$run_id/checkpoints/snapshots/"*.pkl; do
  [ -e "$checkpoint" ] || continue
  stem="$(basename "$checkpoint" .pkl)"
  t1eval="${run_id}_${stem}_task1"; t2eval="${run_id}_${stem}_task2"
  "$py" -m experiments.run --config "$t1" --mode evaluate --device cpu --task 1 --agent optimized_double_q_lambda_agent --checkpoint "$checkpoint" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t1eval"
  "$py" -m experiments.run --config "$t2" --mode evaluate --device cpu --task 2 --agent optimized_double_q_lambda_agent --checkpoint "$checkpoint" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t2eval"
  candidate+=(--candidate "$checkpoint" "runs/$t1eval" "runs/$t2eval")
done
"$py" -m experiments.select_task2_snapshots --agent optimized_double_q_lambda_agent --parent-task1 "$parent_t1" --parent-task2 "runs/${run_id}_parent_task2" --gate experiments/task2_quality_gate.json --output "runs/$run_id/best_task2_selection.json" "${candidate[@]}"
