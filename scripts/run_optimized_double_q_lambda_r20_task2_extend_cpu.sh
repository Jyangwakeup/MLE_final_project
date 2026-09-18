#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda-r20-ext
#SBATCH --output=qlambda_r20_ext_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent_t2="${1:?usage: sbatch $0 PARENT_T2 PARENT_TASK1_EVAL PARENT_TASK2_EVAL}"
parent_t1_eval="${2:?missing parent Task 1 evaluation}"
parent_task2_eval="${3:?missing parent Task 2 evaluation}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_r20_t2_200k_s11_j${job_id}"
py=.venv-compute/bin/python
t1=experiments/configs/optimized_double_q_lambda_r20_task1.json
t2=experiments/configs/optimized_double_q_lambda_r20_task2_200k.json
"$py" -m experiments.run --config "$t2" --mode train --device cpu --task 2 --agent optimized_double_q_lambda_agent --seed 11 --run-id "$run_id" --resume-from "$parent_t2" --task2-success-task1-config "$t1" --task2-success-parent-task1 "$parent_t1_eval" --task2-success-parent-task2 "$parent_task2_eval" --task2-success-gate experiments/task2_quality_gate.json
candidate=()
for checkpoint in "runs/$run_id/checkpoints/snapshots/"*.pkl; do
  [ -e "$checkpoint" ] || continue
  stem="$(basename "$checkpoint" .pkl)"
  t1eval="${run_id}_${stem}_task1"; t2eval="${run_id}_${stem}_task2"
  "$py" -m experiments.run --config "$t1" --mode evaluate --device cpu --task 1 --agent optimized_double_q_lambda_agent --checkpoint "$checkpoint" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t1eval"
  "$py" -m experiments.run --config "$t2" --mode evaluate --device cpu --task 2 --agent optimized_double_q_lambda_agent --checkpoint "$checkpoint" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t2eval"
  candidate+=(--candidate "$checkpoint" "runs/$t1eval" "runs/$t2eval")
done
"$py" -m experiments.select_task2_snapshots --agent optimized_double_q_lambda_agent --parent-task1 "$parent_t1_eval" --parent-task2 "$parent_task2_eval" --gate experiments/task2_quality_gate.json --output "runs/$run_id/best_task2_selection.json" "${candidate[@]}"
