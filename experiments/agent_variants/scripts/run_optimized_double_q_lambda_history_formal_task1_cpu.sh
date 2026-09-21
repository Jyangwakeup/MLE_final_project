#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda-hist-t1
#SBATCH --output=qlambda_history_t1_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_history_r20_t1_s11_j${job_id}"
gate_id="${run_id}_stage_gate"
py=.venv-compute/bin/python
config=experiments/configs/optimized_double_q_lambda_history_r20_task1.json
agent=optimized_double_q_lambda_history_agent
"$py" -m experiments.run --config "$config" --mode train --device cpu --task 1 --agent "$agent" --seed 11 --run-id "$run_id"
"$py" -m experiments.package_agent --agent "$agent" --checkpoint "runs/$run_id/checkpoints/final.pkl" --output "runs/$run_id/submission-smoke.zip"
"$py" -m experiments.run --config "$config" --mode evaluate --device cpu --task 1 --agent "$agent" --checkpoint "runs/$run_id/checkpoints/final.pkl" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$gate_id"
"$py" -m experiments.audit_task1_promotion --training-run "runs/$run_id" --stage-gate "runs/$gate_id" --agent "$agent" --package "runs/$run_id/submission-smoke.zip" --output "runs/$run_id/promotion_audit.json"
