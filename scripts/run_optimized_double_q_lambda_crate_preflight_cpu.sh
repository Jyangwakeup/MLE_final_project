#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda_crate_check
#SBATCH --output=qlambda_crate_check_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_AUDIODRIVER=dummy
py=.venv-compute/bin/python
agent=optimized_double_q_lambda_crate_agent
config=experiments/configs/optimized_double_q_lambda_crate_smoke.json
run_id="qlambda_crate_smoke_j${SLURM_JOB_ID}"
"$py" -m unittest tests.test_optimized_double_q_lambda_crate_agent tests.test_optimized_double_q_lambda_agent tests.test_new_agents -q
"$py" -m experiments.run --config "$config" --agent "$agent" --mode train --task 1 --device cpu --n-rounds 3 --run-id "$run_id"
"$py" -m experiments.run --config "$config" --agent "$agent" --mode evaluate --task 1 --device cpu --checkpoint "runs/$run_id/checkpoints/final.pkl" --seeds 7001 --n-rounds 3 --run-id "${run_id}_reload"
export CRATE_SMOKE_CHECKPOINT="runs/$run_id/checkpoints/final.pkl"
"$py" -m unittest tests.test_optimized_double_q_lambda_crate_package -q
