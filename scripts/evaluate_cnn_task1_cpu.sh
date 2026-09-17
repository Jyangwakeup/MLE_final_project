#!/usr/bin/env bash
# The project venv targets Python 3.10, which is currently available on the
# student GPU nodes but not the main/compute CPU nodes (Python 3.8 only).
# Inference itself remains CPU-only by experiment contract.
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-t1-eval
#SBATCH --output=%x_%j.txt

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Please submit this job from the project root}"

checkpoint="${CHECKPOINT:-runs/cnn_task1_r5_coin_potential_s11_j471409/checkpoints/final.pt}"
run_id="${RUN_ID:-cnn_task1_frozen_eval_j${SLURM_JOB_ID}}"

srun .venv/bin/python -u -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json \
  --mode evaluate --task 1 --agent cnn_double_dqn_agent --device cpu \
  --reward-id r5_coin_potential \
  --checkpoint "$checkpoint" \
  --seeds 10001 10002 10003 10004 10005 --n-rounds 20 \
  --run-id "$run_id"
