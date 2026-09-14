#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-path-eval
#SBATCH --output=%x_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Please submit this job from the project root}"

checkpoint="${CHECKPOINT:?Set CHECKPOINT to a completed path CNN checkpoint}"
run_id="${RUN_ID:-cnn_path_t1_eval_j${SLURM_JOB_ID}}"

srun .venv/bin/python -u -m experiments.run \
  --config experiments/configs/cnn_path_task1_r3.json \
  --mode evaluate --task 1 --agent cnn_path_double_dqn_agent --device cpu \
  --checkpoint "$checkpoint" \
  --seeds 10001 10002 10003 10004 10005 --n-rounds 20 \
  --run-id "$run_id"
