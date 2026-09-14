#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-path-v2
#SBATCH --output=%x_%j.txt
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
config=experiments/configs/cnn_path_task1_e2_r3.json
run_id="cnn_path_v2_r3_s11_j${SLURM_JOB_ID}"
srun .venv/bin/python -m experiments.run --config "$config" --mode train \
  --task 1 --agent cnn_path_double_dqn_agent --device cpu --seed 11 \
  --n-rounds 3 --run-id "${run_id}_smoke"
srun .venv/bin/python -m experiments.run --config "$config" --mode evaluate \
  --task 1 --agent cnn_path_double_dqn_agent --device cpu --seed 10001 \
  --n-rounds 1 --checkpoint "runs/${run_id}_smoke/checkpoints/final.pt" \
  --run-id "${run_id}_smoke_eval"
srun .venv/bin/python -m experiments.run --config "$config" --mode train \
  --task 1 --agent cnn_path_double_dqn_agent --device cuda --seed 11 \
  --n-rounds 2000 --target-stage-action-steps 100000 --min-rounds 1 \
  --run-id "$run_id"
srun .venv/bin/python scripts/evaluate_path_snapshots.py "runs/${run_id}" --config "$config"
