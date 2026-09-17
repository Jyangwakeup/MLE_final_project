#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-path-t1
#SBATCH --output=%x_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Please submit from the project root}"

config="${CONFIG:?Set CONFIG to an experiment JSON file}"
label="${LABEL:?Set LABEL to a short experiment name}"
seed="${TRAIN_SEED:-11}"
steps="${TARGET_STEPS:-100000}"
run_id="cnn_path_${label}_s${seed}_j${SLURM_JOB_ID}"

srun .venv/bin/python -m experiments.run --config "$config" --mode train \
  --task 1 --agent cnn_path_double_dqn_agent --device cpu --seed "$seed" \
  --n-rounds 3 --run-id "${run_id}_smoke"
srun .venv/bin/python -m experiments.run --config "$config" --mode evaluate \
  --task 1 --agent cnn_path_double_dqn_agent --device cpu --seed 10001 \
  --n-rounds 1 --checkpoint "runs/${run_id}_smoke/checkpoints/final.pt" \
  --run-id "${run_id}_smoke_eval"
srun .venv/bin/python -u -m experiments.run --config "$config" --mode train \
  --task 1 --agent cnn_path_double_dqn_agent --device cuda --seed "$seed" \
  --n-rounds 2000 --target-stage-action-steps "$steps" --min-rounds 1 \
  --run-id "$run_id"
srun .venv/bin/python scripts/evaluate_path_snapshots.py "runs/${run_id}" --config "$config"
