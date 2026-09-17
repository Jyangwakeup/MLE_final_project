#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-task1
#SBATCH --output=%x_%j.txt

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Please submit this job from the project root}"

train_seed="${TRAIN_SEED:-11}"
train_rounds="${TRAIN_ROUNDS:-1000}"
reward_id="${REWARD_ID:-r5_coin_potential}"
run_id="cnn_task1_${reward_id}_s${train_seed}_j${SLURM_JOB_ID}"

srun echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
srun .venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated no CUDA device; do not run training on this node")
print(f"PyTorch={torch.__version__}")
print(f"CUDA={torch.version.cuda}")
print(f"GPU={torch.cuda.get_device_name(0)}")
PY

srun .venv/bin/python -u -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json \
  --mode train --task 1 --agent cnn_double_dqn_agent --device cuda \
  --reward-id "$reward_id" --seed "$train_seed" --n-rounds "$train_rounds" \
  --run-id "$run_id"
