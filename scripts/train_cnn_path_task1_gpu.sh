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
cd "${SLURM_SUBMIT_DIR:?Please submit this job from the project root}"

train_seed="${TRAIN_SEED:-11}"
target_steps="${TARGET_STEPS:-100000}"
run_id="${RUN_ID:-cnn_path_t1_e1_s${train_seed}_j${SLURM_JOB_ID}}"

srun .venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated no CUDA device")
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name(0)}")
PY

srun .venv/bin/python -u -m experiments.run \
  --config experiments/configs/cnn_path_task1_r3.json \
  --mode train --task 1 --agent cnn_path_double_dqn_agent --device cuda \
  --seed "$train_seed" --n-rounds 500 \
  --target-stage-action-steps "$target_steps" --min-rounds 250 \
  --run-id "$run_id"
