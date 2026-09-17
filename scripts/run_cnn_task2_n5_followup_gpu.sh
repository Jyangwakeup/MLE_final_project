#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-t2-n5
#SBATCH --output=%x_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"

srun .venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated no CUDA device")
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name(0)}")
PY

srun .venv/bin/python -u -m experiments.run_cnn_task2_followup \
  --stage n5 \
  --work-directory runs/cnn_distilled_t2_n5_followup
