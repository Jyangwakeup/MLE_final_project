#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-distill-t1
#SBATCH --output=%x_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"

run_name="${RUN_NAME:-cnn_distilled_t1_j${SLURM_JOB_ID}}"
teacher="${TEACHER_CHECKPOINT:-agent_code/double_dqn_continuous_v2_agent/final.pt}"

srun .venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated no CUDA device")
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name(0)}")
PY

srun .venv/bin/python -u -m experiments.run_cnn_distillation_pipeline \
  --teacher-checkpoint "$teacher" --work-directory "runs/$run_name"
