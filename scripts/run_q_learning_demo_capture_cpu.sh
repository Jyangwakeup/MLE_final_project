#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qdemo-capture
#SBATCH --output=qdemo_capture_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_AUDIODRIVER=dummy
py=.venv-compute/bin/python
job_id="${SLURM_JOB_ID:?submit through sbatch}"
work="runs/qdemo_dataset_j${job_id}"
teacher=agent_code/double_dqn_continuous_v2_agent/final.pt
student=runs/qlambda_r20_t2_200k_s11_j473159/checkpoints/snapshots/step_0200000.pkl
"$py" -m experiments.collect_q_learning_demonstrations \
  --teacher "$teacher" --student "$student" --work-directory "$work"
"$py" -m experiments.pretrain_q_learning_demonstrations \
  --dataset "$work/student_transitions.npz" \
  --initial-checkpoint "$student" --output "$work/pretrained.pkl" \
  --passes 5 --seed 20260919
