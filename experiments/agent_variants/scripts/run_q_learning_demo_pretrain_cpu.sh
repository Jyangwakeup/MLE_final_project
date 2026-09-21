#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qdemo-pretrain
#SBATCH --output=qdemo_pretrain_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

dataset="${1:?missing demonstration dataset}"
initial="${2:?missing initial student checkpoint}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
output="runs/qdemo_pretrain_j${job_id}/pretrained.pkl"

.venv-compute/bin/python -m experiments.pretrain_q_learning_demonstrations \
  --dataset "$dataset" --initial-checkpoint "$initial" \
  --output "$output" --passes 5 --seed 20260919
