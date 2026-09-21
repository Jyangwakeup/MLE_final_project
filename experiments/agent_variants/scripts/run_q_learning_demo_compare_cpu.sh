#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --qos=batch
#SBATCH --job-name=qdemo-compare
#SBATCH --output=qdemo_compare_%j.txt
set -euo pipefail
cd /home/students/ji/mles
demo="${1:?missing demo run}"; control="${2:?missing control run}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
.venv-compute/bin/python -m experiments.compare_q_learning_demo \
  --demo "$demo" --control "$control" \
  --output "runs/qdemo_comparison_j${job_id}.json"
