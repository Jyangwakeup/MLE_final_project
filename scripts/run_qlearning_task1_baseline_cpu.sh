#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qbaseline_t1
#SBATCH --output=qbaseline_t1_%j.txt

set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qbaseline_t1_s11_j${job_id}"
python_bin="/home/students/ji/mles/.venv/bin/python"

"${python_bin}" -m experiments.run \
  --config experiments/configs/q_learning_task1_baseline.json \
  --mode train --device cpu --task 1 --agent q_learning_agent --seed 11 \
  --run-id "${run_id}"

eval_runs=()
for checkpoint in "runs/${run_id}/checkpoints/snapshots/"*.pkl; do
  [ -e "${checkpoint}" ] || continue
  stem="$(basename "${checkpoint}" .pkl)"
  evaluation="${run_id}_eval_${stem}"
  "${python_bin}" -m experiments.run \
    --config experiments/configs/q_learning_task1_baseline.json \
    --mode evaluate --device cpu --task 1 --agent q_learning_agent \
    --checkpoint "${checkpoint}" --seeds 10000 10001 10002 10003 10004 \
    --n-rounds 20 --run-id "${evaluation}"
  eval_runs+=("runs/${evaluation}")
done
selection_args=()
for evaluation_run in "${eval_runs[@]}"; do
  selection_args+=(--evaluation-run "${evaluation_run}")
done
"${python_bin}" -m experiments.select_task1_snapshots --agent q_learning_agent \
  --output "runs/${run_id}/best_task1_selection.json" \
  "${selection_args[@]}"
