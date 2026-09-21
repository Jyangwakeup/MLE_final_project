#!/usr/bin/env bash
# One-shot reserved 100-seed Task 1/2 validation for the selected candidate.
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda_v4_main
#SBATCH --output=qlambda_v4_main_%j.txt

set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

checkpoint="${1:?checkpoint}"
task1_config="${2:?task1 config}"
task2_config="${3:?task2 config}"
parent_task1="${4:?parent Task 1 evaluation}"
parent_task2="${5:?parent Task 2 evaluation}"
label="${6:?label}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_v4_${label}_main100_j${job_id}"
python_bin="/home/students/ji/mles/.venv-compute/bin/python"

"${python_bin}" -m experiments.run --config "${task1_config}" --mode evaluate \
  --device cpu --task 1 --agent optimized_double_q_lambda_v4_agent \
  --checkpoint "${checkpoint}" --seeds $(seq 11000 11099) --n-rounds 1 \
  --run-id "${run_id}_task1"
"${python_bin}" -m experiments.run --config "${task2_config}" --mode evaluate \
  --device cpu --task 2 --agent optimized_double_q_lambda_v4_agent \
  --checkpoint "${checkpoint}" --seeds $(seq 11000 11099) --n-rounds 1 \
  --run-id "${run_id}_task2"
"${python_bin}" -m experiments.select_task2_snapshots \
  --agent optimized_double_q_lambda_v4_agent \
  --parent-task1 "${parent_task1}" --parent-task2 "${parent_task2}" \
  --gate experiments/task2_quality_gate.json --expected-episodes 100 \
  --candidate "${checkpoint}" "runs/${run_id}_task1" "runs/${run_id}_task2" \
  --output "runs/${run_id}_main_validation.json"
