#!/usr/bin/env bash
# Independent Task 1 confirmation for the selected frozen Q(lambda) checkpoint.
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda_t1_confirm
#SBATCH --output=qlambda_t1_confirm_%j.txt

set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_t1_confirm_s11000_j${job_id}"
python_bin="/home/students/ji/mles/.venv-compute/bin/python"
checkpoint="runs/qlambda_opt_t1_s11_j472872/checkpoints/snapshots/step_0100008.pkl"

"${python_bin}" -m experiments.run \
  --config experiments/configs/optimized_double_q_lambda_task1.json \
  --mode evaluate --device cpu --task 1 --agent optimized_double_q_lambda_agent \
  --checkpoint "${checkpoint}" --seeds $(seq 11000 11099) --n-rounds 1 \
  --run-id "${run_id}"
