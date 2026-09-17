#!/usr/bin/env bash
# Formal v4 Task 1 lineage for one immutable reward contract.
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda_v4_t1
#SBATCH --output=qlambda_v4_t1_%j.txt

set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

config="${1:?usage: sbatch $0 CONFIG LABEL [SEED]}"
label="${2:?usage: sbatch $0 CONFIG LABEL [SEED]}"
seed="${3:-11}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_v4_${label}_t1_s${seed}_j${job_id}"
gate_id="${run_id}_stage_gate"
python_bin="/home/students/ji/mles/.venv-compute/bin/python"
checkpoint="runs/${run_id}/checkpoints/final.pkl"
package="runs/${run_id}/submission-smoke.zip"

"${python_bin}" -m experiments.run \
  --config "${config}" --mode train --device cpu --task 1 \
  --agent optimized_double_q_lambda_v4_agent --seed "${seed}" --run-id "${run_id}"

"${python_bin}" -m experiments.package_agent \
  --agent optimized_double_q_lambda_v4_agent --checkpoint "${checkpoint}" \
  --output "${package}"

"${python_bin}" -m experiments.run \
  --config "${config}" --mode evaluate --device cpu --task 1 \
  --agent optimized_double_q_lambda_v4_agent --checkpoint "${checkpoint}" \
  --seeds $(seq 10000 10019) --n-rounds 1 --run-id "${gate_id}"

"${python_bin}" -m experiments.audit_task1_promotion \
  --training-run "runs/${run_id}" --stage-gate "runs/${gate_id}" \
  --agent optimized_double_q_lambda_v4_agent --package "${package}" \
  --output "runs/${run_id}/promotion_audit.json"
