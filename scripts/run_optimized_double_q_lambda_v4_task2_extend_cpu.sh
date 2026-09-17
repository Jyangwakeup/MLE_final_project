#!/usr/bin/env bash
# Extend the selected 100k Task 2 run to 200k without changing its contract.
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda_v4_ext
#SBATCH --output=qlambda_v4_ext_%j.txt

set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

parent_t2="${1:?usage: sbatch $0 PARENT_T2 TASK1_EVAL PARENT_TASK2_EVAL TASK1_CONFIG TASK2_200K_CONFIG LABEL [SEED]}"
parent_t1_eval="${2:?missing parent Task 1 evaluation}"
parent_task2_eval="${3:?missing parent Task 2 baseline evaluation}"
t1_config="${4:?missing Task 1 config}"
t2_config="${5:?missing Task 2 200k config}"
label="${6:?missing label}"
seed="${7:-11}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_v4_${label}_t2_200k_s${seed}_j${job_id}"
python_bin="/home/students/ji/mles/.venv-compute/bin/python"

"${python_bin}" -m experiments.run --config "${t2_config}" --mode train \
  --device cpu --task 2 --agent optimized_double_q_lambda_v4_agent \
  --seed "${seed}" --run-id "${run_id}" --resume-from "${parent_t2}"

candidate_args=()
for checkpoint in "runs/${run_id}/checkpoints/snapshots/"*.pkl; do
  [ -e "${checkpoint}" ] || continue
  stem="$(basename "${checkpoint}" .pkl)"
  t1_eval="${run_id}_${stem}_task1"
  t2_eval="${run_id}_${stem}_task2"
  "${python_bin}" -m experiments.run --config "${t1_config}" --mode evaluate \
    --device cpu --task 1 --agent optimized_double_q_lambda_v4_agent \
    --checkpoint "${checkpoint}" --seeds $(seq 10000 10019) --n-rounds 1 \
    --run-id "${t1_eval}"
  "${python_bin}" -m experiments.run --config "${t2_config}" --mode evaluate \
    --device cpu --task 2 --agent optimized_double_q_lambda_v4_agent \
    --checkpoint "${checkpoint}" --seeds $(seq 10000 10019) --n-rounds 1 \
    --run-id "${t2_eval}"
  candidate_args+=(--candidate "${checkpoint}" "runs/${t1_eval}" "runs/${t2_eval}")
done

"${python_bin}" -m experiments.select_task2_snapshots \
  --agent optimized_double_q_lambda_v4_agent \
  --parent-task1 "${parent_t1_eval}" --parent-task2 "${parent_task2_eval}" \
  --gate experiments/task2_quality_gate.json \
  --output "runs/${run_id}/best_task2_selection.json" "${candidate_args[@]}"
