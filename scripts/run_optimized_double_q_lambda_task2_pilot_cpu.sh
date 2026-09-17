#!/usr/bin/env bash
# Seed-11 Task 2 pilot from one qualified formal Task 1 parent run.
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda_t2_pilot
#SBATCH --output=qlambda_t2_pilot_%j.txt

set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

parent_run="${1:?usage: sbatch $0 runs/<qualified-task1-run>}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"
run_id="qlambda_t2_pilot_s11_j${job_id}"
python_bin="/home/students/ji/mles/.venv-compute/bin/python"
t1_config="experiments/configs/optimized_double_q_lambda_formal_task1.json"
t2_config="experiments/configs/optimized_double_q_lambda_task2.json"
parent_checkpoint="${parent_run}/checkpoints/final.pkl"
parent_name="$(basename "${parent_run}")"
parent_t1_eval="runs/${parent_name}_stage_gate"
parent_t2_eval="runs/${run_id}_parent_task2"

"${python_bin}" - "${parent_run}/promotion_audit.json" <<'PY'
import json
import sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
if not audit.get("passed"):
    raise SystemExit("Task 1 parent did not pass promotion audit")
PY

"${python_bin}" -m experiments.run \
  --config "${t2_config}" --mode evaluate --device cpu --task 2 \
  --agent optimized_double_q_lambda_agent --checkpoint "${parent_checkpoint}" \
  --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$(basename "${parent_t2_eval}")"

"${python_bin}" -m experiments.run \
  --config "${t2_config}" --mode train --device cpu --task 2 \
  --agent optimized_double_q_lambda_agent --seed 11 --run-id "${run_id}" \
  --resume-from "${parent_run}"

candidate_args=()
for checkpoint in "runs/${run_id}/checkpoints/snapshots/"*.pkl; do
  [ -e "${checkpoint}" ] || continue
  stem="$(basename "${checkpoint}" .pkl)"
  t1_eval="${run_id}_${stem}_task1"
  t2_eval="${run_id}_${stem}_task2"
  "${python_bin}" -m experiments.run \
    --config "${t1_config}" --mode evaluate --device cpu --task 1 \
    --agent optimized_double_q_lambda_agent --checkpoint "${checkpoint}" \
    --seeds $(seq 10000 10019) --n-rounds 1 --run-id "${t1_eval}"
  "${python_bin}" -m experiments.run \
    --config "${t2_config}" --mode evaluate --device cpu --task 2 \
    --agent optimized_double_q_lambda_agent --checkpoint "${checkpoint}" \
    --seeds $(seq 10000 10019) --n-rounds 1 --run-id "${t2_eval}"
  candidate_args+=(--candidate "${checkpoint}" "runs/${t1_eval}" "runs/${t2_eval}")
done

"${python_bin}" -m experiments.select_task2_snapshots \
  --agent optimized_double_q_lambda_agent \
  --parent-task1 "${parent_t1_eval}" --parent-task2 "${parent_t2_eval}" \
  --gate experiments/task2_quality_gate.json \
  --output "runs/${run_id}/best_task2_selection.json" \
  "${candidate_args[@]}"
