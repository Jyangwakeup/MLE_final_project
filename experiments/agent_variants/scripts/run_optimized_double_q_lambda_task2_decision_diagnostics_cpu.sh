#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda-t2-diag
#SBATCH --output=qlambda_t2_diag_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_AUDIODRIVER=dummy
py=.venv-compute/bin/python
config=experiments/configs/optimized_double_q_lambda_task2_decision_diagnostic.json
job=${SLURM_JOB_ID:?submit through sbatch}
run_one() {
  local agent=$1
  local checkpoint=$2
  local label=$3
  local run_id="qlambda_t2_diag_${label}_j${job}"
  "$py" -m experiments.run --config "$config" --mode evaluate --task 2 --device cpu \
    --agent "$agent" --checkpoint "$checkpoint" --seeds $(seq 10000 10019) \
    --n-rounds 1 --run-id "$run_id" --replay-policy none
  mapfile -t traces < <(find "runs/$run_id" -name q_learning_decisions.jsonl -type f | sort)
  "$py" -m experiments.summarize_q_learning_decisions \
    "${traces[@]}" --output "runs/$run_id/decision_summary.json"
}
run_one optimized_double_q_lambda_agent \
  runs/qlambda_t2_pilot_s11_j472902/checkpoints/snapshots/step_0175200.pkl old_v2
run_one optimized_double_q_lambda_crate_agent \
  runs/qlambda_crate_r7_t2_100k_s11_j473117/checkpoints/snapshots/step_0100000.pkl crate_100k
