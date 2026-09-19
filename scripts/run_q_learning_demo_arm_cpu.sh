#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qdemo-arm
#SBATCH --output=qdemo_arm_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_AUDIODRIVER=dummy
arm="${1:?usage: sbatch $0 demo|control INIT_CHECKPOINT}"
initial="${2:?missing warm-start checkpoint}"
case "$arm" in demo|control) ;; *) echo "arm must be demo or control" >&2; exit 2;; esac
py=.venv-compute/bin/python
job_id="${SLURM_JOB_ID:?submit through sbatch}"
agent=optimized_double_q_lambda_demo_agent
config=experiments/configs/optimized_double_q_lambda_demo_task2_50k.json
t1=experiments/configs/optimized_double_q_lambda_r20_task1.json
run_id="qdemo_${arm}_s11_j${job_id}"
"$py" -m experiments.run --config "$config" --mode train --device cpu \
  --task 2 --agent "$agent" --seed 11 --run-id "$run_id" \
  --init-from-checkpoint "$initial"
candidates=()
evaluate_candidate() {
  local checkpoint=$1 label=$2
  local t1run="${run_id}_${label}_task1" t2run="${run_id}_${label}_task2"
  "$py" -m experiments.run --config "$t1" --mode evaluate --device cpu \
    --task 1 --agent "$agent" --checkpoint "$checkpoint" \
    --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t1run"
  "$py" -m experiments.run --config "$config" --mode evaluate --device cpu \
    --task 2 --agent "$agent" --checkpoint "$checkpoint" \
    --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t2run"
  candidates+=(--candidate "$checkpoint" "runs/$t1run" "runs/$t2run")
}
evaluate_candidate "$initial" initial
for checkpoint in "runs/$run_id/checkpoints/snapshots/"*.pkl; do
  [ -e "$checkpoint" ] || continue
  stem="$(basename "$checkpoint" .pkl)"
  evaluate_candidate "$checkpoint" "$stem"
done
"$py" -m experiments.select_task2_snapshots --agent "$agent" \
  --parent-task1 runs/qlambda_r20_t1_s11_j473126_stage_gate \
  --parent-task2 runs/qlambda_r20_t2_100k_s11_j473127_parent_task2 \
  --gate experiments/task2_quality_gate.json \
  --output "runs/$run_id/best_task2_selection.json" "${candidates[@]}"
