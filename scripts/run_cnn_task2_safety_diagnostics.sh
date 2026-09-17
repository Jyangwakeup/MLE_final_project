#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --qos=batch
#SBATCH --job-name=cnn-t2-maskdiag
#SBATCH --output=%x_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"

run_id="cnn_t2_d01_maskdiag_j${SLURM_JOB_ID}"
.venv/bin/python -u -m experiments.run \
  --config experiments/configs/task2_cnn_decision_diagnostics.json \
  --mode evaluate --task 2 --agent cnn_distilled_double_dqn_agent \
  --reward-id r7_safe_credit_sparse \
  --checkpoint runs/cnn_distilled_t2_j471458_transfer_s11/checkpoints/snapshots/policy_200000.pt \
  --device cpu --n-rounds 1 \
  --seeds 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 \
          10010 10011 10012 10013 10014 10015 10016 10017 10018 10019 \
  --run-id "$run_id" --replay-policy none

mapfile -t diagnostics < <(find "runs/$run_id" -name cnn_decisions.jsonl -type f | sort)
.venv/bin/python -m experiments.summarize_cnn_decisions \
  "${diagnostics[@]}" --output "runs/$run_id/decision_summary.json"
