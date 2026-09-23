#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda-r21-t2
#SBATCH --output=qlambda_r21_t2_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_AUDIODRIVER=dummy
parent_run="${1:?missing qualified Task 1 run}"; seed="${2:-11}"; replicate="${3:-yes}"
job_id="${SLURM_JOB_ID:?submit through sbatch}"; py=.venv-compute/bin/python
agent=optimized_double_q_lambda_agent
t1=experiments/configs/optimized_double_q_lambda_r21_task1.json
t2=experiments/configs/optimized_double_q_lambda_r21_task2_100k.json
run_id="qlambda_r21_t2_100k_s${seed}_j${job_id}"
"$py" - "$parent_run/promotion_audit.json" <<'PY'
import json,sys
if not json.load(open(sys.argv[1],encoding="utf-8")).get("passed"):
    raise SystemExit("Task 1 parent did not pass promotion audit")
PY
"$py" -m experiments.run --config "$t2" --mode evaluate --device cpu --task 2 --agent "$agent" --checkpoint "$parent_run/checkpoints/final.pkl" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "${run_id}_parent_task2"
"$py" -m experiments.run --config "$t2" --mode train --device cpu --task 2 --agent "$agent" --seed "$seed" --run-id "$run_id" --resume-from "$parent_run"
parent_t1="$($py - "$parent_run/promotion_audit.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["stage_gate"])
PY
)"
evaluate_run() {
  local target_run=$1 config=$2; local candidates=()
  for checkpoint in "runs/$target_run/checkpoints/snapshots/"*.pkl; do
    [ -e "$checkpoint" ] || continue
    local stem t1eval t2eval
    stem="$(basename "$checkpoint" .pkl)"; t1eval="${target_run}_${stem}_task1"; t2eval="${target_run}_${stem}_task2"
    "$py" -m experiments.run --config "$t1" --mode evaluate --device cpu --task 1 --agent "$agent" --checkpoint "$checkpoint" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t1eval"
    "$py" -m experiments.run --config "$config" --mode evaluate --device cpu --task 2 --agent "$agent" --checkpoint "$checkpoint" --seeds $(seq 10000 10019) --n-rounds 1 --run-id "$t2eval"
    candidates+=(--candidate "$checkpoint" "runs/$t1eval" "runs/$t2eval")
  done
  "$py" -m experiments.select_task2_snapshots --agent "$agent" --parent-task1 "$parent_t1" --parent-task2 "runs/${run_id}_parent_task2" --gate experiments/task2_quality_gate.json --output "runs/$target_run/best_task2_selection.json" "${candidates[@]}"
}
evaluate_run "$run_id" "$t2"
readarray -t decision < <("$py" - "runs/$run_id/best_task2_selection.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); b=p["best"]; m=b["task2"]
extend=(m["mean_coins"]>3.70 and m["mean_crates"]>=50.25 and m["long_ping_pong_loop_rate"]<0.70 and b["gates"]["suicide_rate"] and b["gates"]["survived_bomb_rate"] and b["gates"]["task1_retention"])
print("passed" if b["passed"] else "failed"); print("extend" if extend else "stop")
PY
)
final_run="$run_id"
if [[ "${decision[0]}" == failed && "${decision[1]}" == extend ]]; then
  t2ext=experiments/configs/optimized_double_q_lambda_r21_task2_200k.json
  final_run="qlambda_r21_t2_200k_s${seed}_j${job_id}"
  "$py" -m experiments.run --config "$t2ext" --mode train --device cpu --task 2 --agent "$agent" --seed "$seed" --run-id "$final_run" --resume-from "runs/$run_id" --task2-success-task1-config "$t1" --task2-success-parent-task1 "$parent_t1" --task2-success-parent-task2 "runs/${run_id}_parent_task2" --task2-success-gate experiments/task2_quality_gate.json
  evaluate_run "$final_run" "$t2ext"
  decision[0]="$($py - "runs/$final_run/best_task2_selection.json" <<'PY'
import json,sys
print("passed" if json.load(open(sys.argv[1],encoding="utf-8"))["best"]["passed"] else "failed")
PY
)"
fi
"$py" - "$parent_run" "runs/$run_id" "runs/$final_run" > "runs/$run_id/campaign_result.json" <<'PY'
import json,sys
parent,base,final=sys.argv[1:]; selection=json.load(open(final+"/best_task2_selection.json",encoding="utf-8"))
print(json.dumps({"parent_task1_run":parent,"base_task2_run":base,"final_task2_run":final,"passed":selection["best"]["passed"],"best":selection["best"]},indent=2,sort_keys=True))
PY
if [[ "${decision[0]}" == passed && "$replicate" == yes ]]; then
  child_jobs=(); child_runs=()
  for child_seed in 22 33; do
    t1job=$(sbatch --parsable scripts/run_optimized_double_q_lambda_r21_formal_task1_cpu.sh "$child_seed")
    t1run="runs/qlambda_r21_t1_s${child_seed}_j${t1job}"
    t2job=$(sbatch --parsable --dependency="afterok:${t1job}" scripts/run_optimized_double_q_lambda_r21_task2_cpu.sh "$t1run" "$child_seed" no)
    child_jobs+=("$t2job"); child_runs+=("runs/qlambda_r21_t2_100k_s${child_seed}_j${t2job}")
  done
  final_job=$(sbatch --parsable --dependency="afterok:${child_jobs[0]}:${child_jobs[1]}" scripts/run_optimized_double_q_lambda_r21_main_validation_cpu.sh "runs/$run_id" "${child_runs[0]}" "${child_runs[1]}")
  "$py" - "$final_job" "${child_jobs[@]}" > "runs/$run_id/replication_jobs.json" <<'PY'
import json,sys
print(json.dumps({"main_validation_job":sys.argv[1],"task2_jobs":sys.argv[2:]},indent=2))
PY
fi
