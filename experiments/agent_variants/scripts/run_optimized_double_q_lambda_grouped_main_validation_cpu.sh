#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda-group-main
#SBATCH --output=qlambda_grouped_main_%j.txt
set -euo pipefail
cd /home/students/ji/mles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_AUDIODRIVER=dummy
base11="${1:?missing seed11 base run}"; base22="${2:?missing seed22 base run}"; base33="${3:?missing seed33 base run}"
py=.venv-compute/bin/python
agent=optimized_double_q_lambda_grouped_agent
job_id="${SLURM_JOB_ID:?submit through sbatch}"
winner="runs/qlambda_grouped_main_j${job_id}_winner.json"
"$py" - "$base11" "$base22" "$base33" > "$winner" <<'PY'
import json,sys
items=[]
for base in sys.argv[1:]:
    result=json.load(open(base+"/campaign_result.json",encoding="utf-8"))
    if not result["passed"]: raise SystemExit("all three training seeds must pass before main validation")
    best=result["best"]; metrics=best["task2"]
    rank=(-sum(best["gates"].values()), -best["balanced_capability"], metrics["long_wait_loop_rate"]+metrics["long_ping_pong_loop_rate"], best["checkpoint"])
    items.append((rank,result))
print(json.dumps(min(items,key=lambda item:item[0])[1],indent=2,sort_keys=True))
PY
checkpoint="$($py - "$winner" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["best"]["checkpoint"])
PY
)"
parent="$($py - "$winner" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["parent_task1_run"])
PY
)"
base="$($py - "$winner" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["base_task2_run"])
PY
)"
t1=experiments/configs/optimized_double_q_lambda_grouped_r20_task1.json
t2=experiments/configs/optimized_double_q_lambda_grouped_r20_task2_200k.json
run="qlambda_grouped_main_j${job_id}"
"$py" -m experiments.run --config "$t1" --mode evaluate --device cpu --task 1 --agent "$agent" --checkpoint "$checkpoint" --seeds $(seq 11000 11099) --n-rounds 1 --run-id "${run}_task1"
"$py" -m experiments.run --config "$t2" --mode evaluate --device cpu --task 2 --agent "$agent" --checkpoint "$checkpoint" --seeds $(seq 11000 11099) --n-rounds 1 --run-id "${run}_task2"
parent_t1="$($py - "$parent/promotion_audit.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["stage_gate"])
PY
)"
"$py" -m experiments.select_task2_snapshots --agent "$agent" --parent-task1 "$parent_t1" --parent-task2 "${base}_parent_task2" --gate experiments/task2_quality_gate.json --expected-episodes 100 --candidate "$checkpoint" "runs/${run}_task1" "runs/${run}_task2" --output "runs/$run/main_validation.json"
