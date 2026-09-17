#!/usr/bin/env bash
# Short controller job: inspect completed dependencies and submit the next DAG stage.
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --qos=batch
#SBATCH --job-name=qlambda_v4_ctl
#SBATCH --output=qlambda_v4_ctl_%j.txt

set -euo pipefail
cd /home/students/ji/mles
state="${1:?usage: sbatch $0 CAMPAIGN_STATE}"
/home/students/ji/mles/.venv-compute/bin/python \
  -m experiments.run_q_lambda_v4_task2_campaign --state "${state}" advance
