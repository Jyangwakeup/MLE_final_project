#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
prefix="rainbow_lite_v7_r21_s11_task4_r370"
trap 'printf "\n"; exit 0' INT TERM
while true; do
  completed=0; evals=0; active=""; chunk_rounds=0
  for cumulative in $(seq 100 100 3000); do
    suffix=$(printf "%04d" "$cumulative"); dir="runs/${prefix}_c${suffix}"
    if [[ -f "$dir/metadata.json" && "$(jq -r '.status' "$dir/metadata.json")" == "completed" ]]; then
      completed=$cumulative
    elif [[ -d "$dir" ]]; then
      active="$dir"; [[ -f "$dir/training.csv" ]] && chunk_rounds=$(($(wc -l < "$dir/training.csv")-1)); break
    else break; fi
  done
  for cumulative in $(seq 100 100 3000); do
    suffix=$(printf "%04d" "$cumulative"); id="${prefix}_c${suffix}_diag10"
    [[ -f "runs/$id/${id}_summary/summary.csv" ]] && evals=$((evals+1))
  done
  total=$((completed+chunk_rounds)); reward="-"; reward10="-"; score10="-"; kills10="-"; suicides10="-"; survival10="-"
  if [[ -n "$active" && -s "$active/training.csv" ]]; then
    m=$(tail -n +2 "$active/training.csv" | awk -F, '{x[NR]=$4;r=$4}END{s=NR-9;if(s<1)s=1;z=0;for(i=s;i<=NR;i++)z+=x[i];printf "%.2f\t%.2f",r,z/(NR-s+1)}')
    IFS=$'\t' read -r reward reward10 <<< "$m"
  fi
  if [[ -n "$active" && -s "$active/episodes.jsonl" ]]; then
    m=$(tail -n 10 "$active/episodes.jsonl" | jq -rs '[.[]|.agents[]|select(.name=="rainbow_lite_v7_agent")]as$a|[($a|map(.score)|add/length),($a|map(.kills)|add),($a|map(.suicides)|add),($a|map(if .survived then 1 else 0 end)|add/length)]|@tsv' -r)
    IFS=$'\t' read -r score10 kills10 suicides10 survival10 <<< "$m"
  fi
  printf '\r\033[K V7/R21 %4d/3000 | chunk %3d/100 | eval %d/30 | reward %.2f avg10 %.2f | score10 %.2f | K/S %s/%s | survive %.0f%%' "$total" "$chunk_rounds" "$evals" "$reward" "$reward10" "$score10" "$kills10" "$suicides10" "$(awk -v x="$survival10" 'BEGIN{print 100*x}')"
  sleep 3
done
