#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

prefix="rainbow_lite_v5_r18_s11_task3_retrain"
total_rounds=1500
trap 'printf "\n"; exit 0' INT TERM

while true; do
  completed=0 active="" chunk_rounds=0 evals=0
  for cumulative in $(seq 100 100 "$total_rounds"); do
    suffix=$(printf "%04d" "$cumulative")
    dir="runs/${prefix}_c${suffix}"
    if [[ -f "$dir/metadata.json" ]] && [[ "$(jq -r '.status' "$dir/metadata.json")" == "completed" ]]; then
      completed=$cumulative
    elif [[ -d "$dir" ]]; then
      active="$dir"
      [[ -f "$dir/training.csv" ]] && chunk_rounds=$(($(wc -l < "$dir/training.csv")-1))
      break
    else
      break
    fi
  done
  for cumulative in $(seq 100 100 "$total_rounds"); do
    suffix=$(printf "%04d" "$cumulative")
    id="${prefix}_c${suffix}_diag10"
    [[ -f "runs/$id/${id}_summary/summary.csv" ]] && evals=$((evals+1))
  done

  total=$((completed+chunk_rounds))
  (( total > total_rounds )) && total=$total_rounds
  reward="-" reward10="-" score10="-" kills10="-" suicides10="-" survival10="-"
  metric_dir="$active"
  [[ -n "$metric_dir" ]] || {
    suffix=$(printf "%04d" "$completed")
    metric_dir="runs/${prefix}_c${suffix}"
  }
  if [[ -n "$metric_dir" && -s "$metric_dir/training.csv" ]]; then
    metrics=$(tail -n +2 "$metric_dir/training.csv" | awk -F, '{x[NR]=$4;r=$4}END{s=NR-9;if(s<1)s=1;z=0;for(i=s;i<=NR;i++)z+=x[i];printf "%.2f\t%.2f",r,z/(NR-s+1)}')
    IFS=$'\t' read -r reward reward10 <<< "$metrics"
  fi
  if [[ -n "$metric_dir" && -s "$metric_dir/episodes.jsonl" ]]; then
    metrics=$(tail -n 10 "$metric_dir/episodes.jsonl" | jq -rs '[.[]|.agents[]|select(.name=="rainbow_lite_v5_agent")]as$a|if ($a|length)>0 then [($a|map(.score)|add/length),($a|map(.kills)|add),($a|map(.suicides)|add),($a|map(if .survived then 1 else 0 end)|add/length)] else [0,0,0,0] end|@tsv' -r)
    IFS=$'\t' read -r score10 kills10 suicides10 survival10 <<< "$metrics"
  fi
  percent=$((100*total/total_rounds))
  filled=$((percent/5)); empty=$((20-filled))
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
  survival_pct=$(awk -v x="$survival10" 'BEGIN{if(x=="-")print "-";else printf "%.0f",100*x}')
  printf '\r\033[K Task3 [%s] %4d/%d %3d%% | eval %2d/15 | reward %s avg10 %s | score10 %s | K/S %s/%s | survive %s%%' \
    "$bar" "$total" "$total_rounds" "$percent" "$evals" "$reward" "$reward10" "$score10" "$kills10" "$suicides10" "$survival_pct"
  sleep 3
done
