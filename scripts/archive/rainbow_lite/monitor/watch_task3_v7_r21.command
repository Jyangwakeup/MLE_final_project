#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../../../.."
prefix="rainbow_lite_v7_r21_s11_task3"
total=1500
trap 'printf "\n"; exit 0' INT TERM
while true; do
  done_rounds=0 active="" local_rounds=0 evals=0
  for n in $(seq 100 100 "$total"); do
    s=$(printf "%04d" "$n"); d="runs/${prefix}_c${s}"
    if [[ -f "$d/metadata.json" && "$(jq -r '.status' "$d/metadata.json")" == "completed" ]]; then done_rounds=$n
    elif [[ -d "$d" ]]; then active="$d"; [[ -f "$d/training.csv" ]] && local_rounds=$(($(wc -l < "$d/training.csv")-1)); break
    else break; fi
  done
  for n in $(seq 100 100 "$total"); do s=$(printf "%04d" "$n"); id="${prefix}_c${s}_diag10"; [[ -f "runs/$id/${id}_summary/summary.csv" ]] && evals=$((evals+1)); done
  current=$((done_rounds+local_rounds)); ((current>total)) && current=$total
  d="$active"; [[ -n "$d" ]] || { s=$(printf "%04d" "$done_rounds"); d="runs/${prefix}_c${s}"; }
  reward="-" avg="-" score="-" kills="-" suic="-" survival="-"
  if [[ -s "$d/training.csv" ]]; then m=$(tail -n +2 "$d/training.csv"|awk -F, '{x[NR]=$4;r=$4}END{s=NR-9;if(s<1)s=1;z=0;for(i=s;i<=NR;i++)z+=x[i];printf "%.2f\t%.2f",r,z/(NR-s+1)}'); IFS=$'\t' read -r reward avg <<< "$m"; fi
  if [[ -s "$d/episodes.jsonl" ]]; then m=$(tail -n 10 "$d/episodes.jsonl"|jq -rs '[.[]|.agents[]|select(.name=="rainbow_lite_v7_agent")]as$a|[($a|map(.score)|add/length),($a|map(.kills)|add),($a|map(.suicides)|add),($a|map(if .survived then 1 else 0 end)|add/length)]|@tsv' -r); IFS=$'\t' read -r score kills suic survival <<< "$m"; fi
  pct=$((100*current/total)); fill=$((pct/5)); blank=$((20-fill)); bar="$(printf '%*s' "$fill" ''|tr ' ' '#')$(printf '%*s' "$blank" ''|tr ' ' '-')"; sp=$(awk -v x="$survival" 'BEGIN{if(x=="-")print "-";else printf "%.0f",100*x}')
  printf '\r\033[K V7/R21-T3 [%s] %4d/%d %3d%% | eval %2d/15 | reward %s avg10 %s | score10 %s | K/S %s/%s | survive %s%%' "$bar" "$current" "$total" "$pct" "$evals" "$reward" "$avg" "$score" "$kills" "$suic" "$sp"
  sleep 3
done
