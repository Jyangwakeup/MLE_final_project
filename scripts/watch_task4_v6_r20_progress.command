#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."

prefix="rainbow_lite_v6_r20_s11_task4_r370"
completed=0
active_rounds=0
active_suffix=""
active_dir=""
evaluations=0

for cumulative in $(seq 100 100 3000); do
  suffix=$(printf "%04d" "$cumulative")
  run_dir="runs/${prefix}_c${suffix}"
  run_status=""
  if [[ -f "${run_dir}/metadata.json" ]]; then
    run_status=$(jq -r '.status // "unknown"' "${run_dir}/metadata.json")
  fi
  if [[ "$run_status" == "completed" ]]; then
    completed=$cumulative
  elif [[ -d "$run_dir" ]]; then
    active_suffix="$suffix"
    active_dir="$run_dir"
    if [[ -f "${run_dir}/training.csv" ]]; then
      lines=$(wc -l < "${run_dir}/training.csv")
      active_rounds=$((lines > 0 ? lines - 1 : 0))
    fi
    break
  else
    break
  fi
done

for cumulative in $(seq 100 100 3000); do
  suffix=$(printf "%04d" "$cumulative")
  eval_id="${prefix}_c${suffix}_diag10"
  if [[ -f "runs/${eval_id}/${eval_id}_summary/summary.csv" ]]; then
    evaluations=$((evaluations + 1))
  fi
done

current=$((completed + active_rounds))
(( current > 3000 )) && current=3000
percent=$((current * 100 / 3000))
filled=$((percent / 2))
empty=$((50 - filled))
bar=""
for _ in $(seq 1 "$filled" 2>/dev/null); do bar+="#"; done
for _ in $(seq 1 "$empty" 2>/dev/null); do bar+="-"; done

reward_now="-"; reward10="-"; loss="-"; epsilon="-"; replay="-"
score10="-"; kills10="-"; suicides10="-"; survival10="-"
if [[ -n "$active_dir" && -s "${active_dir}/training.csv" ]]; then
  metrics=$(tail -n +2 "${active_dir}/training.csv" | awk -F, '
    {r[NR]=$4; last_reward=$4; loss=$9; epsilon=$7; replay=$11}
    END {
      start=NR-9; if (start<1) start=1
      sum=0; for(i=start;i<=NR;i++) sum+=r[i]
      if (NR>0) printf "%.3f\t%.3f\t%.4f\t%.4f\t%s", last_reward, sum/(NR-start+1), loss, epsilon, replay
    }')
  IFS=$'\t' read -r reward_now reward10 loss epsilon replay <<< "$metrics"
fi
if [[ -n "$active_dir" && -s "${active_dir}/episodes.jsonl" ]]; then
  episode_metrics=$(tail -n 10 "${active_dir}/episodes.jsonl" | jq -rs '
    [.[] | .agents[] | select(.name == "rainbow_lite_v6_agent")] as $a |
    if ($a|length)==0 then "-\t-\t-\t-" else
      [($a|map(.score)|add/length), ($a|map(.kills)|add),
       ($a|map(.suicides)|add), ($a|map(if .survived then 1 else 0 end)|add/length)] |
      @tsv end' -r)
  IFS=$'\t' read -r score10 kills10 suicides10 survival10 <<< "$episode_metrics"
fi

if [[ -n "$active_suffix" ]]; then
  phase="train-c${active_suffix}:${active_rounds}/100"
elif (( completed == 3000 && evaluations < 30 )); then
  phase="final-evaluation"
elif (( completed == 3000 )); then
  phase="complete"
else
  phase="waiting"
fi

printf '[%s] %3d%% | rounds %d/3000 | %s | eval %d/30 | TRAIN reward now/avg10 %s/%s | TRAIN score avg10 %s | kills10 %s | suicides10 %s | survival10 %s | loss %s | eps %s | replay %s\n' \
  "$bar" "$percent" "$current" "$phase" "$evaluations" \
  "$reward_now" "$reward10" "$score10" "$kills10" "$suicides10" \
  "$survival10" "$loss" "$epsilon" "$replay"
