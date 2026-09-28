#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/../../../.."

trap 'printf "\n"; exit 0' INT TERM
while true; do
  line=$(scripts/archive/rainbow_lite/monitor/watch_task4_v6_r20_progress.command)
  printf '\r\033[K%s' "$line"
  sleep 5
done
