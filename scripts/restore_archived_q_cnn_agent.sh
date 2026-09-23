#!/usr/bin/env bash
# Restore one archived Q-learning/CNN variant for a reproducibility run.
set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
  echo "usage: $0 <archived-agent-name> [--destination <directory>]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "$0")/.." && pwd)
agent_name=$1
source_dir="$repo_root/experiments/agent_variants/$agent_name"
destination_root="$repo_root"
if [ "$#" -eq 3 ]; then
  if [ "$2" != "--destination" ]; then
    echo "expected --destination as second argument" >&2
    exit 2
  fi
  destination_root=$(cd "$3" && pwd)
fi
target_dir="$destination_root/agent_code/$agent_name"

if [ ! -d "$source_dir" ]; then
  echo "unknown archived agent: $agent_name" >&2
  exit 2
fi
if [ -e "$target_dir" ] || [ -L "$target_dir" ]; then
  echo "refusing to overwrite existing target: $target_dir" >&2
  exit 3
fi

mkdir -p "$destination_root/agent_code"
cp -a "$source_dir" "$target_dir"
echo "restored $agent_name to $target_dir"
