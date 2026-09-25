#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
exec .venv/bin/python scripts/dashboard_task3_v5_r18_maskv5.py
