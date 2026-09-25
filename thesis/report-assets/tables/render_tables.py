#!/usr/bin/env python3
"""Render the five report tables from evidence CSV files."""
from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent
NAMES = ["model_inventory", "task1_task2", "task3_paired", "task4_attempts", "die_hardest"]
def render(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0])
    lines = [f"# {path.stem.replace('_', ' ').title()}", "", "由同名 CSV 自动生成；修改 CSV 后运行 python3 render_tables.py。", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(row[field].replace("|", "\\|") for field in fields) + " |" for row in rows)
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
for index, name in enumerate(NAMES, 1):
    render(ROOT / f"table{index}_{name}.csv")
