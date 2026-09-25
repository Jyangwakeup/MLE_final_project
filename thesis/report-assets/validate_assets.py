#!/usr/bin/env python3
"""Fail when a report asset loses its source, protocol, or evidence status."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TABLES = Path(__file__).resolve().parent / "tables"
FIGURE_DATA = Path(__file__).resolve().parent / "figures" / "data"
ALLOWED = {"verified_raw", "verified_summary", "packaging_verified", "not_comparable", "unrun", "evidence_gap"}
MISSING = {"", "not_reported", "not_applicable"}


def validate_csv(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [f"{path}: no rows"]
    required = {"evidence_status", "source_path"}
    if path.name.startswith("table"):
        required.add("protocol_id" if path.name == "table2_task1_task2.csv" else "evidence_status")
    absent = required.difference(rows[0])
    if absent:
        return [f"{path}: missing columns {sorted(absent)}"]
    for index, row in enumerate(rows, 2):
        status, source = row["evidence_status"], row["source_path"]
        if status not in ALLOWED:
            errors.append(f"{path}:{index}: invalid evidence_status {status!r}")
        if source in MISSING:
            errors.append(f"{path}:{index}: missing source_path")
        elif not (REPO / source).exists():
            errors.append(f"{path}:{index}: source not found: {source}")
        for key, value in row.items():
            if key.endswith("rate") and value == "0" and status == "unrun":
                errors.append(f"{path}:{index}: unrun result must not use zero for {key}")
    return errors


def main() -> int:
    paths = sorted(TABLES.glob("table*.csv")) + sorted(FIGURE_DATA.glob("*.csv"))
    errors = [error for path in paths for error in validate_csv(path)]
    if errors:
        print("Report asset validation failed:", *errors, sep="\n- ")
        return 1
    print(f"Validated {len(paths)} CSV evidence files; all sources and statuses are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
