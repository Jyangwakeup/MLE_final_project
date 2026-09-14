"""Metrics and gates for the preregistered phase-aware Task 3 iteration."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import events as e


def phase_episode_metrics(run_directory: Path, agent_name: str) -> dict[str, float]:
    """Reconstruct weighted resource and late-mobility facts from raw traces."""
    directory = Path(run_directory)
    event_rows = [
        json.loads(line) for line in (directory / "phase_events.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    event_rows = [row for row in event_rows if row["agent_name"] == agent_name]
    timing_rows = [
        json.loads(line) for line in (directory / "timing.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    timing_rows = [
        row for row in timing_rows
        if row["agent_name"] == agent_name and "phase" in row
    ]
    early_steps = sum(float(row["phase"]["early_weight"]) for row in event_rows)
    early_coins = sum(
        float(row["phase"]["early_weight"]) * row["events"].count(e.COIN_COLLECTED)
        for row in event_rows)
    early_crates = sum(
        float(row["phase"]["early_weight"]) * row["events"].count(e.CRATE_DESTROYED)
        for row in event_rows)
    late_weight = sum(float(row["phase"]["late_weight"]) for row in timing_rows)
    late_crisis = sum(
        float(row["phase"]["late_weight"])
        * float(row["phase"]["late_mobility_crisis"])
        for row in timing_rows)
    return {
        "early_weighted_coins_per_100_steps": (
            100.0 * early_coins / early_steps if early_steps else 0.0),
        "early_weighted_crates_per_100_steps": (
            100.0 * early_crates / early_steps if early_steps else 0.0),
        "late_mobility_crisis_rate": (
            late_crisis / late_weight if late_weight else 0.0),
        "geometric_edge_rate": (
            mean(float(row["phase"].get("geometric_edge", False)) for row in timing_rows)
            if timing_rows else 0.0),
    }


def summarize_phase(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot summarize empty phase evidence")
    fields = (
        "early_weighted_coins_per_100_steps",
        "early_weighted_crates_per_100_steps",
        "late_mobility_crisis_rate", "geometric_edge_rate",
    )
    return {field: mean(float(row[field]) for row in rows) for field in fields}


def phase_gate_checks(
    parent_rows: Sequence[dict[str, Any]],
    child_rows: Sequence[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, bool]:
    parent = summarize_phase(parent_rows)
    child = summarize_phase(child_rows)
    coin_threshold = float(gates["early_weighted_coins_per_100_retention"])
    crate_threshold = float(gates["early_weighted_crates_per_100_retention"])
    crisis_limit = float(gates["late_mobility_crisis_rate"])
    reduction = float(gates["late_mobility_crisis_relative_reduction"])

    def retained(field: str, threshold: float) -> bool:
        return child[field] >= threshold * parent[field]

    crisis_pass = child["late_mobility_crisis_rate"] <= crisis_limit
    if parent["late_mobility_crisis_rate"] > 0:
        crisis_pass = crisis_pass or child["late_mobility_crisis_rate"] <= (
            (1.0 - reduction) * parent["late_mobility_crisis_rate"])
    return {
        "early_coin_rate_retention": retained(
            "early_weighted_coins_per_100_steps", coin_threshold),
        "early_crate_rate_retention": retained(
            "early_weighted_crates_per_100_steps", crate_threshold),
        "late_mobility_crisis": crisis_pass,
    }


def rank_arms(arms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the frozen worst-seed-first shortlist ordering."""
    def key(arm):
        seeds = arm["seed_results"]
        return (
            -sum(bool(item["passed"]) for item in seeds),
            max(float(item["suicide_rate"]) for item in seeds),
            -min(float(item["score_gain"]) for item in seeds),
            -mean(float(item["task3_score"]) for item in seeds),
            -mean(float(item["combat_metric"]) for item in seeds),
            -mean(float(item["resource_metric"]) for item in seeds),
            -mean(float(item["mobility_metric"]) for item in seeds),
            -min(float(item["retention_metric"]) for item in seeds),
            max(float(item["act_p95_seconds"]) for item in seeds),
            str(arm["run_id"]),
        )
    return sorted(arms, key=key)
