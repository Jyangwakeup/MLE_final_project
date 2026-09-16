"""Aggregate raw experiment episodes into a CSV summary and score chart."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev, stdev
from typing import Any, Iterable, Sequence

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt


EPISODE_SCHEMA_VERSION = "episode-v1"
TIMING_SCHEMA_VERSION = "timing-v1"
AGENT_METRICS = ("coins", "kills", "suicides", "crates", "invalid")
SUMMARY_FIELDS = (
    "run_id",
    "agent_name",
    "episode_count",
    "total_score",
    "rank_by_total_score",
    "mean_score",
    "median_score",
    "score_std",
    "mean_score_ci95_low",
    "mean_score_ci95_high",
    "total_round_steps",
    "coins",
    "mean_coins",
    "min_coins_per_round",
    "max_coins_per_round",
    "zero_coin_round_count",
    "zero_coin_round_rate",
    "coins_per_100_steps",
    "steps_per_coin",
    "all_coins_count",
    "all_coins_rate",
    "mean_all_coins_completion_steps",
    "max_steps_count",
    "max_steps_rate",
    "long_wait_loop_count",
    "long_wait_loop_rate",
    "long_ping_pong_loop_count",
    "long_ping_pong_loop_rate",
    "exploration_disabled",
    "kills",
    "mean_kills",
    "suicides",
    "suicide_rate",
    "killed_by_opponent",
    "killed_by_opponent_rate",
    "crates",
    "mean_crates",
    "mean_bombs",
    "zero_utility_bombs",
    "zero_utility_bomb_rate",
    "crates_per_bomb",
    "zero_bomb_round_rate",
    "bombs_resolved",
    "bombs_survived",
    "survived_bomb_rate",
    "invalid_actions",
    "invalid_action_rate",
    "survival_rate",
    "mean_survival_steps",
    "exclusive_wins",
    "exclusive_win_rate",
    "tied_first",
    "tied_first_rate",
    "zero_score_ties",
    "zero_score_tie_rate",
    "act_count",
    "act_mean_time",
    "act_p95_time",
    "act_max_time",
    "act_timeout_count",
    "act_skipped_count",
    "up_action_count",
    "up_action_rate",
    "right_action_count",
    "right_action_rate",
    "down_action_count",
    "down_action_rate",
    "left_action_count",
    "left_action_rate",
    "wait_action_count",
    "wait_action_rate_all_actions",
    "bomb_action_count",
    "bomb_action_rate",
    "median_longest_wait_streak",
    "min_longest_wait_streak",
    "max_longest_wait_streak",
    "safety_decisions",
    "safety_interventions",
    "safety_intervention_rate",
    "safety_fallbacks",
    "safety_fallback_rate",
    "navigation_decisions",
    "coin_distance_comparable_count",
    "coin_target_observation_count",
    "coin_target_continuity_opportunity_count",
    "wait_action_rate",
    "immediate_reverse_rate",
    "coin_distance_reducing_rate",
    "coin_target_switch_rate",
    "multiple_nearest_coin_rate",
    "conditional_loop_count",
    "conditional_loop_rate",
    "avoidable_wait_count",
    "wait_penalized_count",
    "wait_exempt_current_danger_count",
    "wait_exempt_next_danger_count",
    "wait_exempt_no_reachable_coin_count",
    "wait_exempt_no_safe_progress_move_count",
    "unseen_q_states",
    "q_decisions",
    "unseen_q_state_rate",
)

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def _mean_ci95(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Return a normal-approximation 95% CI for a sample mean."""
    if len(values) < 2:
        return None, None
    half_width = 1.96 * stdev(values) / math.sqrt(len(values))
    centre = mean(values)
    return centre - half_width, centre + half_width


def _invalid(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}: line {line_number}: {message}")


def _number(value: Any, field: str, path: Path, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _invalid(path, line_number, f"{field} must be a finite number")
    return float(value)


def _validate_episode(value: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(path, line_number, "episode must be a JSON object")
    if value.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise _invalid(path, line_number, f"schema_version must be {EPISODE_SCHEMA_VERSION!r}")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise _invalid(path, line_number, "run_id must be a non-empty string")
    if not isinstance(value.get("round_index"), int) or isinstance(value["round_index"], bool):
        raise _invalid(path, line_number, "round_index must be an integer")
    if not isinstance(value.get("round_steps"), int) or isinstance(value["round_steps"], bool):
        raise _invalid(path, line_number, "round_steps must be an integer")

    agents = value.get("agents")
    if not isinstance(agents, list) or not agents:
        raise _invalid(path, line_number, "agents must be a non-empty list")

    names: set[str] = set()
    for agent in agents:
        if not isinstance(agent, dict):
            raise _invalid(path, line_number, "each agent must be an object")
        name = agent.get("name")
        if not isinstance(name, str) or not name:
            raise _invalid(path, line_number, "agent.name must be a non-empty string")
        if name in names:
            raise _invalid(path, line_number, f"duplicate agent name: {name}")
        names.add(name)
        _number(agent.get("score"), "agent.score", path, line_number)
        for metric in AGENT_METRICS:
            _number(agent.get(metric), f"agent.{metric}", path, line_number)
        if "zero_utility_bombs" in agent:
            zero_utility = _number(
                agent["zero_utility_bombs"], "agent.zero_utility_bombs",
                path, line_number)
            if zero_utility < 0 or zero_utility > float(agent.get("bombs", 0)):
                raise _invalid(
                    path, line_number,
                    "agent.zero_utility_bombs must be between zero and agent.bombs")
        survived = agent.get("survived")
        dead = agent.get("dead")
        if not isinstance(survived, bool) or not isinstance(dead, bool) or survived == dead:
            raise _invalid(path, line_number, "agent.survived and agent.dead must be opposite booleans")
        death_step = agent.get("death_step")
        if death_step is not None and (
            not isinstance(death_step, int) or isinstance(death_step, bool) or death_step < 1
        ):
            raise _invalid(path, line_number, "agent.death_step must be null or a positive integer")
        killed_by_opponent = agent.get("killed_by_opponent", False)
        if not isinstance(killed_by_opponent, bool):
            raise _invalid(path, line_number, "agent.killed_by_opponent must be a boolean")
        killed_by_self = agent.get("killed_by_self", bool(agent.get("suicides", 0)))
        if not isinstance(killed_by_self, bool):
            raise _invalid(path, line_number, "agent.killed_by_self must be a boolean")
        death_causes = agent.get("death_causes", [])
        if not isinstance(death_causes, list):
            raise _invalid(path, line_number, "agent.death_causes must be a list")
        for cause in death_causes:
            if not isinstance(cause, dict):
                raise _invalid(path, line_number, "each death cause must be an object")
            if cause.get("type") not in {"self_bomb", "opponent_bomb"}:
                raise _invalid(path, line_number, "death cause type must be self_bomb or opponent_bomb")
            if not isinstance(cause.get("owner"), str) or not cause["owner"]:
                raise _invalid(path, line_number, "death cause owner must be a non-empty string")
    return value


def _read_episodes(run_directory: Path) -> list[dict[str, Any]]:
    episodes_path = run_directory / "episodes.jsonl"
    if not episodes_path.is_file():
        raise ValueError(f"{run_directory}: missing episodes.jsonl")

    episodes: list[dict[str, Any]] = []
    for line_number, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise _invalid(episodes_path, line_number, "blank lines are not valid episode records")
        try:
            raw_episode = json.loads(line)
        except json.JSONDecodeError as exception:
            raise _invalid(episodes_path, line_number, f"invalid JSON: {exception.msg}") from exception
        episode = _validate_episode(raw_episode, episodes_path, line_number)
        if episode["run_id"] != run_directory.name:
            raise _invalid(
                episodes_path,
                line_number,
                f"run_id {episode['run_id']!r} does not match directory name {run_directory.name!r}",
            )
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"{episodes_path}: contains no episode records")
    return episodes


def _validate_timing(value: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(path, line_number, "timing record must be a JSON object")
    if value.get("schema_version") != TIMING_SCHEMA_VERSION:
        raise _invalid(path, line_number, f"schema_version must be {TIMING_SCHEMA_VERSION!r}")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise _invalid(path, line_number, "run_id must be a non-empty string")
    if not isinstance(value.get("agent_name"), str) or not value["agent_name"]:
        raise _invalid(path, line_number, "agent_name must be a non-empty string")
    for field in ("skipped", "timed_out"):
        if not isinstance(value.get(field), bool):
            raise _invalid(path, line_number, f"{field} must be a boolean")
    think_time = value.get("think_time")
    if think_time is not None:
        _number(think_time, "think_time", path, line_number)
    navigation = value.get("navigation")
    if navigation is not None:
        if not isinstance(navigation, dict):
            raise _invalid(path, line_number, "navigation must be an object")
        for field in (
            "distance_comparable", "distance_reduced", "multiple_nearest_coins",
            "previous_target_still_available",
            "target_switched_while_previous_available", "waited",
            "immediate_reverse", "movement_legal_in_observed_state",
        ):
            if not isinstance(navigation.get(field), bool):
                raise _invalid(
                    path, line_number, f"navigation.{field} must be a boolean")
        for field in ("conditional_loop", "avoidable_wait"):
            if field in navigation and not isinstance(navigation[field], bool):
                raise _invalid(
                    path, line_number, f"navigation.{field} must be a boolean")
        if navigation.get("wait_disposition", "not_wait") not in {
            "not_wait", "penalized", "current_danger", "next_danger",
            "no_reachable_coin", "no_safe_progress_move",
        }:
            raise _invalid(
                path, line_number, "navigation.wait_disposition is invalid")
        for field in (
            "nearest_coin_distance_before",
            "predicted_nearest_coin_distance_after",
        ):
            if navigation.get(field) is not None:
                _number(navigation[field], f"navigation.{field}", path, line_number)
        if (
            not isinstance(navigation.get("coin_count_before"), int)
            or isinstance(navigation["coin_count_before"], bool)
            or navigation["coin_count_before"] < 0
        ):
            raise _invalid(
                path, line_number,
                "navigation.coin_count_before must be a non-negative integer",
            )
        if "selected_target" not in navigation:
            raise _invalid(
                path, line_number, "navigation.selected_target is required")
        selected_target = navigation["selected_target"]
        if selected_target is not None and (
            not isinstance(selected_target, list)
            or len(selected_target) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in selected_target)
        ):
            raise _invalid(
                path, line_number,
                "navigation.selected_target must be null or an integer pair",
            )
    return value


def _read_timing(run_directory: Path) -> list[dict[str, Any]]:
    timing_path = run_directory / "timing.jsonl"
    if not timing_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(timing_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise _invalid(timing_path, line_number, "blank lines are not valid timing records")
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exception:
            raise _invalid(timing_path, line_number, f"invalid JSON: {exception.msg}") from exception
        record = _validate_timing(raw_record, timing_path, line_number)
        if record["run_id"] != run_directory.name:
            raise _invalid(
                timing_path,
                line_number,
                f"run_id {record['run_id']!r} does not match directory name {run_directory.name!r}",
            )
        records.append(record)
    return records


def _read_q_diagnostics(run_directory: Path) -> dict[str, dict[str, float]]:
    diagnostics_path = run_directory / "q_diagnostics.jsonl"
    if not diagnostics_path.exists():
        return {}
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"unseen_q_states": 0.0, "q_decisions": 0.0}
    )
    for line_number, line in enumerate(diagnostics_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise _invalid(diagnostics_path, line_number, "blank lines are not valid Q diagnostic records")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exception:
            raise _invalid(diagnostics_path, line_number, f"invalid JSON: {exception.msg}") from exception
        if not isinstance(record, dict):
            raise _invalid(diagnostics_path, line_number, "Q diagnostic record must be an object")
        if record.get("run_id") != run_directory.name:
            raise _invalid(diagnostics_path, line_number, "run_id does not match directory name")
        agent_name = record.get("agent_name")
        if not isinstance(agent_name, str) or not agent_name:
            raise _invalid(diagnostics_path, line_number, "agent_name must be a non-empty string")
        totals[agent_name]["unseen_q_states"] += _number(
            record.get("unseen_q_states", 0), "unseen_q_states", diagnostics_path, line_number
        )
        totals[agent_name]["q_decisions"] += _number(
            record.get("q_decisions", 0), "q_decisions", diagnostics_path, line_number
        )
    return totals


def _ranking(agents: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    scores = {agent["name"]: float(agent["score"]) for agent in agents}
    highest_score = max(scores.values())
    first = {name for name, score in scores.items() if score == highest_score}
    if highest_score == 0 and all(score == 0 for score in scores.values()):
        return set(), set(), set(scores)
    if len(first) == 1:
        return first, set(), set()
    return set(), first, set()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _summary_row(
    run_id: str,
    agent_name: str,
    samples: list[dict[str, Any]],
    timing: dict[str, Any],
    q_diagnostics: dict[str, float],
) -> dict[str, Any]:
    scores = [sample["score"] for sample in samples]
    episode_count = len(samples)
    total_round_steps = sum(sample["round_steps"] for sample in samples)
    total_coins = sum(sample["coins"] for sample in samples)
    total_bombs = sum(sample["bombs"] for sample in samples)
    total_crates = sum(sample["crates"] for sample in samples)
    total_zero_utility_bombs = sum(
        sample["zero_utility_bombs"] for sample in samples)
    completion_steps = [
        sample["round_steps"] for sample in samples if sample["all_coins"]
    ]
    act_times = timing.get("think_times", [])
    navigation_decisions = timing.get("navigation_decisions", 0)
    comparable = timing.get("coin_distance_comparable", 0)
    target_observations = timing.get("coin_target_observations", 0)
    continuity_opportunities = timing.get(
        "coin_target_continuity_opportunities", 0)
    q_decisions = q_diagnostics.get("q_decisions", 0.0)
    unseen_q_states = q_diagnostics.get("unseen_q_states", 0.0)
    score_ci_low, score_ci_high = _mean_ci95(scores)
    action_counts = timing.get("action_counts", {})
    action_total = sum(action_counts.values())
    wait_streaks = timing.get("longest_wait_streaks", [])
    return {
        "run_id": run_id,
        "agent_name": agent_name,
        "episode_count": episode_count,
        "total_score": sum(scores),
        "rank_by_total_score": None,
        "mean_score": mean(scores),
        "median_score": median(scores),
        "score_std": pstdev(scores),
        "mean_score_ci95_low": score_ci_low,
        "mean_score_ci95_high": score_ci_high,
        "total_round_steps": total_round_steps,
        "coins": total_coins,
        "mean_coins": total_coins / episode_count,
        "min_coins_per_round": min(sample["coins"] for sample in samples),
        "max_coins_per_round": max(sample["coins"] for sample in samples),
        "zero_coin_round_count": sum(sample["coins"] == 0 for sample in samples),
        "zero_coin_round_rate": mean(sample["coins"] == 0 for sample in samples),
        "coins_per_100_steps": 100.0 * total_coins / max(1.0, total_round_steps),
        "steps_per_coin": total_round_steps / total_coins if total_coins else None,
        "all_coins_count": sum(sample["all_coins"] for sample in samples),
        "all_coins_rate": mean(sample["all_coins"] for sample in samples),
        "mean_all_coins_completion_steps": (
            mean(completion_steps) if completion_steps else None),
        "max_steps_count": sum(sample["max_steps"] for sample in samples),
        "max_steps_rate": mean(sample["max_steps"] for sample in samples),
        "long_wait_loop_count": sum(sample["long_wait_loop"] for sample in samples),
        "long_wait_loop_rate": mean(sample["long_wait_loop"] for sample in samples),
        "long_ping_pong_loop_count": sum(
            sample["long_ping_pong_loop"] for sample in samples),
        "long_ping_pong_loop_rate": mean(
            sample["long_ping_pong_loop"] for sample in samples),
        "exploration_disabled": all(
            sample["exploration_disabled"] for sample in samples),
        "kills": sum(sample["kills"] for sample in samples),
        "mean_kills": sum(sample["kills"] for sample in samples) / episode_count,
        "suicides": sum(sample["suicides"] for sample in samples),
        "suicide_rate": sum(sample["suicides"] for sample in samples) / episode_count,
        "killed_by_opponent": sum(sample["killed_by_opponent"] for sample in samples),
        "killed_by_opponent_rate": sum(sample["killed_by_opponent"] for sample in samples) / episode_count,
        "crates": total_crates,
        "mean_crates": total_crates / episode_count,
        "mean_bombs": total_bombs / episode_count,
        "zero_utility_bombs": total_zero_utility_bombs,
        "zero_utility_bomb_rate": (
            total_zero_utility_bombs / total_bombs if total_bombs else None),
        "crates_per_bomb": total_crates / total_bombs if total_bombs else None,
        "zero_bomb_round_rate": mean(sample["bombs"] == 0 for sample in samples),
        "bombs_resolved": sum(sample["bombs_resolved"] for sample in samples),
        "bombs_survived": sum(sample["bombs_survived"] for sample in samples),
        "survived_bomb_rate": (
            sum(sample["bombs_survived"] for sample in samples)
            / sum(sample["bombs_resolved"] for sample in samples)
            if sum(sample["bombs_resolved"] for sample in samples) else None),
        "invalid_actions": sum(sample["invalid"] for sample in samples),
        "invalid_action_rate": (
            sum(sample["invalid"] for sample in samples)
            / max(1.0, sum(sample["round_steps"] for sample in samples))
        ),
        "survival_rate": mean(sample["survived"] for sample in samples),
        "mean_survival_steps": mean(sample["survival_steps"] for sample in samples),
        "exclusive_wins": sum(sample["exclusive_win"] for sample in samples),
        "exclusive_win_rate": sum(sample["exclusive_win"] for sample in samples) / episode_count,
        "tied_first": sum(sample["tied_first"] for sample in samples),
        "tied_first_rate": sum(sample["tied_first"] for sample in samples) / episode_count,
        "zero_score_ties": sum(sample["zero_score_tie"] for sample in samples),
        "zero_score_tie_rate": sum(sample["zero_score_tie"] for sample in samples) / episode_count,
        "act_count": timing.get("act_count", 0),
        "act_mean_time": mean(act_times) if act_times else None,
        "act_p95_time": _percentile(act_times, 95),
        "act_max_time": max(act_times) if act_times else None,
        "act_timeout_count": timing.get("timeout_count", 0),
        "act_skipped_count": timing.get("skipped_count", 0),
        **{
            f"{action.lower()}_action_count": action_counts.get(action, 0)
            for action in ACTIONS
        },
        **{
            ("wait_action_rate_all_actions" if action == "WAIT"
             else f"{action.lower()}_action_rate"):
            (action_counts.get(action, 0) / action_total if action_total else None)
            for action in ACTIONS
        },
        "median_longest_wait_streak": (
            median(wait_streaks) if wait_streaks else None),
        "min_longest_wait_streak": min(wait_streaks) if wait_streaks else None,
        "max_longest_wait_streak": max(wait_streaks) if wait_streaks else None,
        "safety_decisions": timing.get("safety_decisions", 0),
        "safety_interventions": timing.get("safety_interventions", 0),
        "safety_intervention_rate": (
            timing.get("safety_interventions", 0) / timing.get("safety_decisions", 0)
            if timing.get("safety_decisions", 0) else None),
        "safety_fallbacks": timing.get("safety_fallbacks", 0),
        "safety_fallback_rate": (
            timing.get("safety_fallbacks", 0) / timing.get("safety_decisions", 0)
            if timing.get("safety_decisions", 0) else None),
        "navigation_decisions": navigation_decisions,
        "coin_distance_comparable_count": comparable,
        "coin_target_observation_count": target_observations,
        "coin_target_continuity_opportunity_count": continuity_opportunities,
        "wait_action_rate": (
            timing.get("wait_actions", 0) / navigation_decisions
            if navigation_decisions else None),
        "immediate_reverse_rate": (
            timing.get("immediate_reversals", 0) / navigation_decisions
            if navigation_decisions else None),
        "coin_distance_reducing_rate": (
            timing.get("coin_distance_reducing", 0) / comparable
            if comparable else None),
        "coin_target_switch_rate": (
            timing.get("coin_target_switches", 0) / continuity_opportunities
            if continuity_opportunities else None),
        "multiple_nearest_coin_rate": (
            timing.get("multiple_nearest_coins", 0) / target_observations
            if target_observations else None),
        "conditional_loop_count": timing.get("conditional_loops", 0),
        "conditional_loop_rate": (
            timing.get("conditional_loops", 0) / navigation_decisions
            if navigation_decisions else None),
        "avoidable_wait_count": timing.get("avoidable_waits", 0),
        "wait_penalized_count": timing.get("wait_penalized", 0),
        "wait_exempt_current_danger_count": timing.get("wait_current_danger", 0),
        "wait_exempt_next_danger_count": timing.get("wait_next_danger", 0),
        "wait_exempt_no_reachable_coin_count": timing.get("wait_no_reachable_coin", 0),
        "wait_exempt_no_safe_progress_move_count": timing.get("wait_no_safe_progress_move", 0),
        "unseen_q_states": unseen_q_states,
        "q_decisions": q_decisions,
        "unseen_q_state_rate": unseen_q_states / q_decisions if q_decisions else None,
    }


def summarize_runs(run_directories: Iterable[Path]) -> list[dict[str, Any]]:
    """Return one provenance-preserving summary row for every run and agent."""
    samples_by_agent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    timing_by_agent: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "think_times": [], "act_count": 0, "timeout_count": 0,
            "skipped_count": 0, "navigation_decisions": 0,
            "wait_actions": 0, "immediate_reversals": 0,
            "coin_distance_comparable": 0, "coin_distance_reducing": 0,
            "coin_target_observations": 0,
            "coin_target_continuity_opportunities": 0,
            "coin_target_switches": 0,
            "multiple_nearest_coins": 0,
            "conditional_loops": 0, "avoidable_waits": 0,
            "wait_penalized": 0, "wait_current_danger": 0,
            "wait_next_danger": 0, "wait_no_reachable_coin": 0,
            "wait_no_safe_progress_move": 0,
            "action_counts": defaultdict(int),
            "actions_by_round": defaultdict(list),
            "longest_wait_streaks": [],
            "safety_decisions": 0, "safety_interventions": 0,
            "safety_fallbacks": 0,
        }
    )
    q_by_agent: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"unseen_q_states": 0.0, "q_decisions": 0.0}
    )
    processed_directories: set[Path] = set()
    for candidate in run_directories:
        run_directory = Path(candidate).resolve()
        if not run_directory.is_dir():
            raise ValueError(f"{run_directory}: run directory does not exist")
        if run_directory in processed_directories:
            raise ValueError(f"{run_directory}: duplicate run directory")
        processed_directories.add(run_directory)

        for episode in _read_episodes(run_directory):
            exclusive, tied, zero_score = _ranking(episode["agents"])
            for agent in episode["agents"]:
                samples_by_agent[(episode["run_id"], agent["name"])].append(
                    {
                        "score": float(agent["score"]),
                        "coins": float(agent["coins"]),
                        "all_coins": float(agent.get("all_coins", agent["coins"] >= 50)),
                        "max_steps": float(agent.get("max_steps", episode["round_steps"] >= 400)),
                        "long_wait_loop": float(agent.get("long_wait_loop", False)),
                        "long_ping_pong_loop": float(
                            agent.get("long_ping_pong_loop", False)),
                        "exploration_disabled": bool(
                            episode.get("exploration_disabled", False)),
                        "kills": float(agent["kills"]),
                        "suicides": float(agent["suicides"]),
                        "crates": float(agent["crates"]),
                        "bombs": float(agent.get("bombs", 0)),
                        "zero_utility_bombs": float(
                            agent.get("zero_utility_bombs", 0)),
                        "bombs_resolved": float(agent.get("bombs_resolved", 0)),
                        "bombs_survived": float(agent.get("bombs_survived", 0)),
                        "invalid": float(agent["invalid"]),
                        "survived": float(agent["survived"]),
                        "killed_by_opponent": float(agent.get("killed_by_opponent", False)),
                        "round_steps": float(episode["round_steps"]),
                        "survival_steps": float(
                            episode["round_steps"]
                            if agent["survived"]
                            else agent.get("death_step") or episode["round_steps"]
                        ),
                        "exclusive_win": float(agent["name"] in exclusive),
                        "tied_first": float(agent["name"] in tied),
                        "zero_score_tie": float(agent["name"] in zero_score),
                    }
                )

        for record in _read_timing(run_directory):
            bucket = timing_by_agent[(record["run_id"], record["agent_name"])]
            action = record.get("action")
            if action in ACTIONS:
                bucket["action_counts"][action] += 1
                bucket["actions_by_round"][record.get("round_index")].append(action)
            if record["think_time"] is not None:
                bucket["think_times"].append(float(record["think_time"]))
                bucket["act_count"] += 1
            if record["timed_out"]:
                bucket["timeout_count"] += 1
            if record["skipped"]:
                bucket["skipped_count"] += 1
            safety = record.get("safety")
            if safety is not None:
                bucket["safety_decisions"] += 1
                bucket["safety_interventions"] += int(
                    safety.get("intervened", False))
                bucket["safety_fallbacks"] += int(safety.get("fallback", False))
            navigation = record.get("navigation")
            if navigation is not None:
                bucket["navigation_decisions"] += 1
                bucket["wait_actions"] += int(navigation["waited"])
                bucket["immediate_reversals"] += int(
                    navigation["immediate_reverse"])
                bucket["coin_distance_comparable"] += int(
                    navigation["distance_comparable"])
                bucket["coin_distance_reducing"] += int(
                    navigation["distance_reduced"])
                bucket["coin_target_observations"] += int(
                    navigation["selected_target"] is not None)
                bucket["coin_target_continuity_opportunities"] += int(
                    navigation["previous_target_still_available"])
                bucket["coin_target_switches"] += int(
                    navigation["target_switched_while_previous_available"])
                bucket["multiple_nearest_coins"] += int(
                    navigation["multiple_nearest_coins"])
                bucket["conditional_loops"] += int(navigation.get("conditional_loop", False))
                bucket["avoidable_waits"] += int(navigation.get("avoidable_wait", False))
                disposition = navigation.get("wait_disposition", "not_wait")
                if disposition != "not_wait":
                    bucket[f"wait_{disposition}"] += 1

        for (run_id, agent_name), bucket in timing_by_agent.items():
            if run_id != run_directory.name:
                continue
            for actions in bucket["actions_by_round"].values():
                longest = current = 0
                for action in actions:
                    current = current + 1 if action == "WAIT" else 0
                    longest = max(longest, current)
                bucket["longest_wait_streaks"].append(longest)

        for agent_name, diagnostics in _read_q_diagnostics(run_directory).items():
            bucket = q_by_agent[(run_directory.name, agent_name)]
            bucket["unseen_q_states"] += diagnostics["unseen_q_states"]
            bucket["q_decisions"] += diagnostics["q_decisions"]

    rows = [
        _summary_row(
            run_id,
            agent_name,
            samples,
            timing_by_agent[(run_id, agent_name)],
            q_by_agent[(run_id, agent_name)],
        )
        for (run_id, agent_name), samples in sorted(samples_by_agent.items())
    ]
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_run[row["run_id"]].append(row)
    for run_rows in rows_by_run.values():
        ordered_scores = sorted({row["total_score"] for row in run_rows}, reverse=True)
        for row in run_rows:
            row["rank_by_total_score"] = ordered_scores.index(row["total_score"]) + 1
    return rows


def _weighted_mean(rows: list[dict[str, Any]], field: str, weight_field: str = "episode_count") -> float | None:
    weighted_values = [
        (float(row[field]), float(row[weight_field]))
        for row in rows
        if row.get(field) is not None and float(row[weight_field]) > 0
    ]
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_agent[row["agent_name"]].append(row)

    averages: list[dict[str, Any]] = []
    for agent_name, agent_rows in sorted(rows_by_agent.items()):
        mean_scores = [float(row["mean_score"]) for row in agent_rows]
        score_ci_low, score_ci_high = _mean_ci95(mean_scores)
        q_decisions = sum(float(row["q_decisions"]) for row in agent_rows)
        unseen_q_states = sum(float(row["unseen_q_states"]) for row in agent_rows)
        total_round_steps = sum(
            float(row["total_round_steps"]) for row in agent_rows)
        total_coins = sum(float(row["coins"]) for row in agent_rows)
        total_bombs = sum(
            float(row["mean_bombs"]) * float(row["episode_count"])
            for row in agent_rows)
        total_crates = sum(float(row["crates"]) for row in agent_rows)
        total_zero_utility_bombs = sum(
            float(row["zero_utility_bombs"]) for row in agent_rows)
        all_coins_count = sum(
            float(row["all_coins_count"]) for row in agent_rows)
        act_times = [row["act_mean_time"] for row in agent_rows if row["act_mean_time"] is not None]
        p95_times = [row["act_p95_time"] for row in agent_rows if row["act_p95_time"] is not None]
        max_times = [row["act_max_time"] for row in agent_rows if row["act_max_time"] is not None]
        averages.append(
            {
                "run_id": "AVERAGE",
                "agent_name": agent_name,
                "episode_count": sum(int(row["episode_count"]) for row in agent_rows),
                "total_score": sum(float(row["total_score"]) for row in agent_rows),
                "rank_by_total_score": mean(float(row["rank_by_total_score"]) for row in agent_rows),
                "mean_score": _weighted_mean(agent_rows, "mean_score"),
                "median_score": median(mean_scores),
                "score_std": pstdev(mean_scores) if len(mean_scores) > 1 else 0.0,
                "mean_score_ci95_low": score_ci_low,
                "mean_score_ci95_high": score_ci_high,
                "total_round_steps": total_round_steps,
                "coins": total_coins,
                "mean_coins": _weighted_mean(agent_rows, "mean_coins"),
                "min_coins_per_round": min(
                    float(row["min_coins_per_round"]) for row in agent_rows),
                "max_coins_per_round": max(
                    float(row["max_coins_per_round"]) for row in agent_rows),
                "zero_coin_round_count": sum(
                    float(row["zero_coin_round_count"]) for row in agent_rows),
                "zero_coin_round_rate": _weighted_mean(
                    agent_rows, "zero_coin_round_rate"),
                "coins_per_100_steps": (
                    100.0 * total_coins / total_round_steps
                    if total_round_steps else None),
                "steps_per_coin": (
                    total_round_steps / total_coins if total_coins else None),
                "all_coins_count": all_coins_count,
                "all_coins_rate": _weighted_mean(agent_rows, "all_coins_rate"),
                "mean_all_coins_completion_steps": _weighted_mean(
                    agent_rows, "mean_all_coins_completion_steps", "all_coins_count"),
                "max_steps_count": sum(float(row["max_steps_count"]) for row in agent_rows),
                "max_steps_rate": _weighted_mean(agent_rows, "max_steps_rate"),
                "long_wait_loop_count": sum(
                    float(row["long_wait_loop_count"]) for row in agent_rows),
                "long_wait_loop_rate": _weighted_mean(
                    agent_rows, "long_wait_loop_rate"),
                "long_ping_pong_loop_count": sum(
                    float(row["long_ping_pong_loop_count"]) for row in agent_rows),
                "long_ping_pong_loop_rate": _weighted_mean(
                    agent_rows, "long_ping_pong_loop_rate"),
                "exploration_disabled": all(
                    bool(row["exploration_disabled"]) for row in agent_rows),
                "kills": sum(float(row["kills"]) for row in agent_rows),
                "mean_kills": _weighted_mean(agent_rows, "mean_kills"),
                "suicides": sum(float(row["suicides"]) for row in agent_rows),
                "suicide_rate": _weighted_mean(agent_rows, "suicide_rate"),
                "killed_by_opponent": sum(float(row["killed_by_opponent"]) for row in agent_rows),
                "killed_by_opponent_rate": _weighted_mean(agent_rows, "killed_by_opponent_rate"),
                "crates": total_crates,
                "mean_crates": _weighted_mean(agent_rows, "mean_crates"),
                "mean_bombs": (
                    total_bombs / sum(float(row["episode_count"]) for row in agent_rows)),
                "zero_utility_bombs": total_zero_utility_bombs,
                "zero_utility_bomb_rate": (
                    total_zero_utility_bombs / total_bombs if total_bombs else None),
                "crates_per_bomb": (
                    total_crates / total_bombs if total_bombs else None),
                "zero_bomb_round_rate": _weighted_mean(
                    agent_rows, "zero_bomb_round_rate"),
                "bombs_resolved": sum(
                    float(row["bombs_resolved"]) for row in agent_rows),
                "bombs_survived": sum(
                    float(row["bombs_survived"]) for row in agent_rows),
                "survived_bomb_rate": _weighted_mean(
                    agent_rows, "survived_bomb_rate", "bombs_resolved"),
                "invalid_actions": sum(float(row["invalid_actions"]) for row in agent_rows),
                "invalid_action_rate": _weighted_mean(agent_rows, "invalid_action_rate"),
                "survival_rate": _weighted_mean(agent_rows, "survival_rate"),
                "mean_survival_steps": _weighted_mean(agent_rows, "mean_survival_steps"),
                "exclusive_wins": sum(float(row["exclusive_wins"]) for row in agent_rows),
                "exclusive_win_rate": _weighted_mean(agent_rows, "exclusive_win_rate"),
                "tied_first": sum(float(row["tied_first"]) for row in agent_rows),
                "tied_first_rate": _weighted_mean(agent_rows, "tied_first_rate"),
                "zero_score_ties": sum(float(row["zero_score_ties"]) for row in agent_rows),
                "zero_score_tie_rate": _weighted_mean(agent_rows, "zero_score_tie_rate"),
                "act_count": sum(int(row["act_count"]) for row in agent_rows),
                "act_mean_time": mean(act_times) if act_times else None,
                "act_p95_time": mean(p95_times) if p95_times else None,
                "act_max_time": max(max_times) if max_times else None,
                "act_timeout_count": sum(int(row["act_timeout_count"]) for row in agent_rows),
                "act_skipped_count": sum(int(row["act_skipped_count"]) for row in agent_rows),
                **{
                    f"{action.lower()}_action_count": sum(
                        int(row[f"{action.lower()}_action_count"])
                        for row in agent_rows)
                    for action in ACTIONS
                },
                **{
                    ("wait_action_rate_all_actions" if action == "WAIT"
                     else f"{action.lower()}_action_rate"):
                    (
                        sum(int(row[f"{action.lower()}_action_count"])
                            for row in agent_rows)
                        / sum(
                            int(row[f"{candidate.lower()}_action_count"])
                            for row in agent_rows for candidate in ACTIONS)
                        if sum(
                            int(row[f"{candidate.lower()}_action_count"])
                            for row in agent_rows for candidate in ACTIONS) else None
                    )
                    for action in ACTIONS
                },
                "median_longest_wait_streak": median([
                    float(row["median_longest_wait_streak"])
                    for row in agent_rows
                    if row["median_longest_wait_streak"] is not None
                ]) if any(row["median_longest_wait_streak"] is not None
                          for row in agent_rows) else None,
                "min_longest_wait_streak": min([
                    float(row["min_longest_wait_streak"])
                    for row in agent_rows
                    if row["min_longest_wait_streak"] is not None
                ], default=None),
                "max_longest_wait_streak": max([
                    float(row["max_longest_wait_streak"])
                    for row in agent_rows
                    if row["max_longest_wait_streak"] is not None
                ], default=None),
                "navigation_decisions": sum(
                    int(row["navigation_decisions"]) for row in agent_rows),
                "coin_distance_comparable_count": sum(
                    int(row["coin_distance_comparable_count"])
                    for row in agent_rows),
                "coin_target_observation_count": sum(
                    int(row["coin_target_observation_count"])
                    for row in agent_rows),
                "coin_target_continuity_opportunity_count": sum(
                    int(row["coin_target_continuity_opportunity_count"])
                    for row in agent_rows),
                "wait_action_rate": _weighted_mean(
                    agent_rows, "wait_action_rate", "navigation_decisions"),
                "immediate_reverse_rate": _weighted_mean(
                    agent_rows, "immediate_reverse_rate", "navigation_decisions"),
                "coin_distance_reducing_rate": _weighted_mean(
                    agent_rows, "coin_distance_reducing_rate",
                    "coin_distance_comparable_count"),
                "coin_target_switch_rate": _weighted_mean(
                    agent_rows, "coin_target_switch_rate",
                    "coin_target_continuity_opportunity_count"),
                "multiple_nearest_coin_rate": _weighted_mean(
                    agent_rows, "multiple_nearest_coin_rate",
                    "coin_target_observation_count"),
                "conditional_loop_count": sum(
                    int(row["conditional_loop_count"]) for row in agent_rows),
                "conditional_loop_rate": _weighted_mean(
                    agent_rows, "conditional_loop_rate", "navigation_decisions"),
                "avoidable_wait_count": sum(
                    int(row["avoidable_wait_count"]) for row in agent_rows),
                "wait_penalized_count": sum(
                    int(row["wait_penalized_count"]) for row in agent_rows),
                "wait_exempt_current_danger_count": sum(
                    int(row["wait_exempt_current_danger_count"]) for row in agent_rows),
                "wait_exempt_next_danger_count": sum(
                    int(row["wait_exempt_next_danger_count"]) for row in agent_rows),
                "wait_exempt_no_reachable_coin_count": sum(
                    int(row["wait_exempt_no_reachable_coin_count"]) for row in agent_rows),
                "wait_exempt_no_safe_progress_move_count": sum(
                    int(row["wait_exempt_no_safe_progress_move_count"]) for row in agent_rows),
                "unseen_q_states": unseen_q_states,
                "q_decisions": q_decisions,
                "unseen_q_state_rate": unseen_q_states / q_decisions if q_decisions else None,
            }
        )
    return averages


def _write_summary(rows: list[dict[str, Any]], output_directory: Path) -> Path:
    summary_path = output_directory / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows([*rows, *_average_rows(rows)])
    return summary_path


def _write_mean_score_chart(rows: list[dict[str, Any]], output_directory: Path) -> Path:
    labels = [f"{row['run_id']}\n{row['agent_name']}" for row in rows]
    scores = [row["mean_score"] for row in rows]
    figure, axis = plt.subplots(figsize=(max(6, len(rows) * 1.3), 4.5))
    axis.bar(range(len(rows)), scores)
    axis.set_ylabel("Mean score per round")
    axis.set_title("Experiment mean scores")
    axis.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
    figure.tight_layout()
    chart_path = output_directory / "mean_score.png"
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return chart_path


def analyze_runs(run_directories: Iterable[Path], output_directory: Path) -> list[dict[str, Any]]:
    """Aggregate raw episodes and write ``summary.csv`` plus ``mean_score.png``."""
    rows = summarize_runs(run_directories)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_summary(rows, output)
    _write_mean_score_chart(rows, output)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analyze_runs(args.runs, args.output)
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
