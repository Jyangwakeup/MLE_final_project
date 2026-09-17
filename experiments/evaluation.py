"""Frozen single- and multi-seed evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from experiments.devices import resolve_device
from experiments.agent_contracts import resolve_agent_contract
from experiments.progress_plugin import INLINE_PROGRESS
import settings as s


def run_multi_seed_evaluation(
    config_path: Path, seeds: Sequence[int], n_rounds: int,
    run_id_prefix: str, agent: str, opponents: Sequence[str], scenario: str,
    checkpoint: Path | None, task_name: str, replay_policy: str, replay_interval: int,
    *, runs_root: Path, project_root: Path, run_session: Callable,
    analyze_runs: Callable, write_json: Callable, device_info: dict[str, Any],
    feature_id_override: str | None = None,
    reward_id_override: str | None = None,
) -> Path:
    checkpoint = None if checkpoint is None else checkpoint.resolve()
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be non-empty and unique")
    evaluation_root = runs_root / run_id_prefix
    summary = evaluation_root / f"{run_id_prefix}_summary"
    if evaluation_root.exists():
        raise FileExistsError(
            "Refusing to overwrite existing output directory: "
            + str(evaluation_root.relative_to(project_root))
        )
    evaluation_root.mkdir(parents=True)
    run_directories = []
    with INLINE_PROGRESS.create(
        total=len(seeds) * n_rounds, desc="Evaluation", unit="round",
    ) as progress:
        for seed in seeds:
            output = evaluation_root / f"{run_id_prefix}_s{seed}"
            run_directories.append(run_session(
                config_path, "evaluate", int(seed), output, agent, opponents,
                scenario, n_rounds, checkpoint, task_name, replay_policy,
                replay_interval, device_info=device_info,
                feature_id_override=feature_id_override,
                reward_id_override=reward_id_override,
                show_progress=False,
            ))
            progress.update(n_rounds)
    analyze_runs(run_directories, summary)
    write_json(summary / "fixed_evaluation.json", {
        "agent": agent,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "exploration_disabled": True,
        "navigation_metrics": {
            "all_coins_target": int(s.SCENARIOS[scenario]["COIN_COUNT"]),
            "long_loop_threshold_steps": 10,
            "max_round_steps": 400,
            "reported": [
                "mean_coins", "min_coins_per_round", "max_coins_per_round",
                "zero_coin_round_rate", "all_coins_rate", "max_steps_rate",
                "long_wait_loop_rate", "long_ping_pong_loop_rate",
                "zero_utility_bomb_rate", "crates_per_bomb",
                "coins_per_100_steps", "steps_per_coin",
                "mean_all_coins_completion_steps", "wait_action_rate",
                "immediate_reverse_rate", "coin_distance_reducing_rate",
                "coin_target_switch_rate", "multiple_nearest_coin_rate",
            ],
        },
        "n_rounds_per_seed": n_rounds, "opponents": list(opponents),
        "replay_interval": replay_interval, "replay_policy": replay_policy,
        "scenario": scenario, "seeds": list(seeds), "task": task_name,
        "device": device_info,
    })
    return summary


def run_evaluation_mode(
    args: Any, config: dict[str, Any], configured_seeds: Sequence[int],
    configured_rounds: int, task_name: str, scenario: str,
    opponents: Sequence[str], *, project_root: Path,
    output_directory: Callable, run_session: Callable,
    run_multi_seed: Callable, analyze_runs: Callable,
) -> Path:
    contract = resolve_agent_contract(args.agent)
    raw_checkpoint = args.checkpoint or config.get("checkpoint")
    if raw_checkpoint is None and contract.checkpoint_name is not None:
        raise ValueError("Evaluation requires --checkpoint or config.checkpoint")
    checkpoint = None if raw_checkpoint is None else Path(raw_checkpoint)
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = (project_root / checkpoint).resolve()
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("config.evaluation must be an object")
    algorithm = contract.algorithm
    requested_device = args.device or evaluation.get("device", "cpu")
    device_info = resolve_device(algorithm, "evaluate", requested_device)
    n_rounds = args.n_rounds or configured_rounds
    replay_interval = args.replay_interval or 500
    if args.seed is not None:
        output, _ = output_directory(args.run_id, args.output)
        completed = run_session(
            args.config, "evaluate", args.seed, output, args.agent, opponents,
            scenario, n_rounds, checkpoint, task_name,
            "none" if args.replay_policy == "auto" else args.replay_policy,
            replay_interval,
            device_info=device_info,
            feature_id_override=getattr(args, "feature_id", None),
            reward_id_override=getattr(args, "reward_id", None),
        )
        summary = completed / "summary"
        analyze_runs([completed], summary)
        return summary
    if args.output is not None:
        raise ValueError("Multi-seed evaluation requires --run-id")
    seeds = tuple(args.seeds) if args.seeds is not None else configured_seeds
    return run_multi_seed(
        args.config, seeds, n_rounds, args.run_id, args.agent, opponents,
        scenario, checkpoint, task_name,
        "all" if args.replay_policy == "auto" else args.replay_policy,
        replay_interval,
        device_info=device_info,
        feature_id_override=getattr(args, "feature_id", None),
        reward_id_override=getattr(args, "reward_id", None),
    )
