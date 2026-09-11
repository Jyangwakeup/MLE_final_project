"""Frozen single- and multi-seed evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence


def run_multi_seed_evaluation(
    config_path: Path, seeds: Sequence[int], n_rounds: int,
    run_id_prefix: str, agent: str, opponents: Sequence[str], scenario: str,
    checkpoint: Path, task_name: str, replay_policy: str, replay_interval: int,
    *, runs_root: Path, project_root: Path, run_session: Callable,
    analyze_runs: Callable, write_json: Callable,
) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
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
    for seed in seeds:
        output = evaluation_root / f"{run_id_prefix}_s{seed}"
        run_directories.append(run_session(
            config_path, "evaluate", int(seed), output, agent, opponents,
            scenario, n_rounds, checkpoint, task_name, replay_policy,
            replay_interval,
        ))
    analyze_runs(run_directories, summary)
    write_json(summary / "fixed_evaluation.json", {
        "agent": agent, "checkpoint": str(checkpoint),
        "n_rounds_per_seed": n_rounds, "opponents": list(opponents),
        "replay_interval": replay_interval, "replay_policy": replay_policy,
        "scenario": scenario, "seeds": list(seeds), "task": task_name,
    })
    return summary


def run_evaluation_mode(
    args: Any, config: dict[str, Any], configured_seeds: Sequence[int],
    configured_rounds: int, task_name: str, scenario: str,
    opponents: Sequence[str], *, project_root: Path,
    output_directory: Callable, run_session: Callable,
    run_multi_seed: Callable, analyze_runs: Callable,
) -> Path:
    raw_checkpoint = args.checkpoint or config.get("checkpoint")
    if raw_checkpoint is None:
        raise ValueError("Evaluation requires --checkpoint or config.checkpoint")
    checkpoint = Path(raw_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (project_root / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    n_rounds = args.n_rounds or configured_rounds
    if args.seed is not None:
        output, _ = output_directory(args.run_id, args.output)
        completed = run_session(
            args.config, "evaluate", args.seed, output, args.agent, opponents,
            scenario, n_rounds, checkpoint, task_name,
            "none" if args.replay_policy == "auto" else args.replay_policy,
            args.replay_interval,
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
        args.replay_interval,
    )
