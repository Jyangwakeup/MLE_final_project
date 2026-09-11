"""Train or evaluate one agent on a reproducible Bomberman task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings as s
from environment import BombeRLeWorld, WorldArgs
from environment import Trophy
from experiments.analyze import analyze_runs
from experiments.analyze_training import analyze_training
from experiments.evaluation import run_evaluation_mode, run_multi_seed_evaluation
from experiments.training import (
    TrainingEarlyStopping,
    early_stopping_config as _early_stopping_config,
    run_training_mode,
)
from main import world_controller


RUNS_ROOT = PROJECT_ROOT / "runs"
TASKS = {
    1: {"name": "coin_navigation", "scenario": "coin-heaven", "opponents": ()},
    2: {"name": "crate_navigation", "scenario": "classic", "opponents": ()},
    3: {
        "name": "weak_opponents",
        "scenario": "classic",
        "opponents": ("peaceful_agent", "coin_collector_agent"),
    },
    4: {"name": "full_match", "scenario": "classic", "opponents": ("rule_based_agent",)},
}
EPISODE_SCHEMA_VERSION = "episode-v1"
TIMING_SCHEMA_VERSION = "timing-v1"
DEFAULT_EVALUATION_SEEDS = (10001, 10002, 10003, 10004, 10005)
DEFAULT_EVALUATION_ROUNDS = 20
DEFAULT_REPLAY_INTERVAL = 500
REPLAY_POLICIES = ("none", "failures", "sampled", "all")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _append_json_line(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


class ExperimentWorld(BombeRLeWorld):
    """Official world extension that appends one record after each completed round."""

    def __init__(
        self, args: WorldArgs, agents, output: Path, run_id: str,
        environment_seed: int, stage: str = "evaluation",
        replay_policy: str = "none", replay_interval: int = DEFAULT_REPLAY_INTERVAL,
    ):
        self._episodes_path = output / "episodes.jsonl"
        self._timing_path = output / "timing.jsonl"
        self._episodes_path.touch(exist_ok=False)
        self._timing_path.touch(exist_ok=False)
        self._experiment_run_id = run_id
        self._environment_seed = environment_seed
        self._stage = stage
        self._output = output
        self._replay_policy = replay_policy
        self._replay_interval = replay_interval
        self._death_steps: dict[str, int] = {}
        self._death_causes: dict[str, list[dict[str, str]]] = {}
        super().__init__(args, agents)

    def new_round(self) -> None:
        self._death_steps = {}
        self._death_causes = {}
        super().new_round()

    def evaluate_explosions(self) -> None:
        alive_before = {agent.name for agent in self.active_agents if not agent.dead}
        hits: dict[str, list[dict[str, str]]] = {}
        for explosion in self.explosions:
            if not explosion.is_dangerous():
                continue
            for agent in self.active_agents:
                if (not agent.dead) and (agent.x, agent.y) in explosion.blast_coords:
                    owner_name = explosion.owner.name
                    hits.setdefault(agent.name, []).append(
                        {
                            "type": "self_bomb" if agent is explosion.owner else "opponent_bomb",
                            "owner": owner_name,
                        }
                    )
        super().evaluate_explosions()
        alive_after = {agent.name for agent in self.active_agents if not agent.dead}
        for agent_name in alive_before - alive_after:
            self._death_steps.setdefault(agent_name, int(self.step))
            self._death_causes.setdefault(agent_name, hits.get(agent_name, []))

    def _append_timing(
        self,
        agent,
        action: str,
        requested_action: str | None,
        think_time: float | None,
        skipped: bool,
        timed_out: bool,
        available_before: float,
        available_after: float,
    ) -> None:
        _append_json_line(
            self._timing_path,
            {
                "schema_version": TIMING_SCHEMA_VERSION,
                "run_id": self._experiment_run_id,
                "round_index": int(self.round),
                "step": int(self.step),
                "agent_name": agent.name,
                "action": action,
                "requested_action": requested_action,
                "think_time": think_time,
                "skipped": skipped,
                "timed_out": timed_out,
                "available_before": available_before,
                "available_after": available_after,
            },
        )

    def poll_and_run_agents(self) -> None:
        for agent in self.active_agents:
            state = self.get_state_for_agent(agent)
            agent.store_game_state(state)
            agent.reset_game_events()
            if agent.available_think_time > 0:
                agent.act(state)

        permutation = self.rng.permutation(len(self.active_agents))
        self.replay["permutations"].append(permutation)
        for index in permutation:
            agent = self.active_agents[index]
            available_before = float(agent.available_think_time)
            requested_action = None
            think_time = None
            timed_out = False
            skipped = False
            if agent.available_think_time > 0:
                try:
                    requested_action, think_time = agent.wait_for_act()
                    action = requested_action
                except KeyboardInterrupt:
                    raise
                except Exception:
                    if not self.args.silence_errors:
                        raise
                    action = "ERROR"
                    requested_action = "ERROR"
                    think_time = float("inf")

                self.logger.info(
                    f"Agent <{agent.name}> chose action {action} in {think_time:.2f}s."
                )
                if think_time > available_before:
                    timed_out = True
                    next_think_time = agent.base_timeout - (think_time - available_before)
                    self.logger.warning(
                        f"Agent <{agent.name}> exceeded think time by "
                        f"{think_time - available_before:.2f}s. Setting action to "
                        f'"WAIT" and decreasing available time for next round to '
                        f"{next_think_time:.2f}s."
                    )
                    action = "WAIT"
                    agent.trophies.append(Trophy.time_trophy)
                    agent.available_think_time = next_think_time
                else:
                    self.logger.info(
                        f"Agent <{agent.name}> stayed within acceptable think time."
                    )
                    agent.available_think_time = agent.base_timeout
            else:
                self.logger.info(
                    f"Skipping agent <{agent.name}> because of last slow think time."
                )
                skipped = True
                agent.available_think_time += agent.base_timeout
                action = "WAIT"

            self._append_timing(
                agent,
                action,
                requested_action,
                None if think_time is None or not math.isfinite(think_time) else float(think_time),
                skipped,
                timed_out,
                available_before,
                float(agent.available_think_time),
            )
            self.replay["actions"][agent.name].append(action)
            self.perform_agent_action(agent, action)

    def end_round(self) -> None:
        super().end_round()
        agents = []
        for agent in self.agents:
            statistics = agent.statistics
            suicides = int(statistics.get("suicides", 0))
            dead = bool(agent.dead)
            death_causes = self._death_causes.get(agent.name, [])
            killed_by_opponent = any(cause["type"] == "opponent_bomb" for cause in death_causes)
            killed_by_self = any(cause["type"] == "self_bomb" for cause in death_causes)
            agents.append(
                {
                    "name": agent.name,
                    "score": int(agent.score),
                    "coins": int(statistics.get("coins", 0)),
                    "kills": int(statistics.get("kills", 0)),
                    "suicides": int(statistics.get("suicides", 0)),
                    "crates": int(statistics.get("crates", 0)),
                    "bombs": int(statistics.get("bombs", 0)),
                    "invalid": int(statistics.get("invalid", 0)),
                    "survived": not bool(agent.dead),
                    "dead": dead,
                    "death_step": self._death_steps.get(agent.name),
                    "death_causes": death_causes,
                    "killed_by_self": killed_by_self or bool(suicides),
                    "killed_by_opponent": killed_by_opponent,
                }
            )

        target = agents[0]
        failed = bool(target["dead"] or target["invalid"] or target["suicides"])
        sampled = self.round == 1 or self.round % self._replay_interval == 0
        save_reason = None
        if self._replay_policy == "all":
            save_reason = "all"
        elif self._replay_policy == "failures" and failed:
            save_reason = "failure"
        elif self._replay_policy == "sampled" and sampled:
            save_reason = "periodic_sample"
        if save_reason is not None:
            replay_directory = self._output / "replays"
            replay_directory.mkdir(exist_ok=True)
            replay_path = replay_directory / f"round_{self.round:05d}.pt"
            self.replay["n_steps"] = self.step
            with replay_path.open("wb") as file:
                pickle.dump(self.replay, file)
            _append_json_line(
                replay_directory / "manifest.jsonl",
                {
                    "reason": save_reason,
                    "replay": replay_path.name,
                    "round_index": int(self.round),
                    "score": target["score"],
                    "seed": self._environment_seed,
                    "target_agent": target["name"],
                },
            )

        _append_json_line(
            self._episodes_path,
            {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "run_id": self._experiment_run_id,
                "round_index": int(self.round),
                "seed": self._environment_seed,
                "environment_seed": self._environment_seed,
                "scenario": self.args.scenario,
                "stage": self._stage,
                "round_steps": int(self.step),
                "agents": agents,
            },
        )


def _source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _evaluation_seeds(experiment_seed: int) -> dict[str, int]:
    """Return the validation/test seed plan fixed by the implementation guide."""
    return {
        "experiment_seed": experiment_seed,
        "environment_seed": experiment_seed,
        "official_opponent_seed": experiment_seed + 100000,
    }


def _seed_official_rng(official_opponent_seed: int) -> None:
    """Seed official agents' shared Python and NumPy RNGs at lifecycle boundaries."""
    random.seed(official_opponent_seed)
    np.random.seed(official_opponent_seed)


def _initial_metadata(
    config_path: Path, mode: str, run_id: str, seeds: dict[str, int]
) -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for distribution in ("numpy", "tqdm", "pygame"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None

    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": None,
        "dependencies": dependencies,
        "ended_at": None,
        "error": None,
        "expanded_config": None,
        "hardware": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "mode": mode,
        "python_version": platform.python_version(),
        "run_id": run_id,
        "seed": seeds["experiment_seed"],
        "seeds": seeds,
        "source_commit": _source_commit(),
        "started_at": _utc_now(),
        "status": "running",
    }


def _read_config(config_path: Path) -> dict[str, Any]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Experiment config must contain a JSON object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _configured_evaluation(config: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    evaluation = config.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("config.evaluation must be an object")
    raw_seeds = evaluation.setdefault("seeds", list(DEFAULT_EVALUATION_SEEDS))
    n_rounds = evaluation.setdefault("n_rounds", DEFAULT_EVALUATION_ROUNDS)
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("config.evaluation.seeds must be a non-empty list")
    seeds: list[int] = []
    for index, seed in enumerate(raw_seeds):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"config.evaluation.seeds[{index}] must be an integer")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("config.evaluation.seeds must not contain duplicates")
    return tuple(seeds), _positive_int(n_rounds, "config.evaluation.n_rounds")


def _output_directory(run_id: str | None, output: Path | None) -> tuple[Path, str]:
    if run_id is not None:
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise ValueError("--run-id must be a single directory name")
        return RUNS_ROOT / run_id, run_id

    assert output is not None
    candidate = output if output.is_absolute() else PROJECT_ROOT / output
    candidate = candidate.resolve()
    if candidate.parent != RUNS_ROOT.resolve() or candidate.name in {"", ".", ".."}:
        raise ValueError("--output must name runs/<run_id>")
    return candidate, candidate.name


def _world_args(output: Path, scenario: str, seed: int, run_id: str) -> WorldArgs:
    return WorldArgs(
        no_gui=True,
        fps=0,
        turn_based=False,
        update_interval=0.0,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir=str(output / "logs"),
        save_stats=str(output / "official_stats.json"),
        match_name=run_id,
        seed=seed,
        silence_errors=False,
        scenario=scenario,
    )


def _agent_code_exists(agent: str, training: bool) -> None:
    agent_directory = PROJECT_ROOT / "agent_code" / agent
    required = [agent_directory / "callbacks.py"]
    if training:
        required.append(agent_directory / "train.py")
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Agent {agent!r} is missing: {', '.join(missing)}")


def _custom_agents(agent: str, opponents: Sequence[str], training: bool):
    names = [agent, *opponents]
    if not 1 <= len(names) <= s.MAX_AGENTS:
        raise ValueError(f"agent plus opponents must contain 1 to {s.MAX_AGENTS} agents")
    _agent_code_exists(agent, training)
    for opponent in opponents:
        _agent_code_exists(opponent, False)
    return [(agent, training), *((opponent, False) for opponent in opponents)]


def _task_settings(task: int, opponents: Sequence[str] | None):
    task_config = TASKS[task]
    resolved_opponents = tuple(task_config["opponents"] if opponents is None else opponents)
    if task in {1, 2} and resolved_opponents:
        raise ValueError(f"Task {task} does not use opponents")
    if task == 3 and opponents is not None:
        raise ValueError("Task 3 opponents are fixed to peaceful_agent and coin_collector_agent")
    if task == 4 and not resolved_opponents:
        raise ValueError("Task 4 requires at least one opponent")
    return str(task_config["name"]), str(task_config["scenario"]), resolved_opponents


def _training_seeds(experiment_seed: int) -> dict[str, int]:
    return {
        "experiment_seed": experiment_seed,
        "environment_seed": 1000 + experiment_seed,
        "official_opponent_seed": 3000 + experiment_seed,
    }


def _checkpoint_name(agent: str) -> str:
    return "final.pt" if "dqn" in agent.lower() else "final.pkl"


def run_agent_session(
    config_path: Path,
    mode: str,
    seed: int,
    output: Path,
    agent: str,
    opponents: Sequence[str],
    scenario: str,
    n_rounds: int,
    checkpoint: Path,
    task_name: str,
    replay_policy: str = "none",
    replay_interval: int = DEFAULT_REPLAY_INTERVAL,
    early_stopping_config: dict[str, Any] | None = None,
) -> Path:
    """Run one isolated training or frozen-evaluation session."""
    training = mode == "train"
    if mode not in {"train", "evaluate"}:
        raise ValueError(f"Unsupported agent session mode: {mode}")
    if scenario not in s.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    n_rounds = _positive_int(n_rounds, "n_rounds")
    if replay_policy not in REPLAY_POLICIES:
        raise ValueError(f"replay_policy must be one of {', '.join(REPLAY_POLICIES)}")
    replay_interval = _positive_int(replay_interval, "replay_interval")
    specs = _custom_agents(agent, opponents, training)
    checkpoint = checkpoint.resolve()
    if not training and not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    output.mkdir(parents=True, exist_ok=False)
    if training:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    seeds = _training_seeds(seed) if training else _evaluation_seeds(seed)
    metadata = _initial_metadata(config_path, mode, output.name, seeds)
    expanded = _read_config(config_path)
    expanded["seed"] = seed
    expanded["execution"] = {
        "agent": agent,
        "allow_bomb": task_name != "coin_navigation",
        "checkpoint": str(checkpoint),
        "n_rounds": n_rounds,
        "opponents": list(opponents),
        "replay_interval": replay_interval,
        "replay_policy": replay_policy,
        "scenario": scenario,
        "task": task_name,
    }
    metadata["expanded_config"] = expanded
    metadata["config_sha256"] = hashlib.sha256(
        json.dumps(expanded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata_path = output / "metadata.json"
    _write_json(metadata_path, metadata)

    previous = {
        name: os.environ.get(name)
        for name in (
            "BOMBERMAN_CHECKPOINT", "BOMBERMAN_CONFIG", "BOMBERMAN_RUN_DIR",
            "BOMBERMAN_RUN_ID", "BOMBERMAN_TRAINING_TASK",
            "BOMBERMAN_ALLOW_BOMB",
        )
    }
    try:
        (output / "logs").mkdir()
        os.environ["BOMBERMAN_CHECKPOINT"] = str(checkpoint)
        os.environ["BOMBERMAN_CONFIG"] = str(config_path.resolve())
        os.environ["BOMBERMAN_RUN_DIR"] = str(output.resolve())
        os.environ["BOMBERMAN_RUN_ID"] = output.name
        os.environ["BOMBERMAN_TRAINING_TASK"] = task_name
        # Task 1 isolates coin navigation. Keep its action space free of bombs
        # for both training and frozen evaluation so agents are compared under
        # the same curriculum constraint.
        os.environ["BOMBERMAN_ALLOW_BOMB"] = (
            "false" if task_name == "coin_navigation" else "true"
        )
        _seed_official_rng(seeds["official_opponent_seed"])
        world = ExperimentWorld(
            _world_args(output, scenario, seeds["environment_seed"], output.name),
            specs,
            output,
            output.name,
            seeds["environment_seed"],
            stage=f"{task_name}_{mode}",
            replay_policy=replay_policy,
            replay_interval=replay_interval,
        )
        _seed_official_rng(seeds["official_opponent_seed"])
        early_stopping = (
            TrainingEarlyStopping(output / "training.csv", early_stopping_config)
            if training and early_stopping_config is not None else None
        )
        completed_rounds = world_controller(
            world, n_rounds, gui=None, every_step=False, turn_based=False,
            make_video=False, update_interval=0.0, show_progress=True,
            stop_condition=early_stopping,
        )
        if training and not checkpoint.is_file():
            raise RuntimeError(f"Training did not write checkpoint: {checkpoint}")
        if training:
            analyze_training(output)
        metadata["ended_at"] = _utc_now()
        metadata["termination"] = {
            "completed_rounds": completed_rounds,
            "early_stopping": None if early_stopping is None else early_stopping.result,
            "requested_rounds": n_rounds,
        }
        metadata["status"] = "completed"
        _write_json(metadata_path, metadata)
        return output
    except BaseException as exception:
        metadata["ended_at"] = _utc_now()
        metadata["error"] = {"message": str(exception), "type": type(exception).__name__}
        metadata["status"] = "failed"
        _write_json(metadata_path, metadata)
        raise
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_agent_evaluation(
    config_path: Path,
    seeds: Sequence[int],
    n_rounds: int,
    run_id_prefix: str,
    agent: str,
    opponents: Sequence[str],
    scenario: str,
    checkpoint: Path,
    task_name: str,
    replay_policy: str = "all",
    replay_interval: int = DEFAULT_REPLAY_INTERVAL,
) -> Path:
    return run_multi_seed_evaluation(
        config_path, seeds, n_rounds, run_id_prefix, agent, opponents,
        scenario, checkpoint, task_name, replay_policy, replay_interval,
        runs_root=RUNS_ROOT, project_root=PROJECT_ROOT,
        run_session=run_agent_session, analyze_runs=analyze_runs,
        write_json=_write_json,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("train", "evaluate"))
    parser.add_argument("--task", required=True, type=int, choices=TASKS)
    parser.add_argument("--agent", required=True)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--n-rounds", type=int)
    parser.add_argument("--opponents", nargs="*")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--replay-policy", choices=("auto", *REPLAY_POLICIES), default="auto",
        help="Replay retention: auto, none, failures, sampled, or all",
    )
    parser.add_argument(
        "--replay-interval", type=int,
        help=("Rounds between sampled training replays; by default train mode "
              "saves at each 10%% progress milestone"),
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--run-id")
    location.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _read_config(args.config)
        configured_seeds, configured_rounds = _configured_evaluation(
            json.loads(json.dumps(config))
        )
        task_name, scenario, opponents = _task_settings(args.task, args.opponents)
        if args.mode == "train":
            run_training_mode(
                args, config, task_name, scenario, opponents,
                output_directory=_output_directory,
                checkpoint_name=_checkpoint_name,
                run_session=run_agent_session,
            )
        else:
            run_evaluation_mode(
                args, config, configured_seeds, configured_rounds,
                task_name, scenario, opponents, project_root=PROJECT_ROOT,
                output_directory=_output_directory,
                run_session=run_agent_session,
                run_multi_seed=run_agent_evaluation,
                analyze_runs=analyze_runs,
            )
    except KeyboardInterrupt:
        return 130
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
