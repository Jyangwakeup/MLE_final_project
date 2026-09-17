"""Train or evaluate one agent on a reproducible Bomberman task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
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
from experiments.devices import resolve_device
from agent_code.team_agent.exploration import resolve_exploration_spec
from experiments.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    LoadedSnapshot,
    commit_training_snapshot,
    materialize_learner_checkpoint,
    materialize_migrated_checkpoint,
    materialize_task3_transfer_checkpoint,
)
from experiments.performance_stopping import (
    Task1PerformanceStopping, frozen_score_assessor, load_committed_history,
)
from experiments.training import (
    CompositeTrainingStop, TrainingActionBudget, TrainingEarlyStopping,
    early_stopping_config as _early_stopping_config,
    run_training_mode,
)
from main import world_controller
from agent_code.team_agent.feature_system import ACTIONS, normalize_feature_id
from agent_code.team_agent.rewards import REWARD_VERSION, resolve_reward_spec
from agent_code.team_agent.safety import resolve_safety_spec
from experiments.agent_contracts import resolve_agent_contract
from experiments.navigation_diagnostics import navigation_diagnostic
from agent_code.learning_common.training_spec import resolve_retention_spec


RUNS_ROOT = PROJECT_ROOT / "runs"
TASKS = {
    1: {"name": "coin_navigation", "scenario": "coin-heaven", "opponents": ()},
    2: {"name": "crate_navigation", "scenario": "classic", "opponents": ()},
    3: {
        "name": "weak_opponents",
        "scenario": "classic",
        "opponents": ("peaceful_agent", "coin_collector_agent"),
    },
    4: {
        "name": "full_match",
        "scenario": "classic",
        "opponents": ("rule_based_agent", "rule_based_agent", "rule_based_agent"),
    },
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


def _checkpoint_reward_contract(
    checkpoint: Path, algorithm: str,
) -> dict[str, Any]:
    """Read reward provenance without changing frozen checkpoint behavior."""
    if checkpoint.suffix not in {".pkl", ".pt"}:
        return {"reward_version": None, "reward_spec": None}
    if checkpoint.suffix == ".pkl":
        with checkpoint.open("rb") as file:
            payload = pickle.load(file)
    else:
        try:
            import torch
        except ImportError as exception:
            raise RuntimeError(
                "PyTorch is required to inspect a DQN checkpoint"
            ) from exception
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        return {"reward_version": None, "reward_spec": None}
    embedded_version = payload.get("reward_version")
    embedded_spec = payload.get("reward_spec")
    return {
        "embedded_reward_version": embedded_version,
        "reward_version": payload.get("reward_id", embedded_version),
        "reward_spec": embedded_spec,
    }


def _checkpoint_feature_contract(
    checkpoint: Path, algorithm: str,
) -> dict[str, Any]:
    """Read frozen feature provenance, including the 78-dimensional adapter."""
    if checkpoint.suffix == ".pkl":
        with checkpoint.open("rb") as file:
            payload = pickle.load(file)
    elif checkpoint.suffix == ".pt":
        try:
            import torch
        except ImportError as exception:
            raise RuntimeError(
                "PyTorch is required to inspect a neural checkpoint"
            ) from exception
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        return {"feature_id": None, "feature_schema": None, "runtime_adapter": None}
    if not isinstance(payload, dict):
        return {"feature_id": None, "feature_schema": None, "runtime_adapter": None}
    feature_id = payload.get("feature_id", payload.get("feature_version"))
    schema = payload.get("feature_schema")
    raw_shape = schema.get("vector_shape") if isinstance(schema, dict) else None
    shape = tuple(raw_shape) if raw_shape is not None else ()
    adapter = (
        "continuous-v2-legacy78"
        if feature_id == "continuous-v2" and shape == (78,) else None
    )
    return {
        "feature_id": feature_id,
        "feature_schema": schema,
        "runtime_adapter": adapter,
    }


class ExperimentWorld(BombeRLeWorld):
    """Official world extension that appends one record after each completed round."""

    def __init__(
        self, args: WorldArgs, agents, output: Path, run_id: str,
        environment_seed: int, stage: str = "evaluation",
        replay_policy: str = "none", replay_interval: int = DEFAULT_REPLAY_INTERVAL,
        snapshot_config: dict[str, Any] | None = None,
        navigation_diagnostics: bool = False,
    ):
        self._episodes_path = output / "episodes.jsonl"
        self._timing_path = output / "timing.jsonl"
        self._phase_events_path = output / "phase_events.jsonl"
        self._episodes_path.touch(exist_ok=False)
        self._timing_path.touch(exist_ok=False)
        self._phase_events_path.touch(exist_ok=False)
        self._episodes_file = self._episodes_path.open(
            "a", encoding="utf-8", buffering=1)
        self._timing_file = self._timing_path.open(
            "a", encoding="utf-8", buffering=1)
        self._phase_events_file = self._phase_events_path.open(
            "a", encoding="utf-8", buffering=1)
        self._experiment_run_id = run_id
        self._environment_seed = environment_seed
        self._stage = stage
        self._output = output
        self._replay_policy = replay_policy
        self._replay_interval = replay_interval
        self._snapshot_config = snapshot_config
        self._navigation_diagnostics = navigation_diagnostics
        self._navigation_previous_action: dict[str, str] = {}
        self._navigation_previous_target: dict[str, tuple[int, int] | None] = {}
        self._training_rewards: list[float] = []
        self._training_csv_offset = 0
        self._training_reward_column: int | None = None
        self._death_steps: dict[str, int] = {}
        self._death_causes: dict[str, list[dict[str, str]]] = {}
        self._bomb_owners_exploded_this_step = []
        self._latest_phase_facts: dict[str, dict[str, Any]] = {}
        super().__init__(args, agents)

    def new_round(self) -> None:
        self._death_steps = {}
        self._death_causes = {}
        self._navigation_previous_action = {}
        self._navigation_previous_target = {}
        self._bomb_owners_exploded_this_step = []
        self._latest_phase_facts = {}
        super().new_round()

    def update_bombs(self) -> None:
        self._bomb_owners_exploded_this_step = [
            bomb.owner for bomb in self.bombs if bomb.timer <= 0
        ]
        super().update_bombs()

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
        for owner in self._bomb_owners_exploded_this_step:
            owner.note_stat("bombs_resolved")
            if not owner.dead:
                owner.note_stat("bombs_survived")
        self._bomb_owners_exploded_this_step = []

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
        game_state: dict[str, Any],
    ) -> None:
        record = {
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
            }
        if self._navigation_diagnostics:
            diagnostic, target = navigation_diagnostic(
                game_state,
                action,
                previous_target=self._navigation_previous_target.get(agent.name),
                previous_action=self._navigation_previous_action.get(agent.name),
            )
            record["navigation"] = diagnostic
            self._navigation_previous_target[agent.name] = target
            self._navigation_previous_action[agent.name] = action
        runner = getattr(agent.backend, "runner", None)
        safety = getattr(
            getattr(runner, "fake_self", None), "last_safety_diagnostic", None)
        if isinstance(safety, dict):
            record["safety"] = safety
        fake_self = getattr(runner, "fake_self", None)
        if fake_self is not None and (
            getattr(fake_self, "feature_id", None) == "continuous-phase-v1"
            or self._navigation_diagnostics
        ):
            from agent_code.team_agent.phase import phase_facts_for_owner
            phase = phase_facts_for_owner(fake_self, game_state)
            x, y = game_state["self"][3]
            phase["geometric_edge"] = bool(
                x in {1, game_state["field"].shape[0] - 2}
                or y in {1, game_state["field"].shape[1] - 2})
            record["phase"] = phase
            self._latest_phase_facts[agent.name] = phase
        self._timing_file.write(json.dumps(record, sort_keys=True) + "\n")

    def send_game_events(self) -> None:
        """Persist event/phase pairs before training callbacks consume events."""
        for agent in self.agents:
            phase = self._latest_phase_facts.get(agent.name)
            if phase is None:
                continue
            self._phase_events_file.write(json.dumps({
                "schema_version": "phase-events-v1",
                "run_id": self._experiment_run_id,
                "round_index": int(self.round), "step": int(self.step),
                "agent_name": agent.name, "phase": phase,
                "events": list(agent.events),
            }, sort_keys=True) + "\n")
        super().send_game_events()

    def poll_and_run_agents(self) -> None:
        states: dict[str, dict[str, Any]] = {}
        for agent in self.active_agents:
            state = self.get_state_for_agent(agent)
            states[agent.name] = state
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

                self.logger.debug(
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
                    self.logger.debug(
                        f"Agent <{agent.name}> stayed within acceptable think time."
                    )
                    agent.available_think_time = agent.base_timeout
            else:
                self.logger.debug(
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
                states[agent.name],
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
            actions = self.replay["actions"].get(agent.name, [])
            longest_wait = wait_streak = 0
            longest_ping_pong = ping_pong_streak = 0
            previous_action = None
            opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
            for action in actions:
                wait_streak = wait_streak + 1 if action == "WAIT" else 0
                longest_wait = max(longest_wait, wait_streak)
                if action in opposites and previous_action == opposites[action]:
                    ping_pong_streak = ping_pong_streak + 1 if ping_pong_streak else 2
                else:
                    ping_pong_streak = 1 if action in opposites else 0
                longest_ping_pong = max(longest_ping_pong, ping_pong_streak)
                previous_action = action
            agents.append(
                {
                    "name": agent.name,
                    "score": int(agent.score),
                    "coins": int(statistics.get("coins", 0)),
                    "kills": int(statistics.get("kills", 0)),
                    "suicides": int(statistics.get("suicides", 0)),
                    "crates": int(statistics.get("crates", 0)),
                    "bombs": int(statistics.get("bombs", 0)),
                    "bombs_resolved": int(statistics.get("bombs_resolved", 0)),
                    "bombs_survived": int(statistics.get("bombs_survived", 0)),
                    "invalid": int(statistics.get("invalid", 0)),
                    "survived": not bool(agent.dead),
                    "dead": dead,
                    "death_step": self._death_steps.get(agent.name),
                    "death_causes": death_causes,
                    "killed_by_self": killed_by_self or bool(suicides),
                    "killed_by_opponent": killed_by_opponent,
                    "all_coins": int(statistics.get("coins", 0)) >= 50,
                    "max_steps": int(self.step) >= int(s.MAX_STEPS),
                    "longest_wait_streak": longest_wait,
                    "longest_ping_pong_streak": longest_ping_pong,
                    "long_wait_loop": longest_wait >= 10,
                    "long_ping_pong_loop": longest_ping_pong >= 10,
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

        self._episodes_file.write(
            json.dumps({
                "schema_version": EPISODE_SCHEMA_VERSION,
                "run_id": self._experiment_run_id,
                "round_index": int(self.round),
                "seed": self._environment_seed,
                "environment_seed": self._environment_seed,
                "scenario": self.args.scenario,
                "stage": self._stage,
                "round_steps": int(self.step),
                "exploration_disabled": self._stage.endswith("_evaluate"),
                "agents": agents,
            }, sort_keys=True) + "\n"
        )
        if self._snapshot_config is not None:
            training_path = self._output / "training.csv"
            self._read_new_training_rewards(training_path)
            local_rewards = self._training_rewards
            inherited = self._snapshot_config.get("inherited_rewards", [])
            commit_training_snapshot(
                self._output,
                self._snapshot_config["checkpoint"],
                algorithm=self._snapshot_config["algorithm"],
                task=self._snapshot_config["task"],
                seed=self._snapshot_config["seed"],
                round_index=int(self.round),
                world_rng_state=self.rng.bit_generator.state,
                python_rng_state=random.getstate(),
                numpy_rng_state=np.random.get_state(),
                early_stopping_rewards=[*inherited, *local_rewards],
                source_commit=self._snapshot_config["source_commit"],
                source_hash=self._snapshot_config["source_hash"],
                cumulative_completed_rounds=(
                    self._snapshot_config["parent_cumulative"] + len(local_rewards)
                ),
                early_stopping_config=self._snapshot_config["early_stopping_config"],
                performance_stopping=self._snapshot_config.get("performance_stopping"),
                performance_history=self._snapshot_config.get("performance_history", []),
            )

    def end(self) -> None:
        """Close buffered experiment streams after the official world stops."""
        try:
            super().end()
        finally:
            self._episodes_file.close()
            self._timing_file.close()
            self._phase_events_file.close()

    def _read_new_training_rewards(self, training_path: Path) -> None:
        """Consume only CSV rows appended since the previous round."""
        if not training_path.is_file():
            return
        with training_path.open(newline="", encoding="utf-8") as file:
            if self._training_reward_column is None:
                header = next(csv.reader([file.readline()]))
                try:
                    self._training_reward_column = header.index("reward")
                except ValueError as exception:
                    raise ValueError(
                        "training.csv is missing the reward column") from exception
                self._training_csv_offset = file.tell()
            else:
                file.seek(self._training_csv_offset)
            rows = file.readlines()
            self._training_csv_offset = file.tell()
        for row in csv.reader(rows):
            self._training_rewards.append(
                float(row[self._training_reward_column]))


def _source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_hash() -> str:
    """Hash runtime Python/config sources, including uncommitted files."""
    digest = hashlib.sha256()
    roots = [PROJECT_ROOT / "agent_code", PROJECT_ROOT / "experiments"]
    paths = [PROJECT_ROOT / name for name in ("agents.py", "environment.py", "items.py", "settings.py")]
    for root in roots:
        paths.extend(root.rglob("*.py"))
        paths.extend(root.rglob("*.json"))
    for path in sorted({item for item in paths if item.is_file()}):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
        "config_source_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
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


def _last_training_integer(path: Path, field: str) -> int | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not rows[-1].get(field):
        return None
    return int(rows[-1][field])


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


def _task_settings(
    task: int, opponents: Sequence[str] | None, *,
    task3_curriculum_stage: str | None = None,
):
    task_config = TASKS[task]
    resolved_opponents = tuple(task_config["opponents"] if opponents is None else opponents)
    if task in {1, 2} and resolved_opponents:
        raise ValueError(f"Task {task} does not use opponents")
    if task3_curriculum_stage is not None:
        if task != 3:
            raise ValueError("Task 3 opponent curriculum is only valid for Task 3")
        stages = {
            "stationary": ("stationary_target_agent", "stationary_target_agent"),
            "moving": ("no_bomb_random_agent", "no_bomb_random_agent"),
            "standard": TASKS[3]["opponents"],
        }
        resolved_opponents = tuple(stages[task3_curriculum_stage])
    elif task == 3 and opponents is not None:
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
    return resolve_agent_contract(agent).checkpoint_name


def _algorithm_name(agent: str) -> str:
    return resolve_agent_contract(agent).algorithm


def _close_output_log_handlers(output: Path) -> None:
    """Release per-run files so tests and callers can move completed runs."""
    root = output.resolve()
    loggers = [logging.getLogger()]
    loggers.extend(
        item for item in logging.Logger.manager.loggerDict.values()
        if isinstance(item, logging.Logger)
    )
    for logger in loggers:
        for handler in list(logger.handlers):
            filename = getattr(handler, "baseFilename", None)
            if filename is None:
                continue
            try:
                Path(filename).resolve().relative_to(root)
            except ValueError:
                continue
            logger.removeHandler(handler)
            handler.close()


def run_agent_session(
    config_path: Path,
    mode: str,
    seed: int,
    output: Path,
    agent: str,
    opponents: Sequence[str],
    scenario: str,
    n_rounds: int,
    checkpoint: Path | None,
    task_name: str,
    replay_policy: str = "none",
    replay_interval: int = DEFAULT_REPLAY_INTERVAL,
    early_stopping_config: dict[str, Any] | None = None,
    *,
    resume_snapshot: LoadedSnapshot | None = None,
    resume_kind: str | None = None,
    parent_metadata: dict[str, Any] | None = None,
    init_checkpoint: Path | None = None,
    device_info: dict[str, Any] | None = None,
    action_budget_config: dict[str, Any] | None = None,
    safe_exploration: bool = False,
    safety_spec: dict[str, Any] | None = None,
    n_step: int = 1,
    retention_spec: dict[str, Any] | None = None,
    adaptation_triggers: Sequence[str] = (),
    feature_id_override: str | None = None,
    reward_id_override: str | None = None,
    show_progress: bool = True,
    progress_leave: bool = True,
    performance_stopping: dict[str, Any] | None = None,
    migration: bool = False,
    task3_transfer: bool = False,
    distillation_path: Path | None = None,
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
    checkpoint = None if checkpoint is None else checkpoint.resolve()
    init_checkpoint = (
        None if init_checkpoint is None else init_checkpoint.expanduser().resolve())
    if init_checkpoint is not None and not training:
        raise ValueError("Warm-start checkpoints are only valid in training mode")
    if init_checkpoint is not None and resume_snapshot is not None:
        raise ValueError("Warm-start and resume cannot be used together")
    if init_checkpoint is not None and not init_checkpoint.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {init_checkpoint}")
    expanded = _read_config(config_path)
    algorithm = _algorithm_name(agent)
    training_config = expanded.get("training", {})
    if not isinstance(training_config, dict):
        raise ValueError("config.training must be an object")
    exploration_spec = resolve_exploration_spec(training_config.get("exploration"))
    training_config["exploration"] = exploration_spec
    if not isinstance(safe_exploration, bool):
        raise ValueError("safe_exploration must be a boolean")
    configured_safety = expanded.get("safety")
    if configured_safety is not None and "safe_exploration" in training_config:
        raise ValueError("config.safety conflicts with training.safe_exploration")
    if safety_spec is None:
        safety_spec = resolve_safety_spec(
            configured_safety,
            legacy_safe_exploration=(
                safe_exploration if configured_safety is None and training else None),
        )
    else:
        safety_spec = resolve_safety_spec(safety_spec)
        if configured_safety is not None and resolve_safety_spec(
            configured_safety) != safety_spec:
            raise ValueError("resolved safety specification conflicts with config.safety")
    safe_exploration = bool(
        training and safety_spec["mode"] in {"exploration", "all"})
    if n_step not in {1, 4}:
        raise ValueError("n_step must be 1 or 4")
    retention_spec = resolve_retention_spec(retention_spec)
    adaptation_triggers = tuple(adaptation_triggers)
    allowed_adaptation_triggers = {
        "suicide", "retention", "q_capability", "dqn_capability",
    }
    unknown_triggers = sorted(set(adaptation_triggers) - allowed_adaptation_triggers)
    if unknown_triggers:
        raise ValueError(
            "Unknown adaptation trigger(s): " + ", ".join(unknown_triggers)
        )
    if len(set(adaptation_triggers)) != len(adaptation_triggers):
        raise ValueError("Adaptation triggers must be unique")
    action_budget_config = action_budget_config or {
        "target_stage_action_steps": None, "min_rounds": 1,
    }
    training_config["safe_exploration"] = safe_exploration
    training_config["n_step"] = n_step
    training_config["retention"] = retention_spec
    training_config["action_budget"] = action_budget_config
    training_config["performance_stopping"] = performance_stopping
    expanded["training"] = training_config
    expanded["safety"] = safety_spec
    expanded["adaptation_triggers"] = list(adaptation_triggers)
    evaluation_config = expanded.get("evaluation", {})
    if not isinstance(evaluation_config, dict):
        raise ValueError("config.evaluation must be an object")
    navigation_diagnostics = evaluation_config.get(
        "navigation_diagnostics", False)
    if not isinstance(navigation_diagnostics, bool):
        raise ValueError("config.evaluation.navigation_diagnostics must be a boolean")
    navigation_diagnostics = bool(navigation_diagnostics and not training)
    agent_seed = int(seed)
    if device_info is None:
        section = expanded.get("training" if training else "evaluation", {})
        if not isinstance(section, dict):
            raise ValueError("Device config section must be an object")
        requested_device = section.get("device", "auto" if training else "cpu")
        device_info = resolve_device(algorithm, mode, requested_device)
    configured_id = feature_id_override or expanded.get("feature_id")
    configured_legacy = (
        None if feature_id_override is not None else expanded.get("feature_version"))
    requested_feature_id = (
        None if configured_id is None and configured_legacy is None
        else normalize_feature_id(configured_id, configured_legacy)
    )
    contract = resolve_agent_contract(agent, requested_feature_id)
    algorithm = contract.algorithm
    configured_algorithm = expanded.get("algorithm")
    if configured_algorithm not in (None, algorithm):
        raise ValueError("Configured algorithm does not match the selected agent")
    configured_reward_id = reward_id_override or expanded.get("reward_id")
    legacy_reward_id = (
        None if reward_id_override is not None else expanded.get("reward_version"))
    if (
        configured_reward_id is not None
        and legacy_reward_id is not None
        and configured_reward_id != legacy_reward_id
    ):
        raise ValueError("Configured reward_id conflicts with reward_version")
    reward_version = configured_reward_id or legacy_reward_id or REWARD_VERSION
    resolved_rewards = resolve_reward_spec(reward_version)
    if resume_snapshot is not None and not training:
        raise ValueError("Resume snapshots are only valid in training mode")
    if (resume_snapshot is None) != (resume_kind is None):
        raise ValueError("Resume snapshot and resume kind must be provided together")
    if not training and checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    checkpoint_reward_contract = (
        None if training or checkpoint is None
        else _checkpoint_reward_contract(checkpoint, algorithm)
    )
    checkpoint_feature_contract = (
        None if training or checkpoint is None
        else _checkpoint_feature_contract(checkpoint, algorithm)
    )
    output.mkdir(parents=True, exist_ok=False)
    if training:
        assert checkpoint is not None
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    transfer_contract = None
    if resume_snapshot is not None:
        if task3_transfer:
            if distillation_path is None:
                raise ValueError("Task 3 transfer requires a distillation dataset")
            transfer_contract = materialize_task3_transfer_checkpoint(
                resume_snapshot, checkpoint,
                child_contract={
                    "algorithm": algorithm, "seed": seed, "task": task_name,
                    "feature_id": contract.feature_id,
                    "feature_schema": contract.feature_schema,
                    "reward_id": reward_version, "reward_spec": resolved_rewards,
                    "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                    "actions": list(ACTIONS),
                    "training_device_name": device_info["name"],
                    "training_device_type": device_info["type"],
                    "agent_seed": seed, "exploration_spec": exploration_spec,
                    "safe_exploration": safe_exploration,
                    "safety_spec": safety_spec, "n_step": n_step,
                    "retention_spec": retention_spec,
                    "training_budget": action_budget_config,
                    "network_spec": contract.network_spec,
                    "hyperparameters": contract.hyperparameters,
                },
                distillation_path=distillation_path,
            )
        elif migration:
            materialize_migrated_checkpoint(
                resume_snapshot, checkpoint, training_budget=action_budget_config)
        else:
            materialize_learner_checkpoint(resume_snapshot, checkpoint)

    seeds = _training_seeds(seed) if training else _evaluation_seeds(seed)
    metadata = _initial_metadata(config_path, mode, output.name, seeds)
    source_commit = metadata["source_commit"]
    source_hash = _source_hash()
    metadata["source_hash"] = source_hash
    expanded["algorithm"] = algorithm
    expanded["feature_id"] = contract.feature_id
    expanded["feature_version"] = (
        "v1" if contract.feature_id == "discrete-v1" else None)
    expanded["reward_id"] = reward_version
    expanded["reward_version"] = reward_version
    expanded["resolved_rewards"] = resolved_rewards
    expanded["seed"] = seed
    expanded["execution"] = {
        "agent": agent,
        "allow_bomb": task_name != "coin_navigation",
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "init_from_checkpoint": (
            None if init_checkpoint is None else str(init_checkpoint)),
        "n_rounds": n_rounds,
        "opponents": list(opponents),
        "replay_interval": replay_interval,
        "replay_policy": replay_policy,
        "scenario": scenario,
        "task": task_name,
        "device": device_info,
        "curriculum_action_mask": {"BOMB": task_name != "coin_navigation"},
        "checkpoint_reward_contract": checkpoint_reward_contract,
        "checkpoint_feature_contract": checkpoint_feature_contract,
        "navigation_diagnostics": navigation_diagnostics,
    }
    metadata["expanded_config"] = expanded
    metadata["agent"] = agent
    metadata["algorithm"] = algorithm
    metadata["agent_seed"] = agent_seed
    metadata["checkpoint_schema"] = CHECKPOINT_SCHEMA_VERSION
    metadata["exploration_spec"] = exploration_spec
    metadata["safe_exploration"] = bool(safe_exploration and training)
    metadata["safety_spec"] = safety_spec
    metadata["n_step"] = n_step
    metadata["retention_spec"] = retention_spec
    metadata["training_budget"] = action_budget_config
    metadata["performance_stopping"] = performance_stopping
    metadata["transfer_contract"] = transfer_contract
    metadata["adaptation_triggers"] = list(adaptation_triggers)
    metadata["feature_id"] = contract.feature_id
    metadata["feature_schema"] = contract.feature_schema
    metadata["feature_version"] = (
        "v1" if contract.feature_id == "discrete-v1" else None)
    metadata["reward_id"] = reward_version
    metadata["reward_version"] = reward_version
    metadata["action_order"] = list(ACTIONS)
    metadata["checkpoint"] = None if checkpoint is None else str(checkpoint)
    metadata["network_spec"] = contract.network_spec
    metadata["hyperparameters"] = contract.hyperparameters
    metadata["curriculum_action_mask"] = {
        "BOMB": task_name != "coin_navigation"
    }
    metadata["rewards"] = resolved_rewards
    metadata["checkpoint_reward_contract"] = checkpoint_reward_contract
    metadata["checkpoint_feature_contract"] = checkpoint_feature_contract
    metadata["task"] = task_name
    metadata["device"] = device_info
    metadata["exploration_disabled"] = not training
    parent_cumulative = 0
    if resume_snapshot is not None:
        parent_cumulative = int(
            resume_snapshot.runner_state["cumulative_completed_rounds"]
        )
    elif parent_metadata is not None:
        termination = parent_metadata.get("termination") or {}
        parent_cumulative = int(termination.get(
            "cumulative_completed_rounds", termination.get("completed_rounds", 0)
        ))
    metadata["lineage"] = (
        {
            "kind": "warm_start",
            "source_checkpoint": str(init_checkpoint),
            "inherited": ["policy_weights"],
            "reset": [
                "target_network", "optimizer", "replay", "epsilon",
                "agent_rng", "early_stopping", "round_state",
            ],
        }
        if init_checkpoint is not None else None
    ) if resume_snapshot is None else {
        "fallback_reason": resume_snapshot.fallback_reason,
        "fallback_lost_rounds": resume_snapshot.lost_rounds,
        "parent_generation": resume_snapshot.generation,
        "parent_generation_hash": resume_snapshot.generation_hash,
        "parent_run": str(resume_snapshot.run_directory),
        "parent_source_commit": resume_snapshot.source_commit,
        "parent_source_hash": resume_snapshot.source_hash,
        "resume_kind": resume_kind,
        "schema_migration": (
            {
                "from": "training-resume-v6",
                "to": CHECKPOINT_SCHEMA_VERSION,
                "history_initialized": "empty",
            }
            if migration else None
        ),
        "task3_transfer": transfer_contract,
    }
    metadata["config_sha256"] = hashlib.sha256(
        json.dumps(expanded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata_path = output / "metadata.json"
    _write_json(metadata_path, metadata)

    previous = {
        name: os.environ.get(name)
        for name in (
            "BOMBERMAN_CHECKPOINT", "BOMBERMAN_INIT_CHECKPOINT",
            "BOMBERMAN_CONFIG", "BOMBERMAN_RUN_DIR",
            "BOMBERMAN_RUN_ID", "BOMBERMAN_TRAINING_TASK",
            "BOMBERMAN_ALLOW_BOMB", "BOMBERMAN_FEATURE_ID",
            "BOMBERMAN_REWARD_ID", "BOMBERMAN_REWARD_VERSION",
            "BOMBERMAN_TORCH_DEVICE", "BOMBERMAN_AGENT_SEED",
            "BOMBERMAN_EXPLORATION_SPEC",
            "BOMBERMAN_SAFE_EXPLORATION", "BOMBERMAN_N_STEP",
            "BOMBERMAN_SAFETY_SPEC",
            "BOMBERMAN_RETENTION_SPEC", "BOMBERMAN_TRAINING_BUDGET",
            "BOMBERMAN_CAPTURE_ENVIRONMENT_SEED", "BOMBERMAN_CAPTURE_TASK_ID",
        )
    }
    try:
        (output / "logs").mkdir()
        if checkpoint is None:
            os.environ.pop("BOMBERMAN_CHECKPOINT", None)
        else:
            os.environ["BOMBERMAN_CHECKPOINT"] = str(checkpoint)
        if init_checkpoint is None:
            os.environ.pop("BOMBERMAN_INIT_CHECKPOINT", None)
        else:
            os.environ["BOMBERMAN_INIT_CHECKPOINT"] = str(init_checkpoint)
        os.environ["BOMBERMAN_CONFIG"] = str(config_path.resolve())
        os.environ["BOMBERMAN_RUN_DIR"] = str(output.resolve())
        os.environ["BOMBERMAN_RUN_ID"] = output.name
        os.environ["BOMBERMAN_TRAINING_TASK"] = task_name
        os.environ["BOMBERMAN_FEATURE_ID"] = contract.feature_id
        os.environ["BOMBERMAN_REWARD_ID"] = reward_version
        os.environ["BOMBERMAN_REWARD_VERSION"] = reward_version
        os.environ["BOMBERMAN_TORCH_DEVICE"] = device_info["actual"]
        os.environ["BOMBERMAN_AGENT_SEED"] = str(agent_seed)
        os.environ["BOMBERMAN_EXPLORATION_SPEC"] = json.dumps(
            exploration_spec, sort_keys=True, separators=(",", ":")
        )
        os.environ["BOMBERMAN_SAFE_EXPLORATION"] = (
            "true" if training and safe_exploration else "false")
        os.environ["BOMBERMAN_SAFETY_SPEC"] = json.dumps(
            safety_spec, sort_keys=True, separators=(",", ":"))
        os.environ["BOMBERMAN_N_STEP"] = str(n_step)
        os.environ["BOMBERMAN_RETENTION_SPEC"] = json.dumps(
            retention_spec, sort_keys=True, separators=(",", ":"))
        os.environ["BOMBERMAN_TRAINING_BUDGET"] = json.dumps(
            action_budget_config, sort_keys=True, separators=(",", ":"))
        if (os.getenv("BOMBERMAN_DISTILLATION_CAPTURE")
                or os.getenv("BOMBERMAN_CNN_TEACHER_CAPTURE")):
            os.environ["BOMBERMAN_CAPTURE_ENVIRONMENT_SEED"] = str(
                seeds["environment_seed"])
            os.environ["BOMBERMAN_CAPTURE_TASK_ID"] = str(
                {"coin_navigation": 1, "crate_navigation": 2,
                 "weak_opponents": 3, "full_match": 4}[task_name])
        # Task 1 isolates coin navigation. Keep its action space free of bombs
        # for both training and frozen evaluation so agents are compared under
        # the same curriculum constraint.
        os.environ["BOMBERMAN_ALLOW_BOMB"] = (
            "false" if task_name == "coin_navigation" else "true"
        )
        _seed_official_rng(seeds["official_opponent_seed"])
        inherited_rewards = (
            resume_snapshot.runner_state.get("early_stopping_rewards", [])
            if resume_snapshot is not None
            and resume_kind in {"same_task", "v6_migration"} else []
        )
        snapshot_performance_history = (
            resume_snapshot.runner_state.get("performance_history", [])
            if resume_snapshot is not None and resume_kind == "same_task" else []
        )
        performance_history = (
            load_committed_history(
                resume_snapshot.run_directory,
                snapshot_performance_history,
                performance_stopping,
            )
            if performance_stopping is not None
            and resume_snapshot is not None and resume_kind == "same_task"
            else []
        )
        world = ExperimentWorld(
            _world_args(output, scenario, seeds["environment_seed"], output.name),
            specs,
            output,
            output.name,
            seeds["environment_seed"],
            stage=f"{task_name}_{mode}",
            replay_policy=replay_policy,
            replay_interval=replay_interval,
            # The path CNN owns an agent-local feature/checkpoint contract.
            # It deliberately does not use shared exact-resume snapshots.
            snapshot_config=(
                {
                    "algorithm": algorithm,
                    "checkpoint": checkpoint,
                    "inherited_rewards": inherited_rewards,
                    "early_stopping_config": early_stopping_config,
                    "parent_cumulative": parent_cumulative,
                    "seed": seed,
                    "source_commit": source_commit,
                    "source_hash": source_hash,
                    "task": task_name,
                    "performance_stopping": performance_stopping,
                    "performance_history": performance_history,
                }
                if training and agent not in {
                    "cnn_path_double_dqn_agent", "cnn_distilled_double_dqn_agent",
                } else None
            ),
            navigation_diagnostics=navigation_diagnostics,
        )
        if resume_snapshot is not None and resume_kind in {"same_task", "v6_migration"}:
            runner_state = resume_snapshot.runner_state
            world.rng.bit_generator.state = runner_state["world_rng_state"]
            world.round = resume_snapshot.round_index
            random.setstate(runner_state["python_rng_state"])
            np.random.set_state(runner_state["numpy_rng_state"])
        else:
            _seed_official_rng(seeds["official_opponent_seed"])
        early_stopping = (
            TrainingEarlyStopping(
                output / "training.csv", early_stopping_config, inherited_rewards
            )
            if training and early_stopping_config is not None else None
        )
        action_budget = (
            TrainingActionBudget(
                output / "training.csv",
                action_budget_config["target_stage_action_steps"],
                action_budget_config["min_rounds"],
            )
            if training and action_budget_config["target_stage_action_steps"] is not None
            else None
        )
        performance_stop = None
        if training and performance_stopping is not None:
            assessor = frozen_score_assessor(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                run_directory=output,
                agent=agent,
                checkpoint=checkpoint,
                seeds=performance_stopping["evaluation_seeds"],
                rounds_per_seed=performance_stopping["rounds_per_seed"],
            )
            performance_stop = Task1PerformanceStopping(
                performance_stopping,
                run_directory=output,
                base_cumulative_rounds=parent_cumulative,
                assessor=assessor,
                history=performance_history,
            )
            world._snapshot_config["performance_history"] = performance_stop.history
            if migration:
                performance_stop.assess(
                    parent_cumulative,
                    resume_snapshot.generation,
                    resume_snapshot.generation_hash,
                )
        stopping = CompositeTrainingStop(
            early_stopping, action_budget, performance_stop)
        completed_rounds = world_controller(
            world, n_rounds, gui=None, every_step=False, turn_based=False,
            make_video=False, update_interval=0.0, show_progress=show_progress,
            progress_leave=progress_leave,
            stop_condition=stopping if stopping.conditions else None,
        )
        if training and (checkpoint is None or not checkpoint.is_file()):
            raise RuntimeError(f"Training did not write checkpoint: {checkpoint}")
        if training:
            analyze_training(output)
            if algorithm == "dqn":
                import torch
                checkpoint_metadata = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )
                metadata["device"]["peak_memory_bytes"] = int(
                    checkpoint_metadata.get("peak_cuda_memory_bytes", 0)
                )
        metadata["ended_at"] = _utc_now()
        final_stage_steps = _last_training_integer(
            output / "training.csv", "stage_action_steps") if training else None
        budget_reached = (
            action_budget_config["target_stage_action_steps"] is None
            or (
                final_stage_steps is not None
                and final_stage_steps >= action_budget_config["target_stage_action_steps"]
                and completed_rounds >= action_budget_config["min_rounds"]
            )
        )
        metadata["termination"] = {
            "completed_rounds": completed_rounds,
            "cumulative_completed_rounds": parent_cumulative + completed_rounds,
            "early_stopping": None if early_stopping is None else early_stopping.result,
            "local_completed_rounds": completed_rounds,
            "requested_rounds": n_rounds,
            "stage_action_steps": final_stage_steps,
            "action_budget": action_budget_config,
            "action_budget_reached": budget_reached,
            "performance_stopping": (
                None if performance_stop is None else {
                    "result": performance_stop.result,
                    "history": performance_stop.history,
                    "converged": bool(
                        performance_stop.result
                        and performance_stop.result["reason"] == "task1_score_converged"),
                }
            ),
            "completion_reason": (
                performance_stop.result["reason"]
                if performance_stop is not None and performance_stop.result is not None
                else "stage_action_target_reached"
                if action_budget is not None and action_budget.result is not None
                else "round_cap_reached"
            ),
        }
        metadata["status"] = (
            "not_converged"
            if performance_stop is not None and (
                performance_stop.result is None
                or performance_stop.result["reason"] == "task1_score_not_converged")
            else "early_stopped" if early_stopping is not None and early_stopping.result
            else "completed"
        )
        _write_json(metadata_path, metadata)
        return output
    except BaseException as exception:
        metadata["ended_at"] = _utc_now()
        metadata["error"] = {"message": str(exception), "type": type(exception).__name__}
        metadata["status"] = (
            "interrupted" if isinstance(exception, KeyboardInterrupt) else "error"
        )
        _write_json(metadata_path, metadata)
        raise
    finally:
        _close_output_log_handlers(output)
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
    checkpoint: Path | None,
    task_name: str,
    replay_policy: str = "all",
    replay_interval: int = DEFAULT_REPLAY_INTERVAL,
    *,
    device_info: dict[str, Any] | None = None,
    feature_id_override: str | None = None,
    reward_id_override: str | None = None,
) -> Path:
    if device_info is None:
        device_info = resolve_device(_algorithm_name(agent), "evaluate", "cpu")
    return run_multi_seed_evaluation(
        config_path, seeds, n_rounds, run_id_prefix, agent, opponents,
        scenario, checkpoint, task_name, replay_policy, replay_interval,
        runs_root=RUNS_ROOT, project_root=PROJECT_ROOT,
        run_session=run_agent_session, analyze_runs=analyze_runs,
        write_json=_write_json,
        device_info=device_info,
        feature_id_override=feature_id_override,
        reward_id_override=reward_id_override,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("train", "evaluate"))
    parser.add_argument("--task", required=True, type=int, choices=TASKS)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--feature-id")
    parser.add_argument("--reward-id")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--n-rounds", type=int)
    parser.add_argument(
        "--target-stage-action-steps", type=int,
        help="Stop training after this cumulative action count within the Task",
    )
    parser.add_argument(
        "--min-rounds", type=int,
        help="Minimum local rounds before an action target may stop training",
    )
    parser.add_argument(
        "--adaptation-trigger", action="append", default=[],
        choices=("suicide", "retention", "q_capability", "dqn_capability"),
        help="Round-3 failure signal recorded in metadata; repeat as needed",
    )
    parser.add_argument("--opponents", nargs="*")
    parser.add_argument(
        "--task3-opponent-curriculum",
        choices=("stationary", "moving", "standard"),
        help="Preregistered Task 3 combat-only fallback stage",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"),
        help="DQN training device; evaluation is always resolved to CPU",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-from", type=Path,
        help="Parent v7 training run used for exact or curriculum resume",
    )
    resume_group.add_argument(
        "--migrate-resume-from", type=Path,
        help="Complete v6 Task 1 run migrated explicitly into a v7 child",
    )
    resume_group.add_argument(
        "--transfer-task3-from", type=Path,
        help="Complete v7 Task 2 run explicitly transferred into a v8 phase agent",
    )
    parser.add_argument(
        "--distillation-dataset", type=Path,
        help="Optional Task 1/2 teacher dataset; otherwise use the parent-derived path",
    )
    parser.add_argument(
        "--init-from-checkpoint", type=Path,
        help=("Development-only neural warm start: load policy weights while "
              "resetting optimizer, replay, epsilon, RNG, and round state"),
    )
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
        agent_contract = resolve_agent_contract(args.agent, args.feature_id)
        configured_seeds, configured_rounds = _configured_evaluation(
            json.loads(json.dumps(config))
        )
        if args.task3_opponent_curriculum is not None and args.mode != "train":
            raise ValueError("Task 3 opponent curriculum is training-only")
        task_name, scenario, opponents = _task_settings(
            args.task, args.opponents,
            task3_curriculum_stage=args.task3_opponent_curriculum)
        if args.mode == "train":
            if not agent_contract.trainable:
                raise ValueError(f"Agent {args.agent!r} does not support training")
            run_training_mode(
                args, config, task_name, scenario, opponents,
                output_directory=_output_directory,
                checkpoint_name=_checkpoint_name,
                run_session=run_agent_session,
                source_commit=_source_commit(),
                source_hash=_source_hash(),
            )
        else:
            if (args.resume_from is not None or args.migrate_resume_from is not None
                    or args.transfer_task3_from is not None):
                raise ValueError("resume and migration options are only valid with --mode train")
            if args.adaptation_trigger:
                raise ValueError(
                    "--adaptation-trigger is only valid with --mode train")
            if args.init_from_checkpoint is not None:
                raise ValueError(
                    "--init-from-checkpoint is only valid with --mode train")
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
