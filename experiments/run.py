"""Run a reproducible official-agent baseline experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
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
from main import world_controller


RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_BASELINE_AGENTS = (
    "rule_based_agent",
    "random_agent",
    "peaceful_agent",
    "coin_collector_agent",
)
OFFICIAL_BASELINE_AGENTS = frozenset(DEFAULT_BASELINE_AGENTS)
EPISODE_SCHEMA_VERSION = "episode-v1"


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

    def __init__(self, args: WorldArgs, agents, output: Path, run_id: str, seed: int):
        self._episodes_path = output / "episodes.jsonl"
        self._episodes_path.touch(exist_ok=False)
        self._experiment_run_id = run_id
        self._environment_seed = seed
        super().__init__(args, agents)

    def end_round(self) -> None:
        super().end_round()
        agents = []
        for agent in self.agents:
            statistics = agent.statistics
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
                    "dead": bool(agent.dead),
                }
            )

        _append_json_line(
            self._episodes_path,
            {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "run_id": self._experiment_run_id,
                "round_index": int(self.round),
                "seed": self._environment_seed,
                "scenario": self.args.scenario,
                "stage": "official_baseline",
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


def _initial_metadata(config_path: Path, mode: str, run_id: str, seed: int) -> dict[str, Any]:
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
        "seed": seed,
        "source_commit": _source_commit(),
        "started_at": _utc_now(),
        "status": "running",
    }


def _read_config(config_path: Path) -> dict[str, Any]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Experiment config must contain a JSON object")
    return value


def _expanded_config(config: dict[str, Any], seed: int) -> tuple[dict[str, Any], tuple[str, ...], str, int, int]:
    expanded = json.loads(json.dumps(config))
    baseline = expanded.setdefault("baseline", {})
    if not isinstance(baseline, dict):
        raise ValueError("config.baseline must be an object")

    agents = baseline.setdefault("agents", list(DEFAULT_BASELINE_AGENTS))
    scenario = baseline.setdefault("scenario", "classic")
    n_rounds = baseline.setdefault("n_rounds", 5)
    opponent_seed = baseline.setdefault("opponent_seed", seed + 100000)
    expanded["seed"] = seed

    if not isinstance(agents, list) or not all(isinstance(agent, str) for agent in agents):
        raise ValueError("config.baseline.agents must be a list of agent names")
    if not 1 <= len(agents) <= s.MAX_AGENTS:
        raise ValueError(f"config.baseline.agents must contain 1 to {s.MAX_AGENTS} agents")
    unsupported = sorted(set(agents) - OFFICIAL_BASELINE_AGENTS)
    if unsupported:
        raise ValueError(f"Only official baseline agents are supported: {', '.join(unsupported)}")
    if scenario not in s.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if isinstance(n_rounds, bool) or not isinstance(n_rounds, int) or n_rounds < 1:
        raise ValueError("config.baseline.n_rounds must be a positive integer")
    if isinstance(opponent_seed, bool) or not isinstance(opponent_seed, int):
        raise ValueError("config.baseline.opponent_seed must be an integer")

    return expanded, tuple(agents), scenario, n_rounds, opponent_seed


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


def run_baseline(config_path: Path, mode: str, seed: int, run_id: str | None, output: Path | None) -> Path:
    """Create one run directory and execute an official-agent baseline."""
    output_directory, resolved_run_id = _output_directory(run_id, output)
    output_directory.mkdir(parents=True, exist_ok=False)
    metadata_path = output_directory / "metadata.json"
    metadata = _initial_metadata(config_path, mode, resolved_run_id, seed)
    _write_json(metadata_path, metadata)

    try:
        config = _read_config(config_path)
        expanded, agents, scenario, n_rounds, opponent_seed = _expanded_config(config, seed)
        metadata["expanded_config"] = expanded
        metadata["config_sha256"] = hashlib.sha256(
            json.dumps(expanded, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _write_json(metadata_path, metadata)

        if mode != "baseline":
            raise ValueError("Only --mode baseline is supported in C2")

        (output_directory / "logs").mkdir()
        world = ExperimentWorld(
            _world_args(output_directory, scenario, seed, resolved_run_id),
            [(agent, False) for agent in agents],
            output_directory,
            resolved_run_id,
            seed,
        )
        random.seed(opponent_seed)
        np.random.seed(opponent_seed)
        world_controller(
            world,
            n_rounds,
            gui=None,
            every_step=False,
            turn_based=False,
            make_video=False,
            update_interval=0.0,
        )
        if not (output_directory / "official_stats.json").is_file():
            raise RuntimeError("Official framework did not write official_stats.json")

        metadata["ended_at"] = _utc_now()
        metadata["status"] = "completed"
        _write_json(metadata_path, metadata)
        return output_directory
    except BaseException as exception:
        metadata["ended_at"] = _utc_now()
        metadata["error"] = {
            "message": str(exception),
            "type": type(exception).__name__,
        }
        metadata["status"] = "failed"
        _write_json(metadata_path, metadata)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", required=True, type=int)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--run-id")
    location.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_baseline(args.config, args.mode, args.seed, args.run_id, args.output)
    except KeyboardInterrupt:
        return 130
    except Exception as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
