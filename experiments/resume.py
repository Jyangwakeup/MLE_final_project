"""Crash-safe, versioned training snapshots for curriculum runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import uuid
from typing import Any

import numpy as np

from agent_code.team_agent.feature_system import (
    feature_schema_contract,
    normalize_feature_id,
    validate_checkpoint_feature_contract,
)


CHECKPOINT_SCHEMA_VERSION = "training-resume-v8"
MIGRATABLE_CHECKPOINT_SCHEMA = "training-resume-v6"
TASK3_TRANSFER_PARENT_SCHEMA = "training-resume-v7"
TASK_ORDER = ("coin_navigation", "crate_navigation", "weak_opponents", "full_match")
RETAINED_GENERATIONS = 2
TABLE_ALGORITHMS = frozenset(("q_learning", "double_q_learning"))
TORCH_ALGORITHMS = frozenset((
    "dqn", "double_dqn", "cnn_double_dqn", "hybrid_dueling_double_dqn", "rainbow_lite",
))
PICKLE_ALGORITHMS = frozenset(("expected_sarsa_lambda", "double_q_lambda"))


@dataclass(frozen=True)
class LoadedSnapshot:
    run_directory: Path
    generation: str
    generation_hash: str
    algorithm: str
    task: str
    seed: int
    round_index: int
    source_commit: str | None
    source_hash: str | None
    runner_state: dict[str, Any]
    learner_payload: dict[str, Any] | None
    learner_path: Path | None
    source_hash_scope: str | None = None
    lost_rounds: int = 0
    fallback_reason: str | None = None

    @property
    def contract(self) -> dict[str, Any]:
        metadata = self.runner_state["contract"]
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "task": self.task,
            "feature_id": metadata.get("feature_id", "discrete-v1"),
            "feature_schema": metadata.get("feature_schema"),
            "feature_version": metadata.get("feature_version"),
            "reward_id": metadata.get("reward_id", metadata["reward_version"]),
            "reward_version": metadata["reward_version"],
            "reward_spec": metadata["reward_spec"],
            "checkpoint_schema": metadata["checkpoint_schema"],
            "training_device_name": metadata["training_device_name"],
            "training_device_type": metadata["training_device_type"],
            "actions": metadata["actions"],
            "agent_seed": metadata["agent_seed"],
            "exploration_spec": metadata["exploration_spec"],
            "source_commit": self.source_commit,
            "source_hash": self.source_hash,
            "source_hash_scope": self.source_hash_scope,
            "network_spec": metadata.get("network_spec"),
            "hyperparameters": metadata.get("hyperparameters", {}),
            "safe_exploration": metadata.get("safe_exploration", False),
            "safety_spec": metadata.get("safety_spec"),
            "n_step": metadata.get("n_step", 1),
            "retention_spec": metadata.get("retention_spec", {}),
            "training_budget": metadata.get("training_budget", {}),
            "performance_stopping": metadata.get("performance_stopping"),
            "transfer_contract": metadata.get("transfer_contract"),
        }


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "__tuple__" in value:
            return tuple(_decode(item) for item in value["__tuple__"])
        if "__ndarray__" in value:
            array = np.asarray(value["__ndarray__"], dtype=value["dtype"])
            return array.reshape(value["shape"])
        return {key: _decode(item) for key, item in value.items()}
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_encode(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    """Snapshot an atomically replaced checkpoint without duplicating its bytes."""
    try:
        os.link(source, destination)
    except OSError:
        # Cross-device filesystems and platforms without hard links retain the
        # previous portable behaviour.
        shutil.copy2(source, destination)


def _read_checkpoint_metadata(checkpoint: Path, algorithm: str) -> dict[str, Any]:
    if algorithm in TABLE_ALGORITHMS:
        with checkpoint.open("rb") as file:
            return pickle.load(file)
    if algorithm in PICKLE_ALGORITHMS:
        with checkpoint.open("rb") as file:
            return pickle.load(file)
    if algorithm in TORCH_ALGORITHMS:
        try:
            import torch
        except ImportError as exception:
            raise RuntimeError("PyTorch is required to snapshot a DQN run") from exception
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


def commit_training_snapshot(
    run_directory: Path,
    checkpoint: Path,
    *,
    algorithm: str,
    task: str,
    seed: int,
    round_index: int,
    world_rng_state: Any,
    python_rng_state: Any,
    numpy_rng_state: Any,
    early_stopping_rewards: list[float],
    source_commit: str | None,
    source_hash: str | None = None,
    source_hash_scope: str | None = None,
    cumulative_completed_rounds: int | None = None,
    early_stopping_config: dict[str, Any] | None = None,
    performance_stopping: dict[str, Any] | None = None,
    performance_history: list[dict[str, Any]] | None = None,
) -> Path:
    """Atomically publish one complete round-boundary snapshot."""
    run_directory = Path(run_directory).resolve()
    resume_root = run_directory / "resume"
    resume_root.mkdir(parents=True, exist_ok=True)
    generation = f"generation-{round_index:08d}"
    final_directory = resume_root / generation
    temporary = resume_root / f".{generation}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        learner = _read_checkpoint_metadata(checkpoint, algorithm)
        required = {
            "checkpoint_schema", "algorithm", "actions", "feature_id",
            "feature_schema", "reward_id", "reward_version", "reward_spec",
            "training_task", "training_device_name", "training_device_type",
            "agent_seed", "exploration_spec", "network_spec", "hyperparameters",
            "safe_exploration", "n_step", "retention_spec", "training_budget",
            "safety_spec", "safety_decisions", "safety_interventions",
            "safety_fallbacks",
            "total_action_steps", "stage_action_steps", "n_step_state",
            "agent_rng_state", "action_history_state",
            "safe_exploration_decisions", "safe_exploration_fallbacks",
        }
        if algorithm in TABLE_ALGORITHMS:
            required.update(
                {"q_table"} if algorithm == "q_learning"
                else {"q_table_a", "q_table_b"})
        elif algorithm in PICKLE_ALGORITHMS:
            required.update({"weights", "traces", "tile_coder", "updates", "learner_rng_state"})
        else:
            required.update({
                "policy", "target", "optimizer", "replay", "torch_rng_state",
                "updates", "teacher",
            })
        learner_feature_id = normalize_feature_id(
            learner.get("feature_id"), learner.get("feature_version"))
        if learner_feature_id == "continuous-phase-v1":
            required.update({"phase_history_state", "transfer_contract"})
            required.update({
                "distillation_dataset", "distillation_dataset_hash",
                "distillation_rng_state",
            })
        missing = sorted(required.difference(learner))
        if missing:
            raise ValueError(
                "Checkpoint is not resumable; missing fields: " + ", ".join(missing)
            )
        if learner["checkpoint_schema"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Checkpoint uses an incompatible resume schema")
        if learner["algorithm"] != algorithm or learner["training_task"] != task:
            raise ValueError("Checkpoint algorithm/task does not match the run")
        feature_id = learner_feature_id
        feature_schema = learner.get(
            "feature_schema", feature_schema_contract(feature_id))
        board_contract = feature_schema.get("board_shape")
        board_shape = (
            None if board_contract is None
            or any(value is None for value in board_contract[1:])
            else tuple(board_contract[1:])
        )
        validate_checkpoint_feature_contract(
            learner, feature_id, tuple(learner["actions"]),
            board_shape=board_shape)
        reward_id = learner.get("reward_id", learner["reward_version"])
        if reward_id != learner["reward_version"]:
            raise ValueError("Checkpoint reward_id conflicts with reward_version")

        learner_files: list[str]
        if algorithm in TABLE_ALGORITHMS:
            table_names = ("q_table",) if algorithm == "q_learning" else (
                "q_table_a", "q_table_b")
            arrays = {}
            for table_name in table_names:
                table = learner.pop(table_name)
                arrays[table_name + "_keys"] = np.asarray(list(table), dtype=np.int16)
                arrays[table_name + "_values"] = (
                    np.stack([np.asarray(table[key], dtype=np.float32) for key in table])
                    if table else np.empty((0, len(learner["actions"])), dtype=np.float32)
                )
            np.savez_compressed(temporary / "q_table.npz", **arrays)
            (temporary / "learner.json").write_bytes(_json_bytes(learner))
            learner_files = ["learner.json", "q_table.npz"]
        elif algorithm in PICKLE_ALGORITHMS:
            _link_or_copy(checkpoint, temporary / "learner.pkl")
            learner_files = ["learner.pkl"]
        else:
            # Agent checkpoint writers publish by atomic replace.  A hard link
            # therefore pins this immutable inode even when the working path is
            # replaced after the next round.
            _link_or_copy(checkpoint, temporary / "learner.pt")
            learner_files = ["learner.pt"]

        runner_state = {
            "contract": {
                "actions": list(learner["actions"]),
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                "agent_seed": learner["agent_seed"],
                "exploration_spec": learner["exploration_spec"],
                "training_device_name": learner["training_device_name"],
                "training_device_type": learner["training_device_type"],
                "feature_id": feature_id,
                "feature_schema": feature_schema,
                "feature_version": learner.get("feature_version"),
                "reward_id": reward_id,
                "reward_spec": learner["reward_spec"],
                "reward_version": learner["reward_version"],
                "network_spec": learner.get("network_spec"),
                "hyperparameters": learner.get("hyperparameters", {}),
                "safe_exploration": learner["safe_exploration"],
                "safety_spec": learner["safety_spec"],
                "n_step": learner["n_step"],
                "retention_spec": learner["retention_spec"],
                "training_budget": learner["training_budget"],
                "performance_stopping": performance_stopping,
                "transfer_contract": learner.get("transfer_contract"),
            },
            "cumulative_completed_rounds": (
                int(round_index)
                if cumulative_completed_rounds is None
                else int(cumulative_completed_rounds)
            ),
            "early_stopping_config": early_stopping_config,
            "early_stopping_rewards": list(early_stopping_rewards),
            "performance_history": list(performance_history or []),
            "numpy_rng_state": numpy_rng_state,
            "python_rng_state": python_rng_state,
            "world_rng_state": world_rng_state,
        }
        (temporary / "runner_state.json").write_bytes(_json_bytes(runner_state))
        files = learner_files + ["runner_state.json"]
        hashes = {name: _sha256(temporary / name) for name in files}
        manifest = {
            "agent_seed": learner["agent_seed"],
            "algorithm": algorithm,
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "exploration_spec": learner["exploration_spec"],
            "files": hashes,
            "round_index": int(round_index),
            "seed": int(seed),
            "source_commit": source_commit,
            "source_hash": source_hash,
            "source_hash_scope": source_hash_scope,
            "task": task,
        }
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        generation_hash = _sha256(temporary / "manifest.json")
        if final_directory.exists():
            raise FileExistsError(f"Snapshot generation already exists: {generation}")
        temporary.replace(final_directory)

        previous: list[str] = []
        latest_path = resume_root / "latest.json"
        if latest_path.is_file():
            previous = json.loads(latest_path.read_text(encoding="utf-8")).get(
                "generations", []
            )
        generations = [generation, *(name for name in previous if name != generation)]
        generations = generations[:RETAINED_GENERATIONS]
        _atomic_write(latest_path, _json_bytes({
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "generation_hash": generation_hash,
            "generations": generations,
        }))
        keep = set(generations)
        for candidate in resume_root.glob("generation-*"):
            if candidate.is_dir() and candidate.name not in keep:
                shutil.rmtree(candidate)
        return final_directory
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_generation(
    run_directory: Path, generation: str, expected_manifest_hash: str | None = None,
    *, expected_schema: str = CHECKPOINT_SCHEMA_VERSION,
) -> LoadedSnapshot:
    directory = run_directory / "resume" / generation
    manifest_path = directory / "manifest.json"
    if expected_manifest_hash is not None and _sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Snapshot manifest failed latest-pointer validation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_schema") != expected_schema:
        raise ValueError("Snapshot manifest has an incompatible schema")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Snapshot file failed integrity validation: {name}")
    runner = _decode(json.loads((directory / "runner_state.json").read_text()))
    algorithm = manifest["algorithm"]
    payload = None
    learner_path = None
    if algorithm in TABLE_ALGORITHMS:
        payload = _decode(json.loads((directory / "learner.json").read_text()))
        with np.load(directory / "q_table.npz", allow_pickle=False) as archive:
            table_names = ("q_table",) if algorithm == "q_learning" else (
                "q_table_a", "q_table_b")
            for table_name in table_names:
                keys = archive[table_name + "_keys"]
                values = archive[table_name + "_values"]
                payload[table_name] = {
                    tuple(int(item) for item in key): value.astype(np.float32, copy=True)
                    for key, value in zip(keys, values)
                }
    elif algorithm in PICKLE_ALGORITHMS:
        learner_path = directory / "learner.pkl"
    else:
        learner_path = directory / "learner.pt"
    return LoadedSnapshot(
        run_directory=run_directory,
        generation=generation,
        generation_hash=_sha256(manifest_path),
        algorithm=algorithm,
        task=manifest["task"],
        seed=int(manifest["seed"]),
        round_index=int(manifest["round_index"]),
        source_commit=manifest.get("source_commit"),
        source_hash=manifest.get("source_hash"),
        source_hash_scope=manifest.get("source_hash_scope"),
        runner_state=runner,
        learner_payload=payload,
        learner_path=learner_path,
    )


def _load_training_snapshot(run_directory: Path, *, expected_schema: str) -> LoadedSnapshot:
    """Load the newest valid committed generation, falling back once if needed."""
    run_directory = Path(run_directory).resolve()
    latest_path = run_directory / "resume" / "latest.json"
    if not latest_path.is_file():
        raise ValueError("Parent run has no complete resume snapshot")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("checkpoint_schema") != expected_schema:
        raise ValueError("Resume pointer has an incompatible checkpoint schema")
    generations = latest.get("generations", [])
    if not generations:
        raise ValueError("Parent run has no committed resume generation")
    newest_round = None
    failures = []
    for index, generation in enumerate(generations[:RETAINED_GENERATIONS]):
        try:
            loaded = _load_generation(
                run_directory,
                generation,
                latest.get("generation_hash") if index == 0 else None,
                expected_schema=expected_schema,
            )
            newest_round = loaded.round_index if newest_round is None else newest_round
            return LoadedSnapshot(
                **{
                    **loaded.__dict__,
                    "lost_rounds": max(0, newest_round - loaded.round_index),
                    "fallback_reason": "; ".join(failures) if failures else None,
                }
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exception:
            if newest_round is None:
                try:
                    newest_round = int(generation.rsplit("-", 1)[1])
                except (IndexError, ValueError):
                    newest_round = 0
            failures.append(f"{generation}: {exception}")
    raise ValueError("No valid resume snapshot: " + "; ".join(failures))


def load_training_snapshot(run_directory: Path) -> LoadedSnapshot:
    """Load a v7 snapshot for exact continuation or curriculum promotion."""
    return _load_training_snapshot(
        run_directory, expected_schema=CHECKPOINT_SCHEMA_VERSION)


def load_migration_snapshot(run_directory: Path) -> LoadedSnapshot:
    """Load a complete v6 snapshot through the explicit one-way migration path."""
    return _load_training_snapshot(
        run_directory, expected_schema=MIGRATABLE_CHECKPOINT_SCHEMA)


def load_task3_transfer_snapshot(run_directory: Path) -> LoadedSnapshot:
    """Load a complete v7 Task 2 parent through the explicit transfer path."""
    return _load_training_snapshot(
        run_directory, expected_schema=TASK3_TRANSFER_PARENT_SCHEMA)


def materialize_learner_checkpoint(snapshot: LoadedSnapshot, destination: Path) -> None:
    """Create the child run's working checkpoint from a validated snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if snapshot.algorithm in TABLE_ALGORITHMS:
        with temporary.open("wb") as file:
            pickle.dump(snapshot.learner_payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        assert snapshot.learner_path is not None
        shutil.copy2(snapshot.learner_path, temporary)
    temporary.replace(destination)


def materialize_migrated_checkpoint(
    snapshot: LoadedSnapshot,
    destination: Path,
    *,
    training_budget: dict[str, Any],
) -> None:
    """Copy a validated v6 learner into a v7 child without touching the parent."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if snapshot.algorithm in TABLE_ALGORITHMS:
        assert snapshot.learner_payload is not None
        payload = dict(snapshot.learner_payload)
        payload["checkpoint_schema"] = CHECKPOINT_SCHEMA_VERSION
        payload["training_budget"] = dict(training_budget)
        with temporary.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        assert snapshot.learner_path is not None
        try:
            import torch
        except ImportError as exception:
            raise RuntimeError("PyTorch is required to migrate a DQN snapshot") from exception
        payload = torch.load(snapshot.learner_path, map_location="cpu", weights_only=True)
        payload["checkpoint_schema"] = CHECKPOINT_SCHEMA_VERSION
        payload["training_budget"] = dict(training_budget)
        torch.save(payload, temporary)
    temporary.replace(destination)


def _expanded_network_state(state: dict[str, Any], new_input_size: int = 117):
    """Copy an 84-input MLP into a larger input while zeroing new columns."""
    import torch

    expanded = {name: value.clone() for name, value in state.items()}
    name = "layers.0.weight"
    source = state[name]
    if tuple(source.shape) != (128, 84):
        raise ValueError("Task 3 transfer requires an 84x128 parent first layer")
    weight = torch.zeros(
        (source.shape[0], new_input_size), dtype=source.dtype,
        device=source.device)
    weight[:, :source.shape[1]].copy_(source)
    expanded[name] = weight
    return expanded


def load_distillation_dataset(path: Path) -> tuple[dict[str, Any], str]:
    """Load and hash one immutable 117-dimensional teacher dataset."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Task 3 distillation dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"states", "legal_masks", "teacher_q", "environment_seeds", "task_ids"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError("Distillation dataset is missing: " + ", ".join(missing))
        states = archive["states"].astype(np.float32, copy=True)
        legal = archive["legal_masks"].astype(bool, copy=True)
        teacher_q = archive["teacher_q"].astype(np.float32, copy=True)
        environment_seeds = archive["environment_seeds"].astype(np.int64, copy=True)
        task_ids = archive["task_ids"].astype(np.int8, copy=True)
    if states.ndim != 2 or states.shape[1] != 117:
        raise ValueError("Distillation states must have shape (N,117)")
    if legal.shape != (len(states), 6) or teacher_q.shape != legal.shape:
        raise ValueError("Distillation masks/Q values must have shape (N,6)")
    if environment_seeds.shape != (len(states),) or task_ids.shape != (len(states),):
        raise ValueError("Distillation provenance arrays have an incompatible shape")
    if set(np.unique(task_ids)) != {1, 2}:
        raise ValueError("Distillation dataset must include Task 1 and Task 2")
    if any(np.count_nonzero(task_ids == task) > 10_000 for task in (1, 2)):
        raise ValueError("Distillation dataset exceeds the per-Task 10,000-row cap")
    if not set(environment_seeds).issubset(set(range(6000, 6100))):
        raise ValueError("Distillation data must use training-only seeds 6000-6099")
    if not legal.any(axis=1).all() or not np.isfinite(teacher_q).all():
        raise ValueError("Distillation dataset contains invalid masks or Q values")
    return {
        "states": states, "legal_masks": legal, "teacher_q": teacher_q,
        "environment_seeds": environment_seeds, "task_ids": task_ids,
    }, _sha256(path)


def materialize_task3_transfer_checkpoint(
    snapshot: LoadedSnapshot,
    destination: Path,
    *,
    child_contract: dict[str, Any],
    distillation_path: Path,
) -> dict[str, Any]:
    """Create a fresh v8 phase learner from one validated v7 Task 2 parent."""
    if snapshot.learner_path is None:
        raise ValueError("Task 3 transfer requires a neural parent")
    try:
        import torch
    except ImportError as exception:
        raise RuntimeError("PyTorch is required for Task 3 transfer") from exception
    parent = torch.load(snapshot.learner_path, map_location="cpu", weights_only=True)
    dataset, dataset_hash = load_distillation_dataset(distillation_path)
    seed = int(child_contract["seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch_rng_state = torch.get_rng_state()
    policy = _expanded_network_state(parent["policy"])
    target = _expanded_network_state(parent.get("target", parent["policy"]))
    transfer_contract = {
        "version": "task2-to-phase-v1",
        "parent_run": str(snapshot.run_directory),
        "parent_generation": snapshot.generation,
        "parent_generation_hash": snapshot.generation_hash,
        "parent_checkpoint_sha256": _sha256(snapshot.learner_path),
        "parent_feature_id": "continuous-v2",
        "parent_input_columns": 84,
        "new_zero_columns": 33,
        "distillation_dataset": str(Path(distillation_path).resolve()),
        "distillation_dataset_sha256": dataset_hash,
    }
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "double_dqn",
        "actions": list(child_contract["actions"]),
        "feature_id": child_contract["feature_id"],
        "feature_version": None,
        "feature_schema": child_contract["feature_schema"],
        "reward_id": child_contract["reward_id"],
        "reward_version": child_contract["reward_id"],
        "reward_spec": child_contract["reward_spec"],
        "hyperparameters": child_contract["hyperparameters"],
        "network_spec": child_contract["network_spec"],
        "policy": policy, "target": target,
        "replay": {
            "format": "vector-replay-task-partitioned-v2", "count": 0,
            "capacity": int(child_contract["retention_spec"]["per_task_capacity"]),
            "current_task": child_contract["task"],
            "rng_state": random.Random(seed).getstate(),
        },
        "torch_rng_state": torch_rng_state,
        "training_device_name": child_contract["training_device_name"],
        "training_device_type": child_contract["training_device_type"],
        "peak_cuda_memory_bytes": 0,
        "updates": 0, "teacher": None,
        "retention_spec": child_contract["retention_spec"],
        "double_dqn": True,
        "distillation_dataset": {
            "states": torch.as_tensor(dataset["states"]),
            "legal_masks": torch.as_tensor(dataset["legal_masks"]),
            "teacher_q": torch.as_tensor(dataset["teacher_q"]),
        },
        "distillation_dataset_hash": dataset_hash,
        "distillation_rng_state": random.Random(seed + 104729).getstate(),
        "total_action_steps": int(parent.get("total_action_steps", 0)),
        "stage_action_steps": 0,
        "action_steps": int(parent.get("total_action_steps", 0)),
        "agent_seed": seed,
        "agent_rng_state": random.Random(seed).getstate(),
        "exploration_spec": child_contract["exploration_spec"],
        "safe_exploration": child_contract["safe_exploration"],
        "safe_exploration_decisions": 0, "safe_exploration_fallbacks": 0,
        "safety_spec": child_contract["safety_spec"],
        "safety_decisions": 0, "safety_interventions": 0,
        "safety_fallbacks": 0,
        "action_history_state": None,
        "phase_history_state": None,
        "n_step": child_contract["n_step"], "n_step_state": None,
        "training_budget": child_contract["training_budget"],
        "training_task": child_contract["task"],
        "transfer_contract": transfer_contract,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return transfer_contract


def validate_task3_transfer(
    parent: dict[str, Any], child: dict[str, Any], *, parent_status: str,
) -> None:
    """Validate the intentionally narrow v7 Task 2 to v8 phase seam."""
    if parent.get("checkpoint_schema") != TASK3_TRANSFER_PARENT_SCHEMA:
        raise ValueError("Task 3 transfer requires a training-resume-v7 parent")
    if child.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Task 3 transfer target must use training-resume-v8")
    if parent_status not in {"completed", "early_stopped"}:
        raise ValueError("Task 3 transfer requires a completed Task 2 parent")
    if parent.get("task") != "crate_navigation" or child.get("task") != "weak_opponents":
        raise ValueError("Task 3 transfer must promote Task 2 to Task 3")
    if parent.get("algorithm") != "double_dqn" or child.get("algorithm") != "double_dqn":
        raise ValueError("Task 3 transfer requires Double DQN")
    if parent.get("feature_id") != "continuous-v2":
        raise ValueError("Task 3 transfer requires an 84-dimensional continuous-v2 parent")
    if tuple(parent.get("feature_schema", {}).get("vector_shape", ())) != (84,):
        raise ValueError("Task 3 transfer parent feature shape must be 84")
    if child.get("feature_id") != "continuous-phase-v1":
        raise ValueError("Task 3 transfer child must use continuous-phase-v1")
    if tuple(child.get("feature_schema", {}).get("vector_shape", ())) != (117,):
        raise ValueError("Task 3 transfer child feature shape must be 117")
    for field in ("seed", "agent_seed", "actions"):
        if parent.get(field) != child.get(field):
            raise ValueError(f"Task 3 transfer {field} must match the parent")


def validate_v6_migration(
    parent: dict[str, Any], child: dict[str, Any], *, parent_status: str,
) -> None:
    """Validate the narrow, explicit v6-to-v7 Task 1 migration contract."""
    if parent.get("checkpoint_schema") != MIGRATABLE_CHECKPOINT_SCHEMA:
        raise ValueError("Migration requires a training-resume-v6 parent")
    if child.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Migration target must use training-resume-v8")
    if parent_status != "completed":
        raise ValueError("Migration requires a completed v6 parent run")
    if parent.get("task") != "coin_navigation" or child.get("task") != "coin_navigation":
        raise ValueError("v6 migration is supported only within Task 1")
    for field in (
        "algorithm", "seed", "reward_spec", "training_device_type",
        "training_device_name", "agent_seed", "safe_exploration", "safety_spec",
        "n_step", "retention_spec", "exploration_spec", "network_spec",
        "hyperparameters",
    ):
        if parent.get(field) != child.get(field):
            raise ValueError(f"Migration {field} must match the v6 parent")
    parent_feature = normalize_feature_id(
        parent.get("feature_id"), parent.get("feature_version"))
    child_feature = normalize_feature_id(
        child.get("feature_id"), child.get("feature_version"))
    if parent_feature != child_feature or parent.get("feature_schema") != child.get("feature_schema"):
        raise ValueError("Migration feature contract must match the v6 parent")
    if parent.get("reward_id", parent.get("reward_version")) != child.get(
        "reward_id", child.get("reward_version")):
        raise ValueError("Migration reward ID must match the v6 parent")
    if list(parent.get("actions", ())) != list(child.get("actions", ())):
        raise ValueError("Migration action order must match the v6 parent")


def validate_resume_transition(
    parent: dict[str, Any], child: dict[str, Any], *, parent_status: str
) -> str:
    """Validate curriculum and compatibility contracts; return resume kind."""
    for field in (
        "algorithm", "seed", "checkpoint_schema", "reward_spec",
        "training_device_type", "training_device_name", "agent_seed",
        "source_commit", "safe_exploration", "safety_spec",
    ):
        if parent.get(field) != child.get(field):
            raise ValueError(f"Resume {field} must match the parent run")
    parent_source_scope = parent.get("source_hash_scope")
    child_source_scope = child.get("source_hash_scope")
    if parent_source_scope is not None:
        if parent_source_scope != child_source_scope:
            raise ValueError("Resume source_hash_scope must match the parent run")
        if parent.get("source_hash") != child.get("source_hash"):
            raise ValueError("Resume source_hash must match the parent run")
    try:
        parent_feature = normalize_feature_id(
            parent.get("feature_id"), parent.get("feature_version"))
        child_feature = normalize_feature_id(
            child.get("feature_id"), child.get("feature_version"))
    except ValueError as exception:
        raise ValueError(
            f"Resume feature_version/feature_id is incompatible: {exception}") from exception
    if parent_feature != child_feature:
        raise ValueError("Resume feature_version/feature_id must match the parent run")
    parent_schema = parent.get("feature_schema")
    child_schema = child.get("feature_schema")
    if parent_schema is not None or child_schema is not None:
        expected_schema = feature_schema_contract(parent_feature)
        if parent_schema is None:
            parent_schema = expected_schema
        if child_schema is None:
            child_schema = expected_schema
        if parent_schema != child_schema:
            raise ValueError("Resume feature schema/shape must match the parent run")
    parent_reward = parent.get("reward_id", parent.get("reward_version"))
    child_reward = child.get("reward_id", child.get("reward_version"))
    if parent_reward != child_reward:
        raise ValueError("Resume reward_version/reward_id must match the parent run")
    if parent.get("actions") is not None and child.get("actions") is not None:
        if list(parent["actions"]) != list(child["actions"]):
            raise ValueError("Resume action order must match the parent run")
    for field in ("network_spec", "hyperparameters"):
        if parent.get(field) != child.get(field):
            raise ValueError(f"Resume {field} must match the parent run")
    try:
        parent_index = TASK_ORDER.index(parent["task"])
        child_index = TASK_ORDER.index(child["task"])
    except (KeyError, ValueError) as exception:
        raise ValueError("Resume task is not part of the curriculum") from exception
    if child_index == parent_index:
        for field in (
            "n_step", "retention_spec", "performance_stopping", "transfer_contract",
        ):
            if parent.get(field) != child.get(field):
                raise ValueError(f"Same-Task resume {field} must match the parent run")
        if parent.get("exploration_spec") != child.get("exploration_spec"):
            raise ValueError("Same-Task resume exploration_spec must match the parent run")
        parent_budget = parent.get("training_budget", {})
        child_budget = child.get("training_budget", {})
        if parent_budget.get("min_rounds") != child_budget.get("min_rounds"):
            raise ValueError("Same-Task resume minimum rounds must match the parent run")
        parent_target = parent_budget.get("target_stage_action_steps")
        child_target = child_budget.get("target_stage_action_steps")
        if (
            parent_target is not None
            and (child_target is None or int(child_target) < int(parent_target))
        ):
            raise ValueError("Same-Task resume action target cannot decrease")
        return "same_task"
    if child_index != parent_index + 1:
        raise ValueError("Resume must use the same or direct next Task")
    if parent_status not in {"completed", "early_stopped"}:
        raise ValueError("Promotion to the next Task requires a completed parent run")
    if parent_feature == "continuous-phase-v1" and (
        parent.get("transfer_contract") != child.get("transfer_contract")
    ):
        raise ValueError("Resume transfer_contract must match the parent run")
    return "next_task"
