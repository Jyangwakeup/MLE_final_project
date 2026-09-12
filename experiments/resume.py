"""Crash-safe, versioned training snapshots for curriculum runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import uuid
from typing import Any

import numpy as np


CHECKPOINT_SCHEMA_VERSION = "training-resume-v1"
TASK_ORDER = ("coin_navigation", "crate_navigation", "weak_opponents", "full_match")
RETAINED_GENERATIONS = 2


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
    lost_rounds: int = 0
    fallback_reason: str | None = None

    @property
    def contract(self) -> dict[str, Any]:
        metadata = self.runner_state["contract"]
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "task": self.task,
            "feature_version": metadata["feature_version"],
            "reward_version": metadata["reward_version"],
            "checkpoint_schema": metadata["checkpoint_schema"],
            "actions": metadata["actions"],
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


def _read_checkpoint_metadata(checkpoint: Path, algorithm: str) -> dict[str, Any]:
    if algorithm == "q_learning":
        with checkpoint.open("rb") as file:
            return pickle.load(file)
    if algorithm == "dqn":
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
    cumulative_completed_rounds: int | None = None,
    early_stopping_config: dict[str, Any] | None = None,
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
            "checkpoint_schema", "algorithm", "actions", "feature_version",
            "reward_version", "reward_spec", "training_task",
        }
        missing = sorted(required.difference(learner))
        if missing:
            raise ValueError(
                "Checkpoint is not resumable; missing fields: " + ", ".join(missing)
            )
        if learner["checkpoint_schema"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Checkpoint uses an incompatible resume schema")
        if learner["algorithm"] != algorithm or learner["training_task"] != task:
            raise ValueError("Checkpoint algorithm/task does not match the run")

        learner_files: list[str]
        if algorithm == "q_learning":
            q_table = learner.pop("q_table")
            keys = np.asarray(list(q_table), dtype=np.int16)
            values = (
                np.stack([np.asarray(q_table[key], dtype=np.float32) for key in q_table])
                if q_table else np.empty((0, len(learner["actions"])), dtype=np.float32)
            )
            np.savez_compressed(temporary / "q_table.npz", keys=keys, values=values)
            (temporary / "learner.json").write_bytes(_json_bytes(learner))
            learner_files = ["learner.json", "q_table.npz"]
        else:
            shutil.copy2(checkpoint, temporary / "learner.pt")
            learner_files = ["learner.pt"]

        runner_state = {
            "contract": {
                "actions": list(learner["actions"]),
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                "feature_version": learner["feature_version"],
                "reward_spec": learner["reward_spec"],
                "reward_version": learner["reward_version"],
            },
            "cumulative_completed_rounds": (
                int(round_index)
                if cumulative_completed_rounds is None
                else int(cumulative_completed_rounds)
            ),
            "early_stopping_config": early_stopping_config,
            "early_stopping_rewards": list(early_stopping_rewards),
            "numpy_rng_state": numpy_rng_state,
            "python_rng_state": python_rng_state,
            "world_rng_state": world_rng_state,
        }
        (temporary / "runner_state.json").write_bytes(_json_bytes(runner_state))
        files = learner_files + ["runner_state.json"]
        hashes = {name: _sha256(temporary / name) for name in files}
        manifest = {
            "algorithm": algorithm,
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "files": hashes,
            "round_index": int(round_index),
            "seed": int(seed),
            "source_commit": source_commit,
            "source_hash": source_hash,
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
) -> LoadedSnapshot:
    directory = run_directory / "resume" / generation
    manifest_path = directory / "manifest.json"
    if expected_manifest_hash is not None and _sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Snapshot manifest failed latest-pointer validation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Snapshot manifest has an incompatible schema")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Snapshot file failed integrity validation: {name}")
    runner = _decode(json.loads((directory / "runner_state.json").read_text()))
    algorithm = manifest["algorithm"]
    payload = None
    learner_path = None
    if algorithm == "q_learning":
        payload = _decode(json.loads((directory / "learner.json").read_text()))
        with np.load(directory / "q_table.npz", allow_pickle=False) as archive:
            keys = archive["keys"]
            values = archive["values"]
        payload["q_table"] = {
            tuple(int(item) for item in key): value.astype(np.float32, copy=True)
            for key, value in zip(keys, values)
        }
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
        runner_state=runner,
        learner_payload=payload,
        learner_path=learner_path,
    )


def load_training_snapshot(run_directory: Path) -> LoadedSnapshot:
    """Load the newest valid committed generation, falling back once if needed."""
    run_directory = Path(run_directory).resolve()
    latest_path = run_directory / "resume" / "latest.json"
    if not latest_path.is_file():
        raise ValueError("Parent run has no complete resume snapshot")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
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


def materialize_learner_checkpoint(snapshot: LoadedSnapshot, destination: Path) -> None:
    """Create the child run's working checkpoint from a validated snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if snapshot.algorithm == "q_learning":
        with temporary.open("wb") as file:
            pickle.dump(snapshot.learner_payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        assert snapshot.learner_path is not None
        shutil.copy2(snapshot.learner_path, temporary)
    temporary.replace(destination)


def validate_resume_transition(
    parent: dict[str, Any], child: dict[str, Any], *, parent_status: str
) -> str:
    """Validate curriculum and compatibility contracts; return resume kind."""
    for field in (
        "algorithm", "seed", "feature_version", "reward_version", "checkpoint_schema"
    ):
        if parent.get(field) != child.get(field):
            raise ValueError(f"Resume {field} must match the parent run")
    if parent.get("actions") is not None and child.get("actions") is not None:
        if list(parent["actions"]) != list(child["actions"]):
            raise ValueError("Resume action order must match the parent run")
    try:
        parent_index = TASK_ORDER.index(parent["task"])
        child_index = TASK_ORDER.index(child["task"])
    except (KeyError, ValueError) as exception:
        raise ValueError("Resume task is not part of the curriculum") from exception
    if child_index == parent_index:
        return "same_task"
    if child_index != parent_index + 1:
        raise ValueError("Resume must use the same or direct next Task")
    if parent_status not in {"completed", "early_stopped"}:
        raise ValueError("Promotion to the next Task requires a completed parent run")
    return "next_task"
