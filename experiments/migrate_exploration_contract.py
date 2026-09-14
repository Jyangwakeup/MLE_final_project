"""Create an audited copy of a run with a corrected exploration contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

from experiments.resume import load_training_snapshot
from experiments.run import PROJECT_ROOT, _source_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _replace_checkpoint_spec(path: Path, expected: int, replacement: int) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    spec = payload.get("exploration_spec")
    if not isinstance(spec, dict) or spec.get("decay_action_steps") != expected:
        raise ValueError(f"Unexpected exploration contract in {path}")
    if payload.get("hyperparameters", {}).get("epsilon_decay_action_steps") != replacement:
        raise ValueError(f"Checkpoint does not prove actual {replacement}-step behavior")
    payload["exploration_spec"] = {**spec, "decay_action_steps": replacement}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def migrate(source: Path, destination: Path, expected: int, replacement: int) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source run does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    original_snapshot = load_training_snapshot(source)
    original_hash = original_snapshot.generation_hash
    current_hash = _source_hash()
    current_commit = _source_commit()

    shutil.copytree(source, destination)
    try:
        final_checkpoint = destination / "checkpoints" / "final.pt"
        _replace_checkpoint_spec(final_checkpoint, expected, replacement)

        latest_path = destination / "resume" / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        newest_manifest_hash = None
        for index, generation in enumerate(latest["generations"]):
            generation_dir = destination / "resume" / generation
            learner_path = generation_dir / "learner.pt"
            _replace_checkpoint_spec(learner_path, expected, replacement)

            runner_path = generation_dir / "runner_state.json"
            runner = json.loads(runner_path.read_text(encoding="utf-8"))
            runner_spec = runner["contract"]["exploration_spec"]
            if runner_spec.get("decay_action_steps") != expected:
                raise ValueError(f"Unexpected runner exploration contract in {generation}")
            runner["contract"]["exploration_spec"] = {
                **runner_spec, "decay_action_steps": replacement,
            }
            _write_json(runner_path, runner)

            manifest_path = generation_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["exploration_spec"].get("decay_action_steps") != expected:
                raise ValueError(f"Unexpected manifest exploration contract in {generation}")
            manifest["exploration_spec"] = {
                **manifest["exploration_spec"], "decay_action_steps": replacement,
            }
            manifest["source_commit"] = current_commit
            manifest["source_hash"] = current_hash
            manifest["files"] = {
                name: _sha256(generation_dir / name)
                for name in manifest["files"]
            }
            _write_json(manifest_path, manifest)
            if index == 0:
                newest_manifest_hash = _sha256(manifest_path)

        latest["generation_hash"] = newest_manifest_hash
        _write_json(latest_path, latest)

        metadata_path = destination / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        old_source_hash = metadata.get("source_hash")
        metadata["run_id"] = destination.name
        metadata["checkpoint"] = str(final_checkpoint)
        metadata["source_commit"] = current_commit
        metadata["source_hash"] = current_hash
        metadata["exploration_spec"] = {
            **metadata["exploration_spec"], "decay_action_steps": replacement,
        }
        expanded = metadata["expanded_config"]
        expanded["training"]["exploration"] = {
            **expanded["training"]["exploration"],
            "decay_action_steps": replacement,
        }
        expanded["execution"]["checkpoint"] = str(final_checkpoint)
        metadata["config_sha256"] = hashlib.sha256(
            json.dumps(expanded, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        metadata["migration"] = {
            "kind": "exploration-contract-correction-v1",
            "source_run": str(source),
            "source_generation": original_snapshot.generation,
            "source_generation_hash": original_hash,
            "source_source_hash": old_source_hash,
            "recorded_decay_action_steps": expected,
            "proven_actual_decay_action_steps": replacement,
            "evidence": "checkpoint.hyperparameters.epsilon_decay_action_steps",
            "learner_state_changed": False,
        }
        _write_json(metadata_path, metadata)

        migrated = load_training_snapshot(destination)
        if migrated.contract["exploration_spec"]["decay_action_steps"] != replacement:
            raise ValueError("Migrated snapshot did not retain the corrected contract")
        return destination
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--replacement", type=int, required=True)
    args = parser.parse_args()
    result = migrate(args.source, args.destination, args.expected, args.replacement)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
