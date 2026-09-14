"""Create an audited resume copy after display-only source changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.resume import load_training_snapshot
from experiments.run import _source_commit, _source_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def migrate(source: Path, destination: Path, expected_hash: str) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    snapshot = load_training_snapshot(source)
    if snapshot.source_hash != expected_hash:
        raise ValueError("Source snapshot does not have the verified pre-display hash")

    current_hash = _source_hash()
    current_commit = _source_commit()
    shutil.copytree(source, destination)
    try:
        latest_path = destination / "resume" / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        newest_manifest_hash = None
        for index, generation in enumerate(latest["generations"]):
            generation_dir = destination / "resume" / generation
            manifest_path = generation_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_hash") != expected_hash:
                raise ValueError(f"Unexpected manifest source hash in {generation}")
            manifest["source_hash"] = current_hash
            manifest["source_commit"] = current_commit
            manifest["files"] = {
                name: _sha256(generation_dir / name) for name in manifest["files"]
            }
            _write_json(manifest_path, manifest)
            if index == 0:
                newest_manifest_hash = _sha256(manifest_path)

        latest["generation_hash"] = newest_manifest_hash
        _write_json(latest_path, latest)

        metadata_path = destination / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["run_id"] = destination.name
        metadata["source_hash"] = current_hash
        metadata["source_commit"] = current_commit
        metadata["migration"] = {
            "kind": "display-only-source-contract-v1",
            "source_run": str(source),
            "source_generation": snapshot.generation,
            "source_generation_hash": snapshot.generation_hash,
            "source_source_hash": expected_hash,
            "replacement_source_hash": current_hash,
            "verified_changes": [
                "experiments/evaluation.py",
                "experiments/progress_plugin.py",
            ],
            "learner_state_changed": False,
        }
        _write_json(metadata_path, metadata)

        migrated = load_training_snapshot(destination)
        if migrated.source_hash != current_hash:
            raise ValueError("Migrated snapshot has the wrong source hash")
        return destination
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-hash", required=True)
    args = parser.parse_args()
    print(migrate(args.source, args.destination, args.expected_hash))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
