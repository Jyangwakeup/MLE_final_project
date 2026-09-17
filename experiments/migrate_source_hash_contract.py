"""Create an audited run copy under the corrected agent source-hash contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.resume import load_training_snapshot
from experiments.run import SOURCE_HASH_SCOPE, _source_commit, _source_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def migrate(
    source: Path, destination: Path, *, agent: str, expected_hash: str,
) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    snapshot = load_training_snapshot(source)
    if snapshot.source_hash != expected_hash:
        raise ValueError("Source snapshot does not have the expected source hash")

    replacement_hash = _source_hash(agent)
    replacement_commit = _source_commit()
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
                raise ValueError(f"Unexpected source hash in {generation}")
            manifest["source_hash"] = replacement_hash
            manifest["source_hash_scope"] = SOURCE_HASH_SCOPE
            manifest["source_commit"] = replacement_commit
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
        metadata["run_id"] = destination.name
        source_checkpoint = Path(metadata["checkpoint"])
        metadata["checkpoint"] = str(
            destination / "checkpoints" / source_checkpoint.name)
        metadata["source_hash"] = replacement_hash
        metadata["source_hash_scope"] = SOURCE_HASH_SCOPE
        metadata["source_commit"] = replacement_commit
        metadata["migration"] = {
            "kind": "agent-source-hash-contract-v2",
            "source_run": str(source),
            "source_generation": snapshot.generation,
            "source_generation_hash": snapshot.generation_hash,
            "source_source_hash": expected_hash,
            "replacement_source_hash": replacement_hash,
            "reason": (
                "Separate executable runtime provenance from the independently "
                "recorded per-Task configuration hash."
            ),
            "learner_state_changed": False,
        }
        _write_json(metadata_path, metadata)

        migrated = load_training_snapshot(destination)
        if migrated.source_hash != replacement_hash:
            raise ValueError("Migrated snapshot has the wrong source hash")
        if migrated.source_hash_scope != SOURCE_HASH_SCOPE:
            raise ValueError("Migrated snapshot has the wrong source hash scope")
        return destination
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--expected-hash", required=True)
    args = parser.parse_args()
    print(migrate(
        args.source, args.destination, agent=args.agent,
        expected_hash=args.expected_hash))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
