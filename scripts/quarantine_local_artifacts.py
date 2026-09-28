#!/usr/bin/env python3
"""Audit, quarantine, and restore explicitly approved local artifacts.

Quarantine is a reversible rename into ``.local-artifacts``.  It is not an
archive and this tool intentionally has no deletion command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
RAINBOW_GENERATED_TARGETS = (
    ".scratch/deleted-after-c0100-20260917-2300",
    ".scratch/interrupted-rainbow_lite_v7-r21-control-task3-c0020-20260917",
    ".scratch/interrupted-rainbow_lite_v7_r21-c0300-20260916",
    ".scratch/interrupted-v5-r18-maskv5-task3-c0050-20260917",
    ".scratch/interrupted-v5-r18-maskv5-task3-cache-benchmark-20260917",
    ".scratch/interrupted-v5-r18-maskv5-task3-cache-key-20260917-211304",
    ".scratch/interrupted-v5-r18-maskv5-task3-incomplete-integration-20260917",
    ".scratch/interrupted-v5-r18-maskv5-task3-postpull-20260917-213157",
    ".scratch/interrupted-v5-r18-maskv5-task3-precache-20260917",
    ".scratch/interrupted-v5-r18-maskv5-task3-race-c0050-20260917",
    ".scratch/invalid-v5-r18-maskv5-cross-step-cache-20260917-213956",
    ".scratch/premature-c0350-eval10-20260917_050710",
    ".scratch/restarted-c100-anchor-progress-fix-20260917-2310",
    ".scratch/smoke-v5-r18-maskv5-nocrosscache-error-20260917",
    "agent_code/rainbow_lite_agent/.DS_Store",
    "agent_code/rainbow_lite_agent/__pycache__",
    "agent_code/rainbow_lite_agent/logs",
    "agent_code/rainbow_lite_continuous_v2_agent/__pycache__",
    "agent_code/rainbow_lite_continuous_v2_agent/logs",
    "agent_code/rainbow_lite_no_safety_agent/__pycache__",
    "agent_code/rainbow_lite_no_safety_agent/logs",
    "agent_code/rainbow_lite_spatial_v6_agent/__pycache__",
    "agent_code/rainbow_lite_spatial_v6_agent/logs",
    "agent_code/rainbow_lite_v3_eval_agent",
    "agent_code/rainbow_lite_v5_agent/__pycache__",
    "agent_code/rainbow_lite_v5_agent/logs",
    "all_other_agent_code/rainbow_lite_v10_agent/__pycache__",
    "all_other_agent_code/rainbow_lite_v10_agent/logs",
    "all_other_agent_code/rainbow_lite_v11_agent/__pycache__",
    "all_other_agent_code/rainbow_lite_v11_agent/logs",
    "all_other_agent_code/rainbow_lite_v6_agent/__pycache__",
    "all_other_agent_code/rainbow_lite_v6_agent/logs",
    "all_other_agent_code/rainbow_lite_v6_stable_agent/__pycache__",
    "all_other_agent_code/rainbow_lite_v6_stable_agent/logs",
    "all_other_agent_code/rainbow_lite_v7_agent/__pycache__",
    "all_other_agent_code/rainbow_lite_v7_agent/logs",
    "all_other_agent_code/rainbow_lite_v8_agent/__pycache__",
    "all_other_agent_code/rainbow_lite_v8_agent/logs",
    "all_other_agent_code/rainbow_lite_v9_agent/__pycache__",
    "scripts/archive/rainbow_lite/monitor/__pycache__",
)
ALLOWED_TARGETS = (
    "output",
    ".scratch/archive-consolidation",
    ".scratch/branch-integration",
    ".scratch/die-hardest-submission",
    *RAINBOW_GENERATED_TARGETS,
)
AUDIT_PATHS = (
    "agent_code/die_hardest",
    "experiments/results",
    "archive",
    "runs",
    *ALLOWED_TARGETS,
)
SCHEMA_VERSION = 1


def local_root(root: Path) -> Path:
    return root / ".local-artifacts"


def manifests_root(root: Path) -> Path:
    return local_root(root) / "manifests"


def quarantine_root(root: Path) -> Path:
    return local_root(root) / "quarantine"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, *, follow_symlinks: bool = True) -> list[int]:
    stat = path.stat() if follow_symlinks else path.lstat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def validate_target(root: Path, name: str) -> Path:
    if name not in ALLOWED_TARGETS:
        raise ValueError(f"Target is not approved: {name}")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"Top-level target may not be a symlink: {name}")
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        raise ValueError(f"Target escapes repository: {name}")
    return path


def validate_local_store(root: Path) -> None:
    store = local_root(root)
    if store.is_symlink():
        raise ValueError(".local-artifacts may not be a symlink")
    if not store.resolve(strict=False).is_relative_to(root.resolve()):
        raise ValueError("Local artifact store escapes repository")


def tracked_paths(root: Path, name: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _record(path: Path, base: Path) -> dict[str, object]:
    relative = "." if path == base else path.relative_to(base).as_posix()
    if path.is_symlink():
        return {
            "path": relative,
            "kind": "symlink",
            "link_text": os.readlink(path),
            "identity": identity(path, follow_symlinks=False),
        }
    if path.is_dir():
        return {
            "path": relative,
            "kind": "directory",
            "identity": identity(path, follow_symlinks=False),
        }
    if path.is_file():
        return {
            "path": relative,
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "identity": identity(path),
        }
    raise ValueError(f"Unsupported file kind: {path}")


def scan_tree(base: Path) -> dict[str, object]:
    if not base.exists() or base.is_symlink():
        raise ValueError(f"Expected a real file or directory: {base}")
    records = [_record(base, base)]
    if base.is_dir():
        for current, directories, files in os.walk(base, followlinks=False):
            current_path = Path(current)
            for name in list(directories):
                path = current_path / name
                records.append(_record(path, base))
                if path.is_symlink():
                    directories.remove(name)
            for name in files:
                records.append(_record(current_path / name, base))
    records.sort(key=lambda row: (str(row["path"]), str(row["kind"])))
    stable_records = [
        {key: value for key, value in row.items() if key != "identity"}
        for row in records
    ]
    encoded = json.dumps(stable_records, sort_keys=True, separators=(",", ":")).encode()
    identity_encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "records": records,
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "identity_sha256": hashlib.sha256(identity_encoded).hexdigest(),
        "files": sum(row["kind"] == "file" for row in records),
        "directories": sum(row["kind"] == "directory" for row in records),
        "symlinks": sum(row["kind"] == "symlink" for row in records),
        "logical_bytes": sum(int(row.get("bytes", 0)) for row in records),
    }


def active_users(paths: Iterable[Path]) -> list[dict[str, object]]:
    roots = [str(path.absolute()) for path in paths]
    own = {os.getpid()}
    parent = os.getppid()
    while parent > 1:
        own.add(parent)
        try:
            parent = int(
                Path(f"/proc/{parent}/stat").read_text().split(") ", 1)[1].split()[1]
            )
        except (OSError, ValueError, IndexError):
            break

    def inside(value: str) -> bool:
        return any(value == root or value.startswith(root + "/") for root in roots)

    hits: list[dict[str, object]] = []
    proc = Path("/proc")
    if not proc.exists():
        return hits
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) in own:
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            values: list[str] = []
            for name in ("cwd", "exe"):
                try:
                    values.append(os.readlink(process / name))
                except OSError:
                    pass
            try:
                file_descriptors = list((process / "fd").iterdir())
            except (FileNotFoundError, PermissionError):
                file_descriptors = []
            for descriptor in file_descriptors:
                try:
                    values.append(os.readlink(descriptor))
                except OSError:
                    pass
            try:
                values.extend(
                    value.decode(errors="replace")
                    for value in (process / "cmdline").read_bytes().split(b"\0")
                    if value.startswith(b"/")
                )
            except (FileNotFoundError, PermissionError):
                pass
            matched = sorted({value for value in values if inside(value)})
            if matched:
                hits.append({"pid": int(process.name), "paths": matched})
        except FileNotFoundError:
            pass
        except PermissionError:
            hits.append({"pid": int(process.name), "error": "process cannot be inspected"})
    return hits


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def operation_id(root: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{git_head(root)[:8]}"


def build_quarantine_plan(
    root: Path,
    targets: Iterable[str],
    operation: str,
    *,
    tracked_checker: Callable[[Path, str], list[str]] = tracked_paths,
) -> dict[str, object]:
    validate_local_store(root)
    entries = []
    for name in dict.fromkeys(targets):
        source = validate_target(root, name)
        if not source.exists():
            entries.append({"source": name, "status": "missing"})
            continue
        tracked = tracked_checker(root, name)
        if tracked:
            raise ValueError(f"Target contains tracked paths: {name}: {tracked[:3]}")
        snapshot = scan_tree(source)
        destination = quarantine_root(root) / operation / name
        if os.path.lexists(destination):
            raise ValueError(f"Quarantine destination already exists: {destination}")
        entries.append(
            {
                "source": name,
                "destination": destination.relative_to(root).as_posix(),
                "status": "planned",
                "snapshot": snapshot,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "operation_id": operation,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(root),
        "state": "planned",
        "entries": entries,
    }


def quarantine(
    root: Path,
    targets: Iterable[str],
    *,
    execute: bool,
    operation: str | None = None,
    tracked_checker: Callable[[Path, str], list[str]] = tracked_paths,
    activity_checker: Callable[[Iterable[Path]], list[dict[str, object]]] = active_users,
    rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, object]:
    operation = operation or operation_id(root)
    plan = build_quarantine_plan(
        root, targets, operation, tracked_checker=tracked_checker
    )
    if not execute:
        return plan
    movable = [entry for entry in plan["entries"] if entry["status"] == "planned"]
    if not movable:
        plan["state"] = "no-op"
        return plan
    sources = [root / str(entry["source"]) for entry in movable]
    busy = activity_checker(sources)
    if busy:
        raise RuntimeError(f"Targets are in use: {busy}")
    for entry in movable:
        destination = root / str(entry["destination"])
        if os.path.lexists(destination):
            raise ValueError(f"Quarantine destination appeared after planning: {destination}")
    operation_root = quarantine_root(root) / operation
    operation_root.mkdir(parents=True, exist_ok=False)
    manifest_path = manifests_root(root) / f"{operation}.json"
    plan["manifest"] = manifest_path.relative_to(root).as_posix()
    write_json(manifest_path, plan)
    mover = rename or (lambda source, destination: source.rename(destination))
    try:
        for entry in movable:
            source = root / str(entry["source"])
            destination = root / str(entry["destination"])
            current = scan_tree(source)
            if current["identity_sha256"] != entry["snapshot"]["identity_sha256"]:
                raise RuntimeError(f"Source changed after planning: {entry['source']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(destination):
                raise ValueError(f"Quarantine destination appeared after planning: {destination}")
            if source.stat().st_dev != operation_root.stat().st_dev:
                raise RuntimeError(f"Cross-filesystem rename refused: {entry['source']}")
            mover(source, destination)
            moved = scan_tree(destination)
            if moved["content_sha256"] != entry["snapshot"]["content_sha256"]:
                raise RuntimeError(f"Moved content failed verification: {entry['source']}")
            entry["status"] = "moved"
            write_json(manifest_path, plan)
    except BaseException:
        plan["state"] = "partial"
        write_json(manifest_path, plan)
        raise
    plan["state"] = "complete"
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, plan)
    return plan


def load_manifest(root: Path, path: Path) -> tuple[Path, dict[str, object]]:
    validate_local_store(root)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError("Manifest may not be a symlink")
    resolved = candidate.resolve(strict=True)
    base = manifests_root(root).resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise ValueError("Manifest must be a regular file below .local-artifacts/manifests")
    value = json.loads(resolved.read_text())
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    if not isinstance(value.get("entries"), list):
        raise ValueError("Malformed manifest")
    return resolved, value


def restore(
    root: Path,
    manifest: Path,
    *,
    execute: bool,
    activity_checker: Callable[[Iterable[Path]], list[dict[str, object]]] = active_users,
) -> dict[str, object]:
    manifest_path, value = load_manifest(root, manifest)
    restorable = []
    for entry in value["entries"]:
        name = str(entry["source"])
        validate_target(root, name)
        if entry["status"] == "missing":
            continue
        source = root / name
        destination = root / str(entry["destination"])
        expected = quarantine_root(root) / str(value["operation_id"]) / name
        if destination.absolute() != expected.absolute():
            raise ValueError(f"Manifest destination mismatch: {name}")
        if entry["status"] == "restored":
            if not source.exists() or destination.exists():
                raise ValueError(f"Restored state does not match filesystem: {name}")
            if scan_tree(source)["content_sha256"] != entry["snapshot"]["content_sha256"]:
                raise ValueError(f"Restored content changed: {name}")
            continue
        if entry["status"] in {"planned", "not-moved"} and value.get("state") in {
            "partial", "restoring", "restore-partial", "restored"
        }:
            if (
                source.exists()
                and not os.path.lexists(destination)
                and scan_tree(source)["content_sha256"]
                == entry["snapshot"]["content_sha256"]
            ):
                if execute:
                    entry["status"] = "not-moved"
                continue
            raise ValueError(f"Unmoved entry does not match its partial manifest: {name}")
        if entry["status"] != "moved":
            raise ValueError(f"Entry is not restorable: {name}: {entry['status']}")
        if os.path.lexists(source):
            raise ValueError(f"Restore destination already exists: {name}")
        if not destination.exists() or destination.is_symlink():
            raise ValueError(f"Quarantined target is missing or invalid: {destination}")
        if scan_tree(destination)["content_sha256"] != entry["snapshot"]["content_sha256"]:
            raise ValueError(f"Quarantined content changed: {name}")
        restorable.append((entry, source, destination))
    preview = {
        "operation_id": value["operation_id"],
        "manifest": manifest_path.relative_to(root.resolve()).as_posix(),
        "state": "restore-preview",
        "targets": [str(entry["source"]) for entry, _, _ in restorable],
    }
    if not execute:
        return preview
    busy = activity_checker(destination for _, _, destination in restorable)
    if busy:
        raise RuntimeError(f"Quarantined targets are in use: {busy}")
    value["state"] = "restoring"
    write_json(manifest_path, value)
    try:
        for entry, source, destination in restorable:
            source.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(source)
            if scan_tree(source)["content_sha256"] != entry["snapshot"]["content_sha256"]:
                raise RuntimeError(f"Restored content failed verification: {entry['source']}")
            entry["status"] = "restored"
            write_json(manifest_path, value)
    except BaseException:
        value["state"] = "restore-partial"
        write_json(manifest_path, value)
        raise
    value["state"] = "restored"
    value["restored_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, value)
    preview["state"] = "restored"
    return preview


def path_inventory(path: Path) -> dict[str, object]:
    if not os.path.lexists(path):
        return {"present": False, "files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    if path.is_file() or path.is_symlink():
        stat = path.lstat()
        return {
            "present": True,
            "files": 1,
            "logical_bytes": stat.st_size,
            "allocated_bytes": getattr(stat, "st_blocks", 0) * 512,
        }
    files = logical = allocated = 0
    for current, directories, names in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            candidate = current_path / name
            if candidate.is_symlink():
                directories.remove(name)
                names.append(name)
        for name in names:
            candidate = current_path / name
            stat = candidate.lstat()
            files += 1
            logical += stat.st_size
            allocated += getattr(stat, "st_blocks", 0) * 512
    return {
        "present": True,
        "files": files,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
    }


def audit(root: Path) -> dict[str, object]:
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--short", "--branch"], text=True
    ).splitlines()
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventories = dict(zip(AUDIT_PATHS, pool.map(
            lambda name: path_inventory(root / name), AUDIT_PATHS
        )))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(root),
        "git_status": status,
        "paths": inventories,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit", help="print a read-only repository artifact inventory")
    quarantine_parser = commands.add_parser(
        "quarantine", help="preview or execute quarantine for approved targets"
    )
    quarantine_parser.add_argument(
        "--target", action="append", choices=ALLOWED_TARGETS,
        help="approved target; repeat as needed (default: all approved targets)",
    )
    quarantine_parser.add_argument("--execute", action="store_true")
    restore_parser = commands.add_parser(
        "restore", help="preview or execute restoration from a quarantine manifest"
    )
    restore_parser.add_argument("--manifest", type=Path, required=True)
    restore_parser.add_argument("--execute", action="store_true")
    return parser


def display_result(value: dict[str, object]) -> dict[str, object]:
    """Keep CLI output readable while full file records stay in the manifest."""
    if "entries" not in value:
        return value
    displayed = dict(value)
    displayed["entries"] = []
    for entry in value["entries"]:
        compact = {key: item for key, item in entry.items() if key != "snapshot"}
        snapshot = entry.get("snapshot")
        if snapshot:
            compact["snapshot"] = {
                key: snapshot[key]
                for key in (
                    "content_sha256", "identity_sha256", "files", "directories",
                    "symlinks", "logical_bytes",
                )
            }
        displayed["entries"].append(compact)
    return displayed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            result = audit(ROOT)
        elif args.command == "quarantine":
            result = quarantine(
                ROOT,
                args.target or ALLOWED_TARGETS,
                execute=args.execute,
            )
        else:
            result = restore(ROOT, args.manifest, execute=args.execute)
        print(json.dumps(display_result(result), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
