"""Audited runner-only migration for an otherwise immutable curriculum parent."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.resume import validate_resume_transition


RUNTIME_MIGRATION_VERSION = "runtime-migration-v1"
_ALLOWED_EXACT = frozenset({
    "experiments/run.py",
    "experiments/training.py",
    "experiments/resume.py",
    "experiments/runtime_migration.py",
    "experiments/task2_success_stopping.py",
    "experiments/run_q_lambda_v4_task2_campaign.py",
    "experiments/select_task2_snapshots.py",
})
_ALLOWED_PREFIXES = (
    "experiments/configs/",
    "scripts/",
    "tests/",
    "docs/",
)
_ALLOWED_AGENT_DOC = (
    "agent_code/optimized_double_q_lambda_v4_agent/EXPERIMENT_LOG.md",
    "agent_code/optimized_double_q_lambda_v4_agent/README.md",
)
_IGNORED_DIRTY_PREFIXES = (".vscode/",)


def path_is_allowed(path: str) -> bool:
    normalized = Path(path).as_posix()
    return (
        normalized in _ALLOWED_EXACT
        or normalized in _ALLOWED_AGENT_DOC
        or normalized.startswith(_ALLOWED_PREFIXES)
    )


def validate_changed_paths(paths: Sequence[str]) -> list[str]:
    normalized = sorted({Path(path).as_posix() for path in paths if path})
    rejected = [path for path in normalized if not path_is_allowed(path)]
    if rejected:
        raise ValueError(
            "Runner-only migration rejects runtime changes: " + ", ".join(rejected))
    return normalized


def _git(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=project_root, text=True, capture_output=True)
    if completed.returncode:
        raise ValueError("Runtime migration requires Git provenance: " + completed.stderr.strip())
    return completed.stdout.strip()


def _dirty_runtime_paths(project_root: Path) -> list[str]:
    output = _git(project_root, "status", "--porcelain", "--untracked-files=all")
    paths: list[str] = []
    for line in output.splitlines():
        raw = line[3:]
        path = raw.split(" -> ")[-1]
        if path.startswith(_IGNORED_DIRTY_PREFIXES):
            continue
        if path.startswith(("runs/",)) or path.endswith(".txt") or path.endswith(".zip"):
            continue
        if path in {"slurm-en.md"} or path.startswith((".venv", "__pycache__/")):
            continue
        paths.append(path)
    return sorted(paths)


def build_runtime_migration_audit(
    *, project_root: Path, parent_run: Path, parent_contract: Mapping[str, Any],
    child_contract: Mapping[str, Any], parent_status: str,
) -> dict[str, Any]:
    """Validate a direct Task1->Task2 promotion with runner-only source drift."""
    project_root = Path(project_root).resolve()
    parent_run = Path(parent_run).resolve()
    promotion_path = parent_run / "promotion_audit.json"
    if not promotion_path.is_file():
        raise ValueError("Runtime migration requires promotion_audit.json")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if not promotion.get("passed"):
        raise ValueError("Runtime migration requires a passed Task 1 promotion audit")
    if parent_contract.get("task") != "coin_navigation" or child_contract.get("task") != "crate_navigation":
        raise ValueError("Runtime migration supports only direct Task 1 to Task 2 promotion")

    # Reuse the ordinary strict contract validator while substituting only the
    # two source identity fields whose runner-only drift this audit explains.
    comparable_child = {
        **dict(child_contract),
        "source_hash": parent_contract.get("source_hash"),
        "source_hash_scope": parent_contract.get("source_hash_scope"),
    }
    resume_kind = validate_resume_transition(
        dict(parent_contract), comparable_child, parent_status=parent_status)
    if resume_kind != "next_task":
        raise ValueError("Runtime migration must be a curriculum promotion")

    parent_commit = parent_contract.get("source_commit")
    if not parent_commit:
        raise ValueError("Runtime migration parent has no source commit")
    current_commit = _git(project_root, "rev-parse", "HEAD")
    committed = _git(project_root, "diff", "--name-only", str(parent_commit), current_commit)
    changed_paths = validate_changed_paths(committed.splitlines())
    dirty_paths = _dirty_runtime_paths(project_root)
    if dirty_paths:
        raise ValueError(
            "Runtime migration requires committed relevant changes: "
            + ", ".join(dirty_paths))
    return {
        "version": RUNTIME_MIGRATION_VERSION,
        "parent_run": str(parent_run),
        "parent_commit": str(parent_commit),
        "current_commit": current_commit,
        "parent_source_hash": parent_contract.get("source_hash"),
        "current_source_hash": child_contract.get("source_hash"),
        "source_hash_scope": child_contract.get("source_hash_scope"),
        "changed_paths": changed_paths,
        "promotion_audit": str(promotion_path),
        "promotion_audit_sha256": promotion.get("checkpoint_sha256"),
        "passed": True,
    }
