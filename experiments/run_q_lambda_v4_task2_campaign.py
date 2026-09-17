"""Idempotent Slurm controller for the preregistered v4 Task 2 campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = PROJECT_ROOT / "runs" / "qlambda_v4_task2_campaign" / "campaign.json"
DEADLINE = datetime.fromisoformat("2026-09-20T12:00:00+02:00")
AGENT = "optimized_double_q_lambda_v4_agent"
ARM = {
    "r7": {
        "task1_config": "experiments/configs/optimized_double_q_lambda_v4_r7_task1.json",
        "screen_config": "experiments/configs/optimized_double_q_lambda_v4_r7_task2_100k.json",
        "extend_config": "experiments/configs/optimized_double_q_lambda_v4_r7_task2_200k.json",
    },
    "r10": {
        "task1_config": "experiments/configs/optimized_double_q_lambda_v4_r10_task1.json",
        "screen_config": "experiments/configs/optimized_double_q_lambda_v4_r10_task2_100k.json",
        "extend_config": "experiments/configs/optimized_double_q_lambda_v4_r10_task2_200k.json",
    },
    "r12": {
        "task1_config": "experiments/configs/optimized_double_q_lambda_v4_r12_reward_only_task1.json",
        "screen_config": "experiments/configs/optimized_double_q_lambda_v4_r12_reward_only_task2_100k.json",
        "extend_config": "experiments/configs/optimized_double_q_lambda_v4_r12_reward_only_task2_200k.json",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(*command: str) -> str:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return completed.stdout.strip()


def _submit(script: str, *arguments: str, dependency: Sequence[str] = ()) -> str:
    command = ["sbatch"]
    if dependency:
        command.append("--dependency=afterany:" + ":".join(dependency))
    output = _run(*command, script, *arguments)
    return output.rsplit(maxsplit=1)[-1]


def _schedule_controller(state_path: Path, dependencies: Sequence[str]) -> str:
    return _submit(
        "scripts/run_q_lambda_v4_campaign_controller_cpu.sh",
        str(state_path.resolve()), dependency=dependencies)


def _screen_run(label: str, seed: int, job: str) -> Path:
    return PROJECT_ROOT / "runs" / f"qlambda_v4_{label}_t2_100k_s{seed}_j{job}"


def _extension_run(label: str, seed: int, job: str) -> Path:
    return PROJECT_ROOT / "runs" / f"qlambda_v4_{label}_t2_200k_s{seed}_j{job}"


def _task1_run(label: str, seed: int, job: str) -> Path:
    return PROJECT_ROOT / "runs" / f"qlambda_v4_{label}_t1_s{seed}_j{job}"


def _job_state(job: str) -> str:
    output = _run("sacct", "-j", job, "--format=State", "-n", "-X", "-P")
    return output.splitlines()[0].split("|")[0].split()[0] if output else "UNKNOWN"


def _record_job(state: dict[str, Any], *, stage: str, label: str, seed: int,
                job: str, estimate: str, attempt: int) -> None:
    state.setdefault("jobs", []).append({
        "stage": stage, "label": label, "seed": seed, "job_id": job,
        "estimate": estimate, "attempt": attempt, "submitted_at": _now(),
    })


def _submit_screen(state: dict[str, Any], label: str, seed: int,
                   parent: Path, attempt: int = 1) -> str:
    arm = ARM[label]
    job = _submit(
        "scripts/run_optimized_double_q_lambda_v4_task2_screen_cpu.sh",
        str(parent), arm["task1_config"], arm["screen_config"], label, str(seed))
    _record_job(state, stage="screen", label=label, seed=seed, job=job,
                estimate="35-50 minutes", attempt=attempt)
    return job


def _selected_resume_source(run: Path, checkpoint: str) -> Path:
    stem = Path(checkpoint).stem
    source = run / "checkpoints" / "resume_snapshots" / stem
    if not (source / "resume" / "latest.json").is_file():
        raise ValueError(f"Selected checkpoint has no resumable snapshot: {source}")
    return source


def _submit_extension(state: dict[str, Any], label: str, seed: int,
                      screen_run: Path, selection: dict[str, Any]) -> str:
    winner = selection["winner"]
    source = _selected_resume_source(screen_run, winner["checkpoint"])
    audit = json.loads((screen_run / "runtime_migration.json").read_text(encoding="utf-8"))
    parent_t1 = Path(audit["parent_run"])
    promotion = json.loads((parent_t1 / "promotion_audit.json").read_text(encoding="utf-8"))
    parent_task2 = PROJECT_ROOT / "runs" / f"{screen_run.name}_parent_task2"
    arm = ARM[label]
    job = _submit(
        "scripts/run_optimized_double_q_lambda_v4_task2_extend_cpu.sh",
        str(source), promotion["stage_gate"], str(parent_task2),
        arm["task1_config"], arm["extend_config"], label, str(seed))
    _record_job(state, stage="extension", label=label, seed=seed, job=job,
                estimate="45-70 minutes", attempt=1)
    return job


def _run_ablation(state_path: Path, entries: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    output = state_path.parent / "ablation_selection.json"
    command = [str(PROJECT_ROOT / ".venv-compute" / "bin" / "python"),
               "-m", "experiments.select_q_lambda_v4_ablation"]
    for label, path in entries:
        command.extend(("--selection", label, str(path)))
    command.extend(("--output", str(output)))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.returncode not in (0, 2):
        raise RuntimeError(completed.stderr or completed.stdout)
    return json.loads(output.read_text(encoding="utf-8"))


def _retryable(job: str, run: Path) -> bool:
    state = _job_state(job)
    if state in {"TIMEOUT", "NODE_FAIL", "PREEMPTED", "CANCELLED", "OUT_OF_MEMORY"}:
        return True
    if state == "COMPLETED" and not (run / "best_task2_selection.json").is_file():
        return True
    return False


def start(state_path: Path, r7_parent: Path, r10_parent: Path) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    state: dict[str, Any] = {
        "schema_version": "qlambda-v4-task2-campaign-v1",
        "status": "running", "phase": "screens", "created_at": _now(),
        "deadline": DEADLINE.isoformat(), "max_infrastructure_retries": 1,
        "parents": {"r7": str(r7_parent.resolve()), "r10": str(r10_parent.resolve())},
        "jobs": [], "screen_jobs": {}, "history": [],
    }
    for label, parent in (("r7", r7_parent), ("r10", r10_parent)):
        state["screen_jobs"][label] = _submit_screen(state, label, 11, parent)
    controller = _schedule_controller(state_path, list(state["screen_jobs"].values()))
    state["controller_job"] = controller
    _write(state_path, state)
    return state


def _deadline_reached(state: dict[str, Any]) -> bool:
    return datetime.now(timezone.utc) >= datetime.fromisoformat(state["deadline"]).astimezone(timezone.utc)


def advance(state_path: Path) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "running":
        return state
    if _deadline_reached(state):
        state.update(status="deadline_reached", stopped_at=_now())
        _write(state_path, state)
        return state
    if state["phase"] == "screens":
        selections: list[tuple[str, Path]] = []
        dependencies: list[str] = []
        for label, job in list(state["screen_jobs"].items()):
            run = _screen_run(label, 11, job)
            selection = run / "best_task2_selection.json"
            if selection.is_file():
                selections.append((label, selection))
                continue
            attempts = sum(
                item["stage"] == "screen" and item["label"] == label
                for item in state["jobs"])
            if attempts <= state["max_infrastructure_retries"] and _retryable(job, run):
                new_job = _submit_screen(
                    state, label, 11, Path(state["parents"][label]), attempts + 1)
                state["screen_jobs"][label] = new_job
                dependencies.append(new_job)
            else:
                state.update(status="failed", failure={
                    "stage": "screen", "label": label, "job_id": job,
                    "slurm_state": _job_state(job), "retryable": False})
                _write(state_path, state)
                return state
        if dependencies:
            state["controller_job"] = _schedule_controller(state_path, dependencies)
            _write(state_path, state)
            return state
        ablation = _run_ablation(state_path, selections)
        if ablation["winner"] is None:
            # r12 gets its own formal Task 1 lineage before any Task 2 work.
            job = _submit(
                "scripts/run_optimized_double_q_lambda_v4_formal_task1_cpu.sh",
                ARM["r12"]["task1_config"], "r12", "11")
            _record_job(state, stage="task1_parent", label="r12", seed=11,
                        job=job, estimate="40-60 minutes", attempt=1)
            state.update(phase="r12_parent", r12_parent_job=job)
            state["controller_job"] = _schedule_controller(state_path, [job])
        else:
            winner = ablation["winner"]
            label = winner["label"]
            screen = _screen_run(label, 11, state["screen_jobs"][label])
            job = _submit_extension(state, label, 11, screen, ablation)
            state.update(phase="seed11_extension", winner_label=label,
                         seed11_extension_job=job)
            state["controller_job"] = _schedule_controller(state_path, [job])
        _write(state_path, state)
        return state

    if state["phase"] == "r12_parent":
        run = _task1_run("r12", 11, state["r12_parent_job"])
        audit = run / "promotion_audit.json"
        if not audit.is_file() or not json.loads(audit.read_text())["passed"]:
            state.update(status="failed", failure={"stage": "r12_parent"})
        else:
            job = _submit_screen(state, "r12", 11, run)
            state.update(phase="r12_screen", screen_jobs={"r12": job})
            state["controller_job"] = _schedule_controller(state_path, [job])
        _write(state_path, state)
        return state

    if state["phase"] == "r12_screen":
        job = state["screen_jobs"]["r12"]
        run = _screen_run("r12", 11, job)
        selection_path = run / "best_task2_selection.json"
        if not selection_path.is_file():
            state.update(status="failed", failure={"stage": "r12_screen", "job_id": job})
        else:
            selection = json.loads(selection_path.read_text())
            safe = [item for item in selection["candidates"] if item["task2"]["suicide_rate"] <= .05 and item["task2"]["survived_bomb_rate"] >= .95 and item["task1_retention"] >= .90]
            if not safe:
                state.update(status="completed_without_target", stopped_at=_now())
            else:
                # Reuse the registered ordering through a one-arm ablation.
                ablation = _run_ablation(state_path, [("r12", selection_path)])
                job = _submit_extension(state, "r12", 11, run, ablation)
                state.update(phase="seed11_extension", winner_label="r12",
                             seed11_extension_job=job)
                state["controller_job"] = _schedule_controller(state_path, [job])
        _write(state_path, state)
        return state

    if state["phase"] == "seed11_extension":
        label, job = state["winner_label"], state["seed11_extension_job"]
        run = _extension_run(label, 11, job)
        selection_path = run / "best_task2_selection.json"
        if not selection_path.is_file():
            state.update(status="failed", failure={"stage": "seed11_extension", "job_id": job})
        else:
            selection = json.loads(selection_path.read_text())
            if not selection.get("any_passed"):
                if label != "r12":
                    job = _submit(
                        "scripts/run_optimized_double_q_lambda_v4_formal_task1_cpu.sh",
                        ARM["r12"]["task1_config"], "r12", "11")
                    _record_job(state, stage="task1_parent", label="r12", seed=11,
                                job=job, estimate="40-60 minutes", attempt=1)
                    state.update(phase="r12_parent", r12_parent_job=job)
                    state["controller_job"] = _schedule_controller(state_path, [job])
                else:
                    state.update(status="completed_without_target", stopped_at=_now())
            else:
                parent_jobs = {}
                for seed in (22, 33):
                    parent_job = _submit(
                        "scripts/run_optimized_double_q_lambda_v4_formal_task1_cpu.sh",
                        ARM[label]["task1_config"], label, str(seed))
                    _record_job(state, stage="task1_parent", label=label, seed=seed,
                                job=parent_job, estimate="40-60 minutes", attempt=1)
                    parent_jobs[str(seed)] = parent_job
                state.update(phase="replication_parents", replication_parent_jobs=parent_jobs,
                             seed11_selection=str(selection_path),
                             seed11_extension_run=str(run))
                state["controller_job"] = _schedule_controller(state_path, list(parent_jobs.values()))
        _write(state_path, state)
        return state

    if state["phase"] == "replication_parents":
        label = state["winner_label"]
        screen_jobs: dict[str, str] = {}
        for seed_raw, parent_job in state["replication_parent_jobs"].items():
            seed = int(seed_raw)
            parent = _task1_run(label, seed, parent_job)
            audit_path = parent / "promotion_audit.json"
            if not audit_path.is_file() or not json.loads(audit_path.read_text())["passed"]:
                state.update(status="failed", failure={
                    "stage": "replication_parent", "seed": seed,
                    "job_id": parent_job})
                _write(state_path, state)
                return state
            screen_jobs[seed_raw] = _submit_screen(state, label, seed, parent)
        state.update(phase="replication_screens", replication_screen_jobs=screen_jobs)
        state["controller_job"] = _schedule_controller(state_path, list(screen_jobs.values()))
        _write(state_path, state)
        return state

    if state["phase"] == "replication_screens":
        label = state["winner_label"]
        extension_jobs: dict[str, str] = {}
        screen_runs: dict[str, str] = {}
        for seed_raw, screen_job in state["replication_screen_jobs"].items():
            seed = int(seed_raw)
            run = _screen_run(label, seed, screen_job)
            selection_path = run / "best_task2_selection.json"
            if not selection_path.is_file():
                state.update(status="failed", failure={
                    "stage": "replication_screen", "seed": seed,
                    "job_id": screen_job})
                _write(state_path, state)
                return state
            ablation = _run_ablation(
                state_path.with_name(f"seed_{seed}_campaign.json"),
                [(f"{label}_s{seed}", selection_path)])
            if ablation["winner"] is None:
                state.update(status="completed_without_replication", failure={
                    "stage": "replication_screen", "seed": seed,
                    "reason": "no_safe_candidate"}, stopped_at=_now())
                _write(state_path, state)
                return state
            # _submit_extension resolves configs by base reward label.
            ablation["winner"]["label"] = label
            extension_jobs[seed_raw] = _submit_extension(
                state, label, seed, run, ablation)
            screen_runs[seed_raw] = str(run)
        state.update(phase="replication_extensions",
                     replication_extension_jobs=extension_jobs,
                     replication_screen_runs=screen_runs)
        state["controller_job"] = _schedule_controller(
            state_path, list(extension_jobs.values()))
        _write(state_path, state)
        return state

    if state["phase"] == "replication_extensions":
        label = state["winner_label"]
        selections: list[tuple[str, Path]] = [
            (f"{label}_s11", Path(state["seed11_selection"]))]
        extension_runs: dict[str, str] = {"11": state["seed11_extension_run"]}
        for seed_raw, job in state["replication_extension_jobs"].items():
            seed = int(seed_raw)
            run = _extension_run(label, seed, job)
            selection_path = run / "best_task2_selection.json"
            if not selection_path.is_file():
                state.update(status="failed", failure={
                    "stage": "replication_extension", "seed": seed,
                    "job_id": job})
                _write(state_path, state)
                return state
            payload = json.loads(selection_path.read_text())
            if not payload.get("any_passed"):
                state.update(status="completed_without_replication", failure={
                    "stage": "replication_extension", "seed": seed,
                    "reason": "task2_gates_not_passed"}, stopped_at=_now())
                _write(state_path, state)
                return state
            selections.append((f"{label}_s{seed}", selection_path))
            extension_runs[seed_raw] = str(run)
        final_selection = _run_ablation(
            state_path.with_name("three_seed_campaign.json"), selections)
        winner = final_selection["winner"]
        if winner is None:
            state.update(status="completed_without_replication", stopped_at=_now())
            _write(state_path, state)
            return state
        winner_seed = winner["label"].rsplit("s", 1)[-1]
        checkpoint = winner["checkpoint"]
        if winner_seed == "11":
            # Seed 11's screen parent establishes both frozen baselines.
            seed_screen_job = state["screen_jobs"][label]
            screen_run = _screen_run(label, 11, seed_screen_job)
        else:
            screen_run = Path(state["replication_screen_runs"][winner_seed])
        parent_task2 = PROJECT_ROOT / "runs" / f"{screen_run.name}_parent_task2"
        audit = json.loads((screen_run / "runtime_migration.json").read_text())
        parent = Path(audit["parent_run"])
        promotion = json.loads((parent / "promotion_audit.json").read_text())
        main_job = _submit(
            "scripts/run_optimized_double_q_lambda_v4_task2_main_validation_cpu.sh",
            checkpoint, ARM[label]["task1_config"], ARM[label]["extend_config"],
            promotion["stage_gate"], str(parent_task2),
            f"{label}_s{winner_seed}")
        _record_job(state, stage="main_validation", label=label,
                    seed=int(winner_seed), job=main_job,
                    estimate="20-35 minutes", attempt=1)
        state.update(phase="main_validation", main_validation_job=main_job,
                     main_candidate=winner, three_seed_selection=final_selection,
                     main_validation_consumed=True)
        state["controller_job"] = _schedule_controller(state_path, [main_job])
        _write(state_path, state)
        return state

    if state["phase"] == "main_validation":
        job = state["main_validation_job"]
        matches = sorted((PROJECT_ROOT / "runs").glob(f"*_main100_j{job}_main_validation.json"))
        if len(matches) != 1:
            state.update(status="failed", failure={
                "stage": "main_validation", "job_id": job,
                "reason": "missing_or_ambiguous_result"})
        else:
            payload = json.loads(matches[0].read_text())
            state.update(
                status="accepted" if payload.get("any_passed") else "main_validation_failed",
                stopped_at=_now(), main_validation=str(matches[0]))
        _write(state_path, state)
        return state

    state.update(status="failed", failure={
        "reason": "unknown_campaign_phase", "phase": state["phase"]},
        stopped_at=_now())
    _write(state_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    sub = parser.add_subparsers(dest="command", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("--r7-parent", required=True, type=Path)
    start_parser.add_argument("--r10-parent", required=True, type=Path)
    sub.add_parser("advance")
    args = parser.parse_args(argv)
    state = (
        start(args.state.resolve(), args.r7_parent, args.r10_parent)
        if args.command == "start" else advance(args.state.resolve()))
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state.get("status") not in {"failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
