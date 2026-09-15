"""Build a self-contained tournament archive for one learned agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil
import tempfile
import zipfile

import torch

from experiments.agent_contracts import resolve_agent_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {
    "double_q_compact_agent", "double_dqn_continuous_agent",
    "double_dqn_continuous_v2_agent", "double_dqn_continuous_v3_agent",
    "cnn_double_dqn_agent", "hybrid_dueling_double_dqn_agent",
}


def _checkpoint_payload(path: Path, algorithm: str):
    if algorithm == "double_q_learning":
        with path.open("rb") as file:
            return pickle.load(file)
    return torch.load(path, map_location="cpu", weights_only=True)


def _rewrite_imports(path: Path, replacements: dict[str, str]) -> None:
    for source in path.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        source.write_text(text, encoding="utf-8")


def build_submission(agent: str, checkpoint: Path, output: Path) -> Path:
    if agent not in SUPPORTED:
        raise ValueError(f"self-contained packaging is not supported for {agent!r}")
    contract = resolve_agent_contract(agent)
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.suffix != ".zip":
        raise ValueError("submission output must use the .zip extension")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite submission archive: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    payload = _checkpoint_payload(checkpoint, contract.algorithm)
    expected = {
        "algorithm": contract.algorithm, "feature_id": contract.feature_id,
        "feature_schema": contract.feature_schema,
        "actions": contract.feature_schema["action_order"],
        "network_spec": contract.network_spec,
        "hyperparameters": contract.hyperparameters,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"checkpoint {field} is incompatible with selected agent")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bomberman-submission-") as temporary:
        stage = Path(temporary) / agent
        shutil.copytree(
            PROJECT_ROOT / "agent_code" / agent, stage,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "final.pkl", "final.pt", "logs"),
        )
        vendor = stage / "_vendor"
        vendor.mkdir()
        (vendor / "__init__.py").write_text("\"\"\"Vendored runtime dependencies.\"\"\"\n")
        shutil.copytree(
            PROJECT_ROOT / "agent_code" / "learning_common", vendor / "learning_common",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(
            PROJECT_ROOT / "agent_code" / "dqn_agent" / "model.py",
            vendor / "dqn_model.py",
        )
        team = vendor / "team_agent"
        team.mkdir()
        for name in (
            "__init__.py", "danger.py", "exploration.py", "features.py", "rewards.py",
            "opponent_transitions.py", "safety.py", "temporal_safety_features.py",
        ):
            shutil.copy2(PROJECT_ROOT / "agent_code" / "team_agent" / name, team / name)
        shutil.copytree(
            PROJECT_ROOT / "agent_code" / "team_agent" / "feature_system",
            team / "feature_system", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(checkpoint, stage / contract.checkpoint_name)
        _rewrite_imports(stage, {
            "from agent_code.dqn_agent.model": (
                f"from agent_code.{agent}._vendor.dqn_model"
            ),
            "from agent_code.learning_common": f"from agent_code.{agent}._vendor.learning_common",
            "from agent_code.team_agent": f"from agent_code.{agent}._vendor.team_agent",
        })
        _rewrite_imports(vendor / "learning_common", {
            f"from agent_code.{agent}._vendor.team_agent": "from ..team_agent",
        })
        manifest = {
            "agent": agent, "algorithm": contract.algorithm,
            "feature_id": contract.feature_id, "reward_id": payload["reward_id"],
            "checkpoint": contract.checkpoint_name,
        }
        (stage / "SUBMISSION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(agent) / path.relative_to(stage))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, choices=sorted(SUPPORTED))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(build_submission(args.agent, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
