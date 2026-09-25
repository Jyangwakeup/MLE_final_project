"""Zero-extend a continuous-v7 Rainbow policy for V10 quadrant density."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_code.rainbow_lite_v10_agent.callbacks import (  # noqa: E402
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)


SOURCE_FEATURE_ID = "continuous-v7-phase-aware"
SOURCE_DIM = 162
TARGET_DIM = 170


def migrate(source: Path, output: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("algorithm") != ALGORITHM
        or payload.get("feature_id") != SOURCE_FEATURE_ID
    ):
        raise ValueError("source must be a continuous-v7 Rainbow-lite checkpoint")
    policy = {key: value.clone() for key, value in payload["policy"].items()}
    old = policy["trunk.0.weight"]
    if tuple(old.shape) != (128, SOURCE_DIM):
        raise ValueError("unexpected source first-layer shape")
    expanded = torch.zeros((128, TARGET_DIM), dtype=old.dtype)
    expanded[:, :SOURCE_DIM] = old
    policy["trunk.0.weight"] = expanded
    migrated = {
        "checkpoint_schema": payload["checkpoint_schema"],
        "algorithm": ALGORITHM,
        "actions": list(ACTIONS),
        "feature_id": FEATURE_ID,
        "feature_version": None,
        "feature_schema": FEATURE_SCHEMA,
        "network_spec": NETWORK_SPEC,
        "hyperparameters": HYPERPARAMETERS,
        "policy": policy,
        "migration": {
            "kind": "continuous-v7-to-v10-zero-extended-policy",
            "source": str(source.resolve()),
            "copied_input_dimensions": SOURCE_DIM,
            "zero_initialized_input_dimensions": TARGET_DIM - SOURCE_DIM,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(migrated, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    migrate(args.source, args.output)


if __name__ == "__main__":
    main()
