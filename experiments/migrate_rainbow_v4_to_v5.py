"""Create a policy-only continuous-v5 warm start from a v4 Rainbow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_code.rainbow_lite_v5_agent.callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)


def migrate(source: Path, output: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("algorithm") != ALGORITHM or payload.get("feature_id") != "continuous-v4":
        raise ValueError("source must be a continuous-v4 Rainbow-lite checkpoint")
    policy = {key: value.clone() for key, value in payload["policy"].items()}
    old = policy["trunk.0.weight"]
    expected = int(NETWORK_SPEC["input_shape"][0])
    if tuple(old.shape) != (128, 126) or expected < 126:
        raise ValueError("unexpected source or target first-layer shape")
    expanded = torch.zeros((old.shape[0], expected), dtype=old.dtype)
    expanded[:, :old.shape[1]] = old
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
            "kind": "continuous-v4-to-v5-zero-extended-policy",
            "source": str(source.resolve()),
            "copied_input_dimensions": 126,
            "zero_initialized_input_dimensions": expected - 126,
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
