"""Zero-extend a continuous-v5/v6 Rainbow policy for V7 phase state."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_code.rainbow_lite_v7_agent.callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)


def migrate(source: Path, output: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    source_feature = payload.get("feature_id")
    source_dimensions = {
        "continuous-v5": 140,
        "continuous-v6-opponent-tracking": 160,
    }
    if payload.get("algorithm") != ALGORITHM or source_feature not in source_dimensions:
        raise ValueError("source must be a continuous-v5/v6 Rainbow-lite checkpoint")
    policy = {key: value.clone() for key, value in payload["policy"].items()}
    old = policy["trunk.0.weight"]
    copied_dimensions = source_dimensions[source_feature]
    if tuple(old.shape) != (128, copied_dimensions):
        raise ValueError("unexpected source first-layer shape")
    expanded = torch.zeros((128, 162), dtype=old.dtype)
    expanded[:, :copied_dimensions] = old
    policy["trunk.0.weight"] = expanded
    migrated = {
        "checkpoint_schema": payload["checkpoint_schema"], "algorithm": ALGORITHM,
        "actions": list(ACTIONS), "feature_id": FEATURE_ID, "feature_version": None,
        "feature_schema": FEATURE_SCHEMA, "network_spec": NETWORK_SPEC,
        "hyperparameters": HYPERPARAMETERS, "policy": policy,
        "migration": {"kind": f"{source_feature}-to-v7-zero-extended-policy", "source": str(source.resolve()), "copied_input_dimensions": copied_dimensions, "zero_initialized_input_dimensions": 162 - copied_dimensions},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(migrated, temporary)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(); migrate(args.source, args.output)


if __name__ == "__main__":
    main()
