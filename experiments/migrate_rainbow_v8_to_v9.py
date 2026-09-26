"""Compact a continuous-v8 Rainbow policy without changing its Q outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from all_other_agent_code.rainbow_lite_v9_agent.callbacks import (  # noqa: E402
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)
from agent_code.team_agent.feature_system import continuous_v8, continuous_v9  # noqa: E402


SOURCE_FEATURE_ID = continuous_v8.FEATURE_ID
SOURCE_DIM = continuous_v8.FEATURE_DIM
TARGET_DIM = continuous_v9.FEATURE_DIM

# Each removed alias is folded into the retained field with the same value.
ALIASES = {
    "can_drop_bomb": "bomb_legal",
    "escape_after_bomb": "bomb_survives_horizon",
    "bomb_tracked_opponent_blast_line": "wait_tracked_opponent_blast_line",
}
CONSTANT_ONES = ("wait_legal",)


def _compact_first_layer(policy: dict[str, torch.Tensor]) -> None:
    old_weight = policy["trunk.0.weight"]
    old_bias = policy["trunk.0.bias"]
    if tuple(old_weight.shape) != (128, SOURCE_DIM):
        raise ValueError("unexpected source first-layer shape")
    source_index = {
        name: index for index, name in enumerate(continuous_v8.VECTOR_FIELDS)
    }
    target_index = {
        name: index for index, name in enumerate(continuous_v9.VECTOR_FIELDS)
    }
    weight = old_weight[:, list(continuous_v9.KEEP_INDICES)].clone()
    for removed, retained in ALIASES.items():
        weight[:, target_index[retained]] += old_weight[:, source_index[removed]]
    bias = old_bias.clone()
    for name in CONSTANT_ONES:
        bias += old_weight[:, source_index[name]]
    policy["trunk.0.weight"] = weight
    policy["trunk.0.bias"] = bias


def migrate(source: Path, output: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("algorithm") != ALGORITHM
        or payload.get("feature_id") != SOURCE_FEATURE_ID
    ):
        raise ValueError("source must be a continuous-v8 Rainbow-lite checkpoint")
    policy = {key: value.clone() for key, value in payload["policy"].items()}
    _compact_first_layer(policy)
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
            "kind": "continuous-v8-to-v9-information-preserving-compaction",
            "source": str(source.resolve()),
            "source_input_dimensions": SOURCE_DIM,
            "target_input_dimensions": TARGET_DIM,
            "removed_fields": sorted(continuous_v9.DROPPED_FIELDS),
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
