"""Create a policy-only spatial warm start from a v6 Rainbow checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from all_other_agent_code.rainbow_lite_spatial_v6_agent.callbacks import (  # noqa: E402
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)
from all_other_agent_code.rainbow_lite_spatial_v6_agent.model import SpatialDuelingNetwork  # noqa: E402


def migrate(source: Path, output: Path) -> None:
    parent = torch.load(source, map_location="cpu", weights_only=True)
    if parent.get("algorithm") != ALGORITHM or parent.get("feature_id") != "continuous-v6-opponent-tracking":
        raise ValueError("source must be a continuous-v6 Rainbow-lite checkpoint")
    old = parent["policy"]
    model = SpatialDuelingNetwork()
    policy = model.state_dict()
    for key in ("trunk.0.weight", "trunk.0.bias", "trunk.2.weight",
                "trunk.2.bias", "value.weight", "value.bias",
                "advantage.weight", "advantage.bias"):
        if key not in old or old[key].shape != policy[key].shape:
            raise ValueError(f"source policy is incompatible at {key}")
        policy[key] = old[key].clone()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save({
        "checkpoint_schema": parent.get("checkpoint_schema"),
        "algorithm": ALGORITHM, "actions": list(ACTIONS),
        "feature_id": FEATURE_ID, "feature_version": None,
        "feature_schema": FEATURE_SCHEMA, "network_spec": NETWORK_SPEC,
        "hyperparameters": HYPERPARAMETERS, "policy": policy,
        "migration": {"kind": "v6-to-spatial-policy-only", "source": str(source.resolve()),
                      "replay_and_optimizer": "reset"},
    }, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    migrate(args.source, args.output)


if __name__ == "__main__":
    main()
