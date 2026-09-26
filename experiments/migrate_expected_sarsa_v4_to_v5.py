"""Create a policy-only continuous-v5 warm start from a v4 SARSA checkpoint."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from all_other_agent_code.expected_sarsa_lambda_v5_agent.callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS,
)


def migrate(source: Path, output: Path, seed: int = 11) -> None:
    with source.open("rb") as file:
        payload = pickle.load(file)
    if payload.get("algorithm") != ALGORITHM or payload.get("feature_id") != "continuous-v4":
        raise ValueError("source must be a continuous-v4 Expected SARSA checkpoint")
    old_offsets = np.asarray(payload["tile_coder"]["offsets"], dtype=np.float32)
    old_primes = np.asarray(payload["tile_coder"]["primes"], dtype=np.int64)
    if old_offsets.shape != (8, 41) or old_primes.shape != (41,):
        raise ValueError("unexpected v4 projected tile-coder shape")
    rng = np.random.default_rng(seed)
    offsets = np.concatenate(
        (old_offsets, rng.random((8, 4), dtype=np.float32)), axis=1)
    primes = np.concatenate(
        (old_primes, rng.integers(1, 2**31 - 1, size=4, dtype=np.int64) | 1))
    migrated = {
        "checkpoint_schema": payload["checkpoint_schema"],
        "algorithm": ALGORITHM,
        "actions": list(ACTIONS),
        "feature_id": FEATURE_ID,
        "feature_version": None,
        "feature_schema": FEATURE_SCHEMA,
        "network_spec": None,
        "hyperparameters": HYPERPARAMETERS,
        "weights": np.asarray(payload["weights"], dtype=np.float32).copy(),
        "traces": np.zeros_like(payload["weights"], dtype=np.float32),
        "tile_coder": {"offsets": offsets, "primes": primes},
        "migration": {
            "kind": "continuous-v4-to-v5-projected-tile-policy",
            "source": str(source.resolve()),
            "copied_projected_dimensions": 41,
            "new_projected_dimensions": 4,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(migrated, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    migrate(args.source, args.output, args.seed)


if __name__ == "__main__":
    main()
