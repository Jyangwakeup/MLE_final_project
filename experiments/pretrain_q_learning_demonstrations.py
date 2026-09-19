"""Apply trace-free 1-step Double-Q updates to a warm-start checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle

import numpy as np

from agent_code.learning_common.linear_agent import LinearTransition
from agent_code.optimized_double_q_lambda_demo_agent import callbacks
from experiments.q_learning_demo_data import sha256


def pretrain(dataset: Path, initial_checkpoint: Path, output: Path, *,
             passes=5, seed=20260919, learning_rate=0.02):
    dataset, initial_checkpoint, output = map(
        lambda value: Path(value).resolve(), (dataset, initial_checkpoint, output))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output}")
    data = np.load(dataset, allow_pickle=False)
    with initial_checkpoint.open("rb") as file:
        initial = pickle.load(file)
    expected = {
        "algorithm": callbacks.ALGORITHM,
        "feature_id": callbacks.FEATURE_ID,
        "feature_schema": callbacks.FEATURE_SCHEMA,
        "hyperparameters": callbacks.HYPERPARAMETERS,
        "reward_id": "r20_safe_credit_targeted_wait",
    }
    for key, value in expected.items():
        if initial.get(key) != value:
            raise ValueError(f"initial checkpoint has incompatible {key}")
    model = callbacks.make_model(int(initial.get("agent_seed", 11)))
    model.load_checkpoint(initial, training=True)
    traces_before = model.traces.copy()
    updates_before = model.updates
    learner_rng_before = copy.deepcopy(model.rng.bit_generator.state)
    demo_rng = np.random.default_rng(seed)
    losses = []
    indices = np.arange(len(data["states"]))
    for pass_index in range(int(passes)):
        pass_losses = []
        for index in demo_rng.permutation(indices):
            done = bool(data["dones"][index])
            transition = LinearTransition(
                data["states"][index], int(data["actions"][index]),
                float(data["rewards"][index]),
                None if done else data["next_states"][index], done,
                None if done else data["next_legal"][index],
            )
            pass_losses.append(model.observe_demonstration(
                transition, rng=demo_rng, learning_rate=learning_rate))
        losses.append({
            "pass": pass_index + 1,
            "mean_absolute_td_error": float(np.mean(pass_losses)),
        })
    if not np.array_equal(model.traces, traces_before):
        raise AssertionError("demonstration updates changed eligibility traces")
    if model.updates != updates_before:
        raise AssertionError("demonstration updates changed online update count")
    if model.rng.bit_generator.state != learner_rng_before:
        raise AssertionError("demonstration updates changed online learner RNG")
    payload = dict(initial)
    payload.update(model.checkpoint())
    payload["traces"] = np.zeros_like(model.traces)
    payload["demonstration"] = {
        "schema_version": "q-learning-demonstration-pretrain-v1",
        "dataset_sha256": sha256(dataset),
        "initial_checkpoint_sha256": sha256(initial_checkpoint),
        "passes": int(passes), "learning_rate": float(learning_rate),
        "transitions_per_pass": int(len(indices)),
        "total_demonstration_updates": int(passes) * int(len(indices)),
        "rng_seed": int(seed), "rng_state": demo_rng.bit_generator.state,
        "history": losses,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    output.with_suffix(".json").write_text(
        json.dumps(payload["demonstration"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return payload["demonstration"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(pretrain(
        args.dataset, args.initial_checkpoint, args.output,
        passes=args.passes, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
