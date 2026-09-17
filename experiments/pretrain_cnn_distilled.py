"""Pretrain one CNN candidate on frozen continuous-v2 teacher logits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_code.cnn_distilled_double_dqn_agent.callbacks import (
    ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)
from agent_code.cnn_distilled_double_dqn_agent.distillation import masked_kl
from agent_code.cnn_distilled_double_dqn_agent.learner import DistilledDoubleDQNLearner
from agent_code.cnn_distilled_double_dqn_agent.model import build_network
from agent_code.cnn_distilled_double_dqn_agent.symmetry import (
    TRANSFORMS, transform_actions, transform_board,
)
from agent_code.learning_common.runtime import CHECKPOINT_SCHEMA
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.team_agent.rewards import resolve_reward_spec
from agent_code.team_agent.safety import resolve_safety_spec


def _sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def _augment(boards, legal, teacher_q, names):
    transformed_boards, transformed_legal, transformed_q = [], [], []
    for board, mask, values, name in zip(boards, legal, teacher_q, names):
        transformed_boards.append(transform_board(board, name))
        transformed_legal.append(transform_actions(mask, name))
        transformed_q.append(transform_actions(values, name))
    return np.stack(transformed_boards), np.stack(transformed_legal), np.stack(transformed_q)


def pretrain(dataset_path, output, architecture, *, seed=11, epochs=30,
             patience=5, batch_size=64):
    dataset_path, output = Path(dataset_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output}")
    payload = np.load(dataset_path, allow_pickle=False)
    boards = np.asarray(payload["boards"], dtype=np.float32)
    legal = np.asarray(payload["legal_masks"], dtype=bool)
    teacher_q = np.asarray(payload["teacher_q"], dtype=np.float32)
    seeds = np.asarray(payload["environment_seeds"], dtype=np.int64)
    train_indices = np.flatnonzero((seeds >= 6000) & (seeds <= 6079))
    validation_indices = np.flatnonzero((seeds >= 6080) & (seeds <= 6099))
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("dataset must contain train seeds 6000-6079 and val seeds 6080-6099")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    network = build_network(architecture).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=HYPERPARAMETERS["learning_rate"])
    rng = np.random.default_rng(seed)
    history, best_loss, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_indices)
        network.train()
        losses = []
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            batch_boards, batch_legal, batch_q = boards[indices], legal[indices], teacher_q[indices]
            if architecture == "action_aligned_d4":
                names = rng.choice(TRANSFORMS, size=len(indices))
                batch_boards, batch_legal, batch_q = _augment(
                    batch_boards, batch_legal, batch_q, names)
            board_tensor = torch.as_tensor(batch_boards, dtype=torch.float32, device=device)
            legal_tensor = torch.as_tensor(batch_legal, dtype=torch.bool, device=device)
            q_tensor = torch.as_tensor(batch_q, dtype=torch.float32, device=device)
            loss = masked_kl(network(board_tensor), q_tensor, legal_tensor, temperature=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))
        network.eval()
        validation_losses = []
        with torch.no_grad():
            for start in range(0, len(validation_indices), batch_size):
                indices = validation_indices[start:start + batch_size]
                validation_losses.append(float(masked_kl(
                    network(torch.as_tensor(boards[indices], dtype=torch.float32, device=device)),
                    torch.as_tensor(teacher_q[indices], dtype=torch.float32, device=device),
                    torch.as_tensor(legal[indices], dtype=torch.bool, device=device),
                    temperature=1.0).item()))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-7:
            best_loss, best_state, stale = validation_loss, copy.deepcopy(network.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    network.load_state_dict(best_state)
    learner = DistilledDoubleDQNLearner(
        build_network(architecture), build_network(architecture),
        hyperparameters=HYPERPARAMETERS, seed=seed)
    learner.policy.load_state_dict(network.to("cpu").state_dict())
    learner.target.load_state_dict(learner.policy.state_dict())
    checkpoint = learner.checkpoint()
    teacher_hash = str(payload["teacher_checkpoint_sha256"].item())
    checkpoint.update({
        "checkpoint_schema": CHECKPOINT_SCHEMA, "algorithm": ALGORITHM,
        "actions": list(ACTIONS), "feature_id": FEATURE_ID,
        "feature_schema": FEATURE_SCHEMA, "reward_id": "r5_conditional_loop",
        "reward_version": "r5_conditional_loop",
        "reward_spec": resolve_reward_spec("r5_conditional_loop"),
        "hyperparameters": HYPERPARAMETERS, "network_spec": NETWORK_SPEC,
        "model_architecture": architecture, "action_steps": 0,
        "total_action_steps": 0, "stage_action_steps": 0, "agent_seed": seed,
        "agent_rng_state": random.Random(seed).getstate(),
        "exploration_spec": {"version": "linear-v1", "start": 0.2,
                             "end": 0.05, "decay_action_steps": 25000},
        "safety_spec": resolve_safety_spec({"version": "survival-mask-v1",
                                              "mode": "off", "horizon": 7,
                                              "fallback": "physical_q"}),
        "n_step": 1, "training_task": "coin_navigation",
        "distillation": {"dataset_sha256": _sha256(dataset_path),
                           "teacher_checkpoint_sha256": teacher_hash,
                           "train_seeds": [6000, 6079], "validation_seeds": [6080, 6099],
                           "temperature": 1.0, "best_validation_loss": best_loss,
                           "epochs_completed": len(history)},
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", required=True,
                        choices=("global", "action_aligned", "action_aligned_d4"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    print(pretrain(args.dataset, args.output, args.architecture,
                   seed=args.seed, epochs=args.epochs, patience=args.patience))


if __name__ == "__main__":
    main()
