import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent_code.dqn_agent.model import ReplayBuffer, Transition
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.learning_common.training_spec import (
    n_step_from_environment, resolve_safety_replay_spec,
)
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.rewards import resolve_reward_spec
from agent_code.team_agent.temporal_safety_features import (
    robust_routes_after_first_step,
)
from experiments.resume import (
    CHECKPOINT_SCHEMA_VERSION, LoadedSnapshot,
    materialize_task3_safety_checkpoint, validate_task3_safety_transfer,
)


ROOT = Path(__file__).resolve().parents[1]
V1 = {"version": "survival-mask-v1", "mode": "all", "horizon": 7,
      "fallback": "physical_q"}
V3 = {"version": "survival-mask-v3", "mode": "all", "horizon": 7,
      "required_independent_routes": 2,
      "robust_fallback": "survival-mask-v1", "fallback": "physical_q"}
V4 = {
    "version": "survival-mask-v4", "mode": "all", "horizon": 7,
    "required_independent_routes": 2,
    "robust_fallback": "survival-mask-v1",
    "opponent_transition_horizon": 1,
    "opponent_action_space": "all_physical",
    "include_opponent_bombs": True, "execution_orders": "all",
    "minimum_scenario_routes": 1,
    "opponent_robust_fallback": "survival-mask-v3",
    "fallback": "physical_q",
}


def contract(task, schema, safety):
    return {
        "algorithm": "double_dqn", "seed": 22, "task": task,
        "feature_id": "continuous-v2",
        "feature_schema": feature_schema_contract("continuous-v2"),
        "reward_id": "r7_safe_credit_sparse",
        "reward_version": "r7_safe_credit_sparse",
        "reward_spec": resolve_reward_spec("r7_safe_credit_sparse"),
        "checkpoint_schema": schema, "actions": list(ACTIONS),
        "training_device_name": None, "training_device_type": "cpu",
        "agent_seed": 22, "n_step": 4,
        "retention_spec": {"parent_fraction": 0.75, "distillation_weight": 2.0,
                           "temperature": 1.0, "per_task_capacity": 20000,
                           "current_warmup": 2000},
        "network_spec": {"input_shape": [84]}, "hyperparameters": {"gamma": 0.95},
        "safety_spec": safety,
    }


class Task3EscapeContractTests(unittest.TestCase):
    def test_agent_accepts_preregistered_five_step_environment_contract(self):
        with patch.dict("os.environ", {"BOMBERMAN_N_STEP": "5"}):
            self.assertEqual(n_step_from_environment(), 5)

    def test_five_steps_assign_self_death_to_the_bomb_action(self):
        rewards = [-0.01, -0.01, -0.01, -0.01, -20.01]

        def first_return(n_step):
            accumulator = NStepAccumulator(n_step, 0.95)
            emitted = []
            for index, reward in enumerate(rewards):
                done = index == len(rewards) - 1
                emitted.extend(accumulator.append(Transition(
                    np.asarray([index], dtype=np.float32),
                    ACTIONS.index("BOMB") if index == 0 else ACTIONS.index("WAIT"),
                    reward,
                    None if done else np.asarray([index + 1], dtype=np.float32),
                    done,
                    None if done else np.ones(len(ACTIONS), dtype=bool),
                    np.ones(len(ACTIONS), dtype=bool),
                    "weak_opponents",
                    1,
                    "ordinary",
                )))
            return emitted[0]

        four_step = first_return(4)
        five_step = first_return(5)
        self.assertFalse(four_step.done)
        self.assertAlmostEqual(four_step.reward, -0.03709875)
        self.assertTrue(five_step.done)
        self.assertAlmostEqual(five_step.reward, -16.3353688125)

    def test_route_count_uses_vertex_capacity_and_caps_at_two(self):
        blocked = np.ones((8, 7, 7), dtype=bool)
        blocked[:, 1:6, 1:6] = False
        danger = np.zeros_like(blocked)
        result = robust_routes_after_first_step(
            (3, 3), "WAIT", danger, blocked, required_routes=2)
        self.assertEqual(result.independent_routes, 2)
        danger[7] = True
        danger[7, 3, 3] = False
        bottleneck = robust_routes_after_first_step(
            (3, 3), "WAIT", danger, blocked, required_routes=2)
        self.assertEqual(bottleneck.independent_routes, 1)

    def test_transfer_is_narrow_and_explicit(self):
        parent = contract("crate_navigation", "training-resume-v7", V1)
        child = contract("weak_opponents", CHECKPOINT_SCHEMA_VERSION, V4)
        validate_task3_safety_transfer(parent, child, parent_status="completed")
        credit_child = {**contract(
            "weak_opponents", CHECKPOINT_SCHEMA_VERSION, V1), "n_step": 5}
        validate_task3_safety_transfer(
            parent, credit_child, parent_status="completed")
        with self.assertRaisesRegex(ValueError, "Task 2 directly"):
            validate_task3_safety_transfer(
                parent, {**child, "task": "full_match"}, parent_status="completed")
        with self.assertRaisesRegex(ValueError, "reward_id"):
            validate_task3_safety_transfer(
                {**parent, "reward_id": "r1"}, child, parent_status="completed")

    def test_materialization_preserves_learner_and_resets_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "learner.pt"
            policy = {"weight": torch.arange(4)}
            payload = {
                "checkpoint_schema": "training-resume-v7", "policy": policy,
                "target": {"weight": torch.arange(4) + 1},
                "optimizer": {"state": {1: {"step": torch.tensor(9)}}},
                "replay": {"marker": "kept"}, "teacher": {"weight": torch.ones(4)},
                "torch_rng_state": torch.get_rng_state(),
                "agent_rng_state": random.Random(22).getstate(),
                "total_action_steps": 1234, "stage_action_steps": 999,
                "n_step": 4, "hyperparameters": {"gamma": 0.95},
            }
            torch.save(payload, source)
            snapshot = LoadedSnapshot(
                root, "generation-1", "hash", "double_dqn", "crate_navigation",
                22, 1, "814173b", "source", {"contract": {}}, None, source)
            destination = root / "child.pt"
            safety_replay = resolve_safety_replay_spec()
            materialize_task3_safety_checkpoint(
                snapshot, destination, safety_spec=V4,
                exploration_spec={"version": "linear-v1", "start": 0.3,
                                  "end": 0.05, "decay_action_steps": 120000},
                training_budget={"target_stage_action_steps": None, "min_rounds": 1},
                safety_replay_spec=safety_replay, n_step=5)
            child = torch.load(destination, map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(child["policy"]["weight"], policy["weight"]))
            self.assertEqual(child["optimizer"]["state"][1]["step"], 9)
            self.assertEqual(child["replay"], payload["replay"])
            self.assertTrue(torch.equal(
                child["teacher"]["weight"], payload["teacher"]["weight"]))
            self.assertEqual(child["total_action_steps"], 1234)
            self.assertEqual(child["stage_action_steps"], 0)
            self.assertEqual(child["n_step"], 5)
            self.assertEqual(child["n_step_state"]["n_step"], 5)
            self.assertEqual(child["n_step_state"]["pending"], [])
            self.assertEqual(child["checkpoint_schema"], "training-resume-v11")
            self.assertEqual(child["safety_spec"], V4)
            self.assertEqual(child["opponent_scenarios_evaluated"], 0)

    def test_safety_replay_uses_48_8_8_and_round_trips_labels(self):
        replay = ReplayBuffer(200, seed=4)
        replay.configure("weak_opponents")
        state = np.zeros(84, dtype=np.float32)
        legal = np.ones(6, dtype=bool)
        for index in range(80):
            replay.append(Transition(
                state + index, 0, 0, state, False, legal, legal,
                "crate_navigation", 1, "ordinary"))
        for index in range(40):
            replay.append(Transition(
                state + index, 0, 0, state, False, legal, legal,
                "weak_opponents", 1, "own_bomb" if index < 12 else "ordinary"))
        replay.mark_recent_own_bomb_fatal("weak_opponents", 4)
        restored = ReplayBuffer(200, seed=99)
        restored.load_state_dict(replay.state_dict())
        restored.configure("weak_opponents")
        batch = restored.sample_batch(64, 0.75, {
            "enabled": True, "parent_samples": 48,
            "ordinary_task3_samples": 8, "own_bomb_cycle_samples": 8,
            "fatal_prefix_steps": 4,
        })
        self.assertEqual(int(batch["is_parent"].sum()), 48)
        self.assertEqual(sum(value != "ordinary" for value in batch["safety_classes"]), 8)

    def test_preregistration_keeps_seed_sets_disjoint(self):
        manifest = json.loads((ROOT / "experiments/task3_escape_obligation.json").read_text())
        known = set(manifest["known_death_seeds"])
        development = set(range(16000, 16020))
        confirmation = set(range(17000, 17020))
        validation = set(range(18000, 18100))
        final = set(range(20000, 20100))
        sets = [known, development, confirmation, validation, final]
        for index, left in enumerate(sets):
            for right in sets[index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        for path in manifest["configs"].values():
            json.loads((ROOT / path).read_text())


if __name__ == "__main__":
    unittest.main()
