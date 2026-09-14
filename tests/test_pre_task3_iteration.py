import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import torch

import events as e
from agent_code.dqn_agent.model import DQN, ReplayBuffer, Transition as DQNTransition
from agent_code.dqn_agent import callbacks as dqn_callbacks
from agent_code.learning_common.action_history import init_action_history
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.q_learning_agent.callbacks import act as q_act
from agent_code.team_agent.exploration import survivable_exploration_mask
from agent_code.team_agent.feature_system import (
    ACTIONS, extract_features, get_feature_schema,
)
from agent_code.team_agent.feature_system.discrete_objective_v1 import extract as objective
from agent_code.team_agent.feature_system.continuous_v2 import extract as continuous_v2
from agent_code.team_agent.rewards import REWARD_SPECS, reward_from_events
from experiments.agent_contracts import resolve_agent_contract
from experiments.resume import CHECKPOINT_SCHEMA_VERSION, validate_resume_transition
from experiments.training import TrainingActionBudget
from experiments.run import _parser, run_agent_session
from tests.test_danger import WALL, make_game_state


RETENTION = {
    "parent_fraction": 0.5,
    "distillation_weight": 1.0,
    "temperature": 1.0,
    "per_task_capacity": 20_000,
    "current_warmup": 2_000,
}


class ObjectiveFeatureTests(unittest.TestCase):
    def test_objective_schema_is_60_dimensional_and_factual(self):
        schema = get_feature_schema("discrete-objective-v1")
        state = make_game_state(position=(3, 3))
        state["coins"] = [(5, 3)]
        original = copy.deepcopy(state)
        result = objective(state, previous_action="WAIT", wait_streak=2)

        self.assertEqual(schema.vector_shape, (60,))
        self.assertEqual(len(result.state_key), 14)
        self.assertEqual(float(result.vector.sum()), 14.0)
        self.assertEqual(result.state_key[5], 4)  # RIGHT is closer to the coin.
        self.assertEqual(result.state_key[-3:], (1, 5, 2))
        np.testing.assert_array_equal(state["field"], original["field"])
        self.assertEqual(state["coins"], original["coins"])

    def test_hierarchical_objective_falls_back_to_crate_frontier(self):
        state = make_game_state(position=(3, 3))
        state["field"][5, 3] = 1
        result = objective(state)
        self.assertEqual(result.state_key[-4], 2)  # Distance 1-2.
        self.assertEqual(result.state_key[-3], 2)  # Crate-frontier objective.

    def test_action_history_advances_in_frozen_act_and_resets_by_round(self):
        owner = SimpleNamespace(
            train=False, allow_bomb=False, rng=random.Random(7),
            q_table={}, feature_id="discrete-objective-v1",
            _feature_cache_key=None, _feature_cache_value=None,
        )
        init_action_history(owner)
        first = make_game_state(position=(3, 3))
        first.update(round=1, step=1)
        first_features = objective(first)
        values = np.zeros(len(ACTIONS), dtype=np.float32)
        values[ACTIONS.index("WAIT")] = 1.0
        owner.q_table[first_features.state_key] = values
        self.assertEqual(q_act(owner, first), "WAIT")

        second = make_game_state(position=(3, 3))
        second.update(round=1, step=2)
        observed = objective(second, owner.feature_previous_action, owner.feature_wait_streak)
        self.assertEqual(observed.state_key[-2:], (5, 1))

        next_round = make_game_state(position=(3, 3))
        next_round.update(round=2, step=1)
        from agent_code.learning_common.action_history import action_history_for_state
        self.assertEqual(action_history_for_state(owner, next_round), (None, 0))

    def test_continuous_v2_uses_remote_history_contract(self):
        state = make_game_state(position=(3, 3))
        state["coins"] = [(5, 3)]
        old = extract_features(state, "continuous-v1").vector
        new = continuous_v2(
            state, previous_action="WAIT", previous_position=(3, 2),
            previous_coin_target=(5, 3),
        ).vector
        self.assertEqual(new.shape, (84,))
        np.testing.assert_array_equal(new[:70], old)
        self.assertEqual(new[74], 1.0)
        self.assertEqual(new[77], 1.0)
        self.assertEqual(new[78], 1.0)

    def test_standard_dqn_builds_a_60_input_network_for_objective_features(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "new.pt"
            owner = SimpleNamespace(train=True, logger=Mock())
            with (
                patch.dict("os.environ", {
                    "BOMBERMAN_FEATURE_ID": "discrete-objective-v1",
                    "BOMBERMAN_AGENT_SEED": "11",
                }, clear=True),
                patch.object(dqn_callbacks, "MODEL_FILE", missing),
            ):
                dqn_callbacks.setup(owner)
        self.assertEqual(owner.model.input_size, 60)
        self.assertEqual(owner.model.policy.layers[0].in_features, 60)


class RewardAndExplorationTests(unittest.TestCase):
    def test_preregistered_reward_constants_and_death_deduplication(self):
        expected = {
            "r5_coin_potential", "r6_safe_sparse", "r6_safe_potential",
            "r7_safe_credit_sparse", "r7_safe_credit_potential",
        }
        self.assertTrue(expected.issubset(REWARD_SPECS))
        for reward_id in expected:
            with self.subTest(reward_id=reward_id):
                self.assertEqual(REWARD_SPECS[reward_id]["coin_collected"], 3.0)
                self.assertEqual(REWARD_SPECS[reward_id]["step"], -0.01)
        self.assertAlmostEqual(
            reward_from_events(
                [e.KILLED_SELF, e.GOT_KILLED], "r6_safe_sparse"), -20.01)
        self.assertAlmostEqual(
            reward_from_events([e.GOT_KILLED], "r6_safe_sparse"), -10.01)

    def test_coin_potential_rewards_shortest_path_progress(self):
        old = make_game_state(position=(3, 3))
        new = make_game_state(position=(4, 3))
        old["coins"] = new["coins"] = [(6, 3)]
        closer = reward_from_events(
            [e.MOVED_RIGHT], "r5_coin_potential",
            old_game_state=old, new_game_state=new)
        farther = reward_from_events(
            [e.MOVED_LEFT], "r5_coin_potential",
            old_game_state=new, new_game_state=old)
        self.assertGreater(closer, -0.01)
        self.assertLess(farther, -0.01)

    def test_bomb_credit_distinguishes_safe_useful_and_unescapable_bombs(self):
        useful = make_game_state(position=(3, 3))
        useful["field"][5, 3] = 1
        bomb_reward = reward_from_events(
            [], "r6_safe_sparse", old_game_state=useful,
            new_game_state=useful, action="BOMB")
        wait_reward = reward_from_events(
            [], "r6_safe_sparse", old_game_state=useful,
            new_game_state=useful, action="WAIT")
        self.assertAlmostEqual(bomb_reward - wait_reward, 0.2)

        trapped = make_game_state(position=(3, 3))
        for position in ((2, 3), (4, 3), (3, 2), (3, 4)):
            trapped["field"][position] = WALL
        r6 = reward_from_events(
            [], "r6_safe_sparse", old_game_state=trapped,
            new_game_state=trapped, action="BOMB")
        r7 = reward_from_events(
            [], "r7_safe_credit_sparse", old_game_state=trapped,
            new_game_state=trapped, action="BOMB")
        self.assertAlmostEqual(r7 - r6, -10.0)

    def test_safe_exploration_filters_only_the_stochastic_candidate_mask(self):
        state = make_game_state(position=(3, 3), bombs=[((3, 5), 0)])
        physical = np.ones(len(ACTIONS), dtype=bool)
        physical[ACTIONS.index("BOMB")] = False
        safe, fallback = survivable_exploration_mask(
            state, physical, allow_bomb=False)
        self.assertFalse(fallback)
        self.assertFalse(safe[ACTIONS.index("WAIT")])
        self.assertTrue(safe[ACTIONS.index("RIGHT")])
        np.testing.assert_array_equal(
            physical, [True, True, True, True, True, False])

    def test_safe_exploration_records_physical_fallback(self):
        explosion = np.zeros((17, 17), dtype=int)
        explosion[3, 3] = 1
        state = make_game_state(
            position=(3, 3), explosion_map=explosion, bombs_left=False)
        for position in ((2, 3), (4, 3), (3, 2), (3, 4)):
            state["field"][position] = WALL
        physical = np.asarray([False, False, False, False, True, False])
        safe, fallback = survivable_exploration_mask(
            state, physical, allow_bomb=False)
        self.assertTrue(fallback)
        np.testing.assert_array_equal(safe, physical)


class CreditAndRetentionTests(unittest.TestCase):
    def test_four_step_accumulator_emits_discounted_returns_and_flushes(self):
        Transition = __import__(
            "agent_code.q_learning_agent.train", fromlist=["Transition"]
        ).Transition
        accumulator = NStepAccumulator(4, 0.5)
        emitted = []
        for index in range(4):
            emitted.extend(accumulator.append(Transition(
                (index,), 0, 1.0, (index + 1,), False,
                np.ones(6, dtype=bool))))
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0].reward, 1.875)
        self.assertEqual(emitted[0].steps, 4)
        terminal = accumulator.append(Transition(
            (4,), 0, 2.0, None, True, None))
        self.assertEqual(len(terminal), 4)
        self.assertTrue(all(item.done for item in terminal))

    def test_task_partitioned_replay_samples_half_parent_half_current(self):
        replay = ReplayBuffer(20, seed=13)
        state = np.zeros(4, dtype=np.float32)
        legal = np.asarray([True, False, True, False, True, False])
        replay.configure("coin_navigation")
        for index in range(8):
            replay.append(DQNTransition(
                state + index, 0, 0.0, state, False, legal, legal,
                "coin_navigation", 1))
        replay.configure("crate_navigation")
        for index in range(8):
            replay.append(DQNTransition(
                state + 20 + index, 0, 0.0, state, False, legal, legal,
                "crate_navigation", 1))
        batch = replay.sample_batch(8, parent_fraction=0.5)
        self.assertEqual(int(batch["is_parent"].sum()), 4)

    def test_task2_teacher_is_frozen_and_distillation_update_is_finite(self):
        retention = {
            "parent_fraction": 0.5, "distillation_weight": 1.0,
            "temperature": 1.0, "per_task_capacity": 20,
            "current_warmup": 4,
        }
        parent = DQN(
            4, 6, seed=7, batch_size=4, warmup=4,
            training_task="coin_navigation", retention_spec=retention)
        state = np.zeros(4, dtype=np.float32)
        # Real Bomberman states almost always mask some movement/BOMB actions;
        # the KL implementation must remain finite when teacher probability is
        # exactly zero on those actions.
        legal = np.asarray([True, False, True, False, True, False])
        for index in range(4):
            parent.replay.append(DQNTransition(
                state + index, 0, 1.0, state + index + 1, False,
                legal, legal, "coin_navigation", 1))
        checkpoint = parent.checkpoint()
        checkpoint["training_task"] = "coin_navigation"
        child = DQN(
            4, 6, seed=99, batch_size=4, warmup=4,
            training_task="crate_navigation", retention_spec=retention)
        child.load_checkpoint(
            checkpoint, training=True, training_task="crate_navigation")
        teacher_before = {
            key: value.clone() for key, value in child.teacher.state_dict().items()
        }
        loss = None
        for index in range(4):
            loss = child.observe(DQNTransition(
                state + 10 + index, 0, 0.5, state + 11 + index,
                False, legal, legal, "crate_navigation", 1))
        self.assertTrue(np.isfinite(loss))
        for key, value in child.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, teacher_before[key]))


class PreregistrationAndContractTests(unittest.TestCase):
    def test_round3_adaptation_triggers_are_repeatable_and_validated(self):
        args = _parser().parse_args([
            "--config", "experiments/configs/pre_task3_task2_credit.json",
            "--mode", "train", "--task", "2", "--agent", "q_learning_agent",
            "--run-id", "example", "--adaptation-trigger", "suicide",
            "--adaptation-trigger", "q_capability",
        ])
        self.assertEqual(
            args.adaptation_trigger, ["suicide", "q_capability"])

    def test_action_budget_waits_for_round_minimum_and_action_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text(
                "round,stage_action_steps\n1,100\n", encoding="utf-8")
            budget = TrainingActionBudget(path, 100, 2)
            self.assertFalse(budget(1))
            with path.open("a", encoding="utf-8") as file:
                file.write("2,101\n")
            self.assertTrue(budget(2))
            self.assertEqual(budget.result["reason"], "stage_action_target_reached")

    def test_v5_allows_stage_contract_changes_only_on_task_promotion(self):
        base = {
            "algorithm": "dqn", "seed": 11, "task": "coin_navigation",
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "feature_id": "discrete-objective-v1",
            "feature_schema": get_feature_schema(
                "discrete-objective-v1").to_dict(),
            "reward_id": "r6_safe_potential",
            "reward_version": "r6_safe_potential",
            "reward_spec": REWARD_SPECS["r6_safe_potential"],
            "training_device_type": "cpu", "training_device_name": None,
            "agent_seed": 11, "source_commit": "abc", "source_hash": "def",
            "safe_exploration": True, "n_step": 1,
            "retention_spec": RETENTION,
            "training_budget": {
                "target_stage_action_steps": 100_000, "min_rounds": 1},
            "exploration_spec": {
                "version": "linear-v1", "start": 1.0, "end": 0.1,
                "decay_action_steps": 160_000},
            "actions": list(ACTIONS), "network_spec": None,
            "hyperparameters": {},
        }
        changed = {
            **base, "task": "crate_navigation", "n_step": 4,
            "retention_spec": {**RETENTION, "parent_fraction": 0.75},
            "training_budget": {
                "target_stage_action_steps": 150_000, "min_rounds": 500},
            "exploration_spec": {
                "version": "linear-v1", "start": 0.3, "end": 0.05,
                "decay_action_steps": 120_000},
        }
        self.assertEqual(
            validate_resume_transition(
                base, changed, parent_status="completed"), "next_task")
        for field in ("n_step", "retention_spec"):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field):
                validate_resume_transition(
                    base, {**base, field: changed[field]},
                    parent_status="completed")

    def test_fallback_agent_contracts_are_fixed_to_preregistered_features(self):
        double_q = resolve_agent_contract("double_q_agent")
        double_dqn = resolve_agent_contract("double_dqn_continuous_v2_agent")
        self.assertEqual(double_q.feature_id, "discrete-objective-v1")
        self.assertEqual(double_dqn.feature_id, "continuous-v2")
        self.assertEqual(double_dqn.network_spec["input_shape"], [78])
        self.assertTrue(double_dqn.network_spec["double_dqn"])

    def test_iteration_manifest_separates_dev_validation_and_final_seeds(self):
        manifest = json.loads(Path(
            "experiments/pre_task3_iteration.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["development_seeds"], list(range(10000, 10020)))
        self.assertEqual(
            manifest["main_validation_seeds"], {"start": 11000, "end": 11099})
        self.assertEqual(
            manifest["reserved_final_test_seeds"],
            {"start": 20000, "end": 20099})
        self.assertEqual(manifest["maximum_adaptive_rounds"], 3)
        self.assertFalse(manifest["task3_launch"])

    def test_all_preregistered_configs_have_fixed_stage_contracts(self):
        paths = sorted(Path("experiments/configs").glob("pre_task3_*.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            with self.subTest(path=path):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["evaluation"]["seeds"],
                    (list(range(11000, 11100)) if "main_validation" in path.name
                     else list(range(10000, 10020))),
                )
                self.assertEqual(config["evaluation"]["n_rounds"], 1)
                self.assertEqual(config["training"]["device"], "cpu")
                self.assertTrue(config["training"]["safe_exploration"])
                self.assertFalse(config["training"]["early_stopping"]["enabled"])
                if "main_validation" in path.name:
                    self.assertNotIn(
                        "target_stage_action_steps", config["training"])
                    self.assertEqual(config["training"]["exploration"]["start"], 0.3)
                elif "task2" in path.name:
                    self.assertEqual(
                        config["training"]["target_stage_action_steps"], 150_000)
                    self.assertEqual(config["training"]["min_rounds"], 500)
                    self.assertEqual(config["training"]["exploration"]["start"], 0.3)
                else:
                    self.assertEqual(
                        config["training"]["target_stage_action_steps"],
                        200_000 if "credit" in path.name else 100_000)
                    self.assertEqual(config["training"]["exploration"]["end"], 0.1)


if __name__ == "__main__":
    unittest.main()
