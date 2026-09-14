import copy
import json
from pathlib import Path
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

import events as e
from agent_code.double_dqn_continuous_v2_agent import callbacks as v2_callbacks
from agent_code.double_dqn_continuous_v2_agent import train as v2_train
from agent_code.dqn_agent.model import ReplayBuffer, Transition
from agent_code.learning_common.action_history import (
    action_history_state, init_action_history, load_action_history_state,
    own_bomb_history_for_state, record_selected_action,
)
from agent_code.learning_common.runtime import validate_checkpoint
from agent_code.team_agent.feature_system import ACTIONS, get_feature_schema
from agent_code.team_agent.feature_system.continuous_v3 import extract as continuous_v3
from agent_code.team_agent.feature_system.discrete_compact_v1 import _transform_game_state
from agent_code.team_agent.feature_system.symmetry import symmetries
from agent_code.team_agent.rewards import reward_from_events
from agent_code.team_agent.safety import (
    avoidable_fatal_action, mask_for_decision, resolve_safety_spec,
)
from tests.test_danger import WALL, make_game_state


ALL_SAFETY = {
    "version": "survival-mask-v1", "mode": "all", "horizon": 7,
    "fallback": "physical_q",
}


class SurvivalMaskTests(unittest.TestCase):
    def _danger_state(self):
        return make_game_state(position=(3, 3), bombs=[((3, 5), 0)])

    def test_modes_and_legacy_mapping_are_explicit(self):
        self.assertEqual(
            resolve_safety_spec(legacy_safe_exploration=True)["mode"],
            "exploration",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_safety_spec(ALL_SAFETY, legacy_safe_exploration=True)

        physical = np.ones(6, dtype=bool)
        physical[-1] = False
        state = self._danger_state()
        exploratory, _ = mask_for_decision(
            state, physical, resolve_safety_spec(legacy_safe_exploration=True),
            allow_bomb=False, exploring=True)
        greedy, _ = mask_for_decision(
            state, physical, resolve_safety_spec(legacy_safe_exploration=True),
            allow_bomb=False, exploring=False)
        self.assertFalse(exploratory[ACTIONS.index("WAIT")])
        np.testing.assert_array_equal(greedy, physical)

    def test_frozen_greedy_vetoes_raw_q_argmax_and_records_intervention(self):
        state = self._danger_state()
        owner = SimpleNamespace(
            train=False, curriculum_allows_bomb=False,
            safety_spec=dict(ALL_SAFETY), rng=random.Random(1),
            stage_action_steps=0, total_action_steps=0,
            safety_decisions=0, safety_interventions=0, safety_fallbacks=0,
            safe_exploration_decisions=0, safe_exploration_fallbacks=0,
            _feature_cache_key=None, _feature_cache_value=None,
            model=SimpleNamespace(q_values=lambda _: np.asarray(
                [1.0, 9.0, 2.0, 3.0, 100.0, -5.0], dtype=np.float32)),
        )
        init_action_history(owner)
        action = v2_callbacks.act(owner, state)
        self.assertEqual(action, "RIGHT")
        self.assertEqual(owner.last_safety_diagnostic["raw_action"], "WAIT")
        self.assertTrue(owner.last_safety_diagnostic["intervened"])
        self.assertEqual(owner.safety_interventions, 1)

    def test_no_safe_action_falls_back_to_physical_q_argmax(self):
        state = make_game_state(position=(3, 3), bombs_left=False)
        for position in ((2, 3), (4, 3), (3, 2), (3, 4)):
            state["field"][position] = WALL
        state["explosion_map"][3, 3] = 1
        physical = np.asarray([False, False, False, False, True, False])
        decision, fallback = mask_for_decision(
            state, physical, ALL_SAFETY, allow_bomb=False, exploring=False)
        self.assertTrue(fallback)
        np.testing.assert_array_equal(decision, physical)

    def test_replay_and_n_step_target_keep_actual_survival_mask(self):
        owner = SimpleNamespace(
            curriculum_allows_bomb=False, safety_spec=dict(ALL_SAFETY),
            training_task="crate_navigation",
            _feature_cache_key=None, _feature_cache_value=None,
        )
        init_action_history(owner)
        old = self._danger_state()
        new = copy.deepcopy(old)
        new["step"] = 2
        transition = v2_train._transition(owner, old, "RIGHT", 0.0, new, False)
        self.assertFalse(transition.next_legal[ACTIONS.index("WAIT")])

        replay = ReplayBuffer(8, seed=4)
        replay.configure("crate_navigation")
        replay.append(transition._replace(task_id="crate_navigation"))
        restored = ReplayBuffer(8, seed=99)
        restored.load_state_dict(replay.state_dict())
        np.testing.assert_array_equal(
            restored._all_items()[0].next_legal, transition.next_legal)

    def test_v5_checkpoint_is_frozen_only_and_v6_binds_safety(self):
        payload = {
            "checkpoint_schema": "training-resume-v5",
            "algorithm": v2_callbacks.ALGORITHM,
            "feature_id": v2_callbacks.FEATURE_ID,
            "feature_schema": v2_callbacks.FEATURE_SCHEMA,
            "feature_version": None,
            "actions": list(ACTIONS),
            "reward_id": "r7_safe_credit_sparse",
            "reward_version": "r7_safe_credit_sparse",
            "hyperparameters": v2_callbacks.HYPERPARAMETERS,
            "network_spec": v2_callbacks.NETWORK_SPEC,
            "training_task": "crate_navigation",
        }
        validate_checkpoint(
            payload, algorithm=v2_callbacks.ALGORITHM,
            feature_id=v2_callbacks.FEATURE_ID,
            feature_schema=v2_callbacks.FEATURE_SCHEMA, actions=ACTIONS,
            reward_id="r7_safe_credit_sparse",
            hyperparameters=v2_callbacks.HYPERPARAMETERS,
            network_spec=v2_callbacks.NETWORK_SPEC, training=False,
            training_task="crate_navigation", safety_spec=ALL_SAFETY)
        with self.assertRaisesRegex(ValueError, "frozen-evaluation only"):
            validate_checkpoint(
                payload, algorithm=v2_callbacks.ALGORITHM,
                feature_id=v2_callbacks.FEATURE_ID,
                feature_schema=v2_callbacks.FEATURE_SCHEMA, actions=ACTIONS,
                reward_id="r7_safe_credit_sparse",
                hyperparameters=v2_callbacks.HYPERPARAMETERS,
                network_spec=v2_callbacks.NETWORK_SPEC, training=True,
                training_task="crate_navigation", safety_spec=ALL_SAFETY)


class ContinuousV3Tests(unittest.TestCase):
    def test_shape_ranges_and_own_bomb_facts(self):
        state = make_game_state(position=(3, 3), bombs=[((3, 3), 2)], bombs_left=False)
        result = continuous_v3(
            state, previous_action="RIGHT", wait_streak=2,
            own_bomb={"position": (3, 3), "pending": True, "visible": True, "timer": 2},
        )
        self.assertEqual(get_feature_schema("continuous-v3").vector_shape, (107,))
        self.assertEqual(result.vector.shape, (107,))
        self.assertTrue(np.isfinite(result.vector).all())
        self.assertTrue(np.all(result.vector >= -1.0))
        self.assertTrue(np.all(result.vector <= 1.0))
        np.testing.assert_allclose(result.vector[-4:], [1.0, 1.0, 0.5, 1.0])

    def test_action_conditional_fields_are_equivariant_under_symmetry(self):
        state = make_game_state(position=(3, 3), bombs=[((3, 5), 1)])
        state["coins"] = [(6, 3)]
        state["field"][5, 3] = 1
        original = continuous_v3(state).vector
        for symmetry in symmetries(*state["field"].shape):
            transformed = continuous_v3(_transform_game_state(state, symmetry)).vector
            for world_index, mapped_index in enumerate(
                    symmetry.action_transform.world_to_canonical):
                np.testing.assert_allclose(
                    original[world_index * 10:(world_index + 1) * 10],
                    transformed[mapped_index * 10:(mapped_index + 1) * 10], atol=1e-6)
                np.testing.assert_allclose(
                    original[84 + world_index * 3:87 + world_index * 3],
                    transformed[84 + mapped_index * 3:87 + mapped_index * 3], atol=1e-6)
            np.testing.assert_allclose(original[60:70], transformed[60:70], atol=1e-6)
            np.testing.assert_allclose(original[102:], transformed[102:], atol=1e-6)

    def test_own_bomb_history_resets_and_round_trips(self):
        owner = SimpleNamespace()
        init_action_history(owner)
        state = make_game_state()
        record_selected_action(owner, state, "BOMB")
        saved = action_history_state(owner)
        clone = SimpleNamespace()
        load_action_history_state(clone, saved)
        state["self"] = (*state["self"][:2], False, state["self"][3])
        self.assertEqual(
            own_bomb_history_for_state(clone, state)["position"], (3, 3))
        next_round = make_game_state()
        next_round["round"] = 2
        self.assertIsNone(
            own_bomb_history_for_state(clone, next_round)["position"])


class ConstrainedRewardTests(unittest.TestCase):
    def test_r8_penalizes_only_avoidable_fatal_actions(self):
        state = make_game_state(position=(3, 3), bombs=[((3, 5), 0)])
        physical = np.ones(6, dtype=bool)
        physical[-1] = False
        self.assertTrue(avoidable_fatal_action(
            state, "WAIT", physical, allow_bomb=False))
        penalized = reward_from_events(
            [], "r8_safe_constrained", old_game_state=state,
            new_game_state=state, action="WAIT", avoidable_fatal=True)
        ordinary = reward_from_events(
            [], "r8_safe_constrained", old_game_state=state,
            new_game_state=state, action="WAIT", avoidable_fatal=False)
        self.assertAlmostEqual(penalized - ordinary, -20.0)

        trapped = make_game_state(position=(3, 3), bombs_left=False)
        for position in ((2, 3), (4, 3), (3, 2), (3, 4)):
            trapped["field"][position] = WALL
        trapped["explosion_map"][3, 3] = 1
        physical = np.asarray([False, False, False, False, True, False])
        self.assertFalse(avoidable_fatal_action(
            trapped, "WAIT", physical, allow_bomb=False))

    def test_self_kill_suppresses_same_frame_positive_events(self):
        reward = reward_from_events(
            [e.KILLED_SELF, e.COIN_COLLECTED, e.CRATE_DESTROYED,
             e.KILLED_OPPONENT],
            "r8_safe_constrained")
        self.assertAlmostEqual(reward, -20.01)

    def test_task1_r8_matches_r7_and_removes_useful_bomb_credit(self):
        state = make_game_state(position=(3, 3))
        state["coins"] = [(5, 3)]
        moved = copy.deepcopy(state)
        moved["self"] = (*moved["self"][:3], (4, 3))
        self.assertEqual(
            reward_from_events([], "r7_safe_credit_sparse", old_game_state=state,
                               new_game_state=moved, action="RIGHT"),
            reward_from_events([], "r8_safe_constrained", old_game_state=state,
                               new_game_state=moved, action="RIGHT"),
        )
        state["field"][5, 3] = 1
        r7 = reward_from_events(
            [], "r7_safe_credit_sparse", old_game_state=state,
            new_game_state=state, action="BOMB")
        r8 = reward_from_events(
            [], "r8_safe_constrained", old_game_state=state,
            new_game_state=state, action="BOMB")
        self.assertAlmostEqual(r7 - r8, 0.2)


class SafetyPreregistrationTests(unittest.TestCase):
    def test_manifest_freezes_safety_gates_and_seed_isolation(self):
        manifest = json.loads(Path(
            "experiments/task2_safety_ablation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["screening_seed"], 11)
        self.assertEqual(manifest["replication_seeds"], [22, 33])
        self.assertEqual(
            manifest["main_validation_seeds"], {"start": 11000, "end": 11099})
        self.assertEqual(
            manifest["reserved_final_test_seeds"], {"start": 20000, "end": 20099})
        self.assertEqual(manifest["gates"]["survived_bomb_rate"], 0.95)
        self.assertFalse(manifest["task3_launch"])

    def test_all_ablation_configs_use_complete_all_mode_contract(self):
        paths = sorted(Path("experiments/configs").glob("safety_ablation_*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["safety"], ALL_SAFETY)
            self.assertNotIn("safe_exploration", config["training"])
            self.assertEqual(config["training"]["n_step"], 4)
            self.assertEqual(config["training"]["device"], "cpu")
            if "task2" in path.name:
                self.assertEqual(config["training"]["min_rounds"], 500)
                self.assertEqual(
                    config["training"]["target_stage_action_steps"], 150000)
            else:
                self.assertEqual(
                    config["training"]["target_stage_action_steps"], 200000)


if __name__ == "__main__":
    unittest.main()
