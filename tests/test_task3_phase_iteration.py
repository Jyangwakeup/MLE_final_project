import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import events as e
from agent_code.dqn_agent.model import DQN
from agent_code.team_agent.feature_system import ACTIONS, get_feature_schema
from agent_code.team_agent.feature_system.continuous_phase_v1 import extract
from agent_code.team_agent.feature_system.discrete_compact_v1 import _transform_game_state
from agent_code.team_agent.feature_system.symmetry import symmetries
from agent_code.team_agent.phase import (
    init_phase_history, load_phase_history_state, phase_facts_for_owner,
    phase_history_state, phase_weights,
)
from agent_code.team_agent.rewards import reward_from_events
from agent_code.team_agent.safety import margin_preserving_mask, resolve_safety_spec
from experiments.resume import CHECKPOINT_SCHEMA_VERSION, _expanded_network_state
from experiments.resume import validate_task3_transfer
from experiments.task3_phase import phase_gate_checks, rank_arms
from tests.test_danger import make_game_state


ROOT = Path(__file__).resolve().parents[1]


class PhaseFeatureTests(unittest.TestCase):
    def test_triangular_weights_are_continuous_and_sum_to_one(self):
        expected = {
            0.0: (1.0, 0.0, 0.0),
            0.25: (0.5, 0.5, 0.0),
            0.5: (0.0, 1.0, 0.0),
            0.75: (0.0, 0.5, 0.5),
            1.0: (0.0, 0.0, 1.0),
        }
        for progress, values in expected.items():
            self.assertEqual(phase_weights(progress), values)
            self.assertAlmostEqual(sum(phase_weights(progress)), 1.0)
        left = phase_weights(0.5 - 1e-7)
        right = phase_weights(0.5 + 1e-7)
        np.testing.assert_allclose(left, right, atol=5e-7)

    def test_feature_is_117d_and_zero_initial_denominators_are_defined(self):
        state = make_game_state()
        state["others"] = []
        result = extract(state, initial_crates=0, initial_opponents=0)
        self.assertEqual(get_feature_schema("continuous-phase-v1").vector_shape, (117,))
        self.assertEqual(result.vector.shape, (117,))
        self.assertTrue(np.isfinite(result.vector).all())
        self.assertEqual(float(result.vector[-7]), 0.0)  # crate depletion
        self.assertEqual(float(result.vector[-6]), 0.0)  # opponent depletion

    def test_round_tracker_resets_and_round_trips(self):
        owner = SimpleNamespace()
        init_phase_history(owner)
        state = make_game_state()
        state["field"][5, 5] = 1
        state["others"] = [("other", 2, True, (7, 7))]
        phase_facts_for_owner(owner, state)
        saved = phase_history_state(owner)
        clone = SimpleNamespace()
        load_phase_history_state(clone, saved)
        self.assertEqual(phase_history_state(clone), saved)
        next_state = copy.deepcopy(state)
        next_state["round"] = 2
        next_state["field"][6, 5] = 1
        phase_facts_for_owner(clone, next_state)
        self.assertEqual(clone.phase_initial_crates, 2)

    def test_phase_suffix_is_invariant_under_board_symmetry(self):
        state = make_game_state(position=(3, 3), bombs=[((3, 5), 1)])
        state["field"][5, 3] = 1
        state["coins"] = [(6, 3)]
        state["others"] = [("other", 2, True, (7, 3))]
        original = extract(state, initial_crates=1, initial_opponents=1).vector
        for symmetry in symmetries(*state["field"].shape):
            transformed = extract(
                _transform_game_state(state, symmetry),
                initial_crates=1, initial_opponents=1).vector
            np.testing.assert_allclose(original[-10:], transformed[-10:], atol=1e-6)


class PhaseRewardTests(unittest.TestCase):
    def _phase(self, early, middle, late):
        return {
            "early_weight": early, "middle_weight": middle, "late_weight": late,
            "score_margin": 0.0, "self_mobility": 1.0,
            "safe_action_fraction": 1.0, "opponent_closeness": 0.0,
            "opponent_mobility": 0.0,
        }

    def test_phase_events_use_old_state_mixture(self):
        for phase, expected in (
            (self._phase(1, 0, 0), 3 + .2 + 5 - .01),
            (self._phase(0, 1, 0), 1.5 + .05 + 8 - .01),
            (self._phase(0, 0, 1), 1 + 0 + 7.5 - .01),
        ):
            actual = reward_from_events(
                [e.COIN_COLLECTED, e.CRATE_DESTROYED, e.KILLED_OPPONENT],
                "r9_phase_full", old_phase_facts=phase)
            self.assertAlmostEqual(actual, expected)

    def test_suicide_dominates_all_positive_events(self):
        reward = reward_from_events(
            [e.KILLED_SELF, e.COIN_COLLECTED, e.CRATE_DESTROYED,
             e.KILLED_OPPONENT],
            "r9_phase_full", old_phase_facts=self._phase(0, 1, 0))
        self.assertAlmostEqual(reward, -30.01)


class TransferAndMaskTests(unittest.TestCase):
    def test_84_to_117_transfer_preserves_policy_and_target_q(self):
        parent = DQN(84, 6, seed=3, double_dqn=True, hidden_size=128)
        child = DQN(
            117, 6, seed=4, double_dqn=True, hidden_size=128,
            exact_prefix_size=84)
        checkpoint = parent.checkpoint()
        child.policy.load_state_dict(_expanded_network_state(checkpoint["policy"]))
        child.target.load_state_dict(_expanded_network_state(checkpoint["target"]))
        state = np.random.default_rng(9).normal(size=84).astype(np.float32)
        expanded = np.concatenate((state, np.zeros(33, dtype=np.float32)))
        parent_q = parent.q_values(state)
        child_q = child.q_values(expanded)
        np.testing.assert_allclose(parent_q, child_q, rtol=0, atol=5e-8)
        self.assertEqual(int(np.argmax(parent_q)), int(np.argmax(child_q)))
        with torch.no_grad():
            parent_target = parent.target(torch.as_tensor(state).unsqueeze(0))
            child_target = child.target(torch.as_tensor(expanded).unsqueeze(0))
        torch.testing.assert_close(parent_target, child_target, rtol=0, atol=5e-8)

    def test_mask_v2_keeps_actions_with_75_percent_of_best_area(self):
        spec = resolve_safety_spec({
            "version": "survival-mask-v2", "mode": "all", "horizon": 7,
            "fallback": "physical_q", "escape_area_fraction": 0.75,
        })
        self.assertEqual(spec["escape_area_fraction"], 0.75)
        physical = np.asarray([True, True, True, False, True, False])
        v1 = np.asarray([True, True, True, False, False, False])
        with patch(
            "agent_code.team_agent.safety.survival_mask",
            return_value=(v1, False),
        ), patch(
            "agent_code.team_agent.safety.survival_diagnostics",
            return_value={"escape_area": [8, 6, 5, 0, 0, 0]},
        ):
            decision, physical_fallback, margin_fallback = margin_preserving_mask(
                {}, physical, allow_bomb=False, own_bomb_pending=True)
        np.testing.assert_array_equal(
            decision, [True, True, False, False, False, False])
        self.assertFalse(physical_fallback)
        self.assertFalse(margin_fallback)

    def test_preregistration_seals_seeds_and_v8(self):
        manifest = json.loads((
            ROOT / "experiments" / "task3_phase_iteration.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, "training-resume-v11")
        self.assertEqual(manifest["checkpoint_schema"], "training-resume-v8")  # historical protocol
        self.assertEqual(manifest["round1"]["training_seeds"], [11, 22])
        self.assertEqual(manifest["main_validation"]["seeds"], {
            "start": 15000, "end": 15099})
        self.assertFalse(manifest["outcome"]["launch_task4"])

    def test_transfer_is_narrow_and_v7_is_not_ordinary_v8_resume(self):
        parent = {
            "checkpoint_schema": "training-resume-v7", "task": "crate_navigation",
            "algorithm": "double_dqn", "seed": 22, "agent_seed": 22,
            "actions": list(ACTIONS), "feature_id": "continuous-v2",
            "feature_schema": {"vector_shape": [84]},
        }
        child = {
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION, "task": "weak_opponents",
            "algorithm": "double_dqn", "seed": 22, "agent_seed": 22,
            "actions": list(ACTIONS), "feature_id": "continuous-phase-v1",
            "feature_schema": {"vector_shape": [117]},
        }
        validate_task3_transfer(parent, child, parent_status="completed")
        with self.assertRaisesRegex(ValueError, "shape must be 84"):
            validate_task3_transfer(
                {**parent, "feature_schema": {"vector_shape": [78]}}, child,
                parent_status="completed")

    def test_phase_gates_do_not_gate_geometric_edge_occupancy(self):
        parent = [{
            "early_weighted_coins_per_100_steps": 2.0,
            "early_weighted_crates_per_100_steps": 4.0,
            "late_mobility_crisis_rate": .10, "geometric_edge_rate": 0.0,
        }]
        child = [{
            "early_weighted_coins_per_100_steps": 1.8,
            "early_weighted_crates_per_100_steps": 3.6,
            "late_mobility_crisis_rate": .08, "geometric_edge_rate": 1.0,
        }]
        checks = phase_gate_checks(parent, child, {
            "early_weighted_coins_per_100_retention": .9,
            "early_weighted_crates_per_100_retention": .9,
            "late_mobility_crisis_rate": .05,
            "late_mobility_crisis_relative_reduction": .2,
        })
        self.assertTrue(all(checks.values()))
        self.assertNotIn("geometric_edge", checks)

    def test_arm_ranking_is_worst_seed_first(self):
        def arm(run_id, suicides, gains):
            return {"run_id": run_id, "seed_results": [{
                "passed": True, "suicide_rate": suicide, "score_gain": gain,
                "task3_score": 5, "combat_metric": 1, "resource_metric": 1,
                "mobility_metric": 1, "retention_metric": 1,
                "act_p95_seconds": .01,
            } for suicide, gain in zip(suicides, gains)]}
        ranked = rank_arms([
            arm("higher-risk", [0, .05], [2, 2]),
            arm("stable", [0, 0], [1, 1]),
        ])
        self.assertEqual(ranked[0]["run_id"], "stable")

    def test_published_phase_failure_evidence_is_complete_and_sealed(self):
        result = json.loads((
            ROOT / "experiments" / "task3_phase_results.json"
        ).read_text(encoding="utf-8"))
        evidence_path = ROOT / result["evidence"]["path"]
        with evidence_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

        self.assertEqual(result["designation"], "task3_phase_iteration_failure")
        self.assertFalse(result["qualified_for_task4"])
        self.assertFalse(result["stopping_decision"]["round3_started"])
        self.assertFalse(result["stopping_decision"]["seed33_started"])
        self.assertEqual(len(rows), 720)
        self.assertEqual(result["evidence"]["rows"], len(rows))
        self.assertEqual(result["evidence"]["sha256"], digest)
        self.assertEqual(
            {int(row["environment_seed"]) for row in rows},
            set(range(13000, 13020)),
        )
        self.assertTrue(all(
            float(row["score"]) == float(row["coins"])
            for row in rows if row["task"] == "coin_navigation"
        ))
        self.assertEqual(
            result["best_failed_configuration"]["arm"],
            "phase_r7_mask_v2",
        )


if __name__ == "__main__":
    unittest.main()
