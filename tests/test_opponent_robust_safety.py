import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import settings as s
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.team_agent.opponent_transitions import (
    enumerate_opponent_transition_scenarios,
)
from agent_code.team_agent.controllable_survival import (
    controllable_survival_actions, danger_interval_steps,
)
from agent_code.team_agent.safety import resolve_safety_spec, safety_decision
from agent_code.team_agent.temporal_safety_features import (
    OpponentRobustResult, RobustRouteResult,
    opponent_robust_survival_after_action,
)
from tests.test_danger import make_game_state


V4 = {
    "version": "survival-mask-v4", "mode": "all", "horizon": 7,
    "required_independent_routes": 2,
    "robust_fallback": "survival-mask-v1",
    "opponent_transition_horizon": 1,
    "opponent_action_space": "all_physical",
    "include_opponent_bombs": True,
    "execution_orders": "all",
    "minimum_scenario_routes": 1,
    "opponent_robust_fallback": "survival-mask-v3",
    "fallback": "physical_q",
}
V5 = {
    "version": "survival-mask-v5", "mode": "all", "horizon": 7,
    "opponent_action_space": "all_physical",
    "include_opponent_bombs": True,
    "execution_orders": "all",
    "danger_interval": "own_bomb_and_lingering",
    "search_budget_ms": 400,
    "robust_fallback": "survival-mask-v1",
    "fallback": "physical_q",
}


class OpponentTransitionTests(unittest.TestCase):
    def test_contract_is_complete_and_fixed(self):
        self.assertEqual(resolve_safety_spec(V4), V4)
        with self.assertRaisesRegex(ValueError, "include_opponent_bombs"):
            resolve_safety_spec({**V4, "include_opponent_bombs": False})

    def test_enumeration_includes_moves_wait_bombs_and_deduplicates(self):
        state = make_game_state(position=(3, 3))
        state["others"] = [("enemy", 0, True, (5, 3))]
        scenarios = enumerate_opponent_transition_scenarios(state, "BOMB")
        self.assertGreater(len(scenarios), 1)
        outcomes = {
            (tuple(item.game_state["others"][0][3]),
             tuple(item.game_state["bombs"]))
            for item in scenarios
        }
        self.assertEqual(len(outcomes), len(scenarios))
        self.assertTrue(any(
            ((5, 3), s.BOMB_TIMER) in item.game_state["bombs"]
            for item in scenarios))
        self.assertTrue(any(
            tuple(item.game_state["others"][0][3]) == (4, 3)
            for item in scenarios))

    def test_execution_order_changes_competing_move_result(self):
        state = make_game_state(position=(3, 3))
        state["others"] = [("enemy", 0, False, (5, 3))]
        scenarios = enumerate_opponent_transition_scenarios(state, "RIGHT")
        positions = {
            (tuple(item.game_state["self"][3]),
             tuple(item.game_state["others"][0][3]))
            for item in scenarios
        }
        self.assertIn(((4, 3), (5, 3)), positions)
        self.assertIn(((3, 3), (4, 3)), positions)

    def test_single_failing_scenario_fails_the_action(self):
        state = make_game_state(position=(3, 3))
        state["others"] = [("enemy", 0, True, (5, 3))]
        state["field"][3, 2] = -1
        state["field"][3, 4] = -1
        state["field"][2, 3] = -1
        result = opponent_robust_survival_after_action(state, "BOMB")
        self.assertFalse(result.survives_all)
        self.assertGreater(result.scenario_count, result.passing_scenarios)
        self.assertIsNotNone(result.first_failing_profile)


class OpponentRobustMaskTests(unittest.TestCase):
    def test_pre_bomb_vetoes_when_one_opponent_scenario_fails(self):
        state = make_game_state(position=(3, 3))
        physical = np.ones(6, dtype=bool)
        with (
            patch(
                "agent_code.team_agent.temporal_safety_features.robust_routes_after_first_step",
                return_value=RobustRouteResult(2, 0.5)),
            patch(
                "agent_code.team_agent.temporal_safety_features.opponent_robust_survival_after_action",
                return_value=OpponentRobustResult(
                    False, 12, 4, ("BOMB", "LEFT"), (0, 1))),
        ):
            decision = safety_decision(
                state, physical, V4, allow_bomb=True, exploring=False)
        self.assertFalse(decision.mask[ACTIONS.index("BOMB")])
        self.assertEqual(decision.opponent_scenario_counts[-1], 12)
        self.assertEqual(decision.opponent_passing_counts[-1], 4)

    def test_post_bomb_uses_v4_then_falls_back_to_v3(self):
        state = make_game_state(position=(3, 3), bombs_left=False)
        physical = np.ones(6, dtype=bool)
        physical[-1] = False
        route_results = iter([
            RobustRouteResult(2, 0.5), RobustRouteResult(1, 0.2),
            RobustRouteResult(1, 0.2), RobustRouteResult(1, 0.2),
            RobustRouteResult(1, 0.2),
        ])
        opponent_results = iter([
            OpponentRobustResult(False, 2, 1, ("UP",), (0,)),
        ])
        with (
            patch(
                "agent_code.team_agent.temporal_safety_features.robust_routes_after_first_step",
                side_effect=lambda *args, **kwargs: next(route_results)),
            patch(
                "agent_code.team_agent.temporal_safety_features.opponent_robust_survival_after_action",
                side_effect=lambda *args, **kwargs: next(opponent_results)),
        ):
            decision = safety_decision(
                state, physical, V4, allow_bomb=True, exploring=False,
                own_bomb_pending=True)
        self.assertTrue(decision.opponent_fallback)
        self.assertFalse(decision.robust_fallback)
        np.testing.assert_array_equal(
            decision.mask, [True, False, False, False, False, False])


class ControllableSurvivalTests(unittest.TestCase):
    def test_v5_contract_and_exact_danger_interval(self):
        self.assertEqual(resolve_safety_spec(V5), V5)
        self.assertEqual(danger_interval_steps({}, placing_bomb=True), 6)
        self.assertEqual(danger_interval_steps(
            {"pending": True, "timer": 3}, placing_bomb=False), 5)
        self.assertEqual(danger_interval_steps(
            {"pending": True, "timer": 0}, placing_bomb=False), 2)

    def test_zero_budget_is_fail_closed(self):
        state = make_game_state(position=(3, 3))
        result = controllable_survival_actions(
            state, ("WAIT",), remaining_steps=2, budget_ms=0)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.proven_actions, ())

    def test_pre_bomb_failure_never_falls_back_to_allow_bomb(self):
        state = make_game_state(position=(3, 3))
        physical = np.ones(6, dtype=bool)
        with patch(
            "agent_code.team_agent.controllable_survival."
            "controllable_survival_actions",
            return_value=type("Result", (), {
                "proven_actions": (), "scenarios_evaluated": 7,
                "states_evaluated": 3, "timed_out": False,
                "first_failing_profile": ("BOMB", "LEFT"),
                "first_failing_order": (1, 0),
            })(),
        ):
            decision = safety_decision(
                state, physical, V5, allow_bomb=True, exploring=False,
                own_bomb_state={"pending": False, "timer": None})
        self.assertFalse(decision.mask[ACTIONS.index("BOMB")])
        self.assertFalse(decision.robust_guarantee_loss)

    def test_post_bomb_empty_set_falls_back_to_v1_and_records_loss(self):
        state = make_game_state(position=(3, 3), bombs_left=False)
        physical = np.ones(6, dtype=bool)
        physical[-1] = False
        with patch(
            "agent_code.team_agent.controllable_survival."
            "controllable_survival_actions",
            return_value=type("Result", (), {
                "proven_actions": (), "scenarios_evaluated": 11,
                "states_evaluated": 5, "timed_out": False,
                "first_failing_profile": ("WAIT", "RIGHT"),
                "first_failing_order": (1, 0),
            })(),
        ):
            decision = safety_decision(
                state, physical, V5, allow_bomb=True, exploring=False,
                own_bomb_pending=True,
                own_bomb_state={"pending": True, "timer": 2})
        self.assertTrue(decision.opponent_fallback)
        self.assertTrue(decision.robust_guarantee_loss)
        self.assertTrue(decision.mask.any())


class OpponentRobustPreregistrationTests(unittest.TestCase):
    def test_seed_sets_and_configs_are_fixed_and_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "experiments/task3_opponent_robust.json").read_text())
        sets = [
            set(manifest["known_death_seeds"]),
            set(range(16000, 16020)), set(range(16100, 16200)),
            set(range(17000, 17020)), set(range(18000, 18100)),
            set(range(20000, 20100)),
        ]
        for index, left in enumerate(sets):
            for right in sets[index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        for name, relative in manifest["configs"].items():
            config = json.loads((root / relative).read_text())
            if name.endswith("v4"):
                self.assertEqual(config["safety"], V4)
        self.assertEqual(manifest["checkpoint_schema"], "training-resume-v10")


class ControllableSafetyPreregistrationTests(unittest.TestCase):
    def test_v5_seed_sets_contracts_and_latency_gates_are_fixed(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "experiments/task3_controllable_survival.json").read_text())
        ranges = [
            set(range(19100, 19200)), set(range(19200, 19220)),
            set(range(19300, 19320)), set(range(19400, 19500)),
            set(range(20000, 20100)),
        ]
        regression = set(
            manifest["known_regression_corpus"]["legacy_v7_seeds"]
            + manifest["known_regression_corpus"]["bomb_credit_n5_seeds"])
        self.assertEqual(len(regression), 11)
        for index, left in enumerate([regression, *ranges]):
            for right in [regression, *ranges][index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(manifest["checkpoint_schema"], "training-resume-v11")
        self.assertEqual(manifest["n_step"], 5)
        self.assertEqual(manifest["gates"]["maximum_act_p95_seconds"], 0.25)
        self.assertEqual(manifest["gates"]["maximum_act_seconds"], 0.48)
        for relative in manifest["configs"].values():
            config = json.loads((root / relative).read_text())
            self.assertEqual(config["safety"], V5)

    def test_progressed_transition_explodes_timer_zero_and_clears_crate(self):
        state = make_game_state(position=(3, 3))
        state["others"] = [("enemy", 0, False, (7, 7))]
        state["bombs"] = [((3, 3), 0)]
        state["field"][3, 2] = 1
        scenarios = enumerate_opponent_transition_scenarios(
            state, "WAIT", progress_world=True)
        self.assertTrue(scenarios)
        self.assertTrue(all(not item.self_alive for item in scenarios))
        self.assertTrue(all(item.game_state["field"][3, 2] == 0
                            for item in scenarios))
        self.assertTrue(all(item.game_state["explosion_map"][3, 3] == 1
                            for item in scenarios))


if __name__ == "__main__":
    unittest.main()
