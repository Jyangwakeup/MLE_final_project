import json
from math import exp
from pathlib import Path
import unittest

import events as e
import numpy as np
import settings as s
from agent_code.team_agent.danger import HORIZON
from agent_code.team_agent.feature_system.common import build_context

from agent_code.team_agent.rewards import (
    REWARD_SPECS,
    REWARD_VERSION,
    _state_potential,
    resolve_reward_spec,
    reward_from_events,
)


def make_state(position=(3, 3), coins=(), bombs=()):
    field = np.zeros((s.COLS, s.ROWS), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("reward_test", 0, True, position), "others": [],
        "bombs": list(bombs), "coins": list(coins),
        "explosion_map": np.zeros(field.shape, dtype=int), "user_input": None,
    }


class SharedRewardTestCase(unittest.TestCase):
    def test_r1_is_the_canonical_objective_reward(self):
        self.assertEqual(REWARD_VERSION, "r1")
        self.assertEqual(resolve_reward_spec("r1"), {
            "step": -0.01,
            "coin_collected": 1.0,
            "killed_opponent": 5.0,
            "crate_destroyed": 0.2,
            "death": -10.0,
            "invalid_action": -0.1,
        })
        self.assertEqual(
            resolve_reward_spec("r1_no_crate")["coin_collected"], 3.0)

    def test_r1_combines_repeated_objective_events(self):
        events = [
            e.COIN_COLLECTED,
            e.KILLED_OPPONENT,
            e.CRATE_DESTROYED,
            e.CRATE_DESTROYED,
            e.INVALID_ACTION,
            e.SURVIVED_ROUND,
        ]

        self.assertAlmostEqual(reward_from_events(events), 6.29)

    def test_death_is_penalized_only_once(self):
        self.assertAlmostEqual(
            reward_from_events([e.KILLED_SELF, e.GOT_KILLED]),
            -10.01,
        )

    def test_unmapped_events_only_receive_step_cost(self):
        self.assertAlmostEqual(reward_from_events([e.MOVED_UP]), -0.01)

    def test_survival_has_no_terminal_bonus(self):
        self.assertAlmostEqual(reward_from_events([e.SURVIVED_ROUND]), -0.01)

    def test_r2_balanced_maps_new_events_and_separates_deaths(self):
        events = [
            e.COIN_COLLECTED, e.COIN_FOUND, e.KILLED_OPPONENT,
            e.CRATE_DESTROYED, e.CRATE_DESTROYED, e.INVALID_ACTION,
            e.SURVIVED_ROUND,
        ]
        self.assertAlmostEqual(
            reward_from_events(events, "r2_balanced"), 8.49)
        self.assertAlmostEqual(
            reward_from_events([e.KILLED_SELF, e.GOT_KILLED], "r2_balanced"),
            -7.01,
        )
        self.assertAlmostEqual(
            reward_from_events([e.GOT_KILLED], "r2_balanced"), -5.01)

    def test_every_non_r1_reward_values_collected_coin_at_three(self):
        for reward_id, spec in REWARD_SPECS.items():
            if reward_id != "r1":
                with self.subTest(reward_id=reward_id):
                    self.assertEqual(spec["coin_collected"], 3.0)

    def test_experiment_configs_cover_active_training_rewards(self):
        paths = {
            "r2_balanced": Path("experiments/configs/reward_r2_balanced.json"),
            "r3_potential": Path("experiments/configs/reward_r3_potential.json"),
            "r4_anti_oscillation": Path(
                "experiments/configs/reward_r4_anti_oscillation.json"),
        }
        for reward_id, path in paths.items():
            with self.subTest(reward_id=reward_id):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["reward_id"], reward_id)
                self.assertEqual(config["reward_version"], reward_id)
        self.assertTrue({"r1", "r1_no_crate"}.issubset(REWARD_SPECS))

    def test_r3_potential_rewards_progress_without_action_rules(self):
        old_state = make_state(position=(3, 3), coins=((5, 3),))
        new_state = make_state(position=(4, 3), coins=((5, 3),))
        spec = resolve_reward_spec("r3_potential")
        expected_shaping = (
            spec["potential_gamma"] * _state_potential(new_state, spec)
            - _state_potential(old_state, spec)
        )
        actual = reward_from_events(
            [e.MOVED_RIGHT], "r3_potential", old_game_state=old_state,
            new_game_state=new_state)
        self.assertAlmostEqual(actual, spec["step"] + expected_shaping)
        self.assertGreater(actual, spec["step"])

    def test_r3_terminal_uses_zero_absorbing_state_potential(self):
        state = make_state(position=(3, 3), coins=((5, 3),))
        spec = resolve_reward_spec("r3_potential")
        actual = reward_from_events(
            [e.GOT_KILLED], "r3_potential", old_game_state=state,
            terminal=True)
        self.assertAlmostEqual(
            actual, spec["step"] + spec["got_killed"]
            - _state_potential(state, spec))

    def test_r4_penalizes_only_flagged_repeated_oscillation(self):
        old_state = make_state(position=(3, 3), coins=((8, 3),))
        new_state = make_state(position=(2, 3), coins=((8, 3),))
        baseline = reward_from_events(
            [e.MOVED_LEFT], "r4_anti_oscillation",
            old_game_state=old_state, new_game_state=new_state)
        penalized = reward_from_events(
            [e.MOVED_LEFT], "r4_anti_oscillation",
            old_game_state=old_state, new_game_state=new_state,
            repeated_oscillation=True)
        self.assertAlmostEqual(
            penalized - baseline,
            resolve_reward_spec("r4_anti_oscillation")["oscillation_penalty"])

    def test_r4_idle_penalty_starts_on_second_wait_and_caps(self):
        state = make_state(position=(3, 3), coins=((8, 3),))
        values = [reward_from_events(
            [e.WAITED], "r4_anti_oscillation",
            old_game_state=state, new_game_state=state, idle_streak=streak,
        ) for streak in (1, 2, 3, 4, 8)]
        self.assertAlmostEqual(values[1] - values[0], -0.04)
        self.assertAlmostEqual(values[2] - values[0], -0.08)
        self.assertAlmostEqual(values[3] - values[0], -0.12)
        self.assertAlmostEqual(values[4], values[3])

    def test_optimized_r3_potential_matches_full_context_definition(self):
        state = make_state(
            position=(3, 3), coins=((7, 3),), bombs=(((3, 6), 1),))
        state["field"][5, 3] = 1
        spec = resolve_reward_spec("r3_potential")
        context = build_context(state)
        position = context.position
        coin_distance = float(context.coin_distance[position])
        expected = 0.0
        if np.isfinite(coin_distance):
            expected += spec["potential_coin_weight"] * exp(-coin_distance / 4.0)
        else:
            crate_distance = float(context.crate_frontier_distance[position])
            if np.isfinite(crate_distance):
                expected += spec["potential_crate_weight"] * exp(-crate_distance / 4.0)
        earliest = int(context.earliest_danger[position])
        safety = 1.0 if earliest > HORIZON else max(0.0, earliest - 1) / HORIZON
        expected += spec["potential_safety_weight"] * safety
        self.assertAlmostEqual(_state_potential(state, spec), expected)

if __name__ == "__main__":
    unittest.main()
