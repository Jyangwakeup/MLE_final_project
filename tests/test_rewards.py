from math import exp
from pathlib import Path
import unittest
import json

import events as e
import numpy as np
import settings as s
from agent_code.team_agent.danger import HORIZON
from agent_code.team_agent.feature_system.common import build_context

from agent_code.team_agent.rewards import (
    REWARD_SPECS,
    REWARD_VERSION,
    _state_potential,
    identify_checkpoint_reward_version,
    resolve_reward_spec,
    reward_from_events,
)
from agent_code.learning_common.temporal_reward import (
    init_temporal_reward_state, temporal_reward_context,
)
from types import SimpleNamespace


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
    def test_r13_combines_dense_survival_and_causal_credit_contracts(self):
        spec = resolve_reward_spec("r13_no_safety_survival_credit")
        self.assertEqual(spec["potential_survival_options_weight"], 2.0)
        self.assertEqual(spec["causal_bomb_death_penalty"], -25.0)
        self.assertEqual(spec["avoidable_fatal_action"], -40.0)

        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        placed = make_state()
        placed["self"] = ("reward_test", 0, False, (3, 3))
        context = temporal_reward_context(
            owner, "BOMB", make_state(), placed, [e.BOMB_DROPPED],
            reward_id="r13_no_safety_survival_credit")
        self.assertEqual(context["temporal_adjustment"], 0.0)

    def test_r13_survival_options_are_bounded_in_state_potential(self):
        state = make_state()
        base = _state_potential(state, resolve_reward_spec(
            "r10_bounded_history_anti_loop"))
        shaped = _state_potential(state, resolve_reward_spec(
            "r13_no_safety_survival_credit"))
        self.assertTrue(np.isfinite(shaped))
        self.assertGreaterEqual(shaped - base, 0.0)
        self.assertLessEqual(shaped - base, 2.0)

    def test_r14_only_rewards_resolved_bombs_with_observed_utility(self):
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        placed = make_state()
        placed["self"] = ("reward_test", 0, False, (3, 3))
        temporal_reward_context(
            owner, "BOMB", make_state(), placed, [e.BOMB_DROPPED],
            reward_id="r14_no_safety_useful_bomb_credit")
        resolved = make_state(position=(4, 3))
        empty = temporal_reward_context(
            owner, "RIGHT", placed, resolved, [],
            reward_id="r14_no_safety_useful_bomb_credit")
        self.assertTrue(empty["bomb_resolved_alive"])
        self.assertEqual(empty["resolved_bomb_crates"], 0)
        self.assertEqual(empty["temporal_adjustment"], 0.0)

        init_temporal_reward_state(owner)
        temporal_reward_context(
            owner, "BOMB", make_state(), placed, [e.BOMB_DROPPED],
            reward_id="r14_no_safety_useful_bomb_credit")
        active = make_state(position=(4, 3))
        active["self"] = ("reward_test", 0, False, (4, 3))
        temporal_reward_context(
            owner, "RIGHT", placed, active, [e.CRATE_DESTROYED],
            reward_id="r14_no_safety_useful_bomb_credit")
        useful = temporal_reward_context(
            owner, "WAIT", active, resolved, [],
            reward_id="r14_no_safety_useful_bomb_credit")
        self.assertEqual(useful["resolved_bomb_crates"], 1)
        self.assertAlmostEqual(useful["temporal_adjustment"], 1.5)

    def test_r15_directly_penalizes_zero_utility_bomb_credit(self):
        spec = resolve_reward_spec("r15_no_safety_objective_credit")
        self.assertEqual(spec["causal_bomb_zero_utility_penalty"], -2.0)
        self.assertEqual(spec["potential_coin_weight"], 1.5)
        self.assertEqual(spec["suppress_useful_bomb_when_coin_reachable"], 1.0)

    def test_r11_delays_crate_credit_and_rewards_resolved_survival(self):
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        placed = make_state(); placed["self"] = ("reward_test", 0, False, (3, 3))
        context = temporal_reward_context(owner, "BOMB", make_state(), placed,
            [e.BOMB_DROPPED], reward_id="r11_causal_bomb_credit")
        self.assertEqual(context["temporal_adjustment"], 0.0)
        active = make_state(); active["self"] = ("reward_test", 0, False, (4, 3))
        context = temporal_reward_context(owner, "RIGHT", placed, active,
            [e.CRATE_DESTROYED], reward_id="r11_causal_bomb_credit")
        self.assertEqual(context["temporal_adjustment"], 0.0)
        resolved = make_state(position=(4, 3))
        context = temporal_reward_context(owner, "WAIT", active, resolved, [],
            reward_id="r11_causal_bomb_credit")
        self.assertAlmostEqual(context["temporal_adjustment"], 3.1)
        reward = reward_from_events([e.CRATE_DESTROYED],
            "r11_causal_bomb_credit", temporal_adjustment=context["temporal_adjustment"])
        self.assertAlmostEqual(reward, 3.09)

    def test_r11_self_kill_dominates_same_step_positive_events(self):
        reward = reward_from_events(
            [e.KILLED_SELF, e.COIN_COLLECTED, e.CRATE_DESTROYED],
            "r11_causal_bomb_credit")
        self.assertAlmostEqual(reward, -40.01)

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
        self.assertIn("r1_no_crate", REWARD_SPECS)
        self.assertEqual(resolve_reward_spec("r1_no_crate")["coin_collected"], 1.0)
        self.assertEqual(resolve_reward_spec("r1_coin3")["coin_collected"], 3.0)
        self.assertEqual(resolve_reward_spec("r1_coin3_no_crate")["coin_collected"], 3.0)
        self.assertEqual(resolve_reward_spec("r1_no_crate")["crate_destroyed"], 0.0)
        self.assertEqual(resolve_reward_spec("r1_coin3_no_crate")["crate_destroyed"], 0.0)

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
        self.assertAlmostEqual(
            reward_from_events(events, "r1_coin3"), 8.29)

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

    def test_coin3_and_shaped_rewards_value_collected_coin_at_three(self):
        for reward_id in (
            "r1_coin3", "r1_coin3_no_crate", "r2_balanced",
            "r3_potential", "r4_anti_oscillation", "r5_conditional_loop",
        ):
            spec = REWARD_SPECS[reward_id]
            if reward_id:
                with self.subTest(reward_id=reward_id):
                    self.assertEqual(spec["coin_collected"], 3.0)

    def test_experiment_configs_cover_active_training_rewards(self):
        paths = {
            "r2_balanced": Path("experiments/configs/reward_r2_balanced.json"),
            "r5_conditional_loop": Path(
                "experiments/configs/reward_r5_conditional_loop.json"),
            "r3_potential": Path("experiments/configs/reward_r3_potential.json"),
            "r4_anti_oscillation": Path(
                "experiments/configs/reward_r4_anti_oscillation.json"),
        }
        for reward_id, path in paths.items():
            with self.subTest(reward_id=reward_id):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["reward_id"], reward_id)
                self.assertEqual(config["reward_version"], reward_id)
        self.assertTrue({
            "r1", "r1_no_crate", "r1_coin3", "r1_coin3_no_crate",
        }.issubset(REWARD_SPECS))

    def test_conditional_loop_requires_all_four_conditions(self):
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        first = make_state(position=(3, 3), coins=((8, 3),))
        middle = make_state(position=(4, 3), coins=((8, 3),))
        returned = make_state(position=(3, 3), coins=((8, 3),))

        initial = temporal_reward_context(
            owner, "RIGHT", first, middle, [], reward_id="r5_conditional_loop")
        loop = temporal_reward_context(
            owner, "LEFT", middle, returned, [], reward_id="r5_conditional_loop")
        self.assertFalse(initial["conditional_loop"])
        self.assertTrue(loop["conditional_loop"])

        owner.reward_previous_position = (3, 3)
        owner.reward_previous_coin_target = (8, 3)
        changed_target = make_state(position=(4, 3), coins=((7, 3),))
        self.assertFalse(temporal_reward_context(
            owner, "LEFT", changed_target, returned, [],
            reward_id="r5_conditional_loop")["conditional_loop"])

        owner.reward_previous_position = (3, 3)
        owner.reward_previous_coin_target = (8, 3)
        danger = make_state(
            position=(4, 3), coins=((8, 3),), bombs=(((4, 6), 0),))
        self.assertFalse(temporal_reward_context(
            owner, "LEFT", danger, returned, [],
            reward_id="r5_conditional_loop")["conditional_loop"])

    def test_conditional_wait_only_penalizes_safe_avoidable_wait(self):
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        safe = make_state(position=(3, 3), coins=((5, 3),))
        context = temporal_reward_context(
            owner, "WAIT", safe, safe, [], reward_id="r5_conditional_loop")
        self.assertTrue(context["avoidable_wait"])

        delayed_danger = make_state(
            position=(3, 3), coins=((5, 3),), bombs=(((3, 6), 1),))
        context = temporal_reward_context(
            owner, "WAIT", delayed_danger, delayed_danger, [],
            reward_id="r5_conditional_loop")
        self.assertFalse(context["avoidable_wait"])

        self.assertFalse(context["conditional_loop"])
        self.assertTrue(context["diagnostic"]["wait_next_danger_count"])

        unsafe_shortcut = make_state(
            position=(3, 3), coins=((5, 3),), bombs=(((4, 6), 0),))
        context = temporal_reward_context(
            owner, "WAIT", unsafe_shortcut, unsafe_shortcut, [],
            reward_id="r5_conditional_loop")
        self.assertFalse(context["avoidable_wait"])

    def test_r20_is_r7_plus_only_targeted_wait_signal(self):
        base = resolve_reward_spec("r7_safe_credit_potential")
        targeted = resolve_reward_spec("r20_safe_credit_targeted_wait")
        self.assertEqual(
            {key: value for key, value in targeted.items()
             if key not in {"avoidable_wait_penalty", "useful_bomb_counts_as_wait_progress"}},
            base)
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        state = make_state(position=(3, 3), coins=())
        state["field"][4, 3] = 1
        context = temporal_reward_context(
            owner, "WAIT", state, state, [], reward_id="r20_safe_credit_targeted_wait")
        self.assertTrue(context["avoidable_wait"])
        self.assertAlmostEqual(
            reward_from_events([e.WAITED], "r20_safe_credit_targeted_wait",
                               avoidable_wait=True)
            - reward_from_events([e.WAITED], "r7_safe_credit_potential"),
            targeted["avoidable_wait_penalty"])

    def test_conditional_reward_adds_only_explicit_flags(self):
        baseline = reward_from_events([e.WAITED], "r5_conditional_loop")
        spec = resolve_reward_spec("r5_conditional_loop")
        self.assertAlmostEqual(
            reward_from_events(
                [e.WAITED], "r5_conditional_loop", conditional_loop=True)
            - baseline,
            spec["conditional_loop_penalty"],
        )
        self.assertAlmostEqual(
            reward_from_events(
                [e.WAITED], "r5_conditional_loop", avoidable_wait=True)
            - baseline,
            spec["avoidable_wait_penalty"],
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            reward_from_events(
                [e.WAITED], "r5_conditional_loop",
                conditional_loop=True, avoidable_wait=True)

    def test_r9_safe_bomb_credit_and_zero_utility_penalty(self):
        spec = resolve_reward_spec("r9_safe_credit_anti_loop")
        empty = make_state()
        baseline = reward_from_events(
            [], "r9_safe_credit_anti_loop", old_game_state=empty,
            new_game_state=empty, action="WAIT")
        empty_bomb = reward_from_events(
            [], "r9_safe_credit_anti_loop", old_game_state=empty,
            new_game_state=empty, action="BOMB")
        self.assertAlmostEqual(
            empty_bomb - baseline, spec["useless_bomb_penalty"])

        for crate_count in (1, 2, 3, 4):
            state = make_state()
            for coordinate in ((5, 3), (1, 3), (3, 5), (3, 1))[:crate_count]:
                state["field"][coordinate] = 1
            reward = reward_from_events(
                [], "r9_safe_credit_anti_loop", old_game_state=state,
                new_game_state=state, action="BOMB")
            expected = min(crate_count, 3) * spec["useful_bomb_per_crate"]
            wait = reward_from_events(
                [], "r9_safe_credit_anti_loop", old_game_state=state,
                new_game_state=state, action="WAIT")
            self.assertAlmostEqual(reward - wait, expected)

        opponent = make_state()
        opponent["others"] = [("opponent", 0, True, (5, 3))]
        threatened = reward_from_events(
            [], "r9_safe_credit_anti_loop", old_game_state=opponent,
            new_game_state=opponent, action="BOMB")
        waiting = reward_from_events(
            [], "r9_safe_credit_anti_loop", old_game_state=opponent,
            new_game_state=opponent, action="WAIT")
        self.assertAlmostEqual(threatened, waiting)

        trapped = make_state()
        for coordinate in ((2, 3), (4, 3), (3, 2), (3, 4)):
            trapped["field"][coordinate] = -1
        unsafe_bomb = reward_from_events(
            [], "r9_safe_credit_anti_loop", old_game_state=trapped,
            new_game_state=trapped, action="BOMB")
        trapped_wait = reward_from_events(
            [], "r9_safe_credit_anti_loop", old_game_state=trapped,
            new_game_state=trapped, action="WAIT")
        self.assertAlmostEqual(
            unsafe_bomb - trapped_wait, spec["unsafe_bomb_penalty"])

    def test_r9_task2_loop_detection_uses_crate_frontier(self):
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        first = make_state(position=(3, 3))
        first["field"][7, 3] = 1
        middle = make_state(position=(4, 3))
        middle["field"][7, 3] = 1
        returned = make_state(position=(3, 3))
        returned["field"][7, 3] = 1
        initial = temporal_reward_context(
            owner, "RIGHT", first, middle, [],
            reward_id="r9_safe_credit_anti_loop")
        loop = temporal_reward_context(
            owner, "LEFT", middle, returned, [],
            reward_id="r9_safe_credit_anti_loop")
        self.assertFalse(initial["conditional_loop"])
        self.assertTrue(loop["conditional_loop"])

        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        waiting = temporal_reward_context(
            owner, "WAIT", first, first, [],
            reward_id="r9_safe_credit_anti_loop")
        self.assertTrue(waiting["avoidable_wait"])

    def test_r3_variants_preserve_the_r3_potential_contract(self):
        baseline = resolve_reward_spec("r3_potential")
        for reward_id in ("r4_anti_oscillation", "r5_conditional_loop"):
            spec = resolve_reward_spec(reward_id)
            with self.subTest(reward_id=reward_id):
                for key, value in baseline.items():
                    self.assertEqual(spec[key], value)

    def test_task1_paired_configs_only_change_reward_contract(self):
        baseline_path = Path(
            "experiments/configs/task1_ddqn_continuous_r3_baseline.json")
        candidate_path = Path(
            "experiments/configs/task1_ddqn_continuous_r5_conditional_loop.json")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        for config in (baseline, candidate):
            self.assertEqual(config["algorithm"], "double_dqn")
            self.assertEqual(config["feature_id"], "continuous-v2")
            self.assertEqual(config["seed"], 11)
            self.assertEqual(config["training"]["n_rounds"], 1000)
            self.assertFalse(config["training"]["early_stopping"]["enabled"])
        for key in ("reward_id", "reward_version"):
            baseline.pop(key)
            candidate.pop(key)
        self.assertEqual(baseline, candidate)

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

    def test_r7_potential_fills_visible_coin_shaping_gap(self):
        old_state = make_state(position=(3, 3), coins=((5, 3),))
        new_state = make_state(position=(4, 3), coins=((5, 3),))
        sparse = reward_from_events(
            [e.MOVED_RIGHT], "r7_safe_credit_sparse",
            old_game_state=old_state, new_game_state=new_state)
        potential = reward_from_events(
            [e.MOVED_RIGHT], "r7_safe_credit_potential",
            old_game_state=old_state, new_game_state=new_state)

        self.assertAlmostEqual(sparse, -0.01)
        self.assertGreater(potential, sparse)

    def test_r7_potential_terminal_uses_zero_absorbing_state(self):
        state = make_state(position=(3, 3), coins=((5, 3),))
        spec = resolve_reward_spec("r7_safe_credit_potential")
        actual = reward_from_events(
            [e.GOT_KILLED], "r7_safe_credit_potential",
            old_game_state=state, terminal=True)
        self.assertAlmostEqual(
            actual, spec["step"] + spec["got_killed"]
            - _state_potential(state, spec))

    def test_r4_anti_oscillation_penalizes_only_flagged_reversal(self):
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

    def test_r4_anti_oscillation_idle_penalty_starts_on_second_wait_and_caps(self):
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
