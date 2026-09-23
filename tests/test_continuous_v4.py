import math
from types import SimpleNamespace
import unittest

import numpy as np

from agent_code.learning_common.action_history import (
    advance_observation, action_history_for_state, action_history_state, init_action_history,
    load_action_history_state, record_selected_action,
)
from agent_code.learning_common.temporal_reward import (
    init_temporal_reward_state, temporal_reward_context,
)
from agent_code.learning_common.linear_agent import repeated_cycle_mask
from agent_code.team_agent.feature_system import ACTIONS, get_feature_schema
from agent_code.team_agent.feature_system.continuous_v4 import (
    _cycle_detail, extract,
)
from agent_code.team_agent.rewards import reward_from_events
from tests.test_danger import make_game_state


class ContinuousV4FeatureTests(unittest.TestCase):
    def test_schema_wait_encoding_and_non_move_cycle_fields(self):
        state = make_game_state(position=(3, 3))
        result = extract(
            state, previous_action="WAIT", wait_streak=3,
            position_history=[(3, 3), (4, 3)],
        )
        self.assertEqual(get_feature_schema("continuous-v4").vector_shape, (126,))
        self.assertEqual(result.vector.shape, (126,))
        self.assertAlmostEqual(result.vector[107], math.log1p(3) / math.log1p(8))
        for action in ("WAIT", "BOMB"):
            offset = 108 + ACTIONS.index(action) * 3
            np.testing.assert_array_equal(result.vector[offset:offset + 3], 0.0)

    def test_detects_two_three_and_four_position_cycles(self):
        self.assertEqual(_cycle_detail([(3, 3), (4, 3)], (3, 3)), (True, 2, 1))
        self.assertEqual(
            _cycle_detail([(3, 3), (4, 3), (4, 4)], (3, 3)),
            (True, 3, 1),
        )
        once = [(3, 3), (4, 3), (4, 4), (3, 4)]
        twice = once + [(3, 3), (4, 3), (4, 4), (3, 4)]
        self.assertEqual(_cycle_detail(once, (3, 3)), (True, 4, 1))
        self.assertEqual(_cycle_detail(twice, (3, 3)), (True, 4, 2))

    def test_action_conditional_cycle_fields_only_mark_closing_move(self):
        state = make_game_state(position=(3, 4))
        history = [(3, 3), (4, 3), (4, 4), (3, 4)]
        result = extract(state, position_history=history)
        up = 108 + ACTIONS.index("UP") * 3
        self.assertEqual(result.vector[up], 1.0)
        self.assertAlmostEqual(result.vector[up + 1], 0.5)
        self.assertAlmostEqual(result.vector[up + 2], 0.25)
        down = 108 + ACTIONS.index("DOWN") * 3
        np.testing.assert_array_equal(result.vector[down:down + 3], 0.0)

    def test_r12_masks_repeated_cycle_closure_but_keeps_alternatives(self):
        state = make_game_state(position=(3, 4))
        repeated = [
            (3, 3), (4, 3), (4, 4), (3, 4),
            (3, 3), (4, 3), (4, 4), (3, 4),
        ]
        item = extract(state, position_history=repeated)
        legal = np.ones(len(ACTIONS), dtype=bool)
        masked = repeated_cycle_mask(
            item, legal, "r12_coin_priority_anti_loop")
        self.assertFalse(masked[ACTIONS.index("UP")])
        self.assertTrue(masked[ACTIONS.index("RIGHT")])
        np.testing.assert_array_equal(
            repeated_cycle_mask(item, legal, "r10_bounded_history_anti_loop"),
            legal,
        )

    def test_r12_cycle_constraint_falls_back_when_every_action_is_vetoed(self):
        item = SimpleNamespace(vector=np.zeros(126, dtype=np.float32))
        item.vector[110] = 0.5
        legal = np.zeros(len(ACTIONS), dtype=bool)
        legal[ACTIONS.index("UP")] = True
        np.testing.assert_array_equal(
            repeated_cycle_mask(item, legal, "r12_coin_priority_anti_loop"),
            legal,
        )

    def test_r12_reward_only_does_not_eliminate_cycle_actions(self):
        item = SimpleNamespace(vector=np.zeros(126, dtype=np.float32))
        item.vector[110] = 0.5
        legal = np.ones(len(ACTIONS), dtype=bool)
        np.testing.assert_array_equal(
            repeated_cycle_mask(
                item, legal, "r12_coin_priority_anti_loop_reward_only"),
            legal,
        )

    def test_history_records_each_observed_step_once_and_round_trips(self):
        owner = SimpleNamespace()
        init_action_history(owner)
        first = make_game_state(position=(3, 3))
        first.update(round=1, step=1)
        advance_observation(owner, first)
        advance_observation(owner, first)
        record_selected_action(owner, first, "WAIT")
        second = make_game_state(position=(4, 3))
        second.update(round=1, step=2)
        self.assertEqual(action_history_for_state(owner, second), ("WAIT", 1))
        self.assertEqual(owner.feature_position_history, [(3, 3)])
        advance_observation(owner, second)
        self.assertEqual(owner.feature_position_history, [(3, 3), (4, 3)])

        clone = SimpleNamespace()
        load_action_history_state(clone, action_history_state(owner))
        self.assertEqual(clone.feature_position_history, owner.feature_position_history)
        self.assertEqual(clone.feature_position_history_key, (1, 2))

        next_round = make_game_state(position=(3, 3))
        next_round.update(round=2, step=1)
        self.assertEqual(action_history_for_state(clone, next_round), (None, 0))
        advance_observation(clone, next_round)
        self.assertEqual(clone.feature_position_history, [(3, 3)])


class BoundedHistoryRewardTests(unittest.TestCase):
    @staticmethod
    def _state(position, step):
        state = make_game_state(position=position)
        state.update(round=1, step=step)
        state["coins"] = [(7, 3)]
        return state

    def test_first_loop_is_observed_and_second_loop_is_penalized(self):
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        positions = [
            (3, 3), (4, 3), (4, 4), (3, 4), (3, 3),
            (4, 3), (4, 4), (3, 4), (3, 3),
        ]
        actions = ["RIGHT", "DOWN", "LEFT", "UP"] * 2
        contexts = []
        for index, action in enumerate(actions):
            contexts.append(temporal_reward_context(
                owner, action, self._state(positions[index], index + 1),
                self._state(positions[index + 1], index + 2), [],
                reward_id="r10_bounded_history_anti_loop",
            ))
        self.assertEqual(contexts[3]["loop_repeat_count"], 1)
        self.assertFalse(contexts[3]["conditional_loop"])
        self.assertEqual(contexts[7]["loop_repeat_count"], 2)
        self.assertTrue(contexts[7]["conditional_loop"])

    def test_second_avoidable_wait_gets_escalating_reward(self):
        owner = SimpleNamespace()
        init_temporal_reward_state(owner)
        state = self._state((3, 3), 1)
        first = temporal_reward_context(
            owner, "WAIT", state, state, [],
            reward_id="r10_bounded_history_anti_loop")
        second = temporal_reward_context(
            owner, "WAIT", state, state, [],
            reward_id="r10_bounded_history_anti_loop")
        self.assertEqual(first["avoidable_wait_streak"], 1)
        self.assertEqual(second["avoidable_wait_streak"], 2)
        baseline = reward_from_events([], "r10_bounded_history_anti_loop")
        penalized = reward_from_events(
            [], "r10_bounded_history_anti_loop",
            avoidable_wait_streak=second["avoidable_wait_streak"])
        self.assertAlmostEqual(penalized - baseline, -0.04)

    def test_r12_scales_repeated_loops_more_strongly_than_r10(self):
        baseline_r10 = reward_from_events([], "r10_bounded_history_anti_loop")
        baseline_r12 = reward_from_events([], "r12_coin_priority_anti_loop")
        loop_r10 = reward_from_events(
            [], "r10_bounded_history_anti_loop", loop_repeat_count=6)
        loop_r12 = reward_from_events(
            [], "r12_coin_priority_anti_loop", loop_repeat_count=6)
        self.assertAlmostEqual(loop_r10 - baseline_r10, -0.12)
        self.assertAlmostEqual(loop_r12 - baseline_r12, -1.0)

    def test_r12_reward_only_has_the_same_reward_contract(self):
        for repeat_count in (0, 2, 6):
            original = reward_from_events(
                [], "r12_coin_priority_anti_loop",
                loop_repeat_count=repeat_count, avoidable_wait_streak=3)
            reward_only = reward_from_events(
                [], "r12_coin_priority_anti_loop_reward_only",
                loop_repeat_count=repeat_count, avoidable_wait_streak=3)
            self.assertEqual(reward_only, original)

    def test_r12_suppresses_useful_bomb_bonus_when_coin_is_reachable(self):
        state = self._state((3, 3), 1)
        state["field"][5, 3] = 1
        wait = reward_from_events(
            [], "r12_coin_priority_anti_loop",
            old_game_state=state, new_game_state=state, action="WAIT")
        bomb = reward_from_events(
            [], "r12_coin_priority_anti_loop",
            old_game_state=state, new_game_state=state, action="BOMB")
        self.assertAlmostEqual(bomb, wait)


if __name__ == "__main__":
    unittest.main()
