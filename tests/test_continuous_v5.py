from types import SimpleNamespace
import unittest

from agent_code.rainbow_lite_v5_agent.features import (
    features_for_state, init_action_history, record_selected_action,
)
from agent_code.learning_common.temporal_reward import (
    init_temporal_reward_state, temporal_reward_context,
)
from agent_code.team_agent.feature_system import ACTIONS, get_feature_schema
from agent_code.team_agent.feature_system.continuous_v5 import extract
from agent_code.team_agent.rewards import reward_from_events, resolve_reward_spec
from tests.test_danger import make_game_state


class ContinuousV5Tests(unittest.TestCase):
    def test_directional_density_exposes_crate_rich_half_plane(self):
        state = make_game_state(position=(7, 7))
        state["field"][:] = 0
        state["field"][0, :] = state["field"][-1, :] = -1
        state["field"][:, 0] = state["field"][:, -1] = -1
        for position in ((9, 7), (10, 6), (11, 8), (5, 7)):
            state["field"][position] = 1
        item = extract(state)
        self.assertEqual(get_feature_schema("continuous-v5").vector_shape, (140,))
        self.assertEqual(item.vector.shape, (140,))
        right = 126 + ACTIONS.index("RIGHT") * 2
        left = 126 + ACTIONS.index("LEFT") * 2
        self.assertGreater(item.vector[right], item.vector[left])

    def test_bomb_objective_persists_after_bomb_resolves(self):
        owner = SimpleNamespace()
        init_action_history(owner)
        state = make_game_state(position=(3, 3))
        state.update(round=1, step=1)
        state["field"][5, 3] = 1
        record_selected_action(owner, state, "BOMB")
        self.assertIsNotNone(owner.feature_bomb_objective)
        self.assertEqual(
            extract(state, bomb_objective=owner.feature_bomb_objective).vector[-1],
            1.0,
        )
        resolved = make_game_state(position=(2, 3))
        resolved.update(round=1, step=2)
        resolved["field"][5, 3] = 1
        features_for_state(owner, resolved)
        self.assertIsNotNone(owner.feature_bomb_objective)
        self.assertEqual(owner.feature_bomb_objective_ttl, 23)

    def test_r17_penalizes_immediate_zero_utility_bomb(self):
        state = make_game_state(position=(3, 3))
        wait = reward_from_events(
            [], "r17_global_crate_bomb_discipline",
            old_game_state=state, new_game_state=state, action="WAIT")
        bomb = reward_from_events(
            [], "r17_global_crate_bomb_discipline",
            old_game_state=state, new_game_state=state, action="BOMB")
        self.assertAlmostEqual(bomb - wait, -1.0)

    def test_no_safety_locked_contract_is_independent_copy_of_r15(self):
        locked = resolve_reward_spec("r16_no_safety_locked")
        previous = resolve_reward_spec("r15_no_safety_objective_credit")
        self.assertEqual(locked, previous)
        locked["useless_bomb_penalty"] = 0.0
        self.assertEqual(
            resolve_reward_spec("r16_no_safety_locked")["useless_bomb_penalty"],
            -2.0,
        )

    def test_r18_penalizes_wait_when_safe_useful_bomb_is_only_progress(self):
        state = make_game_state(position=(4, 3))
        state["coins"] = []
        state["field"][5, 3] = 1
        r17_owner = SimpleNamespace()
        init_temporal_reward_state(r17_owner)
        r17 = temporal_reward_context(
            r17_owner, "WAIT", state, state, [],
            reward_id="r17_global_crate_bomb_discipline")
        self.assertEqual(r17["avoidable_wait_streak"], 0)

        r18_owner = SimpleNamespace()
        init_temporal_reward_state(r18_owner)
        first = temporal_reward_context(
            r18_owner, "WAIT", state, state, [],
            reward_id="r18_wait_attractor_escape")
        second = temporal_reward_context(
            r18_owner, "WAIT", state, state, [],
            reward_id="r18_wait_attractor_escape")
        self.assertEqual(first["avoidable_wait_streak"], 1)
        self.assertEqual(second["avoidable_wait_streak"], 2)
        baseline = reward_from_events([], "r18_wait_attractor_escape")
        penalized = reward_from_events(
            [], "r18_wait_attractor_escape",
            avoidable_wait_streak=second["avoidable_wait_streak"])
        self.assertAlmostEqual(penalized - baseline, -0.2)


if __name__ == "__main__":
    unittest.main()
