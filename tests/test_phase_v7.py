from types import SimpleNamespace
import unittest
import events as e

from all_other_agent_code.rainbow_lite_v7_agent.features import features_for_state, init_action_history
from agent_code.learning_common.temporal_reward import init_temporal_reward_state, temporal_reward_context
from agent_code.team_agent.rewards import resolve_reward_spec, reward_from_events
from tests.test_danger import make_game_state


class PhaseV7Tests(unittest.TestCase):
    def test_score_delta_tracks_own_kill_in_frozen_runtime(self):
        owner = SimpleNamespace(); init_action_history(owner)
        state = make_game_state(); state.update(round=1, step=1)
        first = features_for_state(owner, state)
        killed = make_game_state(); killed.update(round=1, step=2)
        killed["self"] = ("test", 5, True, killed["self"][3])
        second = features_for_state(owner, killed)
        self.assertEqual(first.vector.shape, (162,))
        self.assertEqual(second.vector[-2], 1 / 3)
        self.assertEqual(second.vector[-1], 1.0)

    def test_training_context_counts_own_kills(self):
        owner = SimpleNamespace(); init_temporal_reward_state(owner)
        state = make_game_state()
        context = temporal_reward_context(owner, "WAIT", state, state, [e.KILLED_OPPONENT], reward_id="r21_phase_potential")
        self.assertEqual(context["own_kills_before"], 0)
        self.assertEqual(context["own_kills_after"], 1)

    def test_r21_remains_potential_based_for_bomb_pressure(self):
        state = make_game_state(position=(3, 3)); state["others"] = [("enemy", 0, True, (5, 3))]
        wait = reward_from_events([], "r21_phase_potential", old_game_state=state, new_game_state=state, action="WAIT")
        bomb = reward_from_events([], "r21_phase_potential", old_game_state=state, new_game_state=state, action="BOMB")
        self.assertAlmostEqual(wait, bomb)

    def test_r22_uses_fixed_weights_before_first_kill(self):
        spec = resolve_reward_spec("r22_two_phase_potential")
        self.assertNotIn("phase_crate_low", spec)
        self.assertNotIn("phase_crate_high", spec)
        self.assertEqual(spec["potential_coin_weight"], 0.5)
        self.assertEqual(spec["potential_crate_weight"], 0.25)
        self.assertEqual(spec["potential_opponent_weight"], 0.5)

    def test_r22_remains_potential_based_for_bomb_pressure(self):
        state = make_game_state(position=(3, 3)); state["others"] = [("enemy", 0, True, (5, 3))]
        wait = reward_from_events([], "r22_two_phase_potential", old_game_state=state, new_game_state=state, action="WAIT")
        bomb = reward_from_events([], "r22_two_phase_potential", old_game_state=state, new_game_state=state, action="BOMB")
        self.assertAlmostEqual(wait, bomb)

if __name__ == "__main__": unittest.main()
