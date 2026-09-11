import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import events as e
import numpy as np
import torch

from agent_code.dqn_agent.callbacks import (
    FEATURE_VERSION,
    INPUT_SIZE,
    _features_for,
    act,
    legal_actions,
    setup,
    state_to_features,
)
from agent_code.dqn_agent.model import DQN, Transition
from agent_code.dqn_agent.train import (
    end_of_round,
    game_events_occurred,
    reward_from_events,
    setup_training,
)
from agent_code.q_learning_agent.features import features_for_state as q_features_for_state
from agent_code.team_agent.features import FEATURE_DIM
from agent_code.team_agent.rewards import reward_from_events as shared_reward_from_events
from tests.test_danger import make_game_state


class DQNRewardTestCase(unittest.TestCase):
    def test_objective_outcomes_and_time_cost_are_combined(self):
        events = [
            e.COIN_COLLECTED,
            e.KILLED_OPPONENT,
            e.CRATE_DESTROYED,
            e.CRATE_DESTROYED,
            e.INVALID_ACTION,
        ]
        self.assertAlmostEqual(reward_from_events(events), 6.19)

    def test_death_is_penalized_only_once(self):
        self.assertAlmostEqual(
            reward_from_events([e.KILLED_SELF, e.GOT_KILLED]),
            -10.01,
        )

    def test_movement_directions_have_no_supervised_action_reward(self):
        for movement in (e.MOVED_UP, e.MOVED_RIGHT, e.MOVED_DOWN, e.MOVED_LEFT):
            with self.subTest(movement=movement):
                self.assertAlmostEqual(reward_from_events([movement]), -0.01)


class DQNTrainingTaskTestCase(unittest.TestCase):
    def test_shared_flag_disables_bombs_during_training_and_evaluation(self):
        for training in (True, False):
            with self.subTest(training=training):
                agent = SimpleNamespace(
                    train=training,
                    logger=Mock(),
                    rng=Mock(),
                    action_steps=0,
                    allow_bomb=False,
                    _feature_cache_key=None,
                    _feature_cache_value=None,
                    model=Mock(),
                )
                agent.model.q_values.return_value = np.array(
                    [0, 0, 0, 0, 0, 100], dtype=np.float32
                )
                agent.rng.random.return_value = 1.0
                agent.rng.choice.side_effect = lambda choices: choices[0]

                action = act(agent, make_game_state(bombs_left=True))

                self.assertNotEqual(action, "BOMB")

    def test_new_task_resets_progress_but_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "dqn-model.pt"
            model = DQN(INPUT_SIZE, 6)
            checkpoint = model.checkpoint()
            checkpoint.update({
                "feature_version": FEATURE_VERSION,
                "action_steps": 123,
                "training_task": "task1",
            })
            torch.save(checkpoint, model_file)
            expected_parameter = next(model.policy.parameters()).detach().clone()
            agent = SimpleNamespace(train=True, logger=Mock())

            with (
                patch.dict(os.environ, {"BOMBERMAN_TRAINING_TASK": "task2"}),
                patch("agent_code.dqn_agent.callbacks.MODEL_FILE", model_file),
            ):
                setup(agent)

        self.assertEqual(agent.action_steps, 0)
        self.assertEqual(agent.training_task, "task2")
        self.assertTrue(torch.equal(
            next(agent.model.policy.parameters()).detach(),
            expected_parameter,
        ))

    def test_incompatible_checkpoint_is_rejected_during_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "old-model.pt"
            checkpoint = DQN(INPUT_SIZE, 6).checkpoint()
            checkpoint["feature_version"] = "old-features"
            torch.save(checkpoint, model_file)
            agent = SimpleNamespace(train=False, logger=Mock())

            with patch.dict(os.environ, {"BOMBERMAN_CHECKPOINT": str(model_file)}):
                with self.assertRaisesRegex(ValueError, "feature version"):
                    setup(agent)


class DQNSharedFeatureTestCase(unittest.TestCase):
    def test_vector_and_legal_mask_come_from_team_features(self):
        state = make_game_state(bombs_left=True)
        vector = state_to_features(state)
        legal = legal_actions(state)

        self.assertEqual(INPUT_SIZE, FEATURE_DIM)
        self.assertEqual(vector.shape, (40,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(legal.shape, (6,))
        self.assertTrue(legal[-1])

        dqn_features = _features_for(
            SimpleNamespace(_feature_cache_key=None, _feature_cache_value=None),
            state,
        )
        q_features = q_features_for_state(state)
        np.testing.assert_array_equal(dqn_features.vector, q_features.vector)
        np.testing.assert_array_equal(dqn_features.legal_mask, q_features.legal_mask)

    def test_dqn_reward_is_the_shared_reward_function(self):
        self.assertIs(reward_from_events, shared_reward_from_events)

    def test_shared_features_are_cached_per_round_step(self):
        agent = SimpleNamespace(_feature_cache_key=None, _feature_cache_value=None)
        state = make_game_state()
        state.update(round=1, step=1)
        self.assertIs(_features_for(agent, state), _features_for(agent, state))


class DQNTransitionTestCase(unittest.TestCase):
    def _agent(self):
        model = Mock()
        model.checkpoint.return_value = {}
        return SimpleNamespace(
            model=model,
            model_file=Path(self.directory) / "checkpoint.pt",
            action_steps=1,
            training_task="task1",
            logger=Mock(),
            _feature_cache_key=None,
            _feature_cache_value=None,
        )

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = self.temp_directory.name

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_last_step_is_observed_once_as_terminal(self):
        agent = self._agent()
        setup_training(agent)
        old_state = make_game_state()
        old_state.update(round=1, step=4)
        new_state = make_game_state(position=(3, 4))
        new_state.update(round=1, step=5)

        game_events_occurred(agent, old_state, "WAIT", new_state, [e.WAITED])
        end_of_round(agent, old_state, "WAIT", [e.WAITED, e.SURVIVED_ROUND])

        agent.model.observe.assert_called_once()
        transition = agent.model.observe.call_args.args[0]
        self.assertIsInstance(transition, Transition)
        self.assertTrue(transition.done)
        self.assertIsNone(transition.next_state)
        self.assertAlmostEqual(transition.reward, 0.99)


if __name__ == "__main__":
    unittest.main()
