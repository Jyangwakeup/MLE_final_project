"""Regression tests for the isolated history-aware Double Q(lambda) agent."""

from types import SimpleNamespace
import unittest

import numpy as np

from agent_code.learning_common.action_history import (
    advance_observation,
    init_action_history,
    record_selected_action,
)
from agent_code.learning_common.tile_coding import TraceControl
from agent_code.optimized_double_q_lambda_v4_agent.callbacks import (
    ACTION_FEATURE_INDICES,
    HYPERPARAMETERS,
)
from agent_code.optimized_double_q_lambda_v4_agent.features import features_for_state
from experiments.agent_contracts import resolve_agent_contract
from tests.test_danger import make_game_state


class OptimizedDoubleQLambdaV4Tests(unittest.TestCase):
    def test_contract_and_action_projection_cover_v4_history(self):
        contract = resolve_agent_contract("optimized_double_q_lambda_v4_agent")
        self.assertEqual(contract.algorithm, "double_q_lambda")
        self.assertEqual(contract.feature_id, "continuous-v4")
        self.assertEqual(contract.feature_schema["vector_shape"], [126])
        self.assertTrue(HYPERPARAMETERS["watkins_trace_cut"])
        for action, indices in enumerate(ACTION_FEATURE_INDICES):
            self.assertIn(107, indices)
            self.assertEqual(indices[-3:], list(range(
                108 + action * 3, 111 + action * 3)))

    def test_adapter_observes_wait_and_cycle_history_and_resets_round(self):
        owner = SimpleNamespace()
        init_action_history(owner)
        first = make_game_state(position=(3, 3))
        first.update(round=1, step=1)
        advance_observation(owner, first)
        record_selected_action(owner, first, "WAIT")
        second = make_game_state(position=(3, 3))
        second.update(round=1, step=2)
        value = features_for_state(owner, second, previous_action="WAIT", wait_streak=1)
        self.assertEqual(value.vector.shape, (126,))
        self.assertGreater(value.vector[107], 0.0)

        next_round = make_game_state(position=(3, 3))
        next_round.update(round=2, step=1)
        advance_observation(owner, next_round)
        self.assertEqual(owner.feature_wait_streak, 0)
        self.assertEqual(owner.feature_position_history, [(3, 3)])

    def test_watkins_exploration_cuts_old_trace_but_updates_current(self):
        learner = TraceControl(
            126, 6, seed=7, hyperparameters=HYPERPARAMETERS,
            algorithm="double_q_lambda")
        learner.traces[0, 1, 3] = 1.0
        learner.set_behavior_greedy(False)
        transition = SimpleNamespace(
            state=np.zeros(126, dtype=np.float32), action=0, reward=1.0,
            next_state=np.zeros(126, dtype=np.float32), done=False,
            next_legal=np.ones(6, dtype=bool), steps=1)
        learner.observe(transition)
        self.assertEqual(float(learner.traces[0, 1, 3]), 0.0)
        self.assertGreater(float(learner.traces.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
