import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from agent_code.double_dqn_continuous_v2_agent import callbacks, train
from agent_code.learning_common.action_history import action_history_state
from agent_code.learning_common.action_history import (
    snapshot_history, project_history, bomb_history, own_bomb_history_for_state,
)


def state(step=1, available=True, bombs=()):
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    field[2::2, 2::2] = -1
    return dict(round=1, step=step, field=field, coins=[(5, 1)],
                bombs=list(bombs), explosion_map=np.zeros_like(field),
                self=('me', 0, available, (3, 3)), others=[])


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        safety = json.loads(Path('experiments/configs/task3_controllable_safety_train_v5.json').read_text())['safety']
        env = patch.dict(os.environ, {
            'BOMBERMAN_CHECKPOINT': str(Path(self.tmp.name) / 'model.pt'),
            'BOMBERMAN_REWARD_ID': 'r7_safe_credit_sparse',
            'BOMBERMAN_TRAINING_TASK': 'weak_opponents',
            'BOMBERMAN_N_STEP': '5',
            'BOMBERMAN_SAFETY_SPEC': json.dumps(safety),
        })
        env.start()
        self.addCleanup(env.stop)
        self.owner = SimpleNamespace(train=True)
        callbacks.setup(self.owner)
        self.owner.exploration_spec = dict(version='linear-v1', start=0., end=0., decay_action_steps=100)
        train.setup_training(self.owner)
        self.owner.model.q_values = lambda x: np.array([0, 0, 0, 0, 0, 10.], dtype=np.float32)

    def test_bomb_responsibility_survives_actual_training_callback(self):
        old = state()
        new = state(2, False, [((3, 3), 3)])
        action = callbacks.act(self.owner, old)
        self.assertEqual(action, 'BOMB')
        before = copy.deepcopy(action_history_state(self.owner))
        train.game_events_occurred(self.owner, old, action, new, ['BOMB_DROPPED'])
        self.assertEqual(action_history_state(self.owner), before)
        self.assertTrue(self.owner.feature_own_bomb_pending)
        expected = self.owner.pending[1]
        callbacks.act(self.owner, new)
        self.assertTrue(self.owner.last_safety_diagnostic['own_bomb_pending'])
        np.testing.assert_array_equal(expected.next_legal, self.owner._last_decision_mask)
        np.testing.assert_array_equal(expected.next_state, self.owner._feature_cache_value.vector)

    def test_queries_and_next_projection_do_not_mutate_live_state(self):
        old, new = state(), state(2, False, [((3, 3), 3)])
        action = callbacks.act(self.owner, old)
        history = copy.deepcopy(action_history_state(self.owner))
        rng = self.owner.rng.getstate()
        count = self.owner.safety_decisions
        certificate = copy.deepcopy(self.owner._own_bomb_placement_certificate)
        for _ in range(2):
            own_bomb_history_for_state(self.owner, old)
            train._transition(self.owner, old, action, 0., new, False)
        self.assertEqual(action_history_state(self.owner), history)
        self.assertEqual(self.owner.rng.getstate(), rng)
        self.assertEqual(self.owner.safety_decisions, count)
        self.assertEqual(self.owner._own_bomb_placement_certificate, certificate)
        with self.assertRaises(ValueError):
            train._transition(self.owner, new, action, 0., old, False)
        with self.assertRaises(ValueError):
            train._transition(self.owner, old, 'WAIT', 0., new, False)

    def test_responsibility_includes_lingering_flame_then_clears(self):
        callbacks.act(self.owner, state())
        history = snapshot_history(self.owner)
        for step, timer in enumerate((3, 2, 1, 0, None), 2):
            observed = state(step, False, [] if timer is None else [((3, 3), timer)])
            if timer is None:
                observed['explosion_map'][3, 3] = 1
            projected = project_history(history, observed)
            self.assertTrue(bomb_history(projected, observed)['pending'])
        self.assertFalse(project_history(history, state(7, True)).own_bomb_pending)
        other_round = state()
        other_round['round'] = 2
        reset = project_history(history, other_round)
        self.assertFalse(reset.own_bomb_pending)
        self.assertIsNone(reset.previous_action)

    def test_terminal_replaces_pending_and_clears_caches(self):
        for ordinary_callback in (False, True):
            with self.subTest(ordinary_callback=ordinary_callback):
                old, new = state(), state(2, False, [((3, 3), 3)])
                action = callbacks.act(self.owner, old)
                vector = self.owner._decision_snapshot.vector.copy()
                count = len(self.owner.model.replay)
                if ordinary_callback:
                    train.game_events_occurred(self.owner, old, action, new, ['BOMB_DROPPED'])
                train.end_of_round(self.owner, old, action, ['KILLED_SELF'])
                self.assertEqual(len(self.owner.model.replay), count + 1)
                terminal = self.owner.model.replay._all_items()[-1]
                self.assertTrue(terminal.done)
                np.testing.assert_array_equal(terminal.state, vector)
                self.assertIsNone(self.owner._decision_snapshot)
                self.assertIsNone(self.owner._feature_cache_key)
                self.assertFalse(self.owner.feature_own_bomb_pending)

    def test_move_and_wait_history_match_next_decision(self):
        for action in ('RIGHT', 'WAIT'):
            index = callbacks.ACTIONS.index(action)
            self.owner.model.q_values = lambda x: np.eye(6, dtype=np.float32)[index]
            old = state()
            chosen = callbacks.act(self.owner, old)
            self.assertEqual(chosen, action)
            new = state(2)
            if action == 'RIGHT':
                new['self'] = ('me', 0, True, (4, 3))
            new['coins'] = [(1, 5)]
            train.game_events_occurred(self.owner, old, chosen, new, [])
            transition = self.owner.pending[1]
            callbacks.act(self.owner, new)
            np.testing.assert_array_equal(transition.next_state, self.owner._decision_snapshot.vector)
            train.end_of_round(self.owner, new, action, [])


if __name__ == '__main__':
    unittest.main()
