"""Official same-step callback must project to the following act clock."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from tests import test_task3_lifecycle as lifecycle
state = lifecycle.state
from agent_code.double_dqn_continuous_v2_agent import callbacks,train
from agent_code.learning_common.action_history import action_history_state

class CompactLifecycleTests(unittest.TestCase):
    def setUp(self):
        lifecycle.LifecycleTests.setUp(self)
        self.owner.safety_spec=json.loads(Path('experiments/configs/safety_proven_movement_v9.json').read_text())['safety']

    def test_official_same_step_callback_matches_following_act(self):
        old=state();action=callbacks.act(self.owner,old)
        self.assertEqual(action,'BOMB')
        post=state(1,False,[((3,3),3)]);before=copy.deepcopy(post)
        history=copy.deepcopy(action_history_state(self.owner));rng=self.owner.rng.getstate()
        transition=train._transition(self.owner,old,action,0.,post,False)
        self.assertEqual(post['step'],1);np.testing.assert_array_equal(post['field'],before['field'])
        self.assertEqual(action_history_state(self.owner),history);self.assertEqual(self.owner.rng.getstate(),rng)
        following={**post,'step':2};callbacks.act(self.owner,following)
        np.testing.assert_array_equal(transition.next_state,self.owner._feature_cache_value.vector)
        np.testing.assert_array_equal(transition.next_legal,self.owner._last_decision_mask)
