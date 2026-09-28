import unittest
import numpy as np
import torch
from agent_code.dqn_agent.model import Transition
from agent_code.learning_common.tile_coding import TileCoder, TraceControl
from all_other_agent_code.rainbow_lite_agent.model import DuelingNetwork, PrioritizedReplay, RainbowLite

HP={"gamma":.95,"learning_rate":.1,"lambda":.8,"tilings":4,"bins":8,"memory_size":128}

class CompetitiveLearnerTests(unittest.TestCase):
    def test_tile_coder_is_deterministic_and_bounded(self):
        coder=TileCoder(3,tilings=4,bins=8,memory_size=31,seed=7)
        first=coder.encode(np.array([0.,.5,1.],dtype=np.float32))
        np.testing.assert_array_equal(first,coder.encode(np.array([0.,.5,1.],dtype=np.float32)))
        self.assertEqual(first.shape,(4,)); self.assertTrue(np.all(first<31))
    def test_tile_coder_ternary_quantizes_small_direction_signals(self):
        coder=TileCoder(1,tilings=4,bins=8,memory_size=31,seed=7,
                        signed_indices=(0,),ternary_indices=(0,))
        toward=coder.encode(np.array([1./288],dtype=np.float32))
        away=coder.encode(np.array([-1./288],dtype=np.float32))
        np.testing.assert_array_equal(toward,coder.encode(np.array([.9],dtype=np.float32)))
        np.testing.assert_array_equal(away,coder.encode(np.array([-.9],dtype=np.float32)))
        self.assertFalse(np.array_equal(toward,away))
    def test_action_projection_preserves_local_coin_direction_slot(self):
        from all_other_agent_code.expected_sarsa_lambda_agent.callbacks import HYPERPARAMETERS
        learner=TraceControl(126,6,seed=1,hyperparameters=HYPERPARAMETERS,
                             algorithm="expected_sarsa_lambda")
        self.assertEqual(learner.coder.dimensions,41)
        self.assertEqual(learner.coder.ternary_indices.tolist(),[7])
    def test_expected_sarsa_updates_and_clears_terminal_traces(self):
        learner=TraceControl(3,2,seed=1,hyperparameters=HP,algorithm="expected_sarsa_lambda")
        learner.set_epsilon(.2); before=learner.q_values(np.zeros(3)).copy()
        learner.observe(type("T",(),{"state":np.zeros(3),"action":0,"reward":1.,"next_state":None,"done":True,"next_legal":None,"steps":1})())
        self.assertGreater(learner.q_values(np.zeros(3))[0],before[0]); self.assertFalse(learner.traces.any())
    def test_retrospective_penalty_updates_only_credited_action(self):
        learner=TraceControl(3,2,seed=1,hyperparameters=HP,algorithm="expected_sarsa_lambda")
        state=np.zeros(3); before=learner.q_values(state).copy()
        learner.retrospective_penalty(state,1,-25.)
        after=learner.q_values(state)
        self.assertEqual(after[0],before[0]); self.assertLess(after[1],before[1])
    def test_double_q_updates_only_one_estimator(self):
        learner=TraceControl(3,2,seed=1,hyperparameters=HP,algorithm="double_q_lambda")
        learner.observe(type("T",(),{"state":np.zeros(3),"action":0,"reward":1.,"next_state":None,"done":True,"next_legal":None,"steps":1})())
        changed=[bool(table.any()) for table in learner.weights]; self.assertEqual(sum(changed),1)
    def test_learning_rate_decay_reduces_late_trace_updates(self):
        hp={**HP,"learning_rate_final":.02,"learning_rate_decay_steps":1}
        learner=TraceControl(3,2,seed=1,hyperparameters=hp,algorithm="double_q_lambda")
        transition=type("T",(),{"state":np.zeros(3),"action":0,"reward":1.,"next_state":None,"done":True,"next_legal":None,"steps":1})()
        learner.observe(transition)
        first_change=float(np.abs(learner.weights).sum())
        learner.weights.fill(0.); learner.observe(transition)
        late_change=float(np.abs(learner.weights).sum())
        self.assertLess(late_change,first_change)
    def test_dueling_output_and_prioritized_sampling(self):
        self.assertEqual(tuple(DuelingNetwork(3,2)(torch.zeros(4,3)).shape),(4,2))
        replay=PrioritizedReplay(4,3,.6)
        for reward in range(4): replay.append(Transition(np.zeros(3),0,float(reward),None,True,None,np.ones(2,dtype=bool),"x",1))
        batch,indices,weights=replay.sample(3,.4); self.assertEqual(len(batch),3); self.assertEqual(weights.shape,(3,))
    def test_prioritized_replay_preserves_high_value_quota(self):
        replay=PrioritizedReplay(4,3,.6,protected_capacity=1,
                                 protected_reward_threshold=4.)
        replay.append(Transition(np.zeros(3),0,5.,None,True,None,
                                 np.ones(2,dtype=bool),"x",1))
        for reward in range(10):
            replay.append(Transition(np.zeros(3),0,float(reward%2),None,True,None,
                                     np.ones(2,dtype=bool),"x",1))
        self.assertEqual(sum(replay.protected),1)
        self.assertTrue(any(item.reward==5. for item in replay.memory))
    def test_rainbow_learns_and_round_trips(self):
        hp={"gamma":.95,"learning_rate":1e-3,"batch_size":2,"replay_capacity":8,"warmup":2,"target_sync_interval":2,"per_alpha":.6,"per_beta_start":.4,"per_beta_steps":10,"per_epsilon":1e-5}
        model=RainbowLite(3,2,seed=4,hyperparameters=hp)
        transition=Transition(np.zeros(3,dtype=np.float32),0,1.,None,True,None,np.ones(2,dtype=bool),"x",1)
        model.observe(transition); self.assertIsNotNone(model.observe(transition))
        restored=RainbowLite(3,2,seed=9,hyperparameters=hp); restored.load_checkpoint(model.checkpoint(),training=True)
        np.testing.assert_allclose(model.q_values(np.zeros(3)),restored.q_values(np.zeros(3)))
    def test_rainbow_train_interval_reduces_update_frequency(self):
        hp={"gamma":.95,"learning_rate":1e-3,"batch_size":1,
            "replay_capacity":8,"warmup":1,"target_sync_interval":10,
            "per_alpha":.6,"per_beta_start":.4,"per_beta_steps":100,
            "per_epsilon":1e-5,"train_interval":4}
        model=RainbowLite(3,2,seed=4,hyperparameters=hp)
        transition=Transition(np.zeros(3,dtype=np.float32),0,1.,None,True,None,
                              np.ones(2,dtype=bool),"x",1)
        for _ in range(8): model.observe(transition)
        self.assertEqual(model.observations,8)
        self.assertEqual(model.updates,2)
    def test_stable_v6_rebinds_runtime_hyperparameters(self):
        from all_other_agent_code.rainbow_lite_agent import callbacks as base_callbacks
        from all_other_agent_code.rainbow_lite_v6_stable_agent import callbacks as stable
        self.assertIs(base_callbacks.HYPERPARAMETERS,stable.HYPERPARAMETERS)
        self.assertEqual(base_callbacks.HYPERPARAMETERS["train_interval"],4)

if __name__ == "__main__": unittest.main()
