import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import events as e
from agent_code.dqn_agent.model import DQN, ReplayBuffer, Transition
from agent_code.learning_common.effective_learning import resolve, VERSION
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.learning_common.training_spec import resolve_retention_spec
from agent_code.double_dqn_continuous_v2_agent import callbacks, train
from agent_code.team_agent.rewards import reward_from_events
from experiments.task4_exploration_transfer import initialize, RETENTION, EXPLORATION, materialize, contract_for, validate_resume
from experiments.task4_exploration_campaign import cutoff, validate_manifest, checkpoint_key, arm_key, final_key, failures
from experiments.task4_transfer import TARGET_SAFETY
from tests.test_task4_transfer import assert_equal
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'agent_code/double_dqn_continuous_v2_agent/task3_validated.pt'


def config(arm):
    return dict(feature_id='continuous-v2',reward_id='r7_safe_credit_sparse' if arm=='E1' else 'task4-score-v1',
        safety=TARGET_SAFETY,learning=dict(callbacks.HYPERPARAMETERS,gamma=.99 if arm=='E3' else .95,learning_rate=1e-4),
        exploration_contract=dict(version=VERSION,arm=arm,manifest='experiments/task4_exploration_manifest.json'),
        training=dict(n_step=5,retention=RETENTION,exploration=EXPLORATION,safety_replay=dict(enabled=False,parent_samples=48,ordinary_task3_samples=8,own_bomb_cycle_samples=8,fatal_prefix_steps=4)))


def transition(done=False,steps=1):
    return Transition(np.ones(84,dtype=np.float32),0,1.,None if done else np.zeros(84,dtype=np.float32),done,None if done else np.ones(6,dtype=bool),np.ones(6,dtype=bool),'full_match',steps)


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=torch.load(PARENT,map_location='cpu',weights_only=True)

    def child(self,arm='E3',seed=22):
        return initialize(self.parent,config(arm),seed,dict(target_stage_action_steps=20000,min_rounds=1),dict(version=VERSION,arm=arm))

    def model(self,child):
        m=DQN(84,6,seed=child['agent_seed'],gamma=child['hyperparameters']['gamma'],learning_rate=1e-4,
              training_task='full_match',retention_spec=RETENTION,double_dqn=True,hidden_size=128)
        m.load_checkpoint(child,training=True,training_task='full_match');return m

    def test_policy_only_transfer_exact_and_rng(self):
        a=self.child();b=self.child();c=self.child(seed=11)
        assert_equal(self,a,b)
        for key in ('policy','target'):assert_equal(self,a[key],self.parent['policy'])
        self.assertEqual(a['optimizer']['state'],{});self.assertEqual(a['optimizer']['param_groups'][0]['lr'],1e-4)
        for key in ('updates','action_steps','total_action_steps','stage_action_steps'):self.assertEqual(a[key],0)
        self.assertEqual(a['replay']['count'],0);self.assertEqual(a['replay']['current_task'],'full_match')
        self.assertIsNone(a['teacher']);self.assertIsNone(a['distillation_dataset'])
        self.assertFalse(a['n_step_state']['pending'])
        for key in ('agent_rng_state','distillation_rng_state'):self.assertNotEqual(a[key],c[key])
        self.assertFalse(torch.equal(a['torch_rng_state'],c['torch_rng_state']))
        self.assertNotEqual(a['replay']['rng_state'],c['replay']['rng_state'])

    def test_current_task_only_warmup_and_foreign_rejection(self):
        m=self.model(self.child())
        for _ in range(1999):self.assertIsNone(m.observe(transition()))
        self.assertEqual(m.updates,0);self.assertIsNotNone(m.observe(transition()))
        batch=m.replay.sample_batch(64,sampling_version='task4-only-v1')
        self.assertEqual(len(batch['states']),64);self.assertFalse(batch['is_parent'].any())
        with self.assertRaises(ValueError):m.observe(transition()._replace(task_id='weak_opponents'))
        bad=self.child();bad['replay']['task_ids']=['weak_opponents']
        with self.assertRaises(ValueError):self.model(bad)
        bad=self.child();bad['teacher']=bad['policy']
        with self.assertRaises(ValueError):self.model(bad)
        bad=dict(RETENTION,sampling_version='wrong')
        with self.assertRaises(ValueError):resolve_retention_spec(bad)
        self.assertEqual(resolve_retention_spec(RETENTION),RETENTION)

    def test_effective_parameters_and_legacy_isolation(self):
        self.assertEqual(resolve(callbacks.HYPERPARAMETERS),callbacks.HYPERPARAMETERS)
        for arm,gamma in [('E1',.95),('E2',.95),('E3',.99)]:
            c=config(arm);p=self.child(arm)
            self.assertEqual(resolve(callbacks.HYPERPARAMETERS,c,p)['gamma'],gamma)
            owner=SimpleNamespace(n_step=5,effective_hyperparameters=p['hyperparameters'])
            train.setup_training(owner);self.assertEqual(owner.accumulator.gamma,gamma)
            bad=copy.deepcopy(p);bad['hyperparameters']['gamma']=.9
            with self.assertRaises(ValueError):resolve(callbacks.HYPERPARAMETERS,c,bad)
            with self.assertRaises(ValueError):resolve(callbacks.HYPERPARAMETERS,c,self.parent)

    def test_analytic_nstep_and_bootstrap(self):
        for gamma in (.95,.99):
            acc=NStepAccumulator(5,gamma);out=[]
            for _ in range(5):out+=acc.append(transition())
            self.assertAlmostEqual(out[0].reward,sum(gamma**k for k in range(5)))
            self.assertEqual(out[0].steps,5)
            m=self.model(self.child('E3' if gamma==.99 else 'E2'))
            for net in (m.policy,m.target):
                for param in net.parameters():param.data.zero_()
            m.target.layers[-1].bias.data.fill_(2.)
            for _ in range(64):m.replay.append(out[0])
            expected=out[0].reward+gamma**5*2
            captured=[]
            original=torch.nn.functional.smooth_l1_loss
            def loss(q,target,*args,**kwargs):
                captured.append(target.detach().clone());return original(q,target,*args,**kwargs)
            with patch('torch.nn.functional.smooth_l1_loss',side_effect=loss):m._learn()
            self.assertTrue(captured);self.assertTrue(torch.allclose(captured[0],torch.full_like(captured[0],expected)))

    def test_round_boundary_learning_restore(self):
        m=self.model(self.child())
        for _ in range(2005):m.observe(transition(done=True))
        checkpoint=dict(self.child(),**m.checkpoint())
        clone=self.model(copy.deepcopy(checkpoint))
        for _ in range(5):
            a=m.observe(transition(done=True));b=clone.observe(transition(done=True));self.assertEqual(a,b)
        assert_equal(self,m.policy.state_dict(),clone.policy.state_dict())
        assert_equal(self,m.optimizer.state_dict(),clone.optimizer.state_dict())
        self.assertEqual(m.replay.random.getstate(),clone.replay.random.getstate())

    def test_pure_score_ignores_every_extra_component(self):
        events=[e.COIN_COLLECTED]*2+[e.KILLED_OPPONENT]*3+[e.KILLED_SELF,e.GOT_KILLED,e.CRATE_DESTROYED,e.INVALID_ACTION]
        self.assertEqual(reward_from_events(events,'task4-score-v1',terminal=True,temporal_adjustment=100,avoidable_fatal=True),17.)
        self.assertEqual(reward_from_events([e.KILLED_SELF],'task4-score-v1'),0.)

    def test_reject_parent_state_and_manifest_identity(self):
        for key,value in [('training_task','full_match'),('lifecycle_version','old'),('n_step_state',{'pending':[1]})]:
            p=dict(self.parent);p[key]=value
            with self.assertRaises(ValueError):initialize(p,config('E1'),22,{}, {})
        if not (ROOT/'experiments/task4_exploration_manifest.json').exists():self.skipTest('registration still running')
        path=ROOT/'experiments/configs/task4_exploration_E1.json';c=contract_for(ROOT,path,22)
        validate_resume(c,root=ROOT,config_path=path,seed=22)
        for key in ('arm','source_hash','config_sha256','manifest_sha256','reward_id','hyperparameters'):
            with self.assertRaises(ValueError):validate_resume(dict(c,**{key:'bad'}),root=ROOT,config_path=path,seed=22)
        with tempfile.TemporaryDirectory() as tmp:
            wrong=Path(tmp)/'wrong.pt';wrong.write_bytes(b'not the parent')
            with self.assertRaises(ValueError):materialize(wrong,Path(tmp)/'out.pt',root=ROOT,config_path=path,seed=22,training_budget={})

    def test_budget_selection_and_unused_data(self):
        for kind,hour in [('arm',18),('selection',20),('evaluation',23),('work',24)]:
            cutoff(0,kind=kind,now=hour*3600-1)
            with self.assertRaises(TimeoutError):cutoff(0,kind=kind,now=hour*3600)
        self.assertEqual(failures({'failures':['zero_bomb_round_rate','invalid_action_rate']}),['invalid_action_rate'])
        path=ROOT/'experiments/task4_exploration_manifest.json'
        if path.exists():
            m=json.loads(path.read_text());validate_manifest(m)
            bad=copy.deepcopy(m);bad['excluded_seeds']+=m['final_seeds']
            with self.assertRaises(ValueError):validate_manifest(bad)

if __name__=='__main__':unittest.main()

class ControllerTests(unittest.TestCase):
    def test_behavior_eliminates_checkpoint_but_engineering_stops(self):
        from experiments.task4_exploration_campaign import Campaign, EngineeringFailure
        obj=Campaign.__new__(Campaign)
        with tempfile.TemporaryDirectory() as tmp:
            for bad,raises in [('invalid_action_rate',False),('unexplained_self_death',True),('task4_act_p95',True)]:
                with patch('experiments.task4_exploration_campaign.audit_run',return_value={'failures':[bad]}):
                    if raises:
                        with self.assertRaises(EngineeringFailure):obj.audit(Path(tmp))
                    else:self.assertEqual(obj.audit(Path(tmp))['failures'],[bad])
            with patch('experiments.task4_exploration_campaign.audit_run',return_value={'failures':['invalid_action_rate']}):
                with self.assertRaises(EngineeringFailure):obj.audit(Path(tmp),diagnostic=True)

    def test_first_final_failure_stops_without_replacement(self):
        from experiments.task4_exploration_campaign import Campaign, EngineeringFailure
        obj=Campaign.__new__(Campaign)
        obj.state={'final_candidate':{'arm':'E1','checkpoint':'weight','checkpoint_sha256':'sha'}}
        obj.m={'parent':{'checkpoint':'parent'},'final_seeds':list(range(200))}
        obj.directory=Path('unused');calls=[]
        obj.report=lambda:None;obj.archive_terminal=lambda:None
        def update(status,**kwargs):obj.state.update(status=status,**kwargs)
        obj.update=update
        def evaluate(*args):calls.append(args);raise EngineeringFailure('stop now')
        obj.evaluate=evaluate
        with patch('experiments.task4_exploration_campaign.sha256',return_value='sha'):obj.run()
        self.assertEqual(len(calls),1);self.assertEqual(obj.state['status'],'engineering_failure')

    def test_arm_and_checkpoint_ranking_score_then_first(self):
        def point(score,first,actions,seed=22):
            return dict(summary=dict(mean_score=score,first_place_rate=first,suicide_rate=0,act_p95_seconds=.02),actual_actions=actions,seed=seed)
        a,b=point(5,.4,40000),point(4,.9,20000)
        self.assertLess(checkpoint_key(a),checkpoint_key(b))
        self.assertLess(arm_key([a,a],'E1'),arm_key([b,b],'E2'))
        self.assertLess(final_key(point(5,.4,20000,11)),final_key(a))

    def test_role_and_timing_boundaries_keep_legacy_separate(self):
        from experiments.compact_audit import policy_gates
        from experiments.task4_protocol import CANDIDATE, LEGACY
        from tests.test_task4_campaign import summary
        s=summary();s.update(act_p95_seconds=.1,act_max_seconds=.3)
        self.assertFalse(policy_gates(4,s,CANDIDATE))
        for field,value in [('act_p95_seconds',.100001),('act_max_seconds',.300001)]:
            bad=dict(s,**{field:value});self.assertTrue(policy_gates(4,bad,CANDIDATE))
            self.assertFalse(policy_gates(4,bad,LEGACY))
