import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import events as e
from agent_code.dqn_agent.model import DQN
from agent_code.learning_common.kill_replay import KillReplayBuffer, VERSION as REPLAY
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.learning_common.effective_learning import resolve
from agent_code.double_dqn_continuous_v2_agent import callbacks
from agent_code.team_agent.rewards import resolve_reward_spec,reward_from_events
from experiments.task4_score_transfer import initialize,retention,VERSION,contract_for,validate_resume,validate_archive
from experiments.task4_score_campaign import Campaign,cutoff,validate_manifest,EngineeringFailure
from tests.test_task4_exploration import transition,config as oldconfig
from tests.test_task4_transfer import assert_equal
ROOT=Path(__file__).resolve().parents[1]

def config(arm):
    c=oldconfig('E1');c.pop('exploration_contract');c['learning']['learning_rate']=1e-4 if arm=='C' else 3e-5
    c['score_contract']=dict(version=VERSION,role='learner',arm=arm)
    c['training']['retention']=retention(arm)
    c['reward_id']='r7_safe_credit_potential' if arm in ('LP','LPK') else 'r7_safe_credit_sparse'
    return c

class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.parent=torch.load(ROOT/'agent_code/double_dqn_continuous_v2_agent/task3_validated.pt',weights_only=True,map_location='cpu')
    def child(self,arm='LPK',seed=22):
        return initialize(self.parent,config(arm),seed,dict(target_stage_action_steps=60000,min_rounds=1),dict(version=VERSION,arm=arm))
    def model(self,p):
        m=DQN(84,6,seed=p['agent_seed'],gamma=.95,learning_rate=p['hyperparameters']['learning_rate'],training_task='full_match',retention_spec=p['retention_spec'],double_dqn=True,hidden_size=128)
        m.load_checkpoint(p,training=True,training_task='full_match');return m
    def test_transfer_and_effective_parameters(self):
        for arm in ('C','L','LP','LPK'):
            p=self.child(arm);assert_equal(self,p,self.child(arm))
            for key in ('policy','target'):assert_equal(self,p[key],self.parent['policy'])
            self.assertEqual(p['optimizer']['state'],{});self.assertEqual(p['updates'],0);self.assertEqual(p['replay']['count'],0)
            self.assertEqual(p['optimizer']['param_groups'][0]['lr'],1e-4 if arm=='C' else 3e-5)
            self.assertIsNone(p['teacher']);self.assertIsNone(p['distillation_dataset'])
            self.assertEqual(resolve(callbacks.HYPERPARAMETERS,config(arm),p),p['hyperparameters'])
            bad=copy.deepcopy(p);bad['hyperparameters']['learning_rate']=.5
            with self.assertRaises(ValueError):resolve(callbacks.HYPERPARAMETERS,config(arm),bad)
        self.assertNotEqual(self.child()['agent_rng_state'],self.child(seed=11)['agent_rng_state'])
    def test_labels_terminal_short_tail_and_rewards(self):
        acc=NStepAccumulator(5,.95);out=[]
        for i in range(7):out+=acc.append(transition(done=i==6)._replace(observed_kill=i==4,reward=-15 if i==4 else 1))
        self.assertEqual([t.observed_kill for t in out],[True]*5+[False]*2)
        self.assertAlmostEqual(out[0].reward,sum(.95**j*(1 if j!=4 else -15) for j in range(5)))
        self.assertEqual([t.steps for t in out],[5,5,5,4,3,2,1])
    def test_capacity_quota_fallback_no_duplicates(self):
        for positive in (0,3,100,200):
            r=KillReplayBuffer(20000,22);r.configure('full_match')
            for i in range(200):r.append(transition(done=True)._replace(state=np.full(84,i,dtype=np.float32),observed_kill=i<positive))
            batch=r.sample_batch(64,sampling_version=REPLAY)
            self.assertEqual(len(set(batch['states'][:,0])),64)
            self.assertEqual(int((batch['states'][:,0]<positive).sum()),min(64,max(min(16,positive),64-(200-positive))))
            self.assertFalse(batch['is_parent'].any())
        for i in range(20001):r.append(transition(done=True)._replace(observed_kill=False))
        self.assertEqual(len(r.ordinary),16000);self.assertGreater(r.counters['ordinary_evicted'],0)
        for i in range(4100):r.append(transition(done=True)._replace(observed_kill=True))
        self.assertEqual(len(r),20000);self.assertEqual(len(r.kills),4000)
        state=r.state_dict();clone=KillReplayBuffer(20000,22);clone.configure('full_match');clone.load_state_dict(state)
        assert_equal(self,state,clone.state_dict())
        assert_equal(self,r.sample_batch(64,sampling_version=REPLAY),clone.sample_batch(64,sampling_version=REPLAY))
        with self.assertRaises(ValueError):clone.load_state_dict(dict(state,observed_kills=[]))
        with self.assertRaises(ValueError):r.append(transition()._replace(task_id='weak_opponents'))
    def test_warmup_and_restore_updates(self):
        p=self.child();a=self.model(p)
        for i in range(1999):self.assertIsNone(a.observe(transition(done=True)._replace(observed_kill=i%50==0)))
        a.observe(transition(done=True));self.assertEqual(a.updates,1)
        b=self.model(dict(p,**copy.deepcopy(a.checkpoint())))
        for i in range(8):
            item=transition(done=True)._replace(observed_kill=i%3==0,reward=float(i))
            self.assertEqual(a.observe(item),b.observe(item))
        assert_equal(self,a.checkpoint(),b.checkpoint())
    def test_coin_potential_only_and_terminal(self):
        a=resolve_reward_spec('r7_safe_credit_sparse');b=resolve_reward_spec('r7_safe_credit_potential')
        changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
        self.assertEqual(changed,{'potential_coin_weight'})
        field=np.zeros((7,7),dtype=int);field[0,:]=field[-1,:]=field[:,0]=field[:,-1]=-1
        old=dict(field=field,coins=[(3,3)],bombs=[],explosion_map=np.zeros_like(field),self=('me',0,True,(2,3)),others=[],step=1,round=1)
        # Terminal new state is ignored and the potential at the old state is removed.
        events=[e.COIN_COLLECTED]*2+[e.KILLED_OPPONENT]*2+[e.GOT_KILLED]
        x=reward_from_events(events,'r7_safe_credit_potential',old_game_state=old,terminal=True)
        y=reward_from_events(events,'r7_safe_credit_sparse',old_game_state=old,terminal=True)
        self.assertAlmostEqual(x-y,-.5*np.exp(-1/4))
    def test_real_callback_kill_label_terminal_dedup(self):
        from tests.test_task3_lifecycle import LifecycleTests,state
        from agent_code.double_dqn_continuous_v2_agent import train
        fixture=LifecycleTests();fixture.setUp()
        try:
            owner=fixture.owner;old=state();new=state(2,False,[((3,3),3)])
            action=callbacks.act(owner,old)
            events=[e.KILLED_OPPONENT,e.KILLED_OPPONENT,e.GOT_KILLED]
            train.game_events_occurred(owner,old,action,new,events)
            self.assertTrue(owner.pending[1].observed_kill)
            train.end_of_round(owner,old,action,events)
            items=owner.model.replay._all_items()
            self.assertEqual(len(items),1)
            self.assertTrue(items[0].observed_kill);self.assertTrue(items[0].done)
            self.assertFalse(owner.accumulator.pending);self.assertIsNone(owner.pending)
        finally:fixture.doCleanups()

    def test_manifest_resume_archive_and_budget(self):
        path=ROOT/'experiments/task4_score_manifest.json'
        if not path.exists():self.skipTest('Registration in progress')
        m=json.loads(path.read_text());validate_manifest(m)
        cpath=ROOT/m['arm_configs']['LPK'];c=contract_for(ROOT,cpath,22)
        validate_resume(c,root=ROOT,config_path=cpath,seed=22)
        for k in ('arm','source_hash','manifest_sha256','config_sha256','opponent_rng_root','kill_replay'):
            with self.assertRaises(ValueError):validate_resume(dict(c,**{k:'wrong'}),root=ROOT,config_path=cpath,seed=22)
        entry=m['archived_controls']['22'][0];payload=torch.load(entry['checkpoint'],weights_only=True,map_location='cpu')
        validate_archive(payload,entry['checkpoint'],m)
        with self.assertRaises(ValueError):validate_resume(payload['transfer_contract'],root=ROOT,config_path=cpath,seed=22)
        for kind,h in [('arm',18),('selection',20),('evaluation',23),('work',24)]:
            cutoff(0,kind=kind,now=h*3600-1)
            with self.assertRaises(TimeoutError):cutoff(0,kind=kind,now=h*3600)
    def test_point_accepts_existing_arm_and_seed(self):
        obj=Campaign.__new__(Campaign)
        obj.m={'development_seeds':[27000]};obj.state={'parent_development':{}}
        obj.update=lambda *a,**k:None
        evaluation={'summary':{'mean_score':4},'raw':{'failures':[]}}
        obj.evaluate=lambda *a,**k:evaluation
        obj.compare=lambda *a,**k:{}
        obj.navigation=lambda *a:{};obj.q_drift=lambda *a:{}
        for role in ('archive','candidate'):
            entry={'arm':'C','seed':22,'actual_actions':20001,'checkpoint':'frozen.pt'}
            result=obj.point('C',22,entry,role)
            self.assertEqual(result['arm'],'C');self.assertEqual(result['seed'],22)
            self.assertEqual(result['evaluation'],evaluation)
            self.assertNotIn('evaluation',entry)

    def test_each_checkpoint_evaluated_before_continuation(self):
        obj=Campaign.__new__(Campaign);obj.state={'arms':{}};obj.start=10**12;calls=[]
        obj.update=lambda status,**kw:None
        def train(arm,seed,target,previous=None):
            calls.append(('train',target,previous))
            return dict(checkpoint=str(target),checkpoint_sha256='sha',actual_actions=target,target_actions=target,run='run'+str(target))
        def point(arm,seed,entry):
            calls.append(('evaluate',entry['target_actions']))
            return dict(entry,failures=[],summary=dict(mean_score=1,first_place_rate=.5))
        obj.train=train;obj.point=point
        result=obj.seed('L',22)
        self.assertTrue(result['complete'])
        self.assertEqual(calls,[('train',20000,None),('evaluate',20000),('train',40000,'run20000'),('evaluate',40000),('train',60000,'run40000'),('evaluate',60000)])
        obj.state={'arms':{}};calls.clear()
        obj.point=lambda *args:(_ for _ in ()).throw(EngineeringFailure('checkpoint failure'))
        with self.assertRaises(EngineeringFailure):obj.seed('L',22)
        self.assertEqual(calls,[('train',20000,None)])

    def test_behavior_gate_and_first_failure(self):
        obj=Campaign.__new__(Campaign)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('experiments.task4_score_campaign.audit_run',return_value={'failures':['invalid_action_rate']}),patch('experiments.task4_frozen_audit.audit_opponents',return_value={'failures':[]}):
                self.assertEqual(obj.audit(Path(tmp))['failures'],['invalid_action_rate'])
                with self.assertRaises(EngineeringFailure):obj.audit(Path(tmp),True)
        obj.state=dict(final_candidate={'checkpoint':'a','checkpoint_sha256':'sha'},matching_control={'checkpoint':'b','checkpoint_sha256':'sha'})
        obj.m={'parent':{'checkpoint':'p'},'final_seeds':[]};obj.report=lambda:None;obj.archive_terminal=lambda:None
        obj.update=lambda status,**kw:obj.state.update(status=status,**kw)
        with patch('experiments.task4_score_campaign.sha256',return_value='sha'):
            obj.evaluate=lambda *args:(_ for _ in ()).throw(EngineeringFailure('first failure'))
            obj.run()
        self.assertEqual(obj.state['status'],'engineering_failure')
