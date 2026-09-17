import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.task4_transfer import materialize, validate_payload, validate_resume
from agent_code.double_dqn_continuous_v2_agent import callbacks, train
from agent_code.dqn_agent.model import DQN

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'agent_code/double_dqn_continuous_v2_agent/task3_validated.pt'


def assert_equal(case,a,b):
    if isinstance(a,torch.Tensor):case.assertTrue(torch.equal(a,b));return
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b);return
    if isinstance(a,dict):
        case.assertEqual(set(a),set(b))
        for k in a:assert_equal(case,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        case.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_equal(case,x,y)
    else:case.assertEqual(a,b)


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=torch.load(PARENT,weights_only=True,map_location='cpu')
        cls.config=json.loads((ROOT/'experiments/configs/task4_A.json').read_text())

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def migrated(self,seed=22,arm='A'):
        path=Path(self.temp.name)/f'{seed}{arm}.pt'
        contract=materialize(PARENT,path,root=ROOT,config_path=ROOT/f'experiments/configs/task4_{arm}.json',
                            seed=seed,training_budget={'target_stage_action_steps':None,'min_rounds':1})
        return path,torch.load(path,weights_only=True,map_location='cpu'),contract

    def test_transfer_preserves_learning_state_and_replaces_teacher(self):
        _,p,c=self.migrated()
        for k in ('policy','target','optimizer','updates','total_action_steps'):
            assert_equal(self,self.parent[k],p[k])
        for k in self.parent['replay']:
            if k not in ('rng_state','current_task'):assert_equal(self,self.parent['replay'][k],p['replay'][k])
        assert_equal(self,p['teacher'],p['policy'])
        self.assertEqual(p['training_task'],'full_match');self.assertEqual(p['stage_action_steps'],0)
        self.assertEqual(p['replay']['current_task'],'full_match')
        self.assertNotIn('full_match',p['replay']['task_ids'])
        self.assertEqual(p['replay']['rng_state'],random.Random(22).getstate())
        path,_,_=self.migrated()
        env={'BOMBERMAN_CHECKPOINT':str(path),'BOMBERMAN_TRAINING_TASK':'full_match',
             'BOMBERMAN_AGENT_SEED':'22','BOMBERMAN_REWARD_ID':self.config['reward_id'],
             'BOMBERMAN_N_STEP':'5','BOMBERMAN_SAFETY_SPEC':json.dumps(self.config['safety']),
             'BOMBERMAN_RETENTION_SPEC':json.dumps(self.config['training']['retention'])}
        with patch.dict(os.environ,env):
            owner=SimpleNamespace(train=True);callbacks.setup(owner)
        self.assertTrue(all(not x.requires_grad for x in owner.model.teacher.parameters()))
        self.assertEqual(owner.model.replay.current_task,'full_match')

    def test_same_seed_is_exact_and_distinct_seeds_reset_all_rngs(self):
        _,a,_=self.migrated(11);_,b,_=self.migrated(11);_,c,_=self.migrated(33)
        assert_equal(self,a,b)
        for key in ('agent_rng_state','distillation_rng_state'):self.assertNotEqual(a[key],c[key])
        self.assertNotEqual(a['replay']['rng_state'],c['replay']['rng_state'])
        self.assertFalse(torch.equal(a['torch_rng_state'],c['torch_rng_state']))

    def test_reject_invalid_parent_and_resume_identity(self):
        for key,value in [('training_task','full_match'),('lifecycle_version','old'),('n_step',4)]:
            p=dict(self.parent);p[key]=value
            with self.assertRaises(ValueError):validate_payload(p,self.config)
        for key,value in [('n_step_state',{'pending':[1]}),('action_history_state',{'own_bomb_pending':True}),
                          ('own_bomb_escape_state',{'placement_certificate':{'x':1}})]:
            p=dict(self.parent);p[key]=value
            with self.assertRaises(ValueError):validate_payload(p,self.config)
        bad=Path(self.temp.name)/'bad.pt';bad.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'SHA'):
            materialize(bad,bad.with_suffix('.new'),root=ROOT,config_path=ROOT/'experiments/configs/task4_A.json',seed=22,training_budget={})
        _,_,contract=self.migrated()
        for key in ('arm','manifest_sha256','opponent_hashes','config_sha256'):
            bad={**contract,key:'wrong'}
            with self.assertRaises(ValueError):validate_resume(bad,root=ROOT,config_path=ROOT/'experiments/configs/task4_A.json',seed=22)

    def test_warmup_and_a_b_sampling_counts(self):
        # Use real transferred replay and real current-task transitions.
        for arm,parents in [('A',48),('B',32)]:
            _,p,_=self.migrated(22,arm)
            model=DQN(84,6,seed=22,batch_size=64,hidden_size=128,double_dqn=True,
                      training_task='full_match',retention_spec=p['retention_spec'])
            model.load_checkpoint(p,training=True,training_task='full_match')
            template=next(iter(model.replay.partitions['weak_opponents']))
            item=template._replace(task_id='full_match')
            for _ in range(1998):model.replay.append(item)
            before=model.updates;self.assertIsNone(model.observe(item));self.assertEqual(model.updates,before)
            model.replay.append(item)
            sample=model.replay.sample_batch(64,p['retention_spec']['parent_fraction'])
            self.assertEqual(int(sample['is_parent'].sum()),parents)

    def test_learning_after_warmup_restores_sampling_weights_and_rng(self):
        _,payload,_=self.migrated()
        def learner(checkpoint):
            model=DQN(84,6,seed=22,batch_size=64,hidden_size=128,double_dqn=True,
                      training_task='full_match',retention_spec=payload['retention_spec'])
            model.load_checkpoint(checkpoint,training=True,training_task='full_match')
            return model
        continuous=learner(payload)
        item=next(iter(continuous.replay.partitions['weak_opponents']))._replace(task_id='full_match')
        for _ in range(2000):continuous.replay.append(item)
        for _ in range(2):continuous.observe(item)
        saved=copy.deepcopy(continuous.checkpoint());saved['training_task']='full_match'
        for _ in range(3):continuous.observe(item)
        expected=copy.deepcopy(continuous.checkpoint())
        resumed=learner(saved)
        for _ in range(3):resumed.observe(item)
        assert_equal(self,expected,resumed.checkpoint())

    def test_actual_runner_round_boundary_resume_matches_uninterrupted(self):
        from experiments.run import main
        import settings
        d=Path(self.temp.name)
        common=['--config',str(ROOT/'experiments/configs/task4_A.json'),'--mode','train',
                '--device','cpu','--task','4','--agent','double_dqn_continuous_v2_agent',
                '--seed','22','--replay-policy','none']
        with patch.object(settings,'MAX_STEPS',3), patch('experiments.run.RUNS_ROOT',d):
            self.assertEqual(main(common+['--n-rounds','4','--output',str(d/'whole'),
                '--transfer-task4-from-checkpoint',str(PARENT)]),0)
            self.assertEqual(main(common+['--n-rounds','2','--output',str(d/'first'),
                '--transfer-task4-from-checkpoint',str(PARENT)]),0)
            self.assertEqual(main(common+['--n-rounds','2','--output',str(d/'second'),
                '--resume-from',str(d/'first')]),0)
        a=torch.load(d/'whole/checkpoints/final.pt',weights_only=True,map_location='cpu')
        b=torch.load(d/'second/checkpoints/final.pt',weights_only=True,map_location='cpu')
        assert_equal(self,a,b)
        def actions(path):
            return [(r['round_index'],r['step'],r['agent_name'],r['action'])
                    for r in map(json.loads,path.read_text().splitlines())]
        self.assertEqual(actions(d/'whole/timing.jsonl'),actions(d/'first/timing.jsonl')+actions(d/'second/timing.jsonl'))
