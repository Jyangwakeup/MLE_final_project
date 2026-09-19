"""Contracts, real frozen policies, scheduler, and terminal controller checks."""
import copy
import itertools
import json
import logging
import tempfile
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.frozen_opponents import assignment, stream_seed, rng_state, Delegate, template, policy_digest, Manager
from experiments.task4_frozen_transfer import contract_for, validate_resume, materialize, VERSION
from experiments.task4_exploration_transfer import initialize
from experiments.task4_frozen_campaign import Campaign, cutoff, validate_manifest, EngineeringFailure
from agent_code.double_dqn_continuous_v2_agent import callbacks
from agent_code.learning_common.effective_learning import resolve
from agent_code.learning_common.action_history import action_history_state
from tests.test_task3_lifecycle import state
from tests.test_task4_transfer import assert_equal
ROOT=Path(__file__).resolve().parents[1]


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ROOT/'experiments/task4_frozen_manifest.json').read_text())
        cls.config_path=ROOT/cls.manifest['arm_configs']['S']
        cls.config=json.loads(cls.config_path.read_text())
        cls.parent=torch.load(ROOT/cls.manifest['parent']['checkpoint'],map_location='cpu',weights_only=True)

    def test_ten_round_mixture_and_full_balanced_cycle(self):
        rosters=[assignment('S',i,17) for i in range(1,481)]
        for i in range(0,len(rosters),10):self.assertEqual(sum(r==('rule',)*3 for r in rosters[i:i+10]),5)
        expected=set(itertools.permutations(self.manifest['pool'],3))
        history=[r for r in rosters if r!=('rule',)*3]
        for i in range(0,len(history),24):self.assertEqual(set(history[i:i+24]),expected)
        self.assertEqual(assignment('C',99,17),('rule',)*3)
        self.assertEqual(rosters,[assignment('S',i,17) for i in range(1,481)])
        self.assertNotEqual(rosters,[assignment('S',i,18) for i in range(1,481)])
        with self.assertRaises(ValueError):assignment('S',0,17)

    def test_isolation_and_original_frozen_decisions(self):
        logger=logging.getLogger('test-frozen')
        entry=self.manifest['models']['e1_22_40k']
        random.seed(83);np.random.seed(84);torch.manual_seed(85);before=rng_state()
        delegate=Delegate('neural',entry,123,logger)
        assert_equal(self,rng_state(),before)
        direct=copy.copy(template(entry));direct.rng=random.Random(123)
        from agent_code.learning_common.action_history import init_action_history
        init_action_history(direct)
        sample=state();action=delegate.act(copy.deepcopy(sample))
        self.assertEqual(action,callbacks.act(direct,copy.deepcopy(sample)))
        assert_equal(self,delegate.owner.last_safety_diagnostic,direct.last_safety_diagnostic)
        assert_equal(self,rng_state(),before)
        other=Delegate('neural',entry,124,logger)
        self.assertIsNot(delegate.owner.feature_position_history,other.owner.feature_position_history)
        self.assertNotEqual(delegate.owner.rng.getstate(),other.owner.rng.getstate())
        self.assertEqual(other.owner.feature_position_history,[])
        rule=Delegate('rule',None,125,logger);assert_equal(self,rng_state(),before)
        a=rule.act(copy.deepcopy(sample));assert_equal(self,rng_state(),before)
        clone=Delegate('rule',None,125,logger);self.assertEqual(a,clone.act(copy.deepcopy(sample)))
        assert_equal(self,rng_state(),before)
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(delegate.owner.model.checkpoint(),Path(tmp)/'state.pt')
            assert_equal(self,rng_state(),before)

    def test_all_models_registered_distinct_readonly(self):
        digests=[]
        for key,entry in self.manifest['models'].items():
            owner=template(entry);digests.append(policy_digest(owner.model))
            self.assertFalse(owner.train);self.assertEqual(len(owner.model.replay),0)
            self.assertIsNone(owner.model.teacher);self.assertFalse(owner.model.optimizer.state)
            self.assertTrue(all(not p.requires_grad for p in owner.model.policy.parameters()))
            self.assertEqual(owner.effective_hyperparameters,entry['hyperparameters'])
        self.assertEqual(len(set(digests)),7)
        wrong=dict(self.manifest['models']['parent'],sha256='wrong')
        with self.assertRaises(ValueError):template(wrong)

    def test_new_transfer_exact_and_legacy_rejected(self):
        contract=contract_for(ROOT,self.config_path,22)
        a=initialize(self.parent,self.config,22,{},contract)
        b=initialize(self.parent,self.config,22,{},contract)
        assert_equal(self,a,b)
        for key in ('policy','target'):assert_equal(self,a[key],self.parent['policy'])
        self.assertEqual(a['replay']['count'],0);self.assertEqual(a['optimizer']['state'],{})
        self.assertIsNone(a['teacher']);self.assertIsNone(a['distillation_dataset'])
        for key in ('updates','action_steps','total_action_steps','stage_action_steps'):self.assertEqual(a[key],0)
        self.assertEqual(resolve(callbacks.HYPERPARAMETERS,self.config,a),self.config['learning'])
        with self.assertRaises(ValueError):resolve(callbacks.HYPERPARAMETERS,self.config,self.parent)
        validate_resume(contract,root=ROOT,config_path=self.config_path,seed=22)
        for key in ('version','arm','source_hash','source_commit','config_sha256','manifest_sha256','pool_sha256','schedule_version','schedule_seed','opponent_rng_root','target_safety'):
            bad=dict(contract,**{key:'wrong'})
            with self.assertRaises(ValueError):validate_resume(bad,root=ROOT,config_path=self.config_path,seed=22)
        with tempfile.TemporaryDirectory() as tmp:
            wrong=Path(tmp)/'wrong.pt';wrong.write_bytes(b'wrong')
            with self.assertRaises(ValueError):materialize(wrong,Path(tmp)/'out.pt',root=ROOT,config_path=self.config_path,seed=22,training_budget={})
        with self.assertRaises(ValueError):contract_for(ROOT,ROOT/self.manifest['reference_config'],22)

    def test_data_budget_and_selection_contract(self):
        validate_manifest(self.manifest)
        for key in ('pool_sha256','schedule_version','schema_version'):
            bad=dict(self.manifest,**{key:'wrong'})
            with self.assertRaises(ValueError):validate_manifest(bad)
        bad=copy.deepcopy(self.manifest);bad['excluded_seeds']+=bad['final_seeds']
        with self.assertRaises(ValueError):validate_manifest(bad)
        for kind,hour in [('arm',18),('selection',20),('evaluation',23),('work',24)]:
            cutoff(0,kind=kind,now=hour*3600-1)
            with self.assertRaises(TimeoutError):cutoff(0,kind=kind,now=hour*3600)

    def test_first_engineering_failure_is_terminal(self):
        obj=Campaign.__new__(Campaign)
        obj.state={'final_candidate':dict(checkpoint='S',checkpoint_sha256='sha',seed=22),
                   'matching_control':dict(checkpoint='C',checkpoint_sha256='sha')}
        obj.m={'parent':{'checkpoint':'parent'},'final_seeds':[1]}
        obj.report=lambda:None;obj.archive_terminal=lambda:None
        obj.update=lambda status,**values:obj.state.update(status=status,**values)
        calls=[]
        def fail(*args,**kwargs):calls.append(args);raise EngineeringFailure('first failure')
        obj.evaluate=fail
        with patch('experiments.task4_frozen_campaign.sha256',return_value='sha'):obj.run()
        self.assertEqual(len(calls),1);self.assertEqual(obj.state['status'],'engineering_failure')

    def test_behavior_and_opponent_failures_stop_at_endpoint(self):
        obj=Campaign.__new__(Campaign)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('experiments.task4_frozen_campaign.audit_run',return_value={'failures':['invalid_action_rate']}),patch('experiments.task4_frozen_audit.audit_opponents',return_value={'failures':[]}):
                self.assertEqual(obj.audit(tmp)['failures'],['invalid_action_rate'])
                with self.assertRaises(EngineeringFailure):obj.audit(tmp,diagnostic=True)
            with patch('experiments.task4_frozen_campaign.audit_run',return_value={'failures':[]}),patch('experiments.task4_frozen_audit.audit_opponents',return_value={'failures':['frozen:responsibility_clock']}):
                with self.assertRaises(EngineeringFailure):obj.audit(tmp)

if __name__=='__main__':unittest.main()
