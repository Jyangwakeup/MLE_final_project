import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent_code.dqn_agent.model import ReplayBuffer, Transition
from agent_code.learning_common.training_spec import resolve_retention_spec
from agent_code.team_agent.rewards import resolve_reward_spec
from experiments import task3_lifecycle_campaign as campaign
from experiments.resume import materialize_task3_safety_checkpoint


def results(retention=0, combat=0, safety=0, qualified=False):
    gates={}
    for i in range(retention): gates[f'task{i}_coins_retention']={'passed':False}
    if combat: gates['task3_combat_gain']={'passed':False}
    if safety: gates['task3_suicide_rate']={'passed':False}
    h=dict(failed_gate_count=len(gates),gate_checks=gates,task3_score=6.,task3_kills=.4,
           task3_first_place_rate=.5,task3_suicide_rate=0.,task3_bomb_survival_rate=1.,
           minimum_retention=1.,act_p95_seconds=.1,checkpoint_run='run',cumulative_round=50)
    return {str(s):dict(status='qualified' if qualified else 'gate_failed',history=[copy.deepcopy(h)],
                       decision={'selected_checkpoint_run':'run' if qualified else None}) for s in (11,22,33)}


class CampaignTests(unittest.TestCase):
    def test_bounded_development_order_and_combination_requires_independent_benefit(self):
        self.assertEqual(campaign.next_arm({}),'A')
        a=results(retention=1,combat=1)
        self.assertEqual(campaign.next_arm({'A':a}),'P')
        self.assertEqual(campaign.next_arm({'A':a,'P':results(combat=1)}),'R')
        self.assertEqual(campaign.next_arm({'A':a,'P':results(combat=1),'R':results(retention=1)}),'PR')
        self.assertIsNone(campaign.next_arm({'A':a,'P':a,'R':results(retention=1)}))
        self.assertIsNone(campaign.next_arm({'A':results(safety=1)}))

    def test_confirmation_failure_never_calls_main_or_another_arm(self):
        c=campaign.Campaign.__new__(campaign.Campaign)
        c.state=dict(status='created',arms={},diagnostics={})
        c.started=0;c.path=Path('/unused');c.manifest={'confirmation_seeds':[1],'main_validation_seeds':[2]}
        c.write=lambda *x:None;c.report=lambda:None;c.check_time=lambda **k:None;c.diagnostics=lambda:True
        with patch.object(c,'seed',side_effect=lambda arm,seed:results(qualified=True)[str(seed)]) as seed:
            with patch.object(c,'stage',return_value=(False,{},'confirmation.json')) as stage:
                result=c.run()
        self.assertEqual(result['status'],'stopped_confirmation_failure')
        self.assertEqual(stage.call_count,1)
        self.assertEqual({call.args[0] for call in seed.call_args_list},{'A'})

    def test_resume_rejects_manifest_and_source_changes(self):
        with tempfile.TemporaryDirectory() as td:
            c=campaign.Campaign.__new__(campaign.Campaign)
            c.identity={'source_commit':'source','manifest_sha256':'manifest'}
            p=Path(td)/'result.json';c.write(p,{'status':'running'})
            self.assertEqual(c.read(p)['status'],'running')
            c.identity['manifest_sha256']='changed'
            with self.assertRaises(ValueError):c.read(p)

    def test_fixed_replay_quotas_do_not_fallback(self):
        quotas={'coin_navigation':16,'crate_navigation':32,'weak_opponents':16}
        replay=ReplayBuffer(100,11);replay.configure('weak_opponents')
        for task,code,count in [('coin_navigation',1,16),('crate_navigation',2,32),('weak_opponents',3,15)]:
            for i in range(count):
                replay.append(Transition(np.array([code,i]),0,0.,None,True,None,None,task))
        with self.assertRaisesRegex(ValueError,'quotas'):replay.sample_batch(64,task_samples=quotas)
        replay.append(Transition(np.array([3,15]),0,0.,None,True,None,None,'weak_opponents'))
        batch=replay.sample_batch(64,task_samples=quotas)
        self.assertEqual(np.unique(batch['states'][:,0],return_counts=True)[1].tolist(),[16,32,16])
        self.assertEqual(int(batch['is_parent'].sum()),48)

    def test_reward_transfer_only_changes_kill_and_rejects_task3_replay(self):
        old=resolve_reward_spec('r7_safe_credit_sparse');new=resolve_reward_spec('r9_task3_score_aligned')
        self.assertEqual(new,{**old,'killed_opponent':15.})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'parent.pt';dst=root/'child.pt'
            payload={'reward_id':'r7_safe_credit_sparse','policy':{'w':torch.ones(1)},
                     'replay':{'task_ids':['coin_navigation','crate_navigation']},
                     'hyperparameters':{'gamma':.95},'n_step':4}
            torch.save(payload,src)
            (root/'episodes.jsonl').write_text(json.dumps({'agents':[{'kills':0}]})+'\n')
            snapshot=SimpleNamespace(learner_path=src,run_directory=root)
            kw=dict(safety_spec={},exploration_spec={},training_budget={},safety_replay_spec={},n_step=5,
                    reward_id='r9_task3_score_aligned')
            materialize_task3_safety_checkpoint(snapshot,dst,**kw)
            child=torch.load(dst,weights_only=True)
            self.assertEqual(child['reward_spec'],new)
            self.assertEqual(child['lifecycle_version'],'decision-snapshot-v1')
            torch.testing.assert_close(child['teacher']['w'],payload['policy']['w'])
            payload['replay']['task_ids'].append('weak_opponents');torch.save(payload,src)
            with self.assertRaisesRegex(ValueError,'Task 3 replay'):
                materialize_task3_safety_checkpoint(snapshot,dst,**kw)

    def test_cutoffs_apply_before_new_work(self):
        c=campaign.Campaign.__new__(campaign.Campaign);c.started=100;c.deadline=86500
        c.manifest={'new_arm_cutoff_hours':18,'evaluation_cutoff_hours':23}
        with patch.object(campaign.time,'time',return_value=100+18*3600):
            with self.assertRaises(TimeoutError):c.check_time(new_arm=True)
            c.check_time(evaluation=True)
        with patch.object(campaign.time,'time',return_value=100+23*3600):
            with self.assertRaises(TimeoutError):c.check_time(evaluation=True)

if __name__=='__main__':unittest.main()
