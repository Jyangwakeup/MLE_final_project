import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from experiments import task4_frozen_campaign as campaign
from experiments.task4_frozen_transfer import contract_for
from experiments.task4_exploration_transfer import initialize
from tests.test_task4_transfer import assert_equal
import torch
ROOT=Path(__file__).resolve().parents[1]


class FrozenControllerTests(unittest.TestCase):
    def setUp(self):
        self.m=json.loads((ROOT/'experiments/task4_frozen_manifest.json').read_text())

    def test_only_training_opponents_differ_and_initial_learning_identical(self):
        configs=[json.loads((ROOT/self.m['arm_configs'][a]).read_text()) for a in ('C','S')]
        a,b=copy.deepcopy(configs)
        a['frozen_opponents'].pop('arm');b['frozen_opponents'].pop('arm')
        self.assertEqual(a,b)
        parent=torch.load(ROOT/self.m['parent']['checkpoint'],map_location='cpu',weights_only=True)
        children=[]
        for arm,config in zip(('C','S'),configs):
            contract=contract_for(ROOT,ROOT/self.m['arm_configs'][arm],22)
            child=initialize(parent,config,22,{},contract);child.pop('transfer_contract');children.append(child)
        assert_equal(self,*children)

    def test_cache_refuses_changed_role_and_artifacts(self):
        obj=campaign.Campaign.__new__(campaign.Campaign)
        obj.m=self.m;obj.identity={'source':'fixed'};obj.start=10**12;obj.check_identity=lambda:None
        with tempfile.TemporaryDirectory() as tmp:
            obj.directory=Path(tmp)
            cache=obj.directory/'evaluations'/'example.json'
            campaign.write(cache,{'request':{'role':'reference'}})
            with self.assertRaisesRegex(ValueError,'cache identity'):
                obj.evaluate(ROOT/self.m['parent']['checkpoint'],'S','example',self.m['development_seeds'])

    def test_retry_directory_keeps_previous_attempt(self):
        obj=campaign.Campaign.__new__(campaign.Campaign);obj.m={'campaign_id':'test'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'ROOT',Path(tmp)):
            (Path(tmp)/'runs/test_eval').mkdir(parents=True)
            self.assertEqual(obj.unique('eval'),'test_eval_retry1')
            (Path(tmp)/'runs/test_eval_retry1').mkdir()
            self.assertEqual(obj.unique('eval'),'test_eval_retry2')

    def test_terminal_and_changed_identity_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=copy.deepcopy(self.m)
            for p in [*m['arm_configs'].values(),m['reference_config']]:
                target=root/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
            manifest=root/'manifest.json';manifest.write_text(json.dumps(m))
            with patch.object(campaign,'ROOT',root),patch.object(campaign.Campaign,'check_identity',lambda self:None),patch('subprocess.check_output',return_value='source\n'):
                obj=campaign.Campaign(manifest)
                for status in campaign.TERMINAL:
                    campaign.write(obj.state_path,dict(obj.state,status=status))
                    with self.assertRaises(ValueError):campaign.Campaign(manifest,resume=True)
                campaign.write(obj.state_path,dict(obj.state,status='infrastructure_interrupted'))
                resumed=campaign.Campaign(manifest,resume=True)
                self.assertEqual(resumed.identity,obj.identity)
                bad=dict(obj.state,status='infrastructure_interrupted',identity={})
                campaign.write(obj.state_path,bad)
                with self.assertRaises(ValueError):campaign.Campaign(manifest,resume=True)

    def test_measured_budget_cannot_reduce_protocol(self):
        obj=campaign.Campaign.__new__(campaign.Campaign)
        obj.start=0;obj.state={'diagnostics':{a:dict(wall_seconds=3600,actual_actions=2000) for a in ('C','S')},'engineering':dict(wall_seconds=1000)}
        obj.update=lambda *args,**kwargs:None
        with self.assertRaises(TimeoutError):obj.estimate()

if __name__=='__main__':unittest.main()

class FrozenFlowTests(unittest.TestCase):
    def exercise(self, second_score=6.2):
        import time
        obj=campaign.Campaign.__new__(campaign.Campaign)
        obj.state={'arms':{},'diagnostics':{}};obj.start=time.time()
        obj.m=json.loads((ROOT/'experiments/task4_frozen_manifest.json').read_text())
        obj.update=lambda status,**values:obj.state.update(status=status,**values)
        obj.report=lambda:None;obj.archive_terminal=lambda:None
        calls=[]
        def train(arm,seed,target,previous=None,diagnostic=False):
            calls.append(('train',arm,seed,target,diagnostic))
            return dict(arm=arm,seed=seed,actual_actions=8000 if diagnostic else 60000,
                        checkpoint=f'{arm}{seed}',checkpoint_sha256='sha',wall_seconds=10.,rounds=20 if diagnostic else 150)
        def evaluate(checkpoint,arm,label,worlds,role='candidate',kind='rules'):
            calls.append(('eval',label,len(worlds),role,kind))
            score=4. if role=='reference' else (5. if arm=='C' else second_score)
            return dict(summary=dict(mean_score=score,first_place_rate=.5,suicide_rate=0,act_p95_seconds=.02),
                        raw={'failures':[]},wall_seconds=10.,directories=[])
        obj.train=train;obj.evaluate=evaluate;obj.compare=lambda *a,**kw:{'test':True}
        with patch.object(campaign,'sha256',return_value='sha'):obj.run()
        return obj,calls

    def test_fixed_endpoints_order_unique_final_and_matching_control(self):
        obj,calls=self.exercise()
        formal=[c[1:4] for c in calls if c[0]=='train' and not c[4]]
        self.assertEqual(formal,[(a,s,60000) for s in (22,11,33) for a in ('C','S')])
        final=[c for c in calls if c[0]=='eval' and c[1].endswith('_final')]
        self.assertEqual([c[2] for c in final],[400,400,400])
        self.assertEqual(obj.state['final_candidate']['seed'],11)
        self.assertEqual(obj.state['matching_control']['seed'],11)
        self.assertEqual(obj.state['status'],'observed_improvement')

    def test_initial_screen_failure_does_not_train_third_or_touch_final(self):
        obj,calls=self.exercise(second_score=3.9)
        self.assertEqual(obj.state['status'],'screen_failed')
        self.assertFalse(any(c[0]=='train' and c[2]==33 for c in calls))
        self.assertFalse(any(c[0]=='eval' and c[1].endswith('_final') for c in calls))

class FrozenSeatLimitsTests(unittest.TestCase):
    def test_candidate_margin_and_all_seat_framework_failures(self):
        from experiments.task4_frozen_worker import owned_failure
        from unittest.mock import patch
        with patch('experiments.task4_frozen_worker.safety_failure',return_value=None):
            self.assertIsNone(owned_failure({},'WAIT',{},learner=True,elapsed=.3))
            self.assertEqual(owned_failure({},'WAIT',{},learner=True,elapsed=.300001),'act_max_exceeded_300ms')
            self.assertIsNone(owned_failure({},'WAIT',{},learner=False,elapsed=.300001))
        with patch('experiments.task4_frozen_worker.safety_failure',return_value='act_timeout_or_skip'):
            self.assertEqual(owned_failure({},'WAIT',{},learner=False,elapsed=.6,timed_out=True),'act_timeout_or_skip')
