import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.task4_campaign import (
    Campaign, EngineeringFailure, validate_manifest, evaluate_gates, b_eligible,
)
from experiments.task4_worker import safety_failure


def summary(score=5):
    return dict(mean_score=score,mean_coins=4.,mean_crates=20.,mean_kills=.5,
                first_place_rate=.4,suicide_rate=0.,bomb_survival_rate=1.,
                zero_bomb_round_rate=0.,invalid_action_rate=0.,act_p95_seconds=.02,
                act_max_seconds=.1,act_timeouts=0,act_skipped=0,
                avoidable_escape_collapses=0,robust_guarantee_losses=0,
                robust_search_timeouts=0,mean_bombs=2.)


def point(seed=22,eligible=True):
    checks={'task4_score_gain':dict(passed=eligible), 'task4_combat_gain':dict(passed=eligible),
            'task1_mean_score_retention':dict(passed=True,actual=1.)}
    return dict(seed=seed,rounds=50,eligible=eligible,gate_checks=checks,
                child=dict(summaries={'task4':summary()}))


class CampaignTests(unittest.TestCase):
    def test_registered_worlds_are_separate_and_fresh(self):
        m=json.loads(Path('experiments/task4_campaign.json').read_text());validate_manifest(m)
        for field in ('excluded_seeds','main_validation_seeds'):
            bad=copy.deepcopy(m)
            if field=='excluded_seeds':bad[field]+=m['development_seeds']
            else:bad[field]=m['confirmation_seeds']
            with self.assertRaises(ValueError):validate_manifest(bad)

    def test_all_gates_and_exact_thresholds(self):
        parent={f'task{t}':summary() for t in range(1,5)}
        parent['task1']['mean_bombs']=0
        child=copy.deepcopy(parent);child['task4']['mean_score']+=.5;child['task4']['first_place_rate']+=.05
        self.assertTrue(evaluate_gates(parent,child)[0])
        child['task3']['mean_score']=4.49
        self.assertFalse(evaluate_gates(parent,child)[0])
        child=copy.deepcopy(parent);child['task4']['mean_score']+=.5;child['task4']['mean_kills']+=.1
        self.assertTrue(evaluate_gates(parent,child)[0])
        child['task4']['robust_search_timeouts']=1
        self.assertFalse(evaluate_gates(parent,child)[0])

    def test_b_requires_every_failed_replica_to_have_only_gain_failure(self):
        r={'22':dict(selected=point(),history=[point()]),'11':dict(selected=None,history=[point(11,False)])}
        self.assertTrue(b_eligible(r))
        r['11']['history'][0]['gate_checks']['retention']=dict(passed=False)
        self.assertFalse(b_eligible(r))
        r['11']['history'].append(point(11,False));self.assertTrue(b_eligible(r))

    def test_confirmation_failure_and_main_failure_are_terminal(self):
        for bad_confirm in (True,False):
            v=Campaign.__new__(Campaign);v.root=Path('.');v.directory=Path('unused');v.path=Path('unused/result.json')
            v.m={'parent':{'checkpoint':'parent'},'confirmation_seeds':list(range(100)),'main_validation_seeds':list(range(100,200))}
            v.state={'status':'confirmation'};v.write=lambda *a:None
            calls=[];v.evaluate=lambda *a,**kw: {}
            def assess(arm,seed,trained,phase,*args):
                calls.append((phase,seed));return point(seed,not ((bad_confirm and phase=='confirmation' and seed==11) or (not bad_confirm and phase=='main_validation')))
            v.assess=assess
            results={str(s):{'selected':point(s)} for s in (11,22,33)}
            v.validation('A',results)
            self.assertEqual(v.state['status'],'stopped_confirmation_failure' if bad_confirm else 'stopped_main_validation_failure')
            self.assertEqual(sum(stage=='main_validation' for stage,_ in calls),0 if bad_confirm else 1)
            if bad_confirm:self.assertEqual(calls,[('confirmation',11)])
            before=list(calls);v.run();self.assertEqual(calls,before)

    def test_deadlines_and_changed_cache_or_checkpoint_are_rejected(self):
        v=Campaign.__new__(Campaign);v.started=0;v.identity={'source_commit':'new'}
        for hour,kwargs in [(18,{'new_arm':True}),(23,{'evaluation':True}),(24,{})]:
            with patch('experiments.task4_campaign.time.time',return_value=hour*3600):
                with self.assertRaises(TimeoutError):v.check_time(**kwargs)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.json';p.write_text('{"source_commit":"old"}')
            with self.assertRaises(ValueError):v.read(p)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'weight changed'):
                v.assess('A',22,{'checkpoint':str(p),'checkpoint_sha256':'wrong'},'confirmation',{},[])

    def test_safety_failure_is_separate_from_strategy_performance(self):
        state={'self':('me',0,False,(1,1))}
        s=dict(diagnostic_version='escape-collapse-v2',own_bomb_pending=True,
               selected_action='WAIT',physical_mask=[True]*6,decision_mask=[True]*6)
        self.assertIsNone(safety_failure(state,'WAIT',s,elapsed=.1))
        for key in ('robust_search_timed_out','robust_guarantee_loss','avoidable_escape_collapse'):
            self.assertEqual(safety_failure(state,'WAIT',{**s,key:True},elapsed=.1),key)
        self.assertEqual(safety_failure(state,'WAIT',{**s,'own_bomb_pending':False},elapsed=.1),'lost_bomb_responsibility')
        self.assertEqual(safety_failure(state,'WAIT',s,elapsed=.49),'act_max_exceeded')
        self.assertEqual(safety_failure(state,'BOMB',s,elapsed=.1),'decision_mask_mismatch')
