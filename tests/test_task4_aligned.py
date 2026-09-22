import copy,json,pickle,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.task4_protocol import ALIGNED_VERSION,CANDIDATE,LEGACY,limits
from experiments.task4_transfer import PARENT_SAFETY,TARGET_SAFETY
from experiments.task4_aligned import run_aligned,validate_protocol
from experiments.task4_campaign import Campaign,EngineeringFailure
from experiments.task4_v2_worker import checked_failure
from experiments.task4_worker import safety_failure

ROOT=Path(__file__).resolve().parents[1]
class World25106(unittest.TestCase):
    def test_proof_duration_and_real_responsibility_chain(self):
        from agent_code.team_agent.safety import safety_decision
        from agent_code.team_agent.controllable_survival import controllable_survival_actions
        rows=pickle.loads((ROOT/'experiments/results/task4_v9_reference_20260918/world25106.pkl').read_bytes())
        for row in rows[-4:]:
            s=row['state'];d=row['safety'];own={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],'placed_step':d['own_bomb_placed_step']}
            actual={k:safety_decision(s,d['physical_mask'],allow_bomb=True,exploring=False,safety_spec=v,own_bomb_pending=d['own_bomb_pending'],own_bomb_state=own) for k,v in [('v5',PARENT_SAFETY),('v9',TARGET_SAFETY)]}
            if s['step']==71:self.assertTrue(actual['v5'].mask[5]);self.assertFalse(actual['v9'].mask[5])
            if s['step']==73:self.assertTrue(actual['v5'].mask[3]);self.assertFalse(actual['v9'].mask[3])
            if s['step']==74:self.assertTrue(actual['v5'].physical_fallback);self.assertEqual(own['placed_step'],71)
        for duration in (6,7):
            for rearm in (False,True):
                a=controllable_survival_actions(rows[-4]['state'],('BOMB',),remaining_steps=duration,budget_ms=400,consider_opponent_rearming=rearm)
                self.assertFalse(a.timed_out);self.assertEqual(bool(a.proven_actions),duration==6)

class AlignedProtocol(unittest.TestCase):
    def test_parent_has_candidate_limits_and_failures_are_not_exempt(self):
        self.assertEqual(limits(ALIGNED_VERSION,'reference'),CANDIDATE)
        self.assertEqual(limits('task4-campaign-v3','reference'),LEGACY)
        self.assertEqual(checked_failure(lambda *a,**k:'avoidable_escape_collapse',CANDIDATE,{},'WAIT',{},elapsed=.01),'avoidable_escape_collapse')
        self.assertIsNone(checked_failure(lambda *a,**k:None,CANDIDATE,{},'WAIT',{},elapsed=.3))
        self.assertEqual(checked_failure(lambda *a,**k:None,CANDIDATE,{},'WAIT',{},elapsed=.300001),'campaign_act_max_exceeded')

    def fake(self):
        c=SimpleNamespace(root=Path('/tmp'),path=Path('/tmp/unused'),m={'parent':{'checkpoint':'parent'},'engineering_seeds':[1],'development_seeds':[2],'diagnostic_training_seeds':{'22':32422,'11':32411,'33':32433}},state={'arms':{},'diagnostics':{}})
        c.write=lambda *a:None;c.check_time=lambda **k:None
        c.evaluate=lambda *a,**k:{'summaries':{}}
        c.train=lambda *a,**k:{'checkpoint':'diagnostic'}
        c.validation=lambda *a: c.state.update(status='passed')
        return c

    def test_engineering_failure_never_starts_training(self):
        c=self.fake();c.evaluate=lambda *a,**k:(_ for _ in ()).throw(EngineeringFailure('baseline'))
        c.train=lambda *a,**k:self.fail('training after failed baseline')
        with patch('experiments.task4_aligned.await_previous'),patch('experiments.task4_aligned.certify_retention'):
            with self.assertRaises(EngineeringFailure):run_aligned(c)

    def test_only_b_and_first_failed_seed_stops(self):
        c=self.fake();calls=[]
        def seed(arm,seed,parent):calls.append((arm,seed));return {'selected':None}
        c.seed=seed;c.validation=lambda *a:self.fail('validation after failed seed')
        with patch('experiments.task4_aligned.await_previous'),patch('experiments.task4_aligned.certify_retention'):run_aligned(c)
        self.assertEqual(calls,[('B',22)]);self.assertEqual(c.state['status'],'stopped_development_failure')

    def test_diagnostic_checkpoints_not_training_parents(self):
        c=self.fake();calls=[]
        def seed(arm,seed,parent):calls.append((arm,seed,parent));return {'selected':{'checkpoint':'formal'}}
        c.seed=seed
        with patch('experiments.task4_aligned.await_previous'),patch('experiments.task4_aligned.certify_retention'):run_aligned(c)
        self.assertEqual([x[:2] for x in calls],[('B',22),('B',11),('B',33)])
        self.assertTrue(all(x[2]=={'summaries':{}} for x in calls))
        self.assertEqual(c.state['status'],'passed')

    def test_p95_boundary(self):
        from experiments.task4_campaign import engineering_checks
        from tests.test_task4_campaign import summary
        s=summary();s['act_p95_seconds']=.1;s['act_max_seconds']=.3
        self.assertEqual(engineering_checks({'task4':s},CANDIDATE),[])
        s['act_p95_seconds']=.100001
        self.assertIn('task4_act_p95',engineering_checks({'task4':s},CANDIDATE))

    def test_manifest_rejects_mixed_contracts_and_used_worlds(self):
        m=json.loads((ROOT/'experiments/task4_v9_reference_campaign.json').read_text())
        validate_protocol(m)
        for change in ('limits','arm_configs','engineering_seeds','reference_observation_policy'):
            b=copy.deepcopy(m)
            if change=='limits':b[change]['reference']['maximum']=.48
            elif change=='arm_configs':b[change]['A']='old.json'
            elif change=='engineering_seeds':b[change][0]=b['confirmation_seeds'][0]
            else:b[change]='ignore'
            with self.assertRaises(ValueError):validate_protocol(b)

    def test_cutoffs(self):
        c=Campaign.__new__(Campaign);c.started=100
        for kwargs,hours in [({'new_arm':True},18),({'evaluation':True},23),({},24)]:
            with patch('experiments.task4_campaign.time.time',return_value=100+hours*3600):
                with self.assertRaises(TimeoutError):c.check_time(**kwargs)

    def test_cache_identity_rejects_role_config_or_manifest_changes(self):
        c=Campaign.__new__(Campaign);c.identity={'manifest_sha256':'new','reference_config_sha256':'v9'}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cache.json';p.write_text(json.dumps({'manifest_sha256':'old','reference_config_sha256':'v5'}))
            with self.assertRaises(ValueError):c.read(p)
