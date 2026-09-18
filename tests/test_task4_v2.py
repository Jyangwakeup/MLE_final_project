import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.task4_campaign import Campaign, EngineeringFailure, evaluate_gates, engineering_checks, validate_manifest
from experiments.task4_protocol import CANDIDATE, LEGACY, VERSION, limits
from experiments.task4_v2_worker import checked_failure
from experiments.task4_worker import safety_failure
from tests.test_task4_campaign import summary


class ProtocolTests(unittest.TestCase):
    def test_boundary_and_original_failure(self):
        state={'self':('me',0,True,(1,1))}
        safe=dict(diagnostic_version='escape-collapse-v2',own_bomb_pending=False,
                  selected_action='WAIT',physical_mask=[True]*6,decision_mask=[True]*6)
        for elapsed in (.2626025676727295,.3):
            self.assertIsNone(checked_failure(safety_failure,CANDIDATE,state,'WAIT',safe,elapsed=elapsed))
        self.assertEqual(checked_failure(safety_failure,CANDIDATE,state,'WAIT',safe,elapsed=.3000001),
                         'campaign_act_max_exceeded')
        self.assertIsNone(checked_failure(safety_failure,LEGACY,state,'WAIT',safe,elapsed=.4))
        self.assertEqual(checked_failure(safety_failure,CANDIDATE,state,'WAIT',
            {**safe,'robust_search_timed_out':True},elapsed=.1),'robust_search_timed_out')

    def test_all_checks_use_same_limits_and_v1_is_unchanged(self):
        parent={f'task{t}':summary() for t in range(1,5)};parent['task1']['mean_bombs']=0
        child=copy.deepcopy(parent);child['task4'].update(mean_score=5.5,mean_kills=.6,
                                                      act_p95_seconds=.1,act_max_seconds=.3)
        self.assertTrue(evaluate_gates(parent,child,CANDIDATE)[0])
        self.assertFalse(engineering_checks(child,CANDIDATE))
        for key,value in [('act_p95_seconds',.100001),('act_max_seconds',.300001)]:
            failed=copy.deepcopy(child);failed['task4'][key]=value
            self.assertFalse(evaluate_gates(parent,failed,CANDIDATE)[0])
            self.assertTrue(engineering_checks(failed,CANDIDATE))
            self.assertTrue(evaluate_gates(parent,failed)[0])
        self.assertEqual(limits(VERSION,'reference'),LEGACY)

    def test_manifest_rejects_threshold_and_seed_changes(self):
        m=json.loads(Path('experiments/task4_300ms_campaign.json').read_text());validate_manifest(m)
        for change in ('limits','reference_config','development_seeds','diagnostic_training_seeds'):
            bad=copy.deepcopy(m)
            if change=='limits':bad[change]['candidate']['maximum']=.31
            elif change=='reference_config':bad[change]=''
            elif change=='development_seeds':bad[change][0]=m['confirmation_seeds'][0]
            else:bad[change]['33']=22434
            with self.assertRaises(ValueError):validate_manifest(bad)

    def test_explicit_reference_config_and_cache_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);v=Campaign.__new__(Campaign);v.root=root;v.directory=root/'campaign';v.commit='abc1234'
            for role,version in [('reference','v5'),('candidate','v9')]:
                (root/f'{role}.json').write_text(json.dumps({'safety':{'version':version}}))
            checkpoint=root/'parent.pt';checkpoint.write_bytes(b'parent')
            v.m=dict(schema_version=VERSION,reference_config='reference.json',arm_configs={'A':'candidate.json'},
                     cpus=[0],campaign_id='test',agent='me')
            v.identity={'manifest_sha256':'01234567'};v.check_time=lambda **kw:None;v.check_identity=lambda:None
            seen=[]
            def execute(args,name,cpu,role):
                seen.append((args,role));folder=root/'runs'/name/(name+'_summary');folder.mkdir(parents=True)
                (folder/'summary.csv').write_text('agent_name,run_id,mean_bombs\nme,AVERAGE,2\n')
            v.execute=execute
            with patch('experiments.task4_campaign.summarize_evaluation',return_value=summary()):
                result=v.evaluate('A',checkpoint,'parent','development',[1],tasks=(4,),role='reference')
            self.assertEqual(seen[0][0][1],'reference.json');self.assertEqual(seen[0][1],'reference')
            self.assertEqual(result['request']['safety']['version'],'v5')
            # The same label/checkpoint cannot reuse a reference result as a candidate.
            with self.assertRaisesRegex(ValueError,'cache request mismatch'):
                v.evaluate('A',checkpoint,'parent','development',[1],tasks=(4,))

    def test_first_diagnostic_failure_never_starts_baseline_or_training(self):
        v=Campaign.__new__(Campaign);v.m={'diagnostic_training_seeds':{'22':22422,'11':22411,'33':22433}}
        v.state=dict(status='created',diagnostics={});v.path=Path('unused');v.write=lambda *a:None
        calls=[]
        def train(*a,**kw):calls.append(a);raise EngineeringFailure('latency')
        v.train=train;v.evaluate=lambda *a,**kw:self.fail('baseline opened after failure')
        result=v.run();self.assertEqual(result['status'],'stopped_engineering_failure');self.assertEqual(len(calls),1)
        v.run();self.assertEqual(len(calls),1)

    def test_reused_evidence_rejects_runtime_or_file_mutation(self):
        from experiments.task4_v2_evidence import validate_evidence
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'evidence').write_text('changed')
            m={'agent':'a','runtime_source_sha256':'old','reused_evidence':{'evidence':'wrong'}}
            with patch('experiments.run._source_hash',return_value='new'):
                with self.assertRaisesRegex(ValueError,'semantics changed'):validate_evidence(root,m)
            with patch('experiments.run._source_hash',return_value='old'):
                with self.assertRaisesRegex(ValueError,'Changed reused evidence'):validate_evidence(root,m)


if __name__=='__main__':unittest.main()
