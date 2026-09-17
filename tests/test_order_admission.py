from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.order_equivalence_admission import OrderAdmission, EngineeringFailure


class AdmissionTests(unittest.TestCase):
    def controller(self):
        c=OrderAdmission.__new__(OrderAdmission)
        c.root=Path('.');c.path=Path('unused');c.state={'status':'created','diagnostics':{}}
        c.m={'diagnostic_training_seeds':{'11':22411,'22':22422,'33':22433}}
        c.write=lambda *a:None
        c.check_time=lambda **kw:None
        return c

    def test_failure_stops_remaining_seeds_and_terminal_cannot_restart(self):
        c=self.controller();calls=[]
        def train(*args,**kwargs):
            calls.append(args[1]);raise EngineeringFailure('timeout')
        c.train=train
        with patch('experiments.order_equivalence_admission.validate_evidence'):
            c.run();self.assertEqual(c.state['status'],'admission_failed');c.run()
        self.assertEqual(calls,[22422])

    def test_only_three_diagnostic_runs_and_four_hour_deadline(self):
        c=self.controller();calls=[];c.deadline=14400
        c.train=lambda *args,**kwargs:calls.append((args,kwargs)) or {'completed':20}
        with patch('experiments.order_equivalence_admission.validate_evidence'):
            c.run()
        self.assertEqual(c.state['status'],'admission_passed')
        self.assertEqual([args[1] for args,kw in calls],[22422,22411,22433])
        self.assertTrue(all(args[2]==20 and kw=={'diagnostic':True} for args,kw in calls))
        with patch('experiments.order_equivalence_admission.time.time',return_value=14400):
            with self.assertRaises(TimeoutError):OrderAdmission.check_time(c)

    def test_final_audit_cannot_cross_deadline_and_claim_success(self):
        c=self.controller();c.train=lambda *a,**kw:{'completed':20}
        def expired(**kwargs):raise TimeoutError('deadline after final audit')
        c.check_time=expired
        with patch('experiments.order_equivalence_admission.validate_evidence'):
            c.run()
        self.assertEqual(c.state['status'],'budget_exhausted')

    def test_success_label_does_not_admit_incomplete_benchmark(self):
        from experiments.order_equivalence_admission import validate_evidence
        manifest={'prerequisites':{'benchmark':{'path':'benchmark.json','sha256':'hash'}}}
        result=dict(status='historical_passed',failed_steps=[],search_budget_ms=400,
                    repetitions=10,warmups=1,cpu=[0],states=[])
        with patch('experiments.order_equivalence_admission.p._sha256',return_value='hash'), \
             patch('experiments.order_equivalence_admission.p._json',return_value=result):
            with self.assertRaisesRegex(ValueError,'corpus incomplete'):
                validate_evidence(Path('.'),manifest)

    def test_unexplained_self_death_is_audited_at_round_end(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock
        from experiments import task4_worker as worker
        with tempfile.TemporaryDirectory() as d:
            world=SimpleNamespace(_experiment_run_id='test',_snapshot_config={'train':True},
                _timing_path=Path(d)/'timing.jsonl',_timing_file=Mock(),_episodes_file=Mock(),round=1,step=42,
                agents=[SimpleNamespace(name=worker.AGENT,dead=True,statistics={'suicides':1})])
            with patch.object(worker.run.ExperimentWorld,'end_round') as original, \
                 patch.object(worker.run,'main',side_effect=lambda argv:worker.run.ExperimentWorld.end_round(world)), \
                 patch('experiments.task4_campaign.audit_training',side_effect=EngineeringFailure('unexplained')) as audit:
                with self.assertRaises(EngineeringFailure):worker.main([])
                original.assert_called_once_with(world)
                audit.assert_called_once_with(Path(d),worker.AGENT,check_timing=False)
                self.assertIs(worker.run.ExperimentWorld.end_round,original)
            self.assertEqual(json.loads((Path(d)/'task4_safety_failure.json').read_text())['reason'],'unexplained_self_death')
