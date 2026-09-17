import tempfile,unittest,json,time
from pathlib import Path
from unittest.mock import Mock,patch
from experiments.compact_admission import Admission,gates

class AdmissionTests(unittest.TestCase):
    def test_strict_latency_margin_and_legacy_gates(self):
        s=dict(mean_bombs=1,act_timeouts=0,act_skipped=0,avoidable_escape_collapses=0,robust_guarantee_losses=0,robust_search_timeouts=0,act_p95_seconds=.099,act_max_seconds=.249,invalid_action_rate=0,suicide_rate=0,bomb_survival_rate=1,zero_bomb_round_rate=0)
        self.assertEqual(gates(4,s),[])
        self.assertIn('act_max_seconds',gates(4,{**s,'act_max_seconds':.251}))
        self.assertIn('act_p95_seconds',gates(4,{**s,'act_p95_seconds':.101}))
        self.assertTrue(gates(4,{**s,'robust_guarantee_losses':1}))
        self.assertTrue(gates(1,s))

    def test_frozen_failure_stops_before_other_stages(self):
        a=Admission.__new__(Admission);a.state={};a.m={'regression_cases':[{'task':4,'seed':24025},{'task':4,'seed':24384}],'cpus':[0]};a.save=Mock();a.evaluate=Mock(side_effect=RuntimeError('engine failure'));a.wave=Mock();a.execute=Mock()
        self.assertEqual(a.run(),1)
        self.assertEqual(a.evaluate.call_count,1);a.wave.assert_not_called();a.execute.assert_not_called()
        self.assertEqual(a.state['status'],'admission_failed')

    def test_cutoff_does_not_start_process(self):
        a=Admission.__new__(Admission);a.m={'batch_deadline':'2000-01-01T00:00:00Z'};a.check_identity=Mock()
        with patch('experiments.compact_admission.subprocess.Popen') as launch:
            with self.assertRaises(TimeoutError):a.execute('late',[],0)
            launch.assert_not_called()

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';p.write_text('{}')
            a=Admission.__new__(Admission);a.manifest_path=p;a.commit='abc';a.identity={'manifest_sha256':'wrong'};a.m={'runtime_source_sha256':'runtime'}
            with patch('experiments.compact_admission.subprocess.check_output',side_effect=['','abc']),patch('experiments.compact_admission._source_hash',return_value='runtime'):
                with self.assertRaisesRegex(ValueError,'Manifest changed'):a.check_identity()
