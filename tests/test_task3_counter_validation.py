import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.task3_counter_validation import Validation, validate_manifest


class CounterValidationTests(unittest.TestCase):
    def test_reject_reused_or_overlapping_worlds(self):
        m=json.loads(Path('experiments/task3_counter_validation.json').read_text())
        validate_manifest(m)
        for replacement in (m['confirmation_seeds'], list(range(19400,19500))):
            bad=copy.deepcopy(m);bad['main_validation_seeds']=replacement
            with self.assertRaises(ValueError):validate_manifest(bad)

    def campaign(self, failing_seed=None, main_pass=True):
        v=Validation.__new__(Validation)
        v.state={'status':'created'};v.m={'selected_training_seed':22}
        v.path=Path('unused');v.write=lambda *a:None
        calls=[]
        def seed(stage, number):
            calls.append((stage,number))
            passed=(number!=failing_seed) if stage=='confirmation' else main_pass
            return {'status':'passed' if passed else 'gate_failed'}
        v.seed=seed
        return v,calls

    def test_confirmation_failure_never_starts_main(self):
        v,calls=self.campaign(failing_seed=11)
        self.assertEqual(v.run()['status'],'stopped_confirmation_failure')
        self.assertEqual(set(calls),{('confirmation',s) for s in (11,22,33)})

    def test_fixed_main_candidate_failure_is_terminal(self):
        v,calls=self.campaign(main_pass=False)
        self.assertEqual(v.run()['status'],'stopped_main_validation_failure')
        self.assertEqual([c for c in calls if c[0]=='main_validation'],[('main_validation',22)])
        before=list(calls);v.run();self.assertEqual(calls,before)

    def test_changed_resume_identity_rejected(self):
        v=Validation.__new__(Validation);v.identity={'source_commit':'new'}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json';path.write_text('{"source_commit":"old"}')
            with self.assertRaises(ValueError):v.read(path)
