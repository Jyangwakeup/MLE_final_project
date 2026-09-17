import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from experiments.viability_prerequisites import validate

class PrerequisiteTests(unittest.TestCase):
    def test_changed_evidence_is_rejected_before_opening_worlds(self):
        with patch('experiments.viability_prerequisites.sha256',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'Changed viability prerequisite'):
                validate(Path('.'),{'admission_prerequisites':{'benchmark':{'path':'x','sha256':'expected'}}})

    def test_success_label_with_missing_states_is_rejected(self):
        evidence={'status':'historical_passed','failed_states':[],'search_budget_ms':400,
                  'repetitions':10,'warmups':1,'cpu':[0],'states':[]}
        with patch('experiments.viability_prerequisites.sha256',return_value='ok'), \
             patch.object(Path,'read_text',return_value='{}'), \
             patch('experiments.viability_prerequisites.json.loads',return_value=evidence):
            with self.assertRaisesRegex(ValueError,'Incomplete failure corpus'):
                validate(Path('.'),{'admission_prerequisites':{'benchmark':{'path':'x','sha256':'ok'}}})
