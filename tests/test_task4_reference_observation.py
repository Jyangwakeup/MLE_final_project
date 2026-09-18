import json
import pickle
from pathlib import Path
import tempfile
import unittest

from experiments.task4_reference_observation import ReferenceObservation, engineering_view, POLICY
from experiments.task4_worker import safety_failure
from agent_code.team_agent.safety import safety_decision
from experiments.task4_transfer import PARENT_SAFETY, TARGET_SAFETY


def failure_rows():
    root=Path('experiments/results/task4_300ms_20260918/terminal/run_evidence')
    return pickle.loads(next(root.rglob('*s24614/task4_failure_states.pkl')).read_bytes())


class ReferenceTests(unittest.TestCase):
    def test_recorded_unproved_bomb_is_blocked_by_existing_v9(self):
        row=failure_rows()[-2]
        self.assertEqual(row['state']['step'],106)
        decisions=[safety_decision(row['state'],row['safety']['physical_mask'],
            allow_bomb=True,exploring=False,safety_spec=s,own_bomb_pending=False,
            own_bomb_state={}) for s in (PARENT_SAFETY,TARGET_SAFETY)]
        self.assertTrue(decisions[0].mask[5]);self.assertFalse(decisions[1].mask[5])
        self.assertTrue(decisions[1].mask[4])

    def test_exact_known_reference_case_is_observed_without_mutating_flags(self):
        obs=ReferenceObservation();rows=failure_rows()
        for row in rows[-2:]:
            state,action,safety=row['state'],row['action'],row['safety']
            reason=safety_failure(state,action,safety,elapsed=.01)
            actual,record=obs.inspect(state,action,safety,reason,safety_failure,elapsed=.01)
        self.assertIsNone(actual);self.assertEqual(record['placed_step'],106)
        self.assertTrue(safety['robust_guarantee_loss'])
        # The candidate uses the unchanged check and must still stop.
        self.assertEqual(safety_failure(state,action,safety,elapsed=.01),'robust_guarantee_loss')

    def test_unknown_or_certified_placement_loss_is_still_fatal(self):
        row=failure_rows()[-1];obs=ReferenceObservation()
        result,record=obs.inspect(row['state'],row['action'],row['safety'],
            'robust_guarantee_loss',safety_failure,elapsed=.01)
        self.assertEqual(result,'robust_guarantee_loss');self.assertIsNone(record)

    def test_reference_exception_does_not_hide_a_second_failure(self):
        rows=failure_rows();obs=ReferenceObservation();obs.round=1;obs.unproved_placement=106
        row=rows[-1]
        for field in ('robust_search_timed_out',):
            result,record=obs.inspect(row['state'],row['action'],{**row['safety'],field:True},
                'robust_guarantee_loss',safety_failure,elapsed=.01)
            self.assertEqual(result,field);self.assertIsNone(record)
        result,record=obs.inspect(row['state'],row['action'],row['safety'],
            'robust_guarantee_loss',safety_failure,elapsed=.49)
        self.assertEqual(result,'act_max_exceeded');self.assertIsNone(record)

    def test_same_unproved_placement_collapse_is_recorded(self):
        row=failure_rows()[-1];obs=ReferenceObservation();obs.round=1;obs.unproved_placement=106
        safety={**row['safety'],'robust_guarantee_loss':False,'avoidable_escape_collapse':True}
        result,record=obs.inspect(row['state'],row['action'],safety,'avoidable_escape_collapse',safety_failure,elapsed=.01)
        self.assertIsNone(result);self.assertEqual(record['events'],['avoidable_escape_collapse'])

    def test_episode_boundary_cannot_inherit_observation_exception(self):
        obs=ReferenceObservation();obs.round=0;obs.unproved_placement=106;row=failure_rows()[-1]
        result,record=obs.inspect(row['state'],row['action'],row['safety'],
            'robust_guarantee_loss',safety_failure,elapsed=.01)
        self.assertEqual(result,'robust_guarantee_loss')

    def test_reported_counter_is_not_changed_or_unrecognized_loss_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'world';path.mkdir()
            (path/'reference_observations.jsonl').write_text(json.dumps(dict(policy=POLICY,reason='robust_guarantee_loss',events=['robust_guarantee_loss']))+'\n')
            summary={'robust_guarantee_losses':2,'avoidable_escape_collapses':0}
            checked,records=engineering_view(temp,summary)
            self.assertEqual(summary['robust_guarantee_losses'],2)
            self.assertEqual(checked['robust_guarantee_losses'],1)
            self.assertEqual(len(records),1)


if __name__=='__main__':unittest.main()
