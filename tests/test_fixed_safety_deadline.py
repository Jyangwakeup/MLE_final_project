"""Proof obligations retain their placement deadline across observations."""
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from agent_code.team_agent.safety import safety_decision
from agent_code.learning_common.action_history import (
    init_action_history, record_selected_action, snapshot_history,
    project_history, bomb_history, action_history_state, load_action_history_state,
)
from tests.test_danger import make_game_state
from tests.test_opponent_rearming import V7, record as rearming_record

V8 = {**V7,'version':'survival-mask-v8','danger_interval':'own_bomb_capacity_deadline',
      'proof_clock':'recorded_placement_step'}
ROOT=Path(__file__).resolve().parents[1]


def deadline_record(step):
    rows=json.loads((ROOT/'experiments/results/safety_v7_retention_failure_20260917/failure_states.json').read_text())
    r=next(r for r in rows if r['state']['step']==step);s=r['state']
    for k in ('field','explosion_map'):s[k]=np.asarray(s[k])
    s['self']=(*s['self'][:3],tuple(s['self'][3]))
    s['others']=[(*a[:3],tuple(a[3])) for a in s['others']]
    s['bombs']=[(tuple(p),t) for p,t in s['bombs']]
    return r


def decide(r, placed, spec=V8):
    d=r['safety']
    return safety_decision(r['state'],np.asarray(d['physical_mask']),spec,
        allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],
        own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],
                        'placed_step':placed})


class FixedDeadlineTest(unittest.TestCase):
    def test_recorded_rolling_horizon_failure_has_proven_continuation(self):
        r=deadline_record(162)
        self.assertTrue(decide(r,159,V7).robust_guarantee_loss)
        result=decide(r,159)
        self.assertFalse(result.robust_guarantee_loss)
        self.assertEqual(result.mask.tolist(),[False,True,False,False,False,False])

    def test_rearming_trap_is_still_rejected(self):
        result=decide(rearming_record(182),181)
        self.assertFalse(result.mask[2])
        self.assertTrue(result.mask[0])
        self.assertFalse(result.robust_guarantee_loss)

    def test_origin_is_recorded_and_old_queries_do_not_change_it(self):
        owner=SimpleNamespace();init_action_history(owner)
        state=make_game_state(position=(7,7));state['step']=159
        record_selected_action(owner,state,'BOMB');history=snapshot_history(owner)
        self.assertEqual(bomb_history(history,state)['placed_step'],159)
        next_state=make_game_state(position=(7,6),bombs_left=False,bombs=[((7,7),3)])
        next_state['step']=160
        projected=project_history(history,next_state)
        self.assertEqual(bomb_history(projected,next_state)['placed_step'],159)
        project_history(history,state)
        self.assertEqual(snapshot_history(owner),history)
        saved=action_history_state(owner);restored=SimpleNamespace()
        load_action_history_state(restored,saved)
        self.assertEqual(snapshot_history(restored),history)
        released=project_history(projected,{**next_state,'self':('me',0,True,(7,6))})
        self.assertIsNone(bomb_history(released,next_state)['placed_step'])
        new_round=project_history(projected,{**next_state,'round':2})
        self.assertIsNone(bomb_history(new_round,next_state)['placed_step'])


class FixedDeadlineClockTest(unittest.TestCase):
    def test_countdown_and_invisible_flame_keep_absolute_endpoint(self):
        from agent_code.team_agent.safety import fixed_deadline_remaining
        for step, timer in ((160, 3), (161, 2), (162, 1), (163, 0),
                            (164, None), (165, None)):
            with self.subTest(step=step):
                remaining = fixed_deadline_remaining(
                    {'step': step}, {'placed_step': 159, 'timer': timer},
                    pending=True, duration=7)
                self.assertEqual(step + remaining, 166)
        self.assertEqual(fixed_deadline_remaining(
            {'step': 166}, {}, pending=False, duration=7), 7)

    def test_missing_or_inconsistent_clock_is_rejected(self):
        from agent_code.team_agent.safety import fixed_deadline_remaining
        for step, placed, timer in ((160, None, 3), (160, True, 3),
                                   (159, 159, 4), (158, 159, 4),
                                   (166, 159, None), (161, 159, 3)):
            with self.subTest(step=step, placed=placed, timer=timer):
                with self.assertRaises(ValueError):
                    fixed_deadline_remaining(
                        {'step': step}, {'placed_step': placed, 'timer': timer},
                        pending=True, duration=7)

    def test_actual_callback_chain_preserves_clock_and_projection(self):
        from tests.test_task3_lifecycle import LifecycleTests, state
        from agent_code.double_dqn_continuous_v2_agent import callbacks, train
        fixture = LifecycleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        owner = fixture.owner
        owner.safety_spec = V8
        old, new = state(), state(2, False, [((3, 3), 3)])
        action = callbacks.act(owner, old)
        self.assertEqual(action, 'BOMB')
        self.assertEqual(snapshot_history(owner).own_bomb_placed_step, 1)
        train.game_events_occurred(owner, old, action, new, ['BOMB_DROPPED'])
        projected_transition = owner.pending[1]
        self.assertEqual(snapshot_history(owner).own_bomb_placed_step, 1)
        selected = callbacks.act(owner, new)
        np.testing.assert_array_equal(projected_transition.next_legal,
                                      owner._last_decision_mask)
        np.testing.assert_array_equal(projected_transition.next_state,
                                      owner._decision_snapshot.vector)
        self.assertEqual(owner.last_safety_diagnostic['own_bomb_placed_step'], 1)
        train.end_of_round(owner, new, selected, [])
        self.assertIsNone(snapshot_history(owner).own_bomb_placed_step)
