"""Exact differential checks against the pre-optimization enumerator."""
from dataclasses import asdict
from itertools import permutations
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from agent_code.team_agent import opponent_transitions as new
from agent_code.team_agent import controllable_survival as survival
from tests import reference_opponent_transitions as old
from tests.test_danger import make_game_state

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'experiments/results/task4_shared_parent_20260917/failure_states.json'


def native(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    return value


def historical_states():
    records = json.loads(CORPUS.read_text())
    for record in records:
        s = record['state']
        for key in ('field', 'explosion_map'): s[key] = np.asarray(s[key], dtype=int)
        s['self'] = (*s['self'][:3], tuple(s['self'][3]))
        s['others'] = [(*a[:3], tuple(a[3])) for a in s['others']]
        s['bombs'] = [(tuple(p), timer) for p, timer in s['bombs']]
        s['coins'] = [tuple(p) for p in s['coins']]
    return records


def search_arguments(record):
    d = record['safety']
    if d['own_bomb_pending']:
        actions = tuple(a for i, a in enumerate(new.ACTIONS)
                        if a != 'BOMB' and d['survives_horizon'][i])
        remaining = survival.danger_interval_steps(
            {'pending': True, 'timer': d['own_bomb_timer']}, placing_bomb=False)
    else:
        actions = ('BOMB',) if d['survives_horizon'][5] and d['physical_mask'][5] else ()
        remaining = survival.danger_interval_steps({}, placing_bomb=True)
    return actions, remaining


class OrderEquivalenceTests(unittest.TestCase):
    def compare(self, state, actions=new.ACTIONS):
        original = native(state)
        for progress in (False, True):
            for action in actions:
                with self.subTest(action=action, progress=progress):
                    expected = old.enumerate_opponent_transition_scenarios(state, action, progress_world=progress)
                    actual = new.enumerate_opponent_transition_scenarios(state, action, progress_world=progress)
                    self.assertEqual([native(asdict(x)) for x in expected],
                                     [native(asdict(x)) for x in actual])
        self.assertEqual(original, native(state))

    def test_action_phase_conflicts_chains_cycles_and_bomb_list(self):
        cases = [
            ([(3,3),(9,9),(11,11),(13,13)], ('UP','DOWN','LEFT','WAIT')),
            ([(3,3),(5,3),(9,9)], ('RIGHT','LEFT','WAIT')),
            ([(3,3),(4,3),(5,3)], ('RIGHT','RIGHT','RIGHT')),
            ([(3,3),(4,3)], ('RIGHT','LEFT')),
            ([(3,3),(4,3),(4,4),(3,4)], ('RIGHT','DOWN','LEFT','UP')),
            ([(3,3),(4,3)], ('RIGHT','BOMB')),
            ([(3,3),(9,9),(11,11),(13,13)], ('BOMB','BOMB','BOMB','BOMB')),
            ([(3,3),(9,9)], ('WAIT','WAIT')),
        ]
        for positions, profile in cases:
            state = make_game_state(position=positions[0])
            state['others'] = [(str(i),0,True,p) for i,p in enumerate(positions[1:])]
            actors = [state['self'],*state['others']]
            def outcomes(orders):
                result = {}
                for order in orders:
                    value = native(old._apply_action_phase(state,profile,order))
                    key = json.dumps(value,sort_keys=True)
                    result.setdefault(key,order)
                return list(result.items())
            self.assertEqual(outcomes(permutations(range(len(actors)))),
                             outcomes(new._execution_orders(actors,profile)))
            self.compare(state)
        self.assertEqual(len(new._representative_orders(4,0)),1)
        self.assertEqual(len(new._representative_orders(4,63)),24)

    def test_bombs_capacity_crates_flames_deaths_and_illegal_own_moves(self):
        for capacity in (False, True):
            state = make_game_state(position=(3,3), bombs_left=capacity)
            state['others'] = [('a',0,capacity,(4,3)),('b',0,True,(7,3)),('c',0,False,(9,3))]
            state['bombs'] = [((3,3),3),((5,3),0),((8,3),0)]
            state['field'][6,3] = 1
            state['explosion_map'][9,3] = 1
            self.compare(state)
        with self.assertRaises(ValueError):new.enumerate_opponent_transition_scenarios(state,'INVALID')

    def test_256_fixed_synthetic_states_zero_to_three_opponents(self):
        rng = np.random.default_rng(20260917)
        for index in range(256):
            state = make_game_state(position=(1,1))
            field = state['field']
            field[1:-1,1:-1] = rng.choice([-1,0,1],size=field[1:-1,1:-1].shape,p=[.12,.65,.23])
            cells = np.argwhere(field == 0)
            selected = cells[rng.choice(len(cells),size=8,replace=False)]
            positions = [tuple(map(int,p)) for p in selected]
            state['self'] = ('me',0,bool(rng.integers(2)),positions[0])
            state['others'] = [(str(i),0,bool(rng.integers(2)),positions[i+1]) for i in range(index%4)]
            state['bombs'] = [(positions[4+i],int(rng.integers(4))) for i in range(index%3)]
            if index%5 == 0:state['explosion_map'][positions[7]] = 1
            with self.subTest(index=index):self.compare(state)

    def test_twenty_historical_states_complete_proof_and_evidence(self):
        for record in historical_states():
            actions, remaining = search_arguments(record)
            with self.subTest(step=record['state']['step']):
                self.compare(record['state'],actions)
                with patch.object(survival,'enumerate_opponent_transition_scenarios',old.enumerate_opponent_transition_scenarios):
                    expected = survival.controllable_survival_actions(record['state'],actions,remaining_steps=remaining,budget_ms=60000)
                actual = survival.controllable_survival_actions(record['state'],actions,remaining_steps=remaining,budget_ms=60000)
                self.assertFalse(expected.timed_out)
                self.assertFalse(actual.timed_out)
                self.assertEqual(expected,actual)
