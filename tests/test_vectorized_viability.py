"""Complete proof equivalence and real historical timeout regression."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest

import numpy as np
from agent_code.team_agent import controllable_survival as actual
from tests import reference_controllable_survival as reference
from tests.test_order_equivalence import search_arguments, native
from tests.test_danger import make_game_state

ROOT=Path(__file__).resolve().parents[1]
CORPORA=('task4_shared_parent_20260917','order_equivalence_20260917','viability_performance_20260917')

def records():
    out=[]
    for name in CORPORA:
        for r in json.loads((ROOT/'experiments/results'/name/'failure_states.json').read_text()):
            s=r['state'];s['field']=np.asarray(s['field']);s['explosion_map']=np.asarray(s['explosion_map'])
            s['self']=(*s['self'][:3],tuple(s['self'][3]))
            s['others']=[(*a[:3],tuple(a[3])) for a in s['others']]
            s['bombs']=[(tuple(p),t) for p,t in s['bombs']]
            r['corpus']=name;out.append(r)
    return out

class ViabilityTests(unittest.TestCase):
    def compare(self,state,actions,remaining):
        before=native(state)
        expected=reference.controllable_survival_actions(state,actions,remaining_steps=remaining,budget_ms=60000)
        result=actual.controllable_survival_actions(state,actions,remaining_steps=remaining,budget_ms=60000)
        self.assertFalse(expected.timed_out);self.assertFalse(result.timed_out)
        self.assertEqual(asdict(expected),asdict(result));self.assertEqual(before,native(state))

    def test_sixty_historical_complete_proofs_and_evidence(self):
        for r in records():
            with self.subTest(corpus=r['corpus'],step=r['state']['step']):
                actions,remaining=search_arguments(r);self.compare(r['state'],actions,remaining)

    def test_128_random_proofs_wait_bombs_flames_and_boundaries(self):
        rng=np.random.default_rng(2026091702)
        for index in range(128):
            state=make_game_state(position=(1,1));f=state['field']
            f[1:-1,1:-1]=rng.choice([-1,0,1],size=f[1:-1,1:-1].shape,p=[.12,.6,.28])
            cells=np.argwhere(f==0);chosen=cells[rng.choice(len(cells),8,replace=False)]
            pos=[tuple(map(int,p)) for p in chosen]
            state['self']=('me',0,index%2==0,pos[0])
            state['others']=[(str(i),0,bool(rng.integers(2)),pos[i+1]) for i in range(index%4)]
            state['bombs']=[(pos[4+i],int(rng.integers(4))) for i in range(index%3)]
            if index%4==0:state['bombs'].append((pos[0],index%3))
            if index%5==0:state['explosion_map'][pos[7]]=1
            with self.subTest(index=index):self.compare(state,('UP','RIGHT','DOWN','LEFT','WAIT','BOMB'),1+index%6)

    def test_historical_timeout_regression_runtime_budget_unchanged(self):
        r=records()[-1];actions,remaining=search_arguments(r)
        for repeat in range(10):
            result=actual.controllable_survival_actions(r['state'],actions,remaining_steps=remaining,budget_ms=400)
            self.assertFalse(result.timed_out,(repeat,result))
            self.assertEqual(result.proven_actions,actions)

    def test_zero_budget_remains_fail_closed(self):
        state=make_game_state()
        self.assertTrue(actual.controllable_survival_actions(state,('BOMB',),remaining_steps=6,budget_ms=0).timed_out)

    def test_one_environment_board_is_shared_across_own_positions(self):
        from unittest.mock import patch
        state=make_game_state(position=(7,7))
        with patch.object(actual,'temporal_maps',wraps=actual.temporal_maps) as maps:
            result=actual.controllable_survival_actions(state,('UP','RIGHT','DOWN','LEFT','WAIT'),remaining_steps=3,budget_ms=60000)
        self.assertEqual(result.proven_actions,('UP','RIGHT','DOWN','LEFT','WAIT'))
        self.assertEqual(maps.call_count,1)

    def test_cached_board_queries_each_position_without_reusing_a_boolean(self):
        from unittest.mock import patch
        from agent_code.team_agent.opponent_transitions import OpponentTransitionScenario
        state=make_game_state(position=(1,1),bombs=[((7,7),0)])
        for pos in ((6,7),(8,7),(7,6),(7,8)):state['field'][pos]=-1
        trapped={**state,'self':('me',0,True,(7,7))}
        scenarios=(OpponentTransitionScenario(state,('WAIT',),(0,)),OpponentTransitionScenario(trapped,('BOMB',),(0,)))
        with patch.object(actual,'enumerate_opponent_transition_scenarios',return_value=scenarios):
            result=actual.controllable_survival_actions(state,('WAIT',),remaining_steps=3,budget_ms=60000)
        self.assertEqual(result.proven_actions,())
        self.assertEqual(result.first_failing_profile,('BOMB',))
        self.assertEqual(result.scenarios_evaluated,2)
