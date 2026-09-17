"""Independent complete-proof and ordered-transition differential checks."""
from dataclasses import asdict
import unittest
import numpy as np
from agent_code.team_agent.controllable_survival import controllable_survival_actions as solve
from agent_code.team_agent.compact_survival import TransitionContext, BitGrid
from tests.reference_grid_survival import controllable_survival_actions as reference
from tests.reference_compact_transitions import enumerate_opponent_transition_scenarios, ACTIONS
from tests.test_danger import make_game_state
from tests.test_order_equivalence import native


def synthetic_states(count=4096):
    rng=np.random.default_rng(2026091801)
    for index in range(count):
        state=make_game_state(position=(1,1))
        shape=(7+index%11,7+(index//11)%11)
        field=np.full(shape,-1,dtype=int)
        field[1:-1,1:-1]=rng.choice([-1,0,1],size=(shape[0]-2,shape[1]-2),p=[.1,.65,.25])
        cells=np.argwhere(field==0)
        if len(cells)<8:field[1:-1,1:-1]=0;cells=np.argwhere(field==0)
        pos=[tuple(map(int,p)) for p in cells[rng.choice(len(cells),8,replace=False)]]
        state.update(field=field,explosion_map=np.zeros(shape,dtype=int),coins=[],
                     self=('me',0,bool(index%2),pos[0]),
                     others=[(str(i),0,bool(rng.integers(2)),pos[i+1]) for i in range(index%4)],
                     bombs=[(pos[4+i],int(rng.integers(4))) for i in range(index%3)])
        if index%7==0:state['bombs'].append((pos[0],index%4))
        if index%11==0 and state['others']:state['bombs'].append((pos[1],index%4))
        if index%5==0:state['explosion_map'][pos[7]]=1
        yield index,state


class CompactTests(unittest.TestCase):
    def test_4096_complete_proofs_and_state_immutability(self):
        for index,state in synthetic_states():
            before=native(state);kw=dict(remaining_steps=1+index%7,budget_ms=60000,consider_opponent_rearming=bool(index%2))
            expected=reference(state,ACTIONS,**kw);actual=solve(state,ACTIONS,**kw)
            self.assertFalse(expected.timed_out,index);self.assertFalse(actual.timed_out,index)
            self.assertEqual(asdict(expected),asdict(actual),index)
            self.assertEqual(before,native(state),index)

    def test_256_complete_ordered_compact_scenarios(self):
        for index,state in synthetic_states(256):
            context=TransitionContext(state,{})
            for action in ACTIONS:
                expected=enumerate_opponent_transition_scenarios(state,action,progress_world=True)
                actual=[]
                for own,others,bombs,alive,profile,order in context.scenarios(action,lambda:None):
                    actual.append(dict(game_state={**context.background,'self':own,'others':list(others),'bombs':list(bombs)},self_alive=alive,action_profile=profile,execution_order=order))
                self.assertEqual([native(asdict(r)) for r in expected],native(actual),(index,action))

    def test_shifts_do_not_wrap(self):
        grid=BitGrid(np.zeros((5,7),dtype=int))
        self.assertEqual(grid.expand(grid.bit((0,6))),grid.pack_positions([(0,6),(0,5),(1,6)]))
        self.assertEqual(grid.expand(grid.bit((4,0))),grid.pack_positions([(4,0),(3,0),(4,1)]))

    def test_generation_checks_budget_before_entire_product(self):
        s=make_game_state(position=(3,3));s['others']=[('a',0,True,(9,9)),('b',0,True,(11,11)),('c',0,True,(13,13))]
        stats={};ctx=TransitionContext(s,stats);calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==3:raise RuntimeError('deadline')
        with self.assertRaisesRegex(RuntimeError,'deadline'):list(ctx.scenarios('WAIT',check))
        self.assertLess(stats['action_phase_simulations'],10)
