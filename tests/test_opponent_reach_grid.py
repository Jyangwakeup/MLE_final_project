"""Exact proof comparison against the preserved named-set implementation."""
from dataclasses import asdict
from pathlib import Path
import pickle
import unittest
from agent_code.team_agent import controllable_survival as actual
from tests import reference_named_reach_survival as reference


def failure_rows():
    return pickle.loads((Path(__file__).resolve().parents[1] /
        'experiments/results/safety_proven_movement_v9_20260917/failure/task4_failure_states.pkl').read_bytes())


class OpponentReachGridTests(unittest.TestCase):
    def test_complete_timeout_state_stays_within_existing_budget(self):
        state = failure_rows()[-1]['state']
        actions = ('RIGHT', 'LEFT', 'WAIT', 'BOMB')
        result = actual.controllable_survival_actions(
            state, actions, remaining_steps=7, budget_ms=400,
            consider_opponent_rearming=True)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.proven_actions, actions)

    def test_twenty_recorded_states_preserve_complete_proofs(self):
        for row in failure_rows():
            with self.subTest(step=row['state']['step']):
                arguments = dict(remaining_steps=7, budget_ms=60000,
                                 consider_opponent_rearming=True)
                actions = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
                expected = reference.controllable_survival_actions(row['state'], actions, **arguments)
                observed = actual.controllable_survival_actions(row['state'], actions, **arguments)
                self.assertFalse(expected.timed_out)
                self.assertEqual(asdict(expected), asdict(observed))

    def test_random_frontiers_match_named_set_reference(self):
        import numpy as np
        rng = np.random.default_rng(2026091721)
        for index in range(256):
            steps = 1 + index % 7
            shape = (5 + index % 4, 6 + index % 3)
            blocked = rng.random((steps + 1, *shape)) < .25
            before = blocked.copy()
            others = [(str(i if index % 11 else 0), 0, bool(rng.integers(2)),
                       (int(rng.integers(shape[0])), int(rng.integers(shape[1]))))
                      for i in range(index % 4)]
            for rearming in (False, True):
                named = {str(a[0]): {tuple(a[3])} for a in others}
                armed_names = {str(a[0]) for a in others if rearming or a[2]}
                reach = [{tuple(a[3]) for a in others}]
                armed = [{tuple(a[3]) for a in others if rearming or a[2]}]
                for t in range(1, steps + 1):
                    following = {}
                    for name, positions in named.items():
                        following[name] = set()
                        for x, y in positions:
                            for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
                                xx, yy = x + dx, y + dy
                                if 0 <= xx < shape[0] and 0 <= yy < shape[1] and not blocked[t, xx, yy]:
                                    following[name].add((xx, yy))
                    named = following
                    reach.append(set().union(*named.values()))
                    armed.append(set().union(*(p for n, p in named.items() if n in armed_names)))
                result = actual._opponent_reach_masks(blocked, others, steps, rearming)
                for grids, sets in zip(result, (reach, armed)):
                    for t, expected in enumerate(sets):
                        self.assertEqual({tuple(p) for p in np.argwhere(grids[t])}, expected,
                                         (index, rearming, t))
                np.testing.assert_array_equal(blocked, before)
