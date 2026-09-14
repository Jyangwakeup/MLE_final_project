import random

from agent_code.team_agent.feature_system.common import ACTIONS, build_context


def setup(self):
    self.rng = random.Random(0)


def act(self, game_state):
    legal = build_context(game_state).legal_mask.copy()
    legal[ACTIONS.index("BOMB")] = False
    return self.rng.choice([
        action for action, allowed in zip(ACTIONS, legal) if allowed
    ])
