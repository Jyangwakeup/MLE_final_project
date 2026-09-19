"""Training-harness-only seat; policy selection happens before each round."""
def setup(self):
    if self.train:
        raise ValueError('Frozen opponent seats cannot train')
    self.frozen_delegate = None
    self.last_safety_diagnostic = None


def act(self, game_state):
    if self.frozen_delegate is None:
        raise ValueError('Frozen seat was not bound at the round boundary')
    action = self.frozen_delegate.act(game_state)
    self.last_safety_diagnostic = getattr(self.frozen_delegate.owner, 'last_safety_diagnostic', None)
    return action
