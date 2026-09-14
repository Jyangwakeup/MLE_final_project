# Safety-explicit Continuous Double DQN

This Agent uses `continuous-v3`, a learned Double DQN policy, and an optional
versioned survival mask that only removes actions proven fatal within the exact
seven-step bomb/flame horizon. It does not encode a preferred strategic action.
