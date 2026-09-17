# Rainbow-lite Agent

Dueling Double DQN with proportional prioritized replay and fixed 4-step
returns. The recommended training chain uses `continuous-v4` with explicit
action-conditional survival margins, bounded WAIT/loop history, and the shared
survival-mask interface.
