# Shared learning runtime

This package contains representation-independent training utilities used by the four new agents.
It does not compute features or choose strategic actions. It supplies strict checkpoint validation,
epsilon scheduling, pending-transition handling, Double-DQN targets, replay storage, and atomic
checkpoint writes.

Replay sampling and checkpointing use a columnar batch layout. This keeps the
same transition order, RNG state, values, capacity, and update schedule while
avoiding tens of thousands of individually serialized small tensors. The
loader remains compatible with the earlier per-transition checkpoint layout.

Board replay stores the eleven binary `board-v1` channels with `packbits` and stores the bomb-timer
channel as exact `uint8` quarter steps. A classic board therefore occupies 867 bytes per state.
The package is copied into a generated tournament archive by `experiments.package_agent`; the four
development Agent directories do not contain independent copies.
