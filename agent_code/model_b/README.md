# Model B

Keep the environment entry files (`callbacks.py`, `train.py`) thin. They
should delegate state encoding, reward calculation, and learning to the
shared pipeline, while this directory owns Model B's implementation and
hyperparameters.
