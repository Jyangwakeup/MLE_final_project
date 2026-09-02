# Bomberman RL Final Project

This repository keeps the game framework separate from our agent code.  The
two implementations share feature extraction, rewards, evaluation, and
training utilities, so experiments differ only where intended.

## Layout

```text
agent_code/
  shared/                 # Stable, shared training pipeline
    config.py              # Common defaults and reproducible experiment config
    features.py            # game_state -> model input
    rewards.py             # game events -> reward
    memory.py              # Transition storage / replay buffer
    evaluation.py          # Run matches and aggregate win-rate statistics
    interfaces.py          # Common model contract
  model_a/                # First approach (for example: Q-learning)
    callbacks.py           # Environment-required setup() and act()
    train.py               # Environment-required training callbacks
    model.py               # Model A only
    config.py              # Model A hyperparameter overrides
    saved_model.*          # Generated; do not hand-edit
  model_b/                # Second approach (for example: DQN)
    callbacks.py
    train.py
    model.py               # Model B only
    config.py
    saved_model.*
  opponents/              # Frozen checkpoints used for self-play
  experiments/            # Named YAML/JSON configs and experiment notes
  results/                # Generated metrics, plots, and match summaries
```

`model_a` and `model_b` must expose the same interface defined in
`shared/interfaces.py`. This makes the environment, training loop, and
evaluation code reusable across models.

## Team Rule

Treat `shared/` as a reviewed common pipeline. Model experiments should first
change only `model_a/model.py` or `model_b/model.py` and that model's config.
When changing features or rewards, save the experiment configuration so the
comparison remains fair.
