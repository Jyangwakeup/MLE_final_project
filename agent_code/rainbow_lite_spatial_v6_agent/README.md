# Spatial Rainbow-lite V6 experiment

An experimental dueling Double DQN with the existing 160-value continuous-v6
decision features and a 12×17×17 board. The board goes through two small
convolutions; the vector goes through the original 128×2 MLP. Their embeddings
join before the value and advantage heads. PER, four-step returns, r20 rewards,
and survival-mask-v5 are unchanged from the configured Rainbow-lite workflow.

The replay stores the eleven binary board planes with `packbits`, the bomb timer
as exact `uint8` quarters, and the vector as `float32`. The policy sees decoded
`float32` data. This is an **experimental agent**, with no shipped `final.pt`.
The original V6 checkpoint cannot be loaded directly because the feature and
network contracts changed.

## Warm start from a V6 checkpoint

The migration copies the V6 vector MLP and dueling heads. Its fusion layer
initially ignores the new board branch, so Q values match the parent before
learning. Optimizer, replay, target-network history, and training progress are
reset by `--init-from-checkpoint`.

```bash
.venv/bin/python experiments/migrate_rainbow_v6_to_spatial.py \
  /path/to/v6/checkpoints/final.pt /path/to/spatial-warmstart.pt

.venv/bin/python -m experiments.run \
  --config experiments/configs/rainbow_lite_spatial_v6_r20_maskv5_task3.json \
  --mode train --task 3 --agent rainbow_lite_spatial_v6_agent \
  --seed 11 --init-from-checkpoint /path/to/spatial-warmstart.pt \
  --run-id spatial_v6_r20_maskv5_s11_t3
```

Compare a frozen checkpoint against the original V6 on the same seeds,
opponents, and Task 3 protocol. Check score, kills, suicide, bomb survival,
and complete CPU `act` time. The V5 safety search already has a long tail;
the added feature extraction and CNN must stay within the official 0.5 s
decision budget. Do not treat a one-round smoke as performance evidence.
