# Implement continuous-v5 and locked reward contracts

Type: task
Status: resolved

Implement the versioned feature, reward, Rainbow agent, experiment configs,
and regression tests without changing continuous-v4/r10 checkpoint behavior.

## Answer

Added continuous-v5 (140 dimensions), bounded bomb-objective history,
r16_no_safety_locked, r17_global_crate_bomb_discipline, a v5 Rainbow agent,
and focused regression coverage.

## Comments

The v5 specialization shares the tested Rainbow implementation and is intended
for one learning agent per experiment process, matching the experiment runner.
