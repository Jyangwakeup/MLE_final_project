# Fix Task 1 tile representation

Type: task
Status: claimed

Retrain Expected SARSA(lambda) and Double Q(lambda) from a cold Task 1 start
with per-action feature projection and ternary coin-distance direction coding.
Keep `r7_safe_credit_potential` unchanged to measure the representation change.

## Comments

- 2026-09-15: Work started after the original checkpoints scored 41.8 and 39.7
  mean coins and exhibited long-WAIT-loop rates of 100% and 75% respectively.
