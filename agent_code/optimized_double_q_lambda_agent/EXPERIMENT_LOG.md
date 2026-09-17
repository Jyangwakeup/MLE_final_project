# Optimized Double Q(lambda) Experiment Log

## Task 1 protocol

| Item | Contract |
|---|---|
| Main agent | `optimized_double_q_lambda_agent` |
| Algorithm | Double Q(lambda), per-action tile coding, Watkins trace cut |
| Feature | `continuous-v2` (84 values) |
| Reward | `r7_safe_credit_potential` |
| Safety | `survival-mask-v1/off` for Task 1 |
| Training | seed 11, 100,000 action steps, CPU |
| Development evaluation | seeds 10000--10004, 20 rounds each |
| Confirmation | seeds 11000--11099, one round each, only after selection |
| Ranking | all-coins rate, mean coins, coins/100 steps, completion steps, loop rate |

## T1-E01 — submitted

| Candidate | Status | Training steps | Best snapshot | All coins | Mean coins | Notes |
|---|---|---:|---|---:|---:|---|
| Single-table Q baseline | planned | 100,000 | — | — | — | `discrete-q-v2 + r4_anti_oscillation` |
| Watkins Double Q(lambda) | planned | 100,000 | — | — | — | `continuous-v2 + r7_safe_credit_potential` |

Raw run directories, snapshot evaluation JSON, checkpoint SHA-256 and Slurm IDs are appended here after completion.
