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

## T1-E01 — submitted (2026-09-17)

| Candidate | Slurm job | Status | Training steps | Best snapshot | All coins | Mean coins | Notes |
|---|---|---|---:|---|---:|---:|---|
| Single-table Q baseline | `472859` | failed before training | 100,000 | — | — | — | Compute node lacks login venv's `/usr/local/bin/python3.10`; no transitions ran. |
| Watkins Double Q(lambda) | `472860` | failed before training | 100,000 | — | — | — | Same environment issue; no transitions ran. |

Each job saves a live checkpoint and immutable snapshots every 25k action steps, then
runs the five-seed frozen development evaluation and writes
`best_task1_selection.json`. Raw run directories, snapshot evaluation JSON, checkpoint
SHA-256 and completed metrics are appended here after completion.

The replacement submissions use the local, ignored `.venv-compute` environment,
created from `/home/students/ji/.local/bin/python3.10` with the CPU dependencies.

## Environment retry record

| Attempt | Jobs | Result | Cause / resolution |
|---|---|---|---|
| 1 | `472859`, `472860` | failed before training | Login venv links to `/usr/local/bin/python3.10`, absent on compute. |
| 2 | `472864`, `472865` | failed before training | Compute venv lacked CPU PyTorch, required by the shared experiment import path. |
| 3 | `472871`, `472872` | running | `.venv-compute`: Python 3.10.19, NumPy 2.2.6, Pygame 2.6.1, PyTorch 2.5.1+cpu; verified on `compute`. |
