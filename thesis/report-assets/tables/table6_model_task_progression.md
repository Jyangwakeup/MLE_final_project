# Table 6 — Model progression across Task 1–4

This is a display-oriented task matrix. Metrics change by task, so rows must not be read as a common-scale leaderboard. `not_run` means the model family stopped at the previous stage; it is not a zero.

| Model family | Task 1 | Task 2 | Task 3 | Task 4 | Furthest stage |
|---|---|---|---|---|---:|
| Continuous Double DQN | 50.0 mean coins | 7.25/9 coins; 0% suicide | 7.14 score; +1.23 vs parent | Die Hardest source checkpoint: 4.231 score; 50.2% first place | 4 |
| Rainbow Lite | 50/50 coins (development) | 8.45/9 coins; 60% all-coins | 7.80 score; 65% first place | 2.938 score; 57.1% long WAIT | 4 |
| Watkins Double Q($\lambda$) | 48.95 coins; 96% all-coins | 3.95/9 coins; 0% suicide | not run | not run | 2 |
| Distilled CNN Double DQN | 49.84 coins; 96% all-coins | 2.60/9 coins; 0% all-coins | not run | not run | 2 |

Full protocol qualifiers, sample sizes, evidence status, and source paths are retained in `table6_model_task_progression.csv`.
