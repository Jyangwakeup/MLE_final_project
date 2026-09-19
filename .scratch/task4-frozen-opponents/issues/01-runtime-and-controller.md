# Frozen seats, explicit migration and bounded C/S controller
Type: task
Status: resolved

Implement deterministic balanced historical scheduling, independent frozen seats,
strict identity, fixed 60k endpoints, per-neural safety audit, final paired controls.

## Comments
Implementation begins 2026-09-19T18:10:22+02:00. Base 182f516d, independent worktree.

## Answer
Implemented stable frozen seats, deterministic schedules, strict transfer/resume and bounded fixed-endpoint campaign. Full suite: 504 tests, one skip. Targeted frozen/runtime/controller suite: 15 tests. Actual 12-round continuous-versus-resumed training matched all 15,631 decisions, 4,800 Replay entries, 2,801 optimizer updates and full checkpoints; rule-to-historical boundary resume also matched. Evidence: experiments/results/task4_frozen_registration/.
