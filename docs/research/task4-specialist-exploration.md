# Task4 specialist exploration

This protocol supersedes curriculum retention requirements only within the new
specialist experiment. It does not change any archived experiment or submission.

| Arm | Replay | Distillation | Reward | gamma |
|---|---|---|---|---|
| E1 | Task4 only | none | existing r7 | .95 |
| E2 | Task4 only | none | coin +1, kill +5, others zero | .95 |
| E3 | Task4 only | none | same as E2 | .99 |

Each starts the same registered Task3 policy with fresh Adam at 1e-4, empty
Replay (20,000 capacity), batch64 and warmup2000. Target copies policy. Epsilon
runs .20 to .05 across 100,000 stage actions. Every 20,000 actions, the completed
round creates an immutable checkpoint; overshoot is recorded. Seeds22/11 run
all arms; only the winning arm runs33. No diagnostic checkpoint feeds training.

The manifest records a scan of all worktrees: engineering worlds25800–25819,
development25900–25999, final26000–26199. Diagnostic learning streams are registered
separately and use the usual training world/opponent derivation. Engineering
world25800 was used once for the pre-freeze evaluation smoke test; other engineering
worlds remain reserved and are not represented as completed evidence. 20000–20099 remains sealed.

Safety remains exact v9 with 400ms search, complete-act P95<=100ms and max<=300ms.
Suicide<=5%, bomb survival>=95%, invalid<=1%. Zero-bomb rate and crate totals are
diagnostics only. Every death is retained. Unexplained suicide and engineering
failure stop the campaign; ordinary checkpoint behavior failures eliminate only
that checkpoint. Parent uses original weights with the same v9 runtime.

The implementation budget starts 2026-09-19 08:48:16 Europe/Berlin: no new arm
after18h, final selection by20h, no new evaluation after23h, hard stop24h.

## Reward attribution limitation

`environment.py` credits a bomb owner even if already dead. Dead agents no longer
receive per-step game event callbacks, but `round_ended()` still receives their
accumulated events at round end. The pure-score reward counts all coin/kill events
passed to a callback, including multiple kills and death in the same frame.
Posthumous credit can therefore land on the last pre-death transition rather
than the bomb-placement action. Existing transition de-duplication is preserved;
we do not claim logged training reward sums necessarily equal final score or
that delayed kill credit has correct temporal attribution. Final comparisons use
framework scores, not shaped reward.

## Execution and artifacts

From the independent worktree, with the mle Python environment and all BLAS/OpenMP
thread counts set to1:

```bash
python -m unittest discover -s tests
python experiments/task4_exploration_campaign.py --manifest experiments/task4_exploration_manifest.json
```

The controller creates `runs/task4_exploration_20260919/status.json`, `report.md`,
immutable checkpoint references, paired bootstrap results and exact command JSON
files. Only identity-matching infrastructure interruption accepts `--resume`;
terminal failure cannot resume. All retries retain previous output directories.
Final improvement requires safe evaluation and positive score difference; report
uncertainty and any first-place-rate tradeoff. Do not label it curriculum-qualified.
