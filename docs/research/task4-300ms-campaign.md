# Task4 300ms new-protocol campaign

This is a new attempt from `f52f492`, not a reinterpretation of the failed250ms admission. Implementation started2026-09-18T11:17:18Z; the24-hour deadline is2026-09-19T11:17:18Z (13:17:18 Europe/Berlin). New-arm/evaluation cutoffs are05:17:18Z/10:17:18Z respectively on September19.

The compact agent/game runtime SHA remains `5f3cde8415aecd1b1601d95ee39bcbc2ac07f8be922be7492e1533bd44ed98ca`. Only protocol/control/audit/test/documentation code changes. The original v5 parent checkpoint SHA remains `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d`. Parent replay is historical, not retroactively certified.

Candidate P95/max100/300ms, reference250/480ms, search400ms are separate contracts. All reference evaluations explicitly use v5, candidates v9; comparing them measures the combined effect of changed safety behavior and learning. A/B differ only in replay fractions75%/50% (plus arm identity). Feature84/r7/n-step5/network/optimizer/exploration remain unchanged.

The42,128-file scan had no read errors. Mixed diagnostics use24500–24519, development24600–24659, confirmation24700–24799, main24800–24899. Engineering24400–24499, paired24000–24059 and old diagnostic seeds remain reusable engineering evidence, not independent validation. Sealed20000–20099 remain unused.

Reused proof, retention and400-game evidence is hash-bound to the manifest. The new seed33 failure ring has20 states (round13, steps69–88), all complete-reference comparisons passed. On core3 with normal GC,200 compact search measurements had P95/max22.117/30.074ms. These are search measurements, not a replacement for the original262.603ms complete-act failure. The new actual seed33 diagnostic verifies the recorded state/action prefix before continuing through20 rounds.

After full tests and source freeze, the background owner runs three20-round diagnostics from the original parent, then the full finite campaign. Formal training uses seeds22,11,33 independently initialized from that same parent; it never resumes diagnostic learners. First engineering failure ends the attempt. Only a three-seed confirmed arm and its unique100-world main-validation winner can qualify Task4. No progress claim here substitutes for controller results.

Launch from the clean frozen worktree with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/export/data/sfan/miniforge3/envs/mle/bin/python experiments/task4_background.py \
  --manifest experiments/task4_300ms_campaign.json
```

The run directory is `runs/task4_300ms_20260918_<source_commit7>/`. `background.json` records the owner PID, `result.json` records the current phase and identity, and `logs/` retains exact commands and worker output. The top-level detached log is `runs/task4_300ms_background.log`; launch PID is `runs/task4_300ms_background.pid`. Terminal archive/report will be `experiments/results/task4_300ms_20260918/terminal/`; the owner commits those records locally after termination. Old weights, submitted ZIP and previous attempt records are unchanged. No goal mode, push or packaging occurs.

Pre-launch verification: full unittest477 tests,233.796s, OK with one skip;16 protocol/controller tests passed. The complete failure-reference regression passed. Parent weight and submitted archive hashes were rechecked unchanged.
