# Compact exact safety admission

Implementation starts at 2026-09-17 23:04:33 UTC from 3eb3ea4; deadline is 24 hours later. The candidate shares one exact world background, caches base temporal maps by bomb subset (at most 16), uses integer-grid reachability and viability, and streams exact ordered scenarios with internal budget checks. The full v9 masks, first counterexample and logical counts remain equivalent to a preserved independent reference. No process-wide GC change, opponent deletion, reward/network change or new dependency was made.

The explicit v5→v9 Task4 migration accepts the registered b0b9e7a parent only. Training projects the official post-action same-step observation onto the next decision step, keeping immutable snapshots and the original placement clock. Parent replay remains historical and is not retroactively certified.

Preflight: 4,096 synthetic proof states, 256 ordered-transition cases, 160 unique historical failure states; complete proof equality passed. Across 1,600 measured historical calls per implementation, old/new P95 were 116.687/31.913ms and maximum 326.143/71.473ms (neither timed out in this measurement). Complete act over 1,380 historical/nonpending synthetic measurements had P95/max 30.058/65.988ms. The four-player open-junction complete-act probe peaked at 83.006ms. All 471 unittest cases passed with one skip. Test/harness failures encountered during development are preserved alongside the final evidence.

The frozen manifest registers worlds24400–24499 after scanning 38,629 JSON files without read errors, plus paired retention worlds24000–24059 and diagnostic RNG roots22422/22411/22433. Six distinct physical cores are available; no single game or training process uses multiple threads. Full-act P95/max100/250ms and all original safety/retention gates apply to the candidate. First frozen failure is terminal, and the controller has no formal-training stage.

Run with the mle Python environment, OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1:
`python experiments/compact_admission.py --manifest experiments/compact_admission_manifest.json`

Final outcome: **准入失败 (admission failed)**. The frozen attempt is terminal; no replacement candidate or formal training was launched. Submitted weights and archive remain unchanged.


## Frozen outcome and identity

The controller stopped at 2026-09-18T00:36:27Z, about 92 minutes after implementation began, inside the 24-hour budget. Source is `ae2b981b1d5dad6b7ab7654fe7127f678b112038`, branch `safety-compact-admission`, worktree `/export/data/sfan/MLE_final_project_safety_compact`. Subsequent report commits do not change this tested source identity.

- Manifest SHA-256: `38d843cad07bc3bf52cf7b6f1da98380c63fbbed6398eccbb6b5505a69a93436`.
- Candidate config SHA-256: `301f167685b330bc8eff3e569ace03e19bb12d2f483d56bad1346a6799e3ef2f`.
- Runtime source SHA-256: `5f3cde8415aecd1b1601d95ee39bcbc2ac07f8be922be7492e1533bd44ed98ca`.
- Parent `agent_code/double_dqn_continuous_v2_agent/task3_validated.pt`: `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d` (verified unchanged after termination).
- Submitted ZIP SHA-256 remains `b4f96841e573eed9d083b0f21dea9eb4b9941a0722dc850ac015327afa4cb6a4`.

Logical seed33, RNG22433, failed on round13/step88: `compact_act_max_exceeded`, complete act **262.603ms**, 12.603ms above the 250ms margin gate. Action UP completed a proof; search timeout, guarantee loss, avoidable escape collapse and fallback flags were false. Four physical actions each had 432 scenarios, all proved; logical state count was338,680. These logical counts are not measured allocations or simulations. This is a tail-latency admission failure, not evidence of a failed proof or an actual 400ms search timeout. The live failure did not instrument GC/CPU time, so its cause cannot be attributed to GC from the available evidence. No remeasurement, mechanism adjustment or retry was performed after this frozen failure.

The first12 rounds completed; round13 is partial, rounds14–20 were not started. Seed22/11 each completed20 rounds. No Task4-qualified weights were produced; diagnostic checkpoints must not continue into formal training.

## Completed coverage and performance

Independent reference comparison covered4096 valid synthetic proof states,256 ordered scenario cases, and160 historical states, including saved24025/24384 scenes. The offline reference had60s per proof; no partial proof was accepted. Full unittest:471 tests, one skip, OK. The skipped case is recorded in the full log; “all tests passed” means all executed tests passed, not that the skip ran. Historical measurements used CPU3, one warmup and10 alternating old/new repetitions per scene. Normal GC remained enabled.

| Measurement | Old P95 / max (ms) | Compact P95 / max (ms) |
|---|---:|---:|
| Historical complete search,1600 measured calls each |116.687 /326.143|31.913 /71.473|
| Additional500 worst-scene compact repeats, wall time | — |75.570 /90.623|
| Same500 repeats, CPU time | — |75.059 /90.344|
| Historical/nonpending synthetic complete act,1380 calls | — |30.058 /65.988|

The500 repeats had zero timeouts; maximum total observed GC time per call was22.117ms. Complete-act synthetic probes are a smaller performance corpus than the4096 proof-equivalence cases; pending-history semantics are additionally exercised through lifecycle tests and complete-world regressions. The open-junction complete-act probe max was83.006ms. Actual-work maxima over historical scenes were one shared world progression,16 temporal-map builds,1606 action-phase simulations,1260 distinct scenarios,420 viability-board builds,404 temporal-map cache hits and840 viability cache hits. Raw per-scene counts are in `preflight/work_counts.jsonl`. Hardware details and logical/physical core identities are archived; results do not establish speed on the course reference CPU.

Four known-world regressions passed (Task3:24003,24266; Task4:24025,24384). Their complete-act maxima were21.546,23.859,64.804,63.904ms respectively.

## Paired retention

Each task used60 parent and60 candidate worlds24000–24059 (360 games total). Parent and candidate share world/opponent RNG. Bootstrap intervals use10,000 paired world resamples, with no additional significance gate.

| Metric | Retention | Candidate − parent | Paired95% CI |
|---|---:|---:|---:|
|Task1 score|100.00%|0.000|[0.000, 0.000]|
|Task2 coins|100.00%|0.000|[0.000, 0.000]|
|Task2 crates|100.00%|0.000|[0.000, 0.000]|
|Task3 score|94.79%|-0.400|[-0.983, 0.167]|
|Task3 coins|98.60%|-0.067|[-0.283, 0.133]|
|Task3 crates|99.12%|-0.500|[-2.150, 1.050]|

All retention thresholds passed. Candidate full-act maxima for these stages were18.241,186.814,169.152ms respectively. Retained parent replay was not retroactively certified.

## Fresh engineering evaluation and diagnostics

Worlds24400–24499 were used once per task,400 completed games. No formal confirmation, main-validation or sealed20000–20099 data were consumed. P95 below is pooled complete-act timing. Task1 bomb-survival/zero-bomb thresholds are inapplicable; it placed zero bombs.

| Stage | Complete rounds | P95 / max(ms) | Suicide | Bomb survival | Zero-bomb rounds | Invalid actions | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
|fresh_task1|100|10.933 / 16.887|0.00%|0.00%|100.00%|0.000%|passed|
|fresh_task2|100|9.485 / 14.479|0.00%|100.00%|0.00%|0.000%|passed|
|fresh_task3|100|13.613 / 159.213|0.00%|100.00%|0.00%|0.040%|passed|
|fresh_task4|100|22.428 / 211.143|0.00%|100.00%|0.00%|0.284%|passed|
|diagnostic_22|20|24.895 / 66.058|0.00%|100.00%|0.00%|0.371%|passed|
|diagnostic_11|20|22.686 / 69.198|0.00%|100.00%|0.00%|0.246%|passed|
|diagnostic_33|12|26.339 / 262.603|0.00%|100.00%|0.00%|0.115%|failed; round13 partial|

Partial seed33 rates use completed episodes for episode statistics; timing includes the partial round and failing call. All observed search timeout, framework skip, decision timeout, guarantee loss and avoidable-collapse counters were zero. Raw audits found no responsibility or mask failures. Death summaries and saved states are retained; there were no self-deaths to explain. Task4's100-world engineering score was4.10, kills0.21 and first-place rate49%; these are descriptive, not Task4 qualification.

## Artifacts and reproduction

Archived controller state, exact commands, logs, paired comparisons, per-stage audits, metadata and all saved death/failure states are under `experiments/results/compact_admission_20260917/frozen/`. The new failure's20-state ring is `frozen/run_evidence/compact_ae2b981_diagnostic_33/task4_failure_states.pkl`; its exact summary is beside it. `raw_artifact_inventory.json` records6275 raw files with paths, sizes and SHA-256; full timings, replays and learner states remain in the named worktree's `runs/` directories. No raw run was removed. `final_audit.json` contains CPU identity, hashes, timings and diagnostic checkpoint hashes.

Diagnostic checkpoint evidence (never a qualified candidate):

- `runs/compact_ae2b981_diagnostic_11/checkpoints/final.pt` — SHA-256 `1c821752d0fc6d3879fde51b5f072fbc67662155c44e2925f7c951f5e02a10d8`.
- `runs/compact_ae2b981_diagnostic_22/checkpoints/final.pt` — SHA-256 `6e361e025d5351904cba98a402a0546c8def98e2d4e0afe093179e17e833f904`.
- `runs/compact_ae2b981_diagnostic_33/checkpoints/final.pt` — SHA-256 `46c26ae467f668fec65185779730f427e7b68f1108df3d9ff373f6c597ad2a46`.
- `runs/compact_ae2b981_diagnostic_33/resume/generation-00000012/learner.pt` — SHA-256 `46c26ae467f668fec65185779730f427e7b68f1108df3d9ff373f6c597ad2a46`.

For reference, the exact original launch was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/export/data/sfan/miniforge3/envs/mle/bin/python experiments/compact_admission.py \
  --manifest experiments/compact_admission_manifest.json
```

The manifest deadline and existing-directory guards intentionally prevent treating a rerun as this admission. Do not restart this terminal attempt. Individual original taskset/Python commands are archived as `controller/*_command.json`; performance reproduction must use the frozen source, matching environment, parent, CPU affinity and isolated output paths, as a separately identified diagnostic. Failure latency is not guaranteed to recur in a replay.

Correctness tests on the frozen source use the same single-thread environment:

```bash
/export/data/sfan/miniforge3/envs/mle/bin/python -m unittest tests.test_compact_survival tests.test_compact_lifecycle tests.test_task4_transfer tests.test_compact_admission
/export/data/sfan/miniforge3/envs/mle/bin/python -m unittest discover -s tests
```

The delivered implementation and evidence are local commits only. This attempt ended on its first frozen engineering failure; formal Task4 training remains blocked under this protocol. Any further profiling or new admission requires a separate follow-up scope.
