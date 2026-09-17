# Compact exact safety admission

Implementation starts at 2026-09-17 23:04:33 UTC from 3eb3ea4; deadline is 24 hours later. The candidate shares one exact world background, caches base temporal maps by bomb subset (at most 16), uses integer-grid reachability and viability, and streams exact ordered scenarios with internal budget checks. The full v9 masks, first counterexample and logical counts remain equivalent to a preserved independent reference. No process-wide GC change, opponent deletion, reward/network change or new dependency was made.

The explicit v5→v9 Task4 migration accepts the registered b0b9e7a parent only. Training projects the official post-action same-step observation onto the next decision step, keeping immutable snapshots and the original placement clock. Parent replay remains historical and is not retroactively certified.

Preflight: 4,096 synthetic proof states, 256 ordered-transition cases, 160 unique historical failure states; complete proof equality passed. Across 1,600 measured historical calls per implementation, old/new P95 were 116.687/31.913ms and maximum 326.143/71.473ms (neither timed out in this measurement). Complete act over 1,380 historical/nonpending synthetic measurements had P95/max 30.058/65.988ms. The four-player open-junction complete-act probe peaked at 83.006ms. All 471 unittest cases passed with one skip. Test/harness failures encountered during development are preserved alongside the final evidence.

The frozen manifest registers worlds24400–24499 after scanning 38,629 JSON files without read errors, plus paired retention worlds24000–24059 and diagnostic RNG roots22422/22411/22433. Six distinct physical cores are available; no single game or training process uses multiple threads. Full-act P95/max100/250ms and all original safety/retention gates apply to the candidate. First frozen failure is terminal, and the controller has no formal-training stage.

Run with the mle Python environment, OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1:
`python experiments/compact_admission.py --manifest experiments/compact_admission_manifest.json`

Results will be appended after this one frozen admission. Until all phases complete, training admission is unproven. Submitted weights and archive remain unchanged.
