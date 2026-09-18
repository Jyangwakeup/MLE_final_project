# Task4 same-v9 baseline: 验证失败

Source: `7a1f04f4ffba05b42de6492dd3fa9fbd3cf8bae0`; manifest: `b2e6687efc95a3252e8ed006e5e24ac7004522d65b10a45b3226c6199d38ccf2`.
Terminal status: `stopped_development_failure`; error: ``.
Both parent and child use v9 and 100/300ms act limits; no reference exceptions.
Original v5 parent checkpoint and old terminal outcomes remain unchanged.
Parent Replay comes from historical v5 behavior and is not retroactively certified.
Retention: {"mode": "reused_engineering_evidence", "checks": {"1": {"mean_score": 1.0}, "2": {"mean_coins": 1.0, "mean_crates": 1.0}, "3": {"mean_score": 0.947939262472885, "mean_coins": 0.986013986013986, "mean_crates": 0.9912229373902867}}, "evidence": "experiments/results/task4_v9_reference_20260918/retention_evidence.json"}
Baseline certified: True
Engineering baseline: {"task1": {"episodes": 100.0, "mean_score": 50.0, "mean_coins": 50.0, "mean_crates": 0.0, "mean_kills": 0.0, "first_place_rate": 1.0, "suicide_rate": 0.0, "bomb_survival_rate": 0.0, "zero_bomb_round_rate": 1.0, "invalid_action_rate": 0.0, "act_p95_seconds": 0.009937763214111328, "act_max_seconds": 0.017258167266845703, "act_timeouts": 0.0, "act_skipped": 0.0, "avoidable_escape_collapses": 0.0, "robust_guarantee_losses": 0.0, "robust_search_timeouts": 0.0, "mean_bombs": 0.0}, "task2": {"episodes": 100.0, "mean_score": 7.43, "mean_coins": 7.43, "mean_crates": 101.48, "mean_kills": 0.0, "first_place_rate": 1.0, "suicide_rate": 0.0, "bomb_survival_rate": 1.0, "zero_bomb_round_rate": 0.0, "invalid_action_rate": 0.0, "act_p95_seconds": 0.008689165115356445, "act_max_seconds": 0.014677286148071289, "act_timeouts": 0.0, "act_skipped": 0.0, "avoidable_escape_collapses": 0.0, "robust_guarantee_losses": 0.0, "robust_search_timeouts": 0.0, "mean_bombs": 46.4}, "task3": {"episodes": 100.0, "mean_score": 7.78, "mean_coins": 5.03, "mean_crates": 59.68, "mean_kills": 0.55, "first_place_rate": 0.65, "suicide_rate": 0.0, "bomb_survival_rate": 1.0, "zero_bomb_round_rate": 0.0, "invalid_action_rate": 0.000575, "act_p95_seconds": 0.012274026870727539, "act_max_seconds": 0.020652294158935547, "act_timeouts": 0.0, "act_skipped": 0.0, "avoidable_escape_collapses": 0.0, "robust_guarantee_losses": 0.0, "robust_search_timeouts": 0.0, "mean_bombs": 42.56}, "task4": {"episodes": 100.0, "mean_score": 4.59, "mean_coins": 2.94, "mean_crates": 35.26, "mean_kills": 0.33, "first_place_rate": 0.52, "suicide_rate": 0.0, "bomb_survival_rate": 1.0, "zero_bomb_round_rate": 0.0, "invalid_action_rate": 0.0029040106742013972, "act_p95_seconds": 0.01911473274230957, "act_max_seconds": 0.14578628540039062, "act_timeouts": 0.0, "act_skipped": 0.0, "avoidable_escape_collapses": 0.0, "robust_guarantee_losses": 0.0, "robust_search_timeouts": 0.0, "mean_bombs": 41.2}}

## Diagnostics

- Logical seed22: 20 rounds; timing `{'act_p95_seconds': 0.019526624679565424, 'act_max_seconds': 0.1985793113708496}`; run `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_diag_s22422_c0020_7a1f04f`.
- Logical seed11: 20 rounds; timing `{'act_p95_seconds': 0.02031583786010742, 'act_max_seconds': 0.06550002098083496}`; run `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_diag_s22411_c0020_7a1f04f`.
- Logical seed33: 20 rounds; timing `{'act_p95_seconds': 0.02286684513092041, 'act_max_seconds': 0.17434978485107422}`; run `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_diag_s22433_c0020_7a1f04f`.

## Development, confirmation and main validation

- ArmB seed22: selected rounds `None`.
  - c50: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_train_s22_c0050_7a1f04f/checkpoints/final.pt`; SHA `2cab89e085be4320c4f4e65b1e99c8ade59fedb6fd904fb86cfdcaf1aff156c8`.
  - c100: failed gates `['task4_score_gain', 'task4_combat_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_train_s22_c0100_7a1f04f/checkpoints/final.pt`; SHA `48a523f3a89e6feaef2d3962d346ab5eda49267a863d968724afd062dec4b999`.
  - c150: failed gates `['task4_score_gain', 'task4_combat_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_train_s22_c0150_7a1f04f/checkpoints/final.pt`; SHA `2b225cf07027bf905032a1c87951730ec2d363854e0d4ce5ac5750759ac8b0ad`.
  - c200: failed gates `['task2_mean_coins_retention', 'task4_score_gain', 'task4_combat_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_train_s22_c0200_7a1f04f/checkpoints/final.pt`; SHA `8d45d5811807b667fd999b3ec4d2f9ec5e735be03899daeeb08fcfbf9fefbb24`.
  - c250: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_train_s22_c0250_7a1f04f/checkpoints/final.pt`; SHA `4f07a068d8f75ce62b47573787724c73dd7a183542b54df9c0c70a2fdefcc0f1`.
  - c300: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_v9_reference/runs/task4_v9_reference_20260918_B_b2e6687e_train_s22_c0300_7a1f04f/checkpoints/final.pt`; SHA `3f4e040c18ee12fe7109575cf8a4f77e3a8d9644903cfc03ebc29ccd0e9ce8f4`.

Full parent/child metrics and10,000-resample paired intervals are in `controller/evaluations`,
`controller/comparisons`, per-arm seed records and `controller/result.json`.
If a stage was not reached, it has no result; incomplete runs remain in the raw inventory.
Exact taskset/Python commands are retained in `controller/logs/*.command.json`.
All raw timing, replay and training state paths/SHA-256 are in `raw_artifact_inventory.json`.
Diagnostic weights cannot continue formal training. No archive was repackaged or pushed.
