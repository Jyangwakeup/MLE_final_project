# Task4 300ms：验证失败

Source: `93e5e5990451ab525c33e9d3346cb8112978b29a`; manifest digest: `d7594474b595e914daec7f1ffa3f92f75cbdd0841575b6c93d3def128bdf0211`.
Terminal status: `stopped_engineering_failure`; finished: 2026-09-18T13:49:48.576056+02:00; error: `Safety failure: /export/data/sfan/MLE_final_project_task4_300ms/runs/task4_300ms_20260918_A_d7594474_development_parent_t4_93e5e59/task4_300ms_20260918_A_d7594474_development_parent_t4_93e5e59_s24614/task4_safety_failure.json`.

The previous 250ms admission remains failed. Reused engineering evidence is not new validation.
The reference is the original v5 parent; candidate v9 gains include both controller changes and learning.
Candidate complete-act P95/max limits are100/300ms; reference limits250/480ms; search budget400ms.
Historical parent replay is retained without retroactive certification.

## Diagnostics

- Logical seed22: 20 rounds; timing `{'act_p95_seconds': 0.02125486135482788, 'act_max_seconds': 0.1644737720489502}`; run `/export/data/sfan/MLE_final_project_task4_300ms/runs/task4_300ms_20260918_A_d7594474_diag_s22422_c0020_93e5e59`.
- Logical seed11: 20 rounds; timing `{'act_p95_seconds': 0.019392240047454823, 'act_max_seconds': 0.05854225158691406}`; run `/export/data/sfan/MLE_final_project_task4_300ms/runs/task4_300ms_20260918_A_d7594474_diag_s22411_c0020_93e5e59`.
- Logical seed33: 20 rounds; timing `{'act_p95_seconds': 0.02179452180862425, 'act_max_seconds': 0.16808748245239258}`; run `/export/data/sfan/MLE_final_project_task4_300ms/runs/task4_300ms_20260918_A_d7594474_diag_s22433_c0020_93e5e59`.

## Development, confirmation and main validation


Full parent/child metrics and10,000-resample paired intervals are in `controller/evaluations`,
`controller/comparisons`, per-arm seed records and `controller/result.json`.
If a stage was not reached, it has no result; incomplete runs remain in the raw inventory.
Exact taskset/Python commands are retained in `controller/logs/*.command.json`.
All raw timing, replay and training state paths/SHA-256 are in `raw_artifact_inventory.json`.
Diagnostic weights cannot continue formal training. No archive was repackaged or pushed.
