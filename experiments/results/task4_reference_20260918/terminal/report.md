# Task4 300ms：验证失败

Source: `f3f17d756ffc5ebcb8b731f79fc0c0b7f6290d68`; manifest digest: `badcb498bec753ecb754c309f67bc61a0d8bb04d9b85085cd13eb588e0416c1d`.
Terminal status: `stopped_engineering_failure`; finished: 2026-09-18T21:31:41.242817+02:00; error: `Safety failure: /export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_confirmation_parent_t4_f3f17d7/task4_reference_20260918_A_badcb498_confirmation_parent_t4_f3f17d7_s25106/task4_safety_failure.json`.

The previous 250ms admission remains failed. Reused engineering evidence is not new validation.
The reference is the original v5 parent; candidate v9 gains include both controller changes and learning.
Candidate complete-act P95/max limits are100/300ms; reference limits250/480ms; search budget400ms.
Historical parent replay is retained without retroactive certification.
Under v3, only recognized v5 unproved-placement guarantee losses/escape collapses are reference observations; raw counters are unchanged. Candidate gates remain strict.

## Diagnostics

- Logical seed11: 20 rounds; timing `{'act_max_seconds': 0.05854225158691406, 'act_p95_seconds': 0.019392240047454823}`; run `/export/data/sfan/MLE_final_project_task4_300ms/runs/task4_300ms_20260918_A_d7594474_diag_s22411_c0020_93e5e59`.
- Logical seed22: 20 rounds; timing `{'act_max_seconds': 0.1644737720489502, 'act_p95_seconds': 0.02125486135482788}`; run `/export/data/sfan/MLE_final_project_task4_300ms/runs/task4_300ms_20260918_A_d7594474_diag_s22422_c0020_93e5e59`.
- Logical seed33: 20 rounds; timing `{'act_max_seconds': 0.16808748245239258, 'act_p95_seconds': 0.02179452180862425}`; run `/export/data/sfan/MLE_final_project_task4_300ms/runs/task4_300ms_20260918_A_d7594474_diag_s22433_c0020_93e5e59`.

## Development, confirmation and main validation

- ArmA seed22: selected rounds `50`.
  - c50: failed gates `[]`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s22_c0050_f3f17d7/checkpoints/final.pt`; SHA `47154063ea0cedec97c58046475c3ac7d010a3cc2cb8f07c719d83c634a8507d`.
- ArmA seed11: selected rounds `250`.
  - c50: failed gates `['task2_mean_coins_retention', 'task2_mean_crates_retention']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s11_c0050_f3f17d7/checkpoints/final.pt`; SHA `65cfb7148875815edb634acdb3e125fd6e7c51dfbe74d4d49e37c1c3a246d923`.
  - c100: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s11_c0100_f3f17d7/checkpoints/final.pt`; SHA `ae5dfaa094fe459e1be5ae51a346d952fd26251a7536730a7e77dbd61328515d`.
  - c150: failed gates `['task2_mean_coins_retention', 'task4_score_gain', 'task4_combat_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s11_c0150_f3f17d7/checkpoints/final.pt`; SHA `609424d1c33081ad2a00d6d169dd01668dceaf36269bf986bd1bd1138887e663`.
  - c200: failed gates `['task2_mean_coins_retention', 'task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s11_c0200_f3f17d7/checkpoints/final.pt`; SHA `da7439bab11ec08c8bd59a129d00d771ee9c917476931187517e00e6f467d412`.
  - c250: failed gates `[]`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s11_c0250_f3f17d7/checkpoints/final.pt`; SHA `41a774d7d8b2118193ff0e3d1c182d1486a4982d95104268bd97836d9c396498`.
- ArmA seed33: selected rounds `None`.
  - c50: failed gates `['task4_score_gain', 'task4_combat_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s33_c0050_f3f17d7/checkpoints/final.pt`; SHA `5fae5f9072da379f39b68caaef5a93cd5fe213b1453114594e407b7e450b5d6e`.
  - c100: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s33_c0100_f3f17d7/checkpoints/final.pt`; SHA `3de6ee91ee9e752548f9b41af24b2c668f66e20dc7d0263043fc7070989cbfdf`.
  - c150: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s33_c0150_f3f17d7/checkpoints/final.pt`; SHA `9eece7589ba644ea5cf33a12e949108c9de6f22d71ea89fd94bc3b8875bcfe5d`.
  - c200: failed gates `['task2_mean_coins_retention']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s33_c0200_f3f17d7/checkpoints/final.pt`; SHA `c68b243d17240df6f79d7778e4e5d0a7bcabf2125dfc6ebdd3c846e5c7089329`.
  - c250: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s33_c0250_f3f17d7/checkpoints/final.pt`; SHA `dcdb9b0f737478bf2a594491a5615c0e2791330f48688bc4613d3f1350af2ff5`.
  - c300: failed gates `['task3_mean_score_retention']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_A_badcb498_train_s33_c0300_f3f17d7/checkpoints/final.pt`; SHA `669673d7f6ceae9ac5138cb4083c9b968373aae15154b5efe616ad9f46facb17`.
- ArmB seed22: selected rounds `150`.
  - c50: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s22_c0050_f3f17d7/checkpoints/final.pt`; SHA `babea487b1d5fabf3439efcb1ebd8dfbd5d180013ac75df0d522c96cc3863ae9`.
  - c100: failed gates `['task4_score_gain', 'task4_combat_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s22_c0100_f3f17d7/checkpoints/final.pt`; SHA `df882af16e8a762fcb07b38e837afbcd9d437311274ad98e7e148374e03fcedc`.
  - c150: failed gates `[]`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s22_c0150_f3f17d7/checkpoints/final.pt`; SHA `220084f05caefcdd37e8c8577402d32c06eb4a22a52ed7bbbff8c646f9aaf9b3`.
- ArmB seed11: selected rounds `50`.
  - c50: failed gates `[]`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s11_c0050_f3f17d7/checkpoints/final.pt`; SHA `3acb9b5cbd8f4028055cd64d931b486dfa0ef382bbd6aab1f2115935ce775076`.
- ArmB seed33: selected rounds `200`.
  - c50: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s33_c0050_f3f17d7/checkpoints/final.pt`; SHA `22d17910ec5e6c7b1b5990defdcabff96ef99fe331c824746a529a1365cc9229`.
  - c100: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s33_c0100_f3f17d7/checkpoints/final.pt`; SHA `b6df35c9fa20cc0a058d909a331b09875d8440432f81f78fecc627acb447dfd0`.
  - c150: failed gates `['task4_score_gain']`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s33_c0150_f3f17d7/checkpoints/final.pt`; SHA `d3e3cfe20ec3e4c1d3d522639dbd2600407e0882274388234c6062b16d048f75`.
  - c200: failed gates `[]`; checkpoint `/export/data/sfan/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s33_c0200_f3f17d7/checkpoints/final.pt`; SHA `ae5cf37c7efaa9e7c6d2f41bbe0d1a3c8aa2a58c164ade9cf7cf1ee145686f7a`.

Full parent/child metrics and10,000-resample paired intervals are in `controller/evaluations`,
`controller/comparisons`, per-arm seed records and `controller/result.json`.
If a stage was not reached, it has no result; incomplete runs remain in the raw inventory.
Exact taskset/Python commands are retained in `controller/logs/*.command.json`.
All raw timing, replay and training state paths/SHA-256 are in `raw_artifact_inventory.json`.
Diagnostic weights cannot continue formal training. No archive was repackaged or pushed.
