# Task 3 计数修正后的冻结验证结果

**结论：Task 3 通过本项目预注册门槛，可以进入 Task 4。Task 4 尚未启动。**

完成时间：2026-09-17T14:55:17.287185+02:00。旧实验的主验证失败记录保持不变；本结论来自修正后的新验证。

## 修正与正确性

修正 physical_fallback 下把物理合法 WAIT 误当成安全替代动作的证书条件，诊断版本为 escape-collapse-v2。没有修改策略、权重、奖励、特征、网络或门槛，没有重训或重新选模。世界 19489 的完整重放中，全部 1012 个动作和回合指标不变，错误塌缩计数从 1 变为 0；真实塌缩回归用例仍可检出。完整 unittest 共 287 项，跳过 1 项，其余通过。

## 验证划分与身份

三个原检查点各在 21000–21099 的 100 个世界确认；原唯一候选 seed22/c150 在 21100–21199 的 100 个世界主验证。每阶段均评估 Task 1/2/3、父子配对、关闭探索、单线程 CPU、10,000 次配对 bootstrap。未使用封存的 20000–20099。

- 源码：`7cc6f0249c0ecb09a539614965c6ee3f758b26e3`
- Manifest SHA-256：`b68f9df3d40f3962b36442b4a265a1980c6d830d78b4e6eacfbb1a7a23020881`
- 配置 SHA-256：`a41bc49dc26ca5be609c52dca2ea5c4981089d9dc748bef1a9ed446cbb2a8869`
- Manifest：`experiments/task3_counter_validation.json`
- 配置：`experiments/configs/task3_lifecycle_A.json`
- 运行目录：`/export/data/sfan/MLE_final_project_task3_v5/runs/task3_counter_validation_20260917_7cc6f02`

已重新核对所有六份父子权重的 SHA-256、源码与配置身份、100 局样本量、bootstrap 配对数，并从冻结摘要重算所有门槛，结果一致。

## 冻结结果

| 阶段 | 训练种子 | Task 3 得分 | 得分增量 | 击杀增量 | 第一名增量 | 失败门槛 |
|---|---:|---:|---:|---:|---:|---|
| confirmation | 11 | 6.84 | +1.92 | +0.15 | +23 pp | 无 |
| confirmation | 22 | 7.11 | +1.23 | +0.02 | +16 pp | 无 |
| confirmation | 33 | 7.31 | +2.56 | +0.26 | +23 pp | 无 |
| main_validation | 22 | 7.14 | +1.23 | -0.01 | +12 pp | 无 |

主验证得分 5.91 → 7.14，第一名 41% → 53%。击杀 0.45 → 0.44，因此不能声称击杀提高；战斗 OR 门槛由第一名提升 12 个百分点满足。得分增量 95% bootstrap CI 为 [0.27, 2.14]，击杀增量 CI 为 [-0.16, 0.14]。不额外添加事后显著性门槛。

### 保留、安全及 CPU 门槛

| 阶段/种子 | 门槛 | 实际 | 阈值 |
|---|---|---:|---:|
| confirmation/11 | task1_act_max | 0.024247169 | <= 0.48 |
| confirmation/11 | task1_act_p95 | 0.0093589044 | <= 0.25 |
| confirmation/11 | task1_act_skipped | 0 | <= 0.0 |
| confirmation/11 | task1_act_timeouts | 0 | <= 0.0 |
| confirmation/11 | task1_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/11 | task1_invalid_action_rate | 0 | <= 0.01 |
| confirmation/11 | task1_mean_score_retention | 1 | >= 0.9 |
| confirmation/11 | task1_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/11 | task1_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/11 | task2_act_max | 0.028777599 | <= 0.48 |
| confirmation/11 | task2_act_p95 | 0.0085664845 | <= 0.25 |
| confirmation/11 | task2_act_skipped | 0 | <= 0.0 |
| confirmation/11 | task2_act_timeouts | 0 | <= 0.0 |
| confirmation/11 | task2_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/11 | task2_bomb_survival_rate | 1 | >= 0.95 |
| confirmation/11 | task2_invalid_action_rate | 0 | <= 0.01 |
| confirmation/11 | task2_mean_coins_retention | 1.0232198 | >= 0.9 |
| confirmation/11 | task2_mean_crates_retention | 1.0260799 | >= 0.9 |
| confirmation/11 | task2_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/11 | task2_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/11 | task2_suicide_rate | 0 | <= 0.05 |
| confirmation/11 | task3_act_max | 0.2929275 | <= 0.48 |
| confirmation/11 | task3_act_p95 | 0.04790225 | <= 0.25 |
| confirmation/11 | task3_act_skipped | 0 | <= 0.0 |
| confirmation/11 | task3_act_timeouts | 0 | <= 0.0 |
| confirmation/11 | task3_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/11 | task3_bomb_survival_rate | 1 | >= 0.95 |
| confirmation/11 | task3_invalid_action_rate | 0.00045 | <= 0.01 |
| confirmation/11 | task3_mean_coins_retention | 1.3690852 | >= 0.9 |
| confirmation/11 | task3_mean_crates_retention | 1.3568099 | >= 0.9 |
| confirmation/11 | task3_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/11 | task3_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/11 | task3_score_gain | 1.92 | >= 0.5 |
| confirmation/11 | task3_suicide_rate | 0 | <= 0.05 |
| confirmation/11 | task3_zero_bomb_round_rate | 0 | <= 0.1 |
| confirmation/22 | task1_act_max | 0.01590848 | <= 0.48 |
| confirmation/22 | task1_act_p95 | 0.0083093739 | <= 0.25 |
| confirmation/22 | task1_act_skipped | 0 | <= 0.0 |
| confirmation/22 | task1_act_timeouts | 0 | <= 0.0 |
| confirmation/22 | task1_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/22 | task1_invalid_action_rate | 0 | <= 0.01 |
| confirmation/22 | task1_mean_score_retention | 1 | >= 0.9 |
| confirmation/22 | task1_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/22 | task1_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/22 | task2_act_max | 0.014818907 | <= 0.48 |
| confirmation/22 | task2_act_p95 | 0.0084808373 | <= 0.25 |
| confirmation/22 | task2_act_skipped | 0 | <= 0.0 |
| confirmation/22 | task2_act_timeouts | 0 | <= 0.0 |
| confirmation/22 | task2_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/22 | task2_bomb_survival_rate | 1 | >= 0.95 |
| confirmation/22 | task2_invalid_action_rate | 0 | <= 0.01 |
| confirmation/22 | task2_mean_coins_retention | 1.1035503 | >= 0.9 |
| confirmation/22 | task2_mean_crates_retention | 1.0941562 | >= 0.9 |
| confirmation/22 | task2_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/22 | task2_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/22 | task2_suicide_rate | 0 | <= 0.05 |
| confirmation/22 | task3_act_max | 0.35435843 | <= 0.48 |
| confirmation/22 | task3_act_p95 | 0.050662775 | <= 0.25 |
| confirmation/22 | task3_act_skipped | 0 | <= 0.0 |
| confirmation/22 | task3_act_timeouts | 0 | <= 0.0 |
| confirmation/22 | task3_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/22 | task3_bomb_survival_rate | 0.999751 | >= 0.95 |
| confirmation/22 | task3_invalid_action_rate | 0.0004812929 | <= 0.01 |
| confirmation/22 | task3_mean_coins_retention | 1.3029491 | >= 0.9 |
| confirmation/22 | task3_mean_crates_retention | 1.3113732 | >= 0.9 |
| confirmation/22 | task3_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/22 | task3_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/22 | task3_score_gain | 1.23 | >= 0.5 |
| confirmation/22 | task3_suicide_rate | 0 | <= 0.05 |
| confirmation/22 | task3_zero_bomb_round_rate | 0 | <= 0.1 |
| confirmation/33 | task1_act_max | 0.013546467 | <= 0.48 |
| confirmation/33 | task1_act_p95 | 0.0092683458 | <= 0.25 |
| confirmation/33 | task1_act_skipped | 0 | <= 0.0 |
| confirmation/33 | task1_act_timeouts | 0 | <= 0.0 |
| confirmation/33 | task1_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/33 | task1_invalid_action_rate | 0 | <= 0.01 |
| confirmation/33 | task1_mean_score_retention | 1 | >= 0.9 |
| confirmation/33 | task1_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/33 | task1_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/33 | task2_act_max | 0.024934769 | <= 0.48 |
| confirmation/33 | task2_act_p95 | 0.008498342 | <= 0.25 |
| confirmation/33 | task2_act_skipped | 0 | <= 0.0 |
| confirmation/33 | task2_act_timeouts | 0 | <= 0.0 |
| confirmation/33 | task2_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/33 | task2_bomb_survival_rate | 1 | >= 0.95 |
| confirmation/33 | task2_invalid_action_rate | 0 | <= 0.01 |
| confirmation/33 | task2_mean_coins_retention | 1.0914928 | >= 0.9 |
| confirmation/33 | task2_mean_crates_retention | 1.1017283 | >= 0.9 |
| confirmation/33 | task2_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/33 | task2_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/33 | task2_suicide_rate | 0 | <= 0.05 |
| confirmation/33 | task3_act_max | 0.37964296 | <= 0.48 |
| confirmation/33 | task3_act_p95 | 0.053328147 | <= 0.25 |
| confirmation/33 | task3_act_skipped | 0 | <= 0.0 |
| confirmation/33 | task3_act_timeouts | 0 | <= 0.0 |
| confirmation/33 | task3_avoidable_escape_collapses | 0 | <= 0.0 |
| confirmation/33 | task3_bomb_survival_rate | 1 | >= 0.95 |
| confirmation/33 | task3_invalid_action_rate | 0.0004840641 | <= 0.01 |
| confirmation/33 | task3_mean_coins_retention | 1.39375 | >= 0.9 |
| confirmation/33 | task3_mean_crates_retention | 1.3070478 | >= 0.9 |
| confirmation/33 | task3_robust_guarantee_losses | 0 | <= 0.0 |
| confirmation/33 | task3_robust_search_timeouts | 0 | <= 0.0 |
| confirmation/33 | task3_score_gain | 2.56 | >= 0.5 |
| confirmation/33 | task3_suicide_rate | 0 | <= 0.05 |
| confirmation/33 | task3_zero_bomb_round_rate | 0 | <= 0.1 |
| main_validation/22 | task1_act_max | 0.010562658 | <= 0.48 |
| main_validation/22 | task1_act_p95 | 0.0076956773 | <= 0.25 |
| main_validation/22 | task1_act_skipped | 0 | <= 0.0 |
| main_validation/22 | task1_act_timeouts | 0 | <= 0.0 |
| main_validation/22 | task1_avoidable_escape_collapses | 0 | <= 0.0 |
| main_validation/22 | task1_invalid_action_rate | 0 | <= 0.01 |
| main_validation/22 | task1_mean_score_retention | 1 | >= 0.9 |
| main_validation/22 | task1_robust_guarantee_losses | 0 | <= 0.0 |
| main_validation/22 | task1_robust_search_timeouts | 0 | <= 0.0 |
| main_validation/22 | task2_act_max | 0.013463259 | <= 0.48 |
| main_validation/22 | task2_act_p95 | 0.0081179094 | <= 0.25 |
| main_validation/22 | task2_act_skipped | 0 | <= 0.0 |
| main_validation/22 | task2_act_timeouts | 0 | <= 0.0 |
| main_validation/22 | task2_avoidable_escape_collapses | 0 | <= 0.0 |
| main_validation/22 | task2_bomb_survival_rate | 1 | >= 0.95 |
| main_validation/22 | task2_invalid_action_rate | 0 | <= 0.01 |
| main_validation/22 | task2_mean_coins_retention | 1.0430416 | >= 0.9 |
| main_validation/22 | task2_mean_crates_retention | 1.0614266 | >= 0.9 |
| main_validation/22 | task2_robust_guarantee_losses | 0 | <= 0.0 |
| main_validation/22 | task2_robust_search_timeouts | 0 | <= 0.0 |
| main_validation/22 | task2_suicide_rate | 0 | <= 0.05 |
| main_validation/22 | task3_act_max | 0.27773404 | <= 0.48 |
| main_validation/22 | task3_act_p95 | 0.045922928 | <= 0.25 |
| main_validation/22 | task3_act_skipped | 0 | <= 0.0 |
| main_validation/22 | task3_act_timeouts | 0 | <= 0.0 |
| main_validation/22 | task3_avoidable_escape_collapses | 0 | <= 0.0 |
| main_validation/22 | task3_bomb_survival_rate | 1 | >= 0.95 |
| main_validation/22 | task3_invalid_action_rate | 0.0004778672 | <= 0.01 |
| main_validation/22 | task3_mean_coins_retention | 1.3497268 | >= 0.9 |
| main_validation/22 | task3_mean_crates_retention | 1.3469621 | >= 0.9 |
| main_validation/22 | task3_robust_guarantee_losses | 0 | <= 0.0 |
| main_validation/22 | task3_robust_search_timeouts | 0 | <= 0.0 |
| main_validation/22 | task3_score_gain | 1.23 | >= 0.5 |
| main_validation/22 | task3_suicide_rate | 0 | <= 0.05 |
| main_validation/22 | task3_zero_bomb_round_rate | 0 | <= 0.1 |

### 父子配对差值与区间

以下各区间均使用预定的 10,000 次 bootstrap；exclusive_win 是独占第一，tied_first 是并列第一，不能把其中一个区间称为总第一名比例的区间。

| 阶段/种子 | 任务 | 指标 | 父 | 子 | 差值 | 95% CI |
|---|---|---|---:|---:|---:|---|
| confirmation/11 | task1 | score | 50 | 50 | +0 | [0, 0] |
| confirmation/11 | task1 | coins | 50 | 50 | +0 | [0, 0] |
| confirmation/11 | task1 | round_steps | 131.48 | 132.39 | +0.91 | [-1.05, 2.89] |
| confirmation/11 | task2 | score | 6.46 | 6.61 | +0.15 | [-0.28, 0.59] |
| confirmation/11 | task2 | coins | 6.46 | 6.61 | +0.15 | [-0.29, 0.59] |
| confirmation/11 | task2 | crates | 85.89 | 88.13 | +2.24 | [-1.69, 6.07] |
| confirmation/11 | task2 | suicides | 0 | 0 | +0 | [0, 0] |
| confirmation/11 | task2 | survived | 1 | 1 | +0 | [0, 0] |
| confirmation/11 | task2 | survival_steps | 400 | 400 | +0 | [0, 0] |
| confirmation/11 | task3 | score | 4.92 | 6.84 | +1.92 | [1.06, 2.8] |
| confirmation/11 | task3 | coins | 3.17 | 4.34 | +1.17 | [0.81, 1.52] |
| confirmation/11 | task3 | crates | 38.62 | 52.4 | +13.78 | [10.67, 16.8] |
| confirmation/11 | task3 | kills | 0.35 | 0.5 | +0.15 | [0, 0.3] |
| confirmation/11 | task3 | suicides | 0 | 0 | +0 | [0, 0] |
| confirmation/11 | task3 | survived | 1 | 1 | +0 | [0, 0] |
| confirmation/11 | task3 | survival_steps | 400 | 400 | +0 | [0, 0] |
| confirmation/11 | task3 | exclusive_win | 0.24 | 0.52 | +0.28 | [0.17, 0.39] |
| confirmation/11 | task3 | tied_first | 0.06 | 0.01 | -0.05 | [-0.1, 0] |
| confirmation/22 | task1 | score | 50 | 50 | +0 | [0, 0] |
| confirmation/22 | task1 | coins | 50 | 50 | +0 | [0, 0] |
| confirmation/22 | task1 | round_steps | 129.2 | 127.76 | -1.44 | [-3, 0.07] |
| confirmation/22 | task2 | score | 6.76 | 7.46 | +0.7 | [0.29, 1.13] |
| confirmation/22 | task2 | coins | 6.76 | 7.46 | +0.7 | [0.28, 1.12] |
| confirmation/22 | task2 | crates | 91.55 | 100.17 | +8.62 | [3.87, 13.3] |
| confirmation/22 | task2 | suicides | 0 | 0 | +0 | [0, 0] |
| confirmation/22 | task2 | survived | 1 | 1 | +0 | [0, 0] |
| confirmation/22 | task2 | survival_steps | 400 | 400 | +0 | [0, 0] |
| confirmation/22 | task3 | score | 5.88 | 7.11 | +1.23 | [0.16, 2.28] |
| confirmation/22 | task3 | coins | 3.73 | 4.86 | +1.13 | [0.71, 1.55] |
| confirmation/22 | task3 | crates | 44.93 | 58.92 | +13.99 | [10.72, 17.16] |
| confirmation/22 | task3 | kills | 0.43 | 0.45 | +0.02 | [-0.16, 0.2] |
| confirmation/22 | task3 | suicides | 0 | 0 | +0 | [0, 0] |
| confirmation/22 | task3 | survived | 1 | 0.99 | -0.01 | [-0.03, 0] |
| confirmation/22 | task3 | survival_steps | 400 | 394.77 | -5.23 | [-10.87, -1.06] |
| confirmation/22 | task3 | exclusive_win | 0.35 | 0.47 | +0.12 | [-0.01, 0.25] |
| confirmation/22 | task3 | tied_first | 0.04 | 0.08 | +0.04 | [-0.02, 0.1] |
| confirmation/33 | task1 | score | 50 | 50 | +0 | [0, 0] |
| confirmation/33 | task1 | coins | 50 | 50 | +0 | [0, 0] |
| confirmation/33 | task1 | round_steps | 128.1 | 130.49 | +2.39 | [0.88, 3.88] |
| confirmation/33 | task2 | score | 6.23 | 6.8 | +0.57 | [0.09, 1.05] |
| confirmation/33 | task2 | coins | 6.23 | 6.8 | +0.57 | [0.1, 1.04] |
| confirmation/33 | task2 | crates | 86.21 | 94.98 | +8.77 | [3.85, 13.53] |
| confirmation/33 | task2 | suicides | 0 | 0 | +0 | [0, 0] |
| confirmation/33 | task2 | survived | 1 | 1 | +0 | [0, 0] |
| confirmation/33 | task2 | survival_steps | 400 | 400 | +0 | [0, 0] |
| confirmation/33 | task3 | score | 4.75 | 7.31 | +2.56 | [1.59, 3.54] |
| confirmation/33 | task3 | coins | 3.2 | 4.46 | +1.26 | [0.82, 1.7] |
| confirmation/33 | task3 | crates | 40.58 | 53.04 | +12.46 | [8.95, 16.12] |
| confirmation/33 | task3 | kills | 0.31 | 0.57 | +0.26 | [0.1, 0.42] |
| confirmation/33 | task3 | suicides | 0 | 0 | +0 | [0, 0] |
| confirmation/33 | task3 | survived | 1 | 1 | +0 | [0, 0] |
| confirmation/33 | task3 | survival_steps | 399.74 | 392.51 | -7.23 | [-12.92, -2.45] |
| confirmation/33 | task3 | exclusive_win | 0.32 | 0.51 | +0.19 | [0.06, 0.32] |
| confirmation/33 | task3 | tied_first | 0.01 | 0.05 | +0.04 | [0, 0.09] |
| main_validation/22 | task1 | score | 50 | 50 | +0 | [0, 0] |
| main_validation/22 | task1 | coins | 50 | 50 | +0 | [0, 0] |
| main_validation/22 | task1 | round_steps | 126.66 | 126.29 | -0.37 | [-1.86, 1.12] |
| main_validation/22 | task2 | score | 6.97 | 7.27 | +0.3 | [-0.16, 0.78] |
| main_validation/22 | task2 | coins | 6.97 | 7.27 | +0.3 | [-0.17, 0.77] |
| main_validation/22 | task2 | crates | 94.91 | 100.74 | +5.83 | [1.58, 10.03] |
| main_validation/22 | task2 | suicides | 0 | 0 | +0 | [0, 0] |
| main_validation/22 | task2 | survived | 1 | 1 | +0 | [0, 0] |
| main_validation/22 | task2 | survival_steps | 400 | 400 | +0 | [0, 0] |
| main_validation/22 | task3 | score | 5.91 | 7.14 | +1.23 | [0.27, 2.14] |
| main_validation/22 | task3 | coins | 3.66 | 4.94 | +1.28 | [0.87, 1.7] |
| main_validation/22 | task3 | crates | 43.78 | 58.97 | +15.19 | [12.17, 18.16] |
| main_validation/22 | task3 | kills | 0.45 | 0.44 | -0.01 | [-0.16, 0.14] |
| main_validation/22 | task3 | suicides | 0 | 0 | +0 | [0, 0] |
| main_validation/22 | task3 | survived | 1 | 1 | +0 | [0, 0] |
| main_validation/22 | task3 | survival_steps | 400 | 397.6 | -2.4 | [-6.46, 0] |
| main_validation/22 | task3 | exclusive_win | 0.36 | 0.45 | +0.09 | [-0.04, 0.22] |
| main_validation/22 | task3 | tied_first | 0.05 | 0.08 | +0.03 | [-0.03, 0.1] |

## 冻结权重与复现

- seed11：`/export/data/sfan/MLE_final_project_task3_v5/runs/task3_lifecycle_20260917_A_2b402220_s11_c0050_15d4721/checkpoints/final.pt`；SHA-256 `1e7e5bdce8bbc576e4e461a1e8c9bb09407b6799703140057aaff6b418466d47`。
- seed22：`/export/data/sfan/MLE_final_project_task3_v5/runs/task3_lifecycle_20260917_A_2b402220_s22_c0150_15d4721/checkpoints/final.pt`；SHA-256 `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d`。
- seed33：`/export/data/sfan/MLE_final_project_task3_v5/runs/task3_lifecycle_20260917_A_2b402220_s33_c0050_15d4721/checkpoints/final.pt`；SHA-256 `c3dd7f556e99f3998a7db550d4b6ed7cf2490bfb0e1d449091ccc0d73450f7a8`。

唯一交付候选为 seed22/c150，必须与上述修正源码和配置配套使用。原开发过程仅 A 臂，seed11/22/33 训练 150/250/150 局，选择 c50/c150/c50；P/R/PR 未运行。开发结果及旧确认、旧主验证见 [原报告](task3-lifecycle-frozen-results.md)。机器可读证据见 `experiments/results/task3_counter_validation_20260917.json`。

原始启动命令（在上述源码提交、清洁 worktree 和原登记权重路径下执行；运行目录需不存在，已完成产物应保留）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /export/data/sfan/miniforge3/envs/mle/bin/python -u experiments/task3_counter_validation.py --manifest experiments/task3_counter_validation.json
```

该协议包含原始截止时间，过期后控制器会拒绝新运行。将来重新验证需另行登记新协议和预算，不能覆盖本次结果或伪称同一运行续接。

## Task 4 判断与局限

Task 3 的阶段晋级条件已满足，建议冻结 seed22/c150 作为 Task 4 起点，先建立对强对手的冻结基线并记录堵路、死胡同及对手放弹导致的死亡，再确定训练变更。这次修正只消除错误计数，没有改善已经被对手堵死时的决策能力；普通移动对未来对手威胁的预防仍是下一阶段的研究重点。

Task 4 强对手表现、官方机器兼容性、打包与提交均未由本次验证证明。保留父 Replay 用于隔离修复变量，不等于证明旧 Task 2 数据完全不受历史生命周期问题影响。
