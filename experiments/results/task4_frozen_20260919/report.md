# Task4 冻结历史对手对照报告

结论：未观察到收益（初筛未晋级）
源码：4e2b836d4b8482b220de5686ed4e25a08ca2f407
父权重 SHA-256：b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d
仅比较 Task4 固定终点；不代表旧课程协议的 Task4 合格，也未启动分代自博弈。
留出的历史策略与训练池同源，不称为独立未知对手。旧权重和原提交包不变。
初始化清空全部 Replay 与优化器；规则得分优先，历史对局只作迁移诊断。

| 臂 | 种子 | 实际动作 | 回合 | 历史曝光 | 得分 | 金币 | 击杀 | 独占/并列第一 | 自杀率 | 炸弹存活 | P95/最大ms | 失败项 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| C | 22 | 60055 | 158 | 0.000 | 4.0550 | 2.7550 | 0.2600 | 0.3250/0.1000 | 0.0000 | 1.0000 | 19.44/122.13 | [] |

C/22 权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s22_60000/resume/generation-00000158/learner.pt；SHA-256：15c16e161b27d8975c736946ad70be3da3387bc6b2df19bca7fcd6c5297980ab。
留出历史策略结果：{"episodes": 100, "mean_bombs": 38.16, "suicide_rate": 0.0, "bomb_survival_rate": 1.0, "zero_bomb_round_rate": 0.0, "invalid_action_rate": 0.00055, "act_p95_seconds": 0.02774003744125366, "act_max_seconds": 0.13106298446655273, "act_timeouts": 0, "act_skipped": 0, "robust_search_timeouts": 0, "robust_guarantee_losses": 0, "avoidable_escape_collapses": 0, "mean_score": 2.63, "mean_coins": 2.63, "mean_crates": 30.41, "mean_kills": 0.0, "exclusive_first_rate": 0.27, "tied_first_rate": 0.21, "first_place_rate": 0.48}
| C | 11 | 60083 | 162 | 0.000 | 4.2150 | 2.8400 | 0.2750 | 0.3400/0.1000 | 0.0000 | 1.0000 | 19.04/112.05 | [] |

C/11 权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s11_60000/resume/generation-00000162/learner.pt；SHA-256：92bb9ce9459b4d08158519bd869216bfdc02fee441a042b84d68f1d908ba8b8d。
留出历史策略结果：{"episodes": 100, "mean_bombs": 25.4, "suicide_rate": 0.0, "bomb_survival_rate": 1.0, "zero_bomb_round_rate": 0.0, "invalid_action_rate": 0.000375, "act_p95_seconds": 0.029835164546966518, "act_max_seconds": 0.13578414916992188, "act_timeouts": 0, "act_skipped": 0, "robust_search_timeouts": 0, "robust_guarantee_losses": 0, "avoidable_escape_collapses": 0, "mean_score": 2.77, "mean_coins": 2.77, "mean_crates": 31.78, "mean_kills": 0.0, "exclusive_first_rate": 0.3, "tied_first_rate": 0.21, "first_place_rate": 0.51}
| S | 22 | 60345 | 156 | 0.506 | 4.0350 | 2.6600 | 0.2750 | 0.3450/0.1300 | 0.0000 | 1.0000 | 19.30/155.22 | [] |

S/22 权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_S_s22_60000/resume/generation-00000156/learner.pt；SHA-256：2c69c65e79f1d9f6ba1beb82dd7f8c389d91f384f631e3df76c994475ac2317e。
留出历史策略结果：{"episodes": 100, "mean_bombs": 34.14, "suicide_rate": 0.0, "bomb_survival_rate": 1.0, "zero_bomb_round_rate": 0.0, "invalid_action_rate": 0.000225, "act_p95_seconds": 0.027315878868103025, "act_max_seconds": 0.13762354850769043, "act_timeouts": 0, "act_skipped": 0, "robust_search_timeouts": 0, "robust_guarantee_losses": 0, "avoidable_escape_collapses": 0, "mean_score": 2.64, "mean_coins": 2.64, "mean_crates": 29.59, "mean_kills": 0.0, "exclusive_first_rate": 0.24, "tied_first_rate": 0.2, "first_place_rate": 0.44}
| S | 11 | 60170 | 153 | 0.490 | 3.6100 | 2.6850 | 0.1850 | 0.2850/0.0850 | 0.0000 | 1.0000 | 19.79/129.63 | [] |

S/11 权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_S_s11_60000/resume/generation-00000153/learner.pt；SHA-256：d14df3c7914b6a2cc7376b50cd9a8d245b3a4086f6679ed787ab10f7c63dd8f1。
留出历史策略结果：{"episodes": 100, "mean_bombs": 29.24, "suicide_rate": 0.0, "bomb_survival_rate": 1.0, "zero_bomb_round_rate": 0.0, "invalid_action_rate": 0.0005791846087985697, "act_p95_seconds": 0.027933835983276367, "act_max_seconds": 0.16146421432495117, "act_timeouts": 0, "act_skipped": 0, "robust_search_timeouts": 0, "robust_guarantee_losses": 0, "avoidable_escape_collapses": 0, "mean_score": 2.66, "mean_coins": 2.51, "mean_crates": 30.53, "mean_kills": 0.03, "exclusive_first_rate": 0.28, "tied_first_rate": 0.11, "first_place_rate": 0.39}

实际动作、回合、冻结对手审计、全部中间结果见 status.json；复现命令见 logs/*.command.json。
控制器命令：`/export/data/sfan/miniforge3/envs/mle/bin/python experiments/task4_frozen_campaign.py --manifest /export/data/sfan/MLE_final_project_task4_frozen/experiments/task4_frozen_manifest.json`。
