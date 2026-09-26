# 报告表格数据契约

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| Table 1 — Model inventory | 覆盖全部模型家族与证据状态 | model family / representative | input, learner, reward, stages, status | `table1_model_inventory.csv`; manifests and reports | Team, submission review |
| Table 2 — Task 1/2 frozen results | 展示 Q/CNN 的成功与失败，不混合协议 | checkpoint-level candidate | coins, all-coins, crates, loops, safety, episodes, protocol | `table2_task1_task2.csv` | Ji, source-report audit |
| Table 3 — Task 3 paired validation | 支持 Task 3 的唯一正式 improvement claim | parent/child paired main validation | score, coins, crates, kills, first-place, 95% CI | `table3_task3_paired.csv` | Task 3 owner |
| Table 4 — Task 4 attempts | 保留尝试、停止原因和未运行状态 | campaign / arm | protocol, score, first-place, safety, latency, stop cause | `table4_task4_attempts.csv` | Task 4 owner |
| Table 5 — Die Hardest | 区分历史性能与 package verification | benchmark / package check | score, first-place, survival, timing, memory, limitation | `table5_die_hardest.csv` | Submission owner |
| Table 6 — Model × Task progression | 一眼展示主要模型族从 Task 1 到 Task 4 的已测结果与停止点 | model family / task stage | task-specific headline metric, sample size, stop/not-run status | `table6_model_task_progression.csv`; Tables 2--5 and source reports | Team, results synthesis |

数字的最小字段为 `evidence_status`, `source_path`, `protocol_id`, `episodes_or_worlds`。`not_reported` 与 `unrun` 必须保持文本值，禁止改成 0。带 CI 的效应量记录 `ci95_low` / `ci95_high`；没有 raw paired data 时填 `not_reported`，不计算新 CI。
