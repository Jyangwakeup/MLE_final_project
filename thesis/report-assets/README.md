# Task 1–4 报告结果资产

本目录是公开仓库中的**报告数据源**，而不是最终 PDF。所有数值必须能追溯到 `tables/*.csv` 或 `figures/data/*.csv` 中的行、其 `source_path` 和相应的冻结结果/核验报告。中文说明配合英文列名，便于直接导入论文表格工具。

正文的第一阅读入口是 `tables/table6_model_task_progression.csv`：它把四个主要模型族在 Task 1--4 的代表性冻结结果、停止点和证据限制放在同一矩阵中。完整审计入口仍是 [Task 1–4 结果 Notebook](task1_to_task4_results.ipynb)，用于读取明细 CSV、解释指标并展示四张图；历史逐局训练曲线见 [`notebooks/task1_results.ipynb`](../../notebooks/task1_results.ipynb)。

## 目录

- `plan/`：统一协议、阶段门槛和主张—证据追溯。
- `tables/`：五张报告表的契约、CSV 数据与 Markdown 渲染版。
- `figures/`：数据清单、绘图脚本与 PDF 成品。

## 证据状态

- `verified_raw`：可定位到原始 JSON/逐局或冻结统计。
- `verified_summary`：源运行产物不在仓库，但数值来自已核验的项目报告。
- `packaging_verified`：只验证包、路径或行为等价性，不能当作新的性能评估。
- `not_comparable`：真实结果，但协议/模型合同不同，不能作效应量比较。
- `unrun`：计划阶段没有运行；不是零值。

所有表图都不得把不同任务、不同对手、不同 checkpoint 或不同评估集合的均值汇成一个统计估计。写作时以本目录的 CSV 和 Notebook 为准；[`thesis/REPORT_OUTLINE.md`](../../thesis/REPORT_OUTLINE.md) 是团队现有的写作大纲。
