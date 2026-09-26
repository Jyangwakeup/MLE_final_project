# 图表数据清单与图注

| Figure | Data files | Script | Outputs | Evidence / boundary |
|---|---|---|---|---|
| Figure 1 | protocol and traceability documents | scripts/figure_01_curriculum_pipeline.py | output/figure_01_curriculum_pipeline.pdf | 流程图；实线/虚线区分已记录与计划/未完整运行。 |
| Figure 2 | submission manifest and verification JSON | scripts/figure_02_inference_pipeline.py | output/figure_02_inference_pipeline.pdf | 架构图；mask 是准入过滤，不是规则动作排序。 |
| Figure 3 | data/capability_progression.csv | scripts/figure_03_capability_progression.py | output/figure_03_capability_progression.pdf | 分面、独立纵轴；禁止当作跨任务连续曲线。 |
| Figure 4 | data/task4_tradeoff.csv | scripts/figure_04_task4_tradeoff.py | output/figure_04_task4_tradeoff.pdf | 区分冻结尝试、历史 benchmark 与仅打包检查。 |

## 建议图注

- **Figure 1.** 课程训练、冻结选模与提交验证的证据流程。实线表示本仓库有相应结果记录的路径，虚线表示计划门槛或未完成阶段。
- **Figure 2.** `die_hardest` 的推理与动作准入流程。网络输出 Q 值，物理合法性与生存模块仅移除不准入动作。
- **Figure 3.** 按任务分面的冻结能力观测。每个面板使用不同任务指标和协议，因此不应跨面板解释柱高差为同一效应量。
- **Figure 4.** Task 4 专项评估记录的分数、P95 推理延迟与安全注记。历史 B33 benchmark 与 rename/package equivalence 不被作为同一次性能评估。

生成命令：`python3 thesis/report-assets/figures/scripts/render_figures.py`。脚本只读取同目录 CSV 和内置流程文本。论文引用的最终图片统一为从 SVG 源文件导出的矢量 PDF；PNG 仅作为预览或历史产物保留。
