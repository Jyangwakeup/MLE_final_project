# MLE Bomberman Final Report Outline

> 本大纲与 `main_zh.tex` 和 `report-assets/plan/experiment-protocol.md` 同步。实验采用“问题—证据—诊断—解决—验证—结论—决策”的叙事。中文内容冻结前不分配主笔，主要作者栏保留为空白占位；冻结后、英文翻译前必须统一填写真实姓名。

## 1. Introduction `[主要作者：____]`

### 1.1 问题与研究动机 `[主要作者：____]`

- 稀疏反馈、延迟炸弹信用、目标重选和对手不确定性。
- 训练 reward 不等于冻结游戏表现；得分、安全和 0.5 秒 CPU 时限共同决定选模。

### 1.2 研究问题 `[主要作者：____]`

1. 不同价值学习方法能否学会金币导航，并迁移到炸箱和对战？
2. 等待、循环、自杀和目标丢失分别能由哪些特征、奖励和生存约束改善？
3. 三条路线的主要瓶颈是什么，哪些改进有效、无效或证据不足？
4. 在得分、生存、推理时间和可复现性之间，为什么最终选择 B33？

## 2. Background `[主要作者：____]`

- Bomberman 环境与 Task 1–4。
- Q-learning、Double Q、DQN、Double DQN、dueling、PER、n-step 与蒸馏。
- 离散、tile-coded 连续、MLP 连续和 17 通道空间表示。
- 奖励塑形、课程学习与安全动作准入。

## 3. Project Planning `[主要作者：____]`

### 3.1 团队结构与协作 `[主要作者：____]`

- 路线一：Continuous Double DQN Task 1–4 与 survival mask。
- 路线二：Rainbow Lite Task 1–4、feature/reward 与延迟信用。
- 路线三：Q-learning/CNN Task 1–2 与跨学习器迁移。
- 三条路线共享代码、指标、失败案例和选模规则，不描述为三个独立项目。

### 3.2 门槛与证据管理 `[主要作者：____]`

- 训练 → 开发选模 → 独立确认 → 主验证 → 打包。
- 区分 training seed 与 environment seed；未运行记为 `unrun`，不可记为零。
- 冻结 checkpoint、关闭探索、单线程 CPU、配置/checkpoint 哈希和 paired bootstrap。

## 4. Methods `[主要作者：____]`

### 4.1 共享状态与特征 `[主要作者：____]`

- 离散键、84 维 tile coding、70/78/84 维连续表示、117 维 phase 表示、17 通道 CNN。
- Feature v1–v11 按导航、历史/循环、危险、资源、对手和目标可达性归类。
- 版本号仅作复现标识，不代表单调性能顺序。

### 4.2 共享奖励与生存准入 `[主要作者：____]`

- Reward r1–r22 按正式事件、势函数、WAIT/循环、炸弹信用、生存和对手事件归类。
- Feature、reward、mask 是独立版本轴。
- 区分 physical legal action、finite-horizon survivable action、veto-only mask 和 fallback。
- Survival-mask v1–v9 由具体反例驱动；保留 27155 限制。

### 4.3 模型与统一评估 `[主要作者：____]`

- 表格 Q/Double Q/Watkins Double Q(λ)。
- Continuous/CNN Double DQN 与蒸馏 CNN。
- Rainbow Lite = dueling + proportional PER + fixed 4-step，不称完整 Rainbow。
- 指标和可比性遵循 `report-assets/plan/experiment-protocol.md`。

## 5. Training `[主要作者：____]`

- Task 1 导航；Task 2 炸箱与隐藏金币；Task 3 弱对手；Task 4 三个规则对手。
- 当前任务、旧任务保留、安全和运行时共同决定晋级。
- 严格 resume 与 policy-only transfer 分开，合同变化时显式迁移或重训。
- 表格/MLP 主要使用 CPU，CNN 训练使用 GPU，最终评估 CPU-only。
- 失败记录包括 CNN 输入 bug、WAIT/循环、Task 2 自杀、Task 3 计数缺陷及 Task 4 超时/中断。

## 6. Experiments and Results `[主要作者：____]`

每个实验段落回答：问题、证据、诊断、解决方案、验证、结果、有限答案和下一决策。

### 6.0 总体实验逻辑与比较边界 `[主要作者：____]`

- Task 1–4 依次隔离导航、炸箱逃生、弱对手竞争和强对手压力；后续任务必须回测早期能力。
- 三条路线共享 Feature、Reward、Survival Mask 与冻结评估，但针对表示、信用和安全提出不同假设。
- 课程 agents 作为兼容性、课程或压力测试对象；缺少同协议冻结结果时不建立 baseline 数值排名。
- `verified_raw` 支持正式差值，`verified_summary` 描述历史结果，`not_comparable` 仅作诊断，`unrun` 明确为未运行。

### 6.1 可比性与证据状态 `[主要作者：____]`

- 先区分 `verified_raw`、`verified_summary`、`not_comparable`、`unrun` 与打包验证。
- 说明场景、对手、checkpoint、seed 单位和局数一致时才作直接比较。
- 图表展示课程能力，不建立跨协议统一排行榜。

### 6.2 Task 1：全模型导航能力 `[主要作者：____]`

- 路径 CNN r3/r5 展示表示与训练版本的改善过程。
- Double Q(λ)：48.95 coins、96% all-coins；Distilled CNN：49.84 coins、96% all-coins。
- 结论限定为多种复杂度的价值学习器均可获得导航能力。

### 6.3 Task 2：安全炸箱与目标重选 `[主要作者：____]`

- Q r20：3.95/9 coins、53.45 crates、0% suicide、75% ping-pong。
- CNN D02：2.60/9 coins、44.15 crates、0% suicide。
- Continuous 与 Rainbow 结果用于说明安全准入、奖励和长期规划的不同作用。
- 共同瓶颈是箱后目标重选；安全改善不等于任务完成。

### 6.4 Task 3：弱对手与能力保持 `[主要作者：____]`

- seed22/c150 的 100-world 配对：5.91 → 7.14，差 1.23，CI [0.27, 2.14]。
- 金币与炸箱改善，击杀 0.45 → 0.44，不称击杀改善。
- 其他训练 seed 只作为确认链；环境世界不能写成独立训练。

### 6.5 Task 4：强对手、专项实验与工程约束 `[主要作者：____]`

- E1/E2/E3、Frozen C/S、未运行 score campaign 分开报告。
- Rainbow c1200 的 1,000 局诊断揭示长 WAIT；B33 历史 benchmark 保持为异协议证据。
- 打包等价验证不等于 1,000-world 性能复测；B33 不是 Task 4 qualified model。

### 6.6 Feature、Reward、Mask 消融与失败分析 `[主要作者：____]`

- 多因素版本不作单因素因果归因。
- veto-only mask 抑制可证明致命动作，但 27155 反例保留。
- WAIT、ping-pong、目标丢失和 lifecycle 指标 bug 分别归因，不合并为一个失败标签。

### 6.7 跨路线综合与 B33 选择 `[主要作者：____]`

- 同时权衡能力证据、保持性、跨 seed 稳定、CPU 推理和可复现打包。
- B33 的历史 1,000-world 记录与六局 package equivalence 分开。
- 明确缺少 B33–Rainbow 统一 Task 4 横评，因此选择是风险受控的交付决策，不是最优性证明。

## 7. Conclusion `[主要作者：____]`

- 多种价值学习器可完成 Task 1，但迁移深度不同。
- 历史/reward 改变停滞，mask 减少自杀，二者都不自动解决长期规划。
- Task 3 得分改善；Task 4 增量方案缺少稳定收益。
- B33 是证据完整性与工程约束下的最终选择。

## Figures and Tables

- Figure 1：课程训练与证据流程。
- Figure 2：价值推理和 veto-only 动作准入。
- Figure 3：Task 1–4 冻结能力证据（分面指标，不作跨面板排名）。
- Figure 4：Task 4 得分、安全和完整动作延迟记录。
- Table 1：Task 1–4 代表性协议与比较边界。
- Table 2：Task 3 父子配对。
- Table 3：最终候选证据边界与选择。
- Appendix Table：九类模型及证据路径。

## Writing Workflow Gates

1. **Evidence ready**：三个完整证据输入包合计覆盖方法、流程、结果、失败、局限和证据路径（已满足）。
2. **Chinese draft complete**：将三条路线统一重写为完整中文论文，不按成员拼接；除封面外不出现个人认领。
3. **Chinese content frozen**：摘要、七章、图表、引用、主张边界和局限完整，不保留实质性 TODO。
4. **Authorship assigned**：按技术匹配和约 4,000 English words/人填写所有主笔栏及 Project Planning 贡献表。
5. **English reviewed**：完成学术化翻译、分章节主笔复核和全局一致性编辑。
6. **Submission synchronized**：审核通过的 `main_en.tex` 与最终 `main.tex` 完全一致。

## Submission Checks

- 摘要和 Background 使用完整正文，不保留写作提示或示意图占位。
- 报告开头列明最终 Agent 的 NumPy 与 PyTorch 版本。
- 中文冻结前所有主笔栏保持空白；冻结后、英文翻译前为所有章节和小节填写真实主要作者。
- 所有数字可追溯到 `report-assets/tables/*.csv` 或其 `source_path`。
- `not_reported`、`unrun` 不转换为 0。
- 报告 PDF 不上传公开仓库。
- 团队成员理解、核验并以自己的表达修订 AI 辅助初稿。
