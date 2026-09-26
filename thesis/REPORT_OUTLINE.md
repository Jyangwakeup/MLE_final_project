# MLE Bomberman Final Report Outline

> 本大纲与 `main.tex` 和 `report-assets/plan/experiment-protocol.md` 同步。实验采用“问题—证据—诊断—解决—验证—结论—决策”的叙事。主要作者暂保留待定，提交前必须替换为真实姓名。

## 1. Introduction

### 1.1 问题与研究动机 `[主要作者：待定]`

- 稀疏反馈、延迟炸弹信用、目标重选和对手不确定性。
- 训练 reward 不等于冻结游戏表现；得分、安全和 0.5 秒 CPU 时限共同决定选模。

### 1.2 研究问题 `[主要作者：待定]`

1. 不同价值学习方法能否学会金币导航，并迁移到炸箱和对战？
2. 等待、循环、自杀和目标丢失分别能由哪些特征、奖励和生存约束改善？
3. 三条路线的主要瓶颈是什么，哪些改进有效、无效或证据不足？
4. 在得分、生存、推理时间和可复现性之间，为什么最终选择 B33？

## 2. Background

- Bomberman 环境与 Task 1–4。
- Q-learning、Double Q、DQN、Double DQN、dueling、PER、n-step 与蒸馏。
- 离散、tile-coded 连续、MLP 连续和 17 通道空间表示。
- 奖励塑形、课程学习与安全动作准入。

## 3. Project Planning

### 3.1 团队结构与协作 `[主要作者：待定]`

- 路线一：Continuous Double DQN Task 1–4 与 survival mask。
- 路线二：Rainbow Lite Task 1–4、feature/reward 与延迟信用。
- 路线三：Q-learning/CNN Task 1–2 与跨学习器迁移。
- 三条路线共享代码、指标、失败案例和选模规则，不描述为三个独立项目。

### 3.2 门槛与证据管理 `[主要作者：待定]`

- 训练 → 开发选模 → 独立确认 → 主验证 → 打包。
- 区分 training seed 与 environment seed；未运行记为 `unrun`，不可记为零。
- 冻结 checkpoint、关闭探索、单线程 CPU、配置/checkpoint 哈希和 paired bootstrap。

## 4. Methods

### 4.1 共享状态与特征 `[主要作者：待定]`

- 离散键、84 维 tile coding、70/78/84 维连续表示、117 维 phase 表示、17 通道 CNN。
- Feature v1–v11 按导航、历史/循环、危险、资源、对手和目标可达性归类。
- 版本号仅作复现标识，不代表单调性能顺序。

### 4.2 共享奖励与生存准入 `[主要作者：待定]`

- Reward r1–r22 按正式事件、势函数、WAIT/循环、炸弹信用、生存和对手事件归类。
- Feature、reward、mask 是独立版本轴。
- 区分 physical legal action、finite-horizon survivable action、veto-only mask 和 fallback。
- Survival-mask v1–v9 由具体反例驱动；保留 27155 限制。

### 4.3 模型与统一评估 `[主要作者：待定]`

- 表格 Q/Double Q/Watkins Double Q(λ)。
- Continuous/CNN Double DQN 与蒸馏 CNN。
- Rainbow Lite = dueling + proportional PER + fixed 4-step，不称完整 Rainbow。
- 指标和可比性遵循 `report-assets/plan/experiment-protocol.md`。

## 5. Training

- Task 1 导航；Task 2 炸箱与隐藏金币；Task 3 弱对手；Task 4 三个规则对手。
- 当前任务、旧任务保留、安全和运行时共同决定晋级。
- 严格 resume 与 policy-only transfer 分开，合同变化时显式迁移或重训。
- 表格/MLP 主要使用 CPU，CNN 训练使用 GPU，最终评估 CPU-only。
- 失败记录包括 CNN 输入 bug、WAIT/循环、Task 2 自杀、Task 3 计数缺陷及 Task 4 超时/中断。

## 6. Experiments and Results

每个实验段落回答：问题、证据、诊断、解决方案、验证、结果、有限答案和下一决策。

### 6.0 总体实验逻辑与比较边界 `[主要作者：待定]`

- Task 1–4 依次隔离导航、炸箱逃生、弱对手竞争和强对手压力；后续任务必须回测早期能力。
- 三条路线共享 Feature、Reward、Survival Mask 与冻结评估，但针对表示、信用和安全提出不同假设。
- 课程 agents 作为兼容性、课程或压力测试对象；缺少同协议冻结结果时不建立 baseline 数值排名。
- `verified_raw` 支持正式差值，`verified_summary` 描述历史结果，`not_comparable` 仅作诊断，`unrun` 明确为未运行。

### 6.1 Continuous Double DQN：从导航到可控生存 `[主要作者：待定]`

#### 6.1.1 Task 1 → Task 2：为什么会炸箱仍会自杀？

- 早期候选：5.45 coins、76.65 crates、30% suicide。
- 解决：survival mask 用于行为、推理和 bootstrap。
- 主验证：7.25 coins、100.26 crates、0% suicide、100% Task 1 retention。
- 答案：安全准入解决晋级阻塞，但不等于学会全部长期规划。

#### 6.1.2 Task 3：能否改善竞争并保留旧能力？

- 三 seed confirmation + seed22/c150 main validation。
- 三条独立训练链得分增量为 +1.92、+1.23、+2.56；均值 +1.90，样本标准差 0.67。
- 每条链的 100 个世界是环境重复，不写成 300 次独立训练。
- 得分 5.91 → 7.14，差值 1.23，CI [0.27, 2.14]。
- 第一名率 41% → 53%；击杀 0.45 → 0.44，不称击杀改善。

#### 6.1.3 Task 4：专项训练为何没有稳定收益？

- Shared-parent admission failure。
- E1/E2/E3、Frozen C/S 和 score campaign 分开呈现。
- 结论限定于已运行协议。

### 6.2 Rainbow Lite：特征、奖励与延迟信用 `[主要作者：待定]`

#### 6.2.1 导航停滞来自历史缺失还是奖励诱导？

- 固定 v5/r18/mask-v5 主链，其他版本作为问题导向分支。
- Task 2 c0800：8.45/9 coins、60% all-coins、无长 WAIT/往返。
- 多因素组合不作单项因果归因。

#### 6.2.2 对手与阶段信息是否改善 Task 3？

- V5 c0300：7.80 score、0.65 kills、65% first、0% long WAIT。
- c250/c350 与 v11 只作局部诊断，不能组成排名。

#### 6.2.3 Task 4 的残余失败是什么？

- c1200 1,000 局：2.938 score、83.8% survival、0.7% suicide、57.1% long WAIT。
- 50 seeds 是环境 seeds，并非 50 次独立训练。
- 结论：主要失败转为长期停滞；缺少与 B33 同等级的最终比较。

### 6.3 Q-learning 与 CNN：共享成果能否跨模型迁移 `[主要作者：待定]`

#### 6.3.1 Task 1：两种替代学习器能否导航？

- Double Q(λ)：48.95 coins、96% all-coins。
- Distilled CNN：49.84 coins、96% all-coins。
- 额外 TD fine-tuning 退化到 37.21 coins、55% all-coins。

#### 6.3.2 Task 2：安全炸箱为何仍不能完成金币目标？

- Q r20：3.95/9 coins、53.45 crates、0% suicide、75% long ping-pong。
- CNN D02：2.60/9 coins、44.15 crates、0% suicide、0% all-coins。
- 共同瓶颈是箱后长期目标重选；未通过 Task 2，停止进入 Task 3/4。

### 6.4 跨路线综合与最终选择 `[主要作者：待定]`

- 表格总结三条路线的原始问题、方案、证据、结论和决策。
- 不建立跨协议统一排行榜。
- B33 的 1,000-world benchmark 与 Die Hardest 的六局 package equivalence 分开。
- 最终选择依据：完整 Task 1–4 谱系、4.231 历史均分、50.2% 含并列第一、94.7% survival 和可复现打包。
- 限制：27155、Docker/官方硬件未验证、不是普遍最优证明。

## 7. Conclusion

- 多种价值学习器可完成 Task 1，但迁移深度不同。
- 历史/reward 改变停滞，mask 减少自杀，二者都不自动解决长期规划。
- Task 3 得分改善；Task 4 增量方案缺少稳定收益。
- B33 是证据完整性与工程约束下的最终选择。

## Figures and Tables

- Figure 1：课程训练与证据流程。
- Figure 2：Die Hardest 推理和动作准入。
- Table 1：模型族及证据状态。
- Table 2：Task 3 父子配对。
- Table 3：Q/CNN Task 1–2 代表结果。
- Table 4：三条路线问题—解决—结论综合。

## Submission Checks

- 摘要和 Background 使用完整正文，不保留写作提示或示意图占位。
- 报告开头列明最终 Agent 的 NumPy 与 PyTorch 版本。
- 所有标题填写真实主要作者。
- 所有数字可追溯到 `report-assets/tables/*.csv` 或其 `source_path`。
- `not_reported`、`unrun` 不转换为 0。
- 报告 PDF 不上传公开仓库。
- 团队成员理解、核验并以自己的表达修订 AI 辅助初稿。
