# 实验故事与证据缺口写作蓝图

> 文档定位：团队内部的论文写作蓝图，不是可直接提交的报告正文。  
> 适用材料：\`MODEL_EXPERIMENTS.md\`、\`DESIGN_EVOLUTION_AND_GAPS.md\`、\`REPORT_OUTLINE.md\`、\`Writing_Instruction.md\`、\`main.tex\`、\`report-assets/\`、Git 时间线和已有研究记录。  
> 权威约束：课程公告与 MaMPF 高于原始 \`notes/final_project.pdf\`，原始 PDF 高于项目内摘要。本文已直接核对原始 PDF 的 12 页内容，并用 \`PROJECT_REQUIREMENTS.md\` 和 \`Writing_Instruction.md\` 交叉检查。  
> 使用原则：时间线解释问题如何出现，Task 结构回答科学问题；不得把并行开发伪写成严格串行实验。

## 1. 文档目的与使用方式

这份蓝图要解决的不是“怎样把所有实验记录塞进第 6 章”，而是“怎样让读者看见一个问题被发现、被诊断、被解决或被证明仍未解决的过程”。现有材料的主要困难并非实验太少，而是证据分散在模型日志、结果表、Git 提交、打包验证和研究总结中。如果直接按成员或模型逐段拼接，读者会看见许多版本名，却很难回答：为什么要做下一次修改、修改解决了什么、哪项结论真正得到验证，以及为什么最后选择 B33。

蓝图建议把最终报告写成两层结构：

1. 外层以 Task 1 至 Task 4 为科学问题的递进路线。每个 Task 都从原始成功标准出发，展示该标准为何不足，然后引出下一项实验。
2. 内层以模型路线和共享组件解释不同解决方案。Rainbow Lite、Continuous Double DQN、Q-learning 与 CNN Double DQN 不是四个互不相干的个人项目，而是对同一课程问题的并行回答。

本文不能原样复制到论文。正式写作时需要压缩重复背景、补齐引用、核验数字、统一术语，并由对应作者用自己的语言重写。课程明确允许 AI 辅助草拟，但要求主要工作由成员完成；未经理解和改写的 AI 文本不能作为最终提交。

### 1.1 课程写作要求如何约束故事

原始作业文件对实验故事提出了比“报告一个最终分数”更严格的要求：

- 至少开发并比较两个机器学习模型，且报告所有开发过的模型。
- 与课程提供的预置 Agent 建立基线比较。
- 将任务拆成可管理的子目标，在每一阶段做有意义的实验。
- 展示每次设计或实现变化如何提高性能，或如实报告没有改善。
- 最终候选不能依据单局高分、主观回放或训练 reward 选择。
- Methods 要预先说明变体、指标和系统测试方法；Experiments and Results 要系统评价成功与失败。
- 报告应使其他学生能够理解并复现实验。
- 总篇幅约为每位成员 4,000 English words，三人约 12,000 English words，并且“不应明显更多”。中文字符数不能与英文单词数直接等价。

因此，第 6 章的中心不是“哪个模型分数最高”，而是“哪一个观察触发哪一个修改，修改在什么协议下带来什么变化，以及结论边界是什么”。课程预置 Agent 的同协议比较目前尚不完整，必须被列为核心证据缺口，而不能在最终稿中悄悄省略。

### 1.2 证据状态

所有数字和结论都应标记证据状态。状态描述的是证据可追溯性，不是结果好坏。

| 状态 | 含义 | 允许的论文表述 |
|---|---|---|
| \`verified_raw\` | 能追溯到逐局或逐世界原始结果、checkpoint 和协议 | 可报告具体数值，并在协议允许时做统计推断 |
| \`verified_summary\` | 能追溯到已有汇总，但缺少完整逐局数据 | 可报告汇总事实，不应补做无法验证的显著性推断 |
| \`packaging_verified\` | 只验证打包、加载、动作等价或兼容性 | 只能说明交付正确性，不能说明比赛性能 |
| \`pending\` | 统一冻结评估尚未执行或尚未提供 | 写“待统一验证”，不能填写为 0 |
| \`not_run\` | 该训练阶段或评估没有执行 | 写清原因与训练终点，不能推断性能 |
| \`not_reported\` | 实验执行过，但现有来源未记录该指标 | 说明指标缺失，不得从其他指标反推 |
| \`not_comparable\` | 有结果，但场景、对手、种子、样本或指标定义不同 | 可用于路线内部诊断，不进入跨模型排名 |

“配置文件存在”“代码已经实现”“实验计划写过”都不等于“实验完成”。同样，打包后的动作轨迹一致只证明交付包与源模型等价，不证明它在 1,000 局中的历史得分被重新复现。

### 1.3 每段实验的最小论证单元

最终报告中的每个主要实验至少回答以下七项：

> 我们首先观察到什么异常？  
> 该异常使哪个原始成功标准失效？  
> 我们提出了什么可检验假设？  
> 在保持哪些因素不变时，只改变了什么？  
> 用哪个 checkpoint、协议、样本量和指标验证？  
> 结果支持什么，又不能支持什么？  
> 该证据如何影响晋级、停止或下一次修改？

如果一个实验同时改变 Feature、Reward、Mask 和训练预算，它仍可作为系统版本比较，但不能把性能变化随意归因于其中单项。单项因果结论必须依赖受控消融或清楚的反事实。

## 2. 一句话研究故事与研究问题

### 2.1 一句话研究故事

> 项目的主要进展不是不断换用更复杂的模型，而是团队在每个 Task 中发现原有成功指标不足，逐步把问题从“能否收集金币”扩展为“能否避免循环、能否安全放弹、能否在地图变化后保持目标、能否在对手压力下提高正式得分，以及能否在 CPU 与交付约束下可靠运行”。

这句话可以成为 Introduction 末尾、Experiments 开头和 Conclusion 的共同主线。三个位置的功能不同：Introduction 把它写成研究动机，Experiments 用证据展开，Conclusion 回答哪些部分已经成立。

### 2.2 四个研究问题

**RQ1：不同学习器能否沿 Task 1–4 课程获得逐步扩展的能力？**  
重点不是最终是否达到 Task 4，而是每一次晋级是否同时保留旧任务能力。Q-learning 和 CNN Double DQN 停在 Task 2，应被写成统一晋级框架产生的实验结果，而非最初只计划做到 Task 2。

**RQ2：Feature、Reward 和 Survival Mask 分别解决什么类型的失败？**  
Feature 回答模型能看见什么，Reward 回答经验如何获得训练信用，Mask 回答某个动作是否被允许执行。三者作用位置不同，不能互相替代。

**RQ3：一个路线中的成果能否跨学习器迁移？**  
例如 Survival Mask 能否在不改变学习器权重时降低自杀，历史特征和连续目标是否能跨路线改善循环，\`continuous-v2\` 如何被 Continuous Double DQN 复用。迁移成功要区分“组件能运行”“即时安全改善”和“长期任务完成改善”。

**RQ4：在性能、稳定性、旧任务保持、CPU 和交付约束下，为什么选择 B33？**  
B33 的选择应表述为现有历史证据和交付约束下的工程决策，而不是统一冻结协议下已经证明的算法冠军。正式回答 RQ4 仍缺 B33 与 Rainbow 的共同 Task 4 配对评估及课程 baseline。

### 2.3 五幕结构

| 幕 | 表面问题 | 真正发现 | 方法论转变 |
|---|---|---|---|
| 一：建立学习基线 | 能否收集金币 | 高均值掩盖 WAIT、往返和 checkpoint 退化 | 冻结 checkpoint 和评估协议，不能假定训练越久越好 |
| 二：从导航进入炸箱 | 能否放弹炸箱 | 首先暴露的是自杀和动作阻塞 | 将 Survival Mask 作为独立动作准入层 |
| 三：安全以后仍不完成 | 能否活下来 | 安全、炸箱和收齐隐藏金币是三种能力 | 引入历史、目标连续性、WAIT/循环奖励和延迟信用 |
| 四：进入对手环境 | 能否提高分数 | 总分提高不等于击杀提高，基础设施错误也会改变结论 | 使用配对验证、bootstrap CI 和旧任务保持 |
| 五：强对手与交付 | 能否构造更复杂方案 | 复杂方案波动大，长 WAIT 成为主要残余失败 | 以证据链、稳定性、CPU 和可交付性共同选择候选 |

## 3. 真实开发时间线：问题如何出现

时间线的作用是解释研究动机，不是把 Git 提交顺序伪装成严格实验设计。多条路线在同一天并行，部分总结和评估在实现之后补做；最终论文应按科学问题重排，同时在 Planning 和 Training 中保留真实协作关系。

| 时间 | 主要活动 | 发现的问题 | 触发的下一步 | 负责人或成果流动 | 当时仍不能得出的结论 |
|---|---|---|---|---|---|
| 09-02 至 09-08 | 框架熟悉；炸弹、逃生、金币和对手特征 | 原始状态不能直接提供可学习的局部目标与危险结构 | 建立路径、危险和目标相关特征 | 全队形成共享环境理解 | 尚无模型优劣证据 |
| 09-09 至 09-12 | Q-learning、DQN、训练恢复与评估基础设施 | 训练过程可恢复不等于 checkpoint 单调变好；平均金币不足以描述行为 | 保存中间 checkpoint、固定评估和行为指标 | Q/DQN 路线与评估工具并行发展 | 不能用最后 checkpoint 代表最佳 checkpoint |
| 09-13 | 多模型和多表示探索，包括 CNN、Hybrid 等 | 更复杂输入会引入通道语义、历史堆叠和训练成本问题 | 检查输入合同，比较表格与神经模型 | 季嘉怡主导 Q/CNN，其他路线提供共享特征 | 不能因 CNN 理论容量更大就推断表现更好 |
| 09-14 | Task 2 winner 与统一 Survival Mask | 会放弹的模型可能被自己的炸弹困死；危险动作需要运行时约束 | 发展独立动作准入层并冻结 Task 2 winner | 范思卿主导 Mask；\`continuous-v2\` 等成果跨路线复用 | 0% 自杀不等于完成全部隐藏金币 |
| 09-15 | Task 3 pilot、phase reward、CNN 输入与历史问题 | 对手阶段暴露奖励阶段切换、计数和状态历史合同错误 | 修复评估基础设施，增加历史与阶段奖励检查 | 多路线共同修复实验合同 | 修复前后的分数不能直接归因于算法 |
| 09-16 | Rainbow Lite；Feature、Reward、Mask 多轴发展 | 安全后仍会 WAIT、往返或失去目标；单一轴不足 | Feature v3/v4、Reward/信用设计和 Mask 版本并行迭代 | 祝雨嫣主导 Rainbow 与 Feature/Reward | 多轴一起变化时不能识别单项因果效应 |
| 09-17 至 09-20 | Task 3 修复验证；Q/CNN 正式重测；安全工程和 Task 4 specialist | 得分与击杀可能分离；安全搜索增加成本但未稳定增分 | 父子配对、bootstrap CI、专项失败分析 | 范思卿路线形成 Task 3 配对证据；季嘉怡完成 Q/CNN 诊断 | 单一开发集高分不能形成跨模型排行榜 |
| 09-20 至 09-25 | Rainbow 固定 V5 链；B33 打包和证据整理 | Rainbow 的主要残余失败是长 WAIT；新方案没有稳定胜者 | 回到证据最完整候选，验证打包等价与 CPU 行为 | 祝雨嫣完成 Rainbow 大样本诊断；团队整理 B33 证据 | 打包等价不能替代共同 Task 4 性能复测 |

### 3.1 时间线应如何写进论文

Project Planning 应展示并行责任与共享资产：祝雨嫣负责 Rainbow Lite，范思卿负责 Continuous Double DQN，季嘉怡负责 Q-learning 与 CNN Double DQN。不能写成“每人各自完成一个互不相干的模型”，因为原始作业明确反对这种分工。Training 可按阶段描述何时冻结 checkpoint、何时恢复训练、何时迁移组件。Experiments 则不应按日期流水账，而要把同一科学问题的证据集中起来。

停止 Q-learning/CNN 应写成资源分配决策：在 Task 2 的统一能力要求下，它们未达到晋级门槛；继续投入会挤占 Task 3/4 的训练和验证预算。因此团队保留其诊断价值，把资源转向更有证据支持的路线。这不是“失败后被动放弃”，也不是“它们原本只负责 Task 1–2”。

## 4. 按 Task 重构的科学故事

### 4.1 Task 1：基础导航与首次晋级

#### 研究问题

不同表示和学习器能否在无箱子、无对手环境中快速收集可见金币，并且避免无意义 WAIT、非法动作和往返循环？

#### 原始成功标准及其不足

最初容易采用“50 枚金币中的平均收集数”作为主指标。这个指标直观，却可能掩盖三类失败：少数世界长时间停滞、策略通过往返获得局部 shaped reward、以及后期 checkpoint 退化但均值仍高。Task 1 又是单 Agent 场景，因此不应使用 win rate；更合适的是 mean coins/50、all-coins rate、完成步数、invalid rate、误放 BOMB rate 和动作延迟。

#### 观察到的异常

Q-learning、DQN、Continuous Double DQN、CNN 与 Hybrid 的探索说明，能够学习基本导航并不意味着训练过程稳定。Distilled CNN 的 Task 1 结果为 49.84/50、96% all-coins，动作 P95/max 为 8.39/15.55 ms；但额外 TD fine-tuning 后下降到 37.21/50 和 55% all-coins。这是“继续训练不保证继续改善”的清楚例子。Q-learning 的结果在不同来源中出现 48.95 与 49.58 的冲突，两者都伴随 96% all-coins；在回查原始选择文件前只能写“约 49/50，96%”，不能擅自选一个更好看的数。

#### 原因假设与解决路径

- 表格方法通过低维目标和路径特征快速学习，但可能依赖手工表示。
- CNN 可学习更丰富的空间模式，但输入通道、历史堆叠和训练稳定性构成额外合同。
- 中间 checkpoint 的行为可能优于最终 checkpoint，因此需要冻结开发、确认和主验证阶段。
- 平均金币必须与尾部失败指标同时报告，才不会把少数严重停滞平均掉。

#### 验证证据

Task 1 表应包含四个正式模型族，而探索性的 DQN/Hybrid 可在正文作为早期分支或移至附录。每行都写 checkpoint、protocol、N、coins、all-coins、steps、invalid、BOMB、latency 和 evidence status。当前部分字段为 \`not_reported\` 或 \`pending\`，尤其是多条路线的完成步数、非法动作和误放弹率。

#### 问题回答与晋级决定

现有证据支持“多种学习器都能获得基本导航能力”，也支持“复杂模型与更长训练不保证更好”。它尚不能支持四模型在统一 Task 1 冻结协议下的正式排名。晋级 Task 2 的依据应是导航能力达到最低标准且推理可运行，而不是宣称某一模型已成为总冠军。

#### 结论边界

Task 1 的高金币率不能预测 Task 2 的安全放弹，也不能预测 Task 3/4 的对战能力。若没有统一补测，正文必须把 Task 1 数值描述为路线内部开发证据。

### 4.2 Task 2：从炸箱生存到目标重选

#### 研究问题

模型能否在有箱子、无对手环境中安全放弹、逃离爆炸、重新选择目标并在步数限制内找出和收集全部隐藏金币，同时保留 Task 1 导航能力？

#### 原始成功标准及其不足

“炸掉很多箱子”最初似乎代表进展，但它不能说明模型是否活下来、是否找到隐藏金币、是否收集已出现的金币或是否最终完成任务。Task 2 必须将 crates、bomb survival、suicide、coins/9、all-coins、zero-coin、long WAIT、ping-pong 与 Task 1 retention 分开。

#### 第一层异常：会放弹但会自杀

Task 1 策略进入 Task 2 后首先暴露的是即时安全。模型可能学会 BOMB，却在走位中堵住逃生路径，或者选择当前合法但在未来数步必死的动作。范思卿主导的 Survival Mask 将问题从“Q 值偏好什么”分解为“哪些动作有资格进入选择集合”。Continuous Double DQN Task 2 winner 在 N=100 下达到 7.25/9 coins、100.26 crates、0% suicide，并保留 Task 1 能力。该结果支持安全动作阻塞可以被解除，但 all-coins 指标当前 \`not_reported\`。

#### 第二层异常：安全以后仍不完成

0% suicide 并未结束 Task 2。Q-learning r20 在 N=20 下取得 3.95/9 coins、53.45 crates、100% bomb survival、0% suicide，却有 30% long WAIT 和 75% ping-pong。CNN D02 在 N=100 下为 2.60/9 coins、44.15 crates、0% all-coins、0% suicide，WAIT 占 22.56%。这些结果共同说明：即时安全、破坏箱子和完成隐藏金币目标是三种不同能力。

#### 原因假设

安全后停滞可能来自：

1. 状态表示没有记住最近位置、访问历史和先前目标，模型难以区分“重新经过”与“没有进展”。
2. Reward 能惩罚一次 WAIT，却没有把炸弹在数步后揭示金币的结果分配给先前动作。
3. Mask 消除了危险动作，但剩余动作集合可能过度保守，导致安全 WAIT 或在两个等价位置间往返。
4. 学习器的样本效率与表示容量不同，共享组件的效果不一定自动转化为任务完成。

#### 不同解决路径

- Rainbow Lite（祝雨嫣）：发展 Feature v3/v4，以历史状态、目标连续性和循环检测补足观测；随后调整 Reward 和跨时序信用设计。
- Continuous Double DQN（范思卿）：以 Survival Mask 处理自杀与动作阻塞，并通过 Task 2 winner 冻结安全能力。
- Q-learning 与 CNN Double DQN（季嘉怡）：检验共享安全、历史或 Reward 设计能否跨学习器迁移，同时记录表格方法与神经网络的不同失败。

#### 反事实与单变量证据

CNN D01 到 D02 的单变量 WAIT reward 修改比“大版本整体更好”更有解释力，因为它尝试在其他条件固定时检查停滞信号。CNN 同一权重下的 safety-on/off 反事实显示，关闭 Mask 后 crates 从 34.35 降到 3.55，自杀率升至 90%。这强烈支持运行时保护层确实改变了安全行为，但因为权重是在带有相应安全合同的系统中获得的，它不是重新训练后的完整消融，不能证明 Mask 对长期学习效果的净因果效应。

#### 问题回答与停止决定

Task 2 给出最重要的跨路线回答：安全准入可以跨学习器即时迁移，但长期目标重选和信用分配不能自动迁移。Rainbow Lite 与 Continuous Double DQN 因现有证据满足晋级方向而进入 Task 3；Q-learning 和 CNN Double DQN 因未达到 Task 2 的完成与稳定性门槛停止继续训练。停止是实验结论，不是原始研究范围。

#### 结论边界

当前 Task 2 结果来自不同协议和不同样本量，不能直接排成四模型排行榜。Continuous winner 的 all-coins、四模型共同 seeds 以及统一 CPU 延迟仍需补齐。

### 4.3 Task 3：弱对手下的竞争能力

#### 研究问题

在 peaceful_agent 与 coin_collector_agent 等弱对手压力下，模型能否把导航、炸箱和生存能力转化为正式得分、击杀和名次，同时保持旧任务能力？

#### 原始成功标准及其不足

总体 score 是正式目标的一部分，但 score 同时包含金币与击杀。若只报告 score，就无法判断改进来自更有效收集金币，还是来自真正学习追击和布置击杀。Task 3 因此至少报告 score、coins、kills、kill-round rate、first-place、exclusive-first、survival、suicide、opponent deaths、Task 2 retention 和 latency。

#### 基础设施首先成为问题

Task 3 pilot 暴露 phase reward、历史快照和安全计数等实验合同问题。若旧状态被错误复用、阶段事件计数不一致，性能变化可能来自记录错误而非策略改进。因此“实验基础设施也是模型合同的一部分”：修复前后的结果必须分开，不能把错误协议下的分数混入正式比较。

#### Continuous Double DQN 父子配对

父 checkpoint 与 child checkpoint 在相同的 100 个世界上配对，避免地图差异掩盖模型变化。seed22/c150 的结果为：

| 指标 | Parent | Child | 配对差值 | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| score | 5.91 | 7.14 | +1.23 | [0.27, 2.14] |
| coins | 3.66 | 4.94 | +1.28 | [0.87, 1.70] |
| crates | 43.78 | 58.97 | +15.19 | [12.17, 18.16] |
| kills | 0.45 | 0.44 | -0.01 | [-0.16, 0.14] |
| first-place | 41% | 53% | +12 pp | 需按原始定义核对 CI |

三个训练 seed 的 score 差值为 1.92、1.23 和 2.56，说明提升方向并非只存在于一个训练 seed。但世界级配对样本不能被写成 100 次独立训练；跨训练 seed 的均值与标准差应另行报告。

#### 结果回答

这些数据支持 child 提高总体 score、coins 和 crates，也显示 first-place 上升；kills 的置信区间跨越 0，不能说击杀能力得到显著改善。科学故事应把“得分能力”与“战斗击杀能力”拆开：Task 3 的修改主要改善了资源获取和整体竞争结果，而不是证明模型学会更强追杀。

#### Rainbow checkpoint 非单调

Rainbow 的 checkpoint 选择同样表明局部最高分不能直接决定晋级。固定 V5/R18/Mask-v5 链中，Task 3 c0300 的开发结果为 score 7.80、kills 0.65、first-place 65%，但这是单训练 seed 的 development evidence。应利用 checkpoint 曲线解释非单调性，并在固定协议下确认，而不能把开发集峰值称为最终主验证。

#### 晋级与边界

Continuous Double DQN 和 Rainbow Lite 进入 Task 4。Q-learning/CNN 在表中保留为 \`not_run — not promoted from Task 2\`，并标出其最终 Task 2 checkpoint；空缺不是 0。不同开发协议下的 Rainbow 与 Continuous 结果不能形成正式排名。

### 4.4 Task 4：强对手、稳定性与最终交付

#### 研究问题

面对 rule_based_agent 或强自有 Agent，哪些改动能够稳定提高正式得分？当性能证据、CPU 延迟、稳定性和截止时间发生冲突时，应如何选择可交付候选？

#### 复杂方案的结果

E1 的三个训练 seed 得分分别为 4.06、4.10 和 3.27；E2 为 2.34；E3 为 2.79。安全指标可达到 0% suicide 与 100% bomb survival，但复杂的安全搜索或 specialist 没有形成稳定的 score 增益。这些失败不是附带噪声，而是回答“更复杂是否更好”的关键证据：安全性可以提高，整体竞争力却可能因为保守行为、目标丢失或计算开销下降。

#### Rainbow 大样本诊断

固定 V5/R18/Mask-v5 的 Rainbow c1200 在 1,000 局诊断中得到 score 2.938、survival 83.8%、suicide 0.7%、long WAIT 57.1%。样本由 50 个环境 seed 各重复 20 局构成，不是 1,000 次独立训练，也不是预注册的正式质量门。它的价值在于定位失败：主要残余问题不是直接自杀，而是长 WAIT 和策略停滞。该证据支持下一步应优先解决目标持续性或过度保守，而不是继续堆叠即时危险规则。

#### B33 与 Die Hardest

B33 的历史 1,000-world 汇总为 score 4.231、first-place 50.2%、exclusive-first 37.8%、survival 94.7%，证据状态为 \`verified_summary\`。Die Hardest 的打包等价性由 6 个配对游戏和 180 条动作轨迹支持，动作 P95/max 为 14.826/44.601 ms、RSS 319.645 MiB，状态为 \`packaging_verified\`。这两类证据必须分开：历史 benchmark 支持既有性能印象，打包测试支持源模型与提交包行为一致；后者不是 1,000 局性能复跑。

已知 world 27155 的安全合同反例意味着“在测试中未自杀”不能写成全局安全保证。正确措辞是“在给定协议中观察到较低自杀率或 100% bomb survival”，并保留反例作为局限。

#### 最终选择

推荐写法：

> 基于当前可核验的历史性能、稳定性、CPU 推理与交付约束，B33 被选为提交候选。该选择反映了证据完整度和截止日前可交付性，不等同于统一冻结协议下已经证明 B33 是四个模型族的算法冠军。

该结论诚实回答了工程决策，却保留科学边界。若补齐 B33 与 Rainbow 的共同 Task 4 配对验证、课程 baseline 和统一 CPU 协议，才可升级为正式跨模型选择结论。

## 5. Rainbow Lite 路线的准确叙事

本节优先依据祝雨嫣确认的 \`members/祝雨嫣/Rainbow-Lite-Experiment-Timeline.md\`。其中关于分支关系、研究目的和最终组合的描述，优先于较早的综合清单；\`MODEL_EXPERIMENTS.md\` 中的数字可作为历史开发证据，但在回查原始结果前不能反过来改写已确认的实验逻辑。尤其需要区分两件事：

- 已经存在的开发评估、局部测试和 1,000 局诊断；
- “最终形成的 V5/r18/mask-v5 模型在 Task 1–4 上完成统一测试”这一待补实验。

后者目前仍是 \`pending\`。因此，论文可以写“该组合被固定为最终 Rainbow 合同并已有若干阶段性证据”，不能写“该最终模型已经在统一协议下通过 Task 1–4”。

### 5.1 基线：先固定学习器，再分离表示问题

Rainbow Lite 在 Double DQN 上加入 Dueling network、Proportional Prioritized Experience Replay 和固定 4-step return。它没有 distributional RL、noisy network 等完整 Rainbow 组件，因此命名必须保留 Lite，避免夸大实现范围。

最初的 \`rainbow_lite_continuous_v2_agent\` 使用旧 84 维 \`continuous-v2\` 和 Survival Mask v1。这个基线的科学意义不是“第一个 Rainbow 版本”，而是建立一个控制点：在表示仍接近 Continuous Double DQN 的条件下改变 learner，帮助区分算法变化与后续表示变化。若没有同协议直接对照，论文只能把它写成设计动机，不能声称已完成因果隔离。

### 5.2 探索阶段：复杂 Feature 与 Reward 没有形成稳定胜者

在 Mask v1 条件下，路线探索了 V6–V11 和 Spatial V6，并组合 r19–r22 的对手压力或阶段奖励：

| 分支 | 主要变化 | 想回答的问题 | 当前证据状态 | 合理结论 |
|---|---|---|---|---|
| V6 | 对手跟踪、相对运动、火力线、逃逸空间 | 显式对手状态是否改善追击 | 局部实验，协议与 V5 不完全一致 | 未形成可确认的稳定增益 |
| V6 Stable | 减弱对手压力 | 不稳定是否来自攻击激励过强 | 平行消融 | 统一协议下未胜出 |
| V7 | 本局击杀数与是否首杀 | 阶段 Reward 能否依赖可观察状态 | 有训练和中断记录 | 证明信号可实现，不证明效果 |
| V8 | 可生存炸箱机会 | 能否联合表达收益与逃生 | 已实现、局部测试 | 性能证据不足 |
| V9 | 163 维压缩到 139 维 | 删除常量和冗余字段能否降低复杂度 | 有迁移和测试 | 证明可压缩，不等于性能提高 |
| V10 | 中心四象限箱子/对手密度 | 区域密度能否提供长期方向 | 局部实验 | 未证明优于 V5/V7 |
| V11 | 四象限可达箱目标与距离 | 可达目标是否优于简单密度 | 小样本高分 | 10 局不足以支持胜出 |
| Spatial V6 | 空间棋盘表示 | 卷积表示是否优于连续向量 | 独立分支，协议待核对 | 不进入正式排名 |

r19/a/b 调整对手压力，r20 取消预测性 BOMB 正奖励而按实际炸箱和击杀结算，r21 根据箱密度和首杀状态调整势能，r22 简化为首杀前后两个阶段。由于父 checkpoint、Reward、训练预算和评估样本没有全部固定，这组探索最有价值的结论是“继续增加复杂度没有形成可重复的明显增益”，而不是给 V6–V11 排名。

这是一条很好的失败故事：团队没有机械选择编号最高或小样本最高的版本，而是在发现证据不可比、增益不稳定后停止扩张表示。正文应选两到三个有代表性的分支说明假设与失败，把完整版本表移入附录。

### 5.3 固定最终 Rainbow 合同

最终固定组合为：

    Rainbow Lite learner
    + continuous-v5（140 维）
    + r18_wait_attractor_escape
    + survival-mask-v5

Feature v5 增加箱子区域方向和密度、放弹目标距离、跨步保留的炸箱目标以及剩余箱子比例。Reward r18 加强长期 WAIT 惩罚，并把有效 BOMB 视为任务进展。Mask v5 要求在己方炸弹完整危险期内存在反馈可控的生存策略。

三个轴同时改变，所以 V5 组合的表现只能归因于完整系统合同。除非补做受控消融，不能分别写成“Feature v5 带来多少提升”“r18 解决 WAIT”或“Mask v5 提高最终 score”。已有 Task 2/3 开发数字和 Task 4 诊断可以展示该组合的行为，但最终四 Task 统一测试仍须填为 \`pending\`。

### 5.4 Rainbow 路线真正回答了什么

已经可以回答：

- Rainbow Lite 能作为不同于普通 Double DQN 的学习器路线被实现和训练。
- 对手跟踪、阶段奖励、区域目标和更高维表示等复杂扩展没有在现有不统一协议中形成稳定胜出证据。
- 固定 V5/r18/mask-v5 的开发结果暴露出 checkpoint 非单调，Task 4 大样本诊断暴露长 WAIT 是主要残余失败之一。
- 多轴组合实验不能支持单个 Feature、Reward 或 Mask 的独立因果归因。

尚不能回答：

- 最终固定 Rainbow 模型是否在统一冻结协议下通过全部 Task 1–4。
- Rainbow 是否优于 B33、课程 baseline 或其他模型族。
- r18、Feature v5 或 Mask v5 中哪一项贡献最大。
- 单训练 seed 的结果能否推广到训练随机性变化。

## 6. 三条共享设计轴

Feature、Reward 与 Survival Mask 应作为共享方法层，而不是藏在某个模型介绍中。它们分别干预观测、学习信号和动作执行。清楚区分三轴，才能解释为何“安全了却不完成”“reward 上升却正式得分不升”以及“Feature 更复杂但表现不稳定”。

### 6.1 Feature：模型看不到什么

Feature 将原始游戏状态映射为学习器输入：

$$
\phi_t = \phi(s_t, h_t, g_t),
$$

其中 $s_t$ 是当前游戏状态，$h_t$ 表示位置、访问或动作历史，$g_t$ 表示当前目标或阶段信息。Task 1 的最短方向足以支持可见金币导航，但 Task 2 的隐藏金币要求模型记住哪些区域已搜索、炸弹是否产生进展以及目标在地图变化后是否应重选。Task 3/4 又需要对手相对位置、火力线和逃逸空间。

Feature 能解决部分可观测和别名问题，却不能保证学习器正确使用这些信息。V9 的压缩证明某些字段可删除，不等于压缩必然提升性能；V11 的小样本高分也不能证明可达目标特征稳定优于 V5。

### 6.2 Reward：训练信号如何分配

折扣回报为：

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}.
$$

正式比赛的金币与击杀信号稀疏，因此项目使用辅助 Reward 加速学习。然而 shaped reward 只在训练中存在，不能替代正式游戏指标。若“朝金币移动”持续获得正奖励，而“远离金币”惩罚不足，Agent 可能往返刷取局部信号；若有效炸弹的收益数步后才出现，则需要延迟信用把结果连接到先前 BOMB。

Reward 轴应回答：一次行为为何值得被强化，以及信用分配到哪个时间步。它不能弥补 Feature 中完全缺失的信息，也不能保证高 Q 值动作在未来可生存。论文应报告 r18、r20 等设计针对的行为问题，并把训练 reward 曲线降为诊断证据。

### 6.3 Survival Mask：哪些动作不应被执行

动作准入集合可写为：

$$
\mathcal A_{\mathrm{adm}}(s)
=
\mathcal A_{\mathrm{physical}}(s)
\cap
\mathcal A_{\mathrm{survival}}^{(H)}(s).
$$

Double DQN 目标在准入动作集合上计算：

$$
y_t=r_{t+1}+\gamma Q_{\theta^-}
\left(
s_{t+1},
\arg\max_{a\in\mathcal A_{\mathrm{adm}}(s_{t+1})}
Q_\theta(s_{t+1},a)
\right).
$$

Mask 的职责是拒绝明显物理非法或在有限危险视野 $H$ 内不可生存的动作。它不是替学习器规划长期目标，也不是全局安全证明。有限 horizon、环境模型误差、对手动作和已知反例都可能破坏安全合同。Mask 过强还会减少探索或诱导 WAIT，因此必须同时报告安全、能力和延迟。

### 6.4 三轴不能相互替代

| 失败现象 | 主要怀疑轴 | 为什么另外两轴不够 |
|---|---|---|
| 反复走过同一区域但不知已访问 | Feature/历史 | Reward 看不到历史时难以区分重复，Mask 只管危险 |
| 明知目标却长期 WAIT | Reward、目标连续性或 Mask 过保守 | 增加空间通道不保证产生行动 |
| 放弹后被自己的炸弹困死 | Mask 与危险 Feature | 单纯加大自杀惩罚可能学习太慢，且错误已发生 |
| 炸箱很多但不收齐金币 | Feature 与延迟信用 | Mask 只能保证生存，不能重选隐藏目标 |
| 对手环境 score 上升但 kills 不升 | Reward/策略目标 | 安全和地图表示可能只提高金币获取 |

## 7. 最有价值的实验故事

下面的案例不应都写成同等长度。正文优先保留能改变研究方向或最终决策的案例；完整 checkpoint 表、版本清单和回放索引进入附录。

### 7.1 Task 1 表格学习器与 checkpoint 退化

**发现：** 单表 Q-learning 能学习导航，但 Double Q、eligibility trace 和 tile coding 的组合更适合稳定扩展；训练中间点仍可能退化。  
**诊断：** 最终 checkpoint 不是天然最佳，平均金币无法描述完成效率和尾部失败。  
**解决：** 冻结中间 checkpoint，加入 all-coins、完成步数、WAIT/循环和非法动作。  
**验证：** \`MODEL_EXPERIMENTS.md\` 记录约 100k action steps 的 Task 1 比较，但 Q-learning 数字存在 48.95/49.58 冲突。  
**回答：** 表格方法可成为强 Task 1 基线；不能声称某一算法普遍更优。  
**限制：** 需回查原始 CSV 和 checkpoint 选择脚本。

### 7.2 CNN 输入合同与蒸馏后退化

**发现：** CNN 不仅受优化难度影响，也受通道语义、历史堆叠和输入修复影响。  
**诊断：** 如果输入合同错误，网络容量越大并不会自动弥补错误信息。  
**解决：** 修复 17 通道 \`board-path-history-v2\`，通过冻结 Continuous Double DQN teacher 做 masked-KL 蒸馏。  
**验证：** Distilled CNN 得到 49.84/50 与 96% all-coins；继续 TD fine-tuning 降至 37.21/50 与 55%。  
**回答：** 蒸馏成功迁移 Task 1 行为，但在线微调会破坏该行为。  
**限制：** 这不是 CNN 从零训练达到同等性能的证据。

### 7.3 Continuous Double DQN Task 2 winner

**发现：** Task 1 导航策略进入有箱环境后，自杀与动作阻塞成为首要问题。  
**诊断：** Q 值最大动作可能在未来数步无逃生路径。  
**解决：** 使用 \`continuous-v2\`、相应 Reward 与 Survival Mask 的完整合同。  
**验证：** N=100，7.25/9 coins、100.26 crates、0% suicide，并保持 Task 1；all-coins 尚未报告。  
**回答：** 完整系统可通过项目 Task 2 晋级门。  
**限制：** 不能把增益单独归因于 v2、Reward 或 Mask。

### 7.4 CNN D01 到 D02 的 WAIT reward 实验

**发现：** D01 安全但 WAIT 严重，Task 2 收集能力低。  
**假设：** 对“存在安全进展动作却选择 WAIT”的行为施加窄条件惩罚，可减少停滞且不破坏 Task 1。  
**解决：** 在保持主要模型合同不变时修改 WAIT shaping。  
**验证：** 多个 checkpoint 显示 Task 1 维持 50.00，Task 2 coins 有波动，最终 D02 为 2.60/9、44.15 crates、0% all-coins、0% suicide、WAIT 22.56%。  
**回答：** 窄 WAIT 信号改变行为分布，但没有使 CNN 完成 Task 2。  
**限制：** 应核对 D01/D02 的父 checkpoint、局数和唯一变化，避免把多 checkpoint 选择偏差写成因果。

### 7.5 CNN safety-on/off 反事实

**发现：** CNN 的 Task 2 安全表现可能主要来自运行时 Mask，而非网络内部学会逃生。  
**诊断：** 用同一权重关闭 safety，观察行为是否崩溃。  
**验证：** crates 从 34.35 降至 3.55，自杀率升至 90%。  
**回答：** Mask 对这一冻结权重的运行时安全至关重要。  
**限制：** 同权重反事实不是重新训练消融，不能回答无 Mask 训练能否学出不同策略。

### 7.6 Q r20：失败从 WAIT 转移到 ping-pong

**发现：** 缩窄 WAIT 惩罚后，Agent 可能不再静止，却在相邻状态间往返。  
**诊断：** 单步动作频率改善不等于长期进展；行为失败会换一种形式出现。  
**验证：** N=20，3.95/9 coins、53.45 crates、0% suicide、100% bomb survival、30% long WAIT、75% ping-pong。  
**回答：** r20 没有解决长期目标维持，只把部分停滞从 WAIT 转化为循环。  
**限制：** N=20 适合诊断，不宜做精细跨模型排名。

### 7.7 Task 3 父子配对与 bootstrap CI

**发现：** child 的平均 score 看似更高，但地图差异会污染非配对比较。  
**解决：** 在相同世界计算 child-parent 差值，并对世界级配对差值 bootstrap。  
**验证：** score +1.23，95% CI [0.27,2.14]；kills -0.01，95% CI [-0.16,0.14]。  
**回答：** 训练提高总体 score，却没有证据说明击杀提高。  
**限制：** 100 个世界不是 100 次训练；训练 seed 变异另行报告。

### 7.8 Rainbow checkpoint 非单调与分支停止

**发现：** 更晚 checkpoint、更高版本 Feature 和更复杂阶段 Reward 没有稳定占优。  
**诊断：** 局部高分可能来自小样本、父模型差异或协议差异。  
**解决：** 停止继续增加 V6–V11 复杂度，固定 V5/r18/mask-v5。  
**验证：** 各分支有局部测试和中断记录，但不构成统一排行榜；固定链开发结果同样只有单训练 seed。  
**回答：** 复杂度升级不是可靠的版本选择规则。  
**限制：** 没有受控统一消融时，只能归因于完整组合。

### 7.9 Rainbow Task 4 的 1,000 局长 WAIT 诊断

**发现：** 自杀率已低，但 score 仍不理想。  
**诊断：** 50 个环境 seed 各重复 20 局，分解死亡与停滞。  
**验证：** score 2.938、survival 83.8%、suicide 0.7%、long WAIT 57.1%。  
**回答：** 主要残余失败更接近长期目标停滞，而非直接自杀。  
**限制：** 该数据不是最终 Rainbow Task 1–4 统一测试，也不是 1,000 个训练重复。

### 7.10 Task 4 specialist、安全搜索与 B33 回退

**发现：** E1/E2/E3 和更强安全搜索没有稳定超过现有候选。  
**诊断：** 安全增强可能带来保守动作和额外延迟；不同训练 seed 波动明显。  
**验证：** E1 为 4.06、4.10、3.27，E2 为 2.34，E3 为 2.79；部分安全搜索的完整动作延迟超过内部 margin gate。  
**回答：** 新方案未形成稳定胜者，因此团队回到证据链更完整的 B33。  
**限制：** B33 的 4.231 是历史汇总，Die Hardest 是打包等价验证；两者不能拼成一次新主验证。

## 8. 团队协作与成果流动

### 8.1 模型责任不是实验章节边界

| 模型族 | 主要负责人 | 原始目标 | 当前课程终点 |
|---|---|---|---|
| Rainbow Lite | 祝雨嫣 | 沿 Task 1–4 发展完整能力 | 形成最终合同；统一 Task 1–4 测试待补 |
| Continuous Double DQN | 范思卿 | 沿 Task 1–4 发展完整能力 | 进入 Task 4，并形成 B33 交付候选 |
| Q-learning | 季嘉怡 | 沿 Task 1–4 发展完整能力 | Task 2 未过门槛，停止 |
| CNN Double DQN | 季嘉怡 | 沿 Task 1–4 发展完整能力 | Task 2 未过门槛，停止 |

负责人用于署名与可追溯性，不能把第 6 章拆成三个互不相干的故事。课程原始 PDF 明确要求真实团队协作，反对“每个成员各做一个独立模型”。因此 Planning 应重点展示成果流动，Experiments 则按 Task 组织，在每个 Task 内标记路线负责人。

### 8.2 共享成果矩阵

| 共享成果 | 主要来源 | 最初解决的问题 | 首次阶段 | 后续复用路线 | 当前证据与核对事项 |
|---|---|---|---|---|---|
| \`continuous-v2\` | 祝雨嫣相关开发，来源表述待核对 | 历史增强的连续状态表示 | Task 2 | Continuous DDQN、Rainbow 早期对照 | 查提交与 source path，避免仅凭命名归属 |
| Feature v3/v4 及后续 v5 | 祝雨嫣 | 历史状态、循环、目标连续性 | Task 2 | Rainbow；跨学习器使用情况待核对 | Rainbow 时间线较确定，其他路线需负责人确认 |
| Reward/信用设计 | 祝雨嫣主导的 Rainbow 路线 | 稀疏回报、WAIT 与延迟 BOMB 信用 | Task 2–4 | Rainbow；其他适用路线待核对 | 多轴同时变化，不能夸大单项效果 |
| Survival Mask v1–v9 | 范思卿主导 | 放弹自杀、危险动作和安全工程 | Task 2–4 | Continuous DDQN、CNN、Rainbow | on/off 支持运行时效果；版本间公平消融不足 |
| 蒸馏 teacher | Continuous DDQN 冻结模型，具体贡献待确认 | CNN 从零学习不稳 | Task 1 | CNN Double DQN | 证明行为迁移，不证明 CNN 独立学会 |
| 冻结评估与配对统计 | 综合工具，作者待确认 | 协议不一致、地图方差和 checkpoint 选择 | Task 1–4 | 全部模型 | 工具来源、run ID 和脚本路径待补 |
| 打包等价验证 | 综合交付阶段 | 确保提交包与源模型动作一致 | Final | B33/Die Hardest | 只能支持 packaging，不支持性能排名 |

### 8.3 协作故事的推荐写法

可以把协作写成“问题驱动的组件交换”：祝雨嫣在 Rainbow 路线中扩展历史与目标 Feature，并验证复杂版本没有稳定增益；范思卿将即时安全问题抽象成可复用 Survival Mask；季嘉怡通过 Q-learning 和 CNN 检验这些设计是否依赖特定学习器。Q/CNN 的停止本身提供了迁移边界：运行时安全能够迁移，长期规划并未自动迁移。

对于来源仍不确定的组件，使用“由某路线首先系统化”“随后被某路线复用”比武断写“某人发明”更稳妥。最终稿必须由三位负责人共同核对成果来源和作者标记。

## 9. 当前缺少的关键证据

缺口排序依据是“它是否改变论文核心结论”，而不是补测是否容易。每项都同时给出理想补测方案和不补实验时的降级措辞。

### 9.1 P0：直接影响核心结论

#### P0-1 四模型统一 Task 1/2 冻结横评

**缺口：** 四个模型族现有 Task 1/2 数字来自不同协议、局数和 checkpoint 选择。  
**理想补测：** 冻结每个模型最终可用 checkpoint，在完全相同的 scenario、world seeds、agent seeds、局数、硬件和指标定义下运行；同时标明 checkpoint 实际训练终点。  
**若不补：** 只能分别陈述路线内部表现，使用 \`not_comparable\`，不生成跨模型排名或“Rainbow 更好”等结论。

#### P0-2 B33 与 Rainbow 的共同 Task 4 配对验证

**缺口：** B33 历史 benchmark 与 Rainbow 1,000 局诊断协议不同。  
**理想补测：** 在相同对手组成和相同世界上配对运行，报告每世界 score、coins、kills、survival、suicide、WAIT/loop、延迟及配对 CI。  
**若不补：** 写“B33 是历史证据与交付约束下的选择”，不能写“B33 在实验中优于 Rainbow”。

#### P0-3 课程预置 baseline Agent

**缺口：** 原始 PDF 明确要求与课程提供的 Agent 比较，当前统一 baseline 表不完整。  
**理想补测：** 在 Task 3/4 同协议加入 peaceful_agent、coin_collector_agent、rule_based_agent 或适合该 Task 的预置基线，清楚区分它们是对手还是被比较 Agent。  
**若不补：** 在 limitations 中明确“尚未满足同协议课程 baseline 比较”，不要用不同旧运行代替。

#### P0-4 最终 checkpoint 身份与共同 CPU 协议

**缺口：** 需要为所有最终候选记录完整路径、hash、训练终点、Feature/Reward/Mask 版本和 deadline 状态。  
**理想补测：** 生成冻结清单并在接近官方 Ryzen 单线程条件下测完整 \`act\`，包括 feature、mask、model forward 与动作选择，报告 warm-up、P50/P95/max 和 timeout。  
**若不补：** 只能说“在开发机器上满足内部测试”，不能推断官方环境一定满足 0.5 秒预算。

#### P0-5 统一 final-test

**缺口：** Rainbow 最终固定组合的 Task 1–4 测试仍 pending，四模型也没有完整统一 final-test。  
**理想补测：** 每个 Task 建独立冻结表；Q/CNN 可用最终 Task 2 checkpoint 测 Task 3/4 泛化，但必须标注未在这些 Task 训练。  
**若不补：** final comparison 保留占位，并把 RQ4 的回答限定为工程选择。

#### P0-6 Q-learning Task 1 数字冲突

**缺口：** 汇总 CSV 为 48.95，模型实验记录为 49.58，两处均报告 96% all-coins。  
**理想核对：** 找到原始逐局数据、checkpoint、协议 ID 和汇总脚本，重新计算。  
**若不补：** 使用“约 49/50，96% all-coins”，不得任选其一。

#### P0-7 Rainbow 清单与新时间线冲突

**缺口：** 较早清单曾把 Rainbow 写为 Task 1–2 或 evidence gap，而后续记录有 Task 1–4 开发链；新时间线又明确最终模型的 Task 1–4 测试待补。  
**推荐统一：** “路线曾在各 Task 产生开发/诊断证据；最终固定 V5/r18/mask-v5 合同的统一 Task 1–4 测试仍待补。”  
**禁止写法：** “Rainbow 已正式通过 Task 1–4”或“Rainbow 只研究到 Task 2”。

#### P0-8 Continuous Task 2 all-coins 与 Task 1 行为字段

**缺口：** Task 2 winner 有 coins、crates 和 suicide，但 all-coins 未报告；多个模型 Task 1 缺 completion steps、invalid rate、BOMB rate。  
**理想补测：** 用原始逐局数据补算；无法补算时重新冻结评估。  
**若不补：** 对应单元写 \`not_reported\`，不能以高平均金币替代全完成率。

### 9.2 P1：影响因果解释

| 缺口 | 为什么重要 | 理想设计 | 不补时的写法 |
|---|---|---|---|
| Feature v2 与历史/目标 Feature 的受控消融 | 决定长期规划改善能否归因于表示 | 同 learner、reward、mask、父 checkpoint 和预算，只改 Feature | “完整组合改善/变化”，不归因单 Feature |
| Rainbow Reward 单变量比较 | 判断 r18 是否真正减少 WAIT | 固定 Feature v5 与 mask-v5，只改 Reward；多训练 seed | “r18 针对 WAIT 设计”，不写“r18 解决 WAIT” |
| Mask off、v1、v5、v9 权衡 | 安全可能换来保守和延迟 | 同权重反事实加重新训练消融；报告安全、能力、延迟 | on/off 只支持运行时依赖 |
| Rainbow 多训练 seed | 单 seed 可能偶然 | 至少多个训练 seed，各自冻结后统一评估 | 明确“单训练 seed development evidence” |
| 接近官方 CPU 的完整延迟 | 官方每步 0.5 秒且超时会累积 | 单线程、warm-up、完整 \`act\`、P95/max | 仅报告开发机结果 |
| 训练、开发与最终对手重合 | 过拟合对手会夸大泛化 | 列出每阶段 opponent set，保留未见对手 | 公开重合并降低泛化措辞 |

### 9.3 P2：可写入局限或未来工作

- 超参数筛选的完整搜索空间、淘汰规则和负结果。
- 旋转与镜像对称性增强的受控实验。
- 正式游戏指标随训练进度变化的 checkpoint 曲线，而非只画 loss。
- Docker、官方 Ryzen 环境和正式锦标赛复验。
- world 27155 安全合同反例的最小复现、根因和修复验证。
- 更严格的跨 Task 遗忘测试与灾难性遗忘指标。

这些缺口不会阻止报告现有发现，但应阻止绝对化表述。例如可写“在所测协议中降低自杀”，不能写“保证不会自杀”。

## 10. 证据冲突与待核对清单

| 冲突字段 | 来源 A | 来源 B | 当前推荐写法 | 需回查材料 | 核对前禁止的结论 |
|---|---|---|---|---|---|
| Q Task 1 mean coins | \`table2_task1_task2.csv\`: 48.95 | \`MODEL_EXPERIMENTS.md\`: 49.58 | 约 49/50，all-coins 96% | 原始逐局结果、checkpoint 选择、聚合脚本 | 精确报告任一数值 |
| Rainbow 完成阶段 | 旧清单：Task 1–2/evidence gap | 新记录：有 Task 1–4 开发证据 | 路线有完整开发链，最终统一测试 pending | \`members/祝雨嫣/Rainbow-Lite-Experiment-Timeline.md\` 与原始 runs | 宣称已正式通过或只做到 Task 2 |
| Rainbow 最终组合效果 | MODEL 汇总含阶段数字 | 新时间线写最终 Task 1–4 测试待补 | 数字标 development/diagnostic，final-test pending | V5 结果目录和协议文件 | 把开发数字升级为最终横评 |
| B33 性能与包验证 | 1,000-world historical summary | 6 游戏、180 traces package equivalence | 分成 performance summary 与 packaging evidence | benchmark summary、trace comparison | 声称提交包重跑并复现 4.231 |
| Task 1/2 样本量 | 正文草稿可能写“100 次实验” | 表中实际为 100 worlds/games | 写 N=100 evaluation worlds/games | 协议与 run manifest | 写成 100 次独立训练 |
| Continuous Task 2 all-coins | 高平均 coins 与 crates | all-coins 字段缺失 | \`not_reported\` | 原始逐局 CSV | 推断全完成率很高 |
| Rainbow 1,000 局独立性 | 1,000 games | 50 environment seeds ×20 | 大样本行为诊断 | evaluation script | 写成 1,000 training seeds |
| 计划与执行状态 | \`planned_or_skipped\` | 部分文档可能写“实验分支” | 明确 planned/skipped，无数字 | run directory、日志 | 把配置存在写成实验完成 |
| deadline 身份 | 截止日前 B33 包 | 截止日后分析/补测 | 表中单列 pre/post deadline | manifest、Git 时间 | 把后续模型写成参赛提交 |

### 10.1 状态词的严格用法

- \`unrun\` 或 \`not_run\`：没有执行。
- \`planned_or_skipped\`：有计划或配置，但出于资源/条件跳过。
- \`implemented_not_evaluated\`：代码完成，没有性能评估。
- \`pending\`：明确等待统一测试或数据补充。
- \`not_reported\`：运行存在，只是该指标没记录。
- \`not_comparable\`：双方都有结果，但协议不兼容。

这些词不能互换。尤其不能把“implemented”写成“experimented”，也不能把“pending”写成 0。

## 11. 从蓝图转化为七章最终报告

### 11.1 Introduction

用中心命题解释项目价值：Bomberman 将稀疏奖励、部分可观测、延迟后果、多智能体对抗和实时推理约束结合在一起。介绍四个 RQ 和贡献，不在这里宣布 B33 获胜。贡献可概括为：四模型课程探索、Feature/Reward/Mask 三轴分解、配对评估与失败证据、可交付候选选择。

### 11.2 Background

先解释游戏、四个 Task 和 Agent 接口，再概述 Q-learning、Double DQN、CNN、Rainbow Lite、课程学习、Reward shaping 与 action masking。Background 回答“有哪些一般方法”，Methods 回答“本项目如何具体化”。需要引用 RL、Double DQN、Dueling、PER、n-step、reward shaping 和 action masking 的原始或权威文献。

### 11.3 Project Planning

展示共同 Task 4 目标、模型责任、时间线、训练硬件和成果流动。重点说明团队没有按成员隔离，而是通过 shared features、Mask、teacher、评估工具和打包验证协作。Q/CNN 的停止写成统一门槛与资源约束下的决策。

### 11.4 Methods

先定义完整 Agent 合同：

$$
\text{Agent contract}
=
\text{learner}
+\text{feature}
+\text{reward}
+\text{safety}
+\text{training protocol}.
$$

然后分别描述四个 learner 与三条共享轴，预先定义 development、confirmation、main validation 和 uniform frozen final comparison。Methods 只描述选择理由和测试方法，不提前把历史结果写成胜负。

### 11.5 Training

按 Task 说明初始化、resume、checkpoint、蒸馏、训练 seed、课程迁移和旧任务保持。写明 GPU/CPU、并行训练与最终单线程推理的区别。训练 loss、TD error 和 shaped reward 可用于诊断，但不能替代正式 score、coins、kills 和 survival。

### 11.6 Experiments and Results

以 Task 1–4 为一级科学故事。每个 Task 先给四模型统一证据表，再分路线写“发现—假设—改动—验证—回答—决定”。协议不同的历史证据分块显示；缺失字段使用状态词。Task 3 保留 paired difference/CI 图，Task 4 把 B33 historical、Rainbow diagnostic 和 package equivalence 分开。

### 11.7 Conclusion

逐项回答四个 RQ：

- RQ1：多种学习器能通过 Task 1；Continuous 与 Rainbow 进入更高阶段，Q/CNN 在 Task 2 停止，但统一最终横评仍缺。
- RQ2：Mask 解决即时危险准入，Feature 与 Reward 针对长期观测和信用；现有组合实验限制单项归因。
- RQ3：安全层显示跨学习器迁移，长期规划没有自动迁移。
- RQ4：B33 是现有证据和交付约束下的候选，不是已证明的统一算法冠军。

Conclusion 不引入新数字。未来工作直接对应 P0/P1/P2，而不是泛泛写“训练更久”。

### 11.8 可直接改写的实验段落模板

> 我们首先观察到【异常行为或指标矛盾】。该现象说明仅使用【原成功标准】不足以判断【目标能力】。因此我们提出【可检验假设】，并在固定【learner、parent checkpoint、协议、预算等】的条件下，仅改变【主要变量】。我们使用【场景、对手、seeds、N、指标】进行验证。结果显示【数值、差值与不确定性】。这支持【有限结论】，但不能支持【更强结论】。因此该路线被【晋级、停止或保留为诊断分支】，并触发下一步【修改或补测】。

每段最后一句必须推动故事前进。若一个结果没有改变理解或决策，应压缩到表格或附录。

## 12. Typora 公式与统计说明

### 12.1 Q-learning 与 Double DQN

表格 Q-learning 更新可写为：

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+\alpha
\left[
r_{t+1}
+\gamma\max_a Q(s_{t+1},a)
-Q(s_t,a_t)
\right].
$$

Double DQN 将动作选择和目标估计分开：

$$
y_t
=
r_{t+1}
+\gamma
Q_{\theta^-}
\left(
s_{t+1},
\arg\max_a Q_\theta(s_{t+1},a)
\right).
$$

这些公式用于解释方法，不是性能证据。正式结果仍来自冻结游戏评估。

### 12.2 配对差值

在相同世界比较 child 与 parent：

$$
\Delta_i=m_i^{\mathrm{child}}-m_i^{\mathrm{parent}}.
$$

平均配对效应为：

$$
\bar{\Delta}=\frac{1}{N}\sum_{i=1}^{N}\Delta_i.
$$

bootstrap 应以配对世界为重采样单位，不能分别重采样 parent 和 child 后再配对。置信区间必须在图注中说明 bootstrap 次数、区间类型和有效 $N$。

### 12.3 跨训练 seed

若有 $K$ 个独立训练 seed，其结果为 $x_1,\ldots,x_K$：

$$
\bar{x}=\frac{1}{K}\sum_{k=1}^{K}x_k,
\qquad
s=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(x_k-\bar{x})^2}.
$$

环境世界、Agent 随机 seed 和训练重复是不同层级：

- 多个 environment seeds 衡量地图和局面变化。
- 同一训练模型在多个 agent seeds 下运行衡量策略随机性。
- 多个 training seeds 才衡量训练过程的可重复性。

50 个环境 seed 各重复 20 局仍只有一个训练模型，不能写成 1,000 次独立训练。跨世界 CI 也不能替代跨训练 seed 的不确定性。

### 12.4 比率与分母

all-coins、suicide、survival、first-place 和 long WAIT 都必须给出分母定义。例如“suicide 0%”可能是每局比例、每次死亡比例或每次放弹比例，三者意义不同。若多个 Agent 并列第一，first-place 与 exclusive-first 必须分开。

## 13. 证据索引

以下索引是写作入口，不替代最终 run manifest。source path 在提交前应细化到具体文件。

| 重要证据 | checkpoint/合同 | protocol 与 N | evidence | 当前 source | 允许使用的表述 |
|---|---|---|---|---|---|
| Q Task 1 | 具体 checkpoint 待核对 | Task 1，样本量见源表 | 冲突待核对 | \`MODEL_EXPERIMENTS.md\`、\`report-assets/tables/table2_task1_task2.csv\` | 约 49/50、96% all-coins |
| Distilled CNN Task 1 | teacher/student 合同见模型记录 | Task 1，N 待表内核对 | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\` | 蒸馏迁移 Task 1；TD 微调后退化 |
| Continuous Task 2 winner | continuous-v2 完整合同 | N=100 | \`verified_raw\` | \`MODEL_EXPERIMENTS.md\`、结果表 | 7.25/9、100.26 crates、0% suicide；all-coins 未报告 |
| Q r20 Task 2 | 175.2k 附近链路待核对 | N=20 | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\` | 安全但 long WAIT/ping-pong 严重 |
| CNN D02 Task 2 | D02 | N=100 | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\` | 安全炸箱迁移，未完成 Task 2 |
| CNN mask on/off | 同一 D01 权重 | 反事实样本量见源记录 | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\` | 运行时安全依赖 Mask，不是训练消融 |
| Task 3 parent/child | seed22/c150 child 及 parent | 相同 100 worlds | \`verified_raw\` | \`report-assets/\` traceability 与模型记录 | score/coins/crates 提升；kills 未显著提升 |
| Rainbow Task 2 development | V5/r18/mask-v5 c0800 | 20 局 development | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\` | 开发集 8.45/9、60% all-coins；非最终冻结横评 |
| Rainbow Task 3 development | V5/r18/mask-v5 c0300 | development，单训练 seed | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\` | checkpoint 选择证据，不作跨模型排名 |
| Rainbow Task 4 diagnostic | V5/r18/mask-v5 c1200 | 50 env seeds ×20 | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\` | 长 WAIT 是主要残余失败；非最终质量门 |
| E1/E2/E3 | specialist 合同见模型记录 | 协议块分别报告 | 状态按原始结果核对 | \`MODEL_EXPERIMENTS.md\` | 新方案未稳定胜出 |
| B33 historical | Task4 B seed33/c200 | 1,000-world historical | \`verified_summary\` | \`MODEL_EXPERIMENTS.md\`、打包记录 | 历史表现支持候选选择 |
| Die Hardest package | B33 打包等价 | 6 games、180 traces | \`packaging_verified\` | package equivalence 记录 | 包与源模型动作一致；不代表性能复跑 |
| Rainbow 分支逻辑 | V6–V11、Spatial V6 | 协议不统一 | \`not_comparable\` | \`members/祝雨嫣/Rainbow-Lite-Experiment-Timeline.md\` | 复杂扩展未形成稳定增益证据 |
| Rainbow 最终 Task 1–4 | V5/r18/mask-v5 | 统一协议待定义 | \`pending\` | \`members/祝雨嫣/Rainbow-Lite-Experiment-Timeline.md\` | 最终合同已固定，统一测试待补 |

## 14. 最终写作与静态核对清单

### 14.1 故事完整性

- [ ] 每个 Task 都从研究问题和原成功标准开始。
- [ ] 每次修改由前一轮观察触发，而不是版本流水账。
- [ ] 每段都写支持与不支持的结论。
- [ ] 每个晋级或停止都有 checkpoint、协议、N 和证据路径。
- [ ] 时间线承认并行开发，不虚构严格串行因果。
- [ ] Rainbow 以新时间线为主要叙事来源，其他路线注明待负责人核对。

### 14.2 课程合规

- [ ] 报告所有开发过的模型，而非只写最终候选。
- [ ] 至少两个模型完成系统比较。
- [ ] 课程预置 baseline 的缺口被补齐或明确承认。
- [ ] Experiments and Results 是证据最密集的章节。
- [ ] 失败、无改善和未收敛结果被保留。
- [ ] 所有章节或小节标题标明主要作者。
- [ ] 最终稿约每位成员 4,000 English words，且不明显超出；中文草稿需另行确认课程是否接受及如何换算。
- [ ] 包含公开代码仓库 URL，不使用大学 Logo，不把报告 PDF 上传到公开仓库。

### 14.3 统计与措辞

- [ ] 不把 100 worlds 写成 100 次独立训练。
- [ ] 每个 training seed 先单独报告，再给跨 seed 均值和标准差。
- [ ] 相同世界的父子模型使用配对统计。
- [ ] 误差条说明 SD、SE、CI 或 bootstrap CI。
- [ ] 单 Agent Task 1 不使用 win rate。
- [ ] loss、TD error 与 shaped reward 不充当晋级证据。
- [ ] \`pending\`、\`not_run\`、\`not_reported\` 与 \`not_comparable\` 不混用。
- [ ] Q/CNN 的 Task 3/4 空缺不写成 0。
- [ ] Safety Mask 不写成全局安全保证。
- [ ] B33 不称为统一冻结协议下的算法冠军。

### 14.4 一位读者读完后应得到的答案

理想情况下，读者不会只记住“团队试了四种模型”，而会记住以下逻辑：

1. Task 1 证明基本导航可学，但平均金币和训练时长会掩盖策略退化。
2. Task 2 先揭示即时生存问题，Survival Mask 有效阻断部分危险动作。
3. 安全之后，Q、CNN 和 Rainbow 的结果又揭示长期目标重选与信用分配是独立瓶颈。
4. Task 3 的配对证据说明总分可以改善而击杀不一定改善，迫使评估拆分能力。
5. Task 4 说明复杂安全或 specialist 方案未稳定胜出，Rainbow 的主要残余错误转为长 WAIT。
6. 团队最终选择 B33，是因为它在历史性能、稳定性、CPU 和交付证据之间最完整，而非因为一次统一算法竞赛已经证明它绝对最好。

这才是整篇报告需要讲清楚的“发现问题—诊断问题—尝试解决—验证回答—承认边界”的完整故事。
