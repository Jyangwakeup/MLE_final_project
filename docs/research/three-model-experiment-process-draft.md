# 三类强化学习 Agent 的实验过程（供团队核验的中文报告草稿）

> 状态：2026-09-20 的证据重建稿，**不是已完成的最终选优报告**。成员应核对原始运行记录、补全作者署名，并用自己的表述改写后纳入 PDF。本文中的 `[已提交]` 指 Git 可追溯源码或结果，`[工作区]` 指当前未提交代码或本地 `runs/`，`[待补]` 指尚无可比结果。仓库有 202 个可达提交；同一变更有时在分支与合并历史中出现两次，不能按提交数计算独立实验。

## Methods：研究问题、模型与评价方法

完整 Bomberman 要求导航、炸箱、从炸弹逃生和对抗，单一总分无法诊断失败原因。我们按课程建议由 Task 1（金币导航，无箱子和对手、禁 BOMB）推进到 Task 2（炸箱与金币）、Task 3（弱对手）和 Task 4（三名 `rule_based_agent`）。每次训练后冻结权重、关闭探索，以正式游戏得分和行为指标评估；训练 reward 只用于优化和诊断。Task 1 主要看 50 枚金币的平均收集量、全收集率和完成步数；Task 2 增加炸箱、放弹、自杀和旧任务保留；Task 3/4 增加击杀、第一名率和对手压力下的安全性。完整 `act` 必须满足 0.5 秒正式预算。[依据：`PROJECT_REQUIREMENTS.md`、`experiments/CURRENT_TRAINING_EVALUATION_PARAMETERS.md`]

三条主要路线分别是：(1) 离散状态单表 Q-learning，及后续作为同一表格家族改进的 tile-coded Watkins Double Q(λ)；(2) 连续特征 MLP Double DQN；(3) Rainbow Lite，即 dueling Double DQN、比例优先经验回放及固定四步回报的组合。Rainbow Lite 是项目内的“lite”实现，不能写成包含全部原版 Rainbow 组件。所有路线由学习器对动作估值；特征和安全掩码不直接给出最佳动作。[依据：`all_other_agent_code/rainbow_lite_agent/model.py`、`all_other_agent_code/rainbow_lite_agent/README.md`、各 Agent 实现]

以下历史实验首先用于**筛选模型、特征、奖励和训练方案的组合**。当多个因素或训练预算同时变化时，结果不能归因为算法本身。只有固定其余条件的配对实验才支持对单项改动的因果解释。最终三模型排序需要在同一 Task 4 设置下另做冻结横向评估。

## Training：从提交历史重建的设计过程

| 阶段与问题 | 改动及决策 | 结果和证据状态 |
|---|---|---|
| 初期基线：能否学会收金币 | 引入 Q-learning、DQN 和连续特征，比较表格与神经网络的导航能力；随后用 R3 势能、R4 等惩罚诊断 WAIT 与折返。相关提交包括 `7687e16f`、`d5f800a6`、`3fa125ea`。 | [已提交] 历史 Task 1 七个有效候选的汇总见 `docs/experiment-log-runs-1-2.md`。这批结果的特征、reward 和训练时长不完全相同，只用于探索性筛选。 |
| 导航停滞：高平均金币仍不捡满 | 将连续特征从 v1 扩为含上一动作、目标连续性和折返信息的 v2，并比较条件式循环惩罚；开始使用冻结得分停止训练。相关提交 `8f35ca2e`、`b41de9c9`。 | [已提交] v2 与 reward 同时变化的运行不能证明任一因素单独有效；后续 Task 1 收敛与 Task 2 晋级以冻结评估为准。 |
| 炸箱安全：学会放弹后能否存活 | Double DQN 对探索、推理及 TD 目标统一施加生存动作掩码，并以相同父模型和独立验证选择 Task 2 checkpoint。相关提交 `18cb7f91`、`3b752d93`。 | [已提交] Task 2 正式胜者为 `double_dqn_continuous_v2_agent` seed 22；主验证 100 局平均 7.25 金币、100.26 箱、0% 自杀，Task 1 保留 100%。见 `experiments/task2_winner.json`、`experiments/task2_winner_evaluations.csv`。 |
| 弱对手：Task 2 能力能否迁移 | 初次 Task 3 pilot 维持 r7，三训练种子中 seed 22 失败，未晋级；phase reward、炸弹逃生义务及后续历史快照修复逐一针对失败现象。相关提交 `9da58527`、`1c480466`、`b42bec79`、`15d4721d`。 | [已提交] pilot 和 phase 失败见 `experiments/task3_pilot_results.json`、`experiments/task3_phase_results.json`。生命周期原主验证因逃生塌缩计数失败；修正计数后按登记协议重验，seed 22/c150 主验证得分 5.91→7.14、第一名率 41%→53%，击杀 0.45→0.44，不能声称击杀改善。见 `docs/research/task3-counter-validation-results.md`。 |
| Rainbow Lite：扩展资源与对抗表示 | v1→v5 逐步加入历史、安全余量、循环与箱子目标；r17/r18 调整资源与延迟炸弹信用；mask 单独版本化。v5/r18 集成提交 `bea0b74f`、`f9a056b9`。 | [已提交/工作区] 版本关系见 `docs/version-comparison-v-features.md`、`docs/version-comparison-r-rewards.md`、`docs/version-comparison-survival-masks.md`。版本号是特征、奖励和 mask 的独立轴，不能写成一条单变量递进曲线。 |
| Rainbow Lite 后续 Task 3/4 | v6 加对手跟踪，v7 加击杀阶段；v8/v9 研究可安全炸箱与压缩表示，v10/v11 从 v7 分叉测试象限密度和可达箱目标；r19–r22 与 mask-v5 等另行组合。 | [工作区] v6–v11 的 Agent、配置和多数 `runs/` 尚未提交。v10/v11 不是 v9 的后继；训练与评估轮数不同。现有短诊断只能决定补测优先级，不能证明最终最优。 |
| 表格路线重测 | 在单表 Q-learning 后加入两套 tile-coded Q 估计器和 Watkins trace 截断；继续尝试 Task 2 的历史、WAIT、箱距量化等针对性特征。相关提交 `548136bc`、`3004d869`、`23bf2c75`、`99887a3b`、`7cc6c78c`、`8b9a5b88`。 | [已提交] Task 1 改进已确认；Task 2 日志仍写有运行中和后续启动记录，不能在缺少完整冻结评估时称其已晋级。见 `agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`。 |

这种排列是**研究问题的逻辑顺序**。多名成员的分支曾并行开发，表格路线的正式重测发生于 Double DQN 的 Task 2/3 工作之后，不应在最终报告中伪写为一条严格串行的实验时间线。

## Experiments and Results：已能支持的比较

### 导航筛选与失败记录

初期相同 Task 1 场景的归档结果中，`continuous-v1 + Double DQN + R2` 为 49.52/50 金币、86% 全收集；单表 `discrete-q-v2 + R4 idle` 为 47.44、60%；离散 DQN 为 40.66、29%；Hybrid Dueling Double DQN 为 2.05、0%。这说明当时**具体实现组合**的导航表现不同，不能把差距只归功于网络架构。[已提交：`docs/experiment-log-runs-1-2.md`，原始 `runs/1/` 与 `runs/1.2/`]

该批 Q-learning 的旧 R4 与含 idle R4 共用 reward ID，但实际 reward spec 不同；含 idle 版本全收集率由 31% 升到 60%，同时长乒乓率由 11% 升到 21%。两个 R2 评估因 `discrete-v1` 与 `discrete-q-v2` 的特征契约不匹配而报错，**无有效性能数据**。Hybrid 的 100 局坏结果有效，但训练流程还出现绘图库错误，不能把训练状态写成完全正常。[已提交：同上]

在较新的、同为 100k action steps 量级的 Task 1 表格对照中，单表 Q-learning 的最佳冻结快照为 71% 全收集、47.98 金币；Double Q(λ) 为 96%、49.51。后者在独立的 100 局确认集仍为 96%、49.58，因而可说明**该表格改进组合通过 Task 1 门槛**，不能说明其在对战中胜出。两种实现还涉及特征和奖励差异，不是纯算法消融。单表中途在 50k/75k 快照全收集率跌至 0%；Double Q(λ) 在 50k 快照也跌至 40%，故选模必须依据冻结快照，不能只看训练步数增加。[已提交：`agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`]

### 阶段晋级与 Rainbow Lite 诊断

Double DQN 的 Task 2 结果具有三训练种子与一次独立主验证支撑；Task 3 的初次 pilot 因跨种子不稳定失败，经过决策历史与安全计数修复才形成 `task3_validated.pt`。该文件是**Task 3 已验证候选**，其 release manifest 明确写着 `task4_started=false`，不能把 7.14 的弱对手得分当作 Task 4 强对手成绩。[已提交：`experiments/task3_validated_release.json`、`docs/research/task3-counter-validation-results.md`]

Rainbow Lite 的本地结果显示继续探索的理由，也显示过拟合风险。例如 v5/r18/mask-v5 的 Task 2 完整 500 局父链，在 10 局冻结诊断为 8.7 金币、117.2 箱；其 Task 3 c100-anchor/c250 在 20 局为 7.25 分、20% 自杀，同链 c350 为 5.65 分、30% 自杀。v5/r18/mask-v5 的 Task 4 c350-anchor/c100 在 20 局为 3.65 分、0% 自杀；v6/r20/mask-v5 的对应 c100 为 3.00 分、0% 自杀。v11/r20 的 Task 3 c500 在 10 局为 9.9 分、20% 自杀。这些数值来自不同父模型、任务、checkpoint、种子组或训练预算；只能表述为**局部诊断**，不能组成一张公平排名表。[工作区：相应 `runs/**/summary.csv`；详见下方索引]

Task 4 短诊断中多次出现得分下降和较高自杀率，因此“版本号更大”不意味着能力更强。v10 象限密度与 v11 可达箱目标也是不同输入语义的分支，不可把 v11 的结果解释为在 v10 基础上的增益。[工作区：`docs/version-comparison-v-features.md` 与相关运行摘要]

**当前结论：** 已有证据可以解释为何从基础 Q-learning 发展到 Double DQN 和 Rainbow Lite，以及为何需要安全、历史与目标特征；还不能证明三类模型在最终 Task 4 条件下的胜者。课程基线 Agent 与三模型尚无统一、独立的正式横评表，最终选择留待下述实验。

## 最小补实验：冻结候选、横向比较、独立测试

1. **证据与候选审计。** 对每一类列出可加载 checkpoint 的路径、SHA-256、训练 seed、阶段、实际局数/action steps、feature/reward/mask ID、源码提交和运行状态。优先审计单表 Q-learning、Double Q(λ) 的 Task 2/3/4 产物，Double DQN 的 `task3_validated.pt`，以及 Rainbow Lite v5/v6/v7/v11 的已有冻结候选。只用历史开发集固定每类一个竞争候选；若某类没有 Task 4 可加载模型，记录“无同阶段候选”，不能拿 Task 1/2 得分代替。`agent_code/` 顶层目前只发现单表 Q-learning 与 Double DQN 的最终权重，Rainbow Lite 的比较需定位 `runs/*/checkpoints/final.pt` 并验证 agent 契约。[待补]
2. **同条件主比较。** 用 `experiments/configs/main_validation.json` 的 10000–10099（每 seed 一局）对三个候选及课程 `rule_based_agent` 基线分别冻结评估；三名对手均为 `rule_based_agent`，CPU 单线程，完整局结束。先审计这些种子是否已参与候选选择；若已使用，重新登记一组完全未用的连续 100 seeds，所有候选共用，且在运行前冻结清单。记录每局得分、金币、炸箱、击杀、独占/并列第一、自杀、非法动作、超时及完整 `act` P95/max。原始逐局表按世界 seed 配对，报告平均差值与 10,000 次世界级配对 bootstrap 95% 区间；不把区间是否跨零临时增为晋级门槛。[待补]
3. **预先固定胜者规则。** 仅纳入能加载、无评估错误且满足正式 0.5 秒预算及既定安全硬门槛的候选，按主比较平均正式得分最高选一个；同分依次比较较低自杀率、较低 `act` P95、候选 ID 字典序。所有三模型结果和课程基线进入报告，即使某模型明显落后。若只有 Double DQN 的 Task 3 候选可用，则报告“最终横评未成立”，不能命名三模型优胜。[待补]
4. **独立测试与兼容性。** 唯一胜者冻结后，用未参与训练/选模的 `20000–20099` 每 seed 一局；扫描所有 metadata、配置和逐局结果确认这些种子未被使用。若已使用，预先登记 `30000–30099`，并在结果中解释变更。独立测试不得触发重新训练或重新选模。对胜者再做干净官方框架、CPU、三 random opponents 与打包检查。[待补]

冻结评估命令模板（`<config>` 必须与 checkpoint 的 feature/reward/mask 契约匹配；`<run_id>` 使用全新目录）：

```bash
<python310_or_newer> -m experiments.run \
  --config <matching_config.json> --mode evaluate --device cpu \
  --task 4 --agent <agent_name> --checkpoint <checkpoint_path> \
  --seeds <the_100_registered_seeds> --n-rounds 1 \
  --run-id <new_unique_run_id>
```

本机系统 `python3` 为 3.9，读取 `experiments.run` 时会因 `tuple[...] | None` 注解报错；执行者须先选用项目兼容的 Python 3.10+ 环境。本文未启动新的训练或评估。[本次只读检查]

## 复现与核验索引

| 用途 | 首选原始证据 |
|---|---|
| 课程约束和四阶段定义 | `PROJECT_REQUIREMENTS.md`；`experiments/CURRENT_TRAINING_EVALUATION_PARAMETERS.md` |
| 初期 Task 1 七候选与失败 | `docs/experiment-log-runs-1-2.md`；`runs/1/`、`runs/1.2/` |
| 表格改进与确认 | `agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`；`docs/research/q-learning-task1-optimization.md` |
| Double DQN Task 2 与 Task 3 | `experiments/task2_winner.json`、`experiments/task2_winner_evaluations.csv`；`experiments/task3_validated_release.json`；`experiments/results/task3_counter_validation_20260917.json` |
| Rainbow Lite 版本契约 | `docs/version-comparison-v-features.md`、`docs/version-comparison-r-rewards.md`、`docs/version-comparison-survival-masks.md`；Agent 配置和 checkpoint metadata |
| Rainbow Lite 本地诊断 | `runs/v2_r7/eval10_rainbow_lite_v5_r18_s11_task2_full500/`；`runs/rainbow_lite_v5_r18_maskv5_s11_task3_c100_anchor_c0250_eval20/`；`runs/rainbow_lite_v5_r18_maskv5_s11_task4_c350_anchor_c0100_task4_eval20/`；`runs/rainbow_lite_v6_r20_maskv5_s11_task4_c350_anchor_c0100_task4_eval20/`；`runs/1.3/rainbow_lite_v11_r20_s11_task3_from_task2_c0500_eval10/` |

报告中的每张最终表应附训练/评估配置、实际 seed、局数、checkpoint hash 和结果路径。对照原始 `summary.csv` 时，`AVERAGE` 行是汇总行，不能当作额外一局或额外 seed。团队需在提交前复核动态课程公告、当前代码版本、主要作者署名和原始结果。
