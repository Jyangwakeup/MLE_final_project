# Q-learning 与卷积神经网络：Task 1 导航成功，Task 2 迁移受阻

主要作者：Ji（真实姓名待确认）。本文是供团队审阅的本地初稿，不是课程最终 PDF。两条路线的历史目录均作为变体介绍；团队的 Task 3-4 与最终模型选择由相应作者撰写。

## Task 1 导航表现没有预测 Task 2 目标完成

本研究关注一个课程迁移问题：在 Task 1 学会追踪可见金币的策略，能否在 Task 2 的炸箱、逃生和隐藏金币出现后继续完成收集？Task 1 把重点放在导航；Task 2 还要求模型安全放弹、等待爆炸、识别新出现的金币并切换目标。因此，本文分别沿 Q-learning 与 CNN 两条学习路线，先说明状态表示和价值学习器，再报告 Task 1 冻结结果，最后分析 Task 2 的迁移结果与失败模式。

两条路线构成两种表示与学习器组合，不属于同条件算法竞赛。本文用各自的训练链回答路线内问题，并在结尾比较共同现象；只有场景、训练合同、checkpoint 选择和评估种子一致时，数值才支持直接排名。

## Q-learning 路线：Double Q(λ) 学会导航，但仍受循环限制

### 方法：从离散 Q 表到 Watkins Double Q(λ)

Q-learning 根据动作价值选择行为：模型估计六个动作的长期回报，再从可选动作中选取估值最高者。早期单表模型使用离散状态键；连续距离和危险信息变化时，相近局面可能落入不同键，经验难以共享。主模型改用 84 维 `continuous-v2` 特征和按动作 tile coding。Tile coding 将连续特征映射到多组重叠离散格，使相近状态共享部分参数。模型维护两组动作价值估计，一组选择下一动作，另一组评价它；Watkins 资格迹给近期状态-动作对分配更新权重，资格迹系数 λ 控制回报向过去状态传播的程度。行为策略执行非贪心探索动作时，学习器切断旧资格迹。Double Q 的设计动机是减轻最大化偏差，但本项目没有单独测量该偏差。算法依据见 [Q-learning](https://doi.org/10.1007/BF00992698) 与 [Double Q-learning](https://papers.neurips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)。

84 维 `continuous-v2` 向量先为六个动作分别生成 29 维输入。动作编号 `a` 从 0 到 5，以下特征索引均从 0 开始：每个动作取区间 `[10a, 10a+10)` 的 10 个动作专属值、共享区间 `[60, 78)` 的 18 个值，以及动作历史维度 `78+a`，合计 29 维。Tile coder 再用 8 组平移错开的网格（tilings）、每维 8 个离散区间（bins）和 32,768 个哈希槽编码该输入。折扣率为 0.95、资格迹系数为 0.8，学习率从 0.08 降至 0.02。特征包含目标可达性、危险、放弹机会和上一动作信息，不直接指定目标动作。物理合法性先排除不可执行的移动；Task 2 的 `survival-mask-v1/all`（安全动作掩码）再排除预测时域内没有存活路径的动作。策略和 Double Q 更新都使用同一准入集合，剩余动作仍由 Q 值排序。[Agent 合同](../../agent_code/optimized_double_q_lambda_agent/callbacks.py)与[学习器](../../agent_code/learning_common/tile_coding.py)给出实现。

正式 Task 1 父链采用 `r7_safe_credit_potential`。该奖励用相邻状态间折扣后的势能变化提供金币、箱子和安全训练信号；在势函数塑形条件成立时，这类信号保持原任务的最优策略。[势函数塑形原论文](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)给出该性质的条件。Task 2 的 r20 分支只在当前位置与下一步安全、存在可达目标且有安全推进动作或有效放弹时，对 WAIT 额外给 −0.04。该条件惩罚不属于势函数项，不继承上述策略保持性质。官方得分与训练辅助奖励分开统计；不同路线也存在同名 r20，本文使用完整奖励名区分。

### Task 1：Double Q(λ) 学会收集可见金币

Task 1 使用无箱子、无对手的 `coin-heaven` 场景，禁用 BOMB（放置炸弹动作），目标为捡满 50 枚金币。单表 Q 基线与 Double Q(λ) 候选各使用训练随机种子 11、约 100,000 个动作步，并每 25,000 步保存快照。开发评估使用环境 seeds 10000-10004，每个 seed 运行 20 局。单表 Q 的最佳开发结果为 71% 捡满、平均 47.98 枚；Double Q(λ) 的 100,008-step 快照达到 96%、49.51 枚。两者同时改变了状态表示和学习器，因此该差距只支持“组合方案更好”，不能归因于 Double Q 一项。[单表 Q 选择记录](../../runs/qbaseline_t1_s11_j472871/best_task1_selection.json)和[Double Q 选择记录](../../runs/qlambda_opt_t1_s11_j472872/best_task1_selection.json)保存冻结指标与 checkpoint 身份；[快照评估程序](../../experiments/select_task1_snapshots.py)按固定规则选出候选。

固定 Double Q(λ) 模型快照（checkpoint）后，在独立确认集的环境 seeds 11000-11099 各运行一局：96/100 局捡满，平均 **49.58/50** 枚。长 WAIT 问题局率衡量持续等待循环，长往返问题局率衡量重复返回相邻位置的对局比例；两项在本组评估中均为 0%。完整 CPU `act` 的最大测量值为 11.63 ms，低于 500 ms 规则上限；本地测量不等于官方硬件认证。[100 局原始汇总](../../runs/qlambda_t1_confirm_s11000_j472894/qlambda_t1_confirm_s11000_j472894_summary/summary.csv)中 `AVERAGE` 行的 `mean_coins=49.58`、`all_coins_rate=0.96`。团队报告资产目前写 48.95，尚未找到与本次固定 checkpoint 确认相同的原始记录支持该值，不能把它替换成本段结果。

这条确认链只支持该 checkpoint 在 Task 1 协议下完成可见金币导航；它没有测量箱子破坏、隐藏金币搜索或对战能力。

上述 Task 1 checkpoint 关闭安全动作掩码，只作为 Task 1 独立基线。为进入 Task 2，团队另从随机初始化训练启用掩码的正式父链，保持相同训练 seed 和 `r7_safe_credit_potential`；该父链通过三次连续冻结能力检查及 20-seed 晋级审计。冻结评估固定模型权重并关闭探索。正式 Task 2 从这条父链恢复；两条 Task 1 链合同不同，不作为同一训练链的重复结果。[父链晋级审计](../../runs/qlambda_formal_t1_s11_j472900/promotion_audit.json)记录连续检查、独立 stage gate 和 CPU 指标；[晋级审计程序](../../experiments/audit_task1_promotion.py)生成该记录。

### Task 2：安全炸箱没有带来隐藏金币完成

Task 2 改用有箱子和隐藏金币、无对手的 `classic` 场景，开放 BOMB。r20 主线在训练 seed 11 累计 200,000 个 Task 2 动作步后，对各快照用 seeds 10000-10019 各一局冻结评估。20 局评估选出 200,000-step 快照作为该实验的最佳候选。它保留了 Task 1 导航能力并学会安全放弹，却未达到金币、炸箱和循环门槛。表中列出该候选和后续变体的结果；这些都是开发评估，不是主验证。

后续变体检验了“增加历史、改变 tile 泛化或加入示范能否解决循环”的假设。表中金币均值以 9 枚为满分，长 WAIT 和长往返比例按各自日志协议统计。各变体的训练预算、父链和奖励合同不全相同，数值不构成严格的跨行单因素排名。每行链接各自的冻结选择 JSON；[Task 2 快照选择程序](../../experiments/select_task2_snapshots.py)从原始评估结果生成这些指标。

| Q-learning 变体 | 主要变化 | Task 2 冻结结果 | 证据支持的判断 | 原始结果 |
|---|---|---|---|---|
| r20 主线，200k | 84 维、联合 tile、条件 WAIT 信号 | 20 局；3.95 金币、53.45 箱；长 WAIT／往返 30%／75% | 安全放弹，但金币与循环未达标。 | [快照选择 JSON](../../runs/qlambda_r20_t2_200k_s11_j473159/best_task2_selection.json) |
| v4／r12 | 126 维循环历史与 reward-only 反循环 | 最佳 125.2k；0.95 金币、18.40 箱；长 WAIT／往返 0%／35% | 循环局部减少，任务能力更低；特征与奖励同时变化。 | [快照选择 JSON](../../runs/qlambda_v4_r12_t2_200k_s11_j472958/best_task2_selection.json) |
| crate quantized | 箱子前沿距离三值化 | 100k；1.35 金币、23.70 箱；长 WAIT／往返 80%／90% | 实验日志所列旧 v2 的 100k 参照为 2.45 金币、39.60 箱；本变体两项均较低。该比较来自历史运行记录，不是本轮配对重跑。 | [本变体选择 JSON](../../runs/qlambda_crate_r7_t2_100k_s11_j473117/best_task2_selection.json)；[旧 v2 快照选择 JSON](../../runs/qlambda_t2_pilot_s11_j472902/best_task2_selection.json)；[对照说明](../../experiments/agent_variants/optimized_double_q_lambda_crate_agent/EXPERIMENT_LOG.md) |
| history input | 修复上一位置与目标历史输入，仍为 84 维 | 100k；1.65 金币、29.70 箱；长往返 85% | 修复了输入合同，未证明它是 Task 2 失败的唯一根因。 | [快照选择 JSON](../../runs/qlambda_history_r20_t2_100k_s11_j473165/best_task2_selection.json) |
| grouped tiles | 基础事实与历史事实分组编码 | 100k；0 金币、0 箱；长往返 65% | 本次编码未学到 Task 2 能力。 | [快照选择 JSON](../../runs/qlambda_grouped_r20_t2_100k_s11_j473167/best_task2_selection.json) |
| r21 条件折返 | r20 上增加目标未变时的安全折返罚项 | 最佳 75.2k；3.20 金币、46.80 箱；长 WAIT／往返 50%／70% | 未通过任务门槛；完成结果取代旧日志中的 submitted 状态。 | [快照选择 JSON](../../runs/qlambda_r21_t2_100k_s11_j473169/best_task2_selection.json) |
| team-demo | r20 warm start（从既有权重继续初始化）、五遍离线示范、50k 在线动作 | 最佳 50k；3.75 金币、56.90 箱；长 WAIT／往返 65%／75% | 炸箱增加，金币和 WAIT 未改善；另用了离线数据与更新。 | [快照选择 JSON](../../runs/qdemo_demo_s11_j473177/best_task2_selection.json) |

历史输入变体确认了旧 `continuous-v2` 的部分历史位在运行时被置零；修复后的得分仍低，因此不能把这一实现缺陷直接认定为全部失败原因。示范变体使用 40,100 条团队示范转移样本，每条记录状态、动作、奖励和后续状态；训练先进行五遍离线更新，再运行 50k 在线动作。与 r20 主线相比，它额外使用了示范数据和更新次数。全部变体均未通过完整 Task 2 门槛。[变体清单](../../experiments/agent_variants/MANIFEST.md)保留配置、日志与 checkpoint 身份。

Task 2 的证据把“会导航”与“会持续找目标”区分开来。r20 主线学会安全炸箱，但长往返仍常见；历史、编码和示范变体也没有同时提高金币收集并消除循环。由于多项变体改变了不同合同，这些结果定位了未解决的能力缺口，却不能单独证明某个特征或奖励是唯一原因。

## 卷积神经网络（CNN）路线：蒸馏学生学会导航，但隐藏金币仍未完成

### 方法：卷积表示与 Double DQN 价值学习

卷积神经网络（CNN）从带坐标的棋盘张量提取障碍、金币、炸弹和危险的空间关系。深度 Q 网络（DQN）用神经网络估计各动作的长期价值；Double DQN 将动作选择与动作估值交给两个网络，以减轻价值高估。训练时 online 网络选择下一动作，target 网络评价该动作；经验回放重复利用交互样本。CNN 在此充当状态编码器，Double DQN 负责价值学习。[DQN](https://doi.org/10.1038/nature14236)和 [Double DQN](https://doi.org/10.1609/aaai.v30i1.10295)给出方法背景，本项目的结论只依据本地实验。

早期 `cnn_double_dqn_agent` 使用 12 通道及池化卷积，作为空间基线。`cnn_path_double_dqn_agent` 扩展到 `board-path-history-v2` 的 17 通道，加入静态棋盘、金币与玩家、对手和炸弹、时间危险、广度优先搜索（BFS）距离场、上一位置及最近 16 步访问热图。BFS 以格子步数表示到目标的可达距离，不直接指定首选动作。64 通道无降采样残差主干使用空洞率 1、2、4、8，并结合自身位置特征与全局汇聚输出 Q 值。[Path CNN 日志](../../experiments/agent_variants/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md)记录了通道和网络合同。

最终蒸馏变体保留 17 通道编码，比较全局六动作输出头（global head）、动作对齐头及动作对齐加 D4 对称增强。D4 指棋盘的旋转与镜像变换。冻结的团队教师模型（teacher）`double_dqn_continuous_v2_agent` 在 Task 1 提供软 Q 值，即各动作的连续价值分数；学生模型（student）只对物理合法动作计算 Kullback-Leibler（KL）散度，温度设为 1。Teacher 只提供训练期信号，发布的 CNN 推理只加载学生权重，不读取 teacher。蒸馏不替代后续 Double DQN 更新，因此该模型仍使用 CNN 表示和价值学习。[Policy Distillation](https://arxiv.org/abs/1511.06295)给出方法背景；[蒸馏实现](../../agent_code/cnn_distilled_double_dqn_agent/distillation.py)定义本项目的具体损失。

### Task 1：蒸馏学生完成可见金币导航

早期 path CNN 的 self 通道曾错误标记金币坐标；带该 bug 的结果只能说明输入实现失败，不能用于评价奖励。修复 self 通道和历史缓存后，从零训练的 17 通道 r3 在 100 局开发评估中最佳为 0% 捡满、平均 13.35 枚；r5 条件反循环版本的 75,130-step 快照达到 17%、41.78 枚。独立改用 4-step return 的 E06 最佳为 0%、27.15 枚。r5 的 100k 末快照又降至 0%、27.11 枚，表明这条训练链存在后期策略退化。单 seed 的这些结果不支持普遍的奖励优劣结论。[Path CNN 实验日志](../../experiments/agent_variants/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md)列出全部快照。

团队 teacher 在 seeds 6000-6099 的 100 局轨迹产生 12,825 帧；前 80 个 seed 用于蒸馏训练，后 20 个 seed 用于验证。三个 CNN 结构候选最多训练 30 个训练轮次（epoch），并在 20 个独立 seed 上筛选。global、动作对齐、动作对齐加 D4 的捡满率分别为 95%、85%、90%。只有 global head 进入后续评估：100 局开发评估达到 97% 捡满率、平均 49.96 枚；预留确认集（reserved seeds 21000-21099，每个 seed 一局）达到 **96% 捡满、49.84/50 枚**。CPU `act` 延迟的第 95 百分位数和最大值为 8.39 ms 和 15.55 ms。团队没有对另外两种结构运行同规模确认评估，因此该结果只表明 global head 在本轮结构筛选中胜出。[CNN 选模记录](../../runs/cnn_distilled_t1_j471456/selection.json)与[CNN 日志](../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)给出来源。

global pretrained 继续进行 50k 个在线 Double DQN 动作更新后，开发集捡满率按 10k／25k／50k 依次降为 80%／72%／55%。团队据冻结评估保留 pretrained 作为 Task 1 最佳权重。这个单链结果将微调退化限定在本轮 r5、学习率与 KL 配置内。

### Task 2：安全迁移仍未完成隐藏金币

D01 从 Task 1 global checkpoint 迁移，开放 BOMB，采用 r7 sparse、4-step Double DQN 和 safety-all（行为选择与价值目标使用同一安全动作集合），并保留 Task 1 teacher 数据的 KL 正则。训练随机种子 11 的 200k 开发快照仍在 Task 1 取得平均 50/50；Task 2 平均仅 2.00/9 枚金币、34.35 箱，WAIT 动作率 55.20%，长 WAIT 问题局率 75%。同预算从零对照的 Task 2 均值稍高，为 2.50/9 枚金币、41.85 箱，但 Task 1 平均分仅 2.40，而迁移模型为 50.00；因此从零对照的局部 Task 2 优势以丢失旧任务能力为代价。这里的 KL 是旧任务保持信号，并非 Task 2 teacher 蒸馏。两组来自同 seed、同 200k 动作预算及同一 20-seed 开发评估；逐快照数值和权重哈希见 [D01 迁移／从零选择结果](../../runs/cnn_distilled_t2_j471458/selection.json) 与 [D01 实验日志](../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)。

D02 只加入条件可避免 WAIT 的 −0.04 训练奖励。相同训练随机种子 11、200k 快照的 20 局开发结果为平均 2.80 枚、54.20 箱；按预先规定的选择规则选中训练 seed 22 的 200k 快照，其开发结果为 4.20/9 枚、67.25 箱。固定候选后的 100-seed 主验证为 **2.60/9 枚、44.15 箱、0% 捡满**。该主验证中自杀、零放弹和无效动作均为 0%，已结算炸弹存活率为 100%，CPU `act` 延迟的第 95 百分位数和最大值为 8.88 ms 和 19.11 ms。D02 满足其预设的安全与炸箱迁移检查，但没有完成 Task 2 的隐藏金币目标。D03 奖励对照与 D04 5-step 后备链按预设条件跳过，均无性能结果。[D02 选择与验证](../../runs/cnn_distilled_t2_wait_j472342/selection.json)及[CNN 日志](../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)记录了不同评估阶段。

D01 的同 checkpoint、同 20 seeds 安全掩码反事实评估进一步限定了结论：关闭 mask 后，平均炸箱从 34.35 降至 3.55，自杀率从 0% 升至 90%。该结果支持保留 **这一 checkpoint** 的 mask，不能证明所有策略或局面都由它保证安全。[配对诊断](../../runs/cnn_t2_d01_maskpair_j472347/status.json)保存原始记录。

Task 2 的冻结结果显示，安全动作准入和炸箱并未自动解决金币目标搜索。D02 在主验证中保持零自杀和较高炸弹存活率，但平均金币与捡满率仍未达标；安全性提供完成任务的条件，不代表任务已经完成。

## 两条路线共享 Task 2 瓶颈，但结果不可排名

两条路线各自在 Task 1 冻结评估中完成可见金币导航。对应评估使用不同模型合同和种子集合，支持路线内结论，不支持跨路线数值排名。

Task 2 暴露出共同的迁移缺口：两条路线都能安全放弹或破坏箱子，却没有稳定完成隐藏金币收集。两条路线的评估样本量、阶段、训练 seed、奖励和 checkpoint 均不同，因此结果只支持这个定性结论。箱子打开后的目标重选与长程信用分配仍待研究；现有实验没有隔离足够因素来确认单一根因。

本稿只覆盖本人负责的 Q-learning 与 CNN 路线。团队总报告须单列 Task 3-4、课程基线和最终候选，并按一致协议论证选择；本稿的 CPU 测量也不代替官方硬件或 Docker 验证。

## 合并前核对事项（不进入最终报告正文）

1. 团队报告资产将 Q Task 1 均值写为 48.95，而原始独立确认 `summary.csv` 的 `AVERAGE` 行是 49.58；应先追溯 48.95 的具体 run，再修订表图或清楚标注两个不同评估。
2. 团队表格把 Q r20 的 3.95／53.45 标为 175.2k；原始 `best_task2_selection.json` 的最佳快照为 `step_0200000.pkl`。合并前须核对表格 checkpoint 标签。
3. Q r21 的旧实验日志仍写 `submitted`，但完成的 `best_task2_selection.json` 已记录 3.20／46.80。表中采用完成的原始记录。
4. 中文最终报告是否满足课程语言与字数要求、真实姓名和公开仓库 URL，须由团队提交前确认；成员还须逐句理解并改写 AI 辅助草稿。
