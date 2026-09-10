> 当前版本用于团队内部开发，暂以中文维护；项目后期统一翻译为英文。

# 实验接口

本目录是实验编排层。其接口来自 `PROJECT_REQUIREMENTS.md` 与
`IMPLEMENTATION_GUIDE.md`；特征和学习模块的实现细节仍分别由 A 与 B 负责。

## C 的职责

C 负责实验编排、评估、原始结果记录、分析、可复现性、消融配置和最终打包。这些
职责必须使用官方框架接口，且不修改官方核心文件。

## 已冻结接口

### A/C seam

特征接口由 `agent_code/team_agent/features.py` 负责。C 可以使用：

- 当前为 `v1` 的 `FEATURE_VERSION`；
- `Features` 和 `extract_features(game_state)`；
- `Features.legal_mask`；
- 40 维特征向量；以及
- 固定六动作顺序的 `ACTIONS`。

C 不复制这些定义、不改变特征语义，也不依赖特征实现细节。新的特征变体（包括危险
消融）必须由 A 通过相同特征接口提供并版本化。在实验配置选用计划中的
`v1_no_danger` 之前，必须**与 A 确认**其可用性。

### B/C seam

C 仅通过配置、环境变量、官方 agent 回调和 checkpoint 路径驱动未来统一的
`team_agent`。C 不访问或重新定义 `Transition`、Q-table、网络、replay buffer、
学习更新或其他学习实现细节。

C 需要 B 提供的最小信息是：

- 算法标识符；
- 明确的训练或评估模式；
- checkpoint 输入和输出路径；
- checkpoint 版本元数据；以及
- 训练指标的输出位置。

`team_agent` 实际消费的配置键、模式映射、checkpoint schema 与文件名、版本字段及
训练指标 schema 均为**待与 B 确认**项。在确认前，实验代码不得导入
`agent_code/q_learning_agent/` 或 `agent_code/dqn_agent/` 的私有模块。

### 运行时配置

- `BOMBERMAN_CONFIG` 指向一个绝对路径的完整实验配置。未设置时，agent 仅使用包内
  默认配置。配置必须自包含；第一版不支持隐式继承或多层合并。
- `BOMBERMAN_RUN_DIR` 指向绝对路径的运行输出目录。训练输出从该目录解析。
- 包内配置中的模型路径相对 agent 包解析。导出配置与模型路径不得包含开发机器的
  绝对路径。

运行器将在执行前保存实际使用的完整配置及其 SHA256。恢复或评估模式缺少
checkpoint 时应报错，而不是训练或回退到随机动作。

### 运行产物

每次运行独占一个目录，目标布局为：

```text
runs/<run_id>/
  metadata.json
  episodes.jsonl
  training.jsonl
  timing.jsonl
  official_stats.json
  checkpoints/
  sandbox/
```

`metadata.json` 记录展开后的配置、代码/源码身份、依赖、硬件、种子、模式、时间戳和
终态状态。JSONL 文件是追加式原始记录。`official_stats.json` 是用于交叉检查的官方
框架导出。`checkpoints/` 与 `sandbox/` 只属于对应运行。运行目录不得覆盖已有运行。

本文档只冻结接口。运行器、世界扩展、JSONL schema、分析、计时、训练、checkpoint
和打包将在后续阶段实现。

## 基础配置

`configs/base.json` 是完整 JSON 文档，不是继承片段。`algorithm`、`seed` 和
`checkpoint` 为 `null`，表示后续具体运行配置必须显式选择值。空的 checkpoint
元数据和训练指标细节均为**待与 B 确认**项；它们不定义存储或日志 schema。
