# Fix inference history and neural exploration wiring

Type: task
Status: resolved

修复 `continuous-v2` 冻结评估不推进历史特征的问题，并让共享神经训练路径使用
`BOMBERMAN_EXPLORATION_SPEC` 解析得到的探索计划。添加覆盖训练/评估一致性及 epsilon 数据源的
回归测试。

## Comments

- 2026-09-14：训练末 100 局平均 49.97 金币，而冻结评估平均 6.47；检查发现评估模式不调用
  `train.py`，所以原本仅由 reward callback 推进的历史特征始终停留在初始值。
- 2026-09-14：checkpoint 同时记录 exploration spec 的 1,920,000 步和静态 hyperparameters
  的 80,000 步；动作选择实际使用后者。

## Answer

- 冻结评估在每回合首次决策前重置 `continuous-v2` 历史，并在每次动作选择后推进上一动作、
  上一位置与上一金币目标；训练路径保持由结果回调推进，二者特征时序一致。
- 神经动作选择和训练 CSV 统一使用 Runner 注入的 `self.exploration_spec`。
- 新增两项回归测试。针对性 53 项测试通过；完整 184 项测试中 181 通过、2 跳过，唯一失败是
  隔离提交测试夹具缺少 `assets/coin.png`，与本修改无关。
- 使用原 R5 checkpoint 进行 5 局端到端冻结评估：平均 50.0 金币、100% 全收集、平均
  130.8 步、无长等待或长乒乓，验证历史状态修复有效。
