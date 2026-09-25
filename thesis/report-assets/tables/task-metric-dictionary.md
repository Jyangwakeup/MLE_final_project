# Task 1–4 指标字典

| Metric (English field) | 中文定义 / 单位 | 聚合方式 | 适用 Task | 报告边界 |
|---|---|---|---|---|
| `mean_score` | 每局官方得分 | evaluation episodes 的平均值 | 1–4 | Task 1 无对手时等于 collected coins。 |
| `mean_coins` | 每局收集金币数 | 平均值 | 1–4 | Task 1 分母 50；Task 2 分母 9。 |
| `all_coins_rate` | 收齐该场景全部金币的局比例 | 完成局/总局 | 1–2 | 仅在金币总量固定且记录完整时报告。 |
| `mean_crates` | 每局炸毁箱子数 | 平均值 | 2–4 | 不是正式得分，不可替代 coins。 |
| `mean_kills` | 每局击杀数 | 平均值 | 3–4 | Task 3 需父—子配对；CI 跨 0 时不称改善。 |
| `first_place_rate` | 第一名（定义须写明是否含并列）局比例 | 比例 | 3–4 | `exclusive_first_rate` 与 `tied_first_rate` 单列；不得混写。 |
| `suicide_rate` | 自己被炸死的局比例 | 比例 | 2–4 | 与 bomb survival 分开。 |
| `bomb_survival_rate` | 放过炸弹后存活的已结算比例 | 比例 | 2–4 | 分母定义须随源报告保留。 |
| `wait_rate` / `long_wait_rate` | WAIT 动作或长 WAIT 循环比例 | 源协议定义的比例 | 2 | 不同变体的阈值不同则标 `not_comparable`。 |
| `long_ping_pong_rate` | 长往返循环局比例 | 源协议定义的比例 | 2 | 不能用训练中计数替代冻结评估。 |
| `invalid_action_rate` | 不可执行动作比例 | invalid actions / decisions | 1–4 | 课程工程门槛 <=1%。 |
| `act_p95_ms`, `act_max_ms` | 完整 `act` 耗时的 P95/最大值（毫秒） | timing samples 分位数 / 最大值 | 1–4 | 课程硬上限 500 ms；不同机器不得做速度优劣结论。 |
| `timeout`, `skipped_action` | 超时或框架跳过动作计数 | 总计数 | 1–4 | 应为 0；不得省略。 |
