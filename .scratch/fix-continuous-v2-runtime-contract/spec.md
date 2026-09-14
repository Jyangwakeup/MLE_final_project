# Fix continuous-v2 runtime contract

`continuous-v2` 的历史特征必须在训练和冻结评估中使用相同的逐步语义，并在新回合开始时
重置。神经 Agent 的 epsilon 必须使用 Runner 注入并写入 checkpoint 的完整 exploration spec，
不能继续使用 Agent 静态超参数中的旧衰减步数。

