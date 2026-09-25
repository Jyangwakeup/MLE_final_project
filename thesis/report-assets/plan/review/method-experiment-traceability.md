# 方法—实验—主张追溯

| Contribution | Method module | Experiment | Table / Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 课程型 Double Q(lambda) | 84-D `continuous-v2`、tile coding、Watkins traces、safety mask | Task 1 independent confirmation | Table 2; Figure 3 | 在该独立 100 局确认中达到 96% all-coins、48.95 mean coins。 | verified_summary |
| 空间 CNN value learning | 17-channel CNN、Double DQN、masked action selection | Task 1 distilled CNN reserved evaluation | Table 2; Figure 3 | 在该 reserved 100 局评估中达到 96% all-coins、49.84 mean coins。 | verified_summary |
| Team-teacher distillation | frozen team teacher 的软 Q targets；student inference 无 teacher | distilled CNN Task 1/2 | Table 1–2; Figure 2 | 蒸馏支持 Task 1；不能声称完成 Task 2 全金币。 | verified_summary |
| Q-learning Task 2 diagnostics | r20、history/tile/demo ablations | Task 2 frozen checkpoints | Table 2; Figure 3 | 安全炸箱不等于稳定完成隐藏金币；保留循环/WAIT 负结果。 | verified_summary |
| Task 3 curriculum transfer | same-seed parent/child Double DQN | confirmation and paired main validation | Table 3; Figure 3 | 主验证 score +1.23、first-place +12 pp；不得称 kills improved。 | verified_raw |
| Survival-mask engineering | survival-mask versions and CPU timing | historical safety/Task 4 reports | Figure 2; Figure 4 | 只能描述已测候选的安全/延迟观测；不声称完整安全认证。 | verified_raw / not_comparable |
| Die Hardest selection and packaging | Task4 B seed33/c200; rename/package isolation | historical 1,000-world benchmark + six paired equivalence games | Table 5; Figure 4 | 历史 benchmark 与 rename-equivalence 通过；非重新跑的 1,000-world，且未完成 Docker/official hardware/full safety。 | verified_summary / packaging_verified |
| Task 4 alternatives | E1/E2/E3, frozen C/S, score campaign | bounded frozen reports | Table 4; Figure 4 | 这些专用合同中未观察稳定 score gain；未运行阶段必须写 unrun。 | verified_raw / not_comparable |
