# s185 formal_inputs 重建检查报告

## 结论

已从现有 `s185` 数据集 manifest、整数树 artifact、VIPER 提取脚本和当前代码重建 `formal_inputs`，未重新训练 DQN，也未重新运行 VIPER 提取。

## 已通过检查

- 任务数：12
- 特征维度：128
- 动作维度：24
- task/feature 顺序一致：True
- task/action 顺序一致：True
- artifact feature 与 target 一致：True
- artifact action 与 target 一致：True
- artifact taskset seed：185
- fixed-point config hash：`bb255908bec90f8d54945322db333c5c2d2df5af1c5e76fc9016dc8c975a381e`
- canonical taskset fingerprint：`533ea598787c725581ff9252c8f276a12137d5b4de0c79bbb588a7fd8498be12`
- current source semantic hash：`897074ac1be33fff35f80310c94bb5c31baab5f5061b9fb6041e8d41ebe59d1b`

## 权威优先级顺序

```text
mc_lo_0, mc_hi_1, mc_hi_3, mc_hi_0, mc_hi_2, mc_hi_4, mc_hi_5, mc_lo_1, mc_lo_3, mc_lo_5, mc_lo_2, mc_lo_4
```

## 注意事项

训练 manifest 中的 reward mode `interval_qos_v2_single_recovery_full_C5_overinc016_abs005` 在当前代码中已不存在。重建 target 使用当前可用的 `mendes` 作为环境构造占位；reward mode 不进入 P0 formal target 配置，并且验证表明任务集、128 个 feature 与 24 个 action 均与原 artifact 完全一致。

`phase_k_case_map.json` 已按当前代码重新生成，因此它只适用于当前归档源码；以后修改被绑定的运行时代码后需要重新生成。

## 实际单命令接入检查

重建后的 seed 已经成功通过 seed 导入、target factory 构造、tree artifact 检查和 compiler candidate 生成，不再返回 `AUTHORITATIVE_TARGET_MISSING`。

当前单命令继续进入证明链后，暴露的是工具链实现问题，而不是 formal_inputs 缺失：

1. `BOOT_INITIALIZATION` 返回 `FAIL`：当前 `AMCRealRuntimeAdapter.export_boot_transition_contract()` 在 `environment.reset()` 已处理 t=0 到达后读取状态，因此看到 `mode=HI` 且已有运行 job；它与工具链所要求的“boot 前/boot closure 前初始状态”定义不一致。
2. `CANDIDATE_ENVELOPE` 返回 `UNRESOLVED / FINITE_DOMAIN_TOO_LARGE`：12 个任务的笛卡尔预算域状态数为 `5129622227014718695527589632000`，当前实现无法直接全枚举。
3. 完整 verifier 在测试窗口内未结束，但 compiler 已完整落盘 candidate artifacts，说明 formal_inputs 已经被正常消费。

因此，本次重建解决了输入缺失；后续仍需修复 real runtime boot-contract 取值时点和真实 seed 的预算域证明方法。
