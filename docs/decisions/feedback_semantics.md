# 反馈语义决策

决策编号：`FB-001`

状态：`FROZEN`

## 决策

B 口径 `t_dat + customer_id + article_id` 日内二值交互作为 V1 主训练标签和主评估标签。

| 口径 | V1 用途 | 禁止解释 |
| --- | --- | --- |
| A 原始行 | 数据审计、流行度/强度敏感性对照 | 不等同订单数或购买件数 |
| B user-item-day 二值 | 主召回、主训练标签、主评估标签 | 不等同曝光后的偏好概率 |
| C daily interaction count | 辅助强度特征和敏感性实验 | 不得称为真实 `quantity` |

## 证据

- A 共 31,788,324 行，B 共 28,575,395 个二值交互，差异为 3,212,929。
- 全字段重复超额 2,974,905 行，占 9.36%；同日 user-item 超额占 10.11%。
- 完全重复最大次数 569，同日 user-item 最大次数 570。
- 数据缺少 order_id、quantity 和精确时间，因此无法区分多件购买、重复写入或多笔订单。

来源：`artifacts/stage_1a/audit_results.csv`、`artifacts/stage_1b/profile_summary.json`。

## 辅助强度规则

C 口径只在请求时点前计算：

```text
cap = 训练期 daily_interaction_count 的 P99
strength = 1 + log1p(min(daily_interaction_count, cap) - 1)
```

`cap` 必须只从相应训练窗口估计并写入实验配置。不得用 validation/test 计算截断阈值。若 P99 仍受极端值支配，P1 增加 P95 与固定上限对照。

## 敏感性实验

- 主实验固定 B。
- P1 对比 A、B、B+C 强度，不同时改变模型、切分或候选配额。
- 成功标准：在 14 天 validation 上改善目标指标，且 7/28 天方向一致，分群没有不可接受退化。
- 停止条件：增益不稳定、仅来自重复极端值、或资源成本不成比例。

## 风险

- B 会丢弃真实同日多件购买的潜在强度；
- C 仍可能混合业务购买和重复记录；
- A 容易放大极端重复、热门度和复购信号。

正式记录见 `artifacts/stage_1c/architecture_decisions.csv`。
