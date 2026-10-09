# Stage 2B 评估对象协议

状态：`FROZEN`

## 1. Ground truth

每个 snapshot 使用 Stage 2A 的 B 口径标签窗口。一个 `(user_key, article_key)` 在同一窗口只保留一条正样本，并记录：

- `label_date_min` / `label_date_max`
- `user_item_label_days`
- `label_daily_strength_sum`
- `as_of_date`
- `label_start` / `label_end`
- `feedback_version`

Validation 和 test 分开保存。Ground truth 只含内部整数键，不含原始 customer/article ID。

## 2. 人群集合

| 集合 | 定义 | 用途 |
| --- | --- | --- |
| `all_label_users` | 标签窗口至少一个正样本 | 标签总体与覆盖分母 |
| `evaluable_warm_users` | 有标签，cutoff 前至少一个 B 历史交互，内部键有效 | V1 协同召回主评估 |
| `cold_history_label_users` | 有标签，cutoff 前历史为 0 | Cold-history 独立评估 |
| `cold_item_evaluation` | 至少一个正样本商品在 cutoff 前未见 | Cold-item 独立评估 |

后续模型不得自行改变用户集合。任何额外过滤必须注册为新的 population version，并同时报告本协议集合。

## 3. 用户分群

只使用 cutoff 前 B 交互数：

- `cold`: 0
- `short_history`: 1–4
- `warm`: 5–20
- `active`: 21+

阈值来自 Stage 1C 冻结协议，不在 Stage 2B 调参。Cold 用户不得混入协同召回总体而不单列。

## 4. 商品状态

- `seen_item`：cutoff 前至少一个 B 交互；
- `cold_item`：标签窗口出现、history 未出现；
- `zero_history_dimension_item`：articles 维表存在、history 无交易；
- `head/mid/tail`：只对 seen item，按 snapshot history B 频次划分：
  - tail `< P50`
  - mid `>= P50 and < P90`
  - head `>= P90`

阈值按 validation/test history 分别计算，不使用标签。

## 5. 评估对象字段

私有用户级表固定包含：

- `snapshot_id`
- `user_key`
- `user_segment`
- `history_interaction_count`
- `history_distinct_item_count`
- `positive_item_count`
- `warm_positive_count`
- `cold_positive_count`
- `has_cold_positive`
- `is_main_evaluable`
- `as_of_date`
- `ground_truth_fingerprint`

正式公开产物只输出聚合统计。

## 6. 候选商品域

| 评估 | 候选域 |
| --- | --- |
| V1 协同召回主评估 | cutoff 前 `seen_item` |
| Cold-item 评估 | 完整 articles 维表 |
| Cold-history 用户 | 完整维表或明确的 fallback 候选域，必须独立报告 |

数据缺少商品上架时间和库存，因此完整 articles 维表只是身份域，不等于历史时点可售域。无法可靠排除“标签窗口后才业务上架”的商品；任何结果必须保留这一限制。

## 7. 版本与隐私

所有表携带 dataset fingerprint、population version、feedback version、snapshot 和 as-of 日期。用户级 ground truth、评估对象、分群和商品状态文件位于被 Git 忽略的 `artifacts/stage_2b/private/`，并记录文件 fingerprint。
