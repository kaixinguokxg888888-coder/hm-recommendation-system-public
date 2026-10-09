# Stage 2B：标签集、可评用户与评估对象协议

## 1. 范围

只基于 Stage 2A validation/test snapshot 构建 B 口径 ground truth、可评人群、用户分群、商品状态和候选域协议。未实现召回、特征、模型或推荐指标。

## 2. Ground truth 与人群

| Snapshot | user-item positives | label users | evaluable warm | cold-history | cold items | history popularity P50/P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation_14d | 489723 | 141359 | 129705 | 11654 | 1816 | 59 / 721 |
| test_14d | 438380 | 128333 | 117366 | 10967 | 1580 | 59 / 720 |

每个 user-item 在标签窗口只保留一条正样本，同时保存首次/末次标签日期、购买天数和日级强度总和。私有表只含内部 key。

## 3. 每用户正样本分布

| Snapshot | 1 | 2 | 3–5 | 6–10 | 10+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_14d | 39668 | 32341 | 45611 | 18298 | 5441 |
| test_14d | 37108 | 29805 | 40379 | 16201 | 4840 |

## 4. 主评估与分群

- all_label_users：窗口内至少一个正样本。
- evaluable_warm_users：有标签且 cutoff 前至少一个 B 历史交互，是协同召回主评估集合。
- cold_history_label_users：有标签但无历史，必须单独报告。
- 用户分群沿用 Stage 1C：cold=0、short_history=1–4、warm=5–20、active=21+。

| Snapshot | Segment | Users | Positive pairs | Label coverage |
| --- | --- | ---: | ---: | ---: |
| validation_14d | cold | 11654 | 35411 | 7.23% |
| validation_14d | short_history | 9897 | 25560 | 5.22% |
| validation_14d | warm | 33620 | 95068 | 19.41% |
| validation_14d | active | 86188 | 333684 | 68.14% |
| test_14d | cold | 10967 | 33449 | 7.63% |
| test_14d | short_history | 8653 | 22118 | 5.05% |
| test_14d | warm | 29722 | 82408 | 18.80% |
| test_14d | active | 78991 | 300405 | 68.53% |

## 5. 商品状态与流行度

- seen_item：cutoff 前 history 出现。
- cold_item：标签出现但 history 未出现。
- zero_history_dimension_item：维表存在但 history 无交易。
- popularity 只由各 snapshot history B 频次计算：tail < P50，mid 为 P50 至 P90 以下，head >= P90。

| Snapshot | Cold items | Zero-history dimension items | Head labels | Mid labels | Tail labels | Cold-item labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation_14d | 1816 | 4391 | 160001 | 212458 | 85366 | 31898 |
| test_14d | 1580 | 2575 | 142187 | 192933 | 64886 | 38374 |

## 6. 候选域

- V1 主评估使用 history seen items。
- cold-item 评估使用完整 articles 维表。
- 缺少上架时间和库存，无法严格排除标签窗口之后才业务可见的商品；完整维表只代表身份域，不代表历史可售域。
- 本阶段没有生成任何召回候选。

## 7. 防泄漏与隐私

24/24 项 snapshot 级检查通过。Validation/test ground truth 分开保存；分群、热度和 cold 判定只使用各自 cutoff 前 history。公开产物仅含聚合统计，私有用户级文件被 Git 忽略。

## 8. 可复现性

- dataset fingerprint：`f3b4d8bfed7655be122a6b43abf3f11cd526367b89cafefe6daf0c3e8a057927`
- population version：`1.0.0`
- feedback version：`B-primary_C-log1p-p99-v1`
- 所有私有产物均记录 fingerprint，可由 Stage 2A 基础层重建。

## 9. 结论

Stage 2B 具备支持后续 baseline 阶段的 ground truth 和统一评估对象，但进入下一阶段仍需用户明确确认。
