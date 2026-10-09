# Stage 2A：严格时间切分与可复现数据基础层

## 1. 范围

本阶段只落地时间切分、A/B/C 反馈语义、内部键策略、日历审计和防泄漏协议。未进行特征工程、召回、排序、训练或推荐指标实验。

## 2. 正式切分

| Snapshot | History | Labels | A history/labels | B history/labels |
| --- | --- | --- | ---: | ---: |
| validation | 2018-09-20 至 2020-08-25 | 2020-08-26 至 2020-09-08 | 30744934 / 547838 | 27628749 / 498874 |
| test/refit | 2018-09-20 至 2020-09-08 | 2020-09-09 至 2020-09-22 | 31292772 / 495552 | 28127623 / 447772 |

所有边界均为闭区间，且 history_end 严格早于 label_start。

## 3. 反馈语义

- A 保留原始行，仅供审计与敏感性分析。
- B 按 `t_dat + customer_id + article_id` 每日二值化，是主口径。
- C 在 B 上附加原始日内计数、cap 后计数和 `1 + log1p(capped_count - 1)`；不得称为真实 quantity。
- validation cap=3，输入 27628749 条 history B 记录。
- test/refit cap=3，输入 28127623 条 history B 记录。

两个 cap 分别只从各自 history 计算。

## 4. 内部键

采用 `full_dimension_lexicographic_dense_key_v1`：对完整 customers/articles 维表原始 ID 做字典序排序，再分配从 0 开始的稠密整数。customer/article 映射分别为 1371980 / 105542 行，一对一、稠密且覆盖维表。

映射不使用普通 hash；cold entity 也能获得稳定键，但键本身不得生成训练行为特征。含原始 ID 的映射只保存在被 Git 忽略的 `artifacts/stage_2a/private/`。

## 5. 日历审计

- validation 14 天：2 个完整星期，缺失日期 0，周末交易占比 28.2930%。
- test 14 天：2 个完整星期，缺失日期 0，周末交易占比 29.8826%。
- 7/14/28 天窗口分别包含 1/2/4 个完整星期周期。
- validation/test 原始交易分别为 547838 / 495552，日活用户均值分别为 12545.1 / 11307.4。
- validation 渠道 1/2 占比为 33.25%/66.75%；test 为 30.97%/69.03%。

两个 14 天窗口的星期构成一致，但交易量、周末占比、日活与渠道分布并不完全一致；相邻两个窗口也不足以证明季节性一致。详细 weekday、工作日/周末、活跃用户/商品与渠道统计见 `split_calendar_audit.csv`。

## 6. 防泄漏

10/10 项检查通过，包括日期边界、标签隔离、snapshot cap、内部键策略、as-of 和 manifest 一致性。

## 7. 可复现与资源

- dataset fingerprint：`f3b4d8bfed7655be122a6b43abf3f11cd526367b89cafefe6daf0c3e8a057927`
- split version：`1.0.0`
- feedback version：`B-primary_C-log1p-p99-v1`
- DuckDB 单线程，内存限制 1024 MiB，临时盘上限 20 GB。
- 大型反馈明细和 ID 映射不提交 Git；正式 manifest 仅保存计数与 fingerprint。

## 8. 结论与限制

Stage 2A 数据基础具备进入 Stage 2B 的条件，但仍需用户明确确认。当前没有生成召回候选、模型特征或训练样本，也没有运行任何推荐效果指标。
