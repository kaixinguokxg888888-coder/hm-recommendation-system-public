# Stage 0B：数据契约、审计框架与受限 Smoke Test

## 1. Stage 0B 目标

本阶段仅验证数据契约、DuckDB/PyArrow 审计框架、统一结果协议和资源边界；未执行正式全量 transaction 审计、清洗、特征工程、召回、排序或模型开发。

## 2. 数据契约摘要

- 契约版本：`0.1.0`
- 逻辑表：articles、customers、transactions
- 每条规则均包含 check ID、表、字段、规则类型、严重程度、预期和理由。
- sales channel 合法集合不硬编码，只记录 smoke 范围实际观察值。
- postal code、customer ID、article ID 只输出计数或格式，不输出真实值。

证据：`configs/data_contracts.yaml`。

## 3. Smoke 扫描范围

- articles：完整文件 data/raw/articles.parquet；105542 行；1/1 row groups
- customers：完整文件 data/raw/customers.parquet；1371980 行；2/2 row groups
- transactions：文件 data/raw/transactions_train.parquet 的 row group 0 前 100000 行；该 row group 共 1048576 行；文件共 31788324 行/31 row groups
- DuckDB 另对三个 Parquet 各执行一次直接 `LIMIT 1` 可读性探测；该探测不被误作聚合范围。

transactions 的所有聚合均在首个 row group 前 `100000` 行已物化临时表上执行，`LIMIT` 没有放在全表聚合之后。

## 4. 每条检查结果

| Check | Status | Severity | Rows scanned | Seconds | Observed |
| --- | --- | --- | ---: | ---: | --- |
| articles.duckdb.direct_parquet_probe | PASS | error | 1 | 0.017729 | {"rows_returned": 1} |
| customers.duckdb.direct_parquet_probe | PASS | error | 1 | 0.008839 | {"rows_returned": 1} |
| transactions.duckdb.direct_parquet_probe | PASS | error | 1 | 0.011238 | {"rows_returned": 1} |
| articles.schema.required_fields | PASS | error | 0 | 0.000002 | {"field_count": 25, "missing_fields": [], "type_mismatches": {}} |
| articles.schema.fingerprint_stable | PASS | error | 0 | 0.000003 | {"fingerprint_prefix": "b32d5784aa3a8722", "matches_stage_0a": true} |
| customers.schema.required_fields | PASS | error | 0 | 0.000002 | {"field_count": 7, "missing_fields": [], "type_mismatches": {}} |
| customers.schema.fingerprint_stable | PASS | error | 0 | 0.000003 | {"fingerprint_prefix": "d768b8937a423769", "matches_stage_0a": true} |
| transactions.schema.required_fields | PASS | error | 0 | 0.000002 | {"field_count": 5, "missing_fields": [], "type_mismatches": {}} |
| transactions.schema.fingerprint_stable | PASS | error | 0 | 0.000002 | {"fingerprint_prefix": "630a9f8f9b68ca8d", "matches_stage_0a": true} |
| articles.article_id.not_null | PASS | error | 105542 | 0.001885 | {"violation_count": 0} |
| articles.article_id.format | PASS | error | 105542 | 0.015504 | {"violation_count": 0} |
| articles.article_id.candidate_key | PASS | error | 105542 | 0.019226 | {"duplicate_excess_rows": 0} |
| articles.attributes.null_profile | PASS | info | 105542 | 0.050741 | {"colour_group_code": 0, "colour_group_name": 0, "department_name": 0, "department_no": 0, "detail_desc": 416, "garment_group_name": 0, "garment_group_no": 0, "graphical_appearance_name": 0, "graphical_appearance_no": 0, "index_code": 0, "index_group_name": 0, "index_group_no": 0, "index_name": 0, "perceived_colour_master_id": 0, "perceived_colour_master_name": 0, "perceived_colour_value_id": 0, "perceived_colour_value_name": 0, "prod_name": 0, "product_code": 0, "product_group_name": 0, "product_type_name": 0, "product_type_no": 0, "section_name": 0, "section_no": 0} |
| customers.customer_id.not_null | PASS | error | 1371980 | 0.001711 | {"violation_count": 0} |
| customers.customer_id.format | PASS | error | 1371980 | 0.535041 | {"violation_count": 0} |
| customers.customer_id.candidate_key | PASS | error | 1371980 | 0.415980 | {"duplicate_excess_rows": 0} |
| customers.flags.nullability | PASS | info | 1371980 | 0.042121 | {"Active": 907576, "FN": 895050, "age": 15861} |
| customers.age.reasonable | PASS | warning | 1371980 | 0.133808 | {"age_0_17": 9553, "age_18_24": 347616, "age_25_64": 955348, "age_65_120": 43602, "null": 15861, "out_of_range": 0} |
| customers.postal_code.privacy | PASS | error | 1371980 | 0.554096 | {"violation_count": 0} |
| transactions.t_dat.parseable | PASS | error | 100000 | 0.007145 | {"violation_count": 0} |
| transactions.customer_id.not_null | PASS | error | 100000 | 0.000774 | {"violation_count": 0} |
| transactions.article_id.not_null | PASS | error | 100000 | 0.000510 | {"violation_count": 0} |
| transactions.price.not_null | PASS | error | 100000 | 0.000488 | {"violation_count": 0} |
| transactions.price.non_negative | PASS | error | 100000 | 0.000441 | {"violation_count": 0} |
| transactions.sales_channel_id.observed_domain | PASS | info | 100000 | 0.003273 | {"observed_values": [1, 2]} |
| transactions.full_row.duplicates | WARN | warning | 100000 | 0.032552 | {"duplicate_excess_rows": 9500} |
| transactions.customer_id.foreign_key | PASS | warning | 100000 | 0.313124 | {"unmatched_rows": 0} |
| transactions.article_id.foreign_key | PASS | warning | 100000 | 0.026615 | {"unmatched_rows": 0} |

状态汇总：PASS=27，FAIL=0，WARN=1，ERROR=0，NOT_RUN=0。

## 5. 资源使用

- 总耗时：`3.197638` 秒
- 进程峰值 RSS：`301858816` bytes
- DuckDB 内存限制：`458` MiB
- DuckDB threads：`2`
- 临时目录：`artifacts/stage_0b/tmp`
- 临时磁盘峰值：`0` bytes
- 观察到 spill：`False`
- OOM：`False`
- 输出文件大小：`{"artifacts/stage_0b/audit_results.csv": 10881, "artifacts/stage_0b/audit_summary.json": 1416, "artifacts/stage_0b/resource_usage.json": 749, "docs/stages/stage_0b_contract_and_smoke.md": 7569}`

## 6. 测试结果

- 命令：`python -m pytest`
- 结果：`PASS`
- 返回码：`0`
- 耗时：`0.798599` 秒
- pytest 摘要：`10 passed in 0.24s`

测试不依赖完整 H&M 原始数据；临时 Parquet 由 pytest 临时目录创建并清理。

## 7. 失败与异常

- `transactions.full_row.duplicates`：WARN；重复数只描述受限交易样本，不外推到全量。

## 8. 当前框架局限

- transactions 只覆盖一个 row group 的固定前缀，无法代表完整时间跨度、全量唯一值、全量重复或全量外键覆盖。
- articles/customers 的值检查覆盖当前完整维表，但没有与外部权威校验和比较。
- DuckDB spill 结论只针对本次 smoke 查询和当时资源状态。
- schema fingerprint 稳定只表示相对 Stage 0A 未变化，不证明数据内容完整。

## 9. 局部结论边界

任何 smoke PASS 仅说明对应 SQL/契约在报告所列扫描范围内通过，**不得解释为全量 PASS 或数据百分之百完整**。交易检查尤其不能外推到其余 row groups。

## 10. 是否建议进入 Stage 1

结论：**有条件建议进入 Stage 1**。

理由：三个 Parquet 的 DuckDB 受限读取、契约检查、统一协议和测试均可运行；进入 Stage 1 前仍须保留分步执行与资源护栏。

## 11. Stage 1 风险与建议顺序

1. 先冻结契约与结果协议，并为全量查询设置磁盘、内存和超时护栏。
2. 先做低基数空值、日期范围和值域，再做 customer/article 高基数 distinct。
3. 随后执行主键唯一性和 transaction 完全重复检查，监控 spill 与临时磁盘。
4. 最后执行全量外键覆盖率；该步骤可能同时构建大哈希表并扫描交易列。
5. 每一步单独落盘小型指标并可恢复，禁止物化或提交原始记录。
