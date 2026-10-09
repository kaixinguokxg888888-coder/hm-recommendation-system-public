# Stage 1A：全量完整性与基础数据质量审计

## 1. 执行环境与护栏

- 执行时间：`2026-07-27T20:18:42+08:00`
- Python：`3.10.11`（项目 `.venv`）
- DuckDB：`1.5.5`
- memory_limit 策略：每组启动时取可用内存的 35%，限制在 256–1024 MiB；各组实际值见资源表。
- threads：`1`
- temp_directory：`artifacts/stage_1a/tmp`
- max_temp_directory_size：`20GB`
- 单查询软超时：`1800` 秒；超时由 DuckDB interrupt 中断。
- retry：最多 `1` 次；本配置不自动重试重型查询。
- preserve_insertion_order：`False`

所有原始 Parquet 只读；没有 pandas 全量物化、清洗、去重、特征或模型代码。

## 2. 全量扫描范围

| Table | Rows | Row groups | Bytes | Fingerprint |
| --- | --- | --- | --- | --- |
| articles | 105542 | 1 | 6378835 | b32d5784aa3a8722… |
| customers | 1371980 | 2 | 167711266 | d768b8937a423769… |
| transactions | 31788324 | 31 | 774279764 | 630a9f8f9b68ca8d… |

Group A 只读取完整 footer/schema；Group B–F 的正式指标均以 `execution_scope=FULL_SCAN` 标记。`rows_scanned` 表示该检查的逻辑输入行数，不等同于多查询累计物理行访问次数。

## 3. Schema、空值与格式结果

- 三张表均完整可读，schema 与数据契约一致，fingerprint 与 Stage 0A 一致。
- ID、postal code、日期及价格格式结果详见统一结果 CSV。

| Table | Column | Null count | Null rate |
| --- | --- | --- | --- |
| articles | article_id | 0 | 0.0 |
| articles | product_code | 0 | 0.0 |
| articles | prod_name | 0 | 0.0 |
| articles | product_type_no | 0 | 0.0 |
| articles | product_type_name | 0 | 0.0 |
| articles | product_group_name | 0 | 0.0 |
| articles | graphical_appearance_no | 0 | 0.0 |
| articles | graphical_appearance_name | 0 | 0.0 |
| articles | colour_group_code | 0 | 0.0 |
| articles | colour_group_name | 0 | 0.0 |
| articles | perceived_colour_value_id | 0 | 0.0 |
| articles | perceived_colour_value_name | 0 | 0.0 |
| articles | perceived_colour_master_id | 0 | 0.0 |
| articles | perceived_colour_master_name | 0 | 0.0 |
| articles | department_no | 0 | 0.0 |
| articles | department_name | 0 | 0.0 |
| articles | index_code | 0 | 0.0 |
| articles | index_name | 0 | 0.0 |
| articles | index_group_no | 0 | 0.0 |
| articles | index_group_name | 0 | 0.0 |
| articles | section_no | 0 | 0.0 |
| articles | section_name | 0 | 0.0 |
| articles | garment_group_no | 0 | 0.0 |
| articles | garment_group_name | 0 | 0.0 |
| articles | detail_desc | 416 | 0.003941558810710428 |
| customers | customer_id | 0 | 0.0 |
| customers | FN | 895050 | 0.6523783145526902 |
| customers | Active | 907576 | 0.6615081852505138 |
| customers | club_member_status | 6062 | 0.00441843175556495 |
| customers | fashion_news_frequency | 16011 | 0.011669995189434249 |
| customers | age | 15861 | 0.011560664149623172 |
| customers | postal_code | 0 | 0.0 |
| transactions | t_dat | 0 | 0.0 |
| transactions | customer_id | 0 | 0.0 |
| transactions | article_id | 0 | 0.0 |
| transactions | price | 0 | 0.0 |
| transactions | sales_channel_id | 0 | 0.0 |

## 4. 日期和值域结果

- 交易日期：`{"min_date": "2018-09-20", "max_date": "2020-09-22", "parse_failure_count": 0}`
- 价格：`{"min": 1.694915254237288e-05, "max": 0.5915254237288136, "mean": 0.02782927385390111, "quantiles": {"p01": 0.0035762711864406, "p25": 0.0158135593220338, "p50": 0.0254067796610169, "p75": 0.0338813559322033, "p99": 0.1002203389830508}, "negative_count": 0, "zero_count": 0}`
- 年龄：`{"min": 16.0, "max": 99.0, "mean": 36.386964565794, "quantiles": {"p01": 18.0, "p25": 24.0, "p50": 32.0, "p75": 49.0, "p99": 71.0}, "null_count": 15861, "out_of_range": 0, "age_0_17": 9553, "age_18_24": 347616, "age_25_64": 955348, "age_65_120": 43602}`
- sales_channel_id 频数见 `transactions.full.sales_channel.frequency`。

## 5. 主键唯一性

| Table | Rows | Non-null | Distinct | Duplicate keys | Excess rows |
| --- | --- | --- | --- | --- | --- |
| articles | 105542 | 105542 | 105542 | 0 | 0 |
| customers | 1371980 | 1371980 | 1371980 | 0 | 0 |

表中只包含计数，不包含具体 customer_id 或 article_id。

## 6. 主外键覆盖

- transactions distinct 实体：`{"distinct_customer_count": 1362281, "distinct_article_count": 104547}`

| Relationship | Unmatched rows | Unmatched distinct | Row match rate |
| --- | --- | --- | --- |
| transactions.customer_id->customers.customer_id | 0 | 0 | 1.0 |
| transactions.article_id->articles.article_id | 0 | 0 | 1.0 |

外键检查使用全量 anti join，不保存未匹配 ID。

## 7. 交易重复多口径结果

- 全字段完全相同：`{"full_identical_duplicate_groups": 2543908, "full_identical_duplicate_excess_rows": 2974905, "full_identical_max_repetition": 569, "rows_in_repeated_full_groups": 5518813, "duplicate_excess_ratio": 0.09358483322367042, "rows_in_repeated_groups_ratio": 0.17361132345322766, "group_count_buckets": {"1": 26269511, "2": 2285073, "3": 177237, "4": 55412, "5+": 26186}}`
- `t_dat + customer_id + article_id`：`{"same_date_customer_article_multi_groups": 2717844, "same_date_customer_article_excess_rows": 3212929, "max_repetition": 570, "excess_ratio": 0.10107261395725047}`
- 日期分布摘要：`{"date_count": 734, "dates_with_repeated_records": 734, "max_daily_excess_rows": 21718}`
- 当前 transactions 恰好只有五个字段，因此“全字段完全相同”与 `t_dat + customer_id + article_id + price + sales_channel_id` 是同一口径，报告中显式复用同一结果。

这些现象只能称为记录重复或同条件重复购买。由于缺少 order_id、精确时间戳和 quantity，不能判断是数据错误、同单多件、重复购买还是业务导出语义，因此本阶段绝不执行 `drop_duplicates`。

## 8. 资源占用与性能

| Group | Status | Memory MiB | Threads | Seconds | Peak RSS | Peak temp | Spill | Retry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| group_a_metadata_schema | SUCCESS | 512 | 1 | 0.055861 | 50946048 | 0 | False | False |
| group_b_null_format | SUCCESS | 512 | 1 | 29.295556 | 75206656 | 0 | False | False |
| group_c_date_value_ranges | SUCCESS | 512 | 1 | 9.427079 | 551768064 | 0 | False | False |
| group_d_primary_keys | SUCCESS | 512 | 1 | 0.668469 | 206622720 | 0 | False | False |
| group_e_foreign_keys | SUCCESS | 512 | 1 | 20.900176 | 242982912 | 0 | False | False |
| group_f_transaction_duplicates | SUCCESS | 617 | 1 | 100.11211 | 710352896 | 2345404225 | True | True |

汇总：spill=`True`，retry=`True`，OOM=`True`。

历史失败与降级记录：

- `group_f_transaction_duplicates` 历史尝试 1：status=FAILED，seconds=1.882161，peak_rss=590172160 bytes，peak_temp=121733569 bytes，error=`OutOfMemoryException: Out of Memory Error: failed to allocate data of size 2.0 MiB (486.9 MiB/487.3 MiB used)`
- `group_f_transaction_duplicates` 历史尝试 2：status=FAILED，seconds=11.236423，peak_rss=602611712 bytes，peak_temp=2345404225 bytes，error=`OutOfMemoryException: Out of Memory Error: could not allocate block of size 256.0 KiB (489.0 MiB/489.2 MiB used)`

F 组最终通过临时、精确的一对一整数代理键完成重复分组；代理键不落入正式结果，原始 ID 与 Parquet 均未修改。

## 9. 测试结果

- 命令：`python -m pytest`
- 结果：`PASS`
- 摘要：`24 passed in 0.42s`
- 耗时：`1.073263` 秒

测试使用 pytest 临时目录和小型临时 Parquet，不依赖完整 H&M 原始数据。

## 10. 失败、警告和未完成项

- `transactions.full.duplicate.full_row`：WARN；记录重复不等于脏数据；缺少 order_id、时间戳和 quantity，不能判断业务含义。

Stage 1A 没有验证外部发布校验和，也没有赋予交易重复以“脏数据”标签。

## 11. 对数据清洗的影响

- 空值、格式、范围、主外键和重复统计应成为后续清洗决策输入，但本阶段未修改任何值。
- 年龄和价格的边界统计决定后续是否需要异常值策略。
- 重复记录必须结合业务语义或新增 order_id/quantity 证据后才能决定是否处理。

## 12. 对推荐架构的初步影响

- distinct 用户/商品规模决定用户与商品索引、稀疏交互矩阵及候选生成的容量设计。
- 主外键覆盖决定交互能否安全关联用户与商品侧信息。
- 同日同商品重复记录可能影响隐式反馈强度定义，但不能在此阶段直接压缩为一次事件。
- 日期范围和分布决定时间切分、回测窗口与防止未来信息泄露的策略。

## 13. 是否建议进入 Stage 1B

**有条件建议进入 Stage 1B**。

理由：六个全量检查组、扩展测试和资源护栏均完成；进入前应先评审本报告中的警告与重复语义。

本报告没有执行 Stage 1B。
