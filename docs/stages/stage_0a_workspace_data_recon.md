# 阶段 0A：工作区与 Parquet 数据只读侦察

## 1. 执行信息

- 执行时间：`2026-07-27T15:53:37+08:00`
- 工作区：`.`（相对于项目根目录）
- 实际命令：`python scripts/recon_workspace.py`
- 约束：只读取文件系统、Parquet footer/schema 和每张逻辑表首个 row group 的前 5 行；未执行全量 groupby、join、sort、去重或训练。

## 2. 工作区与目录概况

- `artifacts/`：stage_0a/
- `data/`：raw/, README.md
- `docs/`：stages/
- `scripts/`：recon_workspace.py
- `src/`：空目录
- `tests/`：空目录
- `.gitignore`：421 bytes
- `pyproject.toml`：346 bytes
- `README.md`：2,322 bytes

- 文件扩展名汇总：`.csv` 1 个、`.json` 1 个、`.md` 3 个、`.parquet` 3 个、`.py` 1 个、`.toml` 1 个、`<无扩展名>` 1 个
- Git：未初始化；fatal: not a git repository (or any of the parent directories): .git。【证据：环境命令】
- 原始数据位于 `data/raw/`；侦察前其余顶层目录为空，未发现既有 Python、Notebook、README、requirements、pyproject、environment.yml 或 AGENTS.md。数据与最小项目目录处在同一工作区，但没有与代码文件混放在 `data/raw/` 内。【证据：文件系统遍历】

## 3. Parquet 文件清单

| 逻辑表 | 相对路径 | 字节 | 行数 | Row groups | 压缩 | 格式版本 | 可读 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| articles | data/raw/articles.parquet | 6,378,835 | 105542 | 1 | SNAPPY | 2.6 | 是 |
| customers | data/raw/customers.parquet | 167,711,266 | 1371980 | 2 | SNAPPY | 2.6 | 是 |
| transactions | data/raw/transactions_train.parquet | 774,279,764 | 31788324 | 31 | SNAPPY | 2.6 | 是 |

共 `3` 个文件，合计 `948,369,865` bytes（904.44 MiB）。【证据：文件元数据与 Parquet footer】

## 4. 数据组织与逻辑表识别

组织方式判定：**每张识别出的逻辑表对应一个单独 Parquet 文件**；未观察到 Hive `key=value` 分区路径，也未观察到 Hugging Face 风格分片命名。【证据：相对路径与文件名】

| 逻辑表 | 文件数 | 总字节 | 总行数 | Row groups | Schema | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| articles | 1 | 6378835 | 105542 | 1 | 一致 | data/raw/articles.parquet |
| customers | 1 | 167711266 | 1371980 | 2 | 一致 | data/raw/customers.parquet |
| transactions | 1 | 774279764 | 31788324 | 31 | 一致 | data/raw/transactions_train.parquet |

逻辑表名由文件 stem 保守推断，其中 `_train` 后缀仅作为数据集角色后缀去除；未重命名原文件。

## 5. 完整 schema

### `articles`

| 字段 | 物理类型 | Parquet 逻辑类型 | Arrow 类型 | 可空 | 观察到字典编码 | 特殊类型 |
| --- | --- | --- | --- | --- | --- | --- |
| article_id | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| product_code | INT64 | None | int64 | 是 | 是 | 否 |
| prod_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| product_type_no | INT64 | None | int64 | 是 | 是 | 否 |
| product_type_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| product_group_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| graphical_appearance_no | INT64 | None | int64 | 是 | 是 | 否 |
| graphical_appearance_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| colour_group_code | INT64 | None | int64 | 是 | 是 | 否 |
| colour_group_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| perceived_colour_value_id | INT64 | None | int64 | 是 | 是 | 否 |
| perceived_colour_value_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| perceived_colour_master_id | INT64 | None | int64 | 是 | 是 | 否 |
| perceived_colour_master_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| department_no | INT64 | None | int64 | 是 | 是 | 否 |
| department_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| index_code | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| index_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| index_group_no | INT64 | None | int64 | 是 | 是 | 否 |
| index_group_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| section_no | INT64 | None | int64 | 是 | 是 | 否 |
| section_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| garment_group_no | INT64 | None | int64 | 是 | 是 | 否 |
| garment_group_name | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| detail_desc | BYTE_ARRAY | String | string | 是 | 是 | 否 |

分片字段类型一致性：一致。【证据：Parquet schema/footer】
### `customers`

| 字段 | 物理类型 | Parquet 逻辑类型 | Arrow 类型 | 可空 | 观察到字典编码 | 特殊类型 |
| --- | --- | --- | --- | --- | --- | --- |
| customer_id | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| FN | DOUBLE | None | double | 是 | 是 | 否 |
| Active | DOUBLE | None | double | 是 | 是 | 否 |
| club_member_status | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| fashion_news_frequency | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| age | DOUBLE | None | double | 是 | 是 | 否 |
| postal_code | BYTE_ARRAY | String | string | 是 | 是 | 否 |

分片字段类型一致性：一致。【证据：Parquet schema/footer】
### `transactions`

| 字段 | 物理类型 | Parquet 逻辑类型 | Arrow 类型 | 可空 | 观察到字典编码 | 特殊类型 |
| --- | --- | --- | --- | --- | --- | --- |
| t_dat | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| customer_id | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| article_id | BYTE_ARRAY | String | string | 是 | 是 | 否 |
| price | DOUBLE | None | double | 是 | 是 | 否 |
| sales_channel_id | INT64 | None | int64 | 是 | 是 | 否 |

分片字段类型一致性：一致。【证据：Parquet schema/footer】

当前字段集合能够分别对应 H&M 的 article 主数据、customer 主数据和 transaction 事件数据；对应关系来自实际字段语义，不等于对公开版本完整性的证明。未发现可对应 sample submission 的本地文件。【证据：文件名与 schema】

## 6. Parquet 元数据

- `data/raw/articles.parquet`：created_by=`parquet-cpp-arrow version 20.0.0`，format_version=`2.6`，compression=`SNAPPY`，pandas metadata=有，schema fingerprint=`b32d5784aa3a8722…`。【证据：Parquet footer】
- `data/raw/customers.parquet`：created_by=`parquet-cpp-arrow version 20.0.0`，format_version=`2.6`，compression=`SNAPPY`，pandas metadata=有，schema fingerprint=`d768b8937a423769…`。【证据：Parquet footer】
- `data/raw/transactions_train.parquet`：created_by=`parquet-cpp-arrow version 20.0.0`，format_version=`2.6`，compression=`SNAPPY`，pandas metadata=有，schema fingerprint=`630a9f8f9b68ca8d…`。【证据：Parquet footer】

编码说明：“观察到字典编码”来自 row-group column chunk 的 encoding 列表，并不表示 Arrow 字段本身是 dictionary 类型。

## 7. 安全抽样观察

### `articles`

| 字段 | Arrow 类型 | 样本空值数 | 截断后的值形态（最多 2 个） |
| --- | --- | --- | --- |
| article_id | string | 0 | <string:length=10,charset=digits> |
| product_code | int64 | 0 | 108775; 110065 |
| prod_name | string | 0 | Strap top; Strap top (1) |
| product_type_no | int64 | 0 | 253; 306 |
| product_type_name | string | 0 | Vest top; Bra |
| product_group_name | string | 0 | Garment Upper body; Underwear |
| graphical_appearance_no | int64 | 0 | 1010016; 1010017 |
| graphical_appearance_name | string | 0 | Solid; Stripe |
| colour_group_code | int64 | 0 | 9; 10 |
| colour_group_name | string | 0 | Black; White |
| perceived_colour_value_id | int64 | 0 | 4; 3 |
| perceived_colour_value_name | string | 0 | Dark; Light |
| perceived_colour_master_id | int64 | 0 | 5; 9 |
| perceived_colour_master_name | string | 0 | Black; White |
| department_no | int64 | 0 | 1676; 1339 |
| department_name | string | 0 | Jersey Basic; Clean Lingerie |
| index_code | string | 0 | A; B |
| index_name | string | 0 | Ladieswear; Lingeries/Tights |
| index_group_no | int64 | 0 | 1 |
| index_group_name | string | 0 | Ladieswear |
| section_no | int64 | 0 | 16; 61 |
| section_name | string | 0 | Womens E…sics (len=22); Womens Lingerie |
| garment_group_no | int64 | 0 | 1002; 1017 |
| garment_group_name | string | 0 | Jersey Basic; Under-, Nightwear |
| detail_desc | string | 0 | Jersey t…aps. (len=39); Microfib…ort. (len=240) |

上述内容仅来自首个 row group 的前 5 行，不代表全量分布。【证据：少量抽样】
### `customers`

| 字段 | Arrow 类型 | 样本空值数 | 截断后的值形态（最多 2 个） |
| --- | --- | --- | --- |
| customer_id | string | 0 | <string:length=64,charset=hex> |
| FN | double | 4 | null; 1.0 |
| Active | double | 4 | null; 1.0 |
| club_member_status | string | 0 | ACTIVE |
| fashion_news_frequency | string | 0 | NONE; Regularly |
| age | double | 0 | 49.0; 25.0 |
| postal_code | string | 0 | <string:length=64,charset=hex> |

上述内容仅来自首个 row group 的前 5 行，不代表全量分布。【证据：少量抽样】
### `transactions`

| 字段 | Arrow 类型 | 样本空值数 | 截断后的值形态（最多 2 个） |
| --- | --- | --- | --- |
| t_dat | string | 0 | 2018-09-20 |
| customer_id | string | 0 | <string:length=64,charset=hex> |
| article_id | string | 0 | <string:length=10,charset=digits> |
| price | double | 0 | 0.0508305084745762; 0.0304915254237288 |
| sales_channel_id | int64 | 0 | 2 |

上述内容仅来自首个 row group 的前 5 行，不代表全量分布。【证据：少量抽样】

- ID 的存储类型、日期类型、数值精度和空值形态详见表格；没有对类型做自动转换。【证据：Arrow schema 与少量抽样】
- 抽样未发现 Arrow `list`/`struct` 嵌套字段；长 ID/文本已截断，未保存完整原始行。【证据：schema 与少量抽样】
- 是否存在乱码、异常值或 CSV→Parquet 类型漂移，只能作为抽样观察，不能据此推断全量比例。

## 8. 初步完整性结论

### 元数据已确认

- 三张已发现逻辑表的文件数、总行数、row group 数、压缩和 schema 已由 footer 确认。
- 无法读取文件：无。
- 同一逻辑表内 schema：均一致（当前每表仅 1 个文件）。
- 未发现的核心候选表：sample_submission。

### 抽样观察

- 每张可读逻辑表均已读取首个 row group 的前 5 行；抽样结果仅用于值形态检查。

### 尚未验证

- transaction 中 unique customer、unique article、重复交易、日期全量范围、空值率和外键覆盖率。
- customer/article 主表主键唯一性与 transaction 外键一致性。
- 文件内容是否与公开数据逐行一致；总行数相近不能证明 100% 完整。
- “重复或额外分片”只能确认目录中没有同表多文件，不能排除内容级重复。

## 9. 环境与资源

- OS：`Windows-10-10.0.26200-SP0`
- Python：`3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]`；解释器命令 `python.exe`
- 虚拟环境：`未检测到`
- CPU：`8` 个逻辑核心
- 内存：总计 15.80 GiB，可用 2.12 GiB
- 工作区所在磁盘：总计 780.59 GiB，可用 538.78 GiB
- Parquet 总大小：904.44 MiB

| 依赖 | 版本 |
| --- | --- |
| pandas | 2.2.3 |
| pyarrow | 24.0.0 |
| polars | 未安装 |
| duckdb | 未安装 |
| numpy | 2.2.4 |
| scikit-learn | 1.6.1 |
| lightgbm | 4.6.0 |
| pytorch | 2.6.0 |

【证据：Python/platform、Windows GlobalMemoryStatusEx、shutil.disk_usage、importlib.metadata、Git 命令】

路径风险：工作区绝对路径无中文、无空格，当前相对路径较短；Windows 长路径风险低。Python 来自用户级安装目录，未启用虚拟环境，后续存在依赖漂移风险。

## 10. 风险清单

1. 缺少 sample submission：若后续需要复现 Kaggle 提交流程，需要另行确认其是否应在本项目数据范围内。
2. 当前无虚拟环境、无依赖锁定文件，重现性不足。
3. 交易表约 738.41 MiB；解压后的列式扫描与高基数 distinct/groupby 会显著增加内存。
4. footer 不能给出精确 unique customer/article 数量或内容级重复结论。
5. 只有单文件逻辑表，无法通过分片间比较发现 schema 漂移；“一致”仅表示不存在跨分片冲突面。

## 11. 正式数据审计技术方案

| 方案 | 内存风险 | Parquet/聚合 | Windows 与测试 | 工程与展示价值 |
| --- | --- | --- | --- | --- |
| pandas | 高，易误触全量物化 | 读取方便，超大聚合风险高 | 稳定、测试方便 | 适合小结果，不宜主执行 |
| PyArrow Dataset | 低到中，可投影/过滤/分批 | Parquet 元数据与流式扫描强，复杂聚合较底层 | 稳定、易做 schema/批次测试 | 最适合底层 I/O 与契约检查 |
| Polars lazy scan | 低到中，惰性优化 | 扫描聚合强 | 当前未安装，尚未实机验证 | 有展示价值但不是必要复杂度 |
| DuckDB | 低到中，可溢写磁盘 | SQL 聚合、distinct、join 强 | 当前未安装，尚未实机验证 | 审计 SQL 清晰，可解释性强 |
| PyArrow 分批 | 最低，可显式控制 batch | 需要自行实现聚合状态 | 稳定、最易构造 smoke test | 维护成本适中，边界清晰 |

**建议**：下一阶段以 **DuckDB 作为正式审计主执行引擎**（需先最小化安装并验证），PyArrow 作为 footer/schema/批次读取辅助；pandas 只接收已聚合的小结果。若不增加依赖，则采用 PyArrow Dataset 分批扫描作为可运行备选。当前每表单文件且总数据不足 1 GiB，不建议先物理分区；先做 1 个 row group 或受限行数的 smoke test，再做全量列投影扫描。对昂贵 distinct、外键覆盖率和重复检查缓存小型审计结果（JSON/Parquet），不缓存原始数据副本。最可能瓶颈是 transaction 高基数 distinct/groupby 的 CPU、内存与临时磁盘，而不是 footer 读取。

## 12. 尚未验证事项

- 所有需要全量列扫描的唯一性、分布、空值率、键覆盖与时间范围指标。
- DuckDB/Polars 在本机的真实性能与稳定性（当前未安装时）。
- 数据集与任何外部权威校验和或发布清单的一致性。
- sample submission 缺失是有意裁剪还是下载不完整。

## 13. 下一阶段建议

阶段 0B 建议先建立数据契约与 smoke-test 审计：锁定最小依赖；用 DuckDB 或 PyArrow 扫描单个 row group/受限批次；验证审计 SQL/批处理的内存边界；再按列投影执行全量唯一性、空值率、日期范围、主外键覆盖与重复检查，并持久化仅含指标的小型结果。未经确认，本阶段不继续实施。
