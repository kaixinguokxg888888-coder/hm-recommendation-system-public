# Stage 2A 内部 ID 编码策略

状态：`FROZEN`

## 决策

采用 `full_dimension_lexicographic_dense_key_v1`：

1. 从完整 customers/articles 维表读取原始 ID；
2. 按原始 ID 字典序稳定排序；
3. 分别分配从 0 开始的 `customer_key` 和 `article_key`；
4. 映射在相同数据 fingerprint 下确定且一对一；
5. 不使用普通 hash 作为精确键。

## 选择理由

- 完整维表编码让 validation/test snapshot 共享同一键空间；
- cold user/item 即使没有 history 交互，也能获得稳定内部键；
- 键分配只使用身份维表，不使用 validation/test 交易标签；
- 稠密整数适合后续稀疏矩阵，但不改变原始 ID。

维表本身缺少实体创建时间，因此“获得内部键”不能解释为实体在某个历史 cutoff 已经业务可见。训练行为、热度和共现仍必须严格由 snapshot history 决定。

## 验证结果

- customer 映射：1,371,980 行；
- article 映射：105,542 行；
- 两张映射均满足原始 ID 唯一、内部键唯一、从 0 开始连续、维表覆盖完整；
- 映射 fingerprint 记录在 Stage 2A 聚合产物中，但不公开映射内容。

## 隐私和 Git

映射包含原始 ID，只能保存在：

`artifacts/stage_2a/private/`

该目录已被 `.gitignore` 忽略。公开 manifest 只包含映射数量和文件 fingerprint，不包含任何原始 ID、内部键或对应关系。

## Snapshot 与 cold 策略

- validation/test 使用同一数据版本的全量维表键空间；
- cold entity 允许查到内部键；
- cold 键不得使训练统计读取标签行为；
- 新数据版本重新生成映射并比较 fingerprint；
- 如果维表新增更小字典序 ID，稠密键可能整体变化，因此跨数据版本不能假设键稳定，必须以 dataset fingerprint 隔离。
