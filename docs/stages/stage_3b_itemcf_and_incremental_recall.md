# Stage 3B：经典 ItemCF 与多路召回增量价值验证

## 1. 目标和任务边界

本阶段只实现 cosine/IUF ItemCF、固定预算 union、overlap、marginal Recall 与 paired user-block bootstrap。核心问题是 ItemCF 在已有 Trend 7d、Pure Repurchase 和 Popularity 之外是否提供独立增量；未实现 MF、双塔、内容召回、排序或重排。

## 2. 数据与防泄漏

- Validation：`validation_14d`，as-of `2020-08-25`；Test：`test_14d`，as-of `2020-09-08`。
- 反馈固定为 `B-primary_C-log1p-p99-v1`，主协同人口固定为 `evaluable_warm_users`，cold-history 单列。
- ItemCF 只读取各 snapshot cutoff 前 B 口径 history；ground truth 只进入统一评估计算。
- 主候选域为 `history_seen_items_v1`；lock_fingerprint: `c8c8f233f05001a1d600433068c1874a875f9ec47aa7d7b8d612e51cfbe2d134`。
- Validation 锁定 `iuf` 后才运行一次性 Test；Test 未参与变体或配额选择。

## 3. ItemCF 定义

用户内商品先去重，按最近购买日期及匿名 article key 稳定排序，最近 50 个唯一商品参与共现。cosine 使用 `Cij/sqrt(NiNj)`；IUF 将每用户贡献改为 `1/log(1+|Iu|)` 后使用同一支持度归一化。每商品保留 Top-200 邻居；每用户最多取最近 20 个 seed，权重为 `1/log2(rank+1)`；排除所有 seen items 和候选域外商品，最多输出 500 个真实 ItemCF 候选，不足不补齐。

## 4. Validation 变体选择

| Variant | Recall@500 | HitRate@500 | MAP@12 |
| --- | ---: | ---: | ---: |
| cosine | 0.106857 | 0.247068 | 0.006431 |
| iuf | 0.107695 | 0.248495 | 0.006339 |

按 Recall@500、HitRate@500、MAP@12、候选用户覆盖率、完全相同时 cosine 的预注册顺序，锁定 `iuf`。

## 5. 单路效果

锁定 ItemCF 的 Test Recall@500 为 `0.108671`，MAP@12 为 `0.006750`。完整 overall、short-history/warm/active/cold-history、候选不足、catalog coverage 与集中度见公开 CSV；cold-history 的 ItemCF 候选和指标按定义为 0。

## 6. Overlap

Validation/Test 主人口 ItemCF–Trend 平均 Jaccard 分别为 `0.034359` / `0.032496`；Test 仅 ItemCF 命中的 ground-truth 数为 `21037`，有独立 ItemCF 命中的用户数为 `15764`。ItemCF 实际新增位置、与 Repurchase/Base 的重叠及 shared hits 见 overlap summary。

## 7. 固定预算 union

候选总预算始终为 500。Base Union 使用 Repurchase 50、Trend 100、Popularity 补齐；ItemCF Union 加入锁定 ItemCF 300，仍由 Popularity 只补剩余位置。固定插入顺序为 Repurchase、ItemCF、Trend、Popularity；不同来源原始 score 未相加。

| Phase | Base Recall@500 | ItemCF Union Recall@500 | Absolute delta |
| --- | ---: | ---: | ---: |
| Validation | 0.194108 | 0.220501 | 0.026394 |
| Test | 0.183280 | 0.214609 | 0.031329 |

Test 提升/不变/下降用户比例为 `0.129526` / `0.805199` / `0.065274`；固定预算下的下降没有隐藏。每 100 个实际新增 ItemCF 位置带来的净新增 ground-truth hit 为 `0.033202`。

## 8. Bootstrap

Validation delta 的 95% CI 为 `[0.025274, 0.027537]`，bootstrap delta>0 比例为 `1.000000`；Test 对应为 `[0.030150, 0.032557]` 和 `1.000000`。这是离线 paired user-block bootstrap，不是线上显著性或 A/B Test。

## 9. 资源

- 锁定 Test 共现非零无向商品对 `182136894`，邻居行 `17822424`。
- 已记录任务组耗时合计 `18914.748` 秒，峰值 RSS `1206976512` bytes，临时盘峰值 `16965638264` bytes，spill observed `true`，额外重试 `9`。
- spill 标记采用任务期间整个 Stage 3B 临时目录的保守峰值；已完成分区的临时文件会成为后续组的基线，因此该标记证明观察到临时盘占用，但不能逐组归因。pair consolidation 与邻居排序均观察到临时盘继续增长，可确认发生外存计算。
- 只读接管阶段曾用 1024 MiB 对全量字符串键直接 `count(distinct)` 做规模预估并真实 OOM；该查询未产生正式产物。正式实现改用内部键、分区 pair checkpoint 与外存聚合。
- 首次正式 Validation 的单文件 user-item 集合在约 582 MiB DuckDB 限制下 OOM，随后等价改为 32 个 customer-key 分区；首次 Base Union 单分区在约 848 MiB 限制下 OOM，随后保持配额、顺序与输出语义不变，在每个正式分区内再拆 4 个工程子分区。两次失败均未缩小数据或改变评估口径。
- Validation 的 pair 分区 029 异常耗时 `5373.476` 秒，其他主要分区未出现同量级长尾；Test 未复现。Test 首次外层命令在 1 小时达到运行器上限后被中断，随后以同一 lock、参数和 output fingerprint 仅续跑未完成组，没有重新选择或增加 Test comparator。
- 收尾统计曾因错误要求未运行的 Test cosine 文件而失败；修复仅跳过协议上不应存在的未锁定 Test 变体，并通过显式 source-fingerprint 兼容审计。敏感信息扫描的 fingerprint/URL 误报也曾阻断收尾，字段感知规则修复后通过独立审计。

## 10. 限制

最近 50 个共现商品和最近 20 个 seed 都是本阶段固定工程规则，可能丢失长周期关系，不是行业统一标准。数据没有 order/session、曝光、点击、库存或精确上架时间；共现不等于因果关系，完整维表不等于历史可售域。ItemCF 排除 seen item，而 Repurchase 单独保留历史商品。离线 Recall 增量不等于线上 CTR/CVR 或业务收益。

## 11. 面试与简历证据

仓库真实支持经典 cosine/IUF ItemCF、分区共现与 Top-N 邻居、固定预算多路 union、来源 provenance、独立命中、marginal Recall、paired bootstrap、Test fingerprint 门禁和资源审计。不能声称 MF/双塔已实现、已经上线或取得 CTR/CVR 提升。

## 12. 下一阶段建议

建议 ItemCF 进入后续正式 union，但仍需在后续阶段验证配额与排序。 是否实现 MF challenger 应在获得单独阶段授权后，以 ItemCF 的真实独立增量和资源成本作为对照；本阶段不自动进入下一阶段。
