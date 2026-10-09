# Stage 3A：Baseline 召回与统一离线评估

## 1. 目标与范围

本阶段实际实现并全量运行 Popularity、7/14/28 天 Trend、Pure Repurchase、Repurchase + Popularity Fallback，以及与具体 baseline 解耦的统一候选协议和离线评估器。选择这些简单基线，是为了先建立可复现、可审计的推荐效果下界和 Test 封存机制。未实现 ItemCF、MF、内容召回、特征工程、排序、重排或 Stage 3B 能力。

## 2. 数据与协议

- Validation snapshot：`validation_14d`，as-of `2020-08-25`，标签 `2020-08-26` 至 `2020-09-08`。
- Test snapshot：`test_14d`，as-of `2020-09-08`，标签 `2020-09-09` 至 `2020-09-22`。
- feedback：`B-primary_C-log1p-p99-v1`；population：`1.0.0`。
- 主协同候选域：history seen items；cold-history fallback 使用完整 articles 身份域并单列。
- dataset fingerprint：`f3b4d8bfed7655be122a6b43abf3f11cd526367b89cafefe6daf0c3e8a057927`；Stage 3A config fingerprint：`7636082b84edb63a22f6012e663947aaf4cb12db568c112f3f70e67e20710a0d`。
- Validation 只选择 Trend window；Test 在 lock 后只运行固定版本，未参与选择。

## 3. 算法定义

- Popularity：snapshot history 中每个商品的 B 口径 user-item-day 记录数降序，同分按匿名 `article_key` 升序；不删除历史已购。
- Trend：区间 `(as_of_date-window_days, as_of_date]` 内的 B 口径计数降序，再按同 snapshot 全历史 Popularity 降序、`article_key` 升序。
- Pure Repurchase：每用户按不同购买日数降序、最近购买日降序、`article_key` 升序；不足 500 不补造。
- Fallback：完整保留 pure 顺序，按 Popularity 顺序去重补齐至最多 500；补齐分数不与复购分数作跨层比较。

### AP@12 / MAP@12 口径审计

对评估用户 \(u\)，令 \(G_u\) 为 Stage 2B 中去重后的正式 ground-truth 商品集合，\(rel_u(r)\) 表示第 \(r\) 位候选是否属于 \(G_u\)，\(P_u(r)=\sum_{j=1}^{r}rel_u(j)/r\)。本阶段使用：

```text
AP@12(u) =
  [ Σ(r=1..min(12, L_u)) P_u(r) × rel_u(r) ]
  / min(|G_u|, 12)

MAP@12 =
  [ Σ(u in fixed evaluation population) AP@12(u) ]
  / |fixed evaluation population|
```

其中 \(L_u\) 是用户实际候选长度；空 ground truth 用户不进入模型自定义分母，而是直接复用 Stage 2B 固定人口。该实现与 `docs/decisions/evaluation_protocol.md` 的 Top-12、用户级 AP 和固定人口宏平均口径一致。

### 评估人口对账

| Snapshot | all_label_users | evaluable_warm_users | cold_history_label_users |
| --- | ---: | ---: | ---: |
| validation_14d | 141,359 | 129,705 | 11,654 |
| test_14d | 128,333 | 117,366 | 10,967 |

两组总人口均严格等于 warm 主人口与 cold-history 人口之和，并与 Stage 2B `evaluation_population_manifest.csv` 一致。Stage 3A overall 指标分母分别为 129,705 和 117,366；cold-history 始终单列。

## 4. Validation 结果

| Source | Variant | MAP@12 | Recall@12 | Recall@500 | HitRate@12 |
| --- | --- | ---: | ---: | ---: | ---: |
| popularity | global | 0.004156 | 0.010191 | 0.107532 | 0.029860 |
| trend | window_7d | 0.005959 | 0.021451 | 0.280557 | 0.060206 |
| trend | window_14d | 0.005470 | 0.018905 | 0.239664 | 0.053329 |
| trend | window_28d | 0.004889 | 0.013999 | 0.185579 | 0.042203 |
| repurchase | pure | 0.011745 | 0.026142 | 0.037606 | 0.058332 |
| repurchase | popularity_fallback | 0.011922 | 0.027275 | 0.128553 | 0.061347 |

Trend 预注册选择：

| Window | Recall@500 | MAP@12 | Candidate user coverage | Selected |
| ---: | ---: | ---: | ---: | --- |
| 7 | 0.280557 | 0.005959 | 1.000000 | True |
| 14 | 0.239664 | 0.005470 | 1.000000 | False |
| 28 | 0.185579 | 0.004889 | 1.000000 | False |

按 Recall@500、MAP@12、用户候选覆盖率、较短窗口的确定性顺序，锁定 `7` 天。

## 5. Test 结果

Test 仅运行 lock 允许的四个固定版本：

| Source | Variant | MAP@12 | Recall@12 | Recall@500 | HitRate@12 |
| --- | --- | ---: | ---: | ---: | ---: |
| popularity | global | 0.003243 | 0.008648 | 0.096070 | 0.025476 |
| trend | window_7d | 0.006408 | 0.021335 | 0.283676 | 0.061031 |
| repurchase | pure | 0.014224 | 0.030725 | 0.041865 | 0.064405 |
| repurchase | popularity_fallback | 0.014408 | 0.031856 | 0.123520 | 0.067234 |

## 6. 分群分析

| Source | Variant | Segment | MAP@12 | Recall@500 |
| --- | --- | --- | ---: | ---: |
| popularity | global | short_history | 0.003672 | 0.114301 |
| popularity | global | warm | 0.003783 | 0.106474 |
| popularity | global | active | 0.002992 | 0.090158 |
| popularity | global | cold_history_label_users | 0.003536 | 0.106562 |
| trend | window_7d | short_history | 0.008158 | 0.308851 |
| trend | window_7d | warm | 0.007719 | 0.296886 |
| trend | window_7d | active | 0.005723 | 0.275947 |
| trend | window_7d | cold_history_label_users | 0.005735 | 0.289653 |
| repurchase | pure | short_history | 0.022072 | 0.027143 |
| repurchase | pure | warm | 0.016858 | 0.030008 |
| repurchase | pure | active | 0.012373 | 0.047939 |
| repurchase | pure | cold_history_label_users | 0.000000 | 0.000000 |
| repurchase | popularity_fallback | short_history | 0.023609 | 0.133312 |
| repurchase | popularity_fallback | warm | 0.017137 | 0.126824 |
| repurchase | popularity_fallback | active | 0.012373 | 0.121204 |
| repurchase | popularity_fallback | cold_history_label_users | 0.003536 | 0.106562 |

`evaluable_warm_users` 是所有 history>0 的主人口，不等同于 5–20 的 `warm` 分组；cold-history 始终单列，未混入主协同宏平均。分群差异仅描述本次离线购买标签，不预设任一路必然适合某一人群。

本次数据支持的具体观察是：锁定的 7 天 Trend 在 short-history、warm、active 三组的 Test Recall@500 均高于另外三个固定 baseline；Pure Repurchase 的 Test MAP@12 在三个有历史分组均高于 Trend，但 Recall@500 明显受候选长度限制。active 组的 Pure Repurchase Recall@500（0.047939）高于 short-history（0.027143），但这不等于 active 用户在线上必然更适合复购策略。cold-history 的 Pure Repurchase 按定义为空，fallback 正确退化为 Popularity。

Trend 的高 Recall 与 Repurchase/Fallback 的高 MAP 表现出互补潜力，但本阶段没有实际构造 candidate union，也没有计算 union overlap、独立命中或边际 union gain，因此不得表述为“已证明召回互补”。

## 7. 候选覆盖和集中度

以下为 Test 主协同人口：

| Source | Variant | User coverage | Mean candidates | Below 500 rate | Catalog coverage | Top-10 Top-12 share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| popularity | global | 1.000000 | 500.00 | 0.000000 | 0.004856 | 0.833333 |
| trend | window_7d | 1.000000 | 500.00 | 0.000000 | 0.004856 | 0.833333 |
| repurchase | pure | 1.000000 | 51.12 | 0.999182 | 0.897006 | 0.012152 |
| repurchase | popularity_fallback | 1.000000 | 500.00 | 0.000000 | 0.897006 | 0.112012 |

Pure Repurchase 的候选不足是真实覆盖限制；fallback 的作用是保持复购头部顺序并改善候选完整性。集中度指标只是离线推荐列表统计，不代表线上曝光或收益。

## 8. 防泄漏验证

- Popularity、Trend、Repurchase 统计都限定在各 snapshot history 与 as-of 之前。
- ground truth 只在解耦评估器中读取，不参与候选分数、排序或补齐。
- Validation/Test 使用各自独立 history；Test 入口校验 config、dataset、source 和 selection fingerprints。
- 主协同候选逐项验证为 history seen item；cold-history 使用明确的完整身份域版本。
- Test 前后 Validation selection artifact fingerprint 保持不变。

## 9. 资源与工程

- 记录任务组总耗时：3335.957 秒；峰值 RSS：1173610496 bytes。
- DuckDB 内存上限：881 MiB，单线程；临时盘峰值：19103449972 bytes；spill observed：true。
- 失败后额外尝试次数：1；每组 checkpoint 记录输入、配置、源码与输出 fingerprint。
- 候选和用户级指标位于 Git 忽略的 private；公开产物只有聚合。
- 首次 Validation Popularity 在候选 SQL 中使用保留词别名导致 parser error；未产生指标，修复 SQL 别名后以同一算法和配置成功重试。Test 未发生失败或重试。
- 首次最终公开扫描将 manifest/lock 中声明的 SHA-256 误识别为 64 位 customer 值；独立最终审计仅白名单字段名明确为 fingerprint 的值后通过，未放宽实体字段、绝对路径、ignore 或 tracked-file 检查。

## 10. 真实限制

数据无曝光、点击、真实负反馈、库存/可售状态和精确上架时间。完整 articles 维表只表示身份域，不等于历史可售域；未购买不等于真实不喜欢；离线 Recall/MAP 不等于 CTR/CVR 或线上收益；本项目未做线上 A/B Test。

## 11. 可用于简历和面试的真实证据

仓库可以支持：严格相邻时间快照、B 口径基础层、统一候选协议、宏平均评估器、三类 baseline 与分群结果、Validation 选择和 Test fingerprint 封存、DuckDB 大规模执行、checkpoint/资源/隐私测试。不能声称已实现复杂协同或深度模型，也不能将离线购买指标扩写为线上业务提升。

## 12. 下一阶段建议

建议有条件进入 Stage 3B。证据是：7 天 Trend 在 Test 主人口取得四路最高 Recall@500（0.283676），应进入后续 candidate union；Pure Repurchase 与 fallback 分别取得 0.014224 和 0.014408 的 MAP@12，说明复购头部排序值得保留，而 Pure Repurchase 平均仅 51.12 个候选、99.9182% 用户不足 500，不能单独承担完整召回；Popularity 的 Test Recall@500 为 0.096070，保留为固定对照和 fallback。Stage 3B 如获单独授权，应优先验证冻结路线中的经典协同召回能否在 Trend + Repurchase 之外提供独立 Recall/union 增益；不应自动扩展到排序或深度模型。本报告不授权也未实施 Stage 3B。

## 13. Stage 3A 收尾维护

正式 Test 验收后发现，旧扫描器先用正则识别任意 64 位十六进制字符串，再依赖调用方传入的全局 token 集合豁免，无法验证该值是否真的位于 fingerprint 字段。这导致 manifest/lock 中合法 SHA-256 被误报，同时全局 token 参数存在字段无关的过宽豁免风险。

维护修复将豁免收紧为：字段名必须属于显式正式白名单、值必须是完整 64 位十六进制 SHA-256；普通字段、未知字段、错误长度、非十六进制值均不能获得豁免。修复只涉及扫描逻辑、测试和本报告，没有修改 baseline、候选协议、评估计算、Validation 选择、Test lock、Test 候选或 Test 指标，也没有重新运行 Test。
