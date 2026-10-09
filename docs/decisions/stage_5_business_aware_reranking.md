# Stage 5 supplementary business-aware reranking decision

日期：2026-10-07。状态：SUPPLEMENTARY_BUSINESS_INTERPRETATION_ONLY。

本记录基于已冻结的 Validation 聚合结果补充业务解释，不替代、不覆盖，也不重新解释 Stage 5E 的 preregistered selection。最值得保留的 business-aware candidate configuration 是 **R2（product_code cap=2）**；strict offline frozen variant 始终是 **R1**。

## 原冻结结论与补充候选的边界

| 定位 | 配置 | 含义 |
|---|---|---|
| Strict offline configuration | R1 / no-op | Stage 5E 按原预注册规则冻结的版本；Top-12 membership、order、MAP、NDCG、Recall、HitRate 与 R0 完全一致 |
| Business-aware candidate configuration | R2 / product-aware constrained reranking | 保留既有 product_code cap=2 方案作为业务权衡候选；尚未通过原 Stage 5E selection，也未获线上采用结论 |

R1 是 pipeline control，其 identity 不能被表述为 reranking 带来的效果提升。R2、R3 和全部 R4_N20/N30/N50 均未通过原预注册的相对 R0 0.5% MAP/NDCG relevance guard；点估计与 paired 95% CI lower bound 的要求继续有效。0.5% 是原预注册工程容忍预算，不是统计推导的通用阈值。

**不能宣称 R2“通过原始 Stage 5E selection”。** 将 R2 保留为业务候选只是 supplementary business interpretation，不是补发通过状态、放宽 loss budget、重选 winner 或替换正式 V1 配置。本次不修改任何原阈值、metrics、lock、manifest、selection record、preregistration、配置或 Completion Report。

## 证据范围

数值引用自 [Validation ablation summary](../../artifacts/stage_5/validation_ablation_summary.csv)；精确值与已有 [paired comparison](../../artifacts/stage_5/validation_paired_comparison.csv) 保持为原始证据。下表仅展示已有数值的四舍五入与相对变化，不产生新的实验结果。

输入模型为 L1_600，snapshot 为 validation_14d，as-of 为 2020-08-25。下列均为 129,705 位主评估用户的聚合，包含 67,409 个 retrieval-miss / all-zero query；11,654 位 cold-history 用户仍单独诊断，不混入主平均。changed-user ratio 表示 Top-12 相比 R0 至少一个位置变化的用户比例。

| 配置 | Mean unique product@12 | Mean unique garment@12 | Product-cap violation | Garment-cap violation | Changed-user ratio |
|---|---:|---:|---:|---:|---:|
| R0 / R1 | 8.815211 | 3.808334 | 0.436884 | 0.882071 | 0.000000 |
| R2 | 9.334737 | 3.973579 | 0.000000 | 0.866613 | 0.436884 |
| R3 | 10.166647 | 5.455457 | 0.000000 | 0.000000 | 0.907706 |

Violation 的比较口径始终是 product cap=2、garment cap=4，不为本补充解释另设定义。

| 指标 | R0 / R1 | R2 | R2 相对 R0 变化 | R3 | R3 相对 R0 变化 |
|---|---:|---:|---:|---:|---:|
| MAP@12 | 0.025257775 | 0.024713627 | -2.1544% | 0.024208732 | -4.1533% |
| NDCG@12 | 0.041181950 | 0.040131911 | -2.5498% | 0.038907332 | -5.5233% |
| Recall@12 | 0.059937731 | 0.057564115 | -3.9601% | 0.054775869 | -8.6120% |
| HitRate@12 | 0.136756486 | 0.134181412 | -1.8830% | 0.129316526 | -5.4403% |

## R2：保留为优先业务候选

R2 将 product-cap violation 从 0.436884 降到 0，同时将 mean unique product@12 从 8.815211 提升到 9.334737，约增加 0.519525 个 product/query。它直接处理同一 product_code 在 Top-12 中占位过多的列表问题，仅干预约 43.6884% 的主评估用户。

这个收益以 MAP、NDCG、Recall、HitRate 全部下降为代价。保留 R2 的理由是它提供了清晰、较局部的 product diversity 权衡，且比叠加 garment hard cap 的 R3 干预更少、relevance 损失更小。若业务明确重视减少同 product 集中占位，R2 是当前已试方案中最值得保留的候选；它不是已验证的净业务收益，也不能据此认为用户不需要同 product 的多个商品选择。

## R3：garment_group hard cap=4 对当前数据过度干预

R3 在 R2 上进一步提高 product/garment diversity，并将两种 violation 都降到 0。但 changed-user ratio 从 R2 的 0.436884 升至 0.907706；相对 R0 的 MAP 损失扩大至约 4.1533%，NDCG 损失扩大至约 5.5233%，Recall 与 HitRate 也进一步下降。

因此，本补充业务判断认为 garment_group hard cap=4 对当前数据、模型及 Top-12 场景过度干预，不作为优先业务候选。类别集中可能包含用户原有购买偏好，不能仅凭 violation 归零就认定业务改善。这是当前实验范围内的判断，不是对品类多样性的一般性否定，也不授权搜索新的 garment cap。

## R4：不建议进入 V1 主配置

R4 相比 R3 的 product/garment diversity 额外变化很小，relevance 未恢复，changed-user ratio 继续增加。已有结果如下：

| 配置 | Mean unique product@12 | Mean unique garment@12 | Mean tail count@12 | Changed-user ratio |
|---|---:|---:|---:|---:|
| R3 | 10.166647 | 5.455457 | 0.504098 | 0.907706 |
| R4_N20 | 10.169778 | 5.454778 | 0.528815 | 0.916788 |
| R4_N30 | 10.177187 | 5.452558 | 0.597463 | 0.923835 |
| R4_N50 | 10.189083 | 5.452851 | 0.705771 | 0.932740 |

Tail count 确有增加：相对 R3 约为 +0.024718 / +0.093366 / +0.201673。不能把这些增量写成“没有 tail 收益”或“未达到原 tail gain threshold”。但这些历史支持度代理值尚不足以证明额外列表净收益或业务价值；product/garment diversity 几乎没有足够的额外收益，干预范围却继续扩大。因此不建议 tail exploration 进入 V1 主配置。

这项建议是补充业务判断。正式拒绝原因仍以原 selection record 为准：全部 R4 未通过相对 R0 的 relevance guard，R3 本身也未合格；不改写任何已有额外 guard 的通过/失败状态。Tail 语义继续复用既有 history-safe 定义，本次不扩大探索窗口或增加 challenger。

## 真实线上采用的验证要求

真实线上是否采用 R2，需要通过 CTR、CVR、GMV、用户留存与多样性等在线 A/B 指标验证，综合评估 relevance 代理值下降是否被真实用户体验和业务收益抵消。只有 product-cap violation 降低和 unique product 增加，不能证明 CTR/CVR/GMV 或留存提升；当前数据缺少曝光、点击、库存/可售状态等信息，也没有本方案的线上 A/B 证据。

本记录没有执行、授权或设计新的线上实验，没有设立替代原 guard 的新阈值，也没有部署 R2。线上采用须由未来独立授权的业务实验决定；在获得相应证据前，strict offline configuration 保持 R1，R2 仅作为待验证的 business-aware candidate configuration。

## 原始权威记录

- [Stage 5 preregistered protocol](stage_5_reranking_protocol.md)
- [Stage 5E selection record](../../artifacts/stage_5/validation_selection.json)
- [Stage 5 frozen configuration](../../configs/stage_5_reranking_frozen.yaml)
- [Stage 5 manifest](../../artifacts/stage_5/stage_5_manifest.json)
- [Stage 5 Validation lock](../../artifacts/stage_5/stage_5_validation_lock.json)
- [Stage 5 Validation Completion Report](../stages/stage_5_validation_completion.md)

本次只新增这份独立 decision note。未重新运行 Validation 或 Test，未读取 Test labels，未重算 evaluation/bootstrap，也未修改上述原始记录。
