# Stage 3C：Structured-Attribute Content Retrieval 与固定预算增量验证

## 1. 阶段目标与边界

本阶段以结构化商品属性实现可解释 Content Retrieval，并回答其前部候选能否在固定 500 候选预算下替换部分 IUF ItemCF 尾部。未实现文本/图片 embedding、ANN、双塔、矩阵分解、排序、重排或在线服务。

## 2. 开始门禁与权威输入

开工时 `main` 与 `origin/main` 为 0 ahead/0 behind，工作区干净，初始测试 148 passed；Stage 3A/3B lock、manifest、公开指标及私有上游 fingerprint 审计通过。冲突裁定后，Q0/Q1/Q2 统一沿用 Stage 3B 的预去重 `source_rank_cap` 语义。

## 3. 数据、防泄漏与候选域

Validation/Test 分别只使用各自 cutoff 前 history。主候选域固定为 `history_seen_items_v1`；完整 articles 身份域未参与模型或配额选择。DF/IDF、属性矩阵、内容邻居和用户候选均按 snapshot 独立计算；ground truth 只进入评估。cold-history 个性化 Content 候选为 0。

## 4. Canonical 属性审计

建模字段为 taxonomy（product type/group、garment group、section）、appearance（graphical appearance、colour group、perceived colour value/master）和 merchandising/index（department、index、index group）。code/name 只选择一个 canonical 信号，名称只作解释；冲突、缺失率、基数、高频率和候选域覆盖见 `canonical_field_audit.csv`。`detail_desc`、商品名和图片均未使用。

## 5. Content 表示、IDF 与 group balance

token 固定为 `field_name=field_value`；missing 是字段隔离显式 token。IDF 为 `log((N+1)/(df+1))+1`。C2/C3 先组内 L2 归一化、各非空组等先验，再整体归一化。精确 sparse cosine 以分块矩阵乘法计算每商品 Top-200，不构建稠密全矩阵，也未使用 ANN。

## 6. 用户 seed 与候选生成

复用 Stage 3B 最近 20 个唯一 seed 和 `1/log2(rank+1)` 衰减；多 seed 分数为衰减乘 cosine 后求和。Content 排除全部 seen item、过滤到主候选域、最多 500，候选不足不自行补齐。

## 7. C0–C3 Validation 与版本锁定

四个预注册版本均完成 Content 单路和固定 Q1 Union。按 Q1 Recall@500、HitRate、独占命中、下降用户率和复杂度顺序锁定 `C3`；完整指标、overlap 和独占命中见对应公开 CSV。没有依据 Test 或标签新增属性组合。

主评估人口 `evaluable_warm_users` 的 Content 单路核心指标如下：

| Content 版本 | Recall@500 | HitRate@500 | MAP@12 |
| --- | ---: | ---: | ---: |
| C0 | 0.037735 | 0.094715 | 0.000293 |
| C1 | 0.036033 | 0.090976 | 0.000207 |
| C2 | 0.037656 | 0.101392 | 0.000376 |
| C3 | 0.048399 | 0.121337 | 0.000476 |

同一人口下，Phase A 固定 Q1 Union 的核心指标如下。C3 在四个预注册 Content 版本中取得最高 Q1 Recall@500 和 HitRate@500，因此锁定为最佳结构化 Content 表示；这不等于它已经通过独立召回或固定预算增量价值门槛。

| Q1 Content 版本 | Recall@500 | HitRate@500 | MAP@12 |
| --- | ---: | ---: | ---: |
| Q1_C0 | 0.216877 | 0.440854 | 0.012185 |
| Q1_C1 | 0.216928 | 0.440916 | 0.012185 |
| Q1_C2 | 0.218268 | 0.443075 | 0.012185 |
| Q1_C3 | 0.218866 | 0.443923 | 0.012185 |

## 8. Q0/Q1/Q2 固定预算实验

Q0 精确复用 Stage 3B 冻结候选；Q1 比较移除 ItemCF rank 251–300、加入 Content rank 1–50；Q2 比较移除 ItemCF rank 201–300、加入 Content rank 1–100。所有来源先按 source rank 截断，再合并 provenance 和去重，Popularity 补齐至 500。锁定 `Q0`，Validation Recall@500 为 0.220501。

| 配额方案 | Recall@500 | HitRate@500 | MAP@12（完整值） |
| --- | ---: | ---: | ---: |
| Q0 | 0.220501 | 0.446112 | 0.012184852577349603 |
| Q1 | 0.218866 | 0.443923 | 0.012184760793980384 |
| Q2 | 0.216183 | 0.440554 | 0.012184760793980384 |

Q1 Content cap=50、实际 unique occupancy=40.202、重复损失=5.676；ItemCF 实际唯一减少=45.035、Popularity 补齐=109.790；Q2 Content cap=100、实际 unique occupancy=82.645、重复损失=9.111；ItemCF 实际唯一减少=90.102、Popularity 补齐=111.740。配置 cap 与实际 unique occupancy、重复损失、Popularity 补齐、每 100 个 Content rank/实际唯一位置的净命中效率均在 `validation_quota_occupancy.csv` 分开报告。

三者的 MAP@12 写成六位小数时均为 `0.012185`，但并非三者精确一致：Q1 与 Q2 的完整值精确相同，Q0 比二者高约 `9.1783e-08`。基于既有 Validation 候选产物的只读聚合显示，Q1 和 Q2 相对 Q0 都只有 1/129,705 个主评估用户的 Top-12 候选序列发生变化，占约 0.000771%，且该用户有 11 个位置变化。配额替换主要发生在深层候选，绝大多数用户的 Top-12 未变化，这解释了 MAP@12 几乎不变；该补充聚合没有重跑模型、改变选择或触碰 Test。

## 9. 固定预算增量、损失与 bootstrap

锁定方案相对 Q0 的 ΔRecall@500 为 0.000000，95% paired user-block bootstrap CI 为 不适用。上升/不变/下降用户、增加/丢失/净 hits 均公开，未隐藏固定预算替换损失。这是离线稳定性分析，不是线上 A/B Test、CTR/CVR 显著性或业务收益。

候选方案的 paired bootstrap 结果为：Q1 vs Q0 ΔRecall@500=-0.001635, 95% CI=[-0.001947, -0.001283]；Q2 vs Q0 ΔRecall@500=-0.004318, 95% CI=[-0.004783, -0.003835]。二者区间均低于 0，因此 Content 虽有独占命中，仍不足以抵消被替换的 IUF ItemCF 尾部价值。

## 10. 分群、长尾与低协同支持

short-history/warm/active/cold-history 复用 Stage 2B 固定分母；head/mid/tail 复用 Stage 2B cutoff-history 分位规则。Content 独占命中按 popularity bucket、history support decile 和 IUF 邻居覆盖报告；低协同支持不被称为真实新品冷启动。

## 11. Validation lock 与 Test 门禁

Stage 3C lock 同时固化 Content 版本、source-rank cap、配置/源码、Stage 3A/3B locks 与全部选择产物 fingerprint。Validation 锁定 Q0，且 Content 未通过 value gate，因此按协议未运行 Content Test 变体。

## 12. 资源、失败与恢复

各组耗时、峰值 RSS、临时盘、spill、attempt 和输出大小见 `resource_summary.csv`；失败、重试与恢复见 `failure_retry_summary.csv`。大任务以独立 checkpoint 和 output fingerprint 恢复，fingerprint 变化拒绝复用。

恢复过程中隔离并字节级重建了被中断覆盖的 C0 邻居文件；Q0 从错误的 Stage 3B `base_union` 更正为冻结的 IUF Union；item-support 汇总两次 OOM 后改为等价的小命中集合 anti-join，并保留 1 个缺失协同支持度的 `unknown` 桶。以上均在失败/迁移审计中留痕。

## 13. 阶段结论

Stage 3C 已完成，最终状态为 `FAILED_VALUE_GATE`。C3 是四个预注册方案中最佳的结构化 Content 表示，但固定预算 Q1、Q2 均弱于冻结的 Stage 3B Q0；Content 未通过 Validation value gate，不进入当前后续正式多路 Union。按照预注册门禁，Content Test 状态为 `SKIPPED_BY_PROTOCOL`，没有运行，也没有用于选择或包装结果。Stage 3D 尚未获得新的正式 Prompt 授权，不得开始。

## 14. 真实限制

数据没有曝光、点击、真实负反馈、库存、可售状态或精确上架时间。完整维表不等于历史可售域；未购买不等于不喜欢；结构化相似不等于真实偏好；本阶段未解决线上新品冷启动，离线 Recall/MAP 不等于 CTR、CVR、收入或线上收益。

C3 后续只能作为结构化商品属性处理能力和潜在商品塔输入基础复用；本阶段不能声称其独立召回已经有效，也不能据此预设双塔一定提升。是否采用双塔仍需新的正式任务、独立协议和 Validation 证据。

unique-slot fill、RRF、动态配额和 rank fusion 均未在 Stage 3C 验证。它们只能作为后续新 Prompt 授权后的潜在消融，不能用未运行方案否定、修饰或包装本阶段 `FAILED_VALUE_GATE` 的负实验结论。

## 15. 面试知识点与高频追问

- Content-based Retrieval 用商品属性 multi-hot 与 cosine 找相似商品；ItemCF 使用历史共现，两者归纳偏置不同。
- IDF 降低高频属性 token 的支配；group balance 防止字段更多的语义组自然获得更大权重。
- code/name 不同时建模，避免同一概念重复加权；名称只用于解释，不做语义 embedding。
- Content 依赖历史 seed，不能个性化服务零历史用户；Repurchase 负责 seen item，Content/ItemCF 均排除 seen。
- 先做 Content 再考虑双塔，是为了用低复杂度、可解释基线验证商品侧信号；本阶段结果不能预设双塔一定更好。
- 不使用文本和图片，是因为本阶段正式范围只验证结构化属性，避免在弱结果后扩大搜索空间。
- 固定预算 Union 评估边际替换价值；异构原始分数未标定，不能直接相加。
- source-rank cap 不等于实际唯一 occupancy；重复是召回冗余的一部分，本阶段不向后扫描补偿。
- 单路 Recall 高不保证应获更多 cap；真正问题是替换 ItemCF 尾部后的净新增 hits。
- 每 100 个 rank 候选与每 100 个实际唯一位置同时报告，可区分来源质量和重复损失。
- 报告下降用户是为了暴露固定预算替换的异质损失，不能只展示总体均值。
- paired bootstrap 是同一离线用户的稳定性分析，不是线上随机实验。
- Content 对 short-history 不一定更好，必须以本阶段真实分群指标回答。
- history 低支持商品不等于新品；缺少精确上架时间，不能声称解决真实新品冷启动。
- 即使 value gate 失败，canonical 审计、snapshot IDF、结构化属性处理和可恢复 sparse 计算仍可作为潜在商品塔输入基础复用；这不构成独立召回有效或双塔必然提升的证据。
- unique-slot fill、RRF、动态配额与 rank fusion 是未验证的潜在后续实验，不是本阶段缺陷，也不能改变当前负结论。
