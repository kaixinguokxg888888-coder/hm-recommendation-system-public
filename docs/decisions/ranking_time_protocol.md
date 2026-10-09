# Ranking Time Protocol

决策编号：`RANK-TIME-001`

状态：`FROZEN_FOR_STAGE_4B_CONSUMPTION`

Stage 4A 已通过 ChatGPT 第二轮技术验收：`PASSED_PROTOCOL_GATE / FINAL ACCEPTED`。这允许未来在新的独立正式 Prompt 下消费本协议进入 Stage 4B，但本次 Stage 4A closeout 没有执行 Stage 4B。

## 1. 为什么先完成 Stage 4A

排序监督、候选、动态特征和统计量必须属于同一个请求时点。若先生成 ranking rows 或训练模型，再补写时间口径，就无法证明历史候选没有借用未来模型、未来统计或标签注入。Stage 4A 因而只冻结时间、replay、label/negative、feature、Validation/Test 与 final-refit 契约，不生成全量候选、排序数据、特征、预测或模型。

## 2. Single 与 rolling 的裁决

V1 主训练协议冻结为 `single_historical_14d_snapshot_v1`。单历史 snapshot 已足以建立一个严格、可审计且与正式 Validation/Test 对齐的排序链路，成本也与当前离线日批、CPU 工程边界匹配。多个 rolling historical snapshots 能提高时间稳健性和监督规模，但会成倍增加 TT1 historical full-refit、ANN、候选 replay 与特征 snapshot 成本；在单 snapshot 链路尚未验证前不应扩大搜索空间。因此 rolling 仅保留为 P1 challenger，Stage 4A 不实现。

## 3. Ranking calendar

| Snapshot | as-of/history cutoff | request date | future label window | 角色 |
| --- | --- | --- | --- | --- |
| `ranking_train_14d_v1` | 2020-08-11 | 2020-08-12 | 2020-08-12 至 2020-08-25 | 排序训练监督 |
| `validation_14d` | 2020-08-25 | 2020-08-26 | 2020-08-26 至 2020-09-08 | 模型、特征和超参数选择 |
| `test_14d` | 2020-09-08 | 2020-09-09 | 2020-09-09 至 2020-09-22 | 锁后一次正式测试 |

`ranking_train_14d_v1` 选择 2020-08-11，是因为它是冻结 outer Train 中距离正式 Validation 最近、又能完整容纳连续 14 天 future-label window 的历史请求时点。它不是从聊天记录沿用的任意日期。

## 4. Historical Candidate Replay

Stage 3E 只正式物化了 `validation_14d` 和 `test_14d`，不能复制 Validation candidate 后改日期，也不能用完整数据生成后再过滤。Stage 4B 必须以 `history <= 2020-08-11` 重建 `ranking_train_14d_v1` 的 Repurchase、IUF ItemCF、TT1、HNSW ANN、Trend 7d、Popularity fallback 与最终 Fusion，并保持 `stage_3e_candidate_contract_v1` 和 `source_rank_cap_pre_dedup_v1` 零语义漂移。

冻结语义包括：TT1、T100、HNSW cosine/M=32/efConstruction=200/efSearch=200 与 exact rerank；Repurchase 50、ItemCF 200、Two-Tower 100、Trend 100，Popularity 唯一补齐至 500；source-rank cap 在跨源去重前应用；重复候选消耗各自来源 cap 并合并 provenance；不同来源 raw score 不相加。Historical TT1 必须按历史 cutoff 构造严格因果样本，重新初始化后用锁定 epoch=1 对全部 eligible examples full-refit，internal dev=0；embedding、ANN 与 candidate 完整 lineage 到该历史 checkpoint。

Stage 4A 只做 replay capability 静态审计，不执行 historical full candidate replay。现有基础函数支持 cutoff，但部分 Stage 3B–3E 正式 runner 只注册 Validation/Test，Stage 4B 需要增加 historical snapshot/lineage adapter；这不授权改变冻结 Retrieval 语义。

Historical replay 的跨 snapshot 要求是 `historical_retrieval_protocol_identity`：contract/protocol version、TT1、T100、ANN 参数、source caps、插入顺序、duplicate/fallback/source-score 语义必须一致；它不要求历史 candidate membership/rank 等于 Validation 或 Test。

未来第一次正式物化 `ranking_train_14d_v1` 时必须创建该 snapshot 自己的新 `historical_candidate_semantic_fingerprint`。禁止要求它等于 Validation/Test fingerprint。之后对同一 historical snapshot 采用不同合法 partition/scheduling order replay 时，才要求 replay fingerprint 等于 historical formal fingerprint。未来 gate 应分别验证 `HISTORICAL_PROTOCOL_IDENTITY` 与 `HISTORICAL_REPLAY_SEMANTIC_IDENTITY`。

## 5. Ranking training population

`ranking_train_all_label_users` 是在 2020-08-12 至 2020-08-25 至少有一个 B-feedback positive 的用户。主 Ranking optimization population 冻结为 `historical_evaluable_warm_users_v1`，即同时满足：未来标签窗至少一个 B-feedback positive、2020-08-11 cutoff 前至少一个 B-feedback history interaction、customer/article internal key policy 有效。它是 Stage 2B `evaluable_warm_users` 在 historical ranking snapshot 上的等价口径；Stage 4A 不实际物化该全量 population。

`cold_history_label_users` 不进入主 optimization population，也不混入主 evaluation average，必须独立审计和评估。对主 population 用户，如果存在 future positive 但 frozen 500 candidates 中 positive 为 0，则记为 retrieval miss/all-zero relevance group：未来 LambdaRank optimization 排除，但端到端 evaluation 保留。

Offline query population 可以依据冻结的 population/label 口径确定；candidate item membership 仍必须完全 label-agnostic，不能因 query population 或 future positives 改变。

## 6. Label、未观察项与 Candidate miss

标签继续使用 B 口径。在 `(snapshot_id, customer_key, article_key)` 候选行上，商品在相应未来标签窗至少出现一次即 `label=1`；否则为 `label=0`，其唯一允许解释是“该候选商品在该未来标签窗口内未观察到购买”。它不是 dislike、真实负样本、曝光未点击或曝光未购买。

Future ground truth 只能标注 frozen candidate universe，不能改变 membership。未被召回的 future positive 是 retrieval miss，Ranking 无法恢复，也不得强行注入。Query key 冻结为 `(snapshot_id, customer_key)`。有 future positive 但 500 candidates 中无 positive 的 all-zero relevance group 不进入未来 LambdaRank optimization，却必须保留在端到端 MAP@12、NDCG@12、Recall@12 和 HitRate@12 人口中。

未来训练保留全部 candidate positives。`label=0` rows 最多稳定保留 100 条。Sampling 精确冻结为 SHA-256 与 UTF-8；canonical payload 为 `{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}`，计算 `sha256(payload.encode("utf-8")).hexdigest()`，同一 query 按 digest 升序、article key 升序打破并列后取前 100。Filesystem path、partition id、row order、process id、Python runtime hash seed 都不得进入 payload，Python built-in `hash()` 禁止使用。输入不超过 100 时全部保留；Validation/Test 评估始终评分完整 500 candidates。100 已冻结，不是 Stage 4B 搜索空间。

## 7. Feature cutoff

所有动态特征、词表、频次、趋势、归一化、clipping 与 quantile 统计都必须满足 `feature_timestamp <= snapshot_as_of_date`，并只由该 cutoff 前 history 拟合。Final refit 不能用 2020-09-08 history 重算 2020-08-11 或 2020-08-25 rows；历史 rows 必须保留各自 snapshot-time feature。

Article static attributes 没有可靠 effective timestamp。未来可使用仓库冻结的 canonical attributes，但必须保留“无法验证真实历史属性生效时间”的限制，不能伪造上架或属性生效日期。

## 8. Validation、first7d 与 Test isolation

Stage 4B 只能用 `ranking_train_14d_v1` 训练，在 `validation_14d` 选择 feature set、objective、LightGBM 超参数、regularization 与 best iteration。冻结 SPLIT-001 的原 7-day validation 落在主 Test 时段，Stage 4 在 Test 解锁前不得读取它。

`ranking_validation_first7d_diagnostic` 不是新请求 snapshot；它复用 `validation_14d` 的 candidate 与 2020-08-25 feature snapshot，只把 Validation ground truth 截为 2020-08-26 至 2020-09-01，用作 post-selection short-horizon robustness diagnostic。流程固定为：先仅用完整 14-day Validation 完成选择，再冻结唯一 winner，最后只比较该 winner 与 Stage 3E candidate-rank baseline，并且只执行一次。

First7d 不得选择 model、feature、objective、hyperparameter、best iteration 或 negative policy，不得比较多个 challenger 做选择，也不得触发 retuning。如果 14-day winner 的 delta MAP@12 大于 0、first7d delta MAP@12 小于 0，则状态为 `VALIDATION_ROBUSTNESS_NOT_MET`：不自动调参、不自动授权 Test，返回技术评审。

Validation 完成并写入 Ranking lock 后，final refit 可以合并 Ranking Train 与 Validation supervision，但两批 rows 分别保留 2020-08-11 与 2020-08-25 features；超参数和训练轮数来自 Validation lock，internal selection=0。随后才可构造 as-of 2020-09-08 Test features、评分完整 500 candidates、冻结 prediction fingerprint，最后连接 Test labels 并执行一次正式评估。Test 不得参与任何选择、早停、采样、阈值或重排决策。

## 9. 排序评估与边界

未来排序主指标为 MAP@12，关键辅助为 NDCG@12、Recall@12 和 HitRate@12；baseline 是 Stage 3E frozen `candidate_rank`。Recall@500 是 retrieval 指标，不能替代排序主指标。Stage 4A 没有生成 ranking dataset/features、没有训练或选择模型、没有评分 Test、没有实现 reranking，也没有进入 Stage 4B。

当前 `protocol_status=PASSED_PROTOCOL_GATE`、`final_acceptance_status=ACCEPTED`、`stage_4b_authorized=true`。该授权仅表示未来独立 Stage 4B Prompt 可以消费已冻结协议；historical replay、ranking dataset/features、LightGBM、Test Ranking evaluation 与 reranking 在 Stage 4A 均未执行。
