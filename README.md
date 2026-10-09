# H&M Personalized Fashion Recommendations

这是一个面向 2027 届推荐算法/机器学习算法秋招的工程项目。项目以 H&M 隐式购买反馈为基础，强调真实可信、完整工业链路、严格防泄漏与可复现性；模型复杂度服务于问题证据，不为了复杂而堆叠方案。

## 项目任务

当前冻结的业务任务是：在离线日批场景中，为已知用户预测未来 14 天可能购买的商品，主输出为 Top-12。

该任务属于隐式购买反馈 Top-K 推荐，不是无偏 CTR/CVR 预估。数据不包含曝光、点击、真实负反馈、库存、精确时间戳、`order_id` 或真实 `quantity`，因此仓库不声称具备线上 A/B Test、CTR/CVR 或业务收益结论。

## 当前进度

已完成并留有可复现代码、聚合产物或正式文档的阶段：

- **Stage 0A–2B**：完成数据侦察、契约与审计、可行性画像、架构冻结、严格时间基础层及统一评估人口；
- **Stage 3A**：完成 Popularity、Trend、Pure Repurchase、Repurchase + Popularity Fallback、统一候选协议和统一离线评估器，以及 Validation 选择、Test lock 与锁后正式评估；
- **Stage 3B**：完成 cosine/IUF ItemCF、固定预算 Base/ItemCF Union、overlap、独占命中、marginal Recall、paired user bootstrap，以及 Validation 选择、Test lock 与锁后正式评估；
- **Stage 3C**：完成结构化属性 Content Retrieval、canonical 属性审计、snapshot IDF、group-balanced 表示、固定预算 Q0/Q1/Q2、配额占用审计和 paired bootstrap。最终状态为 `FAILED_VALUE_GATE`：C3 是最佳 Content 表示，但 Q1/Q2 均弱于冻结的 Stage 3B Q0；Content Test 为 `SKIPPED_BY_PROTOCOL`，Content 不进入当前正式召回主线。
- **Stage 3D**：Stage 1C 原 MF-before-Two-Tower 路线经用户明确授权调整，本阶段跳过 MF，完成 TT0/TT1 Two-Tower、严格时间因果样本、full-refit、HNSW + exact rerank、ANN overlap audit 与固定预算增量实验。最终技术验收通过，状态为 `PASSED_VALUE_GATE`，Test 为 `SUCCESS`；锁定模型 `TT1`、固定 500 总候选预算下的融合 `T100`，以及 ANN `efSearch=200`。
- **Stage 3E**：完成 Final Retrieval Fusion & Candidate Contract Freeze，将 Stage 3D 锁定的 TT1-T100 结果冻结为 deterministic、snapshot-safe、source-aware、replayable 的正式 500-candidate contract；最终状态为 `PASSED_CONTRACT_GATE`。
- **Stage 4A**：完成 Ranking Time Protocol Freeze，并通过 ChatGPT 最终技术验收；正式状态为 `PASSED_PROTOCOL_GATE / FINAL ACCEPTED`。冻结单历史 14 天训练 snapshot、Ranking population、deterministic negative sampling、historical candidate replay、feature cutoff、Validation/first7d/Test 与 final-refit discipline。
- **Stage 4B-1**：完成 Historical Candidate Replay & Contract Gate，并通过 ChatGPT 最终技术验收；正式状态为 `PASSED_HISTORICAL_REPLAY_GATE / FINAL ACCEPTED`。针对 `ranking_train_14d_v1`（as-of 2020-08-11，request date 2020-08-12）从冻结 cutoff 前历史重新构建 Repurchase、IUF ItemCF、TT1、HNSW ANN、Trend 7d、Popularity 与 T100 fixed-budget fusion。Historical TT1 使用 fresh initialization、全部 eligible historical samples、锁定 epoch=1、internal dev=0；正式合同为每位 request user 500 candidates，保持 Stage 3E contract semantics。130001 位 request users 共生成 65000500 行正式候选，formal 与 reverse-schedule replay semantic fingerprints 完全一致，15 项 Historical Replay Contract Gate 全部 PASS。
- **Stage 4B-2**：完成 Ranking Ground Truth、candidate-level binary relevance、main/cold population、candidate-positive coverage、full500 evaluation view、optimizer sampled view、确定性 negative sampling、query/group contract、semantic fingerprint、replay identity 与 Contract Gate，并通过 ChatGPT 最终技术验收；正式状态为 `PASSED_RANKING_DATASET_CONTRACT_GATE / FINAL ACCEPTED`。正式数据覆盖 130001 位 all-label users，其中 120608 位 main history users、9393 位 cold-history users；每 query 保留 frozen 500 candidates。69111 个 main retrieval-miss/all-zero queries 保留在完整端到端评估中并排除于 optimizer；51497 个 optimizer queries 使用 SHA-256 稳定保留全部 positives，并最多保留 100 个未来窗未观察购买的 candidates。
- **Stage 4B-3**：完成 snapshot-safe Ranking Feature Pipeline，并通过 ChatGPT 最终技术验收；Codex 工程门状态为 `PASSED_RANKING_FEATURE_CONTRACT_GATE`，最终裁决为 `FINAL ACCEPTED`。F0 `core_temporal_retrieval_v1` 含 30 个 model features，F1 `core_plus_affinity_v1` 在 F0 上新增 22 个 category/affinity features，共 52 个。正式 full500 feature view 覆盖 130001 queries、65000500 rows、每 query 500 candidates；optimizer feature view 对 Stage 4B-2 frozen membership 做 exact projection，覆盖 51497 queries、5236056 rows。Candidate membership/rank/provenance 零漂移，formal/replay feature semantic fingerprint 完全一致，future/Validation/Test reads 均为 0，20 项 G1–G20 全部 PASS。current final source 还在固定 partition 0 上独立复现 1997000 rows，与 formal partition 的 row/schema/52-feature/null/query digest/partition semantic fingerprint mismatch 均为 0，且未改写 formal producer output。
- **Stage 4B-4**：完成 LambdaRank Ranking 的训练、Validation 选择、锁后 First7d 诊断、Train+Validation final refit、Test prediction freeze 与一次性 Test evaluation；用户最终裁决为 `STAGE_4B4_PHASE5B_FINAL_ACCEPTED / STAGE_4B4_FINAL_ACCEPTED`。锁定模型 `L1_FINAL_REFIT_600` 使用 F1 的 52 个特征、LightGBM `lambdarank` objective、`MAP@12` 选择规则与 600 棵树；最终 Test value gate 为 `PASSED_FINAL_TEST_VALUE_GATE`，统计状态为 `POSITIVE_AND_SIGNIFICANT`，关键分群状态为 `NO_CRITICAL_SEGMENT_REGRESSION`。

Stage 3D 的固定预算主结果如下：

| Phase | Q0 Recall@500 | TT1-T100 Recall@500 | ΔRecall@500 |
| --- | ---: | ---: | ---: |
| Validation | 0.22050126975251422 | 0.2448639763673124 | +0.02436270661479820 |
| Test | 0.2146086294269453 | 0.24138900164539004 | +0.02678037221844473 |

Validation 与 Test 的 paired bootstrap CI 均严格大于 0。Test 只在 Validation lock 后执行，没有用于模型、配额、epoch 或 ANN 调参。Two-Tower 当前主要改善深层 Recall@500；冻结顺序的 Validation/Test MAP@12 分别为 0.012184760793980384 和 0.014657496029670704，不能据此声称浅层排序质量已经提升。

Stage 3E 保持上述 Stage 3D 指标零语义漂移，没有新增 Recall 或 MAP 增益。Final V1 Retrieval 为：Repurchase 50、IUF ItemCF 200、TT1 Two-Tower 100、Trend 7d 100，Popularity 仅做唯一候选补齐，总候选预算固定为 500。各路 source-rank cap 在跨来源去重前应用；重复候选占用各自来源配额并合并 provenance，不同来源 raw score 不跨源相加。

Stage 4A 主协议为 `single_historical_14d_snapshot_v1`：Ranking Train as-of 2020-08-11、labels 2020-08-12 至 2020-08-25；Validation as-of 2020-08-25、labels 2020-08-26 至 2020-09-08；Test as-of 2020-09-08、labels 2020-09-09 至 2020-09-22。Stage 4B-1 已按历史 cutoff replay Stage 3E frozen retrieval semantics；historical TT1 使用严格因果样本、重新初始化并按锁定 epoch=1 full-refit，并创建该 snapshot 自己的 semantic fingerprint。

Stage 4B-2 已物化 `ranking_dataset_14d_binary_v1`：future B-feedback purchase 标为 `label=1`；`label=0` 仅表示未来标签窗未观察到购买，不是真实负反馈、dislike 或 exposed-not-purchased。Frozen candidate membership/rank/provenance 在 future label join 前确定，future positives 不注入 candidate。完整 500-candidate view 是后续 Validation/Test 的正式 evaluation universe；sampled optimizer supervision 不能替代它。动态特征和统计仍只能使用各 snapshot as-of 前历史；First7d 仅是锁定 14-day winner 后的一次 robustness diagnostic；Test 不得参与选择。

Stage 4B-3 已物化 snapshot-safe canonical full500 与 optimizer feature views。所有动态 user/item/user-item/category affinity 统计只读取 `2020-08-11` cutoff 当日及以前的 B-feedback history；retrieval scores 保持 source-local，`primary_source` 仅供审计且不进入模型，price 因 lineage 不确定而省略。Article static attributes 继续保留 effective timestamp 无法验证的限制。缺失 recency 保持 null，已知 category 但无匹配历史的 affinity 为 0，缺失 user/category 的 affinity 为 null。正式 feature semantic fingerprint 与 8/32 deterministic replay reconciliation fingerprint exact equality。Producer source correction 后，current final source 在独立 temporary 路径复现 partition 0；formal 与 reproduction partition digest exact equality，全部 exact mismatch 为 0，formal producer output 未被覆盖。精确 fingerprints 见正式 stage report、manifest 与 current-source reproduction audit。

### Stage 4B-4 最终学习排序结果

Stage 4B-4 严格沿用 Stage 4A 的 Train → Validation → First7d diagnostic → Train+Validation final refit → Test prediction freeze → one-time Test evaluation 时序。Validation 只用于选择 F0/F1 与树数；First7d 只在 14-day winner 锁定后做一次方向性诊断，没有形成第二选择集；final refit 保留旧 snapshot 特征、使用 fresh initialization 合并 Train 与 Validation optimizer supervision；Test 排序分数和指纹先冻结，之后才进行唯一一次 Test 标签读取与正式评估。Test 没有参与模型、特征、树数、候选或阈值选择。

Test 主评估人口为 117366 位 `evaluable_warm_users`，B0 与 `L1_FINAL_REFIT_600` 均在完整 frozen 500-candidate universe 上评估，主输出仍为 Top-12：

| Version | MAP@12 | NDCG@12 | Recall@12 | HitRate@12 |
| --- | ---: | ---: | ---: | ---: |
| B0 frozen candidate order | 0.014657 | 0.022618 | 0.033072 | 0.068214 |
| L1_FINAL_REFIT_600 | 0.029620 | 0.046586 | 0.065511 | 0.146703 |

MAP@12 绝对增量为 `+0.014963`，相对 B0 约为 `+102.08%`；1000 次 paired user-block percentile bootstrap 的 95% CI 为 `[0.014437, 0.015461]`。完整精确值见 [`test_evaluation_metrics.csv`](artifacts/stage_4b4/test_evaluation_metrics.csv)、[`test_evaluation_bootstrap.csv`](artifacts/stage_4b4/test_evaluation_bootstrap.csv) 与 [`Stage 4B-4 Test evaluation report`](docs/stages/stage_4b4_test_evaluation.md)。这是严格时间切分下的一次性离线 Test 结果，不等于线上 CTR/CVR、收入、业务收益或 A/B Test 结论。

Test access audit 记录 raw Test label source、canonical ground-truth build、prediction-label join、formal evaluation 与 metric materialization 均恰好执行 1 次，Validation/First7d label reads 为 0，标签没有 reopen。Phase 4 当前 final-refit lock 为 `358c25d1...f4bd8`，旧 Phase 4 lock `de1072d4...2a3d6` 已由恢复流程明确 supersede；Phase 5A pre-label lock 为 `34c8dcc5...7e8ca`，Phase 5B final lock 为 `daddf4c1...b1b88`。完整值分别见 [`stage_4b4_final_refit_lock.json`](artifacts/stage_4b4/stage_4b4_final_refit_lock.json)、[`stage_4b4_test_pre_label_lock.json`](artifacts/stage_4b4/stage_4b4_test_pre_label_lock.json) 与 [`stage_4b4_test_evaluation_lock.json`](artifacts/stage_4b4/stage_4b4_test_evaluation_lock.json)。

结果仍受召回上限约束：117366 个主评估 query 中有 62383 个 candidate-positive all-zero query（约 53.15%），其 Top-12 指标按协议保留为 0，排序模型无法恢复 frozen 500 candidates 之外的未来正例。Cold-history 诊断的 MAP@12 增量约为 `+0.000172`，95% CI 跨 0，因此只能称为 inconclusive 且未发现显著退化；cold-item/zero-history 目标切片仍为 0。数据缺少可靠 article effective timestamp，price 因 lineage 不确定而省略；同时没有曝光、点击、真实负反馈、库存或可售状态，因此 `label=0` 只能解释为未来窗未观察到购买。

详细指标与限制见 [`docs/stages/stage_3a_baseline_retrieval_and_evaluation.md`](docs/stages/stage_3a_baseline_retrieval_and_evaluation.md)、[`docs/stages/stage_3b_itemcf_and_incremental_recall.md`](docs/stages/stage_3b_itemcf_and_incremental_recall.md)、[`docs/stages/stage_3c_content_retrieval_and_incremental_recall.md`](docs/stages/stage_3c_content_retrieval_and_incremental_recall.md)、[`docs/stages/stage_3d_two_tower_retrieval_and_incremental_recall.md`](docs/stages/stage_3d_two_tower_retrieval_and_incremental_recall.md)、[`docs/stages/stage_3e_final_retrieval_fusion_and_candidate_contract.md`](docs/stages/stage_3e_final_retrieval_fusion_and_candidate_contract.md)、[`docs/stages/stage_4b1_historical_candidate_replay.md`](docs/stages/stage_4b1_historical_candidate_replay.md)、[`docs/stages/stage_4b2_ranking_dataset_and_negative_sampling.md`](docs/stages/stage_4b2_ranking_dataset_and_negative_sampling.md)、[`docs/stages/stage_4b3_ranking_feature_pipeline.md`](docs/stages/stage_4b3_ranking_feature_pipeline.md)、[`docs/stages/stage_4b4_lambdarank_ranking.md`](docs/stages/stage_4b4_lambdarank_ranking.md)、[`docs/stages/stage_4b4_validation_selection.md`](docs/stages/stage_4b4_validation_selection.md)、[`docs/stages/stage_4b4_first7d_diagnostic.md`](docs/stages/stage_4b4_first7d_diagnostic.md)、[`docs/stages/stage_4b4_locked_final_refit.md`](docs/stages/stage_4b4_locked_final_refit.md)、[`docs/stages/stage_4b4_test_prediction_freeze.md`](docs/stages/stage_4b4_test_prediction_freeze.md) 与 [`docs/stages/stage_4b4_test_evaluation.md`](docs/stages/stage_4b4_test_evaluation.md) 及对应公开聚合产物。

Stage 4B-4 已最终验收通过并冻结。下一阶段尚未开始，必须由新的独立正式 Prompt 授权；不得重开 Test 标签、重算正式 Test 指标/bootstrap、重训或替换 winner、修改 frozen predictions、用 Test 调参，亦不得重新调整 Stage 3E retrieval 或 Stage 4B-3 frozen feature semantics。

## 数据概览

本地原始数据规模如下。原始 Parquet 仅供本地只读使用，不纳入版本控制；准备方式见 [`data/README.md`](data/README.md)。

| 逻辑表 | 行数 | 文件大小（bytes） |
| --- | ---: | ---: |
| articles | 105,542 | 6,378,835 |
| customers | 1,371,980 | 167,711,266 |
| transactions | 31,788,324 | 774,279,764 |
| **合计** | **33,265,846** | **948,369,865** |

正式数据规模、切分边界和评估人口应以对应 manifest 与阶段产物为准：

- [`artifacts/stage_2a/split_manifest.csv`](artifacts/stage_2a/split_manifest.csv)
- [`artifacts/stage_2b/evaluation_population_manifest.csv`](artifacts/stage_2b/evaluation_population_manifest.csv)

## 冻结协议

V1 架构、反馈语义、时间切分和评估规则已经文档化：

- [`docs/architecture/v1_recommendation_architecture.md`](docs/architecture/v1_recommendation_architecture.md)
- [`docs/decisions/feedback_semantics.md`](docs/decisions/feedback_semantics.md)
- [`docs/decisions/time_split_protocol.md`](docs/decisions/time_split_protocol.md)
- [`docs/decisions/internal_id_policy.md`](docs/decisions/internal_id_policy.md)
- [`docs/decisions/evaluation_population_protocol.md`](docs/decisions/evaluation_population_protocol.md)
- [`docs/decisions/evaluation_protocol.md`](docs/decisions/evaluation_protocol.md)
- [`docs/decisions/model_selection_matrix.md`](docs/decisions/model_selection_matrix.md)
- [`docs/decisions/stage_3d_two_tower_route_override.md`](docs/decisions/stage_3d_two_tower_route_override.md)
- [`configs/stage_3e_candidate_contract.yaml`](configs/stage_3e_candidate_contract.yaml)
- [`artifacts/stage_3e/stage_3e_manifest.csv`](artifacts/stage_3e/stage_3e_manifest.csv)
- [`configs/stage_4a_ranking_time_protocol.yaml`](configs/stage_4a_ranking_time_protocol.yaml)
- [`docs/decisions/ranking_time_protocol.md`](docs/decisions/ranking_time_protocol.md)
- [`artifacts/stage_4a/stage_4a_manifest.csv`](artifacts/stage_4a/stage_4a_manifest.csv)
- [`configs/stage_4b1_historical_replay.yaml`](configs/stage_4b1_historical_replay.yaml)
- [`artifacts/stage_4b1/stage_4b1_manifest.csv`](artifacts/stage_4b1/stage_4b1_manifest.csv)
- [`artifacts/stage_4b1/contract_gate.csv`](artifacts/stage_4b1/contract_gate.csv)
- [`configs/stage_4b2_ranking_dataset.yaml`](configs/stage_4b2_ranking_dataset.yaml)
- [`artifacts/stage_4b2/stage_4b2_manifest.csv`](artifacts/stage_4b2/stage_4b2_manifest.csv)
- [`artifacts/stage_4b2/contract_gate.csv`](artifacts/stage_4b2/contract_gate.csv)
- [`configs/stage_4b3_ranking_features.yaml`](configs/stage_4b3_ranking_features.yaml)
- [`artifacts/stage_4b3/stage_4b3_manifest.csv`](artifacts/stage_4b3/stage_4b3_manifest.csv)
- [`artifacts/stage_4b3/contract_gate.csv`](artifacts/stage_4b3/contract_gate.csv)
- [`artifacts/stage_4b3/current_source_reproduction_audit.csv`](artifacts/stage_4b3/current_source_reproduction_audit.csv)
- [`configs/stage_4b4_lambdarank.yaml`](configs/stage_4b4_lambdarank.yaml)
- [`configs/stage_4b4_validation_selection.yaml`](configs/stage_4b4_validation_selection.yaml)
- [`configs/stage_4b4_final_refit.yaml`](configs/stage_4b4_final_refit.yaml)
- [`configs/stage_4b4_test_prediction_freeze.yaml`](configs/stage_4b4_test_prediction_freeze.yaml)
- [`configs/stage_4b4_test_evaluation.yaml`](configs/stage_4b4_test_evaluation.yaml)
- [`artifacts/stage_4b4/stage_4b4_final_refit_lock.json`](artifacts/stage_4b4/stage_4b4_final_refit_lock.json)
- [`artifacts/stage_4b4/stage_4b4_test_pre_label_lock.json`](artifacts/stage_4b4/stage_4b4_test_pre_label_lock.json)
- [`artifacts/stage_4b4/stage_4b4_test_evaluation_lock.json`](artifacts/stage_4b4/stage_4b4_test_evaluation_lock.json)
- [`artifacts/stage_4b4/test_evaluation_contract_gate.csv`](artifacts/stage_4b4/test_evaluation_contract_gate.csv)

仓库级操作约束见 [`AGENTS.md`](AGENTS.md)。

## 尚未实现

- MF/BPR；
- DeepFM 等排序 challenger；
- 多样性、新颖性或长尾控制等 reranking 逻辑；
- 线上服务、监控和部署。

V1 不设置独立粗排；当前不存在独立粗排实现。以上能力均不能从现有离线结果推断为已经实现。Stage 3A–4B-4 的 Recall、HitRate、MAP、NDCG 等均为离线指标，不等于 CTR/CVR、线上收益或线上 A/B Test 结论。

## 目录结构

```text
.
├── artifacts/          # 各阶段可公开的聚合结果、manifest 与检查记录
├── configs/            # 数据契约及 Stage 1A–4B-4 可复现配置
├── data/
│   ├── README.md       # 本地原始数据准备说明
│   └── raw/            # 本地只读 Parquet，已被 Git 忽略
├── docs/
│   ├── architecture/   # V1 推荐架构权威文档
│   ├── decisions/      # 反馈、切分、ID、评估等冻结协议
│   └── stages/         # Stage 0A–4B-4 阶段报告
├── scripts/            # 各已完成阶段的可执行入口
├── src/
│   ├── data_audit/     # Stage 0B–1B 数据审计与画像代码
│   ├── data_foundation/ # Stage 2A–2B 数据基础与评估对象代码
│   ├── retrieval/      # Stage 3A–4B-1 baseline、统一评估、召回与 Candidate Contract 代码
│   ├── ranking_protocol/ # Stage 4A 时间协议与静态审计 helper
│   ├── ranking_dataset/ # Stage 4B-2 labels、populations、views、sampling 与 contract audit
│   ├── ranking_features/ # Stage 4B-3 snapshot-safe feature materialization 与 audit
│   └── ranking_model/  # Stage 4B-4 LambdaRank 训练、选择、锁定预测与一次性评估
├── tests/              # 不依赖完整原始数据的单元测试
├── AGENTS.md           # Codex 与自动化任务的仓库级操作规约
├── pyproject.toml      # 当前阶段的最小 Python 环境配置
└── README.md
```

## 本地环境与验证

项目使用 Python 3.10–3.12，建议始终通过项目虚拟环境运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest
```

命令不会自动下载 H&M 原始数据。未经明确阶段任务，不应运行下一阶段脚本或改变冻结协议。
