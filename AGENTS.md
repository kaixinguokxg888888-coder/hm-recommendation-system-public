# AGENTS.md

本文件是本仓库中 Codex 与其他自动化编码任务的长期操作规约，适用于仓库根目录及全部子目录。
任何新窗口都应先读取本文件，再读取当前任务相关的正式配置、manifest、决策文档和阶段报告。
本文件约束工作方式，不替代各阶段技术协议，也不授权自动进入下一阶段。

## 1. 项目使命

本项目服务于 2027 届推荐算法/机器学习算法秋招。
项目依次重视秋招展示价值、结论真实可信、工业链路完整、数据与实验可复现、工程实现可测试可审计，以及模型复杂度与数据证据匹配。
不得为了复杂而堆叠模型、特征或系统组件。

任何架构或模型选择都必须回答：

- 为什么选择该方案；
- 为什么不用可替代方案；
- H&M 当前数据是否支持；
- 该方案服务哪个用户群或业务问题；
- 如何以离线实验验证；
- 失败时的停止条件是什么。

不得依据模型流行度、论文热度或链路形式完整性替代数据证据。

## 2. 项目任务边界

当前冻结任务是基于隐式购买反馈的未来 14 天 Top-K 商品推荐。
推荐场景是离线日批，为已知用户生成商品列表，主输出为 Top-12。
主任务不是无偏 CTR/CVR 预估。

当前数据不具备：

- 曝光日志；
- 点击日志；
- 真实负反馈；
- 库存和可售状态；
- 商品精确上架时间；
- 日内精确时间戳；
- `order_id`；
- 真实 `quantity`。

因此不得声称：

- 已完成线上 A/B Test；
- 已获得 CTR 或 CVR 提升；
- 离线未购买等于用户不喜欢；
- 重复交易行等于真实购买数量；
- 已验证库存、曝光或线上业务收益。

数据不支持的任务不得通过假设补齐后包装为正式能力。

## 3. 冻结协议索引

下列文件是对应主题的权威来源：

- 业务任务与 V1 总体架构：`docs/architecture/v1_recommendation_architecture.md`
- 反馈 A/B/C 口径：`docs/decisions/feedback_semantics.md`
- Validation、Test 与开发窗口：`docs/decisions/time_split_protocol.md`
- V1 用户分群、模型取舍与适用边界：`docs/decisions/model_selection_matrix.md`
- 内部 customer/article key：`docs/decisions/internal_id_policy.md`
- Ground truth、评估人口与候选域：`docs/decisions/evaluation_population_protocol.md`
- 离线指标、主指标和报告维度：`docs/decisions/evaluation_protocol.md`
- Stage 1C 实验优先级：`artifacts/stage_1c/experiment_priority.csv`
- Stage 2A 可执行切分配置：`configs/stage_2a_split.yaml`
- Stage 2B 标签与评估人口配置：`configs/stage_2b_labels.yaml`
- 已落地切分事实：`artifacts/stage_2a/split_manifest.csv`
- 已落地评估人口事实：`artifacts/stage_2b/evaluation_population_manifest.csv`
- Stage 3A 正式配置：`configs/stage_3a_recall.yaml`
- Stage 3A manifest：`artifacts/stage_3a/stage_3a_manifest.csv`
- Stage 3A Test lock：`artifacts/stage_3a/stage_3a_lock.json`
- Stage 3A 收尾维护 manifest：`artifacts/stage_3a/stage_3a_maintenance_manifest.json`
- Stage 3B 正式配置：`configs/stage_3b_itemcf.yaml`
- Stage 3B manifest：`artifacts/stage_3b/stage_3b_manifest.csv`
- Stage 3B Test lock：`artifacts/stage_3b/stage_3b_lock.json`
- Stage 3C 正式配置：`configs/stage_3c_content.yaml`
- Stage 3C manifest：`artifacts/stage_3c/stage_3c_manifest.csv`
- Stage 3C Validation lock：`artifacts/stage_3c/stage_3c_lock.json`
- Stage 3C 阶段报告：`docs/stages/stage_3c_content_retrieval_and_incremental_recall.md`
- Stage 3D 正式配置：`configs/stage_3d_two_tower.yaml`
- Stage 3D 路线调整 decision：`docs/decisions/stage_3d_two_tower_route_override.md`
- Stage 3D manifest：`artifacts/stage_3d/stage_3d_manifest.csv`
- Stage 3D Test lock：`artifacts/stage_3d/stage_3d_lock.json`
- Stage 3D 阶段报告：`docs/stages/stage_3d_two_tower_retrieval_and_incremental_recall.md`
- Stage 3E 正式配置：`configs/stage_3e_candidate_contract.yaml`
- Stage 3E manifest：`artifacts/stage_3e/stage_3e_manifest.csv`
- Stage 3E Contract Gate：`artifacts/stage_3e/contract_gate.csv`
- Stage 3E Public Finalization audit：`artifacts/stage_3e/public_finalization_audit.csv`
- Stage 3E 阶段报告：`docs/stages/stage_3e_final_retrieval_fusion_and_candidate_contract.md`
- Stage 4A 正式配置：`configs/stage_4a_ranking_time_protocol.yaml`
- Stage 4A Ranking Time Protocol decision：`docs/decisions/ranking_time_protocol.md`
- Stage 4A manifest：`artifacts/stage_4a/stage_4a_manifest.csv`
- Stage 4A Protocol Gate：`artifacts/stage_4a/protocol_gate.csv`
- Stage 4A 阶段报告：`docs/stages/stage_4a_ranking_time_protocol_freeze.md`
- Stage 4B-1 正式配置：`configs/stage_4b1_historical_replay.yaml`
- Stage 4B-1 manifest：`artifacts/stage_4b1/stage_4b1_manifest.csv`
- Stage 4B-1 Historical Replay Contract Gate：`artifacts/stage_4b1/contract_gate.csv`
- Stage 4B-1 阶段报告：`docs/stages/stage_4b1_historical_candidate_replay.md`
- Stage 4B-2 正式配置：`configs/stage_4b2_ranking_dataset.yaml`
- Stage 4B-2 manifest：`artifacts/stage_4b2/stage_4b2_manifest.csv`
- Stage 4B-2 Contract Gate：`artifacts/stage_4b2/contract_gate.csv`
- Stage 4B-2 阶段报告：`docs/stages/stage_4b2_ranking_dataset_and_negative_sampling.md`
- Stage 4B-3 正式配置：`configs/stage_4b3_ranking_features.yaml`
- Stage 4B-3 manifest：`artifacts/stage_4b3/stage_4b3_manifest.csv`
- Stage 4B-3 Contract Gate：`artifacts/stage_4b3/contract_gate.csv`
- Stage 4B-3 Current-source reproduction audit：`artifacts/stage_4b3/current_source_reproduction_audit.csv`
- Stage 4B-3 阶段报告：`docs/stages/stage_4b3_ranking_feature_pipeline.md`
- Stage 4B-4 正式训练配置：`configs/stage_4b4_lambdarank.yaml`
- Stage 4B-4 Validation 选择配置：`configs/stage_4b4_validation_selection.yaml`
- Stage 4B-4 First7d 诊断配置：`configs/stage_4b4_first7d_diagnostic.yaml`
- Stage 4B-4 final-refit 配置：`configs/stage_4b4_final_refit.yaml`
- Stage 4B-4 Test prediction freeze 配置：`configs/stage_4b4_test_prediction_freeze.yaml`
- Stage 4B-4 Test evaluation 配置：`configs/stage_4b4_test_evaluation.yaml`
- Stage 4B-4 manifest：`artifacts/stage_4b4/stage_4b4_manifest.csv`
- Stage 4B-4 Validation lock：`artifacts/stage_4b4/stage_4b4_validation_selection_lock.json`
- Stage 4B-4 current final-refit lock：`artifacts/stage_4b4/stage_4b4_final_refit_lock.json`
- Stage 4B-4 Test pre-label lock：`artifacts/stage_4b4/stage_4b4_test_pre_label_lock.json`
- Stage 4B-4 Test evaluation lock：`artifacts/stage_4b4/stage_4b4_test_evaluation_lock.json`
- Stage 4B-4 Test Contract Gate：`artifacts/stage_4b4/test_evaluation_contract_gate.csv`
- Stage 4B-4 阶段报告：`docs/stages/stage_4b4_lambdarank_ranking.md`、`docs/stages/stage_4b4_validation_selection.md`、`docs/stages/stage_4b4_first7d_diagnostic.md`、`docs/stages/stage_4b4_locked_final_refit.md`、`docs/stages/stage_4b4_test_prediction_freeze.md`、`docs/stages/stage_4b4_test_evaluation.md`

阶段实现与验收证据见：

- `docs/stages/stage_0a_workspace_data_recon.md`
- `docs/stages/stage_0b_contract_and_smoke.md`
- `docs/stages/stage_1a_full_data_quality_audit.md`
- `docs/stages/stage_1b_recommendation_feasibility_profile.md`
- `docs/stages/stage_1c_architecture_freeze.md`
- `docs/stages/stage_2a_time_split_and_foundation.md`
- `docs/stages/stage_2b_labels_and_evaluation_population.md`
- `docs/stages/stage_3a_baseline_retrieval_and_evaluation.md`
- `docs/stages/stage_3b_itemcf_and_incremental_recall.md`
- `docs/stages/stage_3c_content_retrieval_and_incremental_recall.md`
- `docs/stages/stage_3d_two_tower_retrieval_and_incremental_recall.md`
- `docs/stages/stage_3e_final_retrieval_fusion_and_candidate_contract.md`
- `docs/stages/stage_4a_ranking_time_protocol_freeze.md`
- `docs/stages/stage_4b1_historical_candidate_replay.md`
- `docs/stages/stage_4b2_ranking_dataset_and_negative_sampling.md`
- `docs/stages/stage_4b3_ranking_feature_pipeline.md`
- `docs/stages/stage_4b4_lambdarank_ranking.md`
- `docs/stages/stage_4b4_validation_selection.md`
- `docs/stages/stage_4b4_first7d_diagnostic.md`
- `docs/stages/stage_4b4_locked_final_refit.md`
- `docs/stages/stage_4b4_test_prediction_freeze.md`
- `docs/stages/stage_4b4_test_evaluation.md`

除非用户提供新的正式阶段任务并明确要求评审或变更，不得擅自改变冻结协议。
发现配置、manifest、决策文档或阶段报告互相矛盾时，必须停止有争议的实现、列明冲突并等待确认。
不得静默选择有利于当前实现的一种口径。

## 4. 当前已完成阶段

- Stage 0A：完成工作区和原始 Parquet 的只读侦察与脱敏记录。
- Stage 0B：建立数据契约、审计框架和受限 smoke test。
- Stage 1A：完成全量 schema、质量、主外键、重复和取值范围审计。
- Stage 1B：完成用户活跃、序列、商品长尾、复购、冷启动和时间窗口聚合分析。
- Stage 1C：冻结业务任务、反馈、切分、V1 架构、模型路线和评估协议。
- Stage 2A：落地严格时间切分、反馈层、内部键、日历审计和防泄漏基础。
- Stage 2B：落地统一 ground truth、评估人口、用户分群、商品状态和候选域。
- Stage 3A：完成 Popularity、Trend、Repurchase baseline、统一候选协议、统一离线评估器、Validation 选择、Test lock 和锁后正式评估。
- Stage 3B：完成 Cosine/IUF ItemCF、商品邻居与用户候选生成、固定预算多路 Union、overlap、独占命中、marginal Recall、paired user bootstrap、Validation 选择和锁后正式评估。
- Stage 3C：完成结构化属性 Content Retrieval、canonical 属性审计、snapshot IDF、group-balanced 表示、固定预算 Q0/Q1/Q2、配额占用审计和 paired bootstrap；最终状态为 `FAILED_VALUE_GATE`，Content Test 按协议跳过。
- Stage 3D：完成 TT0/TT1 Two-Tower；TT1 融合 Stage 3C canonical 结构化属性；完成严格时间因果训练样本、in-batch false-negative mask、full-refit final checkpoint、HNSW + exact rerank、ANN exact-overlap audit、Q0/T50/T100 固定预算增量实验、paired bootstrap、Validation lock 和锁后 Test；最终状态为 `PASSED_VALUE_GATE`。
- Stage 3E：完成 Final Retrieval Fusion 与 Candidate Contract Freeze，将锁定的 TT1-T100 迁移为确定性、snapshot-safe、source-aware、可 replay 的 500-candidate 正式合同；最终状态为 `PASSED_CONTRACT_GATE`。
- Stage 4A：完成 Ranking Time Protocol Freeze，冻结单历史 14 天训练 snapshot、Ranking population、deterministic negative sampling、historical candidate replay、feature cutoff、Validation/first7d/Test 与 final-refit discipline；状态为 `PASSED_PROTOCOL_GATE / FINAL ACCEPTED`。
- Stage 4B-1：完成 `ranking_train_14d_v1` Historical Candidate Replay，按照 Stage 4A 冻结时间协议从 2020-08-11 cutoff 前历史重新构建 Repurchase、IUF ItemCF、TT1、ANN、Trend 与 Popularity，保持 Stage 3E 500-candidate contract 与 frozen T100 retrieval semantics，完成 historical TT1 full-refit lineage、historical semantic fingerprint 和不同合法 scheduling order 的 deterministic replay；最终状态为 `PASSED_HISTORICAL_REPLAY_GATE / FINAL ACCEPTED`。
- Stage 4B-2：完成 `ranking_train_14d_v1` Ranking ground truth、candidate-level binary relevance、main/cold population、candidate-positive coverage、retrieval miss/all-zero handling、full500 evaluation view、optimizer sampled view、SHA-256 deterministic negative sampling、query/group contract、semantic fingerprint、replay identity 与 22 项 Contract Gate；最终状态为 `PASSED_RANKING_DATASET_CONTRACT_GATE / FINAL ACCEPTED`。
- Stage 4B-3：完成 `ranking_train_14d_v1` snapshot-safe Ranking Feature Pipeline、F0/F1 52-feature contract、full500/optimizer exact projection、cutoff-safe user/item/user-item/category affinity features、deterministic semantic fingerprint、8/32 partition replay、current final source reproduction 与 20 项 Contract Gate；Codex 工程门状态为 `PASSED_RANKING_FEATURE_CONTRACT_GATE`，ChatGPT 最终技术验收状态为 `FINAL ACCEPTED`。
- Stage 4B-4：完成 LambdaRank Ranking 训练、Validation 选择、锁后 First7d 诊断、Train+Validation final refit、Test prediction freeze 与一次性 Test evaluation；最终模型为 F1 52-feature 的 `L1_FINAL_REFIT_600`，Test 主人口 MAP@12 从 B0 的 `0.014657` 提升到 `0.029620`，paired bootstrap 95% CI 为 `[0.014437, 0.015461]`；状态为 `PASSED_FINAL_TEST_VALUE_GATE / STAGE_4B4_PHASE5B_FINAL_ACCEPTED / STAGE_4B4_FINAL_ACCEPTED`。完整精确值以 `artifacts/stage_4b4/test_evaluation_metrics.csv` 与 `artifacts/stage_4b4/test_evaluation_bootstrap.csv` 为准。

仓库已实现 baseline 召回、ItemCF、结构化属性 Content Retrieval、Two-Tower、HNSW 检索与 exact rerank、多路固定预算 Union、统一离线评估、正式 Candidate Contract、Ranking Time Protocol、historical replay、ranking dataset/features、LightGBM LambdaRank、Validation selection、First7d diagnostic、final refit、Test prediction freeze 与一次性 Test Ranking evaluation。
当前尚未实现 MF/BPR、DeepFM 等新 challenger、额外 reranking 策略或线上服务；V1 不设置独立粗排。
Stage 4B-4 已完成并通过最终技术验收。下一阶段尚未开始，只有新的独立正式 Prompt 才能授权；不得重开 Test 标签、重算正式 Test 指标/bootstrap、重训或替换 winner、修改 frozen predictions、使用 Test 调参，亦不得改动 Stage 3E retrieval 或 Stage 4B-3 frozen feature semantics。

## 5. 数据与隐私规则

`data/raw/` 永远只读；原始数据不得修改、覆盖、移动、转换或重写，也不得提交 Git。
`artifacts/**/private/` 和各阶段 `tmp/` 不得提交 Git。
内部 customer/article key 映射属于私有产物。

不得输出或提交真实的：

- `customer_id`；
- `article_id`；
- `postal_code`；
- 原始 ID 片段或可逆样本；
- 本地用户名；
- 本地绝对路径；
- Python 解释器绝对路径；
- 工作区定位和不必要的机器信息。

公开产物只允许保存完成任务所需的聚合、匿名内部 key、fingerprint、状态和可复现元数据。
大型中间产物必须可由正式配置和上游产物重建，并由 Git 忽略。
不得使用 `git add -f` 绕过忽略规则。
不得为方便当前任务覆盖 Stage 0A–2B 的历史正式产物。
需要修正历史结论时，应新增明确的更正记录并获得用户确认。

## 6. 时间与防泄漏规则

所有统计、召回、特征、采样和评估必须携带：

- `snapshot_id`；
- `as_of_date` 或等价 cutoff；
- 输入数据或配置 fingerprint。

Validation 与 Test 必须使用各自独立的 history。
Validation 用于开发、选择和调参；Test 只用于最终锁定后的报告，不得用于调参。
Validation 标签不得进入 base train history；Test 标签不得进入 refit history。
热度、趋势、共现、分位点、归一化参数、负采样分布和候选配额统计只能从对应 snapshot 的 history 计算。
标签窗口只能用于构造 ground truth 和评估，不能反向生成特征或候选。
内部 key 可以覆盖完整 customers/articles 维表，但 key 的存在不能赋予实体在 cutoff 前的业务可见性。
缺少商品上架时间和库存时，必须保留候选可售性与 cold-item 可见性的限制说明。
禁止将随机切分作为正式主评估。
如需 rolling backtest，必须按冻结协议明确优先级和独立 cutoff。

## 7. Ground Truth 与评估人群

B 口径是主训练与主评估反馈：同一 `t_dat + customer_id + article_id` 按 user-item-day 二值化。
A 口径保留原始交易行，仅用于审计和敏感性分析。
C 口径是日级重复强度辅助，可以截断或对数压缩，但不是真实 `quantity`。

主协同评估人口是有标签且 cutoff 前存在历史的 `evaluable_warm_users`。
`cold_history_label_users` 必须单独报告，不得混入主协同平均后掩盖差异。

冻结的 V1 用户分群是：

- cold：历史长度为 0；
- short-history：历史长度为 1–4；
- warm：历史长度为 5–20；
- active：历史长度为 21 及以上。

这些阈值是当前项目的 V1 工程规则，不是行业统一标准。
主 warm candidate universe 是对应 history 中已出现的商品。
cold-item 评估可使用完整 articles 身份域，但不得把完整维表称为历史可售域。
warm candidate universe 与 full universe 不得混淆。
所有模型和 baseline 必须复用 Stage 2B 统一 ground truth、评估人口、分群和分母。
不得由单个模型重新定义 ground truth、可评用户或指标分母。

## 8. 推荐实现约束

已经完成：

- Popularity baseline；
- Trend baseline；
- Repurchase baseline；
- 统一候选协议；
- 统一离线评估器；
- Cosine/IUF ItemCF；
- 固定预算 Union；
- overlap 与独占命中；
- marginal Recall；
- paired user bootstrap；
- 结构化属性 Content Retrieval、canonical 属性审计、snapshot IDF 与 group-balanced 表示；
- Content 固定预算 Q0/Q1/Q2 与 source-rank cap 有效占用审计；
- TT0/TT1 Two-Tower、严格时间因果训练样本与 in-batch false-negative mask；
- full-refit final checkpoint、HNSW + exact rerank 与 ANN exact-overlap audit；
- Q0/T50/T100 固定 500 总候选预算增量实验、Validation lock 与锁后 Test。
- Stage 3E TT1-T100 最终召回合同、零语义漂移、replay audit、source provenance 与公开合同门。
- Stage 4A single historical ranking snapshot、Ranking population、deterministic negative sampling、historical replay/fingerprint、snapshot-time feature、first7d 与 Validation/Test/final-refit protocol。
- Stage 4B-1 `ranking_train_14d_v1` historical full candidate replay、historical TT1 full-refit lineage、ANN audit、frozen T100 fusion、semantic fingerprint、deterministic replay 与 contract gate。
- Stage 4B-2 `ranking_train_14d_v1` Ranking ground truth、candidate binary label、main/cold population、full500 evaluation view、optimizer sampled view、SHA-256 deterministic negative sampling、query/group contract、semantic fingerprint、replay 与 contract gate。
- Stage 4B-3 `ranking_train_14d_v1` snapshot-safe F0/F1 Ranking Feature Pipeline、52-feature schema、full500/optimizer exact projection、deterministic fingerprint/replay、current-source reproduction 与 contract gate。
- Stage 4B-4 F0/F1 LambdaRank、Validation-only selection、First7d post-selection diagnostic、Train+Validation final refit、Test prediction freeze、一次性 Test label join/evaluation、paired bootstrap、segment audit 与 final lock。

在没有新的正式 Prompt 时，不得提前实现：

- MF/BPR；
- DeepFM；
- DIN 或其他序列模型；
- 新的 challenger、额外 reranking 或 retrieval retuning；
- 在线服务。

候选中未购买的商品只能称为“未观察到购买”，不得称为真实负样本或真实不喜欢。
不得预设任何召回路有效；每一路效果必须由固定评估协议和分群结果验证。

不得使用 Test 选择：

- 候选配额；
- 时间窗口；
- 阈值；
- 衰减系数；
- 超参数；
- 模型版本。

不得为展示链路而增加无证据的独立粗排。

### Stage 3A 长期事实

- Trend 7d 是通过 Validation 预注册规则锁定的正式趋势版本；
- Repurchase 保留用户历史购买商品；
- 后续召回器必须复用 Stage 3A 的统一候选协议和评估器；
- Stage 3A Test lock 和正式指标不得覆盖；
- Stage 3A 结果不等于线上 CTR/CVR 或业务收益。

### Stage 3B 长期事实

- IUF 是通过 Validation 预注册规则锁定的 ItemCF 版本；
- 用户历史商品先去重，最近 50 个唯一商品参与共现；
- 每个商品保留 Top-200 邻居，每个用户使用最近 20 个 seed；
- ItemCF 排除 seen items，Repurchase 负责历史商品；
- ItemCF Union 保持固定候选总预算；
- ItemCF 已通过 marginal Recall、独占命中和 paired user bootstrap 验证存在离线新增价值；
- 固定预算替换仍会造成部分用户 Recall 下降；
- 深层 Recall 增益不代表最终 Top-12 排序已经优化；
- Stage 3B Test lock 和正式指标不得覆盖。

### Stage 3C 长期事实

- C3 是通过 Validation 锁定的最佳结构化 Content 版本；
- 固定预算 Q1/Q2 均弱于冻结的 IUF Q0，Content 未通过 value gate；
- Content Test 按协议未运行，状态为 `SKIPPED_BY_PROTOCOL`；
- Content 当前不进入正式多路 Union；
- canonical 属性审计、snapshot IDF、group-balanced 表示等结构化属性基础可以复用，但不能声称 Content 独立召回已经有效，也不能预设双塔一定提升；
- Stage 3C lock、manifest 和正式指标不得覆盖；
- Stage 3C 收尾时不预设后续双塔一定提升；后续 Stage 3D 的授权与通过不改变 Stage 3C 的 `FAILED_VALUE_GATE` 结论。

### Stage 3D 长期事实

- Stage 1C 原 MF-before-Two-Tower 路线经用户明确授权调整；
- MF/BPR 未实现，也不作为当前 V1 后续必做项；
- TT1 是 Stage 3D 锁定模型，T100 是锁定固定预算融合，总候选预算始终为 500；
- source-rank cap 在跨来源去重前应用；
- Two-Tower 排除 seen items；cold-history 不生成个性化 Two-Tower candidates；
- ANN `efSearch=200` 已通过 exact overlap audit；
- Stage 3D 通过固定预算 marginal value gate；
- Test 仅在 Validation lock 后执行，未用于选择模型、配额、epoch 或 ANN 参数；
- Two-Tower 当前主要提升 Recall@500，不能声称 MAP@12 已提升；
- Stage 3D 结果是离线结果，不能称为 CTR/CVR、收入、线上收益或线上 A/B Test；
- full-refit correction 已完成，正式 embedding、ANN 和 candidates 均来自 full-refit final checkpoint；
- Stage 3D lock、manifest 和正式指标不得覆盖；
- Stage 3E 已冻结最终召回合同，未经新的正式技术评审与 Prompt，不得重新调 Test retrieval、TT1、ANN 参数或融合配额；
- Ranking Time Protocol 已由 Stage 4A 冻结，historical replay 已由 Stage 4B-1 完成，Stage 4B-4 已完成 LightGBM LambdaRank 与一次性 Test evaluation；额外 reranking 和线上服务仍未实现，V1 不设置独立粗排。

### Stage 3E 长期事实

- Stage 3E 已通过 ChatGPT 最终技术验收，正式状态为 `PASSED_CONTRACT_GATE`；
- Final V1 retrieval 为 TT1-T100，固定总候选预算 500；nominal caps 为 Repurchase 50、IUF ItemCF 200、Two-Tower 100、Trend 7d 100，Popularity 只做唯一候选补齐；
- source-rank cap 在跨来源去重前应用；插入顺序为 Repurchase、ItemCF、Two-Tower、Trend、Popularity fallback；重复候选占用各自 pre-dedup source cap 并合并 provenance，不同来源 score 不跨源相加；
- 正式合同版本为 `stage_3e_candidate_contract_v1`，协议版本为 `source_rank_cap_pre_dedup_v1`；一行对应一个 snapshot/user/item，`candidate_rank` 确定且连续，并保留 source provenance、source-local rank/score、semantic fingerprint 与 replay lineage；
- 完整 Candidate 产物保持 private、Git ignored 且可由正式配置与上游产物重建；公开产物只保存合同、聚合、状态和 fingerprint；
- Stage 3E 对 Stage 3D 锁定结果保持 zero semantic drift，不是新的 Recall 或 MAP 提升；Stage 3D Two-Tower 的主要收益仍是 Recall@500，不代表 Top-12 排序已经优化；
- Stage 3C 保持 `FAILED_VALUE_GATE`，Stage 3D 保持 `PASSED_VALUE_GATE`，Stage 3E 保持 `PASSED_CONTRACT_GATE`；
- Stage 3E 完成后 V1 主召回路线原则上冻结。未经新的正式技术评审与 Prompt，不得重新打开 Content、MF/BPR、quota、TT1、ANN 或融合协议搜索空间，也不得为排序效果重新调 Test retrieval；
- Ranking Time Protocol 已由 Stage 4A 正式冻结，historical replay、ranking dataset、snapshot-safe features 与 Stage 4B-4 LambdaRank 全链路均已完成；没有新的独立正式 Prompt，不得重开任何已冻结选择、Test 访问或 reranking 搜索空间。

### Stage 4A 长期事实

- Stage 4A 已通过 ChatGPT 最终技术验收，正式状态为 `PASSED_PROTOCOL_GATE / FINAL ACCEPTED`；decision 状态为 `FROZEN_FOR_STAGE_4B_CONSUMPTION`。
- 主训练协议为 `single_historical_14d_snapshot_v1`：Ranking Train as-of 2020-08-11，Validation as-of 2020-08-25，Test as-of 2020-09-08；各自未来标签窗均为连续 14 天。
- 主 Ranking optimization population 为 `historical_evaluable_warm_users_v1`；cold-history 不进入主 optimization population 或主 evaluation average，all-zero relevance retrieval miss 不参与未来 LambdaRank optimization 但保留在端到端评估人口。
- Future training 保留全部 positives；`label=0` 仅表示未来标签窗未观察到购买，每 query 最多按 UTF-8 SHA-256 canonical payload 稳定保留 100 条；query key 为 `(snapshot_id, customer_key)`。
- Candidate membership 必须 label-agnostic；future positives 只能标注 frozen candidates，不能注入 candidate universe。
- `ranking_validation_first7d_diagnostic` 只允许对锁定的 14-day winner 与 Stage 3E baseline 做一次 post-selection diagnostic，不得成为第二选择集或触发自动 retuning/Test authorization。
- Historical replay 必须保持 Stage 3E retrieval protocol identity，但首次物化 `ranking_train_14d_v1` 时创建自己的 historical semantic fingerprint；只要求同一 historical snapshot 的 replay fingerprint 等于其 formal fingerprint，不得要求等于 Validation/Test fingerprint。
- 所有动态 Ranking features 和统计只使用对应 snapshot as-of 前历史；final refit 不得用晚期 history 重算旧 snapshot features。
- Test 只允许在 Validation lock、final refit、Test prediction fingerprint 冻结后执行一次 locked final evaluation，不得参与任何选择。
- Stage 4A 本身没有执行 historical full replay、ranking dataset/features、LightGBM、Test Ranking evaluation 或 reranking；historical full replay 后续已由 Stage 4B-1 完成，ranking dataset/labels 与 deterministic sampling 后续已由 Stage 4B-2 完成，snapshot-safe ranking features 后续已由 Stage 4B-3 完成，LambdaRank 至一次性 Test evaluation 后续已由 Stage 4B-4 完成。
- 新窗口必须先读根目录 `AGENTS.md`，再读 Stage 4B-4 config/manifest/locks/gates/reports、Stage 4B-3 config/manifest/gate/reproduction-audit/report、Stage 4B-2 config/manifest/gate/report、Stage 4B-1 config/manifest/gate/report、Stage 4A config/decision/report、Stage 3E contract 与上游 lock/manifest；不得仅凭聊天记录进入后续阶段。

### Stage 4B-1 长期事实

- Stage 4B-1 已通过 ChatGPT 最终技术验收，正式状态为 `PASSED_HISTORICAL_REPLAY_GATE / FINAL ACCEPTED`。
- `ranking_train_14d_v1` 使用 as-of 2020-08-11、request date 2020-08-12；future window 只用于 request-user scope，candidate membership/rank/provenance 保持 label-agnostic。
- Historical Repurchase、IUF ItemCF、TT1、HNSW ANN、Trend 7d、Popularity 与 T100 fixed-budget fusion 均从 cutoff 前历史重新构建，保持 Stage 3E frozen retrieval 与 500-candidate contract semantics。
- Historical TT1 使用 fresh initialization、全部 eligible historical samples、锁定 epoch=1、internal dev=0；formal checkpoint、embeddings、ANN、Two-Tower candidates 与 final union 具有完整 lineage。
- Formal 与不同合法 scheduling order 的 replay semantic fingerprints 一致，15 项 Historical Replay Contract Gate 全部通过。
- 完整模型、embedding、ANN、候选与中间产物保持 private、Git ignored；公开产物只保存安全聚合、状态、审计与 fingerprint。
- Stage 4B-2 与 Stage 4B-3 完成后，Stage 4B-4 已按独立正式 Prompt 完成 LightGBM、Validation selection、First7d、final refit 与一次性 Test Ranking evaluation；额外 reranking 仍未进入。

### Stage 4B-2 长期事实

- Stage 4B-2 已通过 ChatGPT 最终技术验收，正式状态为 `PASSED_RANKING_DATASET_CONTRACT_GATE / FINAL ACCEPTED`。
- 正式 snapshot 为 `ranking_train_14d_v1`：as-of `2020-08-11`，future label window 为 `2020-08-12` 至 `2020-08-25`。
- Future B-feedback purchase 标为 `label=1`；candidate 在 future window 未观察到购买标为 `label=0`。`label=0` 不等于 dislike、true negative 或 exposed-not-purchased。
- Stage 4B-1 frozen candidate membership/rank/provenance 必须在 future label join 前冻结；future positive 只能标注 frozen candidate，不得注入 candidate universe。
- 正式 populations 为 `ranking_train_all_label_users`、`historical_evaluable_warm_users_v1` 与 `cold_history_label_users`，main/cold 必须分开。`historical_evaluable_warm_users_v1` 不是 `user_segment == "warm"`，而是符合历史非空条件的 short-history、warm 与 active users。
- Main future-positive user 若 frozen500 candidate positive count 为 0，则标记为 retrieval miss / all-zero relevance group：保留在 full end-to-end evaluation，排除于 optimizer。
- Negative sampling 保留全部 positives；每 query 最多保留 100 条 `label=0`，fixed seed 为 `20260811`。稳定顺序使用 UTF-8 SHA-256 payload `{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}`，按 digest ASC、article key ASC；禁止 Python built-in `hash()`、partition-dependent sampling 或 random policy search。
- 后续 Validation/Test 正式 evaluation 必须使用 full500；sampled optimizer supervision 不得替代完整 evaluation candidate universe。
- Candidate projection 保持 zero semantic drift；formal/replay optimizer semantic fingerprint 相等，partition 或 scheduling order 不改变 sampling membership。
- Stage 4B-3 完成 Ranking Feature Pipeline 后，Stage 4B-4 已按独立正式 Prompt 完成 LightGBM、Validation selection、First7d、ranking final-refit 与一次性 Test Ranking evaluation；额外 reranking 尚未进入。

### Stage 4B-3 长期事实

- Stage 4B-3 已通过 ChatGPT 最终技术验收；Codex 工程门状态为 `PASSED_RANKING_FEATURE_CONTRACT_GATE`，最终验收状态为 `FINAL ACCEPTED`，G1–G20 全部 PASS。
- 正式 snapshot 为 `ranking_train_14d_v1`，as-of `2020-08-11`、request date `2020-08-12`；动态特征和统计只读取 cutoff 当日及以前的 B-feedback history，future label、Validation 与 Test reads 均为 0。
- F0 `core_temporal_retrieval_v1` 含 30 个 model features；F1 `core_plus_affinity_v1` 在 F0 上增加 22 个 canonical category code/user-category historical affinity features，共 52 个 model features。`primary_source` 仅供审计，不进入模型；不同 retrieval source 的 raw scores 保持 source-local，不跨源聚合。
- Article static attributes 缺少可靠 effective timestamp，只能按冻结 canonical identity 使用；price 因 lineage 不确定而省略。缺失 recency 为 null；语义上有效的 count absence 为 0；已知 category 但无匹配历史的 affinity 为 0，缺失 user/category 的 affinity 为 null。
- Canonical full500 feature view 覆盖 130001 queries、65000500 rows，每 query 500 rows；optimizer feature view 只对 Stage 4B-2 frozen optimizer keys 做 exact projection，覆盖 51497 queries、5236056 rows，不重新执行 negative sampling。Candidate membership/rank/provenance mismatch 均为 0。
- Frozen candidate projection fingerprint 为 `1448b2dd...b7dc0`；Stage 4B-2 full projection fingerprint 为 `aa04fce0...db343`；Stage 4B-2 optimizer projection fingerprint 为 `3e634e20...49fa8`。完整值以 Stage 4B-3 正式 manifest 与 audit 为准。
- Feature schema fingerprint 为 `5964882d...896a3`；formal 与 8/32 deterministic replay reconciled feature semantic fingerprint 均为 `f840fe3a...9bccc`，feature statistics fingerprint 为 `bd3b42de...eaf4e`；replay 覆盖 16360000 rows，feature value mismatch 为 0。
- Formal producer 完成后只发生 audit/recovery-only source correction。Producer source fingerprint 为 `ff856910...f7227`，current final source fingerprint 为 `59056645...2a9c7`；两者不同不表示 feature semantic drift。
- Current final source 使用相同 formal config/snapshot/as-of/upstream fingerprints，在独立 Git-ignored temporary 路径重建固定 partition 0 的 1997000 rows；row count/key、schema、candidate rank/provenance、全部 52 个 feature values、null/value、query digest 与 partition semantic fingerprint mismatch 均为 0。Formal 与 reproduction partition semantic digest 均为 `b24b1771...89058`，没有覆盖或改写 formal producer output。
- 完整 aggregates、full500/optimizer features、replay、current-source reproduction 与 checkpoints 保持 private/Git ignored，可由正式配置和上游产物重建；公开产物只保存安全聚合、状态、schema/semantic fingerprints 与 audit metadata。
- Stage 4B-3 targeted tests 为 35 passed，全量 pytest 为 334 passed；privacy/path scan、ignore audit、raw-data diff 与 `git diff --check` 均通过。
- Stage 4B-3 本身没有训练 Ranking model，也没有执行 Validation selection、First7d、ranking final-refit、Test Ranking、reranking 或 retrieval retuning；其中 LambdaRank 至一次性 Test evaluation 后续已由 Stage 4B-4 在独立正式授权下完成，reranking 与 retrieval retuning 仍未授权。

### Stage 4B-4 长期事实

- Stage 4B-4 已获用户最终验收，状态为 `STAGE_4B4_PHASE5B_FINAL_ACCEPTED / STAGE_4B4_FINAL_ACCEPTED`；Test value gate 为 `PASSED_FINAL_TEST_VALUE_GATE`，统计状态为 `POSITIVE_AND_SIGNIFICANT`，分群状态为 `NO_CRITICAL_SEGMENT_REGRESSION`。
- 冻结日历严格沿用 Stage 4A：Train as-of `2020-08-11`、labels `2020-08-12` 至 `2020-08-25`；Validation as-of `2020-08-25`、labels `2020-08-26` 至 `2020-09-08`；Test as-of `2020-09-08`、labels `2020-09-09` 至 `2020-09-22`。First7d 仅覆盖锁定 Validation label window 的前 7 天且只作 post-selection diagnostic。
- Validation 选择锁定 F1 `core_plus_affinity_v1` 的 52 个特征、LightGBM `lambdarank` objective、`MAP@12` 主指标和 600 棵树；final model version 为 `L1_FINAL_REFIT_600`，fresh initialization，Train+Validation optimizer supervision 合并训练，旧 snapshot 特征不得用晚期 history 重算。
- Test 主评估人口为 117366 位 `evaluable_warm_users`；cold-history 10967 位单独诊断，不进入主平均。主人口中 62383 个 candidate-positive all-zero query 保留为 0，约占 53.15%，排序模型不能恢复 frozen 500 candidates 之外的正例。
- Test B0 的 MAP@12/NDCG@12/Recall@12/HitRate@12 分别为 `0.014657 / 0.022618 / 0.033072 / 0.068214`；`L1_FINAL_REFIT_600` 分别为 `0.029620 / 0.046586 / 0.065511 / 0.146703`。完整精确值以 `artifacts/stage_4b4/test_evaluation_metrics.csv` 为准。
- Test 主人口 MAP@12 绝对增量为 `0.014963`，相对增量约 `102.08%`；1000 次 paired user-block percentile bootstrap 95% CI 为 `[0.014437, 0.015461]`。完整精确值以 `artifacts/stage_4b4/test_evaluation_bootstrap.csv` 为准。
- Cold-history MAP@12 增量约为 `0.000172`，95% CI 约为 `[-0.000528, 0.000788]`，只能称为 inconclusive 且未发现显著退化；cold-item/zero-history 目标切片仍为 0。
- 防泄漏顺序已冻结为 Validation selection → winner lock → First7d diagnostic → final refit → Test prediction fingerprint freeze → 唯一一次 Test label access/evaluation。Test 不得参与特征、模型、树数、候选、阈值或任何超参数选择。
- Test access audit 冻结为 raw label read、ground-truth build、prediction-label join、formal evaluation、metric materialization 各 1 次，Validation/First7d label reads 为 0，`test_labels_reopened=false`。今后不得重开 Test labels、重算正式指标/bootstrap 或创建第二次正式 Test evaluation。
- 当前 Phase 4 final-refit lock 为 `358c25d1...f4bd8`；旧 Phase 4 lock `de1072d4...2a3d6` 已被恢复流程明确 supersede。Phase 5A pre-label lock 为 `34c8dcc5...7e8ca`；Phase 5B final lock 为 `daddf4c1...b1b88`。完整值分别以 `artifacts/stage_4b4/stage_4b4_final_refit_lock.json`、`artifacts/stage_4b4/stage_4b4_test_pre_label_lock.json` 与 `artifacts/stage_4b4/stage_4b4_test_evaluation_lock.json` 为准。
- 受保护的 final-refit model、prediction fingerprints、ground-truth manifest、metrics、bootstrap、segment diagnostics、access audit、contract gate、locks 与 final report 不得覆盖或改写。完整模型、预测、ground truth、per-query metrics、checkpoints 与中间产物保持 private/Git ignored。
- Article static attributes 仍缺少可靠 effective timestamp；price 因 lineage 不确定而省略；数据仍缺少曝光、点击、真实负反馈、库存/可售状态，因此离线 Test 结果不等于 CTR/CVR、收入、线上收益或线上 A/B Test。
- 下一阶段尚未开始。未经新的独立正式 Prompt，不得重训/替换 winner、修改 frozen predictions、使用 Test 调参、重开 retrieval/feature semantics、追加 challenger/reranking 或进入线上服务。

上述阈值是当前项目工程规则，不是行业统一标准。

## 9. 工程约定

目标开发环境是 Windows 与 PowerShell。
始终优先使用项目 `.venv`，不要误用系统 Python。
Python 版本与依赖边界以 `pyproject.toml` 为准。
代码路径使用 `pathlib.Path` 和项目相对路径。
不得把本地绝对路径写入代码、配置、日志、报告或公开产物。
大规模 Parquet 扫描和聚合使用 DuckDB/PyArrow；pandas 只处理能够证明为小型的聚合结果。
不得把完整 transactions 载入 pandas。

大任务必须具备：

- checkpoint；
- 输入 fingerprint；
- 明确内存限制；
- 临时目录与磁盘限制；
- 查询或任务耗时记录；
- 资源监控；
- 单组失败恢复；
- 可重复运行语义。

单项失败不得导致已完成且 fingerprint 一致的结果丢失。
不得隐藏 OOM、spill、异常、重试或未验证项。
关键业务逻辑使用中文注释解释“为什么”，避免仅翻译代码表面行为。
正式可复用逻辑放在 `src/`，执行入口放在 `scripts/`，测试放在 `tests/`。
单元测试不得依赖完整原始数据，除非明确标记为 integration test。
不得创建后续模块的空壳文件来假装进度。

## 10. 每个任务的强制工作流

### 开始前

1. 读取根目录 `AGENTS.md`。
2. 读取当前阶段相关的正式配置、manifest、决策文档和阶段报告。
3. 检查当前分支与 `git status`。
4. 检查上阶段是否已提交以及本地与远端的同步状态。
5. 检查输入 fingerprint 是否与正式 manifest 一致。
6. 检查任务需要的私有上游产物是否存在且未被跟踪。
7. 检查项目 `.venv` 和实际 Python 解释器。
8. 确认任务授权范围与明确禁止项。

工作区不干净时，不得把不同阶段的修改混在一起。
已有有效修改应先识别归属并保留，不得擅自撤销用户或其他任务的工作。

### 实现中

- 严守当前任务边界；
- 不擅自进入下一阶段；
- 原始数据只读；
- 不自动清洗未知语义数据；
- 不修改冻结协议以迁就代码；
- 不伪造指标、产物、提交状态或测试结果；
- 不把局部 smoke 结论表述为全量结论；
- 不隐藏失败、OOM、spill 或重试；
- 优先复用现有配置、基础层和统一评估对象；
- 只重跑失败、不完整或 fingerprint 不一致的任务组。

### 完成后

必须汇报：

1. 当前完成情况；
2. 实际运行范围；
3. 指标与结果；
4. 资源使用；
5. 测试结果；
6. 新增和修改文件；
7. 数据保护和防泄漏验证；
8. 失败与未完成项；
9. 是否建议进入下一阶段；
10. Git 提交拆分建议。

没有产生指标或资源测量时，应明确写“不适用”，不得补造。
完成后不得自动 commit、push 或进入下一阶段。

### 阶段最终收尾

每个阶段必须按以下顺序关闭：

1. Codex 完成代码、实验、测试、产物和阶段报告；
2. ChatGPT 完成技术验收；
3. Codex 完成验收要求的全部修复；
4. ChatGPT 确认最终验收通过；
5. Codex 同步更新 `README.md` 当前项目进度，以及根目录 `AGENTS.md` 的权威索引、已完成阶段、高层冻结事实、当前未实现能力和下一阶段未授权状态；
6. 运行全量测试、`git diff --check`、敏感信息扫描、ignore 和 fingerprint 检查；
7. 用户明确批准后，才允许 commit 和 push；
8. commit/push 完成并确认本地与远端同步后，才能开始下一阶段。

阶段尚未通过 ChatGPT 最终验收时，不得在 `AGENTS.md` 标记为完成。
`README.md` 与 `AGENTS.md` 必须在同一阶段收尾中同步。
`AGENTS.md` 只写长期有效的高层事实，详细指标保留在阶段报告和公开产物中。
不得等到下一阶段开始后再补更新上一阶段状态。

## 11. Git 规范

每个阶段应按工程代码与测试、可公开的小型聚合产物、决策文档与阶段报告合理拆分。

Commit message 使用约定式前缀，例如：

- `feat:` 新增阶段能力；
- `data:` 新增可公开聚合产物；
- `docs:` 更新协议或报告；
- `chore:` 环境与仓库维护；
- `fix:` 修复已确认缺陷。

raw、private、tmp、模型文件和大型中间产物不得提交。

提交前至少运行：

```powershell
git diff --check
.\.venv\Scripts\python -m pytest
git status --short
```

还必须执行敏感信息扫描、ignore 验证、本次差异范围核对和必要的 fingerprint 一致性验证。
不得修改已经冻结的历史报告来伪装结果或提升指标。
未经用户明确要求，不得创建 commit 或 push。

## 12. 事实来源优先级

发生事实差异时，按以下优先级处理：

1. 当前仓库中的正式配置、manifest 和聚合产物；
2. `docs/architecture/` 与 `docs/decisions/`；
3. 已提交的阶段报告；
4. 当前根目录 `README.md`；
5. 聊天历史仅作补充。

Stage 3A–4A 的配置、lock/decision、manifest 和公开聚合产物高于 README。
README 与更高优先级正式产物冲突时，应在用户明确裁定后修正。
新窗口必须读取最近已完成阶段的配置、lock、manifest 和报告；Codex 聊天记录丢失不影响项目接管，不得依赖旧聊天记录恢复正式事实。

当前用户明确提供的新阶段 Prompt 用于界定本次授权范围，但不能静默改写仓库中已经冻结的事实。
如果新指令与冻结协议冲突，必须指出具体冲突并等待确认。
不得用聊天记忆覆盖正式配置、manifest 或冻结决策。
新窗口应从仓库事实重建上下文，不依赖旧聊天记录判断进度、指标或协议。
