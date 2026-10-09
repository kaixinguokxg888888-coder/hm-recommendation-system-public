# Stage 4A：Ranking Time Protocol Freeze

## 1. 阶段状态

Stage 4A 当前状态为 `PASSED_PROTOCOL_GATE / FINAL ACCEPTED`。ChatGPT technical review = `PASS`。

Decision 状态为 `FROZEN_FOR_STAGE_4B_CONSUMPTION`；`final_acceptance_status=ACCEPTED`。未来独立正式 Prompt 可以消费本协议进入 Stage 4B，但本 closeout 未进入 Stage 4B。

本阶段冻结协议和静态审计，没有执行 historical full candidate replay，没有生成 ranking dataset/features，没有训练、选择或评分 Ranking 模型，也没有进入 Stage 4B。

## 2. 上游正式事实

- Stage 3C：`FAILED_VALUE_GATE`；Content 不进入正式 Union。
- Stage 3D：`PASSED_VALUE_GATE`，Test `SUCCESS`；锁定 TT1、T100、epoch=1 full-refit 与 ANN `efSearch=200`。
- Stage 3E：`PASSED_CONTRACT_GATE`；合同 `stage_3e_candidate_contract_v1`、协议 `source_rank_cap_pre_dedup_v1`，Validation/Test replay 与 zero-semantic-drift 均通过。
- Final Retrieval 固定 500：Repurchase 50、IUF ItemCF 200、TT1 100、Trend 7d 100、Popularity 唯一补齐。

`upstream_fingerprint_manifest.csv` 对正式协议、config、lock、manifest 和 gate 做实际 SHA-256 复核，全部匹配。

## 3. 冻结 Ranking calendar

| Snapshot | as-of | request | history cutoff | label start | label end |
| --- | --- | --- | --- | --- | --- |
| `ranking_train_14d_v1` | 2020-08-11 | 2020-08-12 | 2020-08-11 | 2020-08-12 | 2020-08-25 |
| `validation_14d` | 2020-08-25 | 2020-08-26 | 2020-08-25 | 2020-08-26 | 2020-09-08 |
| `test_14d` | 2020-09-08 | 2020-09-09 | 2020-09-08 | 2020-09-09 | 2020-09-22 |

主训练协议为 `single_historical_14d_snapshot_v1`；rolling historical snapshots 是 P1，未实现。`ranking_validation_first7d_diagnostic` 复用 Validation request/candidate/features，只使用 2020-08-26 至 2020-09-01 ground truth，不是独立 request snapshot。

First7d 必须在完整 14-day Validation 选出并锁定唯一 winner 后，仅对该 winner 与 Stage 3E baseline 执行一次。它不能成为 secondary selection set，不能触发 retuning。若 14-day delta MAP@12 为正而 first7d 为负，状态记为 `VALIDATION_ROBUSTNESS_NOT_MET`，不自动授权 Test并返回技术评审。

## 4. Replay capability

Repurchase、Trend 与 Popularity 的核心函数已直接按 `as_of_date` 过滤。ItemCF、TT1 training/full-refit、ANN 与 Fusion 的核心 cutoff 或参数化能力存在，但正式 runner/lineage 当前只覆盖 Validation/Test，因此记录为 `NEEDS_STAGE4B_ADAPTATION`。未发现必须改变 Stage 3E frozen semantics 才能 replay 的 blocker。

Stage 4B replay 必须从 2020-08-11 历史重建各来源；不得复制晚期 candidate/checkpoint/statistic，不得重开模型、epoch、ANN 或 quota 选择。

跨 snapshot 只要求 `historical_retrieval_protocol_identity`，不要求 historical membership/rank fingerprint 等于 Validation/Test。Stage 4B 第一次正式物化必须创建 `ranking_train_14d_v1` 自己的新 formal semantic fingerprint；之后同 snapshot、不同合法调度顺序的 replay fingerprint 才必须与该 historical formal fingerprint 相等。本阶段未执行这些 replay。

## 5. Label、sampling 与 group

- Ranking Train all-label users：2020-08-12 至 2020-08-25 至少一个 B-feedback positive。
- Main optimization population：`historical_evaluable_warm_users_v1`，要求 future positive、2020-08-11 前至少一个 B-history interaction，以及有效 internal keys。
- Cold-history：不进入主 optimization population或主 evaluation average，独立审计/评估。
- `label=1`：candidate item 在相应未来 14 天 B-feedback 中至少出现一次。
- `label=0`：candidate item 在该未来标签窗口内未观察到购买，不是真实负反馈。
- Candidate miss：future positive 不在 frozen candidate 中时记 retrieval miss，不注入。
- Query key：`snapshot_id + customer_key`。
- Training sampling：保留全部 positives；未观察购买 rows 按 UTF-8 SHA-256 的 `{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}` payload 计算 digest，再按 digest/article key 升序稳定保留最多 100。Python built-in hash、path、partition、row order、process id 和 runtime hash seed 禁止参与。
- All-zero relevance group：不进入未来 LambdaRank optimization，但保留在端到端排序评估人口。
- Validation/Test：始终评分完整 500 candidates，不做 negative downsampling。

## 6. Feature 与 final-refit 时间语义

所有动态特征和统计满足 `feature_timestamp <= snapshot_as_of_date`。Final refit 可合并 Ranking Train 与 Validation supervision，但旧 rows 分别保留 2020-08-11 和 2020-08-25 features，不得用晚期 history 回填。Static article attribute 缺少 effective timestamp，允许使用 canonical attributes 但不能声称已严格证明历史可用性。

## 7. Leakage audit

`leakage_audit.csv` 的 L1–L10 全部 `PASS`：Ranking Train 包含于 outer Train；每个 cutoff 早于 label；Train/Validation/Test 边界不重叠；Test 不可用于选择；first7d 完全包含于 Validation 且不接触 Test；candidate/label 分离；feature cutoff、historical replay 与 retrieval semantic freeze 均成立。

Targeted correction 新增 L11–L15，分别验证 Ranking population、deterministic negative hash、first7d post-selection-only、historical fingerprint semantics，以及 Stage 4B 在 Stage 4A 期间未被执行，全部 `PASS`。

## 8. Protocol gate

`protocol_gate.csv` 的 G1–G19 全部 `PASS`，包括关键的 `G13_NO_RANKING_IMPLEMENTATION`，以及新增的 G15 Ranking population、G16 deterministic negative hash、G17 first7d post-selection-only、G18 historical fingerprint semantics、`G19_STAGE4B_NOT_ENTERED_DURING_STAGE4A`。最终 Stage 4A 状态为 `PASSED_PROTOCOL_GATE / FINAL ACCEPTED`。

## 9. 资源、隐私与限制

本阶段只执行小型 config/calendar/fingerprint/schema/static-code audit。资源实测见 `resource_summary.csv`；没有正式大规模 replay、模型运行、DuckDB spill 或 ranking artifact。公开隐私扫描未发现实体值、本地绝对路径或本地用户名；raw 未修改，既有 private/tmp/checkpoints ignore 状态未改变。

Stage 4A 首轮实现前全量测试为 207 passed，首轮完成后为 239 passed。本次 targeted correction 后 Stage 4A 定向测试为 44 passed，最终全量测试为 251 passed、0 failed。测试不依赖完整原始数据。

数据仍缺少曝光、点击、真实负反馈、库存、精确上架/属性生效时间。离线结果不能表述为 CTR/CVR、业务收益或线上 A/B Test。

## 10. 明确未实现

- Historical full candidate replay：`NOT IMPLEMENTED / NOT EXECUTED`
- Ranking dataset：`NOT IMPLEMENTED`
- Ranking features：`NOT IMPLEMENTED`
- LightGBM Ranking：`NOT IMPLEMENTED`
- Ranking model selection：`NOT EXECUTED`
- Test ranking evaluation：`NOT EXECUTED`
- Reranking：`NOT IMPLEMENTED`
- Stage 4B：`NOT ENTERED`

STOP. Do not enter Stage 4B.
