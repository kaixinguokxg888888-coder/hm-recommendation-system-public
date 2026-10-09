# Stage 4B-4 Phase 5A：Test Candidate / Feature / Prediction Freeze

## 1. 阶段定位与状态

Phase 5A 是一次性 Test evaluation 的无标签阶段。当前正式状态：

```text
STAGE_4B4_PHASE5A_TEST_PREDICTIONS_FROZEN
```

本阶段只冻结 Test candidates、snapshot-safe F1 features、B0 与 final-refit model ranked lists，并创建 Test pre-label lock。Test labels 未读取，Test metrics 未计算，Test evaluation 未执行；Phase 5B 未授权。

## 2. 上游冻结身份

- 当前唯一有效 Phase 4 lock：`358c25d19ecce3006428a293acd61564e0f6cd3fbdc0c156aa636ee41bef4bd8`。
- `de1072d41e921c8cfe39275486a7a7f201dc9ffa26196b9b6fec8a6a6a52a3d6` 是已 supersede 的旧 lock，本阶段未使用。
- Final model：`L1_FINAL_REFIT_600`，F1，LambdaRank，600 rounds / 600 trees，fresh initialization；未重训、未 warm start、未 early stop。
- Model physical / semantic fingerprints：`1f2af19a79ed8d085d68d20fe9437f4eca7da3534a3852eb823f483c6bf96922` / `0adc28c0497db8e300fe7c4ec487b0b34fd115b8a9cc5e978763d4a8ef2c675e`。

## 3. Test 时间边界与访问顺序

```text
snapshot_id  = test_14d
as_of_date   = 2020-09-08
request_date = 2020-09-09
```

动态特征只读取 `history_date <= 2020-09-08`，实际最大动态历史日期为 `2020-09-08`。执行顺序严格为只读 preflight、pre-access audit、正式 config、preregistration，之后才首次打开 Test candidate/feature content。

Preregistration fingerprint 为 `da210aef75b6c39d04c7bcdff7f538f0f69f857485bc11d1ac1c3064412ff5e5`。创建 preregistration 时 Test candidate、feature、label、prediction write 与 metric read counts 全部为 0。

## 4. Test candidate population 与 full500

Test candidates 精确复用 Stage 3E frozen `stage_3e_candidate_contract_v1 / source_rank_cap_pre_dedup_v1`：

| Check | Result |
|---|---:|
| Queries | 128,333 |
| Candidate rows | 64,166,500 |
| Rows/query | 500 |
| Duplicate row keys | 0 |
| Duplicate candidate ranks | 0 |
| Full500 violation queries | 0 |
| Provenance violations | 0 |
| Candidate semantic fingerprint | `fd94f546add97637e15652c800bc636f3c7723bc4ae7d9cb055f82d5e83e9948` |

没有过滤 cold-history、retrieval-miss 或潜在 all-zero relevance query；Phase 5A 不读取 label，因此也未按 relevance 改变 population。Retrieval、quota、dedup、backfill、source priority 与 `candidate_rank` 均未修改。

必须使用 full500，而不是 optimizer sampled view：sampled view 只服务训练计算；Test 的排序对象是完整 candidate set；抽样会改变 query 内竞争关系；sampled rank 不能代表部署时 full500 rank；B0 与 learned ranker 必须在同一 population 冻结；后续 Phase 5B 只能消费本阶段冻结的完整 ranked lists。

## 5. Test F1 feature identity

- Feature set：F1 `core_plus_affinity_v1`，52 个 model-input features。
- Feature schema fingerprint：`5964882d45d0c3453fb49b3dac39fb187969989d364a63e7d4c18b3ed49896a3`。
- Adapter fingerprint：`aed9098285baf2d4480f51b71073b7a53073d31f0ea3a87f8ad870b7297236ed`。
- Feature semantic fingerprint：`179abf06003e880ddf7ecd00d1d98c47282a4a4e078f5516695ca860720479f1`。
- Feature statistics fingerprint：`daba710e957c1ff38a8d44036927d484e46a0b38a35832fdea8db4ade2f9a73b`。
- Feature lineage fingerprint：`0dd955e3d1be653e5e3001b113385dbf2b6942e3b0bf5f9148163aae7cbf3b4c`。

`primary_source` 只用于 audit；不同 retrieval source raw scores 保持 source-local；price 继续 `OMITTED_BY_LINEAGE_UNCERTAINTY`。Recency 缺失保持 NULL，合法 count absence 为 0，已知 category 但无历史匹配的 affinity 为 0，缺失 user/category 的 affinity 为 NULL。Article static attributes 仍保留 effective timestamp 无法验证的限制。

Partition 0 deterministic replay 的 row key、52-feature values、NULL bitmap、feature order 与 semantic fingerprint mismatch 均为 0。全量 non-finite feature count 为 0。

## 6. Prediction freeze

Phase 5A 只生成两个版本：

| Version | Rows | Queries | Global prediction fingerprint | Status |
|---|---:|---:|---|---|
| B0 | 64,166,500 | 128,333 | `b7a1f9b6268923dc4c27441b6ce559e90e339673a4bd5001402e91e05ddc75a9` | FROZEN |
| L1_FINAL_REFIT_600 | 64,166,500 | 128,333 | `5080c3cc2149aa7bebc3cbd3261f93e5162aefbf426a3d987fcda05df8ee9a0e` | FROZEN |

B0 严格由 frozen `candidate_rank` 决定。Learned ranker 精确加载 Phase 4 final model，以冻结 feature order 输入 F1 并使用全部 600 trees；tie-break 为 `raw_score DESC, candidate_rank ASC, article_key ASC`。两版 duplicate、candidate-rank、predicted-rank、non-finite score 与 tie-break violation 均为 0。

公开 manifest 只保存 row/query counts、physical/semantic/query/global/binding fingerprints 和状态。Raw scores、ranked rows 与 query digests 保存在 Git-ignored private artifacts，不公开 entity-level predictions。

## 7. Prediction replay

Partition 0 在正式预测 worker 退出后由 fresh process replay。Final model 被重新加载，未读取 labels；B0 重新按 baseline 规则生成。两版 row key、raw score、predicted rank 与 query digest mismatch 全部为 0，replay 为 2/2 PASS。

## 8. 资源、checkpoint 与恢复

Preregistration 时 available memory 只读取一次，冻结 absolute RSS cap 为 `1,946,567,884` bytes。运行期不重新计算 cap，整个 worker process tree 由外部父进程监控。

| Worker | Peak process-tree RSS | Frozen cap | OOM | Status |
|---|---:|---:|---:|---|
| Feature materialization | 1,252,143,104 | 1,946,567,884 | 0 | PASS |
| Formal predictions | 653,729,792 | 1,946,567,884 | 0 | PASS |
| Prediction replay | 405,598,208 | 1,946,567,884 | 0 | PASS |

Feature producer 建立 165 个成功 component checkpoints；formal prediction 建立 32 个成功 partition checkpoints；prediction replay 建立 1 个成功 checkpoint。每个正式长任务按 query-contained partition 执行，输出先写 temporary path，完整 Parquet 校验后 atomic promotion，随后写 output digest 与 SUCCESS checkpoint。没有 incomplete partition 或失败 checkpoint 晋升为正式产物。

锁前发现 finalizer 的 P5AG18 证据文字仍标为“待工程审计”。该纯工程顺序问题在锁创建前修复并公开记录；未改变 preregistered semantics、candidate/feature/model/tie-break/partition/fingerprint algorithm，未重算成功预测，也未读取 labels。原 preregistered source 与 corrected source 均被保留和绑定。

## 9. Access audit

| Object | Content read count | Authorized |
|---|---:|---:|
| Test candidates | 1 | yes，仅用于 candidate/feature freeze |
| Test features | 1 | yes，仅用于 audit/prediction |
| Test labels | 0 | no |
| Test metrics | 0 | no |
| Validation labels | 0 | no reopen |
| First7d labels | 0 | no reopen |

没有 Test label hash、schema inspection、join、aggregate 或 metric computation。

## 10. Contract Gate 与 pre-label lock

`P5AG1–P5AG20` 全部 PASS，覆盖 current Phase 4 lock、Stage 4A calendar、Stage 3E candidate/population、pre-access order、query/full500、rank/provenance、cutoff、F1/adapter、final model、两版 predictions、tie-break/fingerprints/replay、resource/checkpoint/privacy/tests、zero Test-label/metric access 与 lock self-fingerprint。

正式 Test pre-label lock fingerprint：

```text
34c8dcc52ee18c18766904d234e8fda0f2964e4805fee3175e9b45c375f7e8ca
```

Lock 使用 canonical JSON self-fingerprint 并独立验证，且明确记录：candidates/features/predictions frozen，Test label read count=0，labels 未 join，metrics/evaluation 未执行，Test 未授权，winner 未改变，model 未重训，Phase 5B 等待独立 ChatGPT 授权。

## 11. Tests 与工程审计

```text
Phase 5A targeted: 8 passed, 0 skipped
Stage 4B-4 targeted: 120 passed, 1 skipped
Full pytest: 454 passed, 1 skipped
Python compile: PASS
git diff --check: PASS
raw-data working/staged diff: 0 / 0
privacy/path violations: 0
tracked private artifacts: 0
background workers after execution: 0
```

唯一 skip 是 Phase 4 已 supersede 的旧 P4 gate 可选测试，分类为 `EXPECTED_OBSOLETE_ARTIFACT_SKIP`；它不是 Phase 5A required test。Phase 5A required tests 0 skip。

## 12. 边界与停止状态

本阶段没有报告任何 Test MAP@12、NDCG@12、Recall@12、HitRate@12 或分群效果，不构成 CTR/CVR、收入、线上收益或 A/B Test 证据。数据仍缺少曝光、点击、真实负反馈、库存/可售状态及可靠 article effective timestamp。

README 与 AGENTS 未更新；commit/push 未执行。Phase 5A 在 pre-label lock 后停止，不进入 Phase 5B。任何 Test label join 与一次性正式 evaluation 都仍需新的独立 ChatGPT 授权。
