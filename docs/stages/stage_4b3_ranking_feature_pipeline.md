# Stage 4B-3 — Ranking Feature Pipeline / Snapshot-safe Feature Materialization

## 1. 阶段结论

Stage 4B-3 已完成 `ranking_train_14d_v1` 的 snapshot-safe Ranking Feature Pipeline。Codex 工程门状态与 ChatGPT 最终技术验收裁决分别为：

```text
Codex Contract Gate: PASSED_RANKING_FEATURE_CONTRACT_GATE
ChatGPT Final Acceptance: STAGE_4B3_FINAL_ACCEPTANCE / FINAL ACCEPTED
```

20 项 mandatory Contract Gate 全部 PASS，且 ChatGPT 已完成最终技术验收。`PASSED_RANKING_FEATURE_CONTRACT_GATE` 表示 Codex 产出的 feature contract 工程门结果；`FINAL ACCEPTED` 是独立的最终验收裁决。最终收尾只同步阶段报告、`README.md` 与 `AGENTS.md` 并创建用户授权的本地 commit，不 push，也不进入任何排序模型训练。

正式 snapshot 固定为：

```text
snapshot_id  = ranking_train_14d_v1
as_of_date   = 2020-08-11
request_date = 2020-08-12
```

动态特征只消费 `t_dat <= 2020-08-11` 的 B-feedback history。最大实际历史日期为 `2020-08-11`，future label、Validation 和 Test 读取均为 0。

## 2. 中断恢复

网络中断发生时，磁盘上的 producer 已经完成，公开 final audit、Contract Gate 和阶段报告尚未完成。恢复接管先检查了 worktree、Python 进程、文件时间、181 个 checkpoint、输出 row count、schema 和物理 fingerprint，没有删除或覆盖已有结果。

恢复时未发现仍在运行的旧 Python writer；没有 partial/incomplete artifact、Parquet 损坏或并发写同一 partition。完整 producer 恢复矩阵见 `artifacts/stage_4b3/checkpoint_recovery_manifest.csv`。

| Task group | Previous status | Checkpoint identity | Action | Final status |
|---|---:|---:|---:|---:|
| deterministic smoke | 1/1 SUCCESS | PASS | REUSE | SUCCESS |
| history partitioning | 1/1 SUCCESS | PASS | REUSE | SUCCESS |
| global user/item/static aggregates | 3/3 SUCCESS | PASS | REUSE | SUCCESS |
| user-item aggregates | 32/32 SUCCESS | PASS | REUSE | SUCCESS |
| user-category aggregates | 32/32 SUCCESS | PASS | REUSE | SUCCESS |
| full500 feature partitions | 32/32 SUCCESS | PASS | REUSE | SUCCESS |
| optimizer feature partitions | 32/32 SUCCESS | PASS | REUSE | SUCCESS |
| formal query digests | 32/32 SUCCESS | PASS | REUSE | SUCCESS |
| replay feature partitions | 8/8 SUCCESS | PASS | REUSE | SUCCESS |
| replay query digests | 8/8 SUCCESS | PASS | REUSE | SUCCESS |
| final read-only audit | interrupted/incomplete | N/A | RERUN | SUCCESS |
| public contract finalization | missing | N/A | COMPLETE | SUCCESS |

全部 181 个 producer checkpoint 均逐项验证：snapshot、as-of、config fingerprint、upstream fingerprint、schema fingerprint、partition id、input/output physical fingerprint 和 Parquet row count。恢复后重算 feature partition 数为 0，重复 producer 执行为 0。

Producer 完成后曾只修改 final audit/recovery 代码，因此 manifest 同时保留 producer source fingerprint 与 current audit source fingerprint，并显式标记 `audit_only_source_correction=true`。这些修订没有改变 config、schema、feature SQL、row universe、feature values 或任何 private producer output。

第一轮技术验收要求补充“current final source 可复现 formal producer output”的执行证据。本次窄范围 correction 使用完全相同的 formal config、`ranking_train_14d_v1` snapshot、`2020-08-11` cutoff 和上游 fingerprints，选择既有 8/32 replay coverage 内的固定 query-contained partition `0`。Current source 从相同 cutoff history 独立重建全局 user/item/static aggregates、partition-local history/user-item/user-category aggregates、full feature partition 和 query digests；所有输出只写入 Git-ignored 的 `tmp/current_source_reproduction`，没有覆盖或改写任何 formal producer output。

| Current-source reproduction check | Result |
|---|---:|
| Formal / reproduction rows | 1,997,000 / 1,997,000 |
| Row count mismatch | 0 |
| Row key mismatch | 0 |
| Physical schema mismatch | 0 |
| Candidate rank mismatch | 0 |
| Candidate provenance mismatch | 0 |
| All 52 model-feature value mismatch | 0 |
| Null/value mismatch | 0 |
| Query digest mismatch | 0 |
| Partition semantic fingerprint mismatch | 0 |

Producer source fingerprint 为 `ff856910225074650a6c5bc418387c2d63a037fd9e9095762b5e02dadeff7227`，current final source fingerprint 为 `590566454957fcdb346701021f9582dc241391be0ab177ab038474e8ca32a9c7`。Formal 与 current-source reproduction partition semantic fingerprints 均为 `b24b17711a984d35f51f6bf4b13213a96ad671f276efbc44dfd02852d0c89058`。安全公开证据见 `artifacts/stage_4b3/current_source_reproduction_audit.csv`。

## 3. Git 与环境 Preflight

```text
branch: main
HEAD: dd628cf feat: complete stage 4b2 ranking dataset
main...origin/main: 0 / 0
initial worktree: clean
recovery worktree: only Stage 4B-3 changes
Python: 3.10.11 (.venv)
DuckDB: 1.5.5
PyArrow: 24.0.0
pytest: 8.4.2
initial full pytest: 299 passed
```

上阶段已经提交，本地与远端同步。恢复工作区内没有非 Stage 4B-3 混杂修改；恢复和验收修正期间未执行 reset、restore、clean 或 push。ChatGPT 最终验收后仅授权 Stage 4B-3 本地 closeout commit，仍明确禁止 push。

## 4. 上游冻结身份

Stage 3E、Stage 4A、Stage 4B-1 与 Stage 4B-2 的 gate、calendar、protocol、schema 和关键 projection 均通过 fail-fast 验证；各阶段 fingerprint mismatch 数均为 0。

| Projection | Formal fingerprint | Status |
|---|---|---|
| Historical candidate | `1448b2dd20ffcc3afceaa35143eec6f86f2ec5900a7fb24a3ebfb28ade8b7dc0` | PASS |
| Stage 4B-2 full labeled | `aa04fce0514d8fe77b73d3600c31bfba76b3b0e21f347077fe7395fa354db343` | PASS |
| Stage 4B-2 optimizer | `3e634e20aafbf6737fbd4160b4eadd80a156e020b137462cdef616c26bf49fa8` | PASS |

Optimizer feature view 只做 frozen Stage 4B-2 optimizer keys 与 canonical full feature dataset 的 inner join，没有重新执行 negative sampling。

## 5. Feature contract

正式 model-input feature 共 52 个。实体 key、snapshot metadata、lineage 字段和 `primary_source` 不进入模型。

### 5.1 F0 — `core_temporal_retrieval_v1`（30 features）

User（6）：

```text
user_history_interaction_count
user_history_distinct_item_count
user_recency_days
user_recent_interaction_count_28d
user_repeat_ratio
user_segment
```

Item（4）：

```text
item_history_interaction_count
item_history_distinct_buyer_count
item_recent_interaction_count_7d
item_recency_days
```

User-item（3）：

```text
ui_history_interaction_count
ui_seen_before
ui_recency_days
```

Retrieval（17）：

```text
candidate_rank
source_count
from_repurchase, from_itemcf, from_two_tower, from_trend, from_popularity
repurchase_rank, repurchase_score
itemcf_rank, itemcf_score
two_tower_rank, two_tower_score
trend_rank, trend_score
popularity_rank, popularity_score
```

各来源 raw score 保持 source-local，没有跨来源求和、求平均或全局归一化。`primary_source` 仅供审计。

### 5.2 F1 — `core_plus_affinity_v1`（F0 + 22 features）

F1 复用 Stage 3C 正式 canonical 11 字段，为每个字段增加一个静态 article category code 和一个动态 user-category historical interaction share：

```text
product_type, product_group, garment_group, section,
graphical_appearance, colour_group,
perceived_colour_value, perceived_colour_master,
department, index, index_group
```

静态编码固定为 `0=MISSING`、`1=UNKNOWN`、`2+=known value ASC`。User-category share 只由 cutoff 前 history 计算；cold user 或 candidate category 缺失时为 null，已知 category 但无匹配 history 时为 0。

Article static attributes 缺少可靠 effective timestamp，只能按冻结 canonical identity 使用，不能声称真实历史属性生效时间已验证。Price feature 状态保持 `OMITTED_BY_LINEAGE_UNCERTAINTY`：B daily feedback 没有冻结的同日 price 聚合语义。

完整 dtype、nullable、zero/null/default semantics、lineage、cutoff 与 model eligibility 见 `artifacts/stage_4b3/feature_schema_manifest.csv`；全量统计见 `feature_statistics.csv`。

## 6. Full500 与 optimizer 结果

| Check | Result |
|---|---:|
| Full queries | 130,001 |
| Full rows | 65,000,500 |
| Rows/query min / mean / max | 500 / 500.0 / 500 |
| Duplicate row keys | 0 |
| Duplicate ranks | 0 |
| Full500 violation queries | 0 |
| Candidate missing / extra | 0 / 0 |
| Candidate rank mismatch | 0 |
| Candidate provenance mismatch | 0 |
| Population control mismatch | 0 |
| Optimizer queries | 51,497 |
| Optimizer rows | 5,236,056 |
| Optimizer missing / extra | 0 / 0 |

Full500 是 canonical evaluation feature dataset；sampled optimizer feature dataset 不替代未来完整 500-candidate evaluation universe。

## 7. Leakage、null 与 schema audit

| Audit | Violations |
|---|---:|
| Max history date after cutoff | 0 |
| Future label reads | 0 |
| Validation reads | 0 |
| Test reads | 0 |
| Denylist model-input fields | 0 |
| Static attribute null codes | 0 |
| Category affinity null/default semantics | 0 |
| User/item/user-item recency null semantics | 0 |
| Source-local rank/score null semantics | 0 |
| NaN / Inf model features | 0 |
| Physical Parquet schema mismatches | 0 |

Future labels 和 Stage 4B-2 population/control/sampling 字段均未进入 feature producer 或 model-input schema。`label=0` 仍只表示未来窗口未观察到购买，不表示 dislike、真实负反馈或曝光未购买。

## 8. Deterministic fingerprint 与 replay

| Artifact | Fingerprint |
|---|---|
| Feature schema | `5964882d45d0c3453fb49b3dac39fb187969989d364a63e7d4c18b3ed49896a3` |
| Formal features | `f840fe3ac3fbf11929d24ebd6f0e9a0cc77cda3b311edfc00a37299ef519bccc` |
| Reconciled replay | `f840fe3ac3fbf11929d24ebd6f0e9a0cc77cda3b311edfc00a37299ef519bccc` |
| Feature statistics | `bd3b42def95636cd2d00ecd81dc127f812cfc81e0e2272773c8089bd4eaeaf4e` |

Replay 使用 reverse scheduling 独立物化 8/32 query-contained partitions，共 16,360,000 rows，覆盖正式行的 25.169%。Membership mismatch 为 0；在固定 schema 下比较每个 query 的完整、有序 feature-row digest，feature value mismatch 为 0。将 replay digests 替换到 formal 32-partition digest 集合后，全局 semantic fingerprint 与 formal 完全一致。

## 9. 资源、checkpoint 与恢复披露

| Resource | Observed |
|---|---:|
| Recorded producer + recovery + final-audit wall time | 1,904.042 s |
| Final read-only audit wall time | 424.004 s |
| Peak RSS | 2,313,728,000 bytes |
| Final-audit peak RSS | 2,058,063,872 bytes |
| Peak temp disk | 147,915,221 bytes |
| Spill observed | true |
| Producer OOM | false |
| Read-only audit OOM | 1 |
| Recovery/task retry events | 4 |
| External network interruption events | 1 |
| Successful checkpoints with prior attempt number > 1 | 134 |
| Verified checkpoint reuse | 181 |
| Recomputed partitions after interruption | 0 |
| Private artifact size | 2,277,109,696 bytes |

除 181 个 producer checkpoints 外，最终还建立了 statistics、feature fingerprints 与 contract audits 三个 finalization checkpoints；它们均绑定 snapshot、config/upstream/schema/source、input/output physical fingerprints，并可在下一次恢复时独立验证。

完整资源记录保存在 `artifacts/stage_4b3/operational_audit.csv`。必须保留的恢复事实如下：

1. 初始 per-value text/SHA digest 路径在单 partition 上约 170 秒后按性能停止，未接受其输出；改为固定 dtype/value/null 敏感的双 64-bit row hash + query/global SHA-256 Merkle。
2. Producer 全部成功后，final audit 修正一次 SQL alias 和一次括号错误。
3. 一次 65M-row global group-by 只读 audit 达到 1.5 GiB DuckDB limit；改为对 32 个 query-contained partitions 做等价 exact audit。该 OOM 没有影响 producer outputs。
4. Codex network/UI stream 中断后，恢复检查确认无旧 writer、无 partial artifact、无重复执行，181 个 checkpoint 全部复用。

Private、tmp 和 checkpoint 均被 Git ignore；tracked private artifacts、raw-data diff、公开绝对路径扫描违规均为 0。公开产物只保存聚合、状态、anonymous internal-key contract metadata 和 fingerprints。

## 10. Tests 与工程检查

```text
Stage 4B-3 targeted tests: 35 passed
full pytest: 334 passed
git diff --check: PASS
privacy/path scan: 0 violations
private/tmp/checkpoint ignore audit: PASS
raw data diff: 0
tracked private artifacts: 0
```

Targeted tests 覆盖 cutoff/inclusive windows、cutoff+1/future contamination、user/item/user-item 语义、retrieval source-local contract、primary source exclusion、category encoding/affinity、leakage denylist、projection identity、semantic/schema fingerprint、null-vs-zero、checkpoint recovery 和 downstream prohibition。

## 11. Files

新增配置、入口、模块、测试和阶段报告：

```text
configs/stage_4b3_ranking_features.yaml
scripts/run_stage_4b3_ranking_features.py
src/ranking_features/
tests/test_stage_4b3.py
docs/stages/stage_4b3_ranking_feature_pipeline.md
```

修改 `.gitignore` 以忽略 Stage 4B-3 private/tmp/checkpoints。公开小型产物位于 `artifacts/stage_4b3/`；完整 aggregates、full500 features、optimizer features、replay 与 recovery records 位于 Git-ignored private/checkpoint 目录。

## 12. Contract Gate

G1–G20 全部 PASS：upstream、calendar、row key、candidate/full/optimizer projection、population control、full500、cutoff、statistics、leakage denylist、retrieval semantics、static lineage limitation、null/default、schema、schema fingerprint、feature fingerprint、replay、resource/privacy/checkpoint 和 no-downstream-execution 均通过。Current-source reproduction audit 作为 G18/G19 的补充 reproducibility evidence，不新增 value gate。

本阶段没有 MAP、NDCG、Recall、HitRate 或 value gate；尚未训练 ranking model，因此不得声称排序效果提升。

## 13. 数据限制与明确未进入

本数据没有 exposure/click/真实负反馈、库存/可售状态、可靠的精确上架时间或 article attribute effective timestamp。本阶段只建立离线 feature materialization contract，不是线上 CTR/CVR 系统，也不构成业务收益或 A/B Test 证据。

```text
LightGBM: NOT ENTERED
LGBMRanker / LambdaRank: NOT ENTERED
Ranking model training: NOT EXECUTED
Validation Ranking: NOT EXECUTED
First7d: NOT EXECUTED
Final refit: NOT EXECUTED
Test Ranking: NOT EXECUTED
Reranking: NOT ENTERED
Retrieval retuning: NOT ENTERED
```

## 14. 建议

本阶段 Codex Contract Gate 状态为 `PASSED_RANKING_FEATURE_CONTRACT_GATE`，ChatGPT 最终技术验收裁决为 `STAGE_4B3_FINAL_ACCEPTANCE / FINAL ACCEPTED`。完成用户授权的本地仓库 closeout commit 后停止；不得 push。任何 LightGBM、Validation selection、First7d、ranking final-refit、Test Ranking 或 reranking 阶段仍需要新的独立正式 Prompt。
