# Stage 3E：最终召回融合与 Candidate Contract 冻结

## 1. 阶段状态

Stage 3E preserved the locked Stage 3D T100 retrieval semantics while freezing a deterministic and replayable 500-candidate contract.

当前实现状态：`PASSED_CONTRACT_GATE`，并已通过 ChatGPT 最终技术验收。Stage 3E 已完成 README/根 AGENTS 收尾，但未进入 Ranking。

## 2. Frozen upstream facts

- Stage 3D：`PASSED_VALUE_GATE`；锁定 `TT1`、`T100`、HNSW `efSearch=200`、固定预算 500。
- Validation/Test 正式 Two-Tower 候选均来自各自 `formal_full_refit` checkpoint，不引用 selection-only checkpoint。
- 正式 nominal caps：Repurchase 50、IUF ItemCF 200、TT1 Two-Tower 100、Trend 7d 100，Popularity 只做唯一补齐。
- 配额语义：`source_rank_cap_before_cross_source_dedup`；插入顺序：Repurchase → ItemCF → Two-Tower → Trend → Popularity fallback。
- Stage 3C 保持 `FAILED_VALUE_GATE`，Content 未进入正式 Union。

## 3. Candidate contract

- 版本：`stage_3e_candidate_contract_v1`；协议：`source_rank_cap_pre_dedup_v1`。
- 主键：`snapshot_id + customer_key + article_key`；`candidate_rank` 为 1-based 连续最终 retrieval order。
- provenance 包含五路布尔标记、`source_count`、`primary_source`、各路 rank 与可用 raw score；不同来源 raw score 未相加或校准。
- Validation 与 Test 显式携带 snapshot/as-of/contract/protocol，不能靠日期猜 snapshot。
- 完整候选在 private ignored 分区中；公开文件只保存聚合、匿名 key 范围、状态、schema 和 fingerprint。

## 4. Zero semantic drift

| Snapshot | Users | Rows | Recall@500 | HitRate@500 | MAP@12 | Membership mismatch | Rank mismatch | Metric mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 141359 | 70679500 | 0.2448639763673124 | 0.4802898885933464 | 0.012184760793980384 | 0 | 0 | 0 |
| Test | 128333 | 64166500 | 0.24138900164539004 | 0.46847468602491354 | 0.014657496029670704 | 0 | 0 | 0 |

候选覆盖全部 Stage 3D formal union 用户；主指标仍只使用 Stage 2B `evaluable_warm_users`（Validation 129705、Test 117366），cold-history 单列而未混入主平均。

## 5. Semantic fingerprint 与 replay

语义指纹版本为 `candidate_semantic_sha256_user_merkle_v1`：每用户按 candidate rank/article 生成 canonical digest，再按 snapshot/customer 排序聚合。它不依赖 Parquet compression、row group、文件名或 partition scan 顺序，但 membership/rank 变化会改变结果。

- Validation semantic fingerprint：`1b48ebf8b8e30a847cd7f95a7da1f23f881763f1fed1a9fb1e369a0fb557c2e0`。
- Test semantic fingerprint：`fd94f546add97637e15652c800bc636f3c7723bc4ae7d9cb055f82d5e83e9948`。
- 实际 replay 以反向 partition 调度顺序重新物化；两 个 snapshot 的 replay semantic fingerprint 均与正式产物一致。Physical fingerprint 独立保存，只检测 byte mutation，不冒充 semantic fingerprint。

## 6. Contract gate

- `G1_UPSTREAM_LINEAGE`: `PASS`
- `G2_PROTOCOL_IDENTITY`: `PASS`
- `G3_LOCKED_T100_IDENTITY`: `PASS`
- `G4_CANDIDATE_UNIQUENESS`: `PASS`
- `G5_BUDGET_INVARIANT`: `PASS`
- `G6_RANK_DETERMINISM`: `PASS`
- `G7_PROVENANCE_COMPLETENESS`: `PASS`
- `G8_ZERO_SEMANTIC_DRIFT`: `PASS`
- `G9_SNAPSHOT_ISOLATION`: `PASS`
- `G10_REPLAY_SEMANTIC_FINGERPRINT`: `PASS`
- `G11_PRIVACY`: `PASS`

## 7. 资源与恢复

本轮 checkpoint group 累计 1916.731 秒，观测峰值 RSS 1756053504 bytes，Stage 3E temp 目录峰值 682393600 bytes，DuckDB memory limit followed the configured 35% available-memory rule, bounded to 512–1024 MiB. Across recorded Stage 3E task groups, the observed configured DuckDB memory-limit range was 512–1024 MiB。正式与 replay 均按 snapshot/partition checkpoint；checkpoint 绑定 config、source、upstream、contract、snapshot、partition 与 input/output physical fingerprint，任一变化均拒绝复用。

## 8. 数据保护与限制

raw 未修改；private/tmp/checkpoints 被 Git ignore。公开审计未发现真实 customer/article/postal 值、本地用户名、绝对路径或解释器路径。数据缺少曝光、点击、真实负反馈、库存与精确上架时间；离线 Recall/MAP 不等于 CTR/CVR、收入、线上收益或 A/B Test。

Ranking-training snapshot is NOT frozen. LightGBM Ranking is NOT implemented. Ranking features are NOT implemented. Negative sampling is NOT implemented. Reranking is NOT implemented. Stage 3E 没有生成 ranking label、没有缩小 500 候选、没有独立粗排，也没有进入后续阶段。

## 9. 网络中断恢复与公开审计事件

恢复审计分类为 `C. Stage 3E 已部分运行，有合法 checkpoint`。恢复时已有 119 个 formal/replay 分区 checkpoint 通过 config、source、upstream、snapshot、partition、contract version 与 input/output physical fingerprint 核验；其余分区由原可恢复进程继续完成。候选正式化结束后共有 128/128 个 SUCCESS checkpoint，最终公开收尾直接复用这些 checkpoint，失效 checkpoint 为 0，没有删除或重跑合法候选分区。

第一次公开 finalization 在隐私扫描处停止：一个已声明 SHA-256 的内部十位纯数字子串被通用 article-value 正则误报。修复后的扫描只允许结构化、已验证来源中的完整 SHA-256；待扫描文件不能授权自身出现的未知 64-hex token。该事件没有改变 config/source/upstream/semantic fingerprint，没有重跑候选、evaluator 或 replay，也不是数据泄漏。

正式运行未发生 OOM。`spill_observed` 使用 Stage 3E temp 目录保守占用判定；已有 temp 文件会成为后续组基线，因此它证明运行期间存在 temp 占用，不能把 128 个标记逐组解释为各组独立产生 spill。
