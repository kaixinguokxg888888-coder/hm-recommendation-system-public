# Stage 4B-2：Ranking Dataset / Label / Population / Deterministic Negative Sampling

## 1. 阶段授权与当前状态

- 本阶段只实现 `ranking_train_14d_v1` 的 Ranking Ground Truth、candidate label、query/population、完整 evaluation view、optimizer view、确定性 negative sampling、fingerprint/replay、checkpoint/resource/privacy audit 与 Contract Gate。
- ChatGPT 最终技术验收已通过；正式状态为 `PASSED_RANKING_DATASET_CONTRACT_GATE / FINAL ACCEPTED`。
- Ranking features、LightGBM、Validation selection、First7d、final refit、Test Ranking 与 reranking 均未进入。

## 2. Git、环境与 upstream preflight

- 开工分支为 `main`，HEAD 为 `58e98790095d79e24a05043dd9e8cc94af4074dd`；开工时 worktree clean，`main...origin/main=0/0`。
- 最新提交为 Stage 4B-1 closeout；README 与 AGENTS 已同步标明 Stage 4B-1 `FINAL ACCEPTED`，本地与远端同步。
- 项目 `.venv` 使用 Python 3.10.11、DuckDB 1.5.5、PyArrow 24.0.0。
- 初始全量测试为 `278 passed`。
- 37 个 Stage 4B-1 声明的公开 upstream 文件与 3 个 Stage 2A private inputs 实际 SHA-256 全部匹配。
- Stage 4B-1 正式 32 candidate partitions 重新计算的 semantic fingerprint 为 `1448b2dd20ffcc3afceaa35143eec6f86f2ec5900a7fb24a3ebfb28ade8b7dc0`，与正式 manifest 一致。

## 3. Ranking snapshot 与 Ground Truth

- Protocol：`single_historical_14d_snapshot_v1`。
- Snapshot：`ranking_train_14d_v1`；as-of `2020-08-11`；request `2020-08-12`；future label window `2020-08-12` 至 `2020-08-25`，连续 14 天。
- Ground Truth 严格使用 B-feedback，同一 future-window user/item 无论出现多少购买日，最终只形成一个 binary relevance positive。
- `ranking_train_all_label_users` 为 `130001`；future positive user-item 为 `464982`。
- `label=0` 仅表示该 candidate item 在 future window 内未观察到购买，不是 dislike、真实负反馈或曝光未购买。

## 4. Candidate-before-label 与 Historical Candidate Input

- 输入为 Stage 4B-1 frozen historical candidate：`130001` users、`65000500` rows、每 query 严格 500。
- Candidate contract 为 `stage_3e_candidate_contract_v1`，protocol 为 `source_rank_cap_pre_dedup_v1`。
- 流程先验证 membership/rank/provenance 和 upstream fingerprint，再独立生成 future GT，最后 LEFT JOIN label。Future positive 没有注入 candidate，也没有改变 candidate rank。
- Candidate projection fingerprint 为 `1448b2dd20ffcc3afceaa35143eec6f86f2ec5900a7fb24a3ebfb28ade8b7dc0`；与 Stage 4B-1 formal fingerprint 相等。
- Candidate membership mismatch=`0`、rank/provenance/source-score/lineage mismatch=`0`、label join mismatch=`0`、query partition mismatch=`0`。

## 5. Ranking populations

| Population | Queries | Main average | Optimizer |
| --- | ---: | --- | --- |
| `ranking_train_all_label_users` | 130001 | 否，作为全集审计 | 否，需继续判定 |
| `historical_evaluable_warm_users_v1` | 120608 | 是 | 仅 candidate positive >= 1 |
| `cold_history_label_users` | 9393 | 否，单独汇总 | 否 |

Main population 包含符合历史非空定义的 short-history、warm 和 active users，绝不等于 `user_segment == "warm"`。Main/cold disjoint，population gap、history-definition violation 均为 0；candidate user set 与独立重建 all-label user set 双向差集均为 0。

## 6. Candidate-positive coverage 与 retrieval miss

| Population | Future positives | Candidate positives | Query hit coverage | Macro coverage | Micro coverage | Retrieval miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main | 434040 | 86356 | 0.4269783099 | 0.2096930812 | 0.1989586213 | 69111 / 0.5730216901 |
| Cold-history | 30942 | 4379 | 0.3140636644 | 0.1442781165 | 0.1415228492 | 不混入 main 定义 |
| All-label audit | 464982 | 90735 | 0.4188198552 | 0.2049666347 | 0.1951365859 | main miss count 69111 |

Main segment 与 future-positive-count 分组的完整安全聚合见 `candidate_positive_coverage.csv`。这些值测量 frozen Retrieval ceiling，不是 Stage 4B-2 value gate，也未触发 retrieval、quota、TT1 或 ANN 调整。

## 7. Full Evaluation View

- Canonical full labeled dataset 保留 `130001` queries、`65000500` rows，每 query candidate count min/max/mean=`500/500/500.0`。
- Main full view 为 `120608` queries、`60304000` rows；cold full view 为 `9393` queries、`4696500` rows。
- Main 的 `69111` 个 all-zero relevance / retrieval-miss queries 全部保留在 full view；没有 negative downsampling。
- Full view 通过 canonical dataset + query population/view definition 表达，没有额外复制另一份 65M-row evaluation Parquet。

## 8. Optimizer View 与 deterministic negative sampling

- Optimizer eligibility：main population 且 `candidate_positive_count >= 1`；query count=`51497`，排除 main retrieval miss=`69111`，optimizer all-zero query=`0`。
- Optimizer-eligible full500 rows=`25748500`，其中 positives=`86356`、label-zero rows=`25662144`。
- Sampled optimizer positives=`86356`，全部保留；sampled label-zero rows=`5149700`；sampled total rows=`5236056`。
- Group size min/max/mean=`101/117/101.6769132182`；每 query label-zero max=`100`；candidate rank 未重编号。
- Sampling version `sha256_zero_cap100_v1`：seed=`20260811`，UTF-8 SHA-256 payload 为 `{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}`，按 digest ASC、article key ASC 取前 100。Partition/file/row order 与 Python built-in `hash()` 均不参与。

## 9. Query/group、fingerprint 与 replay

- Ranking row key：`(snapshot_id, customer_key, article_key)`；query key：`(snapshot_id, customer_key)`；query order 按 canonical key 决定。
- Ranking dataset version：`ranking_dataset_14d_binary_v1`；label version：`candidate_purchase_relevance_b14_v1`；optimizer view：`ranking_optimizer_sha256_100_v1`。
- Full labeled semantic fingerprint：`aa04fce0514d8fe77b73d3600c31bfba76b3b0e21f347077fe7395fa354db343`。
- Optimizer formal semantic fingerprint：`3e634e20aafbf6737fbd4160b4eadd80a156e020b137462cdef616c26bf49fa8`。
- Reverse partition scheduling + reverse input scan replay fingerprint 与 formal 相等；membership mismatch=`0`，payload/rank/label mismatch=`0`。

## 10. Checkpoint、resources、warnings 与 retries

- 正式 DAG 使用 GT 32 partitions、history 32 partitions、label join 32 partitions、formal sampling 32 partitions、replay 32 partitions，以及 merge/query/group/audit checkpoints；成功 producer checkpoints 共 164 个。
- 成功 producer component wall time=`2315.429273199523` 秒，closeout 最终成功 audit wall time=`550.2971342999954` 秒，合计 component wall time=`2865.7264074995182` 秒；peak RSS=`1581744128` bytes，private artifacts=`894526867` bytes。
- Peak temp 至少 `18600000000` bytes；spill observed=`true`，OOM observed=`true`，warning count=`1`。该 warning 是 sandbox 无权读取用户级 Git global-ignore 文件；仓库 `.gitignore` 的 private/tmp/checkpoint 检查仍完成并通过。
- 共 7 个恢复事件，详见 `operational_audit.csv`：三次 monolithic GT/history memory OOM、一次全局 provenance audit temp-limit OOM，以及三次 SQL/CSV API 适配。正式成功方案将 GT/history/provenance audit 改为 32 个 deterministic query-contained partitions；没有抽样用户、降低 500/100、修改 future GT 或改变 frozen semantics。所有 formal private outputs 来自成功的 partitioned execution。
- Audit-only 修复后没有静默重写既有成功数据；finalization 逐一验证 producer config/upstream fingerprint 与 output physical fingerprint，并单独记录 producer/current source lineage。

## 11. Contract Gate

`artifacts/stage_4b2/contract_gate.csv` 的 G1–G22 全部 `PASS`：

- G1–G4：upstream、candidate fingerprint、request-user identity、calendar；
- G5–G9：candidate-before-label、zero drift、ranking key、binary label、no positive injection；
- G10–G15：population、positive accounting、retrieval miss、full view、cold isolation、optimizer eligibility；
- G16–G19：sampling contract、replay identity、positive preservation、full/optimizer isolation；
- G20–G22：schema/lineage/fingerprint、resource/privacy/checkpoint、no downstream execution。

最终 Codex recommendation 为 `PASSED_RANKING_DATASET_CONTRACT_GATE`。Coverage 数值没有作为通过阈值。

## 12. Privacy、数据保护与限制

- `data/raw/` 没有写目标；Stage 4B-1 candidate 与 Stage 2A private inputs 全程只读。
- 完整 candidate/query/user/optimizer/replay/checkpoint/tmp 均位于 Git ignored private paths；public artifacts 只含安全聚合、schema、fingerprint、status 与 resource 信息。
- Public artifact 不保存真实 customer/article ID、internal entity values、internal key mapping、本地绝对路径、用户名或解释器路径。
- 数据仍缺少曝光、点击、真实负反馈、库存和精确上架时间；本阶段不能表述为 CTR/CVR、业务收益或线上 A/B Test。

## 13. 测试与明确未进入

- Initial full tests：`278 passed`。
- Stage 4B-2 targeted tests：correction 前实际为 `20 passed`；加入直接锁定 query-level group aggregate 的 regression test 后为 `21 passed`。测试覆盖 calendar、B-feedback dedup、candidate freeze、population、retrieval miss、golden SHA-256、zero cap 边界、order/partition invariance、rank preservation、fingerprint sensitivity、group aggregate reconciliation 与 ignore/path/privacy。早期报告中的 `19 passed` 是新增 materialized-public privacy/path test 后未同步的旧计数，本次 targeted correction 已统一并补回归保护。
- Final full pytest：correction 前为 `298 passed`；新增 group aggregate regression test 后为 `299 passed`；`git diff --check` 通过。
- 12 个 public artifacts 对 raw ID 字段、用户名、本地绝对路径的扫描均为 0；private/tmp/checkpoints Git ignored 且 tracked files=`0`；37 个公开 upstream 文件复核 mismatch=`0`；22 个 contract gates fail=`0`。
- 最终 worktree 仅包含本阶段 `.gitignore`、config、source、runner、tests、12 个公开 artifacts、本报告，以及 closeout 所需的 README/AGENTS 更新；commit/push 未执行。

明确未进入：

```text
Ranking Features: NOT ENTERED
LightGBM: NOT ENTERED
Validation selection: NOT EXECUTED
First7d: NOT EXECUTED
Final refit: NOT EXECUTED
Test Ranking: NOT EXECUTED
Reranking: NOT ENTERED
```

STOP after Stage 4B-2 closeout. 等待用户明确授权 commit/push；不进入下一阶段。
