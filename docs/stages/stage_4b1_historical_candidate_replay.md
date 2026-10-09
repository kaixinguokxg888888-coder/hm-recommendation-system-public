# Stage 4B-1：Historical Candidate Replay & Contract Gate

## 1. 阶段状态

- Codex implementation recommendation：`PASSED_HISTORICAL_REPLAY_GATE`。
- ChatGPT technical review：`FINAL ACCEPTED`。
- Final stage status：`PASSED_HISTORICAL_REPLAY_GATE`。
- 本阶段没有进入 Stage 4B-2。

## 2. Snapshot 与授权边界

- Snapshot：`ranking_train_14d_v1`；as-of/history cutoff：`2020-08-11`；request date：`2020-08-12`。
- Future window 仅用于枚举 request users；没有生成 item-level label、Ranking Dataset、negative sampling、Ranking features 或模型监督。
- Stage 3D/3E formal artifacts 与 raw 均只读。

## 3. Upstream lineage 与历史维护披露

- `artifacts/stage_4b1/upstream_fingerprint_manifest.csv` 分别以 `stage_3b_source_current` 和 `stage_3b_source_at_lock` 保存两个 source fingerprints；二者不相等，未被静默视为相同。
- Stage 3B current source fingerprint：`905832868bdfb7ce9a13ce718a8bf1641c616754de24e77d7b815ef658273803`。
- Stage 3B lock-time source fingerprint：`e5304699d8ee4d0bb25348400cfd5ba5973c477d58916a9dd3e35d3f26f8a84f`。
- 这是仓库已有的历史维护差异。Stage 4B-1 preflight 中 frozen Stage 3B config、Stage 3B lock，以及 Stage 4A/3E/3D 等正式 upstream fingerprints 均匹配。
- `HISTORICAL_PROTOCOL_IDENTITY=PASS` 仅表示 frozen ItemCF/retrieval protocol semantics 被保持；不表示 current Stage 3B source bytes 与 lock-time source bytes 相等，也不作 source hash equality 声明。

## 4. Historical retrieval

- Repurchase、IUF ItemCF、Trend 7d 与 Popularity 全部从 cutoff 前 B feedback 重建。
- Historical TT1 重新初始化，使用全部 `2455457` 个 eligible examples、`878795` 个 users，固定 epoch=1，internal dev=0。
- ANN 固定 HNSW cosine、M=32、efConstruction=200、efSearch=200、exact rerank；审计通过。
- Fusion 保持 Repurchase 50、ItemCF 200、Two-Tower 100、Trend 100，Popularity 唯一补齐至 500；source cap 在跨源去重前应用，raw score 不跨源相加。

## 5. Historical Candidate Contract

- Users：`130001`；rows：`65000500`。
- Candidate count min/max/mean：`500` / `500` / `500.0`。
- Duplicate users：`0`；rank discontinuity users：`0`；provenance incomplete rows：`0`。
- Contract：`stage_3e_candidate_contract_v1`；protocol：`source_rank_cap_pre_dedup_v1`。
- Formal semantic fingerprint：`1448b2dd20ffcc3afceaa35143eec6f86f2ec5900a7fb24a3ebfb28ade8b7dc0`。
- Reverse-schedule replay fingerprint：`1448b2dd20ffcc3afceaa35143eec6f86f2ec5900a7fb24a3ebfb28ade8b7dc0`；membership mismatch=`0`，rank mismatch=`0`。

## 6. Contract Gate

- `G1_UPSTREAM_FINGERPRINTS`: `PASS`
- `G2_HISTORICAL_SNAPSHOT_ISOLATION`: `PASS`
- `G3_HISTORICAL_TT1_CAUSALITY`: `PASS`
- `G4_HISTORICAL_TT1_FULL_REFIT_LINEAGE`: `PASS`
- `G5_HISTORICAL_ANN_AUDIT`: `PASS`
- `G6_HISTORICAL_PROTOCOL_IDENTITY`: `PASS`
- `G7_CANDIDATE_UNIQUENESS`: `PASS`
- `G8_CANDIDATE_BUDGET_INVARIANT`: `PASS`
- `G9_RANK_CONTINUITY`: `PASS`
- `G10_PROVENANCE_COMPLETENESS`: `PASS`
- `G11_LABEL_AGNOSTIC_CANDIDATE_MEMBERSHIP`: `PASS`
- `G12_HISTORICAL_FORMAL_SEMANTIC_FINGERPRINT`: `PASS`
- `G13_HISTORICAL_REPLAY_SEMANTIC_IDENTITY`: `PASS`
- `G14_PRIVACY`: `PASS`
- `G15_STAGE4B2_NOT_ENTERED`: `PASS`

## 7. Leakage、隐私与资源

- History after cutoff：`0`；max history date：`2020-08-11`。
- Public privacy、raw mutation、Stage 3D/3E mutation audits：`PASS`。
- Recorded component wall time：`8145.735763499979` seconds；peak RSS：`2554261504` bytes；peak temp：`16434364416` bytes；private artifact size：`5606976892` bytes。
- Spill observed：`True`；OOM：`False`；warning count：`1`；retry count：`1`；checkpoint reuse count：`278`。
- Retry disclosure：首次 TT1 historical task 因 Stage 4B-1 private directory 尚未初始化而失败；这是工程初始化失败。修复目录初始化后正式执行成功，没有修改 frozen model/retrieval semantics，首次失败保留在 operational audit 中。
- Warning disclosure：运行中曾出现只读 NumPy view 转 tensor warning，随后通过显式复制修复。warning 保留在 operational audit 中；修复不改变模型、candidate membership/rank 或 frozen protocol。
- 上述 retry 与 warning 均未改变任何正式 model/candidate output；成功结果来自修复后的正式执行。

## 8. 明确未实现

- Stage 4B-2：`NOT ENTERED`
- Ranking Dataset：`NOT IMPLEMENTED`
- Ranking features：`NOT IMPLEMENTED`
- LightGBM：`NOT IMPLEMENTED`
- Test Ranking evaluation：`NOT EXECUTED`
- Commit / push：`NOT DONE`

离线 historical replay 不代表 CTR/CVR、业务收益或线上 A/B Test。数据仍缺少曝光、点击、真实负反馈、库存和精确上架时间。
