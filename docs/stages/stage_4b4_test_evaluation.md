# Stage 4B-4 Phase 5B：One-Time Test Label Join and Final Evaluation

## 1. 阶段定位与状态

本阶段在 Phase 5A pre-label lock 后唯一一次读取 Test labels、冻结 canonical ground truth，并只评估冻结的 `B0` 与 `L1_FINAL_REFIT_600`。正式状态为 `STAGE_4B4_PHASE5B_TEST_EVALUATION_COMPLETE`。

## 2. 冻结身份与访问

- Phase 5A pre-label lock：`34c8dcc52ee18c18766904d234e8fda0f2964e4805fee3175e9b45c375f7e8ca`。
- Phase 4 final-refit lock：`358c25d19ecce3006428a293acd61564e0f6cd3fbdc0c156aa636ee41bef4bd8`。
- Test label window：2020-09-09 至 2020-09-22（闭区间，B user-item-day binary）。
- Raw Test label source read count：1；Test labels reopened：NO。
- Ground-truth build / prediction-label join / formal evaluation / metric materialization：均为 1。

## 3. Evaluation population

| Population | Queries |
| --- | ---: |
| All candidate queries | 128333 |
| Main evaluable warm users | 117366 |
| Cold-history diagnostic | 10967 |
| Main candidate-positive nonzero | 54983 |
| Main all-zero relevance | 62383 |

Main evaluation 保留全部 all-zero relevance queries；这些 query 的 AP/NDCG/Recall/HitRate 均为 0。Cold-history 只作独立诊断，没有混入 primary average。

## 4. Frozen Test metrics

| Version | MAP@12 | NDCG@12 | Recall@12 | HitRate@12 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.014657496029671 | 0.022618270768624 | 0.033072257383157 | 0.068213963157984 |
| L1_FINAL_REFIT_600 | 0.029620310798481 | 0.046585786835952 | 0.065510954243462 | 0.146703474600821 |

MAP@12 delta（final - B0）为 `0.014962814768810`；paired query bootstrap 95% CI 为 `[0.014437462108487, 0.015461287205350]`。Bootstrap 固定为 1000 次、seed 20260826、percentile interval。

## 5. Final statistical status

- Value Gate：`PASSED_FINAL_TEST_VALUE_GATE`。
- Statistical status：`POSITIVE_AND_SIGNIFICANT`。
- Segment status：`NO_CRITICAL_SEGMENT_REGRESSION`。

Segment diagnostics 完整保留 short-history、warm、active、single/multi-positive、candidate-positive/all-zero、cold-positive、cold-history 与 seen/cold/head/mid/tail/zero-history target 结果。它们只作诊断，没有触发换模型或二次选择。

## 6. 工程、资源与 lock

- P5BG1–P5BG20：20/20 PASS。
- Final Test evaluation lock：`daddf4c1cb310f9075e06ebf6189932a58eeb0a96346385d0fb930e792bb1b88`。
- 32 个 query-contained partitions 全部完成；resource cap、checkpoint、privacy/path、ignore、raw-data diff 与 access audit 全部 PASS。
- 未重新生成 candidates/features/predictions，未调参、未重训、未改变 winner、未执行 reranking/retrieval retuning。
- README 与 AGENTS 尚未更新；未 commit、未 push。

这些是一次性离线 Test 指标，不是线上 CTR/CVR、收入提升或 A/B Test 结果。数据仍缺少曝光、点击、真实负反馈、库存/可售状态及可靠 article effective timestamp。Test 结果无论好坏均未用于二次选择。
