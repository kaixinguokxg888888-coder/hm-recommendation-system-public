# Stage 4B-4 Phase 1 — LambdaRank Pre-label Prediction Freeze

## Final status

`STAGE_4B4_PHASE1_PREDICTIONS_FROZEN` / `PRE_LABEL CONTRACT GATE: PASSED`.

Validation label join、Validation metrics、winner selection、First7d、final refit、Test、reranking 与 retrieval retuning 均未执行。执行期间未 commit、未 push。

## Training lineage

- 冻结 optimizer：51,497 queries、5,236,056 rows；positive 86,356，sampled zero 5,149,700。
- query-order-v2 的 query/group/positive/zero/row-key/candidate-rank/boundary/duplicate/all-zero 审计均为 0 violations。
- L0/F0 与 L1/F1 均使用全部 optimizer、固定 600 rounds；model 与 matrix/order/boundary/feature fingerprints 完整绑定。
- L0 peak RSS 1,122,066,432 / cap 2,009,026,969 bytes；L1 peak RSS 1,646,505,984 / cap 1,942,920,806 bytes。
- 历史同进程资源失败与旧 partition-major/group-order 冲突证据继续保留；本轮只使用获授权的 query-order-v2 与已验收隔离模型，没有自动重训。

## Interruption recovery

- Partitions 0–3：input/config/model/feature fingerprint、row count、output physical fingerprint 与 SUCCESS checkpoint identity 全部复验通过，直接复用。
- Partition 4：中断时的 7 个无 digest、无 SUCCESS checkpoint 文件标记为 `INCOMPLETE_AFTER_INTERRUPTION`；在独立 temporary attempt 中重建全部 7 版本，完整验证后 atomic promotion。
- Partitions 5–31：仅执行当时缺失部分。长时宿主 shell 在 partition 21 前终止后，再次按 checkpoint identity/physical fingerprint 复验 4–20 并从 21 续跑，未重算成功分区。
- 总体：4 个中断前成功 partition 复用，28 个 partition 在本轮恢复中完成；32/32 prediction checkpoints SUCCESS，0 FAILED。

## Prediction freeze

七个版本 `B0`、`L0_100`、`L0_300`、`L0_600`、`L1_100`、`L1_300`、`L1_600` 均覆盖 32 partitions、141,359 queries、70,679,500 rows，每 query 恰好 500 rows。

全量流式审计确认 query/key 唯一、candidate rank 1–500、predicted rank 1–500、raw score 全部有限、B0 candidate-rank identity 与冻结 learned tie-break 均为 0 violations。每版本的 physical、raw-score semantic、ranked-list semantic、query digest 与 global Merkle fingerprint 已冻结在公开 manifest。

Partition 0 的七版本独立 replay 对 row key、raw score、predicted rank 与 query digest 的 mismatch 全部为 0。

## Gate, tests and privacy

- G1–G20：20/20 PASS。
- Targeted pytest：19 passed；full pytest：353 passed；compile audit：PASS；`git diff --check`：PASS。
- Public privacy/path scan、raw-data diff、private/tmp/checkpoint ignore、tracked private artifact audit：全部 PASS。
- Validation label read count、First7d read count、Test candidate/feature/label/metric read count：全部为 0。

## Pre-label lock

`validation_predictions_frozen=true`；`winner_selected=false`；`final_refit_executed=false`；`test_authorized=false`。Validation label join 仍未执行，当前仅准备好等待下一份独立正式授权。

```text
STAGE_4B4_PHASE1_PREDICTIONS_FROZEN

INTERRUPTION RECOVERY:
SUCCESS

REUSED PREDICTION PARTITIONS:
0–3

RECOMPUTED INCOMPLETE PARTITION:
4

PRE_LABEL CONTRACT GATE:
PASSED

VALIDATION LABEL JOIN:
NOT EXECUTED

READY_FOR_VALIDATION_LABEL_JOIN:
YES
```
