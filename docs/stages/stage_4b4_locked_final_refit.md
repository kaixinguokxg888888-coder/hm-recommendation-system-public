# Stage 4B-4 Phase 4：Locked Final Refit

## 1. 首次 raw-matrix 训练资源失败

首次正式 raw-memmap 训练按冻结资源门停止：cap 为 2800775577 bytes，进程树峰值为 2818605056 bytes，超出 17829479 bytes；未发生 OOM，也未晋升该次临时模型。原失败证据保持 `BLOCKED_BY_RESOURCE`。

## 2. Binary Dataset 资源恢复

恢复阶段精确复用既有 Validation optimizer 与 combined matrix，未重新采样、物化或读取新标签。LightGBM binary Dataset 含 11568070 rows、113793 queries、52 个 F1 features，semantic fingerprint 为 `8fe8fe6ed1c1cc3e7cd82f2f5845c15955a0f1f45b2c7a9f29f31a41dcf52af9`。

## 3. Binary-only 正式训练

独立 worker 仅加载 formal binary Dataset，以 fresh initialization 训练 L1/F1/LambdaRank 恰好 600 rounds/trees；未使用 warm start、internal validation 或 early stopping。模型 semantic fingerprint 为 `0adc28c0497db8e300fe7c4ec487b0b34fd115b8a9cc5e978763d4a8ef2c675e`。

## 4. Post-training manifest schema 缺失

首次训练后工程审计发现 model manifest 缺少 `prediction_replay.raw_score_fingerprint`。该缺陷只影响 manifest schema，不改变模型、binary、combined matrix 或 replay 语义；失败事实和原始字节证据均已保留。

## 5. Git warning auditor 误分类

旧 auditor 将 Git 全局 ignore 权限 warning 的 stderr 误当成 raw-data diff。修复后 working-tree 与 staged raw diff 分别依据命令 return code 和 stdout/path result 判定，warning 仅以脱敏类别记录，不作为 changed file。

## 6. 失败 audit 的隐私问题

旧失败 audit 含本机绝对路径，不能作为公开产物。其原始字节版本已迁入 Git-ignored private failure-evidence，公开失败证据只保留失败类别、原始 SHA-256、缺失字段、warning 误分类、隐私违规计数与修复状态；原失败事实未删除。

## 7. 本次授权修复范围

本次只执行 manifest schema、auditor、公开隐私、测试审计与 lock/gate/report 修复。未重建 binary、未重训模型、未重新运行 prediction/replay、未打开任何标签，也未执行业务指标或模型选择。

## 8. 模型与 replay 未重算

模型 physical fingerprint `1f2af19a79ed8d085d68d20fe9437f4eca7da3534a3852eb823f483c6bf96922` 与 binary physical fingerprint `df18e505648ad70924d9d59b2a5a888600f893427b1d6c543688666366c3e9d1` 保持不变。Replay audit physical fingerprint `7e52482d421bc2bca85c7bce2209e74aa6993d10ddfa71defbc6fd57a775e347` 保持不变；既有 raw-score fingerprint `bb7eb22b48d1f88670dba8e0defad408aab118d3f0ea8c24769f6aa960aa2715` 与 tie-break fingerprint `0567b99499a83d66864589fab34f75cde8622868b703c4b00904a07e5f9e4dec` 被原样绑定到 corrected manifest。

## 9. Manifest correction evidence

Original incomplete manifest physical fingerprint `3a9b6e9228f6bdd26b6a7f33e83dab7399a0495f187bc1c300c318dd68d4ca8f`；corrected physical fingerprint `a68492fde15c0e4f59c93b40e57dbcd7d41d11bff2d850b315223ba9627b76f3`；corrected semantic fingerprint `8e79067006100b0372f85cba15d0c825863c656334ee1cd00e40882fc5d4721d`。只新增既有 replay 证据的嵌套 metadata/schema，其他原字段保持一致。

## 10. 最终工程与隐私审计

Targeted pytest、full pytest、compile、`git diff --check`、raw-data working/staged diff、privacy/path、ignore 与 tracked-private 审计均 PASS。公开审计不含可恢复的绝对路径或用户名。Sanitized failure audit fingerprint 为 `7ba473e1b760502f252771193d9e2288872f38170ba3df416dd8799cd50f6a21`。

## 11. Final-refit lock 与 R1–R16

Final-refit lock 状态为 `FINAL_REFIT_LOCKED_AFTER_POST_TRAINING_RECOVERY`，fingerprint `de1072d41e921c8cfe39275486a7a7f201dc9ffa26196b9b6fec8a6a6a52a3d6`。Resource-recovery gate 为 R1–R16 全部 PASS。

## 12. 访问边界与结论

Validation labels reopened=NO；First7d read count=0；Test candidate/feature/label/prediction/metric counts 全部为 0，Test 未授权。当前状态为 `STAGE_4B4_PHASE4_FINAL_REFIT_COMPLETE`，等待 ChatGPT 技术验收。该结论仅证明离线 final-refit 工程与协议完成，不代表线上 CTR/CVR、收入提升或 A/B Test 结果。

## 13. Pytest skip audit reconciliation

ChatGPT 独立验收发现上一轮自然语言汇报的 targeted/full skip 数与正式 audit 中保存的 pytest 进度片段不一致。本次核验确认根因为 `AUDIT_PARSER_ERROR`：旧 auditor 的 `safe_summary()` 只保留 stdout 最后一条非空行，不能作为结构化统计；上一轮 `1/2` 又混用了锁后另一批输出，并把 targeted 的 skip 误计入 full。

正式重跑命令通过 `scripts/stage_4b4_pytest_structured_worker.py` 调用 pytest，并固定使用 `-ra`。Targeted 收集 60 项，59 passed，1 skipped，failed/errors/xfailed/xpassed 均为 0；Full 收集 447 项，446 passed，1 skipped，failed/errors/xfailed/xpassed 均为 0。

Targeted command：`python scripts/stage_4b4_pytest_structured_worker.py --output <private-results>/targeted.json -- tests/test_stage_4b4_phase4.py tests/test_stage_4b4_phase4_resource_recovery.py tests/test_stage_4b4_phase4_post_training_recovery.py -ra -q`。

Full command：`python scripts/stage_4b4_pytest_structured_worker.py --output <private-results>/full.json -- -ra -q`。

两套 suite 唯一 skip 均为 `tests/test_stage_4b4_phase4.py::test_phase4_gate_has_exactly_thirteen_passes_when_present`，reason 为 `formal gate is created during finalization`，分类为 `EXPECTED_OBSOLETE_ARTIFACT_SKIP`。该测试只检查未生成的旧 `final_refit_contract_gate.csv`（P4G1–P4G13）；当前正式 gate 是 R1–R16，相关 required tests 已实际执行并通过，因此不会掩盖模型、binary、replay、隐私或 Test 防泄漏问题。

原 test/security audit physical fingerprint `f0ca5a19af80da95f6bd8a6054c8d9d54e21c2184baa03c9a0d7f7e7d975f97f`、原 lock physical/semantic fingerprints `1fa37804666881a1d67b0960bdb0a0ec61468789b743b6b93fcf7d709e7ab778` / `de1072d41e921c8cfe39275486a7a7f201dc9ffa26196b9b6fec8a6a6a52a3d6` 与原 gate physical fingerprint `9e04a6acc9e5e4d7b7a4bb92e072fea1066f90081d108a354f401c0282170015` 已作为 superseded evidence 保留。R1–R14 的语义与 evidence 字节保持不变；R15 现绑定 corrected structured test audit，R16 绑定新 lock `358c25d19ecce3006428a293acd61564e0f6cd3fbdc0c156aa636ee41bef4bd8`。

本次未重训模型、未重建 binary/optimizer/combined matrix、未重新计算 prediction/replay、未打开 Validation/First7d/Test labels，也未读取或物化任何 Test artifact。Phase 4 已满足重新技术验收条件；pytest PASS 仅代表工程测试通过，不代表模型业务效果。
