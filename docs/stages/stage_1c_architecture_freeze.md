# Stage 1C：V1 推荐架构证据评审与冻结

状态：`PASSED`

执行范围：只读证据评审、架构决策和实验路线冻结。

未执行：清洗、特征工程、样本生成、召回/排序代码、模型训练、离线实验、Stage 2。

## 1. 门禁

- 分支 `main`，Stage 1B 已由 `86d9d05`、`73bf290`、`79ec138` 提交并推送；
- 开始时工作区干净，`main` 与 `origin/main` 同步；
- Stage 1A 6/6、Stage 1B 7/7 checkpoint 完整；
- 当前与 Stage 1A/1B 输入 fingerprint 均为 `f3b4d8bfed7655be122a6b43abf3f11cd526367b89cafefe6daf0c3e8a057927`；
- 49 项现有测试通过；
- Stage 1A/1B 正式产物未修改。

## 2. 冻结结论

| 决策域 | 冻结结论 |
| --- | --- |
| 业务任务 | 已知 customer 的离线日批、未来 14 天隐式购买 Top-12 推荐 |
| 反馈 | B 为主标签；A 审计/敏感性；C 为训练期截断+log 辅助强度 |
| 时间 | 连续 14 天 validation + 14 天 test；7 天快速、28 天稳健 |
| 分群 | cold=0、short=1–4、warm=5–20、active=21+ |
| 召回 | 热门、趋势、复购、ItemCF、内容进入；MF 为 P0 challenger |
| 深度召回 | 双塔、序列模型不进 V1 主线，列入 P2 |
| 融合 | 固定配额 union，以 article 去重并保留所有来源证据 |
| 粗排 | V1 不设独立粗排；候选超过 3,000 或资源超限才触发 |
| 精排 | LightGBM ranking 为 V1 强基线；DeepFM 为 P1 对照 |
| 重排 | 确定性 article/product/garment/热门度/冷商品约束 |
| 评估 | Recall@500 与 MAP@12 为主，必须分群并报告 7/14/28 稳健性 |

## 3. 关键证据

- 31.79M 原始交易、28.58M 二值交互；
- 主外键完整，商品和客户候选主键唯一；
- 2.97M 全字段重复超额，缺少 quantity/order_id，支持选择 B 为主语义；
- 47.30% 用户有 10+ 二值行为，深度序列模型不能覆盖全部用户；
- Gini 0.7586、Top 10% 贡献 60.75%，热门有效但必须控制偏置；
- 用户重复购买率 49.21%，支持复购召回；
- 14 天有 117,366 可评用户，用户可见率 91.45%，warm 候选覆盖上限 83.95%；
- 14 天冷用户 8.55%、冷商品 6.98%，必须有 fallback 和内容路径；
- Stage 1B 峰值 RSS 约 1.23 GiB、临时盘约 5.72 GiB，Stage 2 必须保持可恢复和资源护栏。

完整映射见 `artifacts/stage_1c/evidence_traceability.csv`。

## 4. P0/P1/P2

- P0：可重放 popularity、repurchase、ItemCF、union、content、MF challenger、LightGBM、基础重排；
- P1：时间衰减、配额/融合、负采样、A/B/C、分群、DeepFM、rolling backtest；
- P2：双塔、active-only 序列、复杂多目标融合/重排。

每项实验的控制变量、成功标准与停止条件见 `artifacts/stage_1c/experiment_priority.csv`。

## 5. 验收判断

业务任务、反馈、时间、用户分群、召回、融合、粗排、精排、重排、评估和实验顺序均已形成可追溯决策。正式 CSV 采用统一决策字段，能直接指导后续实现。

Stage 1C 通过不等于任何模型有效或已实现。进入 Stage 2 仍需用户明确确认。
