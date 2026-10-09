# Stage 3D Two-Tower 路线变更裁定

决策编号：`ROUTE-3D-001`

状态：`FROZEN_FOR_STAGE_3D`

## 原路线与本次授权

Stage 1C 将隐式矩阵分解列为 P0 challenger，将 Two-Tower 列为 P2，并冻结“不设置无证据的独立粗排”和“召回冻结后以 LightGBM Ranking 作为强排序基线”。用户在 Stage 3D 正式 Prompt 中显式授权改变召回 challenger 的执行顺序：当前 V1 主线跳过 MF，直接实现受限 Two-Tower Retrieval Challenger。

本裁定只覆盖执行顺序，不回写或改造 Stage 1C 历史报告、CSV 和冻结事实。独立粗排仍跳过，LightGBM Ranking 仍是最终召回融合与候选冻结后的下一项核心模型。

## 为什么跳过 MF

- 秋招展示需要在有限时间内保留一个具备独立编码、向量检索和工程审计价值的学习式召回案例；
- 项目链路已经有 Popularity、Trend、Repurchase、IUF ItemCF 和失败的结构化 Content 对照，Two-Tower 能补充训练、embedding 导出与 ANN 质量门禁；
- 同时实现 MF、Two-Tower 和多个排序模型会挤压正式实验、复现、资源与泄漏审计时间；
- 模型数量不是目标，固定预算下可验证的真实增量才是目标。

## 证据门槛与停止条件

Two-Tower 必须面对 Stage 3B 冻结的 IUF Q0，候选总预算固定为 500。模型名称、网络复杂度、单路 Recall 或候选独占性不能替代 T50/T100 对 Q0 的净增量、paired user-block bootstrap、独占与丢失命中、分群、ANN overlap 和资源成本证据。

若 Stage 3D 未通过预注册 value/retrieval gate，不补做 MF、BPR、序列召回、扩大双塔、重新打开 Content 或增加融合搜索空间。本阶段结束后只建议进入新的 Stage 3E，完成最终召回融合与候选冻结，再进入 LightGBM Ranking；本裁定不授权自动实现 Stage 3E、LightGBM、粗排或重排。
