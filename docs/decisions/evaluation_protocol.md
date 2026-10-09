# V1 离线评估协议

决策编号：`EVAL-001`

状态：`FROZEN`

## 1. 评估单位与标签

- 评估单位：`request_id + customer`；
- 主标签：未来窗口内 B 口径购买过的 article 集合；
- 主窗口：14 天；7 天用于快速方向检查，28 天用于稳健性；
- 可评 warm 用户：cutoff 前已见且标签窗口至少一个正样本；
- cold 用户单独评估，不与 warm 总体平均混为一个结论；
- 已购商品不从标签中删除，因为 Stage 1B 显示复购信号显著。

## 2. 召回指标

| 指标 | K | 用途 |
| --- | --- | --- |
| Recall@K | 50、100、300、500、1000 | 主指标为 union Recall@500 |
| HitRate@K | 50、100、300、500 | 衡量至少命中一个正样本 |
| Candidate coverage | 各路及 union | 可产生候选的用户比例 |
| Independent contribution | 500 | 仅由该来源命中的正样本 |
| Union gain | 500 | 加入某来源后的边际 Recall |
| Segment Recall | 100、500 | cold/short/warm/active 分群 |

每个来源还需报告候选数量分布、去重率和来源命中数分布。

## 3. 排序指标

Top-12 最终列表：

- 主指标：`MAP@12`；
- 关键辅助：`NDCG@12`；
- 同时报告 `Recall@12`、`HitRate@12`；
- 不解释为 CTR、CVR 或校准概率；
- 对只有一个标签正样本的用户单独报告，避免总体平均掩盖差异。

## 4. 重排与系统指标

- Catalog coverage@12；
- Long-tail coverage@12，至少分别报告非 Top 10% 和非 Top 20% 商品；
- 推荐列表平均历史 popularity 与 Top 10% 热门占比；
- intra-list department、garment group、colour 多样性；
- novelty：基于训练期商品流行度的负对数聚合，仅作相对指标；
- 已购商品比例和复购命中率；
- cold-user、cold-item、short/warm/active 分群指标；
- 每用户候选数、去重率、来源覆盖；
- 运行时间、峰值 RSS、临时盘、产物大小。

多样性与长尾指标是约束/辅助目标，不能单独替代 MAP/NDCG。

## 5. 报告切片

每个核心指标必须至少按以下维度报告：

- cold / short-history / warm / active；
- 7 / 14 / 28 天窗口；
- 各召回来源和 union；
- 标签正样本数为 1 与大于 1；
- warm item 与 cold item。

不得只报告总体平均。

## 6. 调参与测试纪律

1. 所有超参数、反馈口径、配额和重排阈值只在 validation 选择；
2. test 不得用于早停、特征筛选、阈值调整或模型选择；
3. 选择完成后可用 train+validation 重训，test 只执行一次；
4. 候选、特征、截断分位数和归一化参数必须按 cutoff 重放；
5. 任何实验必须记录输入 fingerprint、配置、随机种子、代码版本与 as-of 日期。

## 7. 稳健性与显著性

- 对同一用户上的模型差值做 paired user-block bootstrap，至少 1,000 次，报告 95% 区间；
- P0 至少要求 14 天 validation 改善且 7 天方向一致；
- P1 追加三个或更多 rolling 14 天切点；
- 若置信区间跨 0、分群退化明显或窗口方向相反，结果标记为不确定，不进入主线；
- test 不因显著性不足而反复重跑调参。

## 8. 默认成功门槛

- baseline 构建类实验：结果可复现、无泄漏、指标与资源完整即通过；
- challenger：主指标相对提升且 95% 区间不跨 0，同时关键分群无超过 validation 预注册容忍度的退化；
- 多样性重排：准确率退化不超过预注册容忍度，并改善至少一项 coverage/diversity 指标；
- 具体数值容忍度在首次 P0 baseline 后只用 validation 预注册，不得事后依据 test 设置。
