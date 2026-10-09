# Stage 3D：Two-Tower Retrieval 与固定预算增量召回

## 1. 阶段状态

| 状态项 | 最终值 |
| --- | --- |
| Stage 3D final status | `PASSED_VALUE_GATE` |
| Stage 3D Test status | `SUCCESS` |
| Locked model | `TT1` |
| Locked quota | `T100` |
| Locked ANN efSearch | `200` |

本阶段实现 CPU Two-Tower、HNSW 检索、精确余弦审计和 Q0/T50/T100 固定预算实验。未实现 MF/BPR、序列模型、排序、重排、在线服务或 Stage 3E。

## 2. 路线和冻结边界

Stage 1C 曾把 MF 放在双塔之前；`docs/decisions/stage_3d_two_tower_route_override.md` 只改变本阶段 challenger 的执行顺序，不改写历史事实。TT0/TT1、T50/T100、64 维输出、AdamW、learning rate、batch=512、temperature、seed、ANN 和 source-rank-cap 参数均按原 Validation 选择冻结。本次最终验收修复没有重新比较模型、配额或超参数。

## 3. 正式输入和防泄漏

反馈沿用 B 口径；每用户最多 3 个 first-time acquisition target，context 必须严格早于 target 日，target 不在自身 context，同日其他商品不能进入 context，最近 20 个唯一历史商品按 `1/log2(rank+1)` 池化。Validation/Test 分别按自身 cutoff 构造 snapshot；Test history 可以包含发生时间早于 Test cutoff 的 Validation 时段交易，但 Test label window 不进入 history、训练、候选或参数选择。

公开 `data_leakage_audit.csv` 的 Validation/Test 结果均为：target present in context=0、context without strictly prior event=0、target seen before target date=0、target after snapshot cutoff=0。in-batch 其他 target 只能称为未观察购买项；重复 target 和当前用户历史内 target 被 mask，对角正类保留。

## 4. Internal train/dev 和 user-ID embedding 限制

模型选择阶段按稳定 user-key hash 做 95% train / 5% dev 的用户级拆分。Validation 有 885,882 eligible users、2,475,714 examples；selection train/dev users=841,237/44,645，examples=2,350,832/124,882。

TT0/TT1 的用户塔均含 user-ID embedding。dev 用户完全不参与 selection train，因此这些 dev 用户的 ID embedding 保持由固定 seed 产生的随机初始化值，未被梯度更新；它们不是 unknown/shared embedding，也没有置零。商品 embedding、属性 embedding、历史池化和 projection 参数则从 train 用户得到更新。因此 internal dev loss 可以在这个固定协议内用于相对 epoch 选择，但不能解释为标准的“已知用户 ID embedding 泛化”评估；它混合了未训练 user-ID 行与已训练共享参数的表现。

TT0/TT1 的三轮 selection dev loss 均在 epoch 1 最优，故锁定 epoch=1。internal dev 仅服务 epoch selection，不是正式候选模型训练人口。

## 5. 最终验收发现和 full-refit 修复

第一轮实现把 dev-loss 最优的 `TT1_epoch_1.pt` 直接复制为 `TT1_best.pt`，随后正式 embedding、ANN、candidate 和 lock 都引用该 selection-only checkpoint；Test 也错误地再次使用自身 internal dev 做 epoch 选择。Validation selection checkpoint SHA-256 缩写为 `1ffc…7407`，它与原正式 `TT1_best.pt` 完全相同，证明原流程缺少 full refit；完整值见 `model_lineage.csv`。

修复保持 TT1、T100、epoch=1 和全部超参数不变：

- Validation：重新初始化 TT1，使用 885,882 users / 2,475,714 examples / 0 dev，固定训练 1 epoch；final checkpoint SHA-256 缩写为 `b7a3…67f0`。
- Test：严格复用 Validation lock，重新初始化 TT1，使用 894,251 users / 2,499,357 examples / 0 dev，固定训练 1 epoch；final checkpoint SHA-256 缩写为 `48cf…4848`。
- 原 `*_best.pt`、epoch checkpoint 和 selection history 均保留；新 formal checkpoint 使用独立 `*_final_refit.pt`。
- `model_lineage.csv` 和 private lineage 以 `formal_full_refit` 角色逐层证明 final checkpoint → item/user embedding → ANN index → 32-part candidate 一致。正式 candidate generation 会拒绝 selection-only 或不完整 refit checkpoint。

## 6. 架构和未观察项

TT0 商品塔使用 article-ID embedding；TT1 在相同底座上加入 Stage 3C canonical 属性 embedding，并先做 taxonomy、appearance、merchandising/index 组内均值，再等权拼接。用户塔由 user-ID embedding 与同一商品基础表示的历史池化组成。两塔输出 64 维 L2-normalized vector；商品向量可独立离线预计算。

损失为 temperature=0.07 的 in-batch softmax。batch 其他 target 不是已知负反馈；重复正 target 和用户历史已见 target 被 mask，positive diagonal 始终保留。没有 popularity、uniform、hard 或 ANN negative 实验。

## 7. 修复前后核心结果

### Validation

| 指标 | 修复前 selection-only formal | 修复后 full-refit formal |
| --- | ---: | ---: |
| TT1 single Recall@500 | 0.14258878537105144 | 0.1445136853512506 |
| Q0 Recall@500 | 0.22050126975251422 | 0.22050126975251422 |
| TT1-T50 Recall@500 | 0.23571802048498094 | 0.2359142829954185 |
| TT1-T100 Recall@500 | 0.24442341573808332 | 0.2448639763673124 |
| TT1-T100 ΔRecall@500 vs Q0 | 0.02392214598556910 | 0.02436270661479820 |
| paired bootstrap 95% CI | [约 0.023154, 0.024719] | [0.023524638024452146, 0.025207862443796466] |

修复后 improved/declined users=12,179/3,339；gained/lost/net ground-truth hits=15,453/4,197/11,256。selection-only single 指标由保留的原 checkpoint 在 ignored tmp 中隔离恢复，只用于 before/after 审计，不是 formal candidate 来源。

### Test

| 指标 | 修复前 selection-only formal | 修复后 full-refit formal |
| --- | ---: | ---: |
| Q0 Recall@500 | 0.2146086294269453 | 0.2146086294269453 |
| TT1-T100 Recall@500 | 0.24090065174373818 | 0.24138900164539004 |
| TT1-T100 ΔRecall@500 vs Q0 | 0.02629202231679288 | 0.02678037221844473 |
| paired bootstrap 95% CI | [约 0.025416, 0.027163] | [0.025852566309520694, 0.027732177079706273] |

修复后 improved/declined users=11,422/2,840；gained/lost/net ground-truth hits=14,457/3,657/10,800。Test 只运行锁定 TT1-T100，没有运行 TT0、T50 或参数选择。

## 8. MAP@12 完整精度和 Top-12 序列

| Phase | Variant | MAP@12 完整值 |
| --- | --- | ---: |
| Validation | Q0 | 0.012184852577349603 |
| Validation | TT0-T50 | 0.012184760793980384 |
| Validation | TT1-T50 | 0.012184760793980384 |
| Validation | TT1-T100 | 0.012184760793980384 |
| Test | Q0 | 0.014657496029670704 |
| Test | TT1-T100 | 0.014657496029670704 |

Validation 三个 challenger 的 MAP@12 精确相同，但与 Q0 只是在六位小数上相同；Q0 高约 `9.1783369219e-08`。Test Q0 与 TT1-T100 的 MAP@12 精确相同。

基于既有候选的只读序列审计：Validation T50 和 T100 相对 Q0 都只有 1/129,705 个主评估用户的 Top-12 发生变化，占 `7.709803014532978e-06`，该用户有 11 个位置变化；Test T100 为 0/117,366，比例 0，changed-user 平均变化位置不适用。Two-Tower 的价值主要体现在深层 Recall@500；固定 source insertion 下 Top-12 几乎不变，不能把 Recall@500 提升表述为 MAP@12 同步提升。

## 9. ANN 审计

| Model / phase | mean overlap | median | p5 | ms/user | index bytes | retry rate | efSearch | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TT0 Validation selection | 0.999576 | 1.000000 | 0.998000 | 2.165808502002619 | 53,852,268 | 0.0 | 200 | PASS |
| TT1 Validation final-refit | 0.997159 | 0.998000 | 0.988000 | 1.5784595481818542 | 53,852,268 | 0.0 | 200 | PASS |
| locked TT1 Test final-refit | 0.9971019999999998 | 0.998000 | 0.988000 | 1.603393400553614 | 54,818,508 | 0.0 | 200 | PASS |

HNSW 固定 cosine、M=32、efConstruction=200；粗取后使用原始 64 维向量做 exact cosine rerank，并排除 seen items。没有重新调 ANN 参数。

## 10. 固定预算、overlap 和分群

Q0 继续精确复用冻结 IUF Union；T50/T100 都先按 source rank cap 截断，再合并 provenance 和去重，最后由 Popularity 唯一补齐到 500。不同来源原始分数未相加，cap 不冒充实际 unique occupancy。

Validation TT1-T100 的 short-history/warm/active ΔRecall@500 分别为 0.011558387799241347 / 0.01914351580840868 / 0.02786892022026946；Test 分别为 0.008433762264803522 / 0.020845989690334542 / 0.03102306991095895。cold-history 个性化 Two-Tower 候选按定义为 0，与主协同人口分开。

## 11. 资源、失败、warning 和恢复

当前 checkpoint group 汇总执行时间 27,408.5 秒，观测最大 RSS 2,500,907,008 bytes，最大 temp 2,984,771,584 bytes。Validation/Test final-refit 分别约 1,613.7/1,488.4 秒，峰值 RSS 约 1.62/1.61 GB，无 temp spill。`resource_summary.csv` 的 spill 是对 Stage 3D tmp 目录的保守观测，不能把每个标记都解释为该 group 独立产生 spill。

公开审计保留以下事实：

1. 首次 finalize 的 leakage audit 因 `UNPIVOT` SQL 语法失败；修正后 attempt 2 成功，没有重跑模型或候选。
2. 原 Test Union part 019 耗时 4,633.1609135 秒并观察到 spill；full-refit 后因候选输入变化必须重跑，本次约 18.29 秒，仍观察到 temp 占用。原长尾事实未删除。
3. PyTorch 对只读 NumPy view 发出 warning。触发 tensor 只用于 item embedding lookup/read，没有写操作；warning 本身不改变 output fingerprint。报告不声称 warning 不存在。
4. full-refit 后重跑 Validation/Test TT1 final-refit、embedding、ANN/index/audit、32 个 candidate 分区、single-route、锁定 Union 和下游聚合。只读复用 snapshot history、training target/sample、attributes、user features、Stage 3B Q0 source candidate、TT0 selection 证据和冻结选择。

详细事件见 `operational_event_audit.csv`，group 级执行与 intentional superseded-input rerun 见 `failure_retry_summary.csv` 和 `resource_summary.csv`。

## 12. Lock、fingerprint 和隐私

Validation lock 记录 selection checkpoint、final-refit checkpoint、embedding source、candidate source、final users/examples、locked epoch 和 ANN 参数。Test 使用该 lock 后才运行；最终 lock 另记录 Test final-refit lineage 和执行状态。`model_lineage.csv` 两行均 `lineage_passed=True`。

公开文件只包含聚合、状态和完整 SHA-256，不包含真实 customer/article/postal code、原始 ID 片段或本地绝对路径。raw 只读；private/tmp/checkpoints 均被 Git ignore 且未 tracked。

## 13. 真实限制

数据没有曝光、点击、真实负反馈、库存、可售状态、精确上架时间、日内时间戳、order_id 或真实 quantity。未购买不等于不喜欢；完整 articles 身份域不等于历史可售域；离线 Recall/MAP 不等于 CTR、CVR、收入或线上收益。Two-Tower 没有证明线上 cold-item 能力。

## 14. 最终结论和下一阶段建议

Stage 3D 已通过 Validation value gate，并在锁后 Test 上保持稳定正增量。建议下一阶段仅由新的正式 Prompt 授权 Stage 3E：将锁定 TT1-T100 纳入最终召回方案，冻结固定 500 候选契约、来源 provenance 与可重放候选，为后续 LightGBM Ranking 做准备。不补做 MF/BPR，不重新打开 Content，不增加独立粗排。本报告不授权自动开始 Stage 3E。

## 15. 面试边界

可以陈述 CPU 双塔、结构化商品塔、严格 snapshot 样本、in-batch false-negative mask、full-refit、HNSW+exact rerank、ANN overlap audit、固定预算 Union、paired bootstrap、checkpoint/lineage 和失败恢复。不能声称线上 A/B、CTR/CVR、业务收益、库存过滤、精确新品识别、排序/重排或在线服务。
