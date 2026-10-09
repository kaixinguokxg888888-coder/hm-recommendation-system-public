# Stage 5 Validation Completion Report

Status: STAGE_5E_VALIDATION_CONFIG_FROZEN; final technical acceptance pending. No commit/push or Test execution.

## A. Implementation

New src/reranking package, scripts/run_stage_5_reranking.py, Stage 5 configs, tests, protocol, independent public/private artifacts; only .gitignore is modified upstream of these new files. No Stage 1–4 file is changed.

## B. Algorithm

Greedy scans frozen predicted_rank ASC. Article uniqueness never relaxes. Product cap=2, garment cap=4 where configured; garment relaxes to 5/6, product to 3/4, then original-order unique fill. Original relative order is retained. R0 bypasses greedy; R1 is an exact pipeline control, not a quality gain. Tail replacement requires a complete level-0, no-natural-tail list and a legal strict-cap swap within preregistered Top-20/30/50.

## C. Metadata

Product projection joins the existing article key map to read-only articles; garment codes reuse Stage 4B-3 canonical attributes. Support uses Validation-only cutoff 2020-08-25 aggregates. popularity_thresholds and popularity_bucket are called directly over all history-seen items. Missing support for dimension-only items becomes zero_history; missing support for formal candidates fails. No label-derived item-state flag is consumed by the algorithm. Static attribute effective timestamps remain unverifiable.

Stage 5 tail semantics are identical to the existing project-level history-safe item-state definition.

## D. Ablation

Main population: 129705; cold-history diagnostic: 11654; all 67409 main retrieval-miss/all-zero queries remain. Each variant outputs 1696308 rows for 141359 queries. Table uses main-population means; violations always measure original caps 2/4 even for uncapped/relaxed variants.

| Variant | MAP@12 | NDCG@12 | Recall@12 | HitRate@12 | Unique product | Unique garment | Product violation | Garment violation | Changed users | Relaxation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.025257775 | 0.041181950 | 0.059937731 | 0.136756486 | 8.815211 | 3.808334 | 0.436884 | 0.882071 | 0.000000 | 0.000000 |
| R1 | 0.025257775 | 0.041181950 | 0.059937731 | 0.136756486 | 8.815211 | 3.808334 | 0.436884 | 0.882071 | 0.000000 | 0.000000 |
| R2 | 0.024713627 | 0.040131911 | 0.057564115 | 0.134181412 | 9.334737 | 3.973579 | 0.000000 | 0.866613 | 0.436884 | 0.000000 |
| R3 | 0.024208732 | 0.038907332 | 0.054775869 | 0.129316526 | 10.166647 | 5.455457 | 0.000000 | 0.000000 | 0.907706 | 0.000000 |
| R4_N20 | 0.024206394 | 0.038900470 | 0.054759781 | 0.129262557 | 10.169778 | 5.454778 | 0.000000 | 0.000000 | 0.916788 | 0.000000 |
| R4_N30 | 0.024204985 | 0.038898609 | 0.054750443 | 0.129301106 | 10.177187 | 5.452558 | 0.000000 | 0.000000 | 0.923835 | 0.000000 |
| R4_N50 | 0.024200417 | 0.038885235 | 0.054697027 | 0.129293397 | 10.189083 | 5.452851 | 0.000000 | 0.000000 | 0.932740 | 0.000000 |

Full diversity/intervention/relaxation/exploration and segment results are in validation_diagnostics.csv. Paired main-population four-metric deltas and 95% CIs are in validation_paired_comparison.csv. Exploration eligible/receiving subgroup comparisons are in tail_exploration_subgroup.csv. CIs are marginal, not multiplicity-adjusted.

## E. Final Selection

Frozen variant: **R1**. The immutable preregistered stepwise checks are:

```json
[
  {
    "eligible": false,
    "relevance_guard": false,
    "variant": "R2"
  },
  {
    "eligible": false,
    "relevance_guard": false,
    "variant": "R3"
  },
  {
    "eligible": false,
    "extra_exploration_guard": true,
    "relevance_guard": false,
    "variant": "R4_N20"
  },
  {
    "eligible": false,
    "extra_exploration_guard": true,
    "relevance_guard": false,
    "variant": "R4_N30"
  },
  {
    "eligible": false,
    "extra_exploration_guard": false,
    "relevance_guard": false,
    "variant": "R4_N50"
  }
]
```

0.5% is a preregistered engineering tolerance budget, not a statistically derived universal threshold. Both point delta and paired CI lower bound must remain within MAP/NDCG budgets. No challenger, threshold or rank window was added after results. A failed complex variant never changes the upstream ranker or retrieval.

## F. Trade-off

- map_at_12: +0.000000000000 (+0.000000% relative).
- ndcg_at_12: +0.000000000000 (+0.000000% relative).
- recall_at_12: +0.000000000000 (+0.000000% relative).
- hit_rate_at_12: +0.000000000000 (+0.000000% relative).
- after_unique_product: +0.000000000.
- after_unique_garment: +0.000000000.
- after_product_violation: +0.000000000.
- after_garment_violation: +0.000000000.

## G. Exploration

Final tail exploration enabled: False. The 0.01 tail-count gain and additional 0.1% R3 relevance budget were evaluated without widening any rank window. Eligibility means a tail exists within the guard; receiving means an actual strict-cap replacement. Natural tail exposure is reported separately. See selection checks for the reason an exploration variant was not adopted.

## H. Engineering

One shared formal full500 scan generates all seven variants; a separate 32-query label-free sanity precedes it. 32 query-contained partitions and 64000-row batches; no full500 pandas. Metadata is small and materialized once. Fixed-pass greedy processing is O(500) per user; final sorting only covers at most 12 items.

Recorded completed worker wall time: 2409.635 seconds. Peak recorded worker process-tree RSS: 922042368 bytes; frozen cap: 3210988748 bytes. Private artifact bytes: 297021695. Final freeze worker resource measurement is appended after this report, in freeze_resource_audit.json; it is not included in these earlier completed-worker totals.

Recovered worker failures: 1. Initial targeted collection failed because the editable package registry omitted the new reranking package; a local offline no-dependency editable reinstall fixed collection before ablation. Resource audit preserves the failed attempt. A CLI import failure before monitoring was also corrected using the repository's existing script src-path convention. No experiment settings or results were changed.

Formal producer source: 30e59f21fa27888cb0b02b6168236fda07a584736fa040aa1d366a01ae0e1edd. Final audit source: e20f64701c1652dbee8c727c5957eccb1b8cd5b05b0642061bbf3e3a9f092623. After all ablation results and selection were saved, an audit-only correction distinguished typed RSS byte counts from raw ID strings and removed duplicate pytest quiet flags so test counts are recorded. Only runner audit/tests/report/freeze methods changed. All remaining producer AST, other producer files, frozen metrics/CI/selection and checkpoint output bytes were verified identical; no Validation evaluation/bootstrap was repeated. Evidence: audit_source_correction.json. Its temporary audit receipt utility needed a Path-type correction before recording; no experiment was executed by that utility.

## I. Audit

Targeted pytest: {'passed': 43}. Full pytest: {'passed': 514, 'skipped': 1}. Protected files checked before/after: 359; mutations: 0. R1 membership/order/all-four-metric identity: PASS. Output schema, uniqueness, continuous Top-12, full500 accounting, all-zero retention, checkpoint/fingerprint, Git ignore and public privacy/path scan: PASS.

## J. Test Protocol Conflict

Test labels read: NO. Test formal evaluation: NO. Test bootstrap: NO. No Stage 5 Test result directory exists. Existing Test predictions/digests were hashed solely for upstream protection. Synthetic Test-schema pytest fixtures contain invented internal keys. AGENTS.md forbids reopening Test labels, recomputing formal Test metrics/bootstrap or creating a second formal Test evaluation. A future assessment would require an independent authorized protocol amendment explicitly defining supplementary evaluation and preserving Stage 4B-4 evidence; no amendment or exception is requested here.

## Limitations and interview explanation

This is offline list-constraint engineering, not CTR/CVR, revenue, inventory, exposure or online A/B evidence. True zero-history items are absent from the frozen candidate domain; tail means history-seen low support. Validation has already served ranking selection and training-refit supervision, so downstream tuning is further reuse of that development set; Stage 5 has no independent Test evidence.

The ranker scores 500 frozen candidates per user. My reranker keeps that ranking unless product/category caps require replacements, scans deterministically, and relaxes constraints only to fill the list. I reused history-safe item support and canonical metadata, and tested tail exploration behind a rank-relative guard. I preregistered a small relevance-loss budget, compared every rule on paired Validation users, and adopted only a qualifying configuration. If diversity rules exceed that budget I retain the original list. Inputs, outputs and checkpoints are fingerprinted, and Test remains untouched.

STOP after Stage 5E. Final technical acceptance is pending; README/AGENTS completion facts remain unchanged. Suggested future Git split, only after explicit approval: feat (code/tests/config), data (public Validation aggregates/audits), docs (protocol/report).
