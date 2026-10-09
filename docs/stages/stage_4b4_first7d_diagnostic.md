# Stage 4B-4 Phase 3 — First7d Locked-Winner Robustness Diagnostic

## Status

`FIRST7D_LOCKED_WINNER_DIAGNOSTIC_COMPLETE`

- Frozen winner: `L1_600` (`F1`, iteration 600); winner changed: `false`.
- Diagnostic window: 2020-08-26 through 2020-09-01, reusing `validation_14d` request/candidates/features/predictions.
- B0 MAP@12: 0.015072609291607419
- L1_600 MAP@12: 0.031448630295311433
- Delta MAP@12: 0.016376021003704014
- Direction: `ROBUSTNESS_DIRECTION_PASS`
- Confidence: `POSITIVE_AND_SIGNIFICANT`
- Segment review: `NO_FIRST7D_CRITICAL_SEGMENT_REGRESSION`

First7d was used only once after the 14-day winner lock. It did not select or change a model and did not authorize Test. Final refit, Test, reranking and retrieval retuning were not executed. P3G1–P3G13 are independent contract checks and all passed.

## Post-First7d Resource Contract Recovery

`FIRST7D_RESOURCE_CONTRACT_RECOVERED`

The original Phase 3 lock and P3 gate are preserved as historical evidence but are `SUPERSEDED / NOT_FORMAL_DUE_TO_RESOURCE_CONTRACT_IDENTITY_FAILURE`. The original config registered 65% (`1207145472` bytes), while the authorized formal contract is 70% of the original preregistration available memory: `1857146880 * 0.70 = 1300002816` bytes. The runtime-recomputed `1315638272` bytes is retained only as invalid historical evidence.

Observed peak RSS was `575967232` bytes (`0.44305075720697518` of the corrected cap), so the corrected resource gate passes. Recovery did not reopen First7d labels, rematerialize ground truth/population, recompute metrics/bootstrap/predictions, change the winner, or execute final-refit/Test/reranking/retrieval retuning.

- Recovered lock fingerprint: `f3af337e862b27d36e9a29968c7071482411329c1de8dc9918c80b3398c5bcb7`
- Supersession fingerprint: `2da9055bbb66364a9b32997c1187d00b2e37d9d9d74bd0db8e4423fe65c19aaf`
- Winner: `L1_600`; direction: `ROBUSTNESS_DIRECTION_PASS`; confidence: `POSITIVE_AND_SIGNIFICANT`.
- First7d total label read count: `1`; recovery reopen count: `0`.
- Recovered R1-R14 contract gate: `PASSED`.
