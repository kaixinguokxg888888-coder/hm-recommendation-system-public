# Stage 4B-4 Phase 2 — Validation 14-day Evaluation and Selection

## 1. Status

`VALIDATION_SELECTION_LOCKED / PASSED_VALUE_GATE`

- Winner: `L1_600`
- Segment review: `NO_CRITICAL_SEGMENT_REGRESSION`
- Validation label access: one authorized 14-day session, 2020-08-26 through 2020-09-08.
- First7d, final refit, Test, reranking and retrieval retuning: not executed.

## 2. Pre-access freeze

Before opening Validation labels, the Phase 2 configuration, exact metric formulas, population and denominator rules, slices, seven-version budget, 1000-iteration paired user-block bootstrap, seed, winner rule and value gate were preregistered. The pre-access audit then revalidated the immutable Phase 1 lock, G1-G20, all 32 prediction SUCCESS checkpoint identities, physical outputs, model/matrix/feature lineage and seven global prediction Merkle fingerprints.

Preregistration fingerprint: `1b248fbc3fd407a605bb0d4e81fc35686b25fb26838a921d8974402aaddf8935`.

## 3. Main Validation results

The main population contains 129705 cutoff-history-nonempty label users. All 67409 full500 retrieval-miss queries remain in the denominator with all four query metrics equal to zero.

| Version | MAP@12 | NDCG@12 | Recall@12 | HitRate@12 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.012184761 | 0.019393515 | 0.028460063 | 0.062179561 |
| L0_100 | 0.023283413 | 0.037708269 | 0.054328963 | 0.124497899 |
| L0_300 | 0.023473998 | 0.037812457 | 0.053993036 | 0.124258895 |
| L0_600 | 0.023392434 | 0.037655206 | 0.053718852 | 0.123711499 |
| L1_100 | 0.024380922 | 0.039331319 | 0.056705781 | 0.128769130 |
| L1_300 | 0.025097869 | 0.040921017 | 0.059600927 | 0.135862149 |
| L1_600 | 0.025257775 | 0.041181950 | 0.059937731 | 0.136756486 |

`L1_600` wins the preregistered exact lexicographic comparison. Its MAP@12 delta over B0 is `+0.013073014`, a relative increase of about `107.29%` in this offline Validation evaluation.

## 4. Bootstrap and segment evidence

The paired user-block percentile bootstrap uses 1000 iterations, seed 20260826 and 95% confidence intervals. The `L1_600 - B0` main MAP@12 interval is `[0.012656809, 0.013525050]`, so its lower bound is strictly positive and the value gate passes.

Short-history, warm, active, one-positive, multi-positive and users-with-cold-positive slices all have strictly positive lower bounds. The separately reported cold-history population has delta `+0.000313914` and interval `[-0.000097523, 0.000744123]`; it is inconclusive, not a significant regression because the upper bound is not below zero. No critical slice triggered the preregistered regression review rule.

Cold-item target metrics remain zero because the frozen main candidate universe is history-seen; this is a data/protocol boundary and was not changed for ranking selection.

## 5. Fingerprints, replay and recovery

Each version has 32 query-metric partitions and 32 digest partitions, 141359 queries, a physical fingerprint and a global query-metric Merkle fingerprint in the immutable selection lock. The aggregate metric and bootstrap artifacts have separate physical and semantic fingerprints. Fixed partition 0 formal/replay identity passes for all seven versions; the bootstrap replay is an exact 13-row match.

The first execution completed all 480 query-base, metric and digest checkpoints, then stopped in the post-metric bootstrap wrapper because a NumPy array used scalar boolean emptiness. The recovery-only correction changed that check to an explicit length test, did not alter metric or selection semantics, reused all 480 successful checkpoints and did not reopen Validation labels. A second housekeeping correction restored the shared Stage 3A evaluator to its frozen source fingerprint; Phase 2 NDCG remains in the preregistered Phase 2 module and no metric output changed.

## 6. Gates and engineering audit

- Phase 1 G1-G20: 20/20 PASS and unchanged.
- Phase 2 P2G1-P2G13: 13/13 PASS.
- Final independent audit: 24/24 PASS.
- Targeted tests: 71 passed; full test suite: 361 passed; compile audit: PASS.
- Peak RSS: 601559040 bytes, below the preregistered 1683436339-byte hard limit; no spill observed.
- Privacy/path scan, `git diff --check`, raw-data diff, private/tmp/checkpoint ignore, tracked-private artifact and Validation/First7d/Test access audits: PASS.

## 7. Explicitly not executed

No First7d diagnostic, final refit, Test candidate/feature/label/metric access, Test authorization, reranking, retrieval retuning, commit or push was performed. The result is offline Validation evidence and does not imply CTR/CVR, revenue or online A/B-test improvement.
