# Stage 5 Validation-only Re-ranking Protocol

Status: PREREGISTERED_FOR_STAGE_5B_THROUGH_5E. Design accepted by user; Stage 5 final technical acceptance pending.

## Scope and input

Only frozen L1_600 validation_14d full500 predictions (as-of 2020-08-25) are consumed. No model training, retrieval rebuild, final-refit Validation scoring, Test labels/evaluation/bootstrap, upstream writes, commit or push. Real Test prediction files may only be hashed for immutable upstream protection. Future Test schema compatibility uses synthetic pytest fixtures. No Stage 5 Test result directory is created.

## Metadata and tail

Product metadata is projected once through the existing internal article key map. Garment codes reuse the frozen Stage 4B-3 canonical encoding. Support reuses the snapshot-specific Validation history aggregate. Thresholds call data_foundation.item_state.popularity_thresholds on every history-seen item, with its nearest-rank P50/P90 semantics. Every item bucket calls popularity_bucket directly: <=0 zero_history, >=P90 head, >=P50 mid, otherwise tail. Missing support means zero only for a dimension item absent from the complete history aggregate; missing/zero support for a formal history-seen candidate fails the gate. Missing product/garment metadata also fails the formal gate. Synthetic missing attributes are treated as unconstrained, never as a shared unknown product/category.

Stage 5 tail semantics are identical to the existing project-level history-safe item-state definition.

Static article attributes lack reliable effective timestamps; historical availability cannot be proven. No future-label item-state flag is an algorithm input.

## Algorithm

Input order is predicted_rank ASC, validated against raw_score DESC, candidate_rank ASC, article_key ASC. R0 directly extracts the original Top-12 without greedy. R1 is a pipeline control: membership, order, and all four metrics must exactly equal R0; any mismatch stops ablation.

Greedy retains article uniqueness forever. Strict product cap=2 and garment cap=4 apply where configured. If incomplete, preserve selected items and scan again with garment caps 5 then 6, then product caps 3 then 4, then unique-only final fill. Reasons and levels remain attached to selected items; output is sorted by original predicted_rank. At least 12 unique candidates implies 12 output items. No-conflict, exploration-off lists are exact identities.

Tail exploration is attempted only on complete level-0 lists with no natural tail, at most one insertion. The first unselected tail within original Top-N (20/30/50) may replace the final list item only if strict caps remain satisfied. The result is sorted by original rank. Ineligible users keep their constrained list. Exploration is not forced; natural tail items are not limited to one.

## Fixed experiment and selection

Exactly R0, R1, R2, R3, R4_N20, R4_N30, R4_N50. No additional challengers or parameter changes after results. R0 loss budget for both MAP@12 and NDCG@12 is 0.5% relative to R0. Both point delta and paired 95% CI lower bound must be >= negative budget.

0.5% is a preregistered engineering tolerance budget, not a statistically derived universal threshold.

R2 also requires strictly lower product violation rate and strictly higher average unique products than R0. If R2 fails, freeze R1. R3 must pass the R0 relevance guard, reduce garment violations versus R2, and gain >=0.10 average unique garment/query versus R2. R4 is considered only if R3 qualifies; it must pass the R0 guard, gain >=0.01 average tail/query versus R3, and pass an additional 0.1% relative R3 MAP/NDCG point+CI guard. Choose the smallest passing rank window; otherwise retain R3/R2/R1. These diversity thresholds and rank windows are immutable after preregistration. Report all results even when no constraint variant qualifies.

Main selection population is all 129705 evaluable warm users, including all-zero queries. Cold-history and segments are separately reported. Formula calls existing binary_top12_metrics: AP denominator=min(all future positives,12), recall denominator=all future positives; candidate misses remain in denominators. Paired bootstrap reuses the existing helper, 1000 iterations, seed 20260826, 95% percentile CI. Comparisons use sorted identical customer keys. CIs are marginal, not simultaneous multiple-comparison guarantees.

## Audit and stop

Outputs are private, Git ignored, partitioned, checkpointed, atomic, and bound to snapshot/cutoff/config/source/upstream fingerprints. Resources use a preregistered absolute RSS cap equal to 70% of available memory; the cap is not recomputed at runtime. Successful outputs are reused only after identity and byte verification; failures are disclosed and recovered independently.

Before/after protection covers all Validation L1_600 predictions, final model, frozen Test predictions/digests, Stage 4B-4 public frozen artifacts/locks/configs, and metadata/Validation supervision inputs. Required tests, R1 identity, metadata join, full500, schema, rank/score integrity, output contract, fingerprints, access boundary and resources must pass. Stage 5 stops after Validation freeze. AGENTS.md prohibits reopening Test labels/recomputing formal Test metrics/bootstrap/creating a second formal Test evaluation. Any future Stage 5 Test assessment requires a separately authorized protocol amendment preserving all Stage 4B-4 evidence; no exception is requested or executed here.
