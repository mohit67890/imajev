# imajev-bench v2 pre-registered analysis

This plan is frozen before any model, including imajev, is run on the v2 calibration or test split.
It may be changed only by a dated amendment below, made before test-split scoring.
Status: **draft**. It is frozen when the external reviewer signs it.

## Primary outcomes

1. **Track score.** For each of the text, visual and joint tracks: 100 × the unweighted mean of
   family accuracies on the test split.
   - Unknown counts as correct only when the reference label is Unknown.
   - An error or an unparseable reply scores 0.
2. **Overall score.** The unweighted mean of the three track scores, reported only when all three
   are complete.
3. **Contrast score.** The share of contrast sets in which every member is correct.

## Secondary outcomes

- **Abstention:**
  - Correct-Unknown rate on Unknown items.
  - False-abstention rate on answerable items.
  - Unknown F1.
- **Change-pair accuracy.** Also report the best score an image-blind system could get by repeating
  one answer per set.
- **Image necessity.** Full minus no-image accuracy on answerable visual and joint items. Reported
  for open models only, as a diagnostic.
- **Stability.** Rotation agreement, for direct option scoring only.
- **Probability quality.** Brier score, negative log-likelihood and 10-bin ECE, for submissions that
  include probability distributions. Temperatures are fitted only on the calibration split.
- **Selective risk.** Risk–coverage curves and AURC, with the thresholds fixed on the calibration
  split.
- **Efficiency.** Latency and memory under the timing protocol in plan §8. Contaminated cases are
  excluded, and the number excluded is reported.

## Statistical methods

- **Unit of analysis.** The evidence cluster, computed by `stats.evidence_clusters`. It joins records
  that share an image, a source scene, a group or a contrast set.
- **Intervals.** 95% percentile bootstrap over clusters, with 4,000 resamples and seed 0. No
  interval is reported for fewer than 10 clusters.
- **Pairwise comparisons.** Paired cluster sign-flip test (exact up to 16 discordant clusters,
  otherwise 10,000 Monte Carlo draws, seed 0).
  - Leaderboard neighbours with p ≥ 0.05 share a tie group.
  - No correction is applied across neighbour tests, because tie groups are descriptive.
  - Claims about named model pairs use Holm correction across all claims made in the paper.
- **Interfaces kept apart.** Direct option scoring and structured generation are ranked separately
  and never pooled.
- **Strata.** Results for commissioned photos, public-source photos and synthetic renders are
  reported separately, and judgement-dependent items are also reported separately.

## Hypotheses registered in advance (imajev-specific)

- **H1.** imajev 1.1 9B differs from base Qwen3.5-9B on the overall score. Two-sided, paired cluster
  test.
- **H2.** imajev 1.1 2B differs from base Qwen3.5-2B on the overall score. Two-sided, paired cluster
  test.
- **H3.** On the contrast score, 9B models beat their 2B counterparts. One-sided.

Only H1–H3 may be described as confirmatory. Every other comparison is exploratory and is labelled
as such.

## Exclusions

- **Items.** An item is removed from scoring only if it is quarantined, or found to have a wrong
  label before test scoring. Every removal is listed in the errata with its reason. No item may be
  removed after test results have been seen, except through a published label correction that
  applies to every model equally.
- **Runs.** A model run is excluded only for an infrastructure failure. The failure must be
  documented, and the model is rerun in full.

## Amendments

(none)
