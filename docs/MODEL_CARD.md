# Model card

## Model

- Base: `Qwen/Qwen3.5-4B`.
- Adaptation: LoRA, text-only.
- Production adapter: exact six-delta soup, rank 96 / alpha 192.
- Inputs: risk category, product name, product description.
- Outputs: probability-like yes/no score, thresholded verdict, post-hoc text.

## Intended use

Research and competition-grade ranking of Russian-language marketplace product
cards under the two supplied policy categories. The output is not a fire-safety,
medical or legal certification and requires human review in consequential use.

## Evaluation

Historical public leaderboard: `0.9261642156862745` on the competition metric.
The repository also reports per-category positive-class F1, PR-AUC, ROC-AUC and
confusion counts. OOF metrics must use grouped held-out predictions; scores from
the final soup on its own training rows are diagnostic only.

## Limitations

- descriptions may be inaccurate or omit kit contents;
- images are ignored;
- rare FLAM positives make thresholds and F1 statistically unstable;
- product-family and seller-style shortcuts can transfer poorly;
- empirical gold may contain systematic inconsistencies;
- rationale is post-hoc and has limited exact-span coverage.

