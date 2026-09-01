# Model card

## Model

- Base: `Qwen/Qwen3.5-4B`.
- Adaptation: LoRA, text-only.
- Production adapter: exact five-fold delta soup, rank 80 / alpha 160.
- Inputs: risk category, product name, product description.
- Outputs: yes/no score, thresholded verdict, post-hoc text.

## Intended use

Research and competition-grade ranking of Russian-language marketplace product
cards under two supplied policy categories. The output is not a fire-safety,
medical or legal certification and requires human review in consequential use.

## Evaluation

Historical public leaderboard: `0.9255540166204985` on the competition metric.
The repository reports per-category positive-class F1, PR-AUC, ROC-AUC and
confusion counts. OOF metrics must use grouped held-out predictions; final-soup
scores on training rows are diagnostic only.

## Limitations

- descriptions may be inaccurate or omit kit contents;
- images are ignored;
- rare FLAM positives make thresholds and F1 statistically unstable;
- product-family and seller-style shortcuts can transfer poorly;
- empirical gold may contain systematic inconsistencies;
- rationale is post-hoc and exact-span coverage is incomplete.

