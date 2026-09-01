"""Fits deployment thresholds on out-of-fold predictions and reports the CV score.

  python oof_thresholds.py --preds 'preds/cv_len1024_fold*.csv'

Why this and not the per-fold numbers in the training logs:

  - Легковоспламеняющиеся has ~29 positives in one validation fold and 142 across
    all five. A threshold fitted on 29 is unstable, and one misclassified item
    moves that category's F1 by roughly a point and a half.
  - Fold-to-fold spread is itself the answer to "is this improvement real or
    noise" — it is printed alongside the mean.
  - The transfer check measures what actually happens at submission time: a
    threshold fitted on one sample applied to a different one. If that costs a
    lot, the F1-optimal threshold is too sharp to trust and a flatter choice is
    safer.
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


GRID = np.linspace(0.05, 0.95, 91)


def f1_curve(y, p):
    """F1 at every threshold in GRID at once.

    Calling sklearn once per threshold per bootstrap resample is tens of
    thousands of calls; counting through sorted arrays gives the same numbers
    in one pass. For threshold t: TP is the positives at or above t, FP the
    negatives at or above t, and F1 = 2TP / (TP + P + FP).
    """
    y, p = np.asarray(y), np.asarray(p)
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    n_pos = len(pos)
    tp = n_pos - np.searchsorted(pos, GRID, side="left")
    fp = len(neg) - np.searchsorted(neg, GRID, side="left")
    denom = tp + n_pos + fp
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def best_threshold(y, p):
    """Plain argmax of F1 — optimal on this sample, and partly fitted to its noise."""
    curve = f1_curve(y, p)
    i = int(curve.argmax())
    return float(GRID[i]), float(curve[i])


def robust_threshold(y, p, n_boot=200, seed=0):
    """Threshold from a bootstrap-averaged F1 curve instead of a single one.

    The hope was that averaging over resamples would flatten spikes created by
    one lucky example while keeping the genuine plateau, giving a threshold that
    transfers better. Measured on simulation it does essentially nothing: fitting
    on ~28 positives and scoring on a large held-out sample, the transfer loss
    was 0.0263 for plain argmax against 0.0247 here, and at 40-57 positives the
    two were identical. Kept only so the real out-of-fold data can confirm or
    contradict that; do not expect a gain from it.

    What does reduce transfer loss is simply more positives to fit on — 0.026 at
    28, 0.017 at 40. Hence out-of-fold thresholds over all five folds.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    total = np.zeros_like(GRID)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:      # degenerate resample
            continue
        total += f1_curve(y[idx], p[idx])
    curve = total / n_boot
    i = int(curve.argmax())
    return float(GRID[i]), float(f1_score(y, (p >= GRID[i]).astype(int),
                                          zero_division=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds/cv_len1024_fold*.csv")
    ap.add_argument("--out", default="thresholds_oof.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.preds))
    if not files:
        raise SystemExit(f"no prediction files matched {args.preds}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(files)} folds, {len(df)} out-of-fold predictions\n")

    thresholds, cat_f1 = {}, {}
    for cat, sub in df.groupby("category"):
        t, f1 = best_threshold(sub["label"].values, sub["p_yes"].values)
        thresholds[cat] = round(t, 3)
        cat_f1[cat] = f1
        print(f"{cat:24s} threshold {t:.2f} | F1 {f1:.4f} | "
              f"{len(sub)} rows, {int(sub['label'].sum())} positives")

    macro = float(np.mean(list(cat_f1.values())))
    print(f"\nOOF macro F1 = {macro:.4f}")

    # Spread across folds, scored with the shared OOF thresholds.
    per_fold = []
    for fold, sub in df.groupby("fold"):
        f1s = [f1_score(g["label"], (g["p_yes"] >= thresholds[c]).astype(int),
                        zero_division=0)
               for c, g in sub.groupby("category")]
        per_fold.append(np.mean(f1s))
    print(f"per-fold: {' '.join(f'{v:.4f}' for v in per_fold)}")
    print(f"mean {np.mean(per_fold):.4f} ± {np.std(per_fold):.4f} "
          f"(std — anything smaller than this is noise, not an improvement)")

    # Transfer check: fit on one fold, score on the rest — the submission setting.
    print("\nthreshold transfer (fit on one fold, apply to the others):")
    print(f"  {'category':24s} {'argmax':>18s} {'bootstrap-averaged':>20s}")
    robust = {}
    for cat, sub in df.groupby("category"):
        plain, boot = [], []
        for fold in sorted(sub["fold"].unique()):
            fit, rest = sub[sub["fold"] == fold], sub[sub["fold"] != fold]
            ideal = best_threshold(rest["label"].values, rest["p_yes"].values)[1]
            for fn, acc in ((best_threshold, plain), (robust_threshold, boot)):
                t, _ = fn(fit["label"].values, fit["p_yes"].values)
                acc.append(ideal - f1_score(rest["label"],
                                            (rest["p_yes"] >= t).astype(int),
                                            zero_division=0))
        print(f"  {cat:24s} {np.mean(plain):9.4f} (max {max(plain):.3f}) "
              f"{np.mean(boot):9.4f} (max {max(boot):.3f})")
        rt, rf1 = robust_threshold(sub["label"].values, sub["p_yes"].values)
        robust[cat] = round(rt, 3)
    print("\n  Lower loss means the threshold survives a change of sample.")
    print(f"  robust thresholds on full OOF: {robust}")

    json.dump({"thresholds": thresholds, "thresholds_robust": robust,
               "oof_macro_f1": macro,
               "per_category_f1": cat_f1, "n_folds": len(files)},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
