"""Evaluate ECUP predictions with the official category-balanced metric.

The task page says (in Russian): "an F1 measure is calculated for both
categories, after which the two results are averaged", and points to
``sklearn.metrics.f1_score`` without an ``average=`` override.  The official
baseline follows that wording and calls ``f1_score(y, pred)`` separately in
each product category.  Therefore the primary metric in this module is:

    mean_product_category( F1(positive class, default sklearn semantics) )

The frequently confused alternative
``mean_product_category(mean_label_class(F1_0, F1_1))`` is still reported as a
named diagnostic.  It is *not* called the contest metric here.  Pooling the two
product categories is wrong under either reading because their distributions
are intentionally very different.

The script has no third-party dependencies so it can also be used in the
submission image and on a fresh laptop.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict


CONTEST_CATEGORIES = {"БАД", "Легковоспламеняющиеся"}


def _confusion(labels, scores, threshold):
    tp = fp = fn = tn = 0
    for y, score in zip(labels, scores):
        pred = int(score >= threshold)
        if y and pred:
            tp += 1
        elif not y and pred:
            fp += 1
        elif y and not pred:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _f1(num, left, right):
    denominator = 2 * num + left + right
    return (2 * num / denominator) if denominator else 0.0


def binary_macro_f1(labels, scores, threshold):
    tp, fp, fn, tn = _confusion(labels, scores, threshold)
    return 0.5 * (_f1(tp, fp, fn) + _f1(tn, fn, fp))


def positive_f1(labels, scores, threshold):
    tp, fp, fn, _ = _confusion(labels, scores, threshold)
    return _f1(tp, fp, fn)


def best_threshold(labels, scores, metric="positive_f1"):
    """Exact threshold sweep for a named metric.

    ``positive_f1`` is the official objective.  ``binary_macro_f1`` exists for
    sensitivity analysis only.  Predictions use ``score >= threshold``.
    """

    candidates = sorted(set(float(value) for value in scores), reverse=True)
    if not candidates:
        raise ValueError("cannot tune a threshold on an empty sample")
    scorers = {
        "positive_f1": positive_f1,
        "binary_macro_f1": binary_macro_f1,
    }
    if metric not in scorers:
        raise ValueError(f"unknown threshold metric {metric!r}")
    scorer = scorers[metric]
    candidates.insert(0, math.nextafter(candidates[0], math.inf))
    return max(
        ((scorer(labels, scores, threshold), threshold)
         for threshold in candidates),
        key=lambda pair: (pair[0], -abs(pair[1] - 0.5)),
    )[::-1]


def average_precision(labels, scores):
    """Average precision with exact tie blocks, matching sklearn semantics."""

    positives = sum(labels)
    if positives == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    previous_recall = ap = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        stop = index
        block_pos = 0
        while stop < len(pairs) and pairs[stop][0] == score:
            block_pos += int(pairs[stop][1])
            stop += 1
        tp += block_pos
        fp += stop - index - block_pos
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        index = stop
    return ap


def roc_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(zip(scores, labels))
    rank = 1
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        average_rank = 0.5 * ((rank) + (rank + stop - index - 1))
        positive_rank_sum += average_rank * sum(y for _, y in ordered[index:stop])
        rank += stop - index
        index = stop
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def read_rows(patterns, score_column):
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches or [pattern])
    rows = []
    seen = set()
    for path in files:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                key = (int(raw["id"]), str(raw["category"]))
                if key in seen:
                    raise ValueError(f"duplicate prediction key {key} across inputs")
                seen.add(key)
                rows.append({
                    "id": key[0],
                    "category": key[1],
                    "label": int(raw["label"]),
                    "fold": int(raw["fold"]),
                    "score": float(raw[score_column]),
                })
    if not rows:
        raise ValueError("no prediction rows loaded")
    return rows, files


def parse_thresholds(values):
    result = {}
    for value in values or []:
        category, threshold = value.split("=", 1)
        result[category] = float(threshold)
    return result


def evaluate(rows, thresholds=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    present_categories = set(grouped)
    extra_categories = present_categories - CONTEST_CATEGORIES
    if not present_categories or extra_categories:
        raise ValueError(
            "invalid contest categories: "
            f"present={sorted(present_categories)}, extra={sorted(extra_categories)}"
        )
    result = {"rows": len(rows), "categories": {}, "threshold_source": {}}
    official_parts = []
    binary_macro_parts = []
    for category in sorted(grouped):
        category_rows = grouped[category]
        labels = [row["label"] for row in category_rows]
        scores = [row["score"] for row in category_rows]
        if thresholds and category in thresholds:
            threshold = thresholds[category]
            source = "fixed"
        else:
            threshold, _ = best_threshold(labels, scores, metric="positive_f1")
            source = "same_rows_optimistic"
        tp, fp, fn, tn = _confusion(labels, scores, threshold)
        category_macro = binary_macro_f1(labels, scores, threshold)
        category_positive = positive_f1(labels, scores, threshold)
        official_parts.append(category_positive)
        binary_macro_parts.append(category_macro)
        result["threshold_source"][category] = source
        result["categories"][category] = {
            "rows": len(category_rows),
            "positives": sum(labels),
            "threshold": threshold,
            "binary_macro_f1": category_macro,
            "positive_f1": category_positive,
            "pr_auc": average_precision(labels, scores),
            "roc_auc": roc_auc(labels, scores),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    available_official_mean = sum(official_parts) / len(official_parts)
    available_binary_macro_mean = sum(binary_macro_parts) / len(binary_macro_parts)
    contest_available = present_categories == CONTEST_CATEGORIES
    result["metric_scope"] = {
        "contest_available": contest_available,
        "available_categories": sorted(present_categories),
        "missing_categories": sorted(CONTEST_CATEGORIES - present_categories),
    }
    result["contest_metric"] = (
        available_official_mean if contest_available else None
    )
    result["contest_metric_definition"] = "mean_category(positive_f1)"
    result["contest_metric_positive_f1"] = result["contest_metric"]
    result["available_category_mean_positive_f1"] = available_official_mean
    result["diagnostic_mean_category_binary_macro_f1"] = (
        available_binary_macro_mean if contest_available else None
    )
    result["diagnostic_available_category_mean_binary_macro_f1"] = (
        available_binary_macro_mean
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", help="CSV paths or glob patterns")
    parser.add_argument("--score_column", default="p_yes")
    parser.add_argument(
        "--threshold", action="append", default=[], metavar="CATEGORY=VALUE",
        help="repeat for fixed per-category thresholds; omitted categories are tuned",
    )
    args = parser.parse_args()
    rows, files = read_rows(args.predictions, args.score_column)
    report = evaluate(rows, parse_thresholds(args.threshold))
    report["files"] = files
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
