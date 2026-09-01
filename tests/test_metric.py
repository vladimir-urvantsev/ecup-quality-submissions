from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_competition_metric import evaluate  # noqa: E402


def test_metric_averages_positive_f1_across_product_categories():
    rows = [
        {"id": 1, "category": "БАД", "label": 1, "fold": 0, "score": 0.9},
        {"id": 2, "category": "БАД", "label": 0, "fold": 0, "score": 0.1},
        {"id": 3, "category": "Легковоспламеняющиеся", "label": 1, "fold": 0, "score": 0.9},
        {"id": 4, "category": "Легковоспламеняющиеся", "label": 0, "fold": 0, "score": 0.1},
    ]
    report = evaluate(rows, {"БАД": 0.5, "Легковоспламеняющиеся": 0.5})
    assert report["contest_metric"] == 1.0
    assert report["contest_metric_definition"] == "mean_category(positive_f1)"

