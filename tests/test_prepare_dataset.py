from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prepare_dataset as module  # noqa: E402


def _official_rows():
    rows = []
    product_id = 1
    for category in sorted(module.CATEGORIES):
        for family in range(10):
            for label in (0, 1):
                rows.append(
                    {
                        "id": product_id,
                        "name": f"{category} семейство {family} вариант {label}",
                        "description": f"описание {family} {label}",
                        "category": category,
                        "label": label,
                    }
                )
                product_id += 1
    return rows


def test_builder_is_deterministic_and_group_safe(tmp_path):
    official = tmp_path / "official.csv"
    curated = tmp_path / "curated.csv"
    pd.DataFrame(_official_rows()).to_csv(official, index=False)
    pd.DataFrame(
        [
            {
                "id": 1000,
                "name": "Газ в зажигалке отдельный набор",
                "description": "контрастный пример",
                "category": "Легковоспламеняющиеся",
                "label": 1,
            }
        ]
    ).to_csv(curated, index=False)
    first, first_manifest = module.build(official, curated)
    second, second_manifest = module.build(official, curated)
    assert first == second
    assert first_manifest == second_manifest
    assert all(value == 0 for value in first_manifest["official_family_leaks"].values())
    assert {row["fold"] for row in first if row["source_partition"] == "official"} == set(range(5))
    assert next(row for row in first if row["id"] == 1000)["fold"] == -1


def test_builder_rejects_cross_source_id_collision(tmp_path):
    official = tmp_path / "official.csv"
    curated = tmp_path / "curated.csv"
    rows = _official_rows()
    pd.DataFrame(rows).to_csv(official, index=False)
    pd.DataFrame([rows[0]]).to_csv(curated, index=False)
    try:
        module.build(official, curated)
    except ValueError as error:
        assert "id overlap" in str(error)
    else:
        raise AssertionError("collision was accepted")

