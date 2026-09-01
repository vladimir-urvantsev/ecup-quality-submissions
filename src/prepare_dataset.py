#!/usr/bin/env python3
"""Build grouped, deduplicated text-training JSONL from reviewed CSV inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

CATEGORIES = {"БАД", "Легковоспламеняющиеся"}
REQUIRED = ("id", "name", "description", "category", "label")
FOLDS = 5
SEED = 42
PREFIX_WORDS = 6
FROZEN_SHA256 = "aeb8fa2bebbe7bbba011d6a7f27c3b506d5c8f42894aed45f910101c0e39df04"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: object) -> str:
    text = str(value or "").lower().replace("ё", "е")
    return " ".join(re.sub(r"[^a-zа-я0-9 ]+", " ", text).split())


def short_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def validate_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [column for column in REQUIRED if column not in frame]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")
    frame = frame.loc[:, REQUIRED].copy()
    if frame["id"].isna().any():
        raise ValueError(f"{source}: null id")
    frame["id"] = frame["id"].astype(int)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    frame["label"] = frame["label"].astype(int)
    unknown = set(frame["category"]) - CATEGORIES
    if unknown:
        raise ValueError(f"{source}: unsupported categories {sorted(unknown)}")
    if not set(frame["label"]) <= {0, 1}:
        raise ValueError(f"{source}: labels must be 0/1")
    if frame["id"].duplicated().any():
        ids = frame.loc[frame["id"].duplicated(False), "id"].tolist()[:20]
        raise ValueError(f"{source}: duplicate ids {ids}")
    return frame


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    normalized_name = frame["name"].map(normalize)
    normalized_description = frame["description"].map(normalize)
    frame["dup_key"] = (normalized_name + "\0" + normalized_description).map(short_hash)
    frame["split_group"] = normalized_name.map(
        lambda value: short_hash(" ".join(value.split()[:PREFIX_WORDS]))
    )
    return frame


def assign_official_folds(frame: pd.DataFrame) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    frame = frame.copy()
    exact = frame.groupby("dup_key")["label"].agg(["mean", "count"])
    frame = frame.join(exact, on="dup_key")
    frame["majority_label"] = (frame["mean"] >= 0.5).astype(int)
    frame["fold"] = -1
    for category in sorted(CATEGORIES):
        subset = frame[frame["category"] == category]
        splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for fold, (_, validation_indices) in enumerate(
            splitter.split(subset, subset["majority_label"], groups=subset["split_group"])
        ):
            frame.loc[subset.index[validation_indices], "fold"] = fold
    if (frame["fold"] < 0).any():
        raise AssertionError("official fold assignment incomplete")
    if (frame.groupby("split_group")["fold"].nunique() > 1).any():
        raise AssertionError("official product family crosses folds")
    return frame.drop(columns=["mean", "count"])


def rule_prediction(row: pd.Series) -> int:
    from rules_baseline import predict_bad, predict_flammable

    text = f"{row['name']}\n{row['description']}"
    return predict_bad(text) if row["category"] == "БАД" else predict_flammable(text)


def build(official_path: Path, curated_path: Path | None) -> tuple[list[dict], dict]:
    import pandas as pd

    official = add_keys(validate_frame(pd.read_csv(official_path), "official"))
    official = assign_official_folds(official)
    official["source_partition"] = "official"

    frames = [official]
    curated_source_hash = None
    if curated_path is not None:
        curated = add_keys(validate_frame(pd.read_csv(curated_path), "curated"))
        overlap = set(official["id"]) & set(curated["id"])
        if overlap:
            raise ValueError(f"official/curated id overlap: {sorted(overlap)[:20]}")
        curated_source_hash = sha256_file(curated_path)
        curated = curated[~curated["dup_key"].isin(set(official["dup_key"]))].copy()
        group_fold = official.groupby("split_group")["fold"].first()
        curated["fold"] = curated["split_group"].map(group_fold).fillna(-1).astype(int)
        curated["majority_label"] = curated["label"]
        curated["source_partition"] = "curated"
        frames.append(curated)

    combined = pd.concat(frames, ignore_index=True)
    exact = combined.groupby("dup_key")["label"].agg(["mean", "count"])
    combined = combined.drop(columns=["majority_label"]).join(exact, on="dup_key")
    combined["majority_label"] = (combined["mean"] >= 0.5).astype(int)
    coin_flip = combined["mean"] == 0.5
    coin_flip_groups = int(combined.loc[coin_flip, "dup_key"].nunique())
    representatives = (
        combined.loc[~coin_flip]
        .sort_values("id")
        .drop_duplicates("dup_key", keep="first")
        .copy()
    )
    representatives["rule_pred"] = representatives.apply(rule_prediction, axis=1)
    representatives["weight"] = representatives["count"].map(
        lambda count: math.log2(1 + int(count))
    )
    representatives.loc[
        representatives["rule_pred"] != representatives["majority_label"], "weight"
    ] *= 2.0

    rows = []
    for row in representatives.itertuples(index=False):
        rows.append(
            {
                "id": int(row.id),
                "dup_key": row.dup_key,
                "split_group": row.split_group,
                "fold": int(row.fold),
                "category": row.category,
                "name": row.name,
                "description": row.description,
                "label": int(row.majority_label),
                "soft_label": round(float(row.mean), 4),
                "weight": round(float(row.weight), 4),
                "train_excluded": 0,
                "rule_pred": int(row.rule_pred),
                "images": [],
                "source_partition": row.source_partition,
            }
        )

    fold_leaks = {}
    for fold in range(FOLDS):
        validation_groups = {
            row["split_group"]
            for row in rows
            if row["source_partition"] == "official" and row["fold"] == fold
        }
        train_groups = {
            row["split_group"]
            for row in rows
            if row["source_partition"] == "official" and row["fold"] != fold
        }
        fold_leaks[str(fold)] = len(validation_groups & train_groups)
    if any(fold_leaks.values()):
        raise AssertionError(f"family leakage detected: {fold_leaks}")

    manifest = {
        "schema_version": "prepared-text-dataset-v1",
        "official_source_sha256": sha256_file(official_path),
        "curated_source_sha256": curated_source_hash,
        "rows": len(rows),
        "source_counts": dict(Counter(row["source_partition"] for row in rows)),
        "category_label_counts": {
            f"{category}|{label}": count
            for (category, label), count in sorted(
                Counter((row["category"], row["label"]) for row in rows).items()
            )
        },
        "coin_flip_groups_dropped": coin_flip_groups,
        "official_family_leaks": fold_leaks,
        "folds": FOLDS,
        "seed": SEED,
    }
    return rows, manifest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def verify_frozen(path: Path) -> None:
    actual = sha256_file(path)
    if actual != FROZEN_SHA256:
        raise SystemExit(f"frozen dataset SHA mismatch: {actual} != {FROZEN_SHA256}")
    count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if count != 10028:
        raise SystemExit(f"frozen dataset row mismatch: {count} != 10028")
    print(f"PASS frozen dataset rows={count} sha256={actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--curated", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-frozen", type=Path)
    args = parser.parse_args()
    if args.verify_frozen:
        verify_frozen(args.verify_frozen)
        return
    if not args.official or not args.output or not args.manifest:
        parser.error("--official, --output and --manifest are required when building")
    rows, manifest = build(args.official, args.curated)
    write_jsonl(args.output, rows)
    manifest["output_sha256"] = sha256_file(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
