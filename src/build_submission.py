#!/usr/bin/env python3
"""Assemble a reviewed runtime plus one adapter into a submission directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


RUNTIME_FILES = (
    "run.py",
    "run_base.py",
    "safe_overrides.py",
    "certificate.py",
    "production_reviewer_rationale.py",
    "reviewer_rationale_integration_v1.py",
    "reviewer_safe_rationale_v1.py",
    "compact_vlm_rationale.py",
    "posthoc_rationale.py",
    "rationale_schema.py",
    "text_rule_rationales.py",
    "reviewer_rationale_contract.json",
    "metadata.json",
    "README.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_regular(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"missing regular source file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def build(root: Path, adapter: Path, recipe_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    adapter_config = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    expected_rank = int(recipe["soup"]["expected_rank"])
    expected_alpha = float(recipe["soup"]["expected_alpha"])
    if int(adapter_config["r"]) != expected_rank:
        raise ValueError(f"adapter rank {adapter_config['r']} != {expected_rank}")
    if float(adapter_config["lora_alpha"]) != expected_alpha:
        raise ValueError("adapter alpha mismatch")

    runtime = root / "submission"
    output.mkdir(parents=True)
    for name in RUNTIME_FILES:
        copy_regular(runtime / name, output / name)
    copy_regular(adapter / "adapter_model.safetensors", output / "adapter/adapter_model.safetensors")

    public_config = dict(adapter_config)
    public_config["base_model_name_or_path"] = recipe["base_model"]
    (output / "adapter/adapter_config.json").write_text(
        json.dumps(public_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    thresholds = {
        "schema_version": "production-thresholds-v1",
        "thresholds": recipe["thresholds"],
        "quantiles": {},
        "truncation": recipe["input"]["truncation"],
        "args": {"max_len": recipe["input"]["max_len"]},
        "base_model": recipe["base_model"],
        "solution": recipe["solution"],
    }
    (output / "adapter/thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {
        str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "assembled-public-submission-v1",
        "solution": recipe["solution"],
        "text_only": True,
        "rationale_is_post_verdict": True,
        "files": files,
    }
    (output / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = build(root, args.adapter, args.recipe, args.output_dir)
    print(f"PASS solution={report['solution']} files={len(report['files'])}")


if __name__ == "__main__":
    main()

