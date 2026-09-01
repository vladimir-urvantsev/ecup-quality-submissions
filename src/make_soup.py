#!/usr/bin/env python3
"""Represent the exact mean of compatible LoRA deltas as one concatenated LoRA."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adapter(directory: Path):
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    state = load_file(str(directory / "adapter_model.safetensors"))
    modules = {}
    for key, tensor in state.items():
        if ".lora_A" in key:
            modules.setdefault(key.split(".lora_A")[0], {})["A"] = tensor
        elif ".lora_B" in key:
            modules.setdefault(key.split(".lora_B")[0], {})["B"] = tensor
    incomplete = [name for name, pair in modules.items() if set(pair) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"incomplete LoRA modules in {directory}: {incomplete[:5]}")
    return modules, config


def combine(directories: list[Path], output: Path) -> dict:
    if len(directories) < 2:
        raise ValueError("at least two adapters are required")
    loaded = [load_adapter(directory) for directory in directories]
    module_sets = [set(modules) for modules, _ in loaded]
    if any(modules != module_sets[0] for modules in module_sets[1:]):
        raise ValueError("adapter module inventories differ")
    base = loaded[0][1]
    keys = ("r", "lora_alpha", "target_modules", "base_model_name_or_path")
    for _, config in loaded[1:]:
        if any(config.get(key) != base.get(key) for key in keys):
            raise ValueError("adapter configurations are incompatible")

    n = len(loaded)
    output_state = {}
    for module in sorted(module_sets[0]):
        a = torch.cat([modules[module]["A"].float() for modules, _ in loaded], dim=0)
        b = torch.cat([modules[module]["B"].float() for modules, _ in loaded], dim=1) / n
        output_state[f"{module}.lora_A.weight"] = a
        output_state[f"{module}.lora_B.weight"] = b

    output.mkdir(parents=True, exist_ok=False)
    save_file(output_state, str(output / "adapter_model.safetensors"))
    config = dict(base)
    config["r"] = int(base["r"]) * n
    config["lora_alpha"] = float(base["lora_alpha"]) * n
    if float(config["lora_alpha"]).is_integer():
        config["lora_alpha"] = int(config["lora_alpha"])
    (output / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "exact-lora-delta-soup-v1",
        "operation": "mean_delta_via_rank_concatenation",
        "ingredients": [
            {
                "path": str(directory),
                "weights_sha256": sha256_file(directory / "adapter_model.safetensors"),
                "config_sha256": sha256_file(directory / "adapter_config.json"),
            }
            for directory in directories
        ],
        "modules": len(module_sets[0]),
        "rank": config["r"],
        "lora_alpha": config["lora_alpha"],
        "weights_sha256": sha256_file(output / "adapter_model.safetensors"),
    }
    (output / "soup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", required=True, help="glob for fold adapter directories")
    parser.add_argument("--extra", action="append", default=[], help="additional adapter directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directories = [Path(path) for path in sorted(glob.glob(args.adapters))]
    directories.extend(Path(path) for path in args.extra)
    if len(set(map(str, directories))) != len(directories):
        raise SystemExit("duplicate adapter path")
    report = combine(directories, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

