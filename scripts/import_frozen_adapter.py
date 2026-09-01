#!/usr/bin/env python3
"""Verify a historical ZIP and extract only its adapter into a local directory."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from verify_frozen_release import verify


MEMBERS = (
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "adapter/thresholds.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.manifest, args.archive)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        for member in MEMBERS:
            target = args.output / Path(member).name
            target.write_bytes(archive.read(member))
    print(json.dumps({**report, "adapter_output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

