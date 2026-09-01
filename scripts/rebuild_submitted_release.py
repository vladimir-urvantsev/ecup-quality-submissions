#!/usr/bin/env python3
"""Rebuild the exact submitted ZIP from tracked source and the frozen adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from package_final_submission import build as package_release  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact(path: Path, expected: dict, member: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular source for {member}: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(expected["bytes"]) or digest != expected["sha256"]:
        raise ValueError(
            f"frozen member drift for {member}: "
            f"bytes={size}, sha256={digest}"
        )


def rebuild(adapter_model: Path, output_zip: Path) -> dict:
    release = json.loads(
        (ROOT / "manifests/frozen_release.json").read_text(encoding="utf-8")
    )
    submitted_manifest_path = ROOT / "submission/candidate_manifest.json"
    submitted_manifest_meta = release["candidate_manifest"]
    require_exact(
        submitted_manifest_path,
        submitted_manifest_meta,
        "candidate_manifest.json",
    )
    submitted_manifest = json.loads(
        submitted_manifest_path.read_text(encoding="utf-8")
    )
    members = dict(submitted_manifest["files"])
    members["candidate_manifest.json"] = submitted_manifest_meta
    expected_adapter = members["adapter/adapter_model.safetensors"]
    require_exact(adapter_model, expected_adapter, "adapter/adapter_model.safetensors")

    if output_zip.exists():
        raise ValueError(f"refusing to overwrite: {output_zip}")

    with tempfile.TemporaryDirectory(prefix="ecup-submitted-release-") as temporary:
        package_dir = Path(temporary) / "package"
        for member, expected in members.items():
            source = (
                adapter_model
                if member == "adapter/adapter_model.safetensors"
                else ROOT / "submission" / member
            )
            require_exact(source, expected, member)
            target = package_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        package_release(package_dir, output_zip)

    actual_archive = sha256_file(output_zip)
    expected_archive = release["archive"]["sha256"]
    if actual_archive != expected_archive:
        output_zip.unlink(missing_ok=True)
        raise ValueError(
            f"rebuilt archive drift: {actual_archive} != {expected_archive}"
        )
    return {
        "status": "PASS",
        "archive": str(output_zip.resolve()),
        "sha256": actual_archive,
        "members": len(members),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-model", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            rebuild(args.adapter_model, args.output_zip),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
