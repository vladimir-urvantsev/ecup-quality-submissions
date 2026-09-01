#!/usr/bin/env python3
"""Verify a historical release archive against its public immutable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(manifest_path: Path, archive_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_archive = manifest["archive"]["sha256"]
    actual_archive = sha256_file(archive_path)
    if actual_archive != expected_archive:
        raise ValueError(f"archive SHA mismatch: {actual_archive} != {expected_archive}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"unsafe ZIP member: {name}")
        for name, expected in manifest["critical_members"].items():
            actual = sha256_bytes(archive.read(name))
            if actual != expected:
                raise ValueError(f"critical member drift: {name}")
        candidate_meta = manifest["candidate_manifest"]
        candidate_payload = archive.read(candidate_meta["member"])
        if len(candidate_payload) != int(candidate_meta["bytes"]):
            raise ValueError("candidate manifest byte-size mismatch")
        if sha256_bytes(candidate_payload) != candidate_meta["sha256"]:
            raise ValueError("candidate manifest SHA mismatch")
        candidate = json.loads(candidate_payload)
        expected_members = dict(candidate["files"])
        expected_members[candidate_meta["member"]] = candidate_meta
        if set(names) != set(expected_members):
            raise ValueError("frozen archive inventory differs from candidate manifest")
        for name, expected in expected_members.items():
            payload = archive.read(name)
            if len(payload) != int(expected["bytes"]):
                raise ValueError(f"member byte-size mismatch: {name}")
            if sha256_bytes(payload) != expected["sha256"]:
                raise ValueError(f"member SHA mismatch: {name}")
        adapter = manifest["adapter"]
        payload = archive.read(adapter["member"])
        if len(payload) != int(adapter["bytes"]):
            raise ValueError("adapter byte-size mismatch")
        if sha256_bytes(payload) != adapter["sha256"]:
            raise ValueError("adapter SHA mismatch")
        config = json.loads(archive.read("adapter/adapter_config.json"))
        thresholds = json.loads(archive.read("adapter/thresholds.json"))["thresholds"]
        if int(config["r"]) != int(adapter["rank"]):
            raise ValueError("adapter rank mismatch")
        if float(config["lora_alpha"]) != float(adapter["lora_alpha"]):
            raise ValueError("adapter alpha mismatch")
        if thresholds != manifest["thresholds"]:
            raise ValueError("threshold mismatch")
    return {"archive_sha256": actual_archive, "members": len(names), "status": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest, args.archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
