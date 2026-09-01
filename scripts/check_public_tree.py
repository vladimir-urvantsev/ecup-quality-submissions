#!/usr/bin/env python3
"""Fail on local-user paths, secrets or oversized ordinary Git blobs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".cff"}
BLOCKED = (
    "/us" + "ers/",
    "run" + "pod",
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]{8,}"),
)


def audit(root: Path) -> list[str]:
    issues = []
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(root)
        size = path.stat().st_size
        if size > 90 * 1024 * 1024:
            issues.append(f"oversized ordinary Git blob: {relative} ({size} bytes)")
        if path.suffix.casefold() in {".csv", ".jsonl", ".parquet", ".safetensors", ".zip"}:
            issues.append(f"data/model artifact must not be committed: {relative}")
        if path.suffix.casefold() not in TEXT_SUFFIXES or size > 5 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.casefold()
        for token in BLOCKED:
            if token in lowered:
                issues.append(f"blocked local/private path in {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                issues.append(f"possible secret in {relative}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    issues = audit(args.root.resolve())
    if issues:
        raise SystemExit("PUBLIC TREE AUDIT FAILED\n" + "\n".join(f"- {x}" for x in issues))
    print("PASS public tree audit")


if __name__ == "__main__":
    main()
