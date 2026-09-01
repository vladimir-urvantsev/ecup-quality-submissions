#!/usr/bin/env python3
"""Validate submission ID coverage and the comment/verdict wire grammar."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


WIRE = re.compile(r"^<комментарий>(?P<comment>.*)<вердикт>(?P<verdict>бан|не бан)$")


def ids(path: Path, *, result: bool) -> list[int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "result"} if result else {"id"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    values = [int(row["id"]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate ids")
    if result:
        for row in rows:
            match = WIRE.fullmatch(row["result"])
            if not match:
                raise ValueError(f"id={row['id']}: invalid result grammar")
            comment = match.group("comment")
            if not 50 <= len(comment) <= 300:
                raise ValueError(f"id={row['id']}: comment length {len(comment)}")
            if "<" in comment or ">" in comment:
                raise ValueError(f"id={row['id']}: angle bracket inside comment")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = ids(args.input, result=False)
    actual = ids(args.output, result=True)
    if set(expected) != set(actual) or len(expected) != len(actual):
        raise SystemExit("output ID set differs from input")
    print(f"PASS rows={len(actual)}")


if __name__ == "__main__":
    main()

