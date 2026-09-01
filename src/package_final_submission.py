#!/usr/bin/env python3
"""Safely package an assembled submission directory as a deterministic ZIP."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REQUIRED_ROOT_FILES = ("run.py", "metadata.json")
FORBIDDEN_PARTS = {"__pycache__"}
FORBIDDEN_NAMES = {".ds_store"}
FORBIDDEN_SUFFIXES = {".pyc", ".zip", ".jsonl"}
FORBIDDEN_BANK_TOKENS = ("fact_bank", "vlm_bank", "retrieval")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class PackagingError(ValueError):
    """Raised when a package or archive violates the submission contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden(relative: Path, *, allow_retrieval: bool = False) -> str | None:
    lowered_parts = tuple(part.casefold() for part in relative.parts)
    lowered_name = relative.name.casefold()
    if any(part in FORBIDDEN_PARTS for part in lowered_parts):
        return "__pycache__ path"
    if lowered_name in FORBIDDEN_NAMES:
        return relative.name
    if relative.suffix.casefold() in FORBIDDEN_SUFFIXES:
        return f"{relative.suffix} file"
    if not allow_retrieval and any(
        token in part
        for part in lowered_parts
        for token in FORBIDDEN_BANK_TOKENS
    ):
        return "bank/retrieval artifact name"
    return None


def inventory(
    package_dir: Path, *, allow_retrieval: bool = False
) -> dict[str, tuple[str, int]]:
    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise PackagingError(f"package directory missing: {package_dir}")
    for required in REQUIRED_ROOT_FILES:
        path = package_dir / required
        if not path.is_file() or path.is_symlink():
            raise PackagingError(f"required regular root file missing: {required}")

    files: dict[str, tuple[str, int]] = {}
    for path in sorted(package_dir.rglob("*")):
        relative = path.relative_to(package_dir)
        if path.is_symlink():
            raise PackagingError(f"symlink forbidden: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PackagingError(f"non-regular file forbidden: {relative}")
        reason = _forbidden(relative, allow_retrieval=allow_retrieval)
        if reason:
            raise PackagingError(f"forbidden {reason}: {relative}")
        archive_name = PurePosixPath(*relative.parts).as_posix()
        if archive_name in files:
            raise PackagingError(f"duplicate normalized path: {archive_name}")
        files[archive_name] = (sha256_file(path), path.stat().st_size)
    return files


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if not name or name.endswith("/") or member.is_absolute() or ".." in member.parts:
        raise PackagingError(f"unsafe archive member: {name!r}")
    if member.as_posix() != name or "\\" in name:
        raise PackagingError(f"non-canonical archive member: {name!r}")
    return member


def verify_archive(
    zip_path: Path,
    expected: dict[str, tuple[str, int]],
    *,
    allow_retrieval: bool = False,
) -> None:
    if not zip_path.is_file():
        raise PackagingError(f"archive missing: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = [item.filename for item in archive.infolist()]
        if len(names) != len(set(names)):
            raise PackagingError("archive contains duplicate members")
        members = {_safe_member(name).as_posix() for name in names}
        if members != set(expected):
            raise PackagingError(
                f"archive inventory differs: missing={sorted(set(expected)-members)} "
                f"extra={sorted(members-set(expected))}"
            )
        with tempfile.TemporaryDirectory(prefix="ecup-final-verify-") as temporary:
            root = Path(temporary)
            for item in archive.infolist():
                member = _safe_member(item.filename)
                mode = (item.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise PackagingError(f"archive symlink forbidden: {item.filename}")
                target = root.joinpath(*member.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = archive.read(item)
                target.write_bytes(payload)
            actual = inventory(root, allow_retrieval=allow_retrieval)
            if actual != expected:
                raise PackagingError("extracted SHA-256/size inventory differs")


def build(
    package_dir: Path,
    output_zip: Path,
    *,
    allow_retrieval: bool = False,
) -> dict[str, tuple[str, int]]:
    package_dir = package_dir.resolve()
    output_zip = output_zip.resolve()
    if output_zip == package_dir or package_dir in output_zip.parents:
        raise PackagingError("output ZIP must be outside package directory")
    expected = inventory(package_dir, allow_retrieval=allow_retrieval)
    if output_zip.exists():
        raise PackagingError(f"refusing to overwrite: {output_zip}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_name(f".{output_zip.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise PackagingError(f"temporary archive exists: {temporary}")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(expected):
                source = package_dir.joinpath(*PurePosixPath(name).parts)
                info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.flag_bits |= 0x800
                info.file_size = source.stat().st_size
                with source.open("rb") as source_handle, archive.open(
                    info, "w", force_zip64=True
                ) as archive_handle:
                    shutil.copyfileobj(source_handle, archive_handle, 1 << 20)
        verify_archive(
            temporary, expected, allow_retrieval=allow_retrieval
        )
        os.replace(temporary, output_zip)
        verify_archive(
            output_zip, expected, allow_retrieval=allow_retrieval
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-retrieval", action="store_true")
    args = parser.parse_args()
    package = Path(args.package_dir)
    output = Path(args.output_zip)
    expected = inventory(package, allow_retrieval=args.allow_retrieval)
    if args.verify_only:
        verify_archive(
            output, expected, allow_retrieval=args.allow_retrieval
        )
    else:
        build(
            package, output, allow_retrieval=args.allow_retrieval
        )
    print(
        f"PASS files={len(expected)} bytes={sum(size for _, size in expected.values())} "
        f"zip={output.resolve()}"
    )


if __name__ == "__main__":
    main()
