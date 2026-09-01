from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from package_final_submission import build, verify_archive  # noqa: E402


def test_deterministic_safe_archive(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (package / "metadata.json").write_text("{}\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    expected = build(package, first)
    build(package, second)
    assert first.read_bytes() == second.read_bytes()
    verify_archive(first, expected)

