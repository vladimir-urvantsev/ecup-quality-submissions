from __future__ import annotations

import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))

import production_reviewer_rationale as runtime  # noqa: E402


def _assert_wire(text, category, prediction):
    assert 50 <= len(text) <= 300
    assert "<" not in text and ">" not in text
    lines = text.splitlines()
    assert len(lines) in {3, 4}
    expected = "относится" if prediction else "не относится"
    assert lines[0] == f"ВЫВОД: Товар {expected} к категории «{category}»."
    assert lines[1].startswith("ОСНОВАНИЕ:")
    assert lines[2].startswith("КОНТРАСТ:")
    if len(lines) == 4:
        assert lines[3].startswith("КОНФЛИКТ:")


def test_all_fallbacks_obey_wire_contract():
    for category in (runtime.BAD, runtime.FLAM):
        for prediction in (0, 1):
            _assert_wire(runtime.render_comment(category, "", "", prediction), category, prediction)


def test_exact_spans_and_prediction_invariance():
    text = runtime.render_comment(
        runtime.FLAM, "Газовый баллон", "Баллон пустой, внутри нет газа.", 0
    )
    _assert_wire(text, runtime.FLAM, 0)
    assert "пуст" in text.casefold()
    parameters = set(inspect.signature(runtime.render_comment).parameters)
    assert not {"gold", "label", "score", "p_yes", "logits"} & parameters

