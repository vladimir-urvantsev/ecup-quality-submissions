"""Production adapter for the audited reviewer-rationale renderer.

The classifier verdict is an immutable argument.  This small adapter projects
only ``category``, ``name`` and ``description`` into the audited integration
renderer; it deliberately has no API for labels, probabilities, model scores,
offline fact banks or raw VLM generations.  The latter may enter a future
package only through the separately audited, human-accepted VLM-bank contract.

The evaluator's wire contract is stricter than the research integration
contract: a comment must contain 50..300 characters.  If the integrated
four-line conflict explanation is too long, we fall back to the conservative
three-line text renderer.  Any other failure uses a fixed honest explanation.
No evidence is clipped, because clipping can remove a negation or a role cue.
"""

from __future__ import annotations

from typing import Any

from reviewer_rationale_integration_v1 import build_integrated_reviewer_rationale
from reviewer_safe_rationale_v1 import build_reviewer_safe_rationale


RUNTIME_VERSION = "production-reviewer-rationale-v1.0.0"
MIN_COMMENT_CHARS = 50
MAX_COMMENT_CHARS = 300
BAD = "БАД"
FLAM = "Легковоспламеняющиеся"
VALID_CATEGORIES = frozenset({BAD, FLAM})

_FALLBACK_BASIS = (
    "По имеющимся данным безопасной дословной опоры нет; "
    "сохранён вердикт классификатора."
)
_FALLBACK_CONTRAST = (
    "Для уточнения случая требуется ручная проверка значения и роли "
    "указанных в карточке признаков."
)


def _prediction(value: int | bool) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    raise ValueError("prediction must be 0 or 1")


def _text(value: Any) -> str:
    # Pandas NaN and arbitrary objects must never be stringified into evidence.
    return value if isinstance(value, str) else ""


def _conclusion(category: str, prediction: int) -> str:
    relation = "относится" if prediction else "не относится"
    return f"ВЫВОД: Товар {relation} к категории «{category}»."


def _fixed_fallback(category: str, prediction: int) -> str:
    return "\n".join((
        _conclusion(category, prediction),
        f"ОСНОВАНИЕ: {_FALLBACK_BASIS}",
        f"КОНТРАСТ: {_FALLBACK_CONTRAST}",
    ))


def _valid_comment(text: Any) -> bool:
    if not isinstance(text, str) or not MIN_COMMENT_CHARS <= len(text) <= MAX_COMMENT_CHARS:
        return False
    if "<" in text or ">" in text or "\x00" in text:
        return False
    lines = text.splitlines()
    prefixes = ("ВЫВОД:", "ОСНОВАНИЕ:", "КОНТРАСТ:", "КОНФЛИКТ:")
    return len(lines) in {3, 4} and all(
        line.startswith(prefixes[index]) for index, line in enumerate(lines)
    )


def render_comment(
    category: str,
    name: Any,
    description: Any,
    prediction: int | bool,
    *,
    scored: bool = True,
) -> str:
    """Render one reviewer comment while preserving ``prediction`` exactly.

    ``scored`` exists only for compatibility with the old submission comment
    API.  It is intentionally not evidence and cannot change the explanation.
    """

    del scored
    if category not in VALID_CATEGORIES:
        raise ValueError(f"invalid category: {category!r}")
    final_prediction = _prediction(prediction)
    source = {
        "category": category,
        "name": _text(name),
        "description": _text(description),
        # No accepted visual bank is shipped.  Keeping the image projection
        # empty makes it impossible for raw/unreviewed VLM text to leak in.
        "images": [],
        "image_content_signature": "",
    }
    try:
        integrated = build_integrated_reviewer_rationale(
            source, final_prediction, fact_record=None, vlm_record=None
        )
        if integrated.prediction != final_prediction:
            raise AssertionError("renderer changed immutable prediction")
        if _valid_comment(integrated.text):
            return integrated.text

        # The research integration permits 360 chars for a conflict line, but
        # the evaluator permits 300.  Re-render from the stricter text-only
        # contract instead of truncating a meaning-bearing statement.
        conservative = build_reviewer_safe_rationale(
            source, final_prediction, fact_record=None
        )
        if conservative.prediction != final_prediction:
            raise AssertionError("text renderer changed immutable prediction")
        if _valid_comment(conservative.text):
            return conservative.text
    except Exception:
        # Comment generation is post-hoc: it must never break score inference
        # or mutate the already selected verdict.
        pass

    fallback = _fixed_fallback(category, final_prediction)
    if not _valid_comment(fallback):  # static programming error, never data-dependent
        raise AssertionError("fixed reviewer fallback violates wire contract")
    return fallback


__all__ = [
    "MAX_COMMENT_CHARS",
    "MIN_COMMENT_CHARS",
    "RUNTIME_VERSION",
    "render_comment",
]
