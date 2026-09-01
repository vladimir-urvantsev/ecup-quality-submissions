"""Conservative, deterministic explanations for final classifier verdicts.

The score-producing classifier and this renderer are deliberately separated.
``prediction`` is an input and is never recomputed or overridden here.  A full
explanation is emitted only when a label-blind text rule or a validated
grounded-evidence record supports that same prediction.  Otherwise the output
states honestly that the available evidence is insufficient for a grounded
explanation.

JSON remains an internal audit format.  The moderator-facing result is always
three short Russian lines: ``ВЫВОД / ОСНОВАНИЕ / КОНТРАСТ``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from rationale_schema import (
    SchemaValidationError,
    element_policy_direction,
    evaluate_policy,
    validate_blind_record,
)
from text_rule_rationales import extract_rule


MAX_RATIONALE_CHARS = 300
VALID_CATEGORIES = frozenset({"БАД", "Легковоспламеняющиеся"})

_BLIND_KEYS = frozenset(
    {
        "schema_version",
        "product_id",
        "category",
        "product_form",
        "elements",
        "modality_relation",
        "conflicts",
        "counterfactual",
        "conclusion",
        "official_policy_verdict",
        "policy_rule_ids",
    }
)

# These phrases implement the official policy and the empirical boundaries
# already frozen in text_rule_rationales.py: an external/empty fuel source is
# not contents, an integrated piezo is not a standalone lighter, ordinary dry
# fuel/candles/pneumatic poppers are quarantined unless a decisive cue exists.
_BASIS_BY_RULE = {
    "BAD-POS-DIRECT-MARKING": "это прямая маркировка БАД",
    "BAD-NEG-SPORTS-NUTRITION": "это прямое указание на спортивное питание",
    "BAD-NEG-EXPLICIT-EXCLUSION": "это прямое исключение из БАД",
    "BAD-NEG-MARKING-EXPLICITLY-ABSENT": "маркировка БАД явно отсутствует",
    "FLAM-POS-SEPARATE-IN-KIT": "опасный предмет отдельно входит в комплект",
    "FLAM-POS-FLAMMABLE-CONTENTS": "подтверждено продаваемое горючее содержимое",
    "FLAM-POS-STANDALONE-IGNITION": "сам товар является источником открытого огня",
    "FLAM-NEG-EMPTY-CONTENTS": "горючее содержимое явно отсутствует",
    "FLAM-NEG-NOT-INCLUDED": "опасный предмет явно не входит в комплект",
    "FLAM-NEG-INTEGRATED-IGNITION": "источник зажигания встроен в устройство",
    "FLAM-NEG-COMPONENT": "горючий материал является лишь компонентом изделия",
    "FLAM-NEG-NOT-PART-OF-PRODUCT": "опасный предмет не является частью продажи",
    "FLAM-NEG-NO-OPEN-FLAME": "механизм не является источником открытого огня",
    # Deterministic text extractor's empirical appliance boundary.  It is not
    # persisted by the richer ontology but is safe for post-hoc text evidence.
    "FLAM-NEG-DEVICE-WITHOUT-CONTENT": (
        "это устройство для огня, а не горючее содержимое"
    ),
}

_CONTRAST_BY_RULE = {
    "BAD-POS-DIRECT-MARKING": (
        "Без маркировки БАД или dietary supplement основания для категории нет."
    ),
    "BAD-NEG-SPORTS-NUTRITION": (
        "Прямая маркировка БАД без признака спортивного питания дала бы иной вывод."
    ),
    "BAD-NEG-EXPLICIT-EXCLUSION": (
        "Без исключения потребовалась бы прямая маркировка БАД."
    ),
    "BAD-NEG-MARKING-EXPLICITLY-ABSENT": (
        "Прямая маркировка БАД дала бы противоположный вывод."
    ),
    "FLAM-POS-SEPARATE-IN-KIT": (
        "Без этого предмета одно лишь устройство не относилось бы к категории."
    ),
    "FLAM-POS-FLAMMABLE-CONTENTS": (
        "Пустая конструкция без содержимого не относилась бы к категории."
    ),
    "FLAM-POS-STANDALONE-IGNITION": (
        "Встроенный поджиг без отдельного источника огня дал бы иной вывод."
    ),
    "FLAM-NEG-EMPTY-CONTENTS": (
        "С горючим содержимым внутри товар относился бы к категории."
    ),
    "FLAM-NEG-NOT-INCLUDED": (
        "Отдельный опасный предмет в комплекте дал бы положительный вывод."
    ),
    "FLAM-NEG-INTEGRATED-IGNITION": (
        "Отдельная зажигалка или спички в комплекте изменили бы вывод."
    ),
    "FLAM-NEG-COMPONENT": (
        "Отдельно продаваемое горючее или источник огня дали бы иной вывод."
    ),
    "FLAM-NEG-NOT-PART-OF-PRODUCT": (
        "Явное включение опасного предмета в продажу изменило бы вывод."
    ),
    "FLAM-NEG-NO-OPEN-FLAME": (
        "Самостоятельный источник открытого огня дал бы положительный вывод."
    ),
    "FLAM-NEG-DEVICE-WITHOUT-CONTENT": (
        "Включённое топливо или отдельный источник огня изменили бы вывод."
    ),
}

_FALLBACK_CONTRAST = {
    ("БАД", 1): (
        "Явное «не БАД» или «спортивное питание» поддержало бы иной вывод."
    ),
    ("БАД", 0): (
        "Прямая маркировка БАД или dietary supplement поддержала бы иной вывод."
    ),
    ("Легковоспламеняющиеся", 1): (
        "Явно пустая конструкция или отсутствие опасного предмета в комплекте поддержали бы иной вывод."
    ),
    ("Легковоспламеняющиеся", 0): (
        "Включённое топливо или отдельный источник огня поддержали бы иной вывод."
    ),
}


@dataclass(frozen=True)
class RenderedRationale:
    """Moderator-facing text plus non-prose diagnostics for the caller."""

    text: str
    category: str
    prediction: int
    status: str
    rule_id: str | None = None
    evidence_source: str | None = None


def _prediction(value: int | bool) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    raise ValueError("prediction must be 0 or 1")


def _source(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping")
    category = source.get("category")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")
    name = source.get("name", "")
    description = source.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError("source name/description must be strings")
    images = source.get("images", [])
    if images is None:
        images = []
    if not isinstance(images, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in images
    ):
        raise ValueError("source.images must be a list of non-empty strings")
    ocr = source.get("ocr_text_by_image")
    if ocr is not None and not isinstance(ocr, Mapping):
        raise ValueError("source.ocr_text_by_image must be a mapping")
    return {
        "category": category,
        "name": name,
        "description": description,
        "images": list(images),
        "ocr_text_by_image": ocr,
    }


def _clean(value: Any) -> str:
    return " ".join(str(value).replace("«", "").replace("»", "").split())


def _clip(value: Any, limit: int, *, focus: str | None = None) -> tuple[str, bool]:
    clean = _clean(value)
    if len(clean) <= limit:
        return clean, False
    focus_clean = _clean(focus or "")
    focus_at = clean.casefold().find(focus_clean.casefold()) if focus_clean else -1
    if focus_at >= 0:
        # Preserve the policy-relevant entity even when the visual observation
        # starts with generic scene prose.  The displayed fragment is still a
        # contiguous substring of the validated observation.
        start = max(0, focus_at - max(0, (limit - len(focus_clean)) // 2))
        end = min(len(clean), start + limit)
        start = max(0, end - limit)
        if start:
            next_space = clean.find(" ", start)
            if 0 <= next_space < focus_at:
                start = next_space + 1
        if end < len(clean):
            previous_space = clean.rfind(" ", start, end + 1)
            if previous_space > focus_at + len(focus_clean):
                end = previous_space
        return clean[start:end].strip(" ,;:-"), True
    cut = clean.rfind(" ", 0, limit + 1)
    if cut < max(12, limit // 2):
        cut = limit
    return clean[:cut].rstrip(" ,;:-"), True


def _evidence_text(
    evidence: Mapping[str, Any],
    *,
    max_chars: int = 76,
    focus: str | None = None,
) -> str:
    source = str(evidence.get("source", ""))
    if source in {"name", "description", "ocr"}:
        span, clipped = _clip(evidence.get("exact_span", ""), max_chars)
        quote = f"«{span}»" + (" (фрагмент)" if clipped else "")
        if source == "name":
            return f"в названии указано {quote}"
        if source == "description":
            return f"в описании указано {quote}"
        return f"на изображении {evidence.get('image_id')} прочитано {quote}"
    observation, clipped = _clip(
        evidence.get("observation", ""), max_chars, focus=focus
    )
    suffix = " (описание сокращено)" if clipped else ""
    return (
        f"на изображении {evidence.get('image_id')} видно: "
        f"{observation}{suffix}"
    )


def _conclusion(category: str, prediction: int) -> str:
    relation = "относится" if prediction else "не относится"
    return f"ВЫВОД: Товар {relation} к категории «{category}»."


def _assemble(
    category: str,
    prediction: int,
    basis: str,
    contrast: str,
) -> str:
    text = "\n".join(
        (_conclusion(category, prediction), f"ОСНОВАНИЕ: {basis}", f"КОНТРАСТ: {contrast}")
    )
    if len(text) > MAX_RATIONALE_CHARS:
        raise ValueError(
            f"post-hoc rationale exceeds {MAX_RATIONALE_CHARS} characters: {len(text)}"
        )
    return text


def _render_grounded(
    category: str,
    prediction: int,
    rule_id: str,
    evidence: Mapping[str, Any],
    status: str,
    *,
    focus: str | None = None,
) -> RenderedRationale | None:
    descriptor = _BASIS_BY_RULE.get(rule_id)
    contrast = _CONTRAST_BY_RULE.get(rule_id)
    if not descriptor or not contrast:
        return None
    evidence_phrase = _evidence_text(evidence, focus=focus)
    basis = f"{evidence_phrase}; {descriptor}."
    try:
        text = _assemble(category, prediction, basis, contrast)
    except ValueError:
        # Preserve all three semantic fields and shorten only the displayed
        # evidence.  The displayed text remains a contiguous fragment of the
        # already validated span/observation; no replacement fact is generated.
        evidence_phrase = _evidence_text(evidence, max_chars=38, focus=focus)
        basis = f"{evidence_phrase}; {descriptor}."
        try:
            text = _assemble(category, prediction, basis, contrast)
        except ValueError:
            # An abnormally long image identifier or future rule wording must
            # not break a submission row.  Returning None routes the caller to
            # the honest insufficient template.
            return None
    return RenderedRationale(
        text=text,
        category=category,
        prediction=prediction,
        status=status,
        rule_id=rule_id,
        evidence_source=str(evidence.get("source")),
    )


def _text_rule_rationale(
    source: Mapping[str, Any], prediction: int
) -> RenderedRationale | None:
    extracted = extract_rule(source)
    if extracted is None or int(extracted["rule_prediction"]) != prediction:
        return None
    return _render_grounded(
        str(extracted["category"]),
        prediction,
        str(extracted["rule_id"]),
        extracted["evidence"],
        "grounded_text_rule",
    )


def _structured_rationale(
    source: Mapping[str, Any],
    prediction: int,
    grounded_record: Mapping[str, Any] | None,
    *,
    allow_visual_evidence: bool,
) -> RenderedRationale | None:
    if not isinstance(grounded_record, Mapping):
        return None
    blind = {key: value for key, value in grounded_record.items() if key in _BLIND_KEYS}
    try:
        validated = validate_blind_record(
            blind,
            name_text=str(source["name"]),
            description_text=str(source["description"]),
            ocr_text_by_image=source.get("ocr_text_by_image"),
            # Empty is intentional: visual evidence cannot pass when the caller
            # did not bind it to an actual input image.
            known_image_ids=source.get("images", []),
            strict=True,
        )
    except (SchemaValidationError, TypeError, ValueError):
        return None

    if validated["category"] != source["category"]:
        return None
    expected_verdict = "yes" if prediction else "no"
    if validated["official_policy_verdict"] != expected_verdict:
        return None

    direction = 1 if prediction else -1
    candidates: list[tuple[int, int, int, str, str, Mapping[str, Any]]] = []
    source_rank = {"name": 0, "description": 1, "ocr": 2, "image": 3}
    for element_index, element in enumerate(validated["elements"]):
        if element_policy_direction(validated["category"], element) != direction:
            continue
        element_decision = evaluate_policy(
            {
                "category": validated["category"],
                "elements": [element],
                "modality_relation": "agree",
                "conflicts": [],
            }
        )
        rule_id = next(
            (rule for rule in element_decision.rule_ids if rule in _BASIS_BY_RULE),
            "",
        )
        if not rule_id:
            continue
        for evidence_index, evidence in enumerate(element["evidence"]):
            evidence_source = str(evidence.get("source", ""))
            if evidence.get("polarity") != "support":
                continue
            if evidence_source in {"image", "ocr"} and not allow_visual_evidence:
                continue
            evidence_value = evidence.get(
                "observation" if evidence_source == "image" else "exact_span", ""
            )
            if not _clean(evidence_value):
                continue
            candidates.append(
                (
                    source_rank.get(evidence_source, 9),
                    element_index,
                    evidence_index,
                    rule_id,
                    str(element.get("name", "")),
                    evidence,
                )
            )

    if not candidates:
        return None
    _, _, _, rule_id, element_name, evidence = min(
        candidates, key=lambda item: item[:3]
    )
    return _render_grounded(
        validated["category"],
        prediction,
        rule_id,
        evidence,
        "grounded_structured",
        focus=element_name,
    )


def build_posthoc_rationale(
    source: Mapping[str, Any],
    prediction: int | bool,
    *,
    grounded_record: Mapping[str, Any] | None = None,
    allow_visual_evidence: bool = False,
) -> RenderedRationale:
    """Build an explanation without ever changing ``prediction``.

    Priority is precision-first: deterministic label-blind text evidence,
    followed by a validated structured record.  Image/OCR observations require
    an explicit opt-in and must reference one of ``source['images']``.  The
    opt-in is intended for audited or otherwise certified VLM evidence, not raw
    generations.
    """

    clean_source = _source(source)
    final_prediction = _prediction(prediction)

    rendered = _text_rule_rationale(clean_source, final_prediction)
    if rendered is None:
        rendered = _structured_rationale(
            clean_source,
            final_prediction,
            grounded_record,
            allow_visual_evidence=allow_visual_evidence,
        )
    if rendered is not None:
        assert rendered.prediction == final_prediction
        return rendered

    basis = (
        "Проверяемой цитаты или подтверждённого визуального факта недостаточно; "
        "объяснение фиксирует только итог классификатора."
    )
    text = _assemble(
        clean_source["category"],
        final_prediction,
        basis,
        _FALLBACK_CONTRAST[(clean_source["category"], final_prediction)],
    )
    return RenderedRationale(
        text=text,
        category=clean_source["category"],
        prediction=final_prediction,
        status="insufficient",
    )


def render_posthoc_rationale(
    source: Mapping[str, Any],
    prediction: int | bool,
    *,
    grounded_record: Mapping[str, Any] | None = None,
    allow_visual_evidence: bool = False,
) -> str:
    """Return only the three-line moderator-facing Russian explanation."""

    return build_posthoc_rationale(
        source,
        prediction,
        grounded_record=grounded_record,
        allow_visual_evidence=allow_visual_evidence,
    ).text


__all__ = [
    "MAX_RATIONALE_CHARS",
    "RenderedRationale",
    "build_posthoc_rationale",
    "render_posthoc_rationale",
]
