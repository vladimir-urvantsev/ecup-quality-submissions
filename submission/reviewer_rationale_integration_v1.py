"""Reviewer-safe integration of a frozen verdict, text facts and VLM facts.

This module is deliberately a *renderer*, not a classifier.  ``prediction`` is
an immutable input and there is no score output.  The conservative text
renderer remains the default.  A compact VLM fact may replace that prose only
when its source, selected images, output bytes and a human semantic acceptance
are all bound by the bank contract below.

The moderator-facing output is three short Russian lines
``ВЫВОД / ОСНОВАНИЕ / КОНТРАСТ``.  A fourth ``КОНФЛИКТ`` line is emitted only
when grounded evidence supports the opposite of the frozen verdict.  It is a
review warning, never an override.  Visual non-observation (``не видно``,
``отсутствует`` and similar wording) is never accepted as proof of absence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from compact_vlm_rationale import (
    CONTRACT_VERSION as COMPACT_VLM_CONTRACT_VERSION,
    CompactVLMContractError,
    CompactVLMFact,
    parse_compact_vlm_fact,
)
from reviewer_safe_rationale_v1 import (
    ReviewerSafeRationale,
    build_reviewer_safe_rationale,
    inspect_safe_candidates,
)


INTEGRATION_VERSION = "reviewer-rationale-integration-v1.0.0"
VLM_BANK_SCHEMA = "reviewer-compact-vlm-bank-v1"
MIN_CHARS = 50
MAX_CHARS = 360
BAD = "БАД"
FLAM = "Легковоспламеняющиеся"
VALID_CATEGORIES = frozenset({BAD, FLAM})

# Only rules for which a positive, human-checked visual observation can be a
# useful reviewer fact.  Sports-nutrition is intentionally absent: its
# empirical gold convention is noisy and it must not be narrated categorically.
SAFE_VISUAL_RULES = frozenset({
    "BAD-POS-DIRECT-MARKING",
    "BAD-NEG-EXPLICIT-EXCLUSION",
    "FLAM-POS-SEPARATE-IN-KIT",
    "FLAM-POS-FLAMMABLE-CONTENTS",
    "FLAM-POS-STANDALONE-IGNITION",
    "FLAM-NEG-INTEGRATED-IGNITION",
    "FLAM-NEG-COMPONENT",
    "FLAM-NEG-NOT-PART-OF-PRODUCT",
    "FLAM-NEG-NO-OPEN-FLAME",
})

_FORBIDDEN_BANK_KEYS = frozenset({
    "gold", "label", "majority_label", "soft_label", "p_yes", "prediction",
    "score", "verdict", "official_policy_verdict",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WS = re.compile(r"\s+")

# An image can positively show another object or a printed statement.  It
# cannot establish non-existence from one or several views.  Even a
# human-accepted record containing one of these claims fails closed because
# the claim type itself violates the rationale contract.
_VISUAL_ABSENCE = re.compile(
    r"(?:^|[^а-яёa-z0-9])(?:"
    r"нет|без|пуст\w*|отсутств\w*|"
    r"не\s+(?:вид\w*|показ\w*|замет\w*|присутств\w*|"
    r"име\w*|содерж\w*|вход\w*|включ\w*)|"
    r"no|without|absent|empty|not\s+(?:visible|shown|present|included)"
    r")(?:$|[^а-яёa-z0-9])",
    re.IGNORECASE | re.UNICODE,
)
_VISUAL_UNCERTAINTY = re.compile(
    r"(?:похоже|кажется|вероят\w*|возможно|предполож\w*|неясно|"
    r"может\s+быть|appears?|seems?|probably|possibly|unclear)",
    re.IGNORECASE | re.UNICODE,
)

_VISUAL_BASIS = {
    "BAD-POS-DIRECT-MARKING": "на упаковке прямо показана маркировка БАД",
    "BAD-NEG-EXPLICIT-EXCLUSION": (
        "на упаковке прямо показано исключение из категории БАД"
    ),
    "FLAM-POS-SEPARATE-IN-KIT": (
        "отдельный опасный предмет подтверждён как часть комплекта"
    ),
    "FLAM-POS-FLAMMABLE-CONTENTS": (
        "подтверждено продаваемое горючее содержимое"
    ),
    "FLAM-POS-STANDALONE-IGNITION": (
        "показан самостоятельный источник открытого огня"
    ),
    "FLAM-NEG-INTEGRATED-IGNITION": (
        "источник зажигания подтверждён как встроенная часть изделия"
    ),
    "FLAM-NEG-COMPONENT": (
        "горючий материал подтверждён только как компонент изделия"
    ),
    "FLAM-NEG-NOT-PART-OF-PRODUCT": (
        "опасный предмет показан лишь в контексте использования"
    ),
    "FLAM-NEG-NO-OPEN-FLAME": (
        "подтверждён механизм без самостоятельного открытого огня"
    ),
}

_VISUAL_CONTRAST = {
    "BAD-POS-DIRECT-MARKING": "Явное «не БАД» поддержало бы иной вывод.",
    "BAD-NEG-EXPLICIT-EXCLUSION": (
        "Прямая маркировка БАД без исключения поддержала бы иной вывод."
    ),
    "FLAM-POS-SEPARATE-IN-KIT": (
        "Подтверждение, что предмет не входит в комплект, изменило бы вывод."
    ),
    "FLAM-POS-FLAMMABLE-CONTENTS": (
        "Подтверждённая пустая конструкция поддержала бы иной вывод."
    ),
    "FLAM-POS-STANDALONE-IGNITION": (
        "Только встроенный поджиг без отдельного огня дал бы иной вывод."
    ),
    "FLAM-NEG-INTEGRATED-IGNITION": (
        "Отдельный источник огня в комплекте изменил бы вывод."
    ),
    "FLAM-NEG-COMPONENT": (
        "Самостоятельно продаваемое горючее изменило бы вывод."
    ),
    "FLAM-NEG-NOT-PART-OF-PRODUCT": (
        "Явное включение опасного предмета в продажу изменило бы вывод."
    ),
    "FLAM-NEG-NO-OPEN-FLAME": (
        "Самостоятельный источник открытого огня изменил бы вывод."
    ),
}


@dataclass(frozen=True)
class IntegratedReviewerRationale:
    """Rendered prose plus fail-closed lineage diagnostics."""

    text: str
    category: str
    prediction: int
    status: str
    text_status: str
    vlm_status: str
    rule_id: str | None = None
    evidence_source: str | None = None
    evidence_sha256: str | None = None
    issues: tuple[str, ...] = ()
    integration_version: str = INTEGRATION_VERSION


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        raise ValueError(f"invalid category: {category!r}")
    name = source.get("name", "")
    description = source.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError("name and description must be strings")
    product_id = source.get("id")
    if product_id is not None and (
        isinstance(product_id, bool) or not isinstance(product_id, int)
    ):
        raise ValueError("source id must be an integer when present")
    fold = source.get("fold")
    if fold is not None and (isinstance(fold, bool) or not isinstance(fold, int)):
        raise ValueError("source fold must be an integer when present")
    images = source.get("images", []) or []
    if not isinstance(images, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in images
    ):
        raise ValueError("source images must be non-empty strings")
    image_signature = source.get("image_content_signature", "") or ""
    if not isinstance(image_signature, str):
        raise ValueError("image_content_signature must be a string")
    return {
        "id": product_id,
        "fold": fold,
        "category": category,
        "name": name,
        "description": description,
        "images": list(images),
        "image_content_signature": image_signature,
        "ocr_text_by_image": source.get("ocr_text_by_image"),
    }


def source_projection_sha256(source: Mapping[str, Any]) -> str:
    """Hash the label-blind card projection used by a reviewed VLM row."""

    clean = _source(source)
    return _stable_sha256({
        "id": clean["id"],
        "category": clean["category"],
        "name": clean["name"],
        "description": clean["description"],
        "images": clean["images"],
        "image_content_signature": clean["image_content_signature"],
    })


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in _FORBIDDEN_BANK_KEYS:
                result.append(path)
            result.extend(_forbidden_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return result


def _clean_observation(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _WS.sub(" ", value).strip().replace("«", "").replace("»", "")
    if not cleaned or len(cleaned) > 96:
        return None
    if _VISUAL_ABSENCE.search(cleaned) or _VISUAL_UNCERTAINTY.search(cleaned):
        return None
    return cleaned


def _clean_span(value: Any, *, limit: int = 72) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _WS.sub(" ", value).strip().replace("«", "").replace("»", "")
    return cleaned if cleaned and len(cleaned) <= limit else None


def _validate_vlm_record(
    source: Mapping[str, Any], record: Mapping[str, Any] | None
) -> tuple[CompactVLMFact | None, str, tuple[str, ...], str | None]:
    """Validate a human-accepted visual bank row against the current card."""

    if record is None:
        return None, "not_supplied", (), None
    if not isinstance(record, Mapping):
        return None, "rejected_record", ("VLM bank row must be a mapping",), None
    issues: list[str] = []
    forbidden = _forbidden_paths(record)
    if forbidden:
        issues.append(f"verdict/label fields are forbidden: {forbidden[:5]}")
    if record.get("schema_version") != VLM_BANK_SCHEMA:
        issues.append("unsupported VLM bank schema")
    if record.get("id") != source["id"] or record.get("category") != source["category"]:
        issues.append("VLM bank id/category mismatch")
    if source["id"] is None:
        issues.append("a VLM bank row requires a source id")
    if not _SHA256.fullmatch(source["image_content_signature"]):
        issues.append("source has no bound image_content_signature")
    if record.get("source_projection_sha256") != source_projection_sha256(source):
        issues.append("source projection SHA mismatch")
    selected = record.get("input_image_ids")
    if not isinstance(selected, list) or not selected or any(
        not isinstance(item, str) or not item for item in (selected or [])
    ):
        issues.append("input_image_ids must be a non-empty string list")
        selected = []
    elif selected != source["images"][:len(selected)]:
        issues.append("input_image_ids must be the exact source-image prefix")
    raw = record.get("output")
    if not isinstance(raw, str) or not raw:
        issues.append("VLM output must be a non-empty string")
        raw = ""
    output_sha = _sha256_text(raw)
    if record.get("output_sha256") != output_sha:
        issues.append("VLM output SHA mismatch")
    if record.get("generation_status") != "valid":
        issues.append("generation_status is not valid")
    if record.get("contract_version") != COMPACT_VLM_CONTRACT_VERSION:
        issues.append("compact VLM contract version mismatch")
    if record.get("semantic_certification") != "human_accepted":
        issues.append("visual semantics are not human_accepted")
    reviewer = record.get("reviewed_by")
    if not isinstance(reviewer, str) or not reviewer.strip():
        issues.append("reviewed_by is required for visual acceptance")
    if record.get("reviewed_output_sha256") != output_sha:
        issues.append("human review is not bound to these output bytes")
    if issues:
        return None, "rejected_record", tuple(issues), output_sha

    parse_source = dict(source)
    parse_source["images"] = list(selected)
    try:
        fact = parse_compact_vlm_fact(raw, parse_source)
    except CompactVLMContractError as exc:
        return None, "rejected_output", tuple(exc.issues), output_sha

    if fact.modality == "insufficient" or fact.image_polarity == "none":
        return fact, "vlm_insufficient", (), output_sha
    observation = _clean_observation(fact.image_fact)
    if observation is None:
        return (
            fact,
            "rejected_visual_claim",
            ("image observation is absent, uncertain, unsupported, or too long",),
            output_sha,
        )
    if fact.state in {"empty", "explicitly_absent"}:
        return (
            fact,
            "rejected_visual_claim",
            ("visual evidence may not establish absence or empty state",),
            output_sha,
        )
    return fact, "accepted", (), output_sha


def _conclusion(category: str, prediction: int) -> str:
    relation = "относится" if prediction else "не относится"
    return f"ВЫВОД: Товар {relation} к категории «{category}»."


def _assemble(
    category: str,
    prediction: int,
    basis: str,
    contrast: str,
    *,
    conflict: str | None = None,
) -> str | None:
    lines = [
        _conclusion(category, prediction),
        f"ОСНОВАНИЕ: {basis}",
        f"КОНТРАСТ: {contrast}",
    ]
    if conflict is not None:
        lines.append(f"КОНФЛИКТ: {conflict}")
    text = "\n".join(lines)
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return None
    prefixes = ("ВЫВОД:", "ОСНОВАНИЕ:", "КОНТРАСТ:", "КОНФЛИКТ:")
    if len(lines) not in {3, 4} or any(
        not line.startswith(prefixes[index]) for index, line in enumerate(lines)
    ):
        return None
    return text


def _from_base(
    base: ReviewerSafeRationale,
    *,
    vlm_status: str,
    issues: Sequence[str] = (),
) -> IntegratedReviewerRationale:
    return IntegratedReviewerRationale(
        text=base.text,
        category=base.category,
        prediction=base.prediction,
        status=base.status,
        text_status=base.status,
        vlm_status=vlm_status,
        rule_id=base.rule_id,
        evidence_source=base.evidence_source,
        evidence_sha256=base.evidence_span_sha256,
        issues=tuple(issues),
    )


def _opposite_text_conflict(
    source: Mapping[str, Any],
    prediction: int,
    base: ReviewerSafeRationale,
    *,
    fact_record: Mapping[str, Any] | None,
    vlm_status: str,
    issues: Sequence[str] = (),
) -> IntegratedReviewerRationale | None:
    candidates = [
        candidate for candidate in inspect_safe_candidates(
            source, fact_record=fact_record
        )
        if candidate.direction != prediction
    ]
    directions = {candidate.direction for candidate in candidates}
    if len(directions) != 1 or not candidates:
        return None
    candidate = candidates[0]
    span = _clean_span(candidate.exact_span, limit=64)
    if span is None:
        return None
    location = "названии" if candidate.source == "name" else "описании"
    basis = "Вердикт классификатора сохранён."
    # Keep the warning inside the evaluator's 300-char wire limit even when
    # the exact opposing span uses its full 64-char allowance.
    contrast = "Согласованные данные позволили бы уточнить вывод."
    conflict = (
        f"В {location} дословно «{span}», что поддерживает иной вывод; "
        "нужна ручная проверка."
    )
    text = _assemble(
        source["category"], prediction, basis, contrast, conflict=conflict
    )
    if text is None:
        return None
    return IntegratedReviewerRationale(
        text=text,
        category=source["category"],
        prediction=prediction,
        status="review_text_verdict_conflict",
        text_status=base.status,
        vlm_status=vlm_status,
        rule_id=candidate.rule_id,
        evidence_source=candidate.source,
        evidence_sha256=_sha256_text(candidate.exact_span),
        issues=tuple(issues),
    )


def _vlm_rule(fact: CompactVLMFact) -> tuple[str | None, int | None]:
    verdict = fact.grounded_record.get("official_policy_verdict")
    rules = [
        str(rule) for rule in fact.grounded_record.get("policy_rule_ids", [])
        if str(rule) in SAFE_VISUAL_RULES
    ]
    if verdict not in {"yes", "no"} or len(rules) != 1:
        return None, None
    return rules[0], int(verdict == "yes")


def _render_vlm_support(
    source: Mapping[str, Any],
    prediction: int,
    base: ReviewerSafeRationale,
    fact: CompactVLMFact,
    rule_id: str,
    output_sha: str,
) -> IntegratedReviewerRationale | None:
    image_fact = _clean_observation(fact.image_fact)
    if image_fact is None or fact.image_id is None:
        return None
    if fact.modality == "agree" and fact.text_span:
        span = _clean_span(fact.text_span, limit=56)
        if span is None:
            return None
        basis = (
            f"В тексте дословно «{span}»; на фото {fact.image_id}: "
            f"«{image_fact}»; {_VISUAL_BASIS[rule_id]}."
        )
        status = "certified_text_image_agreement"
        evidence_source = "text+image"
    else:
        basis = (
            f"На фото {fact.image_id}: «{image_fact}»; "
            f"{_VISUAL_BASIS[rule_id]}."
        )
        status = "certified_image_observation"
        evidence_source = "image"
    text = _assemble(
        source["category"], prediction, basis, _VISUAL_CONTRAST[rule_id]
    )
    if text is None:
        return None
    return IntegratedReviewerRationale(
        text=text,
        category=source["category"],
        prediction=prediction,
        status=status,
        text_status=base.status,
        vlm_status="used",
        rule_id=rule_id,
        evidence_source=evidence_source,
        evidence_sha256=output_sha,
    )


def _render_vlm_verdict_conflict(
    source: Mapping[str, Any],
    prediction: int,
    base: ReviewerSafeRationale,
    fact: CompactVLMFact,
    rule_id: str,
    output_sha: str,
) -> IntegratedReviewerRationale | None:
    image_fact = _clean_observation(fact.image_fact)
    if image_fact is None or fact.image_id is None or len(image_fact) > 72:
        return None
    basis = "Итог классификатора сохранён без категоричного объяснения."
    contrast = (
        "Для БАД нужна согласованная прямая маркировка или явное исключение."
        if source["category"] == BAD
        else "Согласованные данные позволили бы дать однозначное обоснование."
    )
    conflict = (
        f"На фото {fact.image_id}: «{image_fact}»; проверенный факт "
        "поддерживает иной вывод, нужна ручная проверка."
    )
    text = _assemble(
        source["category"], prediction, basis, contrast, conflict=conflict
    )
    if text is None:
        return None
    return IntegratedReviewerRationale(
        text=text,
        category=source["category"],
        prediction=prediction,
        status="review_visual_verdict_conflict",
        text_status=base.status,
        vlm_status="evidence_verdict_conflict",
        rule_id=rule_id,
        evidence_source="image",
        evidence_sha256=output_sha,
    )


def _render_modality_conflict(
    source: Mapping[str, Any],
    prediction: int,
    base: ReviewerSafeRationale,
    fact: CompactVLMFact,
    output_sha: str,
) -> IntegratedReviewerRationale | None:
    """Render two positive, human-checked observations without choosing one."""

    span = _clean_span(fact.text_span, limit=48)
    image_fact = _clean_observation(fact.image_fact)
    if (
        span is None
        or image_fact is None
        or len(image_fact) > 64
        or fact.image_id is None
        or fact.text_source not in {"name", "description"}
    ):
        return None
    basis = "Итог классификатора сохранён; согласованной опоры нет."
    contrast = "Совпадение текста и фото позволило бы обосновать вывод однозначно."
    conflict = (
        f"В тексте «{span}», на фото {fact.image_id}: «{image_fact}»; "
        "нужна ручная проверка."
    )
    text = _assemble(
        source["category"], prediction, basis, contrast, conflict=conflict
    )
    if text is None:
        return None
    return IntegratedReviewerRationale(
        text=text,
        category=source["category"],
        prediction=prediction,
        status="review_modality_conflict",
        text_status=base.status,
        vlm_status="source_conflict",
        rule_id="POLICY-REVIEW-MODALITY-CONFLICT",
        evidence_source="text+image",
        evidence_sha256=output_sha,
    )


def build_integrated_reviewer_rationale(
    source: Mapping[str, Any],
    prediction: int | bool,
    *,
    fact_record: Mapping[str, Any] | None = None,
    vlm_record: Mapping[str, Any] | None = None,
) -> IntegratedReviewerRationale:
    """Render ``prediction`` without changing or recomputing it.

    Text evidence comes from :mod:`reviewer_safe_rationale_v1`.  Visual facts
    are optional and fail closed.  Evidence that contradicts the frozen
    verdict can only add a fourth review-warning line.
    """

    clean = _source(source)
    final_prediction = _prediction(prediction)
    base = build_reviewer_safe_rationale(
        clean, final_prediction, fact_record=fact_record
    )
    fact, vlm_status, issues, output_sha = _validate_vlm_record(clean, vlm_record)

    if fact is not None and vlm_status == "accepted":
        if fact.modality == "conflict":
            rendered = _render_modality_conflict(
                clean, final_prediction, base, fact, output_sha or ""
            )
            if rendered is not None:
                return rendered
            vlm_status = "unrenderable_source_conflict"
            issues = (*issues, "source conflict is not concise and positively grounded")
        else:
            rule_id, direction = _vlm_rule(fact)
            if rule_id is not None and direction is not None:
                if direction == final_prediction:
                    rendered = _render_vlm_support(
                        clean, final_prediction, base, fact, rule_id, output_sha or ""
                    )
                else:
                    rendered = _render_vlm_verdict_conflict(
                        clean, final_prediction, base, fact, rule_id, output_sha or ""
                    )
                if rendered is not None:
                    return rendered
            else:
                vlm_status = "policy_not_allowlisted"
                issues = (*issues, "validated visual fact has no safe reviewer rule")

    # A literal safe fact contradicting the anchor is more useful to a reviewer
    # as an explicit warning than as a generic fallback.  This is especially
    # important for noisy BAD annotations: never invent a rationale that hides
    # a direct marking/exclusion disagreement.
    if base.status == "fallback_disagreement":
        conflict = _opposite_text_conflict(
            clean,
            final_prediction,
            base,
            fact_record=fact_record,
            vlm_status=vlm_status,
            issues=issues,
        )
        if conflict is not None:
            return conflict

    return _from_base(base, vlm_status=vlm_status, issues=issues)


def render_integrated_reviewer_rationale(
    source: Mapping[str, Any],
    prediction: int | bool,
    *,
    fact_record: Mapping[str, Any] | None = None,
    vlm_record: Mapping[str, Any] | None = None,
) -> str:
    return build_integrated_reviewer_rationale(
        source,
        prediction,
        fact_record=fact_record,
        vlm_record=vlm_record,
    ).text


__all__ = [
    "INTEGRATION_VERSION",
    "MAX_CHARS",
    "MIN_CHARS",
    "SAFE_VISUAL_RULES",
    "VLM_BANK_SCHEMA",
    "IntegratedReviewerRationale",
    "build_integrated_reviewer_rationale",
    "render_integrated_reviewer_rationale",
    "source_projection_sha256",
]
