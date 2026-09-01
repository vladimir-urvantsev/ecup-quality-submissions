"""Versioned, fail-closed deterministic rationale renderer for reviewers.

The verdict is an immutable input.  This module reads no ``label``, ``gold``,
probability, or score field from the product card and cannot alter the verdict.
It emits exactly three Russian lines in 50..300 characters.  Grounded outputs
quote a literal span from ``name`` or ``description``; everything else becomes
an honest ``по имеющимся данным`` fallback.

An optional label-blind auxiliary fact-bank row may provide stronger lineage,
but only after source SHA, id, offsets, literal span, and the renderer's own
strict semantic allowlist all pass.  Series metadata is never a feature.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from text_rule_rationales import extract_rule, source_text_sha256


RENDERER_VERSION = "reviewer-safe-rationale-v1.0.1"
CONTRACT_SCHEMA = "reviewer-safe-rationale-contract-v1"
BANK_SCHEMA = "series-aware-masked-aux-bank-v1"
MIN_CHARS = 50
MAX_CHARS = 300
BAD = "БАД"
FLAM = "Легковоспламеняющиеся"
VALID_CATEGORIES = frozenset({BAD, FLAM})

SAFE_RULE_ALLOWLIST = frozenset({
    "BAD-POS-DIRECT-MARKING",
    "BAD-NEG-EXPLICIT-EXCLUSION",
    "FLAM-NEG-NOT-INCLUDED",
    "FLAM-NEG-EMPTY-CONTENTS",
    "FLAM-NEG-COMPONENT",
    "FLAM-NEG-INTEGRATED-IGNITION",
    "FLAM-POS-STANDALONE-IGNITION",
    "FLAM-POS-FLAMMABLE-CONTENTS",
    "FLAM-POS-SEPARATE-IN-KIT",
})

_TAG = re.compile(r"<[^>]{0,500}>")
_WS = re.compile(r"\s+")
_EXPLICIT_INTEGRATED = re.compile(
    r"(?:встро\w*|оснащ\w*|оборудован\w*|с\s+(?:автоматическ\w*\s+)?пьезо(?:поджиг|зажиг)\w*)",
    re.IGNORECASE | re.UNICODE,
)
_STRICT_STANDALONE = re.compile(
    r"(?:^|[^а-яёa-z0-9])(?:спичк\w*|зажигалк\w*)(?:$|[^а-яёa-z0-9])",
    re.IGNORECASE | re.UNICODE,
)
_AMBIGUOUS_PYRO = re.compile(
    r"бенгальск|фонтан|пиротех|хлопуш|петард|салют|фейерверк",
    re.IGNORECASE | re.UNICODE,
)
_STRICT_SOLD_GAS = re.compile(
    r"(?:\b(?:газ|бутан|пропан|mapp|мапп)\b[^\n]{0,35}\b(?:для\s+заправк|балл?он)|"
    r"\bбалл?он\w*[^\n]{0,35}\b(?:mapp|мапп|бутан|пропан)\b|"
    r"\b(?:бутан|пропан)\w*\s+для\s+заправк)",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_NOT_INCLUDED = re.compile(
    # All branches require an explicit lexical absence/separate-purchase cue;
    # product/device identity alone never passes.  ``exact_span`` is already
    # constrained to the source and to a FLAM risk item by the upstream,
    # label-blind extractor.  Keep common catalogue inflections here so safe
    # phrases such as "баллон приобретается отдельно" do not needlessly fall
    # back merely because they say "комплектация" instead of "комплект".
    r"(?:"
    r"не\s+(?:вход|включ)\w*\s+в\s+(?:комплект|комплектац|набор|поставк)\w*|"
    r"в\s+(?:комплект|комплектац|набор|поставк)\w*"
    r"(?:\s+поставк\w*)?\s+не\s+(?:вход|включ)\w*|"
    r"(?:приобрета|покупа)\w*\s+отдельно|"
    r"(?:поставля|прода)\w*\s+без\s+|"
    r"без\s+(?:спич|зажигал|балл?он|газ|топлив|угл)|"
    r"(?:балл?он|спич|зажигал|топлив)\w*[^\n]{0,25}"
    r"не\s+(?:прилага|поставля|вход|включ)\w*"
    r")",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_EMPTY = re.compile(
    r"(?:пуст\w*|внутри\s+нет\s+(?:газа|топлива)|без\s+(?:газа|топлива|горючего))",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_NONPYRO = re.compile(
    r"(?:не\s+пиротех\w*|пневмохлопуш\w*)", re.IGNORECASE | re.UNICODE
)
_EXPLICIT_INCLUDED_RISK = re.compile(
    r"(?:\bсо\s+спичк\w*|\bс\s+зажигалк\w*|"
    r"\bкомплектац\w*\s*:\s*[^.!?\n<>]{0,55}\bпакет\s+с\s+угл[ёе]м\b)",
    re.IGNORECASE | re.UNICODE,
)

# These supplemental patterns deliberately require the role/state/mechanism in
# the same short literal span.  They are narrower than the classifier rules:
# their only job is to make a reviewer explanation specific without ever
# changing the already frozen verdict.
_BAD_FULL_DIRECT_MARKING = re.compile(
    # This bypasses upstream sports-term precedence only for a short
    # definitional sentence that names the L-carnitine product itself.  A bare
    # BAD phrase elsewhere in sports-product boilerplate remains quarantined.
    r"\b(?:l|л)[-\s]?карнитин\w*\s*[–—-]\s*"
    r"биологически\s+активн\w*\s+добавк\w*"
    r"(?:\s*\(\s*бад\s*\))?",
    re.IGNORECASE | re.UNICODE,
)
_BAD_ANY_EXCLUSION = re.compile(
    r"\bне\s+(?:явля\w*\s+)?(?:б\.?\s*а\.?\s*д\.?|"
    r"биологически\s+активн\w*\s+добавк\w*)",
    re.IGNORECASE | re.UNICODE,
)
_BAD_DEFINED_SPORTS_NUTRITION = re.compile(
    r"\b(?:жидк\w*\s+)?(?:l|л)[-\s]?карнитин\w*"
    r"(?:\s*\([^)]{0,28}\))?\s*[–—-]\s*это\s+"
    r"(?:эффективн\w*\s+)?спортивн\w*\s+питан\w*",
    re.IGNORECASE | re.UNICODE,
)
_BAD_CONTEXT_ONLY_ACRONYM = re.compile(
    r"(?:лекарств\w*|препарат\w*)\s+(?:и|или)\s+"
    r"б\.?\s*а\.?\s*д\.?\s*$",
    re.IGNORECASE | re.UNICODE,
)
_BAD_CONTEXT_GENERIC_CLASS = re.compile(
    r"\bбиологически\s+активн\w*\s+добавк\w*\s*,?\s*"
    r"содержащ\w*\b",
    re.IGNORECASE | re.UNICODE,
)
_BAD_SPORT_MAIN_NAME = re.compile(
    r"^\s*(?:креатин(?:\s+моногидрат)?|протеин)\b",
    re.IGNORECASE | re.UNICODE,
)
_FLAM_FILLED_CYLINDERS_IN_TITLE = re.compile(
    r"\+\s*\d+\s+(?:цангов\w*\s+)?балл?он\w*\s+с\s+газом\b",
    re.IGNORECASE | re.UNICODE,
)
_FLAM_PACKED_MATCHES = re.compile(
    r"(?:и\s+)?спички\b[^.!?\n<>]{0,48}\.?"
    r"(?:\s*<br\s*/?>){1,2}\s*вс[ёе]\s+это\s+аккуратно\s+упаковано",
    re.IGNORECASE | re.UNICODE,
)
_FLAM_EXPLICIT_PYRO_CHARGE = re.compile(
    r"(?:\bхлопушк\w*\s+с\s+порохов\w*\s+заряд\w*|"
    r"\bпироэлемент\s*:\s*петард\w*[^.!?\n<>]{0,28}|"
    r"\bпиротехническ\w*\s+заряд\w*)",
    re.IGNORECASE | re.UNICODE,
)
_FLAM_EXPLICIT_PNEUMATIC = re.compile(
    r"\bпневмохлопушка\s+[–—-]\s+это\s+не\s+пиротехника\b",
    re.IGNORECASE | re.UNICODE,
)
_FLAM_UNFILLED_DEVICE = re.compile(
    r"(?:\bгорелка\s+поставляется\s+без\s+газа\b|"
    r"\bбалл?он\s+в\s+комплекте\s*[-–—:]?\s*нет\b|"
    r"\bпоставляется\s+не\s+заправленн\w*\b)",
    re.IGNORECASE | re.UNICODE,
)
_FLAM_KIT_LIST_LIGHTER = re.compile(
    r"[-•]\s*пьезо-зажигалка\b", re.IGNORECASE | re.UNICODE
)
_FLAM_ROLL_WITH_IGNITION_STRIP = re.compile(
    r"\bс\s+терк\w*\s+для\s+поджига\b", re.IGNORECASE | re.UNICODE
)
_FLAM_MATCH_INSIDE_ROLL = re.compile(
    r"\bспичка\s+внутри\s+каждого\s+ролла\b", re.IGNORECASE | re.UNICODE
)
_FLAM_CAKE_FOUNTAIN = re.compile(
    r"\bсвеча-фонтан\b",
    re.IGNORECASE | re.UNICODE,
)
_FLAM_BURNING_DURATION = re.compile(
    r"\bвремя\s+горения\s+\d+\s*секунд\w*", re.IGNORECASE | re.UNICODE
)
_FLAM_DRY_FUEL_MATCH_KIT = re.compile(
    r"\bсухое\s+горючее\b[^.!?\n<>]{0,72}\bспичк\w*\b",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class EvidenceCandidate:
    family: str
    direction: int
    source: str
    exact_span: str
    origin: str
    rule_id: str | None = None


@dataclass(frozen=True)
class ReviewerSafeRationale:
    text: str
    category: str
    prediction: int
    status: str
    family: str | None = None
    rule_id: str | None = None
    evidence_source: str | None = None
    evidence_span_sha256: str | None = None
    renderer_version: str = RENDERER_VERSION


_BASIS = {
    "bad_direct_marking": "цитата прямо маркирует товар как БАД",
    "bad_explicit_exclusion": "цитата прямо исключает товар из БАД",
    "bad_explicit_sports_nutrition": "цитата прямо описывает товар как спортивное питание",
    "bad_sports_product": "цитата прямо называет товар протеином или креатином",
    "flam_not_included": "цитата прямо говорит, что опасный предмет не входит в комплект",
    "flam_empty_container": "цитата прямо указывает на пустую ёмкость без содержимого",
    "flam_component": "цитата прямо называет продаваемый аксессуар или компонент",
    "flam_integrated_ignition": "цитата прямо описывает встроенный поджиг устройства",
    "flam_standalone_ignition": "цитата прямо называет самостоятельный источник огня",
    "flam_sold_gas": "цитата прямо называет продаваемый горючий газ",
    "flam_explicit_nonpyro": "цитата прямо исключает пиротехнический механизм",
    "flam_separate_in_kit": "цитата прямо включает опасный предмет в комплект",
    "flam_pyrotechnic_charge": "цитата прямо называет пиротехнический заряд товара",
    "flam_unfilled_device": "цитата прямо указывает, что горелка продаётся без газа",
    "flam_ignition_strip": "цитата прямо включает тёрку для поджига в товар",
    "flam_integrated_match": "цитата прямо описывает спичку внутри каждого ролла",
    "flam_cake_fountain": "цитата прямо описывает горящую свечу-фонтан",
}

_CONTRAST = {
    "bad_direct_marking": "Прямая фраза «не БАД» поддержала бы противоположный вывод.",
    "bad_explicit_exclusion": "Прямая маркировка БАД без такого исключения поддержала бы иной вывод.",
    "bad_explicit_sports_nutrition": "Прямая маркировка товара как БАД поддержала бы иной вывод.",
    "bad_sports_product": "Прямая маркировка товара как БАД поддержала бы иной вывод.",
    "flam_not_included": "Прямое указание, что предмет входит в комплект, поддержало бы иной вывод.",
    "flam_empty_container": "Прямое указание на горючее содержимое поддержало бы иной вывод.",
    "flam_component": "Прямое указание на самостоятельный источник огня поддержало бы иной вывод.",
    "flam_integrated_ignition": "Отдельный источник огня в комплекте поддержал бы иной вывод.",
    "flam_standalone_ignition": "Прямое указание лишь на встроенный поджиг поддержало бы иной вывод.",
    "flam_sold_gas": "Прямое указание на пустую ёмкость поддержало бы иной вывод.",
    "flam_explicit_nonpyro": "Прямое указание на пиротехнический заряд поддержало бы иной вывод.",
    "flam_separate_in_kit": "Если предмет не входит в комплект, вывод был бы иным.",
    "flam_pyrotechnic_charge": "Пневматический механизм без пирозаряда поддержал бы иной вывод.",
    "flam_unfilled_device": "Прямое указание на заправленный товар поддержало бы иной вывод.",
    "flam_ignition_strip": "Если тёрка не входит в товар, вывод был бы иным.",
    "flam_integrated_match": "Отдельная спичка в комплекте поддержала бы иной вывод.",
    "flam_cake_fountain": "Обычная свеча без эффекта фонтана поддержала бы иной вывод.",
}

_RULE_FAMILY_DIRECTION = {
    "BAD-POS-DIRECT-MARKING": ("bad_direct_marking", 1),
    "BAD-NEG-EXPLICIT-EXCLUSION": ("bad_explicit_exclusion", 0),
    "FLAM-NEG-NOT-INCLUDED": ("flam_not_included", 0),
    "FLAM-NEG-EMPTY-CONTENTS": ("flam_empty_container", 0),
    "FLAM-NEG-COMPONENT": ("flam_component", 0),
    "FLAM-NEG-INTEGRATED-IGNITION": ("flam_integrated_ignition", 0),
    "FLAM-POS-STANDALONE-IGNITION": ("flam_standalone_ignition", 1),
    "FLAM-POS-FLAMMABLE-CONTENTS": ("flam_sold_gas", 1),
    "FLAM-POS-SEPARATE-IN-KIT": ("flam_separate_in_kit", 1),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _prediction(value: int | bool) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
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
        not isinstance(item, str) for item in images
    ):
        raise ValueError("images must be a string list")
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
    }


def _source_projection_sha256(source: Mapping[str, Any]) -> str | None:
    if source["id"] is None or source["fold"] is None:
        return None
    return _stable_sha256({
        "id": source["id"],
        "fold": source["fold"],
        "category": source["category"],
        "name": source["name"],
        "description": source["description"],
        "images": source["images"],
        "image_content_signature": source["image_content_signature"],
    })


def _literal_span(
    source: Mapping[str, Any], field: str, span: Any
) -> str | None:
    if field not in {"name", "description"} or not isinstance(span, str):
        return None
    if not span or len(span) > 96 or span not in source[field]:
        return None
    # Rendering removes only markup/formatting after literal membership was
    # proved. Refuse a span whose meaningful text disappears.
    rendered = html.unescape(_TAG.sub(" ", span))
    rendered = _WS.sub(" ", rendered).strip().replace("«", "").replace("»", "")
    if not rendered or len(rendered) > 96:
        return None
    return rendered


def _direct_bad_marker_has_product_referent(
    source: Mapping[str, Any], field: str, literal: str
) -> bool:
    """Reject a bare acronym used only as the object of compatibility prose."""

    text = source.get(field, "")
    matches = list(re.finditer(re.escape(literal), text, re.IGNORECASE | re.UNICODE))
    if not matches:
        return False
    if re.fullmatch(
        r"\s*биологически\s+активн\w*\s+добавк\w*\s*",
        literal,
        re.IGNORECASE | re.UNICODE,
    ):
        return any(
            not _BAD_CONTEXT_GENERIC_CLASS.search(
                text[match.start():min(len(text), match.end() + 32)]
            )
            for match in matches
        )
    if not re.fullmatch(
        r"\s*б\.?\s*а\.?\s*д\.?\s*", literal, re.IGNORECASE | re.UNICODE
    ):
        return True
    for match in matches:
        context = text[max(0, match.start() - 64):match.end()]
        parenthesized = (
            match.start() > 0
            and match.end() < len(text)
            and text[match.start() - 1] == "("
            and text[match.end()] == ")"
        )
        if parenthesized and re.search(
            r"\bлекарственн\w*\s+препарат\w*\b",
            source.get("description", ""),
            re.IGNORECASE | re.UNICODE,
        ):
            continue
        if not _BAD_CONTEXT_ONLY_ACRONYM.search(context):
            return True
    return False


def _regex_candidate(
    source: Mapping[str, Any],
    field: str,
    pattern: re.Pattern[str],
    family: str,
    direction: int,
) -> EvidenceCandidate | None:
    match = pattern.search(source[field])
    if match is None:
        return None
    literal = _literal_span(source, field, match.group(0))
    if literal is None:
        return None
    return EvidenceCandidate(
        family=family,
        direction=direction,
        source=field,
        exact_span=literal,
        origin="deterministic_reviewer_guard",
    )


def _supplemental_candidates(source: Mapping[str, Any]) -> list[EvidenceCandidate]:
    """Small high-precision gaps whose full cue is absent from upstream spans."""

    candidates: list[EvidenceCandidate] = []
    if source["category"] == BAD:
        combined = f"{source['name']}\n{source['description']}"
        if not _BAD_ANY_EXCLUSION.search(combined):
            for field in ("name", "description"):
                candidate = _regex_candidate(
                    source, field, _BAD_FULL_DIRECT_MARKING,
                    "bad_direct_marking", 1,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    break
        for field in ("name", "description"):
            candidate = _regex_candidate(
                source, field, _BAD_DEFINED_SPORTS_NUTRITION,
                "bad_explicit_sports_nutrition", 0,
            )
            if candidate is not None:
                candidates.append(candidate)
                break
        candidate = _regex_candidate(
            source, "name", _BAD_SPORT_MAIN_NAME,
            "bad_sports_product", 0,
        )
        if candidate is not None:
            candidates.append(candidate)
        return candidates

    supplemental = (
        ("name", _FLAM_FILLED_CYLINDERS_IN_TITLE, "flam_separate_in_kit", 1),
        ("description", _FLAM_PACKED_MATCHES, "flam_separate_in_kit", 1),
        ("name", _FLAM_EXPLICIT_PYRO_CHARGE, "flam_pyrotechnic_charge", 1),
        ("description", _FLAM_EXPLICIT_PYRO_CHARGE, "flam_pyrotechnic_charge", 1),
        ("description", _FLAM_EXPLICIT_PNEUMATIC, "flam_explicit_nonpyro", 0),
    )
    for field, pattern, family, direction in supplemental:
        candidate = _regex_candidate(source, field, pattern, family, direction)
        if candidate is not None:
            candidates.append(candidate)
    if re.search(r"\bгорелк\w*\b", source["name"], re.IGNORECASE | re.UNICODE):
        candidate = _regex_candidate(
            source, "description", _FLAM_UNFILLED_DEVICE,
            "flam_unfilled_device", 0,
        )
        if candidate is not None:
            candidates.append(candidate)
    if (
        re.search(r"\b(?:набор|календар)\w*\b", source["name"], re.IGNORECASE | re.UNICODE)
        and re.search(r"\bполн\w*\s+состав\s+набора\b", source["description"], re.IGNORECASE | re.UNICODE)
    ):
        candidate = _regex_candidate(
            source, "description", _FLAM_KIT_LIST_LIGHTER,
            "flam_separate_in_kit", 1,
        )
        if candidate is not None:
            candidates.append(candidate)
    for field, pattern, family, direction in (
        ("name", _FLAM_ROLL_WITH_IGNITION_STRIP, "flam_ignition_strip", 1),
        ("description", _FLAM_MATCH_INSIDE_ROLL, "flam_integrated_match", 0),
        ("name", _FLAM_DRY_FUEL_MATCH_KIT, "flam_separate_in_kit", 1),
    ):
        candidate = _regex_candidate(source, field, pattern, family, direction)
        if candidate is not None:
            candidates.append(candidate)
    if (
        _FLAM_BURNING_DURATION.search(source["description"])
        and re.search(r"\bплам\w*\b", source["description"], re.IGNORECASE | re.UNICODE)
    ):
        candidate = _regex_candidate(
            source, "name", _FLAM_CAKE_FOUNTAIN,
            "flam_cake_fountain", 1,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _strict_rule_candidate(
    source: Mapping[str, Any], extracted: Mapping[str, Any]
) -> EvidenceCandidate | None:
    rule_id = str(extracted.get("rule_id", ""))
    if rule_id not in SAFE_RULE_ALLOWLIST or rule_id not in _RULE_FAMILY_DIRECTION:
        return None
    evidence = extracted.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    evidence_source = str(evidence.get("source", ""))
    literal = _literal_span(source, evidence_source, evidence.get("exact_span"))
    if literal is None:
        return None
    family, direction = _RULE_FAMILY_DIRECTION[rule_id]
    if int(extracted.get("rule_prediction", -1)) != direction:
        return None
    if rule_id == "BAD-POS-DIRECT-MARKING" and not _direct_bad_marker_has_product_referent(
        source, evidence_source, literal
    ):
        return None
    if rule_id == "FLAM-NEG-NOT-INCLUDED" and not _EXPLICIT_NOT_INCLUDED.search(literal):
        return None
    if rule_id == "FLAM-NEG-EMPTY-CONTENTS" and not _EXPLICIT_EMPTY.search(literal):
        return None
    if rule_id == "FLAM-NEG-INTEGRATED-IGNITION" and not _EXPLICIT_INTEGRATED.search(literal):
        return None
    if rule_id == "FLAM-POS-STANDALONE-IGNITION" and (
        not _STRICT_STANDALONE.search(literal) or _AMBIGUOUS_PYRO.search(literal)
    ):
        return None
    if rule_id == "FLAM-POS-FLAMMABLE-CONTENTS" and not _STRICT_SOLD_GAS.search(literal):
        return None
    if rule_id == "FLAM-POS-SEPARATE-IN-KIT" and not _EXPLICIT_INCLUDED_RISK.search(literal):
        return None
    return EvidenceCandidate(
        family=family,
        direction=direction,
        source=evidence_source,
        exact_span=literal,
        origin="deterministic_text_rule",
        rule_id=rule_id,
    )


def _valid_bank_record(
    source: Mapping[str, Any], record: Mapping[str, Any] | None
) -> bool:
    if not isinstance(record, Mapping):
        return False
    if record.get("schema_version") != BANK_SCHEMA:
        return False
    if record.get("id") != source["id"] or record.get("category") != source["category"]:
        return False
    projection_sha = _source_projection_sha256(source)
    if projection_sha is None or record.get("source_projection_sha256") != projection_sha:
        return False
    if record.get("source_text_sha256") != source_text_sha256(source):
        return False
    slots = record.get("slots")
    return isinstance(slots, list) and len(slots) == 2


def _bank_slot_candidate(
    source: Mapping[str, Any], slot: Mapping[str, Any]
) -> EvidenceCandidate | None:
    if not slot.get("observed"):
        return None
    evidence = slot.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    field = str(evidence.get("source", ""))
    raw_span = evidence.get("exact_span")
    literal = _literal_span(source, field, raw_span)
    if literal is None:
        return None
    if not isinstance(raw_span, str):
        return None
    start, end = evidence.get("start"), evidence.get("end")
    if (
        isinstance(start, bool) or not isinstance(start, int)
        or isinstance(end, bool) or not isinstance(end, int)
        or start < 0 or end != start + len(raw_span)
        or source[field][start:end] != raw_span
        or evidence.get("span_sha256")
        != hashlib.sha256(raw_span.encode("utf-8")).hexdigest()
    ):
        return None
    lineages = slot.get("lineages")
    if not isinstance(lineages, list) or not lineages:
        return None
    origins = {str(lineage.get("origin", "")) for lineage in lineages if isinstance(lineage, Mapping)}
    allowed_origins = {"deterministic_exact_text", "human_accepted_grounded_fact"}
    if not origins or not origins.issubset(allowed_origins):
        return None
    origin = "human_accepted_exact_text" if "human_accepted_grounded_fact" in origins else "auxiliary_exact_text"
    fact_name = str(slot.get("fact_name", ""))
    fact_kind = str(slot.get("fact_kind", ""))
    h1 = set(slot.get("h1_topology", {}).get("values", []))
    h2 = set(slot.get("h2_state_mechanism", {}).get("values", []))

    family_direction: tuple[str, int] | None = None
    if fact_kind in {"bad_marking", "dietary_supplement_marking"}:
        if _direct_bad_marker_has_product_referent(source, field, literal):
            family_direction = ("bad_direct_marking", 1)
    elif fact_kind == "explicit_not_bad":
        family_direction = ("bad_explicit_exclusion", 0)
    elif fact_name == "explicit not included risk item" and _EXPLICIT_NOT_INCLUDED.search(literal):
        family_direction = ("flam_not_included", 0)
    elif fact_name == "explicit empty container" and _EXPLICIT_EMPTY.search(literal):
        family_direction = ("flam_empty_container", 0)
    elif fact_name == "component or accessory":
        family_direction = ("flam_component", 0)
    elif (
        fact_name in {"integrated ignition", "пьезоподжиг"}
        and "spark_only" in h2
        and _EXPLICIT_INTEGRATED.search(literal)
    ):
        family_direction = ("flam_integrated_ignition", 0)
    elif (
        fact_name == "standalone ignition object"
        and fact_kind == "ignition_source"
        and "main_or_self_contained" in h1
        and "open_flame" in h2
        and _STRICT_STANDALONE.search(literal)
        and not _AMBIGUOUS_PYRO.search(literal)
    ):
        family_direction = ("flam_standalone_ignition", 1)
    elif (
        fact_kind == "flammable_gas"
        and "main_or_self_contained" in h1
        and "fuel_present" in h2
        and _STRICT_SOLD_GAS.search(literal)
    ):
        family_direction = ("flam_sold_gas", 1)
    elif (
        fact_kind == "non_ignition_device"
        and "pneumatic_or_nonflame" in h2
        and _EXPLICIT_NONPYRO.search(literal)
    ):
        family_direction = ("flam_explicit_nonpyro", 0)
    if family_direction is None:
        return None
    family, direction = family_direction
    return EvidenceCandidate(
        family=family,
        direction=direction,
        source=field,
        exact_span=literal,
        origin=origin,
        rule_id=None,
    )


def inspect_safe_candidates(
    source: Mapping[str, Any],
    *,
    fact_record: Mapping[str, Any] | None = None,
) -> tuple[EvidenceCandidate, ...]:
    """Return deduplicated safe candidates without reading a verdict/label."""

    clean = _source(source)
    candidates: list[EvidenceCandidate] = []
    try:
        extracted = extract_rule(clean)
    except (TypeError, ValueError):
        extracted = None
    if isinstance(extracted, Mapping):
        candidate = _strict_rule_candidate(clean, extracted)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(_supplemental_candidates(clean))
    if _valid_bank_record(clean, fact_record):
        for slot in fact_record.get("slots", []):
            if not isinstance(slot, Mapping):
                continue
            candidate = _bank_slot_candidate(clean, slot)
            if candidate is not None:
                candidates.append(candidate)
    deduplicated: dict[tuple[Any, ...], EvidenceCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.family,
            candidate.direction,
            candidate.source,
            candidate.exact_span.casefold(),
        )
        # Direct deterministic evidence is preferred when the bank repeats it;
        # human-accepted evidence remains available when it is genuinely new.
        if key not in deduplicated or candidate.origin == "deterministic_text_rule":
            deduplicated[key] = candidate
    return tuple(sorted(
        deduplicated.values(),
        key=lambda item: (
            0 if item.source == "name" else 1,
            0 if item.origin == "deterministic_text_rule" else 1,
            item.family,
            item.exact_span.casefold(),
        ),
    ))


def _conclusion(category: str, prediction: int) -> str:
    relation = "относится" if prediction else "не относится"
    return f"ВЫВОД: Товар {relation} к категории «{category}»."


def _assemble(category: str, prediction: int, basis: str, contrast: str) -> str:
    text = "\n".join((
        _conclusion(category, prediction),
        f"ОСНОВАНИЕ: {basis}",
        f"КОНТРАСТ: {contrast}",
    ))
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        raise ValueError(f"renderer length contract failed: {len(text)}")
    lines = text.splitlines()
    if len(lines) != 3 or not all(
        line.startswith(prefix)
        for line, prefix in zip(lines, ("ВЫВОД:", "ОСНОВАНИЕ:", "КОНТРАСТ:"))
    ):
        raise ValueError("renderer three-line contract failed")
    return text


def _fallback(category: str, prediction: int, status: str) -> ReviewerSafeRationale:
    text = _assemble(
        category,
        prediction,
        "По имеющимся данным безопасной дословной опоры нет; сохранён вердикт классификатора.",
        "Для уточнения случая требуется ручная проверка значения и роли "
        "указанных в карточке признаков.",
    )
    return ReviewerSafeRationale(
        text=text,
        category=category,
        prediction=prediction,
        status=status,
    )


def build_reviewer_safe_rationale(
    source: Mapping[str, Any],
    prediction: int | bool,
    *,
    fact_record: Mapping[str, Any] | None = None,
) -> ReviewerSafeRationale:
    """Render a fixed prediction; evidence can explain but never modify it."""

    clean = _source(source)
    final_prediction = _prediction(prediction)
    candidates = inspect_safe_candidates(clean, fact_record=fact_record)
    if not candidates:
        return _fallback(clean["category"], final_prediction, "fallback_no_safe_evidence")
    directions = {candidate.direction for candidate in candidates}
    if len(directions) != 1 or final_prediction not in directions:
        return _fallback(clean["category"], final_prediction, "fallback_disagreement")
    candidate = candidates[0]
    quote = candidate.exact_span.replace("\n", " ")
    basis = f"В {('названии' if candidate.source == 'name' else 'описании')} дословно: «{quote}»; {_BASIS[candidate.family]}."
    try:
        text = _assemble(
            clean["category"],
            final_prediction,
            basis,
            _CONTRAST[candidate.family],
        )
    except ValueError:
        # Evidence is never silently clipped because losing a negation or role
        # phrase could change its meaning. Long evidence therefore fails closed.
        return _fallback(clean["category"], final_prediction, "fallback_length_guard")
    return ReviewerSafeRationale(
        text=text,
        category=clean["category"],
        prediction=final_prediction,
        status="certified_exact_span",
        family=candidate.family,
        rule_id=candidate.rule_id,
        evidence_source=candidate.source,
        evidence_span_sha256=hashlib.sha256(
            candidate.exact_span.encode("utf-8")
        ).hexdigest(),
    )


def render_reviewer_safe_rationale(
    source: Mapping[str, Any],
    prediction: int | bool,
    *,
    fact_record: Mapping[str, Any] | None = None,
) -> str:
    return build_reviewer_safe_rationale(
        source, prediction, fact_record=fact_record
    ).text


__all__ = [
    "MAX_CHARS",
    "MIN_CHARS",
    "RENDERER_VERSION",
    "SAFE_RULE_ALLOWLIST",
    "EvidenceCandidate",
    "ReviewerSafeRationale",
    "build_reviewer_safe_rationale",
    "inspect_safe_candidates",
    "render_reviewer_safe_rationale",
]
