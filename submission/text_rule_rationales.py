"""Conservative, deterministic text-only rationale extraction.

The module intentionally does less than the regex classifier in
``rules_baseline.py``.  It emits a training target only when the product card
contains a short, policy-relevant textual cue whose role is sufficiently clear.
Extraction never reads the observed label.  The label is attached afterwards
solely to mark whether the extracted policy prediction agrees with the dataset.

This is a compact wire format for a label-first auxiliary objective.  It is not
a free-form chain of thought and it does not use images or a language model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Pattern, Sequence


SCHEMA_VERSION = "text-rule-rationale-v1"
GENERATOR_VERSION = "text-rule-rationale-generator-v1.1.1"

BAD_CATEGORY = "БАД"
FLAM_CATEGORY = "Легковоспламеняющиеся"
VALID_CATEGORIES = frozenset({BAD_CATEGORY, FLAM_CATEGORY})


def _rx(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


# БАД: a direct exclusion takes precedence over sports-food wording, which in
# turn takes precedence over a positive marking.  The sports-food patterns are
# deliberately narrower than the old baseline: generic health/training wording
# in the description is not enough.
_BAD_EXPLICIT_NEGATIVE = (
    _rx(
        r"\bне\s+относ\w*\s+к\s+категори\w*\s+"
        r"(?:б\.?\s*а\.?\s*д\.?(?!\w)|биологически\s+активн\w*\s+добавк\w*)"
    ),
    _rx(
        r"\bне\s+явля\w*\s+"
        r"(?:(?:лекарственн\w*\s+средств\w*|лекарств\w*)\s+(?:и|или)\s+)?"
        r"(?:б\.?\s*а\.?\s*д\.?(?!\w)|биологически\s+активн\w*\s+добавк\w*)"
    ),
    _rx(
        r"\bне\s+(?:явля\w*|счита\w*|относ\w*)\s+(?:к\s+)?"
        r"(?:б\.?\s*а\.?\s*д\.?\w*|биологически\s+активн\w*\s+добавк\w*)"
    ),
    _rx(r"\bне\s+б\.?\s*а\.?\s*д\.?(?!\w)"),
    _rx(r"\bб\.?\s*а\.?\s*д\.?ом\s+не\s+явля\w*"),
    _rx(r"\bбиологически\s+активн\w*\s+добавк(?:ой|ами)\s+не\s+явля\w*"),
)
_BAD_SPORT_PHRASE = _rx(r"\bспортивн\w*\s+(?:питан\w*|добавк\w*)")
_BAD_SPORT_EXPLICIT = (
    _rx(
        r"\b(?:явля\w*|счита\w*|относ\w*\s+к|категори\w*|продукт\w*)"
        r"[^,.;!?\n<>]{0,35}\bспортивн\w*\s+(?:питан\w*|добавк\w*)"
    ),
    _rx(r"\bпредназначен\w*\s+(?:для|как)\s+спортивн\w*\s+питан\w*"),
)
_BAD_SPORT_MAIN_PRODUCT = (
    _rx(r"(?<![\w])(?:bcaa|бцаа)(?![\w])"),
    _rx(r"\b(?:l|л)[-\s]?карнитин\w*\b"),
    _rx(r"\b(?:протеин|гейнер|креатин)\w*\b"),
    _rx(r"\bаминокислот\w*\b"),
)
# These phrases describe what an accessory is used with, not the sold product.
# They are quarantined before both sports-food and direct-BAD rules: boilerplate
# in such cards often contains unrelated mentions of "БАД" as well.
_BAD_SPORT_USE_ONLY = (
    _rx(r"^\s*(?:шейкер\w*|бутылк\w*|контейнер\w*|таблетниц\w*)\b"),
    _rx(
        r"^.{0,100}\b(?:для|под|отсек\w*\s+для\s+хранен\w*)\s+"
        r"(?:[^,.;!?\n<>]{0,30}\s+)?(?:протеин\w*|аминокислот\w*)\b"
    ),
    _rx(r"\bдля\b[^/!?\n<>]{0,80}\bаминокислот\w*\b"),
)
_BAD_DIRECT_MARKING = (
    _rx(r"(?<![\w])б\.?\s*а\.?\s*д\.?(?![\w])"),
    _rx(r"\bбиологически\s+активн\w*\s+добавк\w*"),
    _rx(r"\bdietary\s+supplements?\b"),
)


_RISK_ITEM = (
    r"(?:спичк\w*|зажигалк\w*|угол[ье]\w*|угл[её]м|"
    r"(?:газов\w*\s+)?баллон\w*|(?:жидкост\w*|гел\w*)\s+для\s+розжиг\w*|"
    r"топлив(?:о|ом|а|у|е))"
)

# A separate risky item must be explicitly included.  Merely mentioning an
# item as compatible ("переходник для зажигалок") cannot satisfy these forms.
_FLAM_INCLUDED_EXPLICIT = (
    _rx(
        r"\bв\s+комплект\w*\s+(?:товар\w*\s+)?"
        r"(?:вход\w*|включ\w*|име\w*|есть|поставля\w*|прилага\w*)"
        rf"[^.!?\n<>]{{0,45}}?\b{_RISK_ITEM}"
    ),
    _rx(rf"\b{_RISK_ITEM}\s+(?:вход\w*|включ\w*|прилага\w*)\s+в\s+комплект\w*"),
    _rx(rf"\bкомплектац\w*\s*[:\-]\s*(?:\d+\s*\w*\s*)?{_RISK_ITEM}"),
    _rx(rf"\bв\s+комплекте\s*[:\-]\s*(?:\d+\s*\w*\s*)?{_RISK_ITEM}"),
    _rx(rf"\bкомплектац\w*\s*[:\-][^.!?\n<>]{{0,80}}\b{_RISK_ITEM}"),
)
# Some cards serialize ``Комплектация`` as an HTML list.  Keep this separate
# from the plain-text regexes: a locally listed gas cylinder is strong kit
# evidence, while a generic cylinder mentioned elsewhere in a long HTML card
# is usually only compatibility information.
_FLAM_INCLUDED_HTML_LIST = _rx(
    r"\bкомплектац\w*\s*:\s*[^.!?\n]{0,240}"
)
_FLAM_INCLUDED_HTML_GAS_CYLINDER = _rx(
    r"<li(?:\s+[^>]{0,80})?>\s*"
    r"(?P<risk>газов\w*\s+балл?он\w*)\b"
)
_FLAM_INCLUDED_TITLE = (
    _rx(r"\bсо\s+спичк\w*"),
    _rx(r"\bс\s+зажигалк\w*"),
    _rx(r"(?:\+|,\s*и\b)\s*зажигалк\w*"),
    _rx(
        r"(?:\+|,\s*и\b)\s*(?:\d+\s*)?(?:цангов\w*\s+)?"
        r"(?:газов\w*\s+)?балл?он\w*"
    ),
    _rx(r"\bс\s+(?:газов\w*\s+)?балл?оном\b"),
    _rx(r"\b(?:мангал|гриль|барбекю)\b[^,.;!?\n]{0,45}\bс\s+угл[её]м\b"),
    _rx(
        r"\bнабор\b(?!\s+для\s+(?:зажигал|спич))"
        r"[^,.;!?\n<>]{0,75}\b(?:спичк\w*|зажигалк\w*)"
    ),
)
_FALSE_INCLUDED_BRIDGE = _rx(
    r"\b(?:без|переходник\w*|адаптер\w*|аксессуар\w*|лейк\w*|"
    r"подключ\w*|для\s+заправк\w*|кремн\w*|фитил\w*)\b"
)

_FLAM_NOT_INCLUDED = (
    _rx(
        rf"\b{_RISK_ITEM}[^.!?\n]{{0,45}}?"
        r"(?:не\s+(?:вход\w*|включ\w*)\s+в\s+комплект\w*|"
        r"в\s+комплект\w*\s+не\s+(?:вход\w*|включ\w*)|"
        r"приобрета\w*\s+отдельно)"
    ),
    _rx(
        r"\b(?:комплект\w*|набор\w*|товар\w*|поставк\w*)"
        r"[^.!?\n<>]{0,45}\bбез\s+"
        r"(?:спичек|зажигалк\w*|газов\w*\s+баллон\w*)"
    ),
    _rx(
        r"\b(?:поставля\w*|прода[её]тся)\s+без\s+"
        r"(?:спичек|зажигалк\w*|газов\w*\s+баллон\w*)"
    ),
    _rx(
        r"\(\s*без\s+"
        r"(?:спичек|зажигалк\w*|газов\w*\s+баллон\w*)\s*[!)]"
    ),
)
_FLAM_EMPTY_CUE = (
    _rx(
        r"\b(?:пуст\w*\s+(?:газов\w*\s+)?баллон\w*|"
        r"(?:газов\w*\s+)?баллон\w*\s*(?:явля\w*\s+)?пуст\w*)"
    ),
    _rx(r"\bбез\s+(?:газа|топлива|горючего\s+содержим\w*|содержимого)\b"),
    _rx(r"\b(?:не\s*заправлен\w*|прода[её]тся\s+пуст\w*)\b"),
    _rx(
        r"\b(?:поставля\w*|прода[её]тся)\s+без\s+"
        r"(?:газа|топлива|горючего|баллон\w*)"
    ),
)
_FLAM_MAIN_CONTAINER_NAME = _rx(
    r"^\s*(?:(?:пуст\w*\s+)?(?:газов\w*\s+)?баллон\w*|"
    r"(?:газов\w*\s+)?(?:емкост\w*|[её]мкост\w*|резервуар\w*))\b"
)

# Ambiguous product families are quarantined before any broad device rule.
_ORDINARY_CANDLE_NAME = _rx(r"\bсвеч(?:а|и|ей|у|ами|ах|ка|ки|ек|ной|ные|ных)\b")
_PYRO_CANDLE_NAME = _rx(r"\b(?:свеч\w*[-\s]*фонтан\w*|фонтан\w*\s+для\s+торт\w*|бенгальск\w*\s+огн\w*)")
_POPPER_NAME = _rx(r"\bхлопушк\w*")
_EXPLICIT_PYRO_NAME = _rx(r"\bпиротехнич\w*")
_DRY_FUEL_NAME = _rx(
    r"\b(?:сух\w*\s+(?:горюч\w*|топлив\w*)|"
    r"таблетк\w*\s+для\s+розжиг\w*|парафинов\w*\s+ролл\w*|"
    r"ролл\w*\s+для\s+розжиг\w*)"
)
_COMBUSTIBLE_MAIN_PRODUCT_NAME = _rx(
    r"^\s*(?:\d+\s*шт\.?\s*)?(?:брик\w*|ролл\w*|дров\w*\b|угол[ье]\b)"
)

_ACCESSORY_RISK_OBJECT = (
    r"(?:спич\w*|зажигал\w*|горелк\w*|мангал\w*|грил\w*|барбекю\w*|"
    r"балл?он\w*|газов\w*\s+плит\w*|бенгальск\w*\s+огн\w*|"
    r"свеч\w*[-\s]*фонтан\w*)"
)

_FLAM_STRONG_ACCESSORY_NAME = (
    _rx(r"\b(?:спичечниц\w*|спичечн\w*\s+короб\w*|короб\w*\s+для\s+спич\w*)\b"),
    _rx(r"\bугольн\w*\s+фильтр\w*\b"),
    _rx(r"\bфильтр\w*[^,.;!?\n]{0,50}\bактивированн\w*\s+угл\w*"),
    _rx(r"\bугол[ье]\s+для\s+рисован\w*"),
    _rx(
        r"\b(?:пакет\w*|карточк\w*|наклейк\w*|держател\w*|упаковк\w*|"
        r"чехол\w*|подставк\w*)[^,.;!?\n]{0,70}"
        r"\b(?:бенгальск\w*\s+огн\w*|свеч\w*[-\s]*фонтан\w*)"
    ),
    _rx(
        r"^\s*(?:чехол\w*|футляр\w*|кейс\w*|держател\w*|подставк\w*|"
        r"ремкомплект\w*|кремн\w*|фитил\w*|сопл\w*|редуктор\w*|"
        r"переходник\w*|адаптер\w*|запчаст\w*|комплектующ\w*)\b"
        rf"[^,.;!?\n<>]{{0,100}}\b{_ACCESSORY_RISK_OBJECT}"
    ),
    _rx(
        r"^\s*(?:сумк\w*|чехол\w*|крышк\w*|веер\w*|опахал\w*|махалк\w*|"
        r"сковород\w*|реш[её]тк\w*)\b"
        rf"[^,.;!?\n<>]{{0,100}}\b{_ACCESSORY_RISK_OBJECT}"
    ),
    _rx(r"^\s*набор\s+для\s+(?:зажигал\w*|спич\w*)\b"),
)
_FLAM_COMPONENT_NAME = (
    _rx(
        r"\b(?:чехол|футляр|кейс|держател\w*|подставк\w*|ремкомплект\w*|набор\s+для|"
        r"кремн\w*|фитил\w*|сопл\w*|редуктор\w*|переходник\w*|"
        r"адаптер\w*|насадк\w*|запчаст\w*|комплектующ\w*)"
        r"[^,.;!?\n]{0,70}\b(?:зажигалк\w*|горелк\w*|спичк\w*|"
        r"газов\w*\s+(?:баллон\w*|плит\w*))"
    ),
)

_FLAM_STANDALONE_MATCHES = _rx(r"\bспички\b")
_FLAM_STANDALONE_LIGHTER = _rx(r"\bзажигалк(?:а|и|у|ой|е|и)\b")
_NON_FLAME_LIGHTER_NAME = _rx(
    r"\b(?:электрическ\w*|электронн\w*|дугов\w*|плазменн\w*|usb|"
    r"вечн\w*|огнив\w*|кресал\w*|игрушк\w*|прикол\w*|пранк\w*|шок\w*|"
    r"газосвар\w*|искрообраз\w*|кремн\w*|фитил\w*)\b"
)
_FLAM_EXPLICIT_PYRO = (
    _PYRO_CANDLE_NAME,
    _rx(r"\bпиротехнич\w*\s+(?:издел\w*|хлопушк\w*|заряд\w*)"),
    _rx(r"\b(?:петард\w*|бенгальск\w*\s+огн\w*)"),
)

_FLAM_MAIN_CONTENT = (
    _rx(r"\b(?:газ|бутан|пропан)\s+для\s+(?:зажигал\w*|горел\w*|плит\w*|заправк\w*)"),
    _rx(r"\b(?:жидкост\w*|гел\w*)\s+для\s+розжиг\w*"),
    _rx(r"\bтоплив\w*\s+для\s+зажигал\w*"),
    _rx(r"\bбензин\w*[^,.;!?\n<>]{0,35}\bдля\s+(?:заправк\w*\s+)?зажигал\w*"),
    _rx(r"\bтоплив\w*\s*\d+(?:[.,]\d+)?\s*(?:мл|л)\b"),
    _rx(
        r"^\s*(?:газов\w*\s+баллон\w*|баллон\w*\s+(?:с\s+)?газ\w*)"
        r"[^,.;!?\n]{0,45}\b(?:бутан|пропан|\d+\s*(?:г|гр|мл)\b)"
    ),
)
_FLAM_EXPLICIT_CONTENT = (
    _rx(
        r"(?<!не\s)\b(?:содержит\w*|содержится|наполнен\w*|заправлен\w*)"
        r"[^,.;!?\n]{0,65}\b(?:бутан|пропан|горюч\w*\s+газ\w*|"
        r"легковоспламеня\w*|бензин\w*|керосин\w*|бутан\w*|пропан\w*)\b"
    ),
)

_FLAM_DEVICE_NAME = (
    _rx(r"\b(?:мангал\w*|жаровн\w*|биокамин\w*)\b"),
    _rx(r"\b(?:газов\w*\s+)?(?:гриль\w*|барбекю)\b"),
    _rx(r"\bгазов\w*\s+(?:плит\w*|печ\w*|горелк\w*)\b"),
    _rx(r"\b(?:плит\w*|печ\w*|горелк\w*)\s+газов\w*\b"),
    _rx(
        r"^\s*плит\w*[^,.;!?\n<>]{0,100}\b"
        r"(?:газов\w*|конфор\w*|баллон\w*|пьезо\w*|туристическ\w*|походн\w*)"
    ),
    _rx(r"\b(?:туристическ\w*|кухонн\w*|походн\w*)\s+горелк\w*\b"),
    _rx(
        r"^\s*(?:туристическ\w*\s+)?насадк\w*\s*[-/ ]\s*"
        r"(?:газов\w*\s+)?горелк\w*\b"
    ),
    _rx(r"^\s*(?:газов\w*\s+)?горелк\w*\s*[-/ ]\s*насадк\w*\b"),
)
_FLAM_INTEGRATED_IGNITION = (
    _rx(r"\b(?:пьезо|электро)поджиг\w*\b"),
    _rx(r"\bпьезо[-\s]?(?:электрическ\w*\s+)?розжиг\w*\b"),
    _rx(r"\bвстроенн\w*\s+(?:поджиг\w*|воспламенител\w*)"),
    _rx(r"\bоснащен\w*\s+(?:пьезо|электро)поджиг\w*"),
)
_INTEGRATED_IGNITION_TERM = (
    r"(?:пьезо(?:электрическ\w*)?[- ]?(?:поджиг\w*|розжиг\w*)|"
    r"электроподжиг\w*)"
)
_FLAM_INTEGRATED_IGNITION_ABSENT = (
    _rx(
        rf"\b(?:без|нет|отсутств\w*) +(?:встроенн\w* +|систем\w* +)?"
        rf"{_INTEGRATED_IGNITION_TERM}\b"
    ),
    _rx(
        rf"\b{_INTEGRATED_IGNITION_TERM}"
        r" *(?:(?:[:=]|[-–—]) *)?нет\b"
    ),
    _rx(
        rf"\bне +(?:име\w*|оснащ\w*|предусмотр\w*)"
        rf"[^,.;!?\n<>]{{0,30}}\b{_INTEGRATED_IGNITION_TERM}\b"
    ),
)
_UNCERTAIN_SELF_CONTAINED_DEVICE = _rx(
    r"\b(?:перезаправ\w*|заправляем\w*|резервуар\w*|встроенн\w*\s+бак\w*)"
)


_RULE_TEXT = {
    "BAD-NEG-EXPLICIT-EXCLUSION": (
        "Прямое указание {evidence} исключает товар из БАД.",
        "Без этого исключения нужна была бы прямая маркировка БАД.",
    ),
    "BAD-NEG-SPORTS-NUTRITION": (
        "Признак {evidence} относит товар к спортивному питанию.",
        "Прямо маркированная БАД без признака спортивного питания дала бы иной вывод.",
    ),
    "BAD-POS-DIRECT-MARKING": (
        "В карточке есть прямая маркировка {evidence}.",
        "Без маркировки БАД или dietary supplement оснований для категории нет.",
    ),
    "FLAM-POS-SEPARATE-IN-KIT": (
        "Фраза {evidence} подтверждает отдельный рискованный предмет в комплекте.",
        "Без этого предмета одно лишь устройство не относилось бы к категории.",
    ),
    "FLAM-NEG-NOT-INCLUDED": (
        "Фраза {evidence} прямо исключает рискованный предмет из комплекта.",
        "Если бы он входил в комплект отдельно, товар относился бы к категории.",
    ),
    "FLAM-NEG-EMPTY-CONTENTS": (
        "Фраза {evidence} подтверждает отсутствие горючего содержимого.",
        "С горючим содержимым внутри тот же товар относился бы к категории.",
    ),
    "FLAM-NEG-COMPONENT": (
        "Фраза {evidence} описывает аксессуар или компонент, а не рискованный товар.",
        "Отдельно продаваемый источник огня или горючее дали бы иной вывод.",
    ),
    "FLAM-POS-STANDALONE-IGNITION": (
        "Фраза {evidence} называет самостоятельный источник огня.",
        "Аксессуар без источника огня или встроенный поджиг не относился бы к категории.",
    ),
    "FLAM-POS-FLAMMABLE-CONTENTS": (
        "Фраза {evidence} прямо подтверждает продаваемое горючее содержимое.",
        "Пустая конструкция без этого содержимого не относилась бы к категории.",
    ),
    "FLAM-NEG-INTEGRATED-IGNITION": (
        "Фраза {evidence} описывает поджиг, встроенный в устройство.",
        "Отдельная зажигалка или спички в комплекте дали бы положительный вывод.",
    ),
    "FLAM-NEG-DEVICE-WITHOUT-CONTENT": (
        "Фраза {evidence} называет устройство для работы с огнём, не само горючее.",
        "Явно включённое горючее или отдельный источник огня изменили бы вывод.",
    ),
}


@dataclass(frozen=True)
class _Decision:
    rule_id: str
    prediction: int
    source: str
    exact_span: str


def _as_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"source.{field} must be a string or null")
    return value


def _source_fields(source: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping")
    category = source.get("category")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"source.category must be one of {sorted(VALID_CATEGORIES)!r}")
    return {
        "category": str(category),
        "name": _as_text(source.get("name"), "name"),
        "description": _as_text(source.get("description"), "description"),
    }


def _search_view(text: str) -> str:
    # All substitutions preserve string length, so a regex span can be sliced
    # verbatim from the source card afterwards.
    return (
        text.replace("ё", "е")
        .replace("Ё", "Е")
        .replace("\u00a0", " ")
        .replace("\t", " ")
        .replace("\r", " ")
    )


def _find(
    fields: Mapping[str, str],
    patterns: Sequence[Pattern[str]],
    sources: Sequence[str] = ("name", "description"),
) -> tuple[str, str] | None:
    hits: list[tuple[int, int, int, str, re.Match[str]]] = []
    for source_rank, source_name in enumerate(sources):
        text = fields[source_name]
        view = _search_view(text)
        for pattern_rank, pattern in enumerate(patterns):
            match = pattern.search(view)
            if match:
                hits.append((source_rank, match.start(), pattern_rank, source_name, match))
    if not hits:
        return None
    _, _, _, source_name, match = min(hits, key=lambda item: item[:3])
    return source_name, fields[source_name][match.start() : match.end()]


def _find_html_listed_gas_cylinder(
    fields: Mapping[str, str],
) -> tuple[str, str] | None:
    """Return only the grounded cylinder item from a ``Комплектация`` list."""

    hits: list[tuple[int, int, str, int, int]] = []
    for source_rank, source_name in enumerate(("name", "description")):
        view = _search_view(fields[source_name])
        for section in _FLAM_INCLUDED_HTML_LIST.finditer(view):
            item = _FLAM_INCLUDED_HTML_GAS_CYLINDER.search(section.group(0))
            if not item:
                continue
            # Do not turn ``<li>Газовый баллон в комплект не входит`` into
            # contradictory positive evidence.  The ordinary NOT_INCLUDED
            # branch below should own that card.
            local_item = section.group(0)[item.start() :]
            closing_tag = local_item.find("</li>")
            if closing_tag >= 0:
                local_item = local_item[:closing_tag]
            if any(pattern.search(local_item) for pattern in _FLAM_NOT_INCLUDED):
                continue
            start = section.start() + item.start("risk")
            end = section.start() + item.end("risk")
            hits.append((source_rank, start, source_name, start, end))
    if not hits:
        return None
    _, _, source_name, start, end = min(hits, key=lambda item: item[:2])
    return source_name, fields[source_name][start:end]


def _one(pattern: Pattern[str]) -> tuple[Pattern[str], ...]:
    return (pattern,)


def _decision(rule_id: str, prediction: int, hit: tuple[str, str]) -> _Decision:
    return _Decision(rule_id, prediction, hit[0], hit[1])


def _extract_bad(fields: Mapping[str, str]) -> _Decision | None:
    hit = _find(fields, _BAD_EXPLICIT_NEGATIVE)
    if hit:
        return _decision("BAD-NEG-EXPLICIT-EXCLUSION", 0, hit)

    # A shaker "for protein" or a supplement "for amino acids" does not make
    # the sold product sports nutrition.  Fail closed instead of allowing
    # boilerplate elsewhere in the description to decide the category.
    if _find(fields, _BAD_SPORT_USE_ONLY, sources=("name",)):
        return None

    hit = _find(fields, _BAD_SPORT_EXPLICIT)
    if not hit:
        hit = _find(fields, _one(_BAD_SPORT_PHRASE), sources=("name",))
    if not hit:
        hit = _find(fields, _BAD_SPORT_MAIN_PRODUCT, sources=("name",))
    if hit:
        return _decision("BAD-NEG-SPORTS-NUTRITION", 0, hit)

    hit = _find(fields, _BAD_DIRECT_MARKING)
    if hit:
        return _decision("BAD-POS-DIRECT-MARKING", 1, hit)

    # The absence of a direct marker is an official negative rule, but it is
    # not observable evidence suitable for a short grounded rationale.
    return None


def _extract_flammable(fields: Mapping[str, str]) -> _Decision | None:
    included = _find(fields, _FLAM_INCLUDED_EXPLICIT)
    if not included:
        included = _find(fields, _FLAM_INCLUDED_TITLE, sources=("name",))
    if not included:
        included = _find_html_listed_gas_cylinder(fields)
    if included and _FALSE_INCLUDED_BRIDGE.search(_search_view(included[1])):
        included = None
    # An explicitly empty cylinder included with a device is construction, not
    # flammable contents.  This check is intentionally limited to cylinder-kit
    # evidence so an unrelated empty container cannot cancel included matches.
    if (
        included
        and re.search(r"\bбалл?он\w*", _search_view(included[1]), re.IGNORECASE)
        and _find(fields, _FLAM_EMPTY_CUE)
    ):
        included = None
    not_included = _find(fields, _FLAM_NOT_INCLUDED)
    name_view = _search_view(fields["name"])

    # An empty cylinder mentioned in refill instructions is not necessarily the
    # product being sold.  Empty-state evidence is admitted only when the title
    # identifies the sold product itself as the container.
    empty = None
    if _FLAM_MAIN_CONTAINER_NAME.search(name_view):
        empty = _find(fields, _FLAM_EMPTY_CUE)

    # Contradictory text is left for a model/image audit instead of resolving it
    # with arbitrary regex precedence.
    if included and (not_included or empty):
        return None
    if included:
        return _decision("FLAM-POS-SEPARATE-IN-KIT", 1, included)
    if empty:
        return _decision("FLAM-NEG-EMPTY-CONTENTS", 0, empty)

    # Strong accessory heads describe the sold object rather than a compatible
    # burner, grill, cylinder, or lighter mentioned later in its title.
    accessory = _find(fields, _FLAM_STRONG_ACCESSORY_NAME, sources=("name",))
    if accessory:
        return _decision("FLAM-NEG-COMPONENT", 0, accessory)

    # A present fuel remains sufficient even when another risky item (for
    # example the lighter) is explicitly absent from the same kit.
    content = _find(fields, _FLAM_MAIN_CONTENT, sources=("name",))
    if not content:
        content = _find(fields, _FLAM_EXPLICIT_CONTENT)
    if content:
        return _decision("FLAM-POS-FLAMMABLE-CONTENTS", 1, content)

    if not_included:
        return _decision("FLAM-NEG-NOT-INCLUDED", 0, not_included)

    if _ORDINARY_CANDLE_NAME.search(name_view) and not _PYRO_CANDLE_NAME.search(name_view):
        return None
    if _POPPER_NAME.search(name_view) and not _EXPLICIT_PYRO_NAME.search(name_view):
        return None
    if _DRY_FUEL_NAME.search(name_view):
        return None
    if _COMBUSTIBLE_MAIN_PRODUCT_NAME.search(name_view):
        return None

    matches = _find(fields, _one(_FLAM_STANDALONE_MATCHES), sources=("name",))
    if matches:
        return _decision("FLAM-POS-STANDALONE-IGNITION", 1, matches)

    lighter = _find(fields, _one(_FLAM_STANDALONE_LIGHTER), sources=("name",))
    if lighter and not _NON_FLAME_LIGHTER_NAME.search(name_view):
        return _decision("FLAM-POS-STANDALONE-IGNITION", 1, lighter)

    pyro = _find(fields, _FLAM_EXPLICIT_PYRO, sources=("name",))
    if pyro:
        return _decision("FLAM-POS-STANDALONE-IGNITION", 1, pyro)

    device = _find(fields, _FLAM_DEVICE_NAME, sources=("name",))
    if device:
        integrated_absent = _find(fields, _FLAM_INTEGRATED_IGNITION_ABSENT)
        integrated = (
            None
            if integrated_absent
            else _find(fields, _FLAM_INTEGRATED_IGNITION)
        )
        if integrated:
            return _decision("FLAM-NEG-INTEGRATED-IGNITION", 0, integrated)
        # Refillable/self-contained burners are a known boundary case.  Without
        # an explicit statement that fuel is present they are quarantined.
        if _UNCERTAIN_SELF_CONTAINED_DEVICE.search(name_view):
            return None
        return _decision("FLAM-NEG-DEVICE-WITHOUT-CONTENT", 0, device)

    # Residual component wording is evaluated only after a main-product device
    # has had a chance to claim the title.  This prevents "газовая плита с
    # переходником" and "насадка-горелка" from becoming mere components.
    component = _find(fields, _FLAM_COMPONENT_NAME, sources=("name",))
    if component:
        return _decision("FLAM-NEG-COMPONENT", 0, component)

    return None


def extract_rule(source: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract a label-blind policy decision from one source card.

    ``source`` may contain ``label`` and other training metadata, but this
    function reads only category/name/description.  The returned mapping is
    useful for tests and audits; callers should normally persist ``build_record``.
    """

    fields = _source_fields(source)
    if fields["category"] == BAD_CATEGORY:
        decision = _extract_bad(fields)
    else:
        decision = _extract_flammable(fields)
    if decision is None:
        return None
    return {
        "category": fields["category"],
        "rule_prediction": decision.prediction,
        "rule_id": decision.rule_id,
        "evidence": {"source": decision.source, "exact_span": decision.exact_span},
    }


def source_text_sha256(source: Mapping[str, Any]) -> str:
    """Hash the exact category/name/description payload used by extraction."""

    fields = _source_fields(source)
    payload = json.dumps(
        fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_span(span: str, max_chars: int = 88) -> str:
    clean = re.sub(r"\s+", " ", span).strip().replace("«", "").replace("»", "")
    if len(clean) > max_chars:
        clean = clean[: max_chars - 1].rstrip() + "…"
    return f"«{clean}»"


def _render_target(extracted: Mapping[str, Any]) -> str:
    rule_id = str(extracted["rule_id"])
    reason_template, contrast = _RULE_TEXT[rule_id]
    evidence = _display_span(str(extracted["evidence"]["exact_span"]))
    target = "\n".join(
        (
            "ПРИЧИНА: " + reason_template.format(evidence=evidence),
            "КОНТРАСТ: " + contrast,
            "ПРАВИЛО: " + rule_id,
        )
    )
    _validate_target(target, rule_id)
    return target


def _validate_target(target: str, rule_id: str) -> None:
    lines = target.splitlines()
    expected_prefixes = ("ПРИЧИНА: ", "КОНТРАСТ: ", "ПРАВИЛО: ")
    if len(lines) != 3 or any(
        not line.startswith(prefix) for line, prefix in zip(lines, expected_prefixes)
    ):
        raise ValueError("target must contain exactly three canonical lines")
    if lines[-1] != f"ПРАВИЛО: {rule_id}":
        raise ValueError("target must end with its exact rule_id")
    if any(len(line) > 220 for line in lines):
        raise ValueError("target lines must remain short")


def _require_int(source: Mapping[str, Any], field: str) -> int:
    value = source.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"source.{field} must be an integer")
    return value


def build_record(source: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build one canonical rationale record, or ``None`` when quarantined.

    Extraction happens before ``label`` is read.  Consequently the observed
    label can affect only ``observed_label`` and ``label_support``.
    """

    extracted = extract_rule(source)
    if extracted is None:
        return None

    product_id = _require_int(source, "id")
    fold = _require_int(source, "fold")
    observed_label = _require_int(source, "label")
    if observed_label not in (0, 1):
        raise ValueError("source.label must be 0 or 1")
    if fold < 0:
        raise ValueError("source.fold must be non-negative")

    target = _render_target(extracted)
    prediction = int(extracted["rule_prediction"])
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "id": product_id,
        "fold": fold,
        "category": extracted["category"],
        "student_scope": "text",
        "observed_label": observed_label,
        "rule_prediction": prediction,
        "label_support": "supported" if prediction == observed_label else "conflict",
        "rule_id": extracted["rule_id"],
        "evidence": dict(extracted["evidence"]),
        "source_text_sha256": source_text_sha256(source),
        "target": target,
        "target_sha256": _sha256_text(target),
    }


def canonical_record_json(record: Mapping[str, Any]) -> str:
    """Serialize a canonical record deterministically for JSONL storage."""

    return json.dumps(
        dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def validate_record(
    record: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild and strictly compare a serialized record against its source.

    Returning the rebuilt record (rather than the untrusted payload) gives a
    training loader a safe canonical value.  Any source change, hand-edited
    target, stale generator version, or extra/missing field fails closed.
    """

    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")
    canonical = build_record(source)
    if canonical is None:
        raise ValueError("source no longer yields a deterministic rationale")
    if canonical_record_json(record) != canonical_record_json(canonical):
        differing = sorted(
            key
            for key in set(record) | set(canonical)
            if record.get(key) != canonical.get(key)
        )
        raise ValueError("record does not match deterministic rebuild: " + ", ".join(differing))
    return canonical


__all__ = [
    "SCHEMA_VERSION",
    "GENERATOR_VERSION",
    "source_text_sha256",
    "extract_rule",
    "build_record",
    "validate_record",
    "canonical_record_json",
]
