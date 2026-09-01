"""Grounded, label-blind rationale contract for product classification.

The VLM first describes observable evidence without seeing the dataset label.
Only :func:`attach_observed_label` may add the label and ``label_support``.
This separation makes post-hoc rationalization visible in the stored JSONL.

The schema intentionally treats ``not observed`` as ``unknown``.  A model may
use ``explicitly_absent`` or ``empty`` only when an explicit, grounded cue says
so; an unreadable or missing view never proves absence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


# ``SCHEMA_VERSION`` is the persisted rationale format consumed by the current
# training/audit loader.  V9 deliberately keeps that envelope readable so an
# already audited V1 bank does not become unusable merely because extraction
# improved.  The teacher-facing schema is separate: it contains observations
# and concepts only; every decision field is produced below by deterministic
# code.
SCHEMA_VERSION = "grounded-rationale-v1"
EXTRACTION_SCHEMA_VERSION = "grounded-evidence-v2"
COUNTERFACTUAL_SCHEMA_VERSION = "deterministic-counterfactual-v1"

CATEGORIES = frozenset({"БАД", "Легковоспламеняющиеся"})
POLICY_VERDICTS = frozenset({"yes", "no", "review"})
LABEL_SUPPORT_VALUES = frozenset({"supported", "conflict", "insufficient"})

PRODUCT_FORMS = {
    "БАД": frozenset(
        {
            "supplement",
            "sports_nutrition",
            "food",
            "medicine",
            "accessory",
            "kit",
            "unknown",
        }
    ),
    "Легковоспламеняющиеся": frozenset(
        {
            "standalone_product",
            "kit",
            "appliance",
            "container",
            "fuel",
            "pyrotechnic",
            "component",
            "unknown",
        }
    ),
}

ELEMENT_KINDS = {
    "БАД": frozenset(
        {
            "bad_marking",
            "dietary_supplement_marking",
            "sports_nutrition_marking",
            "explicit_not_bad",
            "unknown",
        }
    ),
    "Легковоспламеняющиеся": frozenset(
        {
            "ignition_source",
            "flammable_gas",
            "flammable_liquid",
            "solid_fuel",
            "pyrotechnic_charge",
            "spark_source",
            "non_ignition_device",
            "combustible_component",
            "unknown",
        }
    ),
}

ROLES = frozenset(
    {
        "main_product",
        "separate_in_kit",
        "integrated",
        "content_of",
        "component_of",
        "compatible_only",
        "mentioned_only",
        "pictured_use_only",
        "unknown",
    }
)

STATES = frozenset({"present", "explicitly_absent", "empty", "unknown"})

MECHANISMS = frozenset(
    {
        "open_flame",
        "sustains_burning",
        "flammable_contents",
        "pyrotechnic",
        "spark_only",
        "pneumatic",
        "heat_only",
        "direct_bad_marking",
        "dietary_supplement_marking",
        "sports_nutrition_marking",
        "explicit_exclusion",
        "none",
        "unknown",
    }
)

EVIDENCE_SOURCES = frozenset({"name", "description", "ocr", "image"})
EVIDENCE_POLARITIES = frozenset({"support", "refute", "context"})
MODALITY_RELATIONS = frozenset(
    {"agree", "text_only", "image_only", "conflict", "insufficient"}
)

_POST_LABEL_KEYS = frozenset(
    {
        "label",
        "gold_label",
        "target",
        "observed_label",
        "label_support",
        "blind_record_sha256",
    }
)

_ROOT_KEYS = frozenset(
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

_ELEMENT_KEYS = frozenset(
    {"name", "kind", "role", "state", "mechanism", "evidence"}
)

_EVIDENCE_KEYS = frozenset(
    {"source", "polarity", "exact_span", "image_id", "bbox", "observation"}
)

_COUNTERFACTUAL_KEYS = frozenset(
    {
        "schema_version",
        "single_change",
        "interventions",
        "expected_verdict",
        "resulting_policy_rule_ids",
    }
)

_INTERVENTION_KEYS = frozenset({"op", "element_index", "set"})
_INTERVENTION_FIELDS = frozenset({"kind", "role", "state", "mechanism"})

_EXTRACTION_ROOT_KEYS = frozenset(
    {"schema_version", "product_id", "category", "product_form", "elements", "conflicts"}
)

_ABSENCE_CUES = (
    "без ",
    "без\u00a0",
    "не входит",
    "не входят",
    "не содержит",
    "не является",
    "не заправлен",
    "не заполнен",
    "отсутств",
    "пуст",
    "нет в комплект",
    "нет маркиров",
    "нет указания",
    "нет топлива",
    "нет газа",
    "empty",
    "without ",
    "not included",
    "does not contain",
)

_NON_EVIDENCE_CUES = (
    "не видно",
    "не виден",
    "не видна",
    "не видны",
    "не наблюда",
    "не обнаруж",
    "не удалось прочитать",
    "не читается",
    "отсутствует на фото",
    "отсутствует на фотографии",
    "отсутствует на изображении",
    "на фото нет",
    "на изображении нет",
    "изображение не содержит",
    "not visible",
    "not observed",
    "cannot see",
    "unreadable",
    "absent from the image",
    "not present in the image",
)

RISK_SUBSTANCE_KINDS = frozenset({
    "flammable_gas", "flammable_liquid", "solid_fuel", "pyrotechnic_charge",
})

_RISK_ABSENCE_TARGETS = {
    "flammable_gas": (
        r"(?:горюч\w*|топлив\w*|газ(?:а|у|ом|е|ы|ов|ами|ах)?\b|"
        r"(?:газов\w*\s+)?баллон\w*|бутан\w*|пропан\w*)"
    ),
    "flammable_liquid": (
        r"(?:горюч\w*|топлив\w*|жидк\w*|бензин\w*|керосин\w*|спирт\w*)"
    ),
    "solid_fuel": (
        r"(?:горюч\w*|топлив\w*|угол\w*|угл[её]м|брик[её]т\w*|дров\w*)"
    ),
    "pyrotechnic_charge": (
        r"(?:пиротехн\w*|пиросостав\w*|порох\w*|заряд\w*)"
    ),
}

_RISK_ABSENCE_TARGETS_EN = {
    "flammable_gas": (
        r"(?:gas(?:\s+(?:cylinder|canister))?|(?:cylinder|canister)\s+of\s+gas|"
        r"butane|propane|fuel)\b"
    ),
    "flammable_liquid": (
        r"(?:flammable\s+liquid|gasoline|petrol|kerosene|alcohol|fuel)\b"
    ),
    "solid_fuel": r"(?:solid\s+fuel|charcoal|coal|briquettes?|firewood|fuel)\b",
    "pyrotechnic_charge": (
        r"(?:pyrotechnic(?:\s+charge)?|gunpowder|pyro\s+charge)\b"
    ),
}

# A visual observation carrying ``polarity=support`` must not literally say
# that the very risk concept it is attached to is absent. Keep these targets
# deliberately narrower than the general ontology: this is a contradiction
# guard, not a visual absence detector. In particular, generic properties
# such as "no visible damage" or "without cover" must not match.
_IMAGE_RISK_DENIAL_TARGETS_RU = {
    "ignition_source": (
        r"(?:зажигалк\w*|спич(?:к\w*|ек|ками|ках)|огнив\w*|"
        r"источник\w*\s+(?:зажигания|воспламенения|огня)|"
        r"устройств\w*\s+зажигани\w*)"
    ),
    "spark_source": (
        r"(?:пьезо(?:поджиг|зажигани)\w*|искров\w*\s+источник\w*|"
        r"источник\w*\s+искр\w*|устройств\w*\s+зажигани\w*)"
    ),
    "flammable_gas": (
        r"(?:горюч\w*\s+газ\w*|газов\w*\s+(?:баллон|картридж)\w*|"
        r"баллон\w*\s+с\s+газом|бутан\w*|пропан\w*)"
    ),
    "flammable_liquid": (
        r"(?:горюч\w*\s+жидк\w*|жидк\w*\s+топлив\w*|бензин\w*|"
        r"керосин\w*|спирт\w*)"
    ),
    "solid_fuel": (
        r"(?:тв[её]рд\w*\s+топлив\w*|угол\w*|угл[её]м|брик[её]т\w*|"
        r"дров\w*|свеч\w*)"
    ),
    "pyrotechnic_charge": (
        r"(?:пиротехн\w*(?:\s+(?:заряд|издел)\w*)?|пиросостав\w*|"
        r"порох\w*|фейерверк\w*|свеч\w*[-\s]?фонтан\w*)"
    ),
    "combustible_component": (
        r"(?:горюч\w*|воспламеня\w*)\s+(?:компонент|материал|част)\w*"
    ),
}

_IMAGE_RISK_DENIAL_TARGETS_EN = {
    "ignition_source": (
        r"(?:ignition\s+(?:device|source|mechanism)|source\s+of\s+ignition|"
        r"lighter|matches?|fire[-\s]?starter)"
    ),
    "spark_source": (
        r"(?:ignition\s+(?:device|source|mechanism)|spark\s+(?:source|device|"
        r"igniter)|piezo(?:electric)?\s+(?:ignition|igniter)|igniter)"
    ),
    "flammable_gas": (
        r"(?:flammable\s+gas|gas\s+(?:cylinder|canister|cartridge|contents?)|"
        r"butane|propane)"
    ),
    "flammable_liquid": (
        r"(?:flammable\s+liquid|liquid\s+fuel|gasoline|petrol|kerosene|"
        r"alcohol)"
    ),
    "solid_fuel": (
        r"(?:solid\s+fuel|charcoal|coal|briquettes?|firewood|candles?)"
    ),
    "pyrotechnic_charge": (
        r"(?:pyrotechnic(?:\s+(?:charge|device))?|pyro\s+charge|gunpowder|"
        r"fireworks?|fountain\s+candle)"
    ),
    "combustible_component": (
        r"(?:flammable|combustible)\s+(?:component|material|part)"
    ),
}


def image_support_denies_risk_target(observation: Any, kind: str) -> bool:
    """Detect a target-scoped contradiction inside image support evidence.

    This does *not* turn visual non-observation into refutation. It only
    rejects a self-contradictory extraction where the teacher labelled an
    observation as support while its own words explicitly deny seeing or
    finding that same risk kind.
    """

    target_ru = _IMAGE_RISK_DENIAL_TARGETS_RU.get(kind)
    target_en = _IMAGE_RISK_DENIAL_TARGETS_EN.get(kind)
    if not isinstance(observation, str) or not target_ru or not target_en:
        return False
    text = " ".join(observation.casefold().replace("\u00a0", " ").split())
    nuisance_ru = r"(?:поврежден|дефект|износ|царапин|чехл|крышк|корпус)\w*"
    nuisance_en = r"(?:damage|defects?|wear|scratches?|cover|case|cap|housing)\b"
    target_ru_scoped = rf"{target_ru}(?!\s+{nuisance_ru})"
    target_en_scoped = rf"{target_en}\b(?!\s+{nuisance_en})"
    ru_patterns = (
        rf"(?:не\s+(?:видно|виден|видна|видны|наблюда\w*|обнаруж\w*))\s+"
        rf"(?:никак\w*\s+|видим\w*\s+)?{target_ru_scoped}",
        rf"(?:нет|отсутству\w*)\s+(?:никак\w*\s+|видим\w*\s+)?"
        rf"{target_ru_scoped}",
        rf"\bбез\s+(?:видим\w*\s+)?{target_ru_scoped}",
        rf"{target_ru_scoped}\s+(?:не\s+(?:(?:ясно|ч[её]тко|хорошо)\s+)?"
        rf"(?:виден|видна|видно|видны|наблюда\w*|обнаруж\w*|показан\w*)|"
        rf"отсутству\w*)",
        rf"(?:изображени\w*|фотографи\w*|фото)\s+не\s+"
        rf"(?:показыва\w*|содерж\w*)\s+{target_ru_scoped}",
    )
    en_patterns = (
        rf"\bno\s+(?:(?:(?:clearly\s+)?visible|apparent)\s+)?"
        rf"(?:signs?\s+of\s+)?"
        rf"(?:(?:an?|any)\s+)?{target_en_scoped}",
        rf"\bnot\s+(?:clearly\s+)?visible\s+{target_en_scoped}",
        rf"\bwithout\s+(?:(?:an?|any)\s+)?{target_en_scoped}",
        rf"\b(?:does|do)\s+not\s+(?:show|contain|include|have)\s+"
        rf"(?:(?:an?|any)\s+)?{target_en_scoped}",
        rf"\b{target_en_scoped}\s+(?:(?:is|are)\s+)?"
        rf"(?:not\s+(?:(?:clearly|fully)\s+)?"
        rf"(?:visible|present|shown|seen)|hard\s+to\s+see|"
        rf"cannot\s+be\s+seen(?:\s+clearly)?|"
        rf"can't\s+be\s+seen(?:\s+clearly)?|absent)\b",
    )
    return any(
        re.search(pattern, text, re.I)
        for pattern in (*ru_patterns, *en_patterns)
    )


def image_refute_affirms_risk_target(observation: Any, kind: str) -> bool:
    """Detect positive target presence mislabeled as image refutation."""

    target_ru = _IMAGE_RISK_DENIAL_TARGETS_RU.get(kind)
    target_en = _IMAGE_RISK_DENIAL_TARGETS_EN.get(kind)
    if not isinstance(observation, str) or not target_ru or not target_en:
        return False
    if image_support_denies_risk_target(observation, kind):
        return False
    text = " ".join(observation.casefold().replace("\u00a0", " ").split())
    nuisance_ru = r"(?:поврежден|дефект|износ|царапин|чехл|крышк|корпус)\w*"
    nuisance_en = r"(?:damage|defects?|wear|scratches?|cover|case|cap|housing)\b"
    target_ru_scoped = rf"{target_ru}(?!\s+{nuisance_ru})"
    target_en_scoped = rf"{target_en}\b(?!\s+{nuisance_en})"
    ru_patterns = (
        rf"(?:виден|видна|видно|видны|показан\w*|наблюда\w*|присутству\w*)"
        rf"\s+(?:ч[её]тко\s+|явно\s+)?{target_ru_scoped}",
        rf"{target_ru_scoped}\s+(?:(?:ч[её]тко|явно)\s+)?"
        rf"(?:виден|видна|видно|видны|показан\w*|наблюда\w*|присутству\w*)",
        rf"(?:изображени\w*|фотографи\w*|фото)\s+"
        rf"(?:ч[её]тко\s+|явно\s+)?(?:показыва\w*|содерж\w*)\s+"
        rf"{target_ru_scoped}",
    )
    en_patterns = (
        rf"\b(?:visible|shown|pictured|present)\s+"
        rf"(?:(?:an?|the)\s+)?{target_en_scoped}",
        rf"\b{target_en_scoped}\s+(?:(?:is|are)\s+)?"
        rf"(?:(?:clearly|plainly)\s+)?(?:visible|shown|pictured|present)\b",
        rf"\b(?:image|photo|picture)\s+(?:clearly\s+|plainly\s+)?"
        rf"(?:shows|contains|includes|depicts)\s+(?:(?:an?|the)\s+)?"
        rf"{target_en_scoped}",
    )
    return any(
        re.search(pattern, text, re.I)
        for pattern in (*ru_patterns, *en_patterns)
    )


def is_strong_visual_identity_denial(observation: Any, kind: str) -> bool:
    """Recognise a narrow identity contradiction, not mere non-visibility.

    The live failure was unusually specific: the image observation positively
    identifies an ordinary paracord bracelet and its plastic buckle while also
    saying that no ignition device is present. That supports audit/review when
    text claims a lighter. A cropped, unclear or single-angle non-observation
    remains weak and must not become refutation.
    """

    if not image_support_denies_risk_target(observation, kind):
        return False
    text = " ".join(str(observation).casefold().replace("\u00a0", " ").split())
    uncertainty = re.search(
        r"\b(?:may|might|possibly|perhaps|appears?|seems?|unclear|uncertain|"
        r"cannot\s+determine|can't\s+determine|from\s+this\s+angle|"
        r"partially\s+visible|not\s+(?:clearly|fully)\s+visible|"
        r"hard\s+to\s+see|cannot\s+be\s+seen(?:\s+clearly)?|"
        r"can't\s+be\s+seen(?:\s+clearly)?|cropped|low[-\s]?resolution|"
        r"no\s+apparent|obscured)\b|"
        r"\b(?:возможно|вероятно|предположительно|похоже|неясно|неуверенно|"
        r"с\s+этого\s+ракурса|частично\s+видим|плохо\s+видно|"
        r"трудно\s+разглядеть|не\s+полностью\s+видим|обрезан|"
        r"низк\w*\s+разрешени\w*|перекрыт)\w*",
        text,
        re.I,
    )
    if uncertainty:
        return False
    bracelet = re.search(
        r"\b(?:(?:plain|ordinary|simple|black|paracord|survival|tactical)\s+)"
        r"{1,4}bracelet\b|"
        r"\b(?:(?:обычн|прост|черн|ч[её]рн|паракорд|тактическ)\w*\s+)"
        r"{1,4}браслет\w*",
        text,
        re.I,
    )
    buckle = re.search(
        r"\b(?:plastic|polymer)\s+(?:buckle|clasp)\b|"
        r"\bпластик\w*\s+(?:пряжк|заст[её]жк)\w*",
        text,
        re.I,
    )
    if not bracelet or not buckle:
        return False
    visible_hardware = re.search(
        r"(?<!no\s)(?<!not\s)\bvisible\s+(?:(?:metal|integrated)\s+)?"
        r"(?:lighter|ignition\s+device|striker|flint|nozzle|fuel\s+tank)\b|"
        r"(?<!не\s)\b(?:виден|видна|видно|видны)\s+"
        r"(?:(?:металлическ|встроенн)\w*\s+)?"
        r"(?:зажигалк|поджиг|бо[её]к|кремень|сопл|резервуар)\w*",
        text,
        re.I,
    )
    return visible_hardware is None


def has_targeted_risk_absence(span: Any, kind: str) -> bool:
    """Recognise explicit absence of the risk referent, not generic ``без``.

    Both a delivery/kit/empty cue and a kind-compatible noun are mandatory.
    A nearby property referent rejects the match, preventing phrases such as
    ``запах газа отсутствует`` from becoming absence of the gas itself.
    """

    target = _RISK_ABSENCE_TARGETS.get(kind)
    target_en = _RISK_ABSENCE_TARGETS_EN.get(kind)
    if not isinstance(span, str) or not target or not target_en:
        return False
    text = " ".join(span.casefold().replace("\u00a0", " ").split())
    container = r"(?:баллон\w*|резервуар\w*|[её]мкост\w*|канистр\w*)"
    patterns = [
        rf"(?:поставля\w*|прода\w*|комплекту\w*|ид[её]т)"
        rf"[^.;\n]{{0,24}}\bбез\s+{target}",
        rf"(?:в\s+)?комплект\w*\s+не\s+(?:вход\w*|включ\w*|поставля\w*)"
        rf"[^.;\n]{{0,40}}{target}",
        rf"{target}[^.;\n]{{0,48}}(?:в\s+)?комплект\w*\s+не\s+"
        rf"(?:вход\w*|включ\w*|поставля\w*)",
        rf"{target}[^.;\n]{{0,24}}не\s+(?:вход\w*|включ\w*|поставля\w*)"
        rf"\s+(?:в\s+)?комплект\w*",
        rf"(?:внутри\s+)?нет\s+{target}",
        rf"(?:не\s+содерж\w*|отсутств\w*)\s+{target}",
        rf"{target}\s+(?:отсутств\w*|не\s+поставля\w*)",
        rf"(?:supplied|sold|provided|comes?)\s+without\s+{target_en}",
        rf"{target_en}\s+(?:is\s+)?not\s+included",
        rf"(?:not\s+included|absent)\s*[:\-]?\s*{target_en}",
        rf"(?:empty\s+{target_en}|{target_en}\s+(?:is\s+)?empty)",
    ]
    if kind in {"flammable_gas", "flammable_liquid"}:
        patterns.extend((
            rf"(?:пуст\w*\s+{container}|{container}[^.;\n]{{0,24}}пуст\w*)",
            rf"{container}[^.;\n]{{0,24}}(?:не\s+(?:заправл|заполн)\w*|"
            rf"без\s+(?:содержим\w*|газ\w*|топлив\w*))",
            rf"(?:gas\s+)?(?:cylinder|canister)\s+without\s+"
            rf"(?:contents?|gas|fuel)",
        ))
    nuisance = re.compile(
        r"\b(?:запах|копот|дым|примес)\w*|"
        r"\b(?:odou?r|smell|soot|smoke|impurit)\w*",
        re.I,
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            window = text[max(0, match.start() - 24):match.end()]
            if nuisance.search(window):
                continue
            return True
    return False


def has_targeted_risk_inclusion(span: Any, kind: str) -> bool:
    """Recognise a literal same-card claim that the risk item is included."""

    target = _RISK_ABSENCE_TARGETS.get(kind)
    target_en = _RISK_ABSENCE_TARGETS_EN.get(kind)
    if not isinstance(span, str) or not target or not target_en:
        return False
    text = " ".join(span.casefold().replace("\u00a0", " ").split())
    patterns = (
        rf"(?:в\s+)?комплект\w*\s+(?:вход\w*|включ\w*)"
        rf"[^.;\n]{{0,32}}{target}",
        rf"{target}[^.;\n]{{0,40}}(?:вход\w*|включ\w*)\s+"
        rf"(?:в\s+)?комплект\w*",
        rf"(?:поставля\w*|прода\w*)\s+с\s+{target}",
        rf"(?:includes?|comes\s+with|supplied\s+with)\s+{target_en}",
        rf"{target_en}\s+(?:is\s+)?included",
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)

_BAD_TARGET_PATTERN = (
    r"(?:\bбад(?:ом|ами|ам|ах|ов|ы)?\b|"
    r"биологически\s+активн\w*\s+добавк\w*)"
)
_DIETARY_SUPPLEMENT_PATTERN = r"\b(?:dietary|food)\s+supplement\w*\b"
_NEGATED_BAD_RE = re.compile(
    rf"(?:"
    rf"\bне\s+(?:явля(?:ется|ются)\s+)?{_BAD_TARGET_PATTERN}|"
    rf"\bне\s+явля\w*\s+[^.;\n]{{0,60}}{_BAD_TARGET_PATTERN}|"
    rf"\bне\s+(?:относ\w*\s+к(?:\s+категории)?|"
    rf"классифицир\w*(?:\s+как|ся\s+как)?|"
    rf"зарегистрир\w*(?:\s+как|\s+в\s+качестве)?|"
    rf"использ\w*\s+в\s+качестве|"
    rf"счита\w*(?:ся)?(?:\s+как)?)\s+{_BAD_TARGET_PATTERN}|"
    rf"\bnot\s+(?:an?\s+)?(?:{_DIETARY_SUPPLEMENT_PATTERN})|"
    rf"\b(?:(?:is|are|was|were|does)\s+)?not\s+"
    rf"(?:constitute|qualify(?:\s+as)?|classif\w*(?:\s+as)?|"
    rf"register\w*(?:\s+as)?|market\w*(?:\s+as)?|"
    rf"intend\w*(?:\s+as)?|use\w*(?:\s+as)?)\s+"
    rf"(?:an?\s+)?{_DIETARY_SUPPLEMENT_PATTERN})",
    re.IGNORECASE,
)
_DIRECT_BAD_RE = re.compile(_BAD_TARGET_PATTERN, re.I)
_DIETARY_SUPPLEMENT_RE = re.compile(_DIETARY_SUPPLEMENT_PATTERN, re.I)
_INCIDENTAL_BAD_RE = re.compile(
    rf"(?:"
    rf"\b(?:схож\w*\s+(?:с|со)|похож\w*\s+на)\s+"
    rf"(?:{_BAD_TARGET_PATTERN}|{_DIETARY_SUPPLEMENT_PATTERN})|"
    rf"\bусилива\w*\s+действ\w*[^.;\n]{{0,60}}"
    rf"(?:{_BAD_TARGET_PATTERN}|{_DIETARY_SUPPLEMENT_PATTERN})|"
    rf"\b(?:принима\w*|употребля\w*)[^.;\n]{{0,50}}"
    rf"(?:{_BAD_TARGET_PATTERN}|{_DIETARY_SUPPLEMENT_PATTERN})|"
    rf"\b(?:между|в\s+отличие\s+от)[^.;\n]{{0,80}}"
    rf"(?:{_BAD_TARGET_PATTERN}|{_DIETARY_SUPPLEMENT_PATTERN})|"
    rf"\b(?:similar\s+to|used\s+with|use\s+with|people\s+taking)"
    rf"[^.;\n]{{0,50}}{_DIETARY_SUPPLEMENT_PATTERN})",
    re.IGNORECASE,
)
_SPORTS_NUTRITION_RE = re.compile(
    r"спортивн\w*\s+питан\w*|\bsports?\s+nutrition\b|\bBCAA\b|"
    r"\bL-?карнитин\w*\b|\bL-?carnitine\b|\bпротеин\w*\b|\bprotein\b|"
    r"\bаминокислот\w*\b|\bamino\s+acid\w*\b|\bкреатин\w*\b|\bcreatine\b",
    re.I,
)

_MECHANISMS_BY_KIND = {
    "bad_marking": {"direct_bad_marking"},
    "dietary_supplement_marking": {"dietary_supplement_marking"},
    "sports_nutrition_marking": {"sports_nutrition_marking"},
    "explicit_not_bad": {"explicit_exclusion"},
    "ignition_source": {"open_flame", "sustains_burning", "spark_only", "pyrotechnic"},
    "flammable_gas": {"flammable_contents"},
    "flammable_liquid": {"flammable_contents", "sustains_burning"},
    "solid_fuel": {"flammable_contents", "sustains_burning"},
    "pyrotechnic_charge": {"pyrotechnic"},
    "spark_source": {"spark_only"},
    "non_ignition_device": {"pneumatic", "heat_only", "none"},
    "combustible_component": {"flammable_contents", "sustains_burning", "none"},
    "unknown": {"unknown", "none"},
}


class SchemaValidationError(ValueError):
    """Raised when a generated rationale violates the grounded contract."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        super().__init__("Invalid grounded rationale:\n- " + "\n- ".join(self.issues))


@dataclass(frozen=True)
class PolicyDecision:
    """Deterministic result of applying official policy to grounded concepts."""

    verdict: str
    rule_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "official_policy_verdict": self.verdict,
            "policy_rule_ids": list(self.rule_ids),
            "policy_reasons": list(self.reasons),
        }


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from prose or a Markdown fence.

    Parsing stays strict JSON (no ``eval`` or Python literal fallback), while
    tolerating the leading/trailing text that instruction-tuned models often
    emit around an otherwise valid object.
    """

    if not isinstance(text, str) or not text.strip():
        raise SchemaValidationError(["model output is empty or is not a string"])

    decoder = json.JSONDecoder()
    parse_errors: list[str] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as exc:
            if len(parse_errors) < 3:
                parse_errors.append(
                    f"JSON candidate at character {match.start()}: {exc.msg}"
                )
            continue
        if isinstance(value, dict):
            return value

    detail = "; ".join(parse_errors) if parse_errors else "no opening object found"
    raise SchemaValidationError([f"could not extract a valid JSON object: {detail}"])


def extract_json_object_with_repairs(
    text: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract JSON while losslessly merging harmless duplicate list fields.

    Some instruction-tuned models emit two adjacent ``evidence`` keys.  Normal
    JSON parsing silently keeps only the last one, which can discard the only
    visual or textual grounding.  V9 merges duplicates *only* when both values
    are lists and the field is a list-valued extraction field.  Scalar
    duplicates retain ordinary JSON last-value behaviour and are reported, not
    guessed.  The original model string remains the source of truth in the raw
    journal; callers store the returned repair log beside it.
    """

    if not isinstance(text, str) or not text.strip():
        raise SchemaValidationError(["model output is empty or is not a string"])

    repairs: list[dict[str, Any]] = []
    mergeable = {"elements", "evidence", "observations", "conflicts"}

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key not in result:
                result[key] = value
                continue
            if key in mergeable and isinstance(result[key], list) and isinstance(value, list):
                result[key] = [*result[key], *value]
                repairs.append({
                    "action": "merge_duplicate_list_key",
                    "key": key,
                })
            else:
                repairs.append({
                    "action": "duplicate_scalar_key_last_value",
                    "key": key,
                })
                result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=object_pairs)
    parse_errors: list[str] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as exc:
            if len(parse_errors) < 3:
                parse_errors.append(
                    f"JSON candidate at character {match.start()}: {exc.msg}"
                )
            continue
        if isinstance(value, dict):
            return value, repairs

    detail = "; ".join(parse_errors) if parse_errors else "no opening object found"
    raise SchemaValidationError([f"could not extract a valid JSON object: {detail}"])


def parse_json_object(value: str | Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached dictionary from model text or an existing mapping."""

    if isinstance(value, str):
        return extract_json_object(value)
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise SchemaValidationError(["rationale must be a JSON object or model text"])


def _normalised_text(value: str) -> str:
    return " ".join(value.split())


def _contains_span(haystack: str, span: str) -> bool:
    return span in haystack or _normalised_text(span) in _normalised_text(haystack)


def _text(value: Any, path: str, issues: list[str], *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        issues.append(f"{path} must be a string")
        return ""
    cleaned = value.strip()
    if not cleaned and not allow_empty:
        issues.append(f"{path} must not be empty")
    return cleaned


def _enum(
    value: Any,
    allowed: Iterable[str],
    path: str,
    issues: list[str],
) -> str:
    result = _text(value, path, issues)
    allowed_set = frozenset(allowed)
    if result and result not in allowed_set:
        issues.append(f"{path}={result!r} is not one of {sorted(allowed_set)}")
    return result


def _unexpected_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    path: str,
    issues: list[str],
) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        issues.append(f"{path} has unexpected keys: {extra}")


def _validate_bbox(value: Any, path: str, issues: list[str]) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        issues.append(f"{path} must contain [x1, y1, x2, y2]")
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        issues.append(f"{path} coordinates must be numbers")
        return None
    bbox = [float(item) for item in value]
    if any(item < 0 or item > 1000 for item in bbox):
        issues.append(f"{path} coordinates must use the normalised 0..1000 range")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        issues.append(f"{path} must satisfy x2>x1 and y2>y1")
    return bbox


def _validate_evidence(
    value: Any,
    path: str,
    issues: list[str],
    *,
    name_text: str | None,
    description_text: str | None,
    ocr_text_by_image: Mapping[str, str] | None,
    known_image_ids: frozenset[str] | None,
    strict: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        issues.append(f"{path} must be an object")
        return {}
    value = dict(value)
    if strict:
        _unexpected_keys(value, _EVIDENCE_KEYS, path, issues)

    source = _enum(value.get("source"), EVIDENCE_SOURCES, f"{path}.source", issues)
    polarity = _enum(
        value.get("polarity"), EVIDENCE_POLARITIES, f"{path}.polarity", issues
    )
    exact_span = value.get("exact_span")
    image_id = value.get("image_id")
    observation = value.get("observation")
    bbox = _validate_bbox(value.get("bbox"), f"{path}.bbox", issues)

    if exact_span is not None:
        exact_span = _text(exact_span, f"{path}.exact_span", issues)
        if len(exact_span) > 400:
            issues.append(f"{path}.exact_span is longer than 400 characters")
    if image_id is not None:
        image_id = _text(image_id, f"{path}.image_id", issues)
    if observation is not None:
        observation = _text(observation, f"{path}.observation", issues)

    if source in {"name", "description", "ocr"} and not exact_span:
        issues.append(f"{path}.exact_span is required for source={source!r}")
    if source in {"ocr", "image"} and not image_id:
        issues.append(f"{path}.image_id is required for source={source!r}")
    if source == "image" and not observation:
        issues.append(f"{path}.observation is required for visual evidence")
    if bbox is not None and source not in {"ocr", "image"}:
        issues.append(f"{path}.bbox is only valid for OCR or image evidence")
    if source in {"name", "description"} and image_id:
        issues.append(f"{path}.image_id is not valid for source={source!r}")

    if image_id and known_image_ids is not None and image_id not in known_image_ids:
        issues.append(f"{path}.image_id={image_id!r} is not an input image")

    source_text: str | None = None
    if source == "name":
        source_text = name_text
    elif source == "description":
        source_text = description_text
    elif source == "ocr" and image_id:
        if ocr_text_by_image is None:
            issues.append(
                f"{path}.source='ocr' cannot be grounded without supplied OCR text"
            )
        else:
            source_text = ocr_text_by_image.get(image_id)
            if source_text is None:
                issues.append(f"{path}.image_id={image_id!r} has no supplied OCR text")

    if exact_span and source_text is not None and not _contains_span(source_text, exact_span):
        issues.append(f"{path}.exact_span is not present in the referenced source")

    result: dict[str, Any] = {"source": source, "polarity": polarity}
    if exact_span is not None:
        result["exact_span"] = exact_span
    if image_id is not None:
        result["image_id"] = image_id
    if bbox is not None:
        result["bbox"] = bbox
    if observation is not None:
        result["observation"] = observation
    return result


def _has_explicit_absence_cue(evidence: Sequence[Mapping[str, Any]]) -> bool:
    candidates = []
    for item in evidence:
        if item.get("polarity") != "support":
            continue
        candidates.extend(
            str(item.get(key, "")).casefold() for key in ("exact_span", "observation")
        )
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return False
    for candidate in candidates:
        if any(cue in candidate for cue in _NON_EVIDENCE_CUES):
            continue
        if any(cue in candidate for cue in _ABSENCE_CUES):
            return True
    return False


def _support_claims_nonpresence(
    evidence: Sequence[Mapping[str, Any]], kind: str
) -> bool:
    """Return true only for literal, target-scoped textual non-presence.

    Generic ``без`` properties, visual non-observation and pictured-use scenes
    cannot contradict ``state=present``.  OCR is allowed because its
    ``exact_span`` is literal package text and is grounded independently.
    """

    for item in evidence:
        if (
            item.get("polarity") != "support"
            or item.get("source") not in {"name", "description", "ocr"}
        ):
            continue
        candidate = item.get("exact_span")
        if has_targeted_risk_absence(candidate, kind):
            return True
    return False


def _has_targeted_risk_absence_evidence(
    evidence: Sequence[Mapping[str, Any]], kind: str
) -> bool:
    """Validate an absent/empty risk state against its actual referent."""

    for item in evidence:
        if item.get("polarity") not in {"support", "context"}:
            continue
        source = item.get("source")
        if source in {"name", "description", "ocr"}:
            candidate = item.get("exact_span")
        elif source == "image" and not _is_non_observation_evidence(item, kind):
            candidate = item.get("observation")
        else:
            continue
        if has_targeted_risk_absence(candidate, kind):
            return True
    return False


def _has_explicit_bad_marking_absence(
    kind: str,
    evidence: Sequence[Mapping[str, Any]],
) -> bool:
    """Require absence of the *marking*, not an unrelated absent property."""

    target = (
        _BAD_TARGET_PATTERN
        if kind == "bad_marking"
        else _DIETARY_SUPPLEMENT_PATTERN
    )
    marking = (
        r"(?:маркировк\w*|надпис\w*|указан\w*|"
        r"обозначен\w*|слов\w*|label\w*|marking\w*|statement\w*)"
    )
    absence = (
        r"(?:явно\s+)?(?:нет|отсутств\w*|"
        r"не\s+(?:име\w*|содерж\w*|нанес\w*|указ\w*)|"
        r"no|without|(?:is|are)\s+(?:absent|not\s+present))"
    )
    patterns = (
        rf"{absence}[^.;\n]{{0,60}}{marking}[^.;\n]{{0,60}}{target}",
        rf"{absence}[^.;\n]{{0,60}}{target}[^.;\n]{{0,40}}{marking}",
        rf"{marking}[^.;\n]{{0,60}}{target}[^.;\n]{{0,40}}{absence}",
        rf"{target}[^.;\n]{{0,40}}{marking}[^.;\n]{{0,40}}{absence}",
    )
    for item in evidence:
        if item.get("polarity") != "support":
            continue
        candidate = " ".join(
            str(item.get(key, "")) for key in ("exact_span", "observation")
        ).strip()
        folded = candidate.casefold()
        if not candidate or any(cue in folded for cue in _NON_EVIDENCE_CUES):
            continue
        if any(re.search(pattern, candidate, re.I) for pattern in patterns):
            return True
    return False


def _validate_bad_marking_semantics(
    kind: str,
    role: str,
    state: str,
    evidence: Sequence[Mapping[str, Any]],
    path: str,
    issues: list[str],
) -> None:
    """Reject polarity inversions that exact-span grounding alone cannot catch."""

    marking_kinds = {
        "bad_marking", "dietary_supplement_marking",
        "sports_nutrition_marking", "explicit_not_bad",
    }
    if kind not in marking_kinds:
        return
    if role != "main_product":
        issues.append(
            f"{path}.kind={kind!r} must have role='main_product'; a mention "
            "about another product is not target-package marking"
        )
    if state == "empty":
        issues.append(f"{path}.kind={kind!r} cannot have state='empty'")
    if (state == "explicitly_absent" and
            kind in {"sports_nutrition_marking", "explicit_not_bad"}):
        issues.append(
            f"{path}.kind={kind!r} cannot have state='explicitly_absent'; "
            "encode the actually observed target marking or use unknown"
        )
    if state != "present":
        return
    texts = [
        " ".join(
            str(item.get(key, "")) for key in ("exact_span", "observation")
        ).strip()
        for item in evidence
        if item.get("polarity") == "support"
    ]
    texts = [text for text in texts if text]
    if not texts:
        return

    def compatible(text: str) -> bool:
        negated = bool(_NEGATED_BAD_RE.search(text))
        incidental = bool(_INCIDENTAL_BAD_RE.search(text))
        if kind == "bad_marking":
            return bool(_DIRECT_BAD_RE.search(text)) and not negated and not incidental
        if kind == "dietary_supplement_marking":
            return (
                bool(_DIETARY_SUPPLEMENT_RE.search(text))
                and not negated
                and not incidental
            )
        if kind == "sports_nutrition_marking":
            return bool(_SPORTS_NUTRITION_RE.search(text))
        return negated

    if not any(compatible(text) for text in texts):
        issues.append(
            f"{path}.kind={kind!r} is not supported by a polarity-compatible "
            "marking span/observation"
        )


def _validate_element(
    value: Any,
    index: int,
    category: str,
    issues: list[str],
    *,
    name_text: str | None,
    description_text: str | None,
    ocr_text_by_image: Mapping[str, str] | None,
    known_image_ids: frozenset[str] | None,
    strict: bool,
) -> dict[str, Any]:
    path = f"elements[{index}]"
    if not isinstance(value, Mapping):
        issues.append(f"{path} must be an object")
        return {}
    value = dict(value)
    if strict:
        _unexpected_keys(value, _ELEMENT_KEYS, path, issues)

    element_name = _text(value.get("name"), f"{path}.name", issues)
    kinds = ELEMENT_KINDS.get(category, frozenset())
    kind = _enum(value.get("kind"), kinds, f"{path}.kind", issues)
    role = _enum(value.get("role"), ROLES, f"{path}.role", issues)
    state = _enum(value.get("state"), STATES, f"{path}.state", issues)
    mechanism = _enum(
        value.get("mechanism"), MECHANISMS, f"{path}.mechanism", issues
    )

    evidence_value = value.get("evidence")
    if not isinstance(evidence_value, list):
        issues.append(f"{path}.evidence must be a list")
        evidence_value = []
    if len(evidence_value) > 24:
        issues.append(f"{path}.evidence must contain at most 24 items")
    evidence = [
        _validate_evidence(
            item,
            f"{path}.evidence[{evidence_index}]",
            issues,
            name_text=name_text,
            description_text=description_text,
            ocr_text_by_image=ocr_text_by_image,
            known_image_ids=known_image_ids,
            strict=strict,
        )
        for evidence_index, item in enumerate(evidence_value)
    ]

    if category == "Легковоспламеняющиеся":
        has_text_support = any(
            item.get("source") in {"name", "description", "ocr"}
            and item.get("polarity") == "support"
            for item in evidence
        )
        for evidence_index, item in enumerate(evidence):
            strong_identity_denial = (
                item.get("source") == "image"
                and has_text_support
                and is_strong_visual_identity_denial(
                    item.get("observation"), kind
                )
            )
            support_denies_target = (
                item.get("source") == "image"
                and item.get("polarity") == "support"
                and image_support_denies_risk_target(
                    item.get("observation"), kind
                )
            )
            refute_affirms_target = (
                item.get("source") == "image"
                and item.get("polarity") == "refute"
                and image_refute_affirms_risk_target(
                    item.get("observation"), kind
                )
            )
            strong_identity_context = (
                item.get("polarity") == "context"
                and strong_identity_denial
            )
            if support_denies_target:
                if strong_identity_denial:
                    issues.append(
                        f"{path}.evidence[{evidence_index}] "
                        f"polarity='support' contradicts its target risk "
                        f"kind={kind!r}: the image positively identifies an "
                        "incompatible plain product form and explicitly denies "
                        "the risk element. Use refute only while this strong "
                        "identity evidence remains literal and grounded; "
                        "otherwise keep context/unknown"
                    )
                else:
                    issues.append(
                        f"{path}.evidence[{evidence_index}] "
                        f"polarity='support' contradicts its target risk "
                        f"kind={kind!r}: the image observation explicitly says "
                        "that risk element is not visible/present. Mere visual "
                        "non-observation is not refutation; use context/unknown "
                        "or obtain positive evidence"
                    )
            elif refute_affirms_target:
                issues.append(
                    f"{path}.evidence[{evidence_index}] polarity='refute' "
                    f"contradicts its target risk kind={kind!r}: the image "
                    "observation explicitly says that risk element is "
                    "visible/present. Keep the emitted polarity auditable and "
                    "repair or remove the inconsistent evidence"
                )
            elif strong_identity_context:
                issues.append(
                    f"{path}.evidence[{evidence_index}] polarity='context' "
                    f"cannot hide a strong visual identity denial for risk "
                    f"kind={kind!r} alongside grounded text support; encode "
                    "an explicit grounded refute or keep the case in "
                    "unknown/review"
                )

    supporting_evidence = [item for item in evidence if item.get("polarity") == "support"]
    has_state_evidence = bool(supporting_evidence)
    if (
        not has_state_evidence
        and category == "Легковоспламеняющиеся"
        and kind in RISK_SUBSTANCE_KINDS
        and state in {"explicitly_absent", "empty"}
    ):
        # Preserve a teacher's harmless ``context`` polarity when the literal
        # grounded span itself is an unambiguous target-specific absence cue.
        has_state_evidence = _has_targeted_risk_absence_evidence(evidence, kind)
    allowed_mechanisms = _MECHANISMS_BY_KIND.get(kind, set())
    if mechanism and allowed_mechanisms and mechanism not in allowed_mechanisms:
        issues.append(
            f"{path}.mechanism={mechanism!r} is incompatible with kind={kind!r}; "
            f"expected one of {sorted(allowed_mechanisms)}"
        )
    if state != "unknown" and not has_state_evidence:
        issues.append(f"{path}.state={state!r} requires supporting evidence")
    absence_sensitive_kinds = {
        "flammable_gas", "flammable_liquid", "solid_fuel",
        "pyrotechnic_charge",
    }
    if (state == "present" and kind in absence_sensitive_kinds and
            _support_claims_nonpresence(evidence, kind)):
        issues.append(
            f"{path}.state='present' contradicts its supporting evidence; "
            "encode absent/empty/unknown as appropriate"
        )
    if category == "БАД":
        _validate_bad_marking_semantics(
            kind, role, state, evidence, path, issues
        )
    if state in {"explicitly_absent", "empty"}:
        bad_absence = (
            category == "БАД"
            and kind in {"bad_marking", "dietary_supplement_marking"}
            and state == "explicitly_absent"
        )
        if bad_absence:
            has_absence = _has_explicit_bad_marking_absence(kind, evidence)
        elif category == "БАД" and kind in {
                "bad_marking", "dietary_supplement_marking",
                "sports_nutrition_marking", "explicit_not_bad",
        }:
            # The BAD semantic validator reports the invalid state itself.
            has_absence = True
        elif category == "Легковоспламеняющиеся" and kind in RISK_SUBSTANCE_KINDS:
            has_absence = _has_targeted_risk_absence_evidence(evidence, kind)
        else:
            has_absence = _has_explicit_absence_cue(evidence)
        if not has_absence:
            if bad_absence:
                issues.append(
                    f"{path}.state={state!r} requires explicit evidence that "
                    "the target-category marking itself is absent; an "
                    "unrelated 'without' property is insufficient and 'not "
                    "visible' must be encoded as unknown"
                )
            else:
                issues.append(
                    f"{path}.state={state!r} requires an explicit "
                    "absence/empty cue; 'not visible' must be encoded as "
                    "unknown"
                )

    return {
        "name": element_name,
        "kind": kind,
        "role": role,
        "state": state,
        "mechanism": mechanism,
        "evidence": evidence,
    }


def evaluate_policy(record: Mapping[str, Any]) -> PolicyDecision:
    """Apply task-statement policy to an already validated blind record.

    This executor verifies rationale consistency.  It is not a hard leaderboard
    override: noisy assessor labels and empirical conventions must be measured
    separately on OOF predictions.
    """

    category = record.get("category")
    elements = record.get("elements")
    if category not in CATEGORIES or not isinstance(elements, list):
        raise SchemaValidationError(["evaluate_policy requires a validated record"])

    if record.get("modality_relation") in {"conflict", "insufficient"}:
        rule = (
            "POLICY-REVIEW-MODALITY-CONFLICT"
            if record.get("modality_relation") == "conflict"
            else "POLICY-REVIEW-INSUFFICIENT-EVIDENCE"
        )
        return PolicyDecision(
            "review",
            (rule,),
            ("modalities conflict or decisive evidence is insufficient",),
        )
    if record.get("conflicts"):
        return PolicyDecision(
            "review",
            ("POLICY-REVIEW-EVIDENCE-CONFLICT",),
            ("the record contains unresolved evidence conflicts",),
        )

    if category == "БАД":
        return _evaluate_bad_policy(elements)
    return _evaluate_flammable_policy(elements)


def element_policy_direction(category: str, element: Mapping[str, Any]) -> int:
    """Return the official direction fired by one grounded element.

    This deliberately delegates to the same executor used for complete
    rationales.  Claim exporters must not maintain a second handwritten copy
    of the task policy: even a small drift (for example around spark-only
    mechanisms) can attach the wrong sign to otherwise valid visual evidence.
    ``0`` means that the element alone is not decisive.
    """

    decision = evaluate_policy({
        "category": category,
        "elements": [element],
        "modality_relation": "agree",
        "conflicts": [],
    })
    return {"yes": 1, "no": -1, "review": 0}[decision.verdict]


def _evaluate_bad_policy(elements: Sequence[Mapping[str, Any]]) -> PolicyDecision:
    positive_rules: list[tuple[str, str]] = []
    negative_rules: list[tuple[str, str]] = []

    # BAD is a target-identity/target-package decision.  A mention of a BAD in
    # the contents, usage instructions, or another bundled product is not a
    # marking of the sold target itself.
    target_roles = {"main_product"}
    for element in elements:
        kind = element.get("kind")
        role = element.get("role")
        state = element.get("state")
        if role not in target_roles:
            continue
        if state == "present" and kind in {
            "bad_marking",
            "dietary_supplement_marking",
        }:
            positive_rules.append(
                ("BAD-POS-DIRECT-MARKING", "direct БАД/dietary supplement marking")
            )
        if state == "present" and kind == "sports_nutrition_marking":
            negative_rules.append(
                ("BAD-NEG-SPORTS-NUTRITION", "direct sports-nutrition indication")
            )
        if state == "present" and kind == "explicit_not_bad":
            negative_rules.append(
                ("BAD-NEG-EXPLICIT-EXCLUSION", "explicit statement that it is not БАД")
            )
        if state == "explicitly_absent" and kind in {
            "bad_marking",
            "dietary_supplement_marking",
        }:
            negative_rules.append(
                ("BAD-NEG-MARKING-EXPLICITLY-ABSENT", "marking is explicitly absent")
            )

    if positive_rules and negative_rules:
        return PolicyDecision(
            "review",
            ("BAD-REVIEW-CONFLICT",),
            ("positive and exclusionary category signals conflict",),
        )
    if positive_rules:
        return _decision_from_rules("yes", positive_rules)
    if negative_rules:
        return _decision_from_rules("no", negative_rules)
    return PolicyDecision(
        "review",
        ("BAD-REVIEW-MARKING-UNKNOWN",),
        ("no grounded direct marking or grounded exclusion was found",),
    )


def _evaluate_flammable_policy(
    elements: Sequence[Mapping[str, Any]],
) -> PolicyDecision:
    positive_rules: list[tuple[str, str]] = []
    negative_rules: list[tuple[str, str]] = []
    flammable_kinds = {
        "flammable_gas",
        "flammable_liquid",
        "solid_fuel",
        "pyrotechnic_charge",
    }
    positive_content_mechanisms = {
        "flammable_gas": {"flammable_contents"},
        "flammable_liquid": {"flammable_contents", "sustains_burning"},
        "solid_fuel": {"flammable_contents", "sustains_burning"},
        "pyrotechnic_charge": {"pyrotechnic"},
    }

    for element in elements:
        kind = element.get("kind")
        role = element.get("role")
        state = element.get("state")
        mechanism = element.get("mechanism")

        if state in {"explicitly_absent", "empty"} and kind in flammable_kinds | {
            "ignition_source"
        }:
            rule = (
                "FLAM-NEG-EMPTY-CONTENTS"
                if state == "empty"
                else "FLAM-NEG-NOT-INCLUDED"
            )
            negative_rules.append((rule, "risk element is explicitly absent or empty"))
            continue
        if state != "present":
            continue

        if role == "separate_in_kit" and (
            (kind in flammable_kinds and
             mechanism in positive_content_mechanisms[kind]) or
            (kind == "ignition_source" and mechanism in {
                "open_flame", "sustains_burning", "pyrotechnic"
            })
        ):
            positive_rules.append(
                ("FLAM-POS-SEPARATE-IN-KIT", "flammable item is separately included")
            )
            continue
        if (role in {"main_product", "content_of"} and
                kind in flammable_kinds and
                mechanism in positive_content_mechanisms[kind]):
            positive_rules.append(
                ("FLAM-POS-FLAMMABLE-CONTENTS", "product contains a flammable substance")
            )
            continue
        if (
            role == "main_product"
            and kind == "ignition_source"
            and mechanism in {"open_flame", "sustains_burning", "pyrotechnic"}
        ):
            positive_rules.append(
                ("FLAM-POS-STANDALONE-IGNITION", "main purpose is creating open fire")
            )
            continue

        if role == "integrated" and kind in {"ignition_source", "spark_source"}:
            negative_rules.append(
                ("FLAM-NEG-INTEGRATED-IGNITION", "ignition source is integrated")
            )
        elif role == "component_of" and kind in flammable_kinds | {
            "combustible_component"
        }:
            negative_rules.append(
                ("FLAM-NEG-COMPONENT", "combustible material is only a component")
            )
        elif role in {"compatible_only", "mentioned_only", "pictured_use_only"}:
            negative_rules.append(
                ("FLAM-NEG-NOT-PART-OF-PRODUCT", "risk item is not part of the sold product")
            )
        elif kind == "spark_source" or mechanism in {
            "spark_only",
            "pneumatic",
            "heat_only",
        }:
            negative_rules.append(
                ("FLAM-NEG-NO-OPEN-FLAME", "mechanism does not create open fire")
            )

    # Flammability policy is existential: one qualifying standalone/included
    # risk element is sufficient.  An integrated piezo in the same burner does
    # not cancel a separately included filled gas cylinder.  Genuine evidence
    # contradictions must be represented in `conflicts`/modality_relation and
    # have already returned REVIEW above.
    if positive_rules:
        return _decision_from_rules("yes", positive_rules)
    if negative_rules:
        return _decision_from_rules("no", negative_rules)
    return PolicyDecision(
        "review",
        ("FLAM-REVIEW-RISK-UNKNOWN",),
        ("no grounded decisive risk role was found",),
    )


def _decision_from_rules(
    verdict: str, rules: Sequence[tuple[str, str]]
) -> PolicyDecision:
    rule_ids = tuple(dict.fromkeys(rule for rule, _ in rules))
    reasons = tuple(dict.fromkeys(reason for _, reason in rules))
    return PolicyDecision(verdict, rule_ids, reasons)


def _evidence_modality(evidence: Mapping[str, Any]) -> str | None:
    source = evidence.get("source")
    if source in {"name", "description"}:
        return "text"
    if source in {"image", "ocr"}:
        return "image"
    return None


def _is_non_observation_evidence(
    evidence: Mapping[str, Any], kind: str | None = None
) -> bool:
    """Return true for image non-observation masquerading as absence evidence."""

    if evidence.get("source") not in {"image", "ocr"}:
        return False
    text = " ".join(
        str(evidence.get(key, "")).casefold()
        for key in ("exact_span", "observation")
    )
    if not text:
        return False
    if kind is not None and is_strong_visual_identity_denial(text, kind):
        return False
    return any(cue in text for cue in _NON_EVIDENCE_CUES) or (
        kind is not None and image_support_denies_risk_target(text, kind)
    )


_RISK_ENTITY_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    "ignition_source": (
        (
            "ignition_device",
            r"\b(?:зажигалк\w*|lighter|ignition\s+(?:device|source)|"
            r"source\s+of\s+ignition|устройств\w*\s+зажигани\w*|"
            r"источник\w*\s+(?:зажигания|воспламенения|огня))\b",
        ),
        ("matches", r"\b(?:спич\w*|matches?|matchbox)\b"),
        ("firestarter", r"\b(?:огнив\w*|fire[-\s]?starter)\b"),
        ("burner", r"\b(?:горелк\w*|burner|torch)\b"),
    ),
    "spark_source": (
        (
            "spark_ignition",
            r"\b(?:пьезо\w*|поджиг\w*|искр\w*|piezo\w*|"
            r"spark\s+(?:source|igniter)|igniter|"
            r"ignition\s+(?:device|source)|устройств\w*\s+зажигани\w*)\b",
        ),
    ),
    "flammable_gas": (
        (
            "flammable_gas",
            r"\b(?:газ\w*|бутан\w*|пропан\w*|gas|butane|propane)\b",
        ),
    ),
    "flammable_liquid": (
        (
            "flammable_liquid",
            r"\b(?:горюч\w*\s+жидк\w*|бензин\w*|керосин\w*|спирт\w*|"
            r"flammable\s+liquid|gasoline|petrol|kerosene|alcohol)\b",
        ),
    ),
    "solid_fuel": (
        (
            "solid_fuel",
            r"\b(?:тв[её]рд\w*\s+топлив\w*|угол\w*|дров\w*|свеч\w*|"
            r"solid\s+fuel|charcoal|coal|firewood|candles?)\b",
        ),
    ),
    "pyrotechnic_charge": (
        (
            "pyrotechnic_charge",
            r"\b(?:пиротехн\w*|пиросостав\w*|порох\w*|заряд\w*|"
            r"pyrotechnic\w*|gunpowder|pyro\s+charge)\b",
        ),
    ),
}


def _risk_entity_alias(kind: str, text: str) -> str | None:
    folded = " ".join(str(text).casefold().replace("ё", "е").split())
    for alias, pattern in _RISK_ENTITY_ALIASES.get(kind, ()):
        if re.search(pattern, folded, re.I):
            return alias
    return None


def risk_entity_key(element: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return a conservative identity key for policy-relevant risk evidence."""

    kind = str(element.get("kind", ""))
    name = " ".join(str(element.get("name", "")).casefold().replace("ё", "е").split())
    if not kind or kind in {"unknown", "non_ignition_device"} or not name:
        return None
    grounded = " ".join(
        str(item.get("exact_span", ""))
        for item in element.get("evidence", [])
        if isinstance(item, Mapping)
        and item.get("source") in {"name", "description", "ocr"}
        and item.get("polarity") == "support"
    )
    alias = _risk_entity_alias(kind, f"{name} {grounded}")
    if alias is not None:
        return kind, alias
    normalised = re.sub(r"[^0-9a-zа-я]+", " ", name, flags=re.I).strip()
    if normalised in {
        "элемент", "устройство", "товар", "продукт", "компонент",
        "element", "device", "product", "component",
    }:
        return None
    return (kind, normalised) if normalised else None


def risk_entity_keys(
    element: Mapping[str, Any],
) -> frozenset[tuple[str, str]]:
    """Infer narrow lexical risk identities independent of emitted ``kind``.

    The teacher-authored kind is not a safety boundary.  A literal
    ``зажигалка`` remains an ignition identity even if the model puts it in a
    ``solid_fuel`` or ``flammable_gas`` slot.  Conversely, a grounded gas
    cylinder span yields only the gas identity, so an ignition-image taint does
    not suppress an unrelated gas trigger.
    """

    grounded = " ".join(
        str(item.get("exact_span", ""))
        for item in element.get("evidence", [])
        if isinstance(item, Mapping)
        and item.get("source") in {"name", "description"}
        and item.get("polarity") == "support"
    )
    identity_text = f"{element.get('name', '')} {grounded}".strip()
    keys = set()
    for kind in _RISK_ENTITY_ALIASES:
        alias = _risk_entity_alias(kind, identity_text)
        if alias is not None:
            keys.add((kind, alias))
    return frozenset(keys)


def risk_entity_denial_matches(
    element: Mapping[str, Any],
    denied_keys: Iterable[tuple[str, str]],
) -> frozenset[tuple[str, str]]:
    """Match a policy element to strong-denial identities fail-closed.

    Grounded lexical identities take precedence and must match exactly.  A
    generic *or kind-inconsistent* teacher name/span cannot erase a strong
    contradiction, however: also match the emitted policy risk family. Thus an
    ``ignition_source`` named ``element`` or even ``gas cylinder`` still
    matches an ignition denial, while a genuinely grounded element emitted as
    ``flammable_gas`` remains independent from it.
    """

    denied = frozenset(denied_keys)
    lexical = risk_entity_keys(element)
    emitted_kind = str(element.get("kind", ""))
    matched = set(lexical & denied)
    if emitted_kind in _RISK_ENTITY_ALIASES:
        matched.update(key for key in denied if key[0] == emitted_kind)
    return frozenset(matched)


def visual_risk_denial_entity_keys(
    observation: Any, *, strong_only: bool = False
) -> frozenset[tuple[str, str]]:
    """Infer denied risk families from observation text, not emitted slots."""

    if not isinstance(observation, str):
        return frozenset()
    keys: set[tuple[str, str]] = set()
    for kind in _RISK_ENTITY_ALIASES:
        denied = (
            is_strong_visual_identity_denial(observation, kind)
            if strong_only
            else image_support_denies_risk_target(observation, kind)
        )
        if not denied:
            continue
        alias = _risk_entity_alias(kind, observation)
        if alias is not None:
            keys.add((kind, alias))
    return frozenset(keys)


def visual_denial_scopes_complete_kit(observation: Any) -> bool:
    """Return true only when an observation explicitly covers all kit items."""

    text = " ".join(str(observation).casefold().replace("ё", "е").split())
    incomplete_scope = re.search(
        r"\bnot\s+(?:all|the\s+(?:entire|complete))\s+"
        r"(?:kit|set|package)\s+(?:contents?|items?)\b|"
        r"\b(?:photo|image|picture)\s+(?:does|do)\s+not\s+show\s+"
        r"(?:all|the\s+(?:entire|complete))\s+(?:kit|set|package)\s+"
        r"(?:contents?|items?)\b|"
        r"\bне\s+все\s+(?:предметы|элементы|содержимое)\s+"
        r"(?:набора|комплекта)\b|\b(?:фото|изображение)\s+не\s+"
        r"показыва\w*\s+(?:весь|полный)\s+(?:набор|комплект)\b",
        text,
        re.I,
    )
    if incomplete_scope:
        return False
    return bool(re.search(
        r"\b(?:all|entire|complete)\s+(?:kit|set|package)\s+"
        r"(?:contents?|items?)\b|\b(?:вся|полная)\s+комплектаци\w*|"
        r"\bвсе\s+(?:предметы|элементы|содержимое)\s+(?:набора|комплекта)\b",
        text,
        re.I,
    ))


def _record_wide_visual_support_denial_issues(
    elements: Sequence[Mapping[str, Any]],
    product_form: str,
    issues: list[str],
) -> None:
    """Reject a strong denial mislabeled as support in another element.

    Per-element validation catches the ordinary case.  This second narrow
    pass closes the model-authored kind/name escape hatch: a teacher can put
    the plain-bracelet observation in a separate ``non_ignition_device`` slot,
    but that does not make ``no visible ignition device`` support a grounded
    positive lighter claim elsewhere in the same product card.
    """

    positive_targets: list[tuple[int, Mapping[str, Any], str, str]] = []
    for index, element in enumerate(elements):
        has_text_support = any(
            item.get("source") in {"name", "description"}
            and item.get("polarity") == "support"
            for item in element.get("evidence", [])
        )
        if (
            has_text_support
            and element_policy_direction("Легковоспламеняющиеся", element) > 0
        ):
            positive_targets.append(
                (
                    index,
                    element,
                    str(element.get("role", "")),
                    str(element.get("kind", "")),
                )
            )

    for image_index, element in enumerate(elements):
        for evidence_index, item in enumerate(element.get("evidence", [])):
            if (
                item.get("source") != "image"
                or item.get("polarity") != "support"
            ):
                continue
            denied_keys = visual_risk_denial_entity_keys(
                item.get("observation"), strong_only=True
            )
            if not denied_keys:
                continue
            complete_kit = visual_denial_scopes_complete_kit(
                item.get("observation")
            )
            for text_index, target, role, emitted_kind in positive_targets:
                matched_keys = risk_entity_denial_matches(target, denied_keys)
                # The ordinary same-slot case has a more precise issue from
                # `_validate_element`; avoid duplicating it here.
                if not matched_keys:
                    continue
                if (
                    text_index == image_index
                    and image_support_denies_risk_target(
                        item.get("observation"), emitted_kind
                    )
                ):
                    continue
                if (
                    role == "separate_in_kit"
                    and product_form == "kit"
                    and not complete_kit
                ):
                    continue
                issues.append(
                    f"elements[{image_index}].evidence[{evidence_index}] "
                    "polarity='support' contradicts grounded positive text "
                    f"evidence in elements[{text_index}] for "
                    f"entities={sorted(matched_keys)!r}; preserve the strong image denial "
                    "as refute/REVIEW or return a non-positive extraction"
                )


def derive_modality_relation(
    elements: Sequence[Mapping[str, Any]],
    *,
    product_form: str | None = None,
) -> tuple[str, list[str]]:
    """Derive cross-modal agreement from grounded evidence polarities.

    A teacher-written ``conflicts`` sentence is not evidence.  In particular,
    failure to see a marking on a photograph does not refute a literal text
    statement.  ``conflict`` is emitted only when one modality supports the
    same extracted element and the other contains an explicit grounded
    refutation. The one deliberately fail-closed exception is a narrow strong
    product-identity denial, which remains REVIEW even if emitted as context.
    A teacher may split one semantic entity into duplicate slots, so matching
    evidence is also aggregated across elements.
    """

    modalities: set[str] = set()
    conflicts: list[str] = []
    entity_evidence: dict[
        tuple[str, str], dict[str, dict[str, set[int]]]
    ] = {}
    positive_text_targets: list[tuple[int, Mapping[str, Any], str]] = []
    strong_image_denials: list[
        tuple[int, frozenset[tuple[str, str]], bool, str]
    ] = []
    for index, element in enumerate(elements):
        by_modality: dict[str, set[str]] = {"text": set(), "image": set()}
        generic_by_modality: dict[str, set[str]] = {
            "text": set(), "image": set()
        }
        has_text_support = False
        for item in element.get("evidence", []):
            modality = _evidence_modality(item)
            if modality is None:
                continue
            observation = item.get("observation")
            denied_keys = (
                visual_risk_denial_entity_keys(observation)
                if modality == "image" else frozenset()
            )
            strong_denied_keys = (
                visual_risk_denial_entity_keys(
                    observation, strong_only=True
                )
                if denied_keys else frozenset()
            )
            # A target denial attached to the wrong teacher-authored kind is
            # still non-evidence when weak. Strong identity evidence is handled
            # record-wide below, independently of that unreliable slot.
            if (denied_keys and not strong_denied_keys) or (
                not denied_keys
                and _is_non_observation_evidence(
                    item, str(element.get("kind", ""))
                )
            ):
                continue
            modalities.add(modality)
            polarity = item.get("polarity")
            if polarity in {"support", "refute", "context"}:
                polarity = str(polarity)
                if modality == "text" and polarity == "support":
                    has_text_support = True
                is_strong_denial = (
                    modality == "image"
                    and polarity in {"refute", "context"}
                    and bool(strong_denied_keys)
                )
                if is_strong_denial:
                    complete_kit = visual_denial_scopes_complete_kit(
                        observation
                    )
                    strong_image_denials.append(
                        (index, strong_denied_keys, complete_kit, polarity)
                    )
                    # A refute attached to the same semantic element always
                    # remains a refute.  In particular, a teacher-authored
                    # ``separate_in_kit`` mis-role must not suppress the live
                    # bracelet contradiction.  Context is handled by the
                    # record-wide narrow identity check below.
                    if polarity == "refute":
                        by_modality[modality].add(polarity)
                    # Strong denials use the record-wide,
                    # observation-derived target keys below, not the
                    # model-authored element kind/name.
                    continue
                if polarity == "context":
                    continue
                by_modality[modality].add(polarity)
                generic_by_modality[modality].add(polarity)
        entity_key = risk_entity_key(element)
        if (
            has_text_support
            and element_policy_direction("Легковоспламеняющиеся", element) > 0
        ):
            positive_text_targets.append(
                (index, element, str(element.get("role", "")))
            )
        if entity_key is not None:
            aggregate = entity_evidence.setdefault(
                entity_key,
                {
                    "text": {"support": set(), "refute": set()},
                    "image": {"support": set(), "refute": set()},
                },
            )
            for modality, polarities in generic_by_modality.items():
                for polarity in polarities:
                    aggregate[modality][polarity].add(index)
        cross_refutation = (
            ("support" in by_modality["text"] and "refute" in by_modality["image"])
            or
            ("refute" in by_modality["text"] and "support" in by_modality["image"])
        )
        if cross_refutation:
            conflicts.append(
                f"elements[{index}] has grounded support/refute evidence across modalities"
            )

    def has_distinct_pair(left: set[int], right: set[int]) -> bool:
        return any(left_index != right_index for left_index in left for right_index in right)

    for entity_key, aggregate in entity_evidence.items():
        text_support_image_refute = has_distinct_pair(
            aggregate["text"]["support"], aggregate["image"]["refute"]
        )
        text_refute_image_support = has_distinct_pair(
            aggregate["text"]["refute"], aggregate["image"]["support"]
        )
        if text_support_image_refute or text_refute_image_support:
            involved = sorted({
                index
                for modality in aggregate.values()
                for indices in modality.values()
                for index in indices
            })
            conflicts.append(
                f"elements{involved} share grounded support/refute evidence "
                f"across modalities for entity={entity_key!r}"
            )

    for text_index, target, role in positive_text_targets:
        for image_index, denied_keys, complete_kit, polarity in strong_image_denials:
            matched_keys = risk_entity_denial_matches(target, denied_keys)
            if not matched_keys:
                continue
            if text_index == image_index and polarity == "refute":
                # Already emitted by the within-element support/refute path.
                continue
            if (
                text_index != image_index
                and role == "separate_in_kit"
                and product_form == "kit"
                and not complete_kit
            ):
                continue
            conflicts.append(
                f"elements[{text_index}, {image_index}] have grounded text "
                f"support and a strong visual identity {polarity} denial for "
                f"entities={sorted(matched_keys)!r}"
            )

    if not modalities:
        return "insufficient", []
    if conflicts:
        return "conflict", conflicts
    if modalities == {"text", "image"}:
        return "agree", []
    if modalities == {"text"}:
        return "text_only", []
    return "image_only", []


def apply_counterfactual_interventions(
    record: Mapping[str, Any],
    interventions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the auditable V9 field replacements used for policy simulation."""

    changed = copy.deepcopy(dict(record))
    elements = changed.get("elements")
    if not isinstance(elements, list):
        raise SchemaValidationError(["counterfactual requires an elements list"])
    issues: list[str] = []
    for index, intervention in enumerate(interventions):
        path = f"counterfactual.interventions[{index}]"
        if not isinstance(intervention, Mapping):
            issues.append(f"{path} must be an object")
            continue
        item = dict(intervention)
        _unexpected_keys(item, _INTERVENTION_KEYS, path, issues)
        if item.get("op") != "replace_fields":
            issues.append(f"{path}.op must be 'replace_fields'")
        element_index = item.get("element_index")
        if (isinstance(element_index, bool) or
                not isinstance(element_index, int) or
                not 0 <= element_index < len(elements)):
            issues.append(f"{path}.element_index is outside the elements list")
            continue
        replacements = item.get("set")
        if not isinstance(replacements, Mapping) or not replacements:
            issues.append(f"{path}.set must be a non-empty object")
            continue
        replacements = dict(replacements)
        unexpected = sorted(set(replacements) - _INTERVENTION_FIELDS)
        if unexpected:
            issues.append(f"{path}.set has unsupported fields: {unexpected}")
            continue
        for field, value in replacements.items():
            allowed = {
                "kind": ELEMENT_KINDS.get(changed.get("category"), frozenset()),
                "role": ROLES,
                "state": STATES,
                "mechanism": MECHANISMS,
            }[field]
            if value not in allowed:
                issues.append(f"{path}.set.{field}={value!r} is invalid")
                continue
            elements[element_index][field] = value
    if issues:
        raise SchemaValidationError(issues)
    return changed


def _replace_fields_intervention(
    element_index: int, **fields: str
) -> dict[str, Any]:
    return {
        "op": "replace_fields",
        "element_index": element_index,
        "set": fields,
    }


def _positive_counterfactual_replacement(
    category: str, element_index: int, element: Mapping[str, Any]
) -> dict[str, Any]:
    """Neutralise one positive trigger while leaving all evidence untouched."""

    if category == "БАД":
        return _replace_fields_intervention(
            element_index,
            kind="explicit_not_bad",
            role="main_product",
            state="present",
            mechanism="explicit_exclusion",
        )
    # ``empty`` is a meaningful counterfactual for a substance/container, but
    # not for a separately sold candle, charcoal briquette or fountain.  Those
    # discrete objects must be absent; calling a candle "empty" is formally
    # executable but semantically nonsensical audit slop.
    if element.get("kind") in {"flammable_gas", "flammable_liquid"} or (
        element.get("kind") in {"solid_fuel", "pyrotechnic_charge"}
        and element.get("role") == "content_of"
    ):
        return _replace_fields_intervention(element_index, state="empty")
    return _replace_fields_intervention(element_index, state="explicitly_absent")


def _negative_to_positive_candidates(
    category: str, element_index: int, element: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if category == "БАД":
        return [_replace_fields_intervention(
            element_index,
            kind="bad_marking",
            role="main_product",
            state="present",
            mechanism="direct_bad_marking",
        )]

    kind = element.get("kind")
    role = element.get("role")
    state = element.get("state")
    candidates: list[dict[str, Any]] = []
    if kind in {"flammable_gas", "flammable_liquid", "solid_fuel", "pyrotechnic_charge"}:
        fields: dict[str, str] = {"state": "present"}
        if role in {"compatible_only", "mentioned_only", "pictured_use_only", "component_of"}:
            fields["role"] = "separate_in_kit"
        candidates.append(_replace_fields_intervention(element_index, **fields))
    if kind in {"ignition_source", "spark_source", "non_ignition_device"}:
        candidates.append(_replace_fields_intervention(
            element_index,
            kind="ignition_source",
            role="main_product" if role != "separate_in_kit" else "separate_in_kit",
            state="present",
            mechanism="open_flame",
        ))
    if kind == "combustible_component":
        candidates.append(_replace_fields_intervention(
            element_index,
            kind="solid_fuel",
            role="main_product",
            state="present",
            mechanism="sustains_burning",
        ))
    if not candidates and state != "present":
        candidates.append(_replace_fields_intervention(element_index, state="present"))
    return candidates


def _prefer_negative_flammable_activation(
    category: str, element: Mapping[str, Any]
) -> bool:
    """Prefer the nearest grounded fuel-boundary contrast for FLAM negatives.

    A negative appliance record often contains both an integrated spark and a
    flammable substance which is empty, explicitly absent, or merely
    compatible.  Activating that already extracted substance is the smaller,
    more informative contrast; rewriting the spark into a standalone lighter
    invents a different product.  Candidate execution below still decides
    whether the single intervention actually flips the verdict.
    """

    return (
        category == "Легковоспламеняющиеся"
        and element.get("kind") in {
            "flammable_gas",
            "flammable_liquid",
            "solid_fuel",
            "pyrotechnic_charge",
        }
        and (
            element.get("state") in {"explicitly_absent", "empty"}
            or element.get("role") == "compatible_only"
        )
    )


def derive_counterfactual(
    record: Mapping[str, Any],
    decision: PolicyDecision | None = None,
) -> dict[str, Any]:
    """Build and execute a deterministic minimal policy contrast.

    For an existential positive, every independently sufficient trigger must be
    neutralised.  Consequently ``interventions`` is a minimal hitting set and
    may contain more than one item; V9 never claims that removing just one
    lighter flips a kit that still contains fuel.
    """

    decision = decision or evaluate_policy(record)
    interventions: list[dict[str, Any]] = []
    resulting = decision

    if decision.verdict == "yes":
        decisive_indices = [
            index
            for index, element in enumerate(record.get("elements", []))
            if element_policy_direction(str(record.get("category")), element) == 1
        ]
        interventions = [
            _positive_counterfactual_replacement(
                str(record.get("category")), index, record["elements"][index]
            )
            for index in decisive_indices
        ]
        if interventions:
            resulting = evaluate_policy(
                apply_counterfactual_interventions(record, interventions)
            )
    elif decision.verdict == "no":
        category = str(record.get("category"))
        indexed_elements = list(enumerate(record.get("elements", [])))
        # Stable partition only: within each group preserve extraction order.
        # This changes no candidates and cannot bypass the executable-flip
        # check; it merely tries the grounded absent/empty/external fuel
        # boundary before synthesising a standalone ignition source.
        indexed_elements.sort(
            key=lambda item: not _prefer_negative_flammable_activation(
                category, item[1]
            )
        )
        for index, element in indexed_elements:
            for candidate in _negative_to_positive_candidates(
                category, index, element
            ):
                candidate_result = evaluate_policy(
                    apply_counterfactual_interventions(record, [candidate])
                )
                if candidate_result.verdict == "yes":
                    interventions = [candidate]
                    resulting = candidate_result
                    break
            if interventions:
                break

    if not interventions:
        summary = "решающий машинно проверяемый контраст не построен"
    elif len(interventions) == 1:
        summary = "заменить policy-relevant поля одного решающего элемента"
    else:
        summary = (
            "нейтрализовать все независимые положительные триггеры "
            f"({len(interventions)} элемента)"
        )
    return {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "single_change": summary,
        "interventions": interventions,
        "expected_verdict": resulting.verdict,
        "resulting_policy_rule_ids": list(resulting.rule_ids),
    }


def _derived_conclusion(record: Mapping[str, Any]) -> str:
    elements = record.get("elements", [])
    if not elements:
        return "Решающих grounded-элементов не извлечено."
    facts = [
        (
            f"{element['name']}: kind={element['kind']}, role={element['role']}, "
            f"state={element['state']}, mechanism={element['mechanism']}"
        )
        for element in elements
    ]
    return "Структурированные grounded-факты: " + "; ".join(facts) + "."


def finalize_extraction_record(
    value: str | Mapping[str, Any],
    *,
    name_text: str | None = None,
    description_text: str | None = None,
    ocr_text_by_image: Mapping[str, str] | None = None,
    known_image_ids: Iterable[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate teacher evidence/concepts, then derive every decision field.

    This is the V9 boundary.  The model is not trusted to choose modality
    agreement, apply policy, write a verdict, or invent a contrast.  Legacy V1
    extraction objects are accepted here so raw V8 proposals can be replayed,
    but all model-authored derived fields are ignored by design.
    """

    extraction = parse_json_object(value)
    issues: list[str] = []
    forbidden = sorted(set(extraction) & _POST_LABEL_KEYS)
    if forbidden:
        issues.append(
            "blind extraction contains post-label fields: "
            f"{forbidden}; attach them only after deterministic derivation"
        )
    if strict:
        allowed = _EXTRACTION_ROOT_KEYS | {
            # Replay compatibility only.  Values are intentionally ignored.
            "modality_relation", "counterfactual", "conclusion",
            "official_policy_verdict", "policy_rule_ids",
        }
        _unexpected_keys(extraction, frozenset(allowed), "record", issues)

    version = extraction.get("schema_version", EXTRACTION_SCHEMA_VERSION)
    if version not in {EXTRACTION_SCHEMA_VERSION, SCHEMA_VERSION}:
        issues.append(
            f"schema_version={version!r}, expected {EXTRACTION_SCHEMA_VERSION!r}"
        )
    product_id = extraction.get("product_id")
    if product_id is not None:
        if isinstance(product_id, bool) or not isinstance(product_id, (int, str)):
            issues.append("product_id must be an integer or a non-empty string")
        elif isinstance(product_id, str) and not product_id.strip():
            issues.append("product_id must be an integer or a non-empty string")
        elif isinstance(product_id, str):
            product_id = product_id.strip()

    category = _enum(extraction.get("category"), CATEGORIES, "category", issues)
    product_form = _enum(
        extraction.get("product_form"),
        PRODUCT_FORMS.get(category, frozenset()),
        "product_form",
        issues,
    )
    elements_value = extraction.get("elements")
    if not isinstance(elements_value, list):
        issues.append("elements must be a list")
        elements_value = []
    if len(elements_value) > 12:
        issues.append("elements must contain at most 12 items")
    known_images = frozenset(known_image_ids) if known_image_ids is not None else None
    elements = [
        _validate_element(
            element,
            index,
            category,
            issues,
            name_text=name_text,
            description_text=description_text,
            ocr_text_by_image=ocr_text_by_image,
            known_image_ids=known_images,
            strict=strict,
        )
        for index, element in enumerate(elements_value)
    ]
    if issues:
        raise SchemaValidationError(issues)

    _record_wide_visual_support_denial_issues(
        elements, product_form, issues
    )
    if issues:
        raise SchemaValidationError(issues)

    modality_relation, conflicts = derive_modality_relation(
        elements, product_form=product_form
    )
    derived: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "product_form": product_form,
        "elements": elements,
        "modality_relation": modality_relation,
        "conflicts": conflicts,
    }
    if product_id is not None:
        derived["product_id"] = product_id
    decision = evaluate_policy(derived)
    derived["counterfactual"] = derive_counterfactual(derived, decision)
    derived["conclusion"] = _derived_conclusion(derived)
    # One final pass validates the structured interventions and freezes policy
    # fields in the persisted, backwards-readable rationale envelope.
    return validate_blind_record(
        derived,
        name_text=name_text,
        description_text=description_text,
        ocr_text_by_image=ocr_text_by_image,
        known_image_ids=known_image_ids,
        strict=True,
    )


def validate_blind_record(
    value: str | Mapping[str, Any],
    *,
    name_text: str | None = None,
    description_text: str | None = None,
    ocr_text_by_image: Mapping[str, str] | None = None,
    known_image_ids: Iterable[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Parse, ground, and normalise a rationale generated without the label.

    ``official_policy_verdict`` and ``policy_rule_ids`` are recomputed from the
    concepts.  If the model supplied them, disagreement is a validation error.
    The returned object never contains a dataset label.
    """

    record = parse_json_object(value)
    issues: list[str] = []

    forbidden = sorted(set(record) & _POST_LABEL_KEYS)
    if forbidden:
        issues.append(
            "blind rationale contains post-label fields: "
            f"{forbidden}; attach them only after validation"
        )
    if strict:
        _unexpected_keys(record, _ROOT_KEYS, "record", issues)

    schema_version = record.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        issues.append(
            f"schema_version={schema_version!r}, expected {SCHEMA_VERSION!r}"
        )

    product_id = record.get("product_id")
    if product_id is not None:
        if isinstance(product_id, bool) or not isinstance(product_id, (int, str)):
            issues.append("product_id must be an integer or a non-empty string")
        elif isinstance(product_id, str) and not product_id.strip():
            issues.append("product_id must be an integer or a non-empty string")
        elif isinstance(product_id, str):
            product_id = product_id.strip()

    category = _enum(record.get("category"), CATEGORIES, "category", issues)
    product_form = _enum(
        record.get("product_form"),
        PRODUCT_FORMS.get(category, frozenset()),
        "product_form",
        issues,
    )
    modality_relation = _enum(
        record.get("modality_relation"),
        MODALITY_RELATIONS,
        "modality_relation",
        issues,
    )

    elements_value = record.get("elements")
    if not isinstance(elements_value, list):
        issues.append("elements must be a list")
        elements_value = []
    if len(elements_value) > 12:
        issues.append("elements must contain at most 12 items")
    known_images = frozenset(known_image_ids) if known_image_ids is not None else None
    elements = [
        _validate_element(
            element,
            index,
            category,
            issues,
            name_text=name_text,
            description_text=description_text,
            ocr_text_by_image=ocr_text_by_image,
            known_image_ids=known_images,
            strict=strict,
        )
        for index, element in enumerate(elements_value)
    ]
    if category == "Легковоспламеняющиеся":
        _record_wide_visual_support_denial_issues(
            elements, product_form, issues
        )

    conflicts_value = record.get("conflicts")
    if not isinstance(conflicts_value, list):
        issues.append("conflicts must be a list")
        conflicts_value = []
    conflicts = [
        _text(item, f"conflicts[{index}]", issues)
        for index, item in enumerate(conflicts_value)
    ]

    expected_relation, expected_conflicts = derive_modality_relation(
        elements, product_form=product_form
    )
    if modality_relation != expected_relation:
        issues.append(
            "modality_relation disagrees with deterministic evidence "
            f"derivation: claimed {modality_relation!r}, computed "
            f"{expected_relation!r}"
        )
    # Conflict prose is a deterministic rendering, not a trusted model field.
    # Canonicalise legacy free-text conflict explanations while still requiring
    # the claimed relation itself to match the evidence-derived relation.
    conflicts = expected_conflicts
    modality_relation = expected_relation

    evidence_sources = {
        evidence["source"]
        for element in elements
        for evidence in element["evidence"]
        if evidence.get("source")
    }
    has_text_evidence = bool(evidence_sources & {"name", "description"})
    has_image_evidence = bool(evidence_sources & {"image", "ocr"})
    if modality_relation in {"agree", "conflict"} and not (
            has_text_evidence and has_image_evidence):
        issues.append(
            f"modality_relation={modality_relation!r} requires both text and image evidence"
        )
    if modality_relation == "text_only" and has_image_evidence:
        issues.append("modality_relation='text_only' cannot contain image/OCR evidence")
    if modality_relation == "image_only" and has_text_evidence:
        issues.append("modality_relation='image_only' cannot contain name/description evidence")
    if modality_relation == "conflict" and not conflicts:
        issues.append("modality_relation='conflict' requires a non-empty conflicts list")
    if modality_relation == "agree" and conflicts:
        issues.append("modality_relation='agree' cannot contain unresolved conflicts")

    counterfactual_value = record.get("counterfactual")
    if not isinstance(counterfactual_value, Mapping):
        issues.append("counterfactual must be an object")
        counterfactual_value = {}
    counterfactual_value = dict(counterfactual_value)
    if strict:
        _unexpected_keys(
            counterfactual_value, _COUNTERFACTUAL_KEYS, "counterfactual", issues
        )
    single_change = _text(
        counterfactual_value.get("single_change"),
        "counterfactual.single_change",
        issues,
    )
    expected_verdict = _enum(
        counterfactual_value.get("expected_verdict"),
        POLICY_VERDICTS,
        "counterfactual.expected_verdict",
        issues,
    )
    counterfactual_schema = counterfactual_value.get("schema_version")
    if counterfactual_schema is not None:
        counterfactual_schema = _text(
            counterfactual_schema, "counterfactual.schema_version", issues
        )
        if counterfactual_schema != COUNTERFACTUAL_SCHEMA_VERSION:
            issues.append(
                "counterfactual.schema_version="
                f"{counterfactual_schema!r}, expected "
                f"{COUNTERFACTUAL_SCHEMA_VERSION!r}"
            )
    interventions_value = counterfactual_value.get("interventions")
    interventions: list[dict[str, Any]] | None = None
    if interventions_value is not None:
        if not isinstance(interventions_value, list):
            issues.append("counterfactual.interventions must be a list")
        else:
            interventions = copy.deepcopy(interventions_value)
    resulting_rules_value = counterfactual_value.get("resulting_policy_rule_ids")
    resulting_rules: list[str] | None = None
    if resulting_rules_value is not None:
        if not isinstance(resulting_rules_value, list) or any(
            not isinstance(item, str) for item in resulting_rules_value
        ):
            issues.append(
                "counterfactual.resulting_policy_rule_ids must be a list of strings"
            )
        else:
            resulting_rules = list(resulting_rules_value)
    conclusion = _text(record.get("conclusion"), "conclusion", issues)

    normalised = {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "product_form": product_form,
        "elements": elements,
        "modality_relation": modality_relation,
        "conflicts": conflicts,
        "counterfactual": {
            "single_change": single_change,
            "expected_verdict": expected_verdict,
        },
        "conclusion": conclusion,
    }
    if counterfactual_schema is not None:
        normalised["counterfactual"]["schema_version"] = counterfactual_schema
    if interventions is not None:
        normalised["counterfactual"]["interventions"] = interventions
    if resulting_rules is not None:
        normalised["counterfactual"]["resulting_policy_rule_ids"] = resulting_rules
    if product_id is not None:
        normalised["product_id"] = product_id

    if not issues:
        decision = evaluate_policy(normalised)
        if interventions is not None:
            try:
                changed = apply_counterfactual_interventions(normalised, interventions)
                counterfactual_decision = evaluate_policy(changed)
            except SchemaValidationError as exc:
                issues.extend(exc.issues)
                counterfactual_decision = None
            if (counterfactual_decision is not None and
                    expected_verdict != counterfactual_decision.verdict):
                issues.append(
                    "counterfactual.expected_verdict disagrees with deterministic "
                    f"intervention execution: claimed {expected_verdict!r}, computed "
                    f"{counterfactual_decision.verdict!r}"
                )
            if (counterfactual_decision is not None and resulting_rules is not None and
                    set(resulting_rules) != set(counterfactual_decision.rule_ids)):
                issues.append(
                    "counterfactual.resulting_policy_rule_ids disagree with "
                    "deterministic intervention execution"
                )
            if (counterfactual_decision is not None and
                    decision.verdict in {"yes", "no"} and
                    counterfactual_decision.verdict == decision.verdict):
                issues.append(
                    "counterfactual.interventions must flip a decisive policy verdict"
                )
        elif decision.verdict in {"yes", "no"}:
            # Backward compatibility for the audited V1 free-text contrast.
            opposite = "no" if decision.verdict == "yes" else "yes"
            if expected_verdict != opposite:
                issues.append(
                    "counterfactual.expected_verdict must flip a decisive policy "
                    f"verdict from {decision.verdict!r} to {opposite!r}"
                )
        claimed_verdict = record.get("official_policy_verdict")
        if claimed_verdict is not None and claimed_verdict != decision.verdict:
            issues.append(
                "official_policy_verdict disagrees with the deterministic policy "
                f"executor: claimed {claimed_verdict!r}, computed {decision.verdict!r}"
            )
        claimed_rules = record.get("policy_rule_ids")
        if claimed_rules is not None:
            if not isinstance(claimed_rules, list) or any(
                not isinstance(item, str) for item in claimed_rules
            ):
                issues.append("policy_rule_ids must be a list of strings")
            elif set(claimed_rules) != set(decision.rule_ids):
                issues.append(
                    "policy_rule_ids disagree with the deterministic policy executor: "
                    f"claimed {claimed_rules!r}, computed {list(decision.rule_ids)!r}"
                )
        normalised["official_policy_verdict"] = decision.verdict
        normalised["policy_rule_ids"] = list(decision.rule_ids)

    if issues:
        raise SchemaValidationError(issues)
    return normalised


def attach_observed_label(
    blind_record: str | Mapping[str, Any],
    label: int | bool,
    **validation_context: Any,
) -> dict[str, Any]:
    """Attach a dataset label after blind evidence and report its agreement.

    The stable hash covers the complete validated blind record.  It can be kept
    in JSONL to prove that later gold reconciliation did not alter the evidence.
    """

    if isinstance(label, bool):
        observed_label = int(label)
    elif isinstance(label, int) and label in {0, 1}:
        observed_label = label
    else:
        raise SchemaValidationError(["label must be 0 or 1"])

    record = validate_blind_record(blind_record, **validation_context)
    blind_serialised = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    verdict = record["official_policy_verdict"]
    if verdict == "review":
        label_support = "insufficient"
    elif (verdict == "yes") == bool(observed_label):
        label_support = "supported"
    else:
        label_support = "conflict"

    result = copy.deepcopy(record)
    result["blind_record_sha256"] = hashlib.sha256(blind_serialised).hexdigest()
    result["observed_label"] = observed_label
    result["label_support"] = label_support
    return result


def compact_target(record: Mapping[str, Any]) -> str:
    """Render structured evidence in Claude's compact Russian slot style."""

    # JSONL output also carries audit/training metadata (`id`, `fold`, targets,
    # model path). Formatting its grounded core should work without asking every
    # caller to know and strip those transport fields first.
    blind = {key: value for key, value in record.items() if key in _ROOT_KEYS}
    validated = validate_blind_record(blind)

    form_ru = {
        "standalone_product": "самостоятельный товар",
        "kit": "комплект из нескольких предметов",
        "appliance": "устройство",
        "container": "ёмкость",
        "fuel": "топливо",
        "pyrotechnic": "пиротехническое изделие",
        "component": "компонент другого изделия",
        "supplement": "добавка",
        "sports_nutrition": "спортивное питание",
        "food": "пищевой продукт",
        "medicine": "лекарственное средство",
        "accessory": "аксессуар",
        "unknown": "тип не установлен",
    }
    role_ru = {
        "main_product": "сам товар",
        "separate_in_kit": "вложен в комплект отдельно",
        "integrated": "встроен в изделие",
        "content_of": "содержимое товара",
        "component_of": "компонент другого изделия",
        "compatible_only": "только совместим; вхождение в комплект не доказано",
        "mentioned_only": "только упомянут в тексте",
        "pictured_use_only": "только показан в сценарии использования",
        "unknown": "роль не установлена",
    }
    state_ru = {
        "present": "присутствует",
        "explicitly_absent": "явно отсутствует",
        "empty": "содержимое отсутствует",
        "unknown": "не установлено (это не означает отсутствие)",
    }
    verdict_ru = {
        "yes": "ОТНОСИТСЯ",
        "no": "не относится",
        "review": "требует проверки",
    }

    elements = validated["elements"]
    element_lines = [
        f"ЭЛЕМЕНТ: [{index}] {element['name']} | ВИД: {element['kind']} | "
        f"РОЛЬ: {role_ru[element['role']]} ({element['role']}) | "
        f"СОСТОЯНИЕ: {state_ru[element['state']]} ({element['state']}) | "
        f"МЕХАНИЗМ: {element['mechanism']}"
        for index, element in enumerate(elements, 1)
    ] or [
        "ЭЛЕМЕНТ: [0] не установлен | ВИД: unknown | РОЛЬ: роль не установлена "
        "(unknown) | СОСТОЯНИЕ: не установлено (unknown) | МЕХАНИЗМ: unknown"
    ]

    evidence_parts: list[str] = []
    for element in elements:
        for evidence in element["evidence"]:
            if evidence["source"] in {"name", "description", "ocr"}:
                ref = evidence.get("image_id")
                prefix = f"{evidence['source']}[{ref}]" if ref else evidence["source"]
                evidence_parts.append(f"{prefix}: «{evidence.get('exact_span', '')}»")
            else:
                evidence_parts.append(
                    f"image[{evidence.get('image_id', '?')}]: "
                    f"{evidence.get('observation', '')}"
                )
    evidence_text = "; ".join(evidence_parts) or "решающее доказательство не получено"
    relation_ru = {
        "agree": "текст и изображения согласуются",
        "text_only": "решающий признак подтверждён только текстом",
        "image_only": "решающий признак подтверждён только изображением",
        "conflict": "текст и изображения противоречат друг другу",
        "insufficient": "доказательств недостаточно",
    }[validated["modality_relation"]]

    return "\n".join(
        [
            f"ТИП: {form_ru[validated['product_form']]}",
            *element_lines,
            f"ДОКАЗАТЕЛЬСТВА: {evidence_text}",
            f"СВЕРКА: {relation_ru}",
            f"ПРАВИЛО: {', '.join(validated['policy_rule_ids'])}",
            "КОНТРАСТ: "
            f"{validated['counterfactual']['single_change']} -> "
            f"{verdict_ru[validated['counterfactual']['expected_verdict']]}",
            f"ВЫВОД: {verdict_ru[validated['official_policy_verdict']]}",
        ]
    )


def distillation_target(record: Mapping[str, Any]) -> str:
    """Return the short label-first auxiliary target used by the text student.

    Exact evidence remains stored in the full record and visible in the audit
    rendering.  Repeating long spans in the completion wastes context and can
    truncate the decision concepts; the student needs the small set of key
    concept tokens after the already supplied verdict.
    """

    blind = {key: value for key, value in record.items() if key in _ROOT_KEYS}
    validated = validate_blind_record(blind)
    decision_direction = {"yes": 1, "no": -1, "review": 0}[
        validated["official_policy_verdict"]
    ]
    decisive_indices = {
        index
        for index, element in enumerate(validated["elements"])
        if element_policy_direction(validated["category"], element)
        == decision_direction
    }

    # A negative decision can depend on a *blocked positive candidate* which is
    # deliberately neutral when evaluated by itself.  For example, an external
    # gas cylinder with role=compatible_only/state=unknown must not fire a
    # negative rule (its absence was not proved), yet this boundary is exactly
    # what distinguishes a non-flammable stove from a kit containing a filled
    # cylinder.  V9 counterfactual interventions are machine-applied and
    # validated to flip the policy verdict, so their element indices are a
    # compact, deterministic relevance signal for the student target.
    counterfactual_indices: set[int] = set()
    counterfactual = validated.get("counterfactual", {})
    if (
        counterfactual.get("schema_version") == COUNTERFACTUAL_SCHEMA_VERSION
        and isinstance(counterfactual.get("interventions"), list)
    ):
        counterfactual_indices = {
            int(intervention["element_index"])
            for intervention in counterfactual["interventions"]
        }

    selected_indices = decisive_indices | counterfactual_indices
    if not selected_indices:
        # Preserve legacy/review behaviour when there is neither a decisive
        # element nor a structured, executable counterfactual.
        selected_indices = set(range(len(validated["elements"])))
    decisive = [
        element
        for index, element in enumerate(validated["elements"])
        if index in selected_indices
    ]

    lines = [f"ТИП: {validated['product_form']}"]
    for index, element in enumerate(decisive, 1):
        lines.append(
            f"ЭЛЕМЕНТ: [{index}] {element['name']} | ВИД: {element['kind']} | "
            f"РОЛЬ: {element['role']} | СОСТОЯНИЕ: {element['state']} | "
            f"МЕХАНИЗМ: {element['mechanism']}"
        )
    lines.append(f"ПРАВИЛО: {', '.join(validated['policy_rule_ids'])}")
    # Evidence and counterfactual prose stay in the human-audit rendering.  The
    # validated structured delta selects relevant original facts above; its
    # synthetic replacement values are intentionally not copied into training.
    return "\n".join(lines)


def schema_contract(category: str) -> dict[str, Any]:
    """Return a compact machine-readable contract for a generation prompt.

    The metadata deliberately uses the *actual* JSON property names.  Earlier
    versions documented semantic slots as ``element_kind`` and
    ``evidence_polarity`` even though strict validation expects ``kind`` and
    ``polarity``.  A generative model reasonably copied those aliases and then
    failed the very schema the prompt was meant to explain.

    The two-item limits are generation-prompt limits, not the general storage
    schema limits enforced by :func:`validate_blind_record`.  Human-authored
    audit records may remain richer; teacher generations should be the minimal
    policy-relevant contrast needed for distillation.
    """

    if category not in CATEGORIES:
        raise SchemaValidationError([f"unknown category: {category!r}"])
    mechanisms = sorted({
        mechanism
        for kind in ELEMENT_KINDS[category]
        for mechanism in _MECHANISMS_BY_KIND[kind]
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "required_root_keys": [
            "schema_version", "product_id", "category", "product_form",
            "elements", "modality_relation", "conflicts",
            "counterfactual", "conclusion",
        ],
        "element_keys": [
            "name", "kind", "role", "state", "mechanism", "evidence",
        ],
        "evidence_keys": [
            "source", "polarity", "exact_span", "image_id", "bbox",
            "observation",
        ],
        "counterfactual_keys": ["single_change", "expected_verdict"],
        "limits": {
            "elements_max": 2,
            "evidence_per_element_max": 2,
            "conflicts_max": 2,
        },
        "enums": {
            "product_form": sorted(PRODUCT_FORMS[category]),
            "kind": sorted(ELEMENT_KINDS[category]),
            "role": sorted(ROLES),
            "state": sorted(STATES),
            "mechanism": mechanisms,
            "source": sorted(EVIDENCE_SOURCES),
            "polarity": sorted(EVIDENCE_POLARITIES),
            "modality_relation": sorted(MODALITY_RELATIONS),
            "expected_verdict": sorted(POLICY_VERDICTS),
        },
        "compatible_mechanisms_by_kind": {
            kind: sorted(_MECHANISMS_BY_KIND[kind])
            for kind in sorted(ELEMENT_KINDS[category])
        },
        "absence_guard": (
            "Если признак не виден, не читается или не наблюдается, используйте "
            "state=unknown, а не explicitly_absent/empty: «не наблюдается» не "
            "означает отсутствие."
        ),
    }


def extraction_schema_contract(category: str) -> dict[str, Any]:
    """Return the V9 teacher contract: grounded observations and concepts only."""

    if category not in CATEGORIES:
        raise SchemaValidationError([f"unknown category: {category!r}"])
    mechanisms = sorted({
        mechanism
        for kind in ELEMENT_KINDS[category]
        for mechanism in _MECHANISMS_BY_KIND[kind]
    })
    return {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "category": category,
        "required_root_keys": [
            "schema_version", "product_id", "category", "product_form", "elements",
        ],
        "optional_root_keys": ["conflicts"],
        "forbidden_derived_keys": [
            "modality_relation", "counterfactual", "conclusion",
            "official_policy_verdict", "policy_rule_ids", "observed_label",
            "label_support",
        ],
        "element_keys": [
            "name", "kind", "role", "state", "mechanism", "evidence",
        ],
        "evidence_keys": [
            "source", "polarity", "exact_span", "image_id", "bbox", "observation",
        ],
        "generation_limits": {
            "elements_max": 3,
            "evidence_per_element_max": 2,
        },
        "enums": {
            "product_form": sorted(PRODUCT_FORMS[category]),
            "kind": sorted(ELEMENT_KINDS[category]),
            "role": sorted(ROLES),
            "state": sorted(STATES),
            "mechanism": mechanisms,
            "source": sorted(EVIDENCE_SOURCES),
            "polarity": sorted(EVIDENCE_POLARITIES),
        },
        "compatible_mechanisms_by_kind": {
            kind: sorted(_MECHANISMS_BY_KIND[kind])
            for kind in sorted(ELEMENT_KINDS[category])
        },
        "semantics": {
            "kind": "type of the risk element/substance, never the container shell",
            "role.compatible_only": (
                "external/compatible item; presence in the sold kit is not established"
            ),
            "state.empty": (
                "the named risk substance/content is explicitly absent, not present"
            ),
            "conflict": (
                "only explicit support/refute evidence; not-visible is not refutation"
            ),
        },
    }


def build_empty_record(product_id: int | str, category: str) -> dict[str, Any]:
    """Build a valid ``review`` record that a generator can fill in place."""

    if category not in CATEGORIES:
        raise SchemaValidationError([f"unknown category: {category!r}"])
    record = {
        "schema_version": SCHEMA_VERSION,
        "product_id": product_id,
        "category": category,
        "product_form": "unknown",
        "elements": [],
        "modality_relation": "insufficient",
        "conflicts": [],
        "counterfactual": {
            "single_change": "получить решающее читаемое доказательство о целевом товаре",
            "expected_verdict": "review",
        },
        "conclusion": "Решающее доказательство пока не получено.",
    }
    return validate_blind_record(record)


def extract_json(text: str) -> dict[str, Any]:
    """Compatibility name for JSON extraction used by generation scripts."""

    return extract_json_object(text)


def validate_record(
    record: str | Mapping[str, Any],
    source_record: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return validation issues without raising, convenient for JSONL audits.

    ``source_record`` may contain ``id``, ``category``, ``name``,
    ``description``, ``images`` and ``ocr_text_by_image``.  When supplied, exact
    spans and image references are grounded against that original card.
    """

    try:
        parsed = parse_json_object(record)
    except SchemaValidationError as exc:
        return list(exc.issues)

    issues: list[str] = []
    context: dict[str, Any] = {}
    if source_record is not None:
        context = {
            "name_text": source_record.get("name"),
            "description_text": source_record.get("description"),
            "ocr_text_by_image": source_record.get("ocr_text_by_image"),
            "known_image_ids": source_record.get("images"),
        }
        if source_record.get("category") is not None and parsed.get("category") != source_record.get(
            "category"
        ):
            issues.append("category does not match the source record")
        source_id = source_record.get("id", source_record.get("product_id"))
        if source_id is not None and parsed.get("product_id") != source_id:
            issues.append("product_id does not match the source record")

    try:
        validate_blind_record(parsed, **context)
    except SchemaValidationError as exc:
        issues.extend(exc.issues)
    return issues


def derive_policy(record: str | Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return ``(yes|no|review, rule_ids)`` from grounded blind concepts."""

    parsed = parse_json_object(record)
    blind = {key: value for key, value in parsed.items() if key in _ROOT_KEYS}
    validated = validate_blind_record(blind)
    return validated["official_policy_verdict"], list(validated["policy_rule_ids"])


def annotate_gold(
    record: str | Mapping[str, Any], gold_label: int | bool
) -> dict[str, Any]:
    """Compatibility name for the only allowed post-label operation."""

    return attach_observed_label(record, gold_label)


__all__ = [
    "CATEGORIES",
    "COUNTERFACTUAL_SCHEMA_VERSION",
    "ELEMENT_KINDS",
    "EXTRACTION_SCHEMA_VERSION",
    "EVIDENCE_POLARITIES",
    "EVIDENCE_SOURCES",
    "LABEL_SUPPORT_VALUES",
    "MECHANISMS",
    "MODALITY_RELATIONS",
    "POLICY_VERDICTS",
    "PRODUCT_FORMS",
    "ROLES",
    "RISK_SUBSTANCE_KINDS",
    "SCHEMA_VERSION",
    "STATES",
    "PolicyDecision",
    "SchemaValidationError",
    "annotate_gold",
    "attach_observed_label",
    "apply_counterfactual_interventions",
    "build_empty_record",
    "compact_target",
    "derive_policy",
    "derive_counterfactual",
    "derive_modality_relation",
    "distillation_target",
    "element_policy_direction",
    "evaluate_policy",
    "extract_json",
    "extract_json_object",
    "extract_json_object_with_repairs",
    "extraction_schema_contract",
    "finalize_extraction_record",
    "has_targeted_risk_absence",
    "has_targeted_risk_inclusion",
    "image_refute_affirms_risk_target",
    "image_support_denies_risk_target",
    "is_strong_visual_identity_denial",
    "parse_json_object",
    "risk_entity_key",
    "risk_entity_denial_matches",
    "risk_entity_keys",
    "schema_contract",
    "validate_blind_record",
    "validate_record",
    "visual_denial_scopes_complete_kit",
    "visual_risk_denial_entity_keys",
]
