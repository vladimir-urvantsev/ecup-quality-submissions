"""Regex/keyword baseline: encodes the rules from the task description directly,
no models, no embeddings. Goal: see how far pure text rules get, and dump
misclassified rows for error analysis (especially in the two minority buckets)."""

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, classification_report

HERE = Path(__file__).parent
DATA_CSV = HERE / "data" / "data.csv"
OUT_DIR = HERE / "analysis"
OUT_DIR.mkdir(exist_ok=True)


def _norm(s) -> str:
    return str(s).lower().replace("ё", "е")


# ---------- БАД rules ----------
_BAD_POSITIVE = re.compile(
    r"\bбад\b|биологически\s*активн|dietary\s*supplement|б\.?а\.?д\.?", re.I
)
_BAD_SPORT_NUTRITION = re.compile(
    r"спортивн\w*\s*питани|\bbcaa\b|аминокислот|л-?карнитин|l-?carnitine|"
    r"креатин|\bprotein\b|протеин|гейнер|\bwhey\b", re.I
)
_BAD_EXPLICIT_NEGATIVE = re.compile(
    r"не\s+являет\w*\s+бад|не\s+бад\b|не\s+явля\w*\s+биологически", re.I
)


def predict_bad(text: str) -> int:
    t = _norm(text)
    if _BAD_EXPLICIT_NEGATIVE.search(t):
        return 0
    if _BAD_SPORT_NUTRITION.search(t):
        return 0
    if _BAD_POSITIVE.search(t):
        return 1
    return 0  # no БАД marking at all -> not a supplement per the rules


# ---------- Легковоспламеняющиеся rules ----------
# label=1 means the item GENUINELY satisfies the flammable-category rule (not "banned" —
# "не бан" verdict is derived from label=1 downstream, but the rule check itself is a
# straight yes/no membership test).
_FLAME_SELF_SOURCE = re.compile(
    r"спичк|зажигалк|жидкост\w*\s*для\s*розжиг|гель\s*для\s*розжиг|"
    r"таблетк\w*\s*для\s*розжиг", re.I
)
_FLAME_SUBSTANCE = re.compile(
    r"легковоспламен|огнеопасн|горюч\w*\s*(веществ|газ)|"
    r"ацетон|бензин|керосин|изопропиловый спирт|спирт\b(?!ов)|"
    r"бутан|пропан|эфир\b|растворитель|нитроцеллюлоз", re.I
)
_FLAME_BUILTIN_IGNITION = re.compile(r"пьезоподж|встроенн\w*\s*(поджиг|воспламенител)", re.I)
_FLAME_COMPONENT_EXCEPTION = re.compile(
    r"активированн\w*\s*угол|уголь\s*для\s*рисован|угольный\s*фильтр|"
    r"древесн\w*|парафинов\w*\s*(ролл|роллы)|топливн\w*\s*брикет|"
    r"пеллет|евродров|для\s*камин\w*\s*и\s*бань", re.I
)
_FLAME_EMPTY_CONTAINER = re.compile(r"пуст\w*\s*(газов|баллон)|без\s*содержим|без\s*спичек|без\s*газа", re.I)
_FLAME_CONTENT_INCLUDED = re.compile(
    r"в\s*комплект\w*\s*с\s*угл|уголь:?\s*1\s*шт|пакет\s*с\s*угл|с\s*углём|с\s*углем", re.I
)
_FLAME_DEVICE_NOT_CONTENT = re.compile(
    r"мангал|гриль|барбекю|газов\w*\s*плит|газов\w*\s*горелк|печь|жаровня|"
    r"курительниц|горелк\w*\s*кухон", re.I
)


def predict_flammable(text: str) -> int:
    t = _norm(text)

    if _FLAME_EMPTY_CONTAINER.search(t):
        return 0  # explicitly empty / no content -> not flammable regardless of other markers

    if _FLAME_COMPONENT_EXCEPTION.search(t):
        return 0  # solid wood-based fuel / component-only material -> exception, not flammable

    if _FLAME_SELF_SOURCE.search(t):
        return 1  # self-contained ignition source -> genuinely flammable

    if _FLAME_SUBSTANCE.search(t):
        return 1  # explicitly contains flammable substance/gas -> genuinely flammable

    if _FLAME_BUILTIN_IGNITION.search(t):
        return 0  # ignition source built into the device -> exception, not flammable

    if _FLAME_DEVICE_NOT_CONTENT.search(t):
        if _FLAME_CONTENT_INCLUDED.search(t):
            return 1  # device explicitly bundled WITH flammable content (e.g. charcoal included)
        return 0  # device itself (mangal/grill/stove), no content mentioned -> not flammable

    return 0  # default: no flammable marker found at all -> not genuinely flammable


def predict(row) -> int:
    if row["category"] == "БАД":
        return predict_bad(row["text_full"])
    return predict_flammable(row["text_full"])


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    df["text_full"] = (
        df["name"].astype(str) + " \n " + df["description"].astype(str)
    )

    df["pred"] = df.apply(predict, axis=1)

    print("=== Overall macro F1 (as competition defines it) ===")
    per_cat_f1 = {}
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        f1 = f1_score(sub["label"], sub["pred"])
        per_cat_f1[cat] = f1
        print(f"{cat}: F1 = {f1:.4f}  (n={len(sub)})")
    macro_f1 = sum(per_cat_f1.values()) / len(per_cat_f1)
    print(f"\nMACRO F1 = {macro_f1:.4f}")

    print("\n=== Classification report per category ===")
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        print(f"\n--- {cat} ---")
        print(classification_report(sub["label"], sub["pred"], digits=3))

    # Misclassified rows, with focus on the two minority buckets
    mis = df[df["label"] != df["pred"]].copy()
    mis.to_csv(OUT_DIR / "misclassified_all.csv", index=False)
    print(f"\nTotal misclassified: {len(mis)} / {len(df)} -> saved to analysis/misclassified_all.csv")

    flam_1 = df[(df["category"] == "Легковоспламеняющиеся") & (df["label"] == 1)]
    flam_1_mis = flam_1[flam_1["pred"] != flam_1["label"]]
    flam_1_mis.to_csv(OUT_DIR / "misclassified_flammable_label1.csv", index=False)
    print(
        f"Легковоспламеняющиеся label=1: {len(flam_1)} total, "
        f"{len(flam_1_mis)} misclassified by rules -> analysis/misclassified_flammable_label1.csv"
    )

    bad_0 = df[(df["category"] == "БАД") & (df["label"] == 0)]
    bad_0_mis = bad_0[bad_0["pred"] != bad_0["label"]]
    bad_0_mis.to_csv(OUT_DIR / "misclassified_bad_label0.csv", index=False)
    print(
        f"БАД label=0 (не-БАД): {len(bad_0)} total, "
        f"{len(bad_0_mis)} misclassified by rules -> analysis/misclassified_bad_label0.csv"
    )


if __name__ == "__main__":
    main()
