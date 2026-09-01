"""One rule, and only one.

Contract fixed by run.py's caller: suppressed_by(name, description) -> str|None.
The caller has already filtered to Легковоспламеняющиеся rows whose verdict is
currently positive, so this function neither sees nor needs the category.

An earlier version of this file suppressed "вечная зажигалка / огниво / кресало"
together and cost 0.011 on the leaderboard: огниво and кресало are positive in a
quarter of cases, so it killed 10 true positives out of 13 firings. The narrow
form below matches only the eternal MATCH, in either word order. On the pooled
out-of-fold set it fires 5 times with zero positives among them, correcting 4
false positives: two "Спичка вечная" products and two survival kits listing
"Вечные спички" in their contents.

An eternal match is a ferrocerium rod — it makes a spark and holds no fuel. The
model has learned the bigram "вечная спичка" but not the inverted "Спичка
вечная", which is why exactly those cards leak through.
"""

import re

ETERNAL_MATCH = re.compile(r"вечн\w*\s+спичк|спичк\w*\s+вечн", re.IGNORECASE)


def suppressed_by(name, description=""):
    """Return a rule name when the verdict must be forced negative, else None."""
    if ETERNAL_MATCH.search(f"{name or ''} {description or ''}"):
        return "вечная спичка: искра без топлива"
    return None
