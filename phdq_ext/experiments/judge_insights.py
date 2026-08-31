"""What the judged scores say about the bands, treated as findings rather than
as a validation exercise.

Three questions, each answerable only because texts are scored one by one.

1. How many independent things does the judge actually see? Ten prompts were
   written; if they collapse onto two factors then eight of them are one
   measurement in different words.

2. Can a text keep its dimension intact while a reader sees damage, and can it
   keep its quality while the dimension moves? Both directions matter: the
   first says the geometry is blind to something a reader notices, the second
   says it detects something invisible.

3. Which operations are quiet -- large geometric displacement at human-level
   scores. Those are the ones a detector would catch and a proofreader would
   not, which is the interesting class.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS
from three_bands import BANDS as _B, SUF

BANDS = list(_B)

BASE = os.path.join(HERE, "..")
# the scales on which "human, unremarkable" is 7 and below 5 reads as noise
WELL_FORMED = ["coherence", "literacy", "naturalness", "integration"]


def main():
    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    J = [k for k in PROPS if k in D]
    pd.set_option("display.width", 220)

    print(f"=== 1. что судья различает ({len(J)} шкал)")
    C = D[J].corr(method="spearman")
    C.index = C.columns = [PROPS[k]["ru"] for k in J]
    print(C.round(2).to_string())
    ev = np.linalg.eigvalsh(D[J].corr().fillna(0))[::-1]
    print("\nдоли дисперсии между шкалами:", np.round(ev / ev.sum(), 2)[:5])

    print("\n\n=== 2. тексты, которые судья считает нормальными")
    ok = D[(D[WELL_FORMED] >= 7).all(axis=1)]
    bad = D[(D[WELL_FORMED] <= 4).any(axis=1)]
    print(f"все четыре шкалы >= 7: {len(ok)} текстов из {len(D)}")
    print(f"хотя бы одна <= 4:     {len(bad)} текстов")
    rows = []
    for band in BANDS:
        rows.append({
            "полоса": band,
            "норм: медиана": ok[band].median(),
            "норм: |сдвиг| медиана": ok[band].abs().median(),
            "норм: 90-й проц.": ok[band].abs().quantile(0.9),
            "битые: |сдвиг| медиана": bad[band].abs().median(),
        })
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    print("\n  самые сильные сдвиги среди текстов, которые судья не забраковал:")
    for band in BANDS:
        t = ok.reindex(ok[band].abs().sort_values(ascending=False).index)
        top = t.head(4)[["perturbation", band] + WELL_FORMED]
        print(f"\n  -- {band}")
        print(top.round(1).to_string(index=False))

    print("\n\n=== 3. тихие операции: большой сдвиг при человеческих оценках")
    g = D.groupby("perturbation")
    P = pd.DataFrame({
        "n": g.size(),
        "оценки": g[WELL_FORMED].mean().mean(axis=1),
        **{b: g[b].mean() for b in BANDS},
    })
    P["макс |сдвиг|"] = P[list(BANDS)].abs().max(axis=1)
    quiet = P[(P["оценки"] >= 6.5) & (P["макс |сдвиг|"] >= 10)]
    loud = P[(P["оценки"] < 5) & (P["макс |сдвиг|"] < 10)]
    print("\n  тихие — читаются как человеческий текст, но геометрия ушла:")
    print(quiet.sort_values("макс |сдвиг|", ascending=False).round(1).to_string())
    print("\n  шумные — судья видит поломку, а геометрия почти на месте:")
    print(loud.sort_values("оценки").round(1).to_string())

    bimodal(D, J)
    P.round(2).to_csv(os.path.join(BASE, "results", f"judge_by_pert{SUF}.csv"))


def bimodal(D, J):
    """The scales split texts into 'reads like a person' and 'broken'.

    A correlation computed across both halves is largely a between-halves
    effect; what matters is whether the scale still says anything inside the
    human half, where every text is well-formed and the differences are the
    ones the property was written to capture.
    """
    print("\n\n=== 4. двугорбость шкал")
    rows = []
    for k in J:
        v = D[k].dropna()
        lo = (v <= 4).mean()
        hi = (v >= 7).mean()
        mid = ((v > 4) & (v < 7)).mean()
        rows.append({"шкала": PROPS[k]["ru"], "n": len(v),
                     "<=4 сломано": lo, "5-6 серая зона": mid,
                     ">=7 по-человечески": hi, "медиана": v.median()})
    B = pd.DataFrame(rows).sort_values("<=4 сломано", ascending=False)
    print(B.round(2).to_string(index=False))

    print("\n  корреляция с полосами: все тексты / только человеческая половина")
    human = D[(D[WELL_FORMED] >= 6).all(axis=1)]
    print(f"  человеческая половина: {len(human)} текстов")
    out = []
    for k in J:
        r = {"шкала": PROPS[k]["ru"]}
        for band in BANDS:
            a = D[[k, band]].dropna()
            h = human[[k, band]].dropna()
            r[f"{band} все"] = a[k].corr(a[band], method="spearman")
            r[f"{band} человеч."] = (h[k].corr(h[band], method="spearman")
                                     if h[k].nunique() > 2 else np.nan)
        out.append(r)
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
