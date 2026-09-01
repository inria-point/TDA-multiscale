"""Do the flagged texts actually read as damaged?

Two readings of the same table. The judged scores say whether a reader finds
anything wrong, group by group. The filter view turns that into the only
question a dataset owner asks: if I drop what the geometry flags, what fraction
of what I drop was worth dropping, and how much good text goes with it.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS
from three_bands import BANDS as _B

B = list(_B)
BASE = os.path.join(HERE, "..")
CUT = 5
WELL = ["coherence", "literacy", "naturalness", "integration"]
LO, HI = "#c53030", "#2b6cb0"


def main():
    D = pd.read_csv(os.path.join(BASE, "results", "outlier_judged.csv"))
    J = [k for k in PROPS if k in D]
    D["худшая оценка"] = D[WELL].min(axis=1)
    D["средняя оценка"] = D[WELL].mean(axis=1)
    D["подозрительный"] = D["худшая оценка"] < CUT
    base = D[D["профиль"] == "000"]

    pd.set_option("display.width", 220)
    R = D.groupby("профиль").agg(
        n=("id", "size"), сила=("сила", "mean"),
        **{PROPS[k]["ru"][:14]: (k, "mean") for k in J})
    R["худшая"] = D.groupby("профиль")["худшая оценка"].mean()
    R["% подозр."] = D.groupby("профиль")["подозрительный"].mean() * 100
    R = R.sort_values("худшая")
    print("средние судейские оценки по профилям "
          f"(000 = обычный текст, n={len(base)})\n")
    print(R.round(1).to_string())

    print("\n\nотличие от обычных текстов (профиль 000), критерий Манна-Уитни")
    rows = []
    for p, g in D.groupby("профиль"):
        if p == "000":
            continue
        r = {"профиль": p, "n": len(g), "сила": g["сила"].mean()}
        for k in J + ["худшая оценка"]:
            a, b = g[k].dropna(), base[k].dropna()
            if len(a) > 5 and len(b) > 5:
                r[PROPS[k]["ru"][:14] if k in J else "худшая"] = (
                    a.mean() - b.mean())
                if k == "худшая оценка":
                    r["p"] = st.mannwhitneyu(a, b).pvalue
        rows.append(r)
    T = pd.DataFrame(rows).sort_values("худшая")
    print(T.round(2).to_string(index=False))

    print("\n\nпрототип фильтра: выбросить всё за порогом по данному критерию")
    print("точность — какая доля выброшенного действительно подозрительна;")
    print("полнота — какая доля подозрительных поймана\n")
    flags = {"любая полоса": D["сила"] > 0,
             "две и более": (D[[f"h_{b}" for b in B]] != 0).sum(1) >= 2,
             "все три": (D[[f"h_{b}" for b in B]] != 0).all(1),
             "сила > 15": D["сила"] > 15, "сила > 25": D["сила"] > 25}
    for b in B:
        flags[f"только {b}"] = D[f"h_{b}"] != 0
        flags[f"{b} вниз"] = D[f"h_{b}"] < 0
    out = []
    p0 = D["подозрительный"].mean()
    for name, f in flags.items():
        if f.sum() == 0:
            continue
        tp = (f & D["подозрительный"]).sum()
        out.append({"правило": name, "выброшено": f.sum(),
                    "точность": tp / f.sum(),
                    "полнота": tp / D["подозрительный"].sum(),
                    "подъём": (tp / f.sum()) / p0})
    F = pd.DataFrame(out).sort_values("точность", ascending=False)
    print(F.round(2).to_string(index=False))
    print(f"\nбазовая доля подозрительных в выборке: {p0:.1%}")
    F.round(3).to_csv(os.path.join(BASE, "results", "outlier_filter.csv"),
                      index=False)
    figure(D, R)


def figure(D, R):
    order = R.index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2))
    ax = axes[0]
    for i, p in enumerate(order):
        g = D[D["профиль"] == p]["худшая оценка"]
        c = LO if g.mean() < base_mean(D) else HI
        ax.scatter(g + np.random.default_rng(i).normal(0, .07, len(g)),
                   [i] * len(g), s=16, color=c, alpha=.5, linewidths=0)
        ax.plot([g.median()] * 2, [i - .3, i + .3], color=c, lw=2.6)
    ax.axvline(CUT, c="#555", ls="--", lw=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontfamily="monospace")
    ax.set_xlabel("худшая из четырёх оценок связности/грамотности/"
                  "естественности/уместности")
    ax.set_title("Оценка судьи по профилю отклонения\n"
                 "штрих — порог 5, ниже которого текст ближе к шуму",
                 fontsize=10)
    ax.grid(axis="x", alpha=.25)

    ax = axes[1]
    ax.scatter(D["сила"], D["худшая оценка"], s=18,
               c=[LO if s else HI for s in D["подозрительный"]], alpha=.6,
               linewidths=0)
    r = st.spearmanr(D["сила"], D["худшая оценка"])
    ax.axhline(CUT, c="#555", ls="--", lw=1.2)
    ax.set_xlabel("сила отклонения (сумма превышений над 10%)")
    ax.set_ylabel("худшая оценка")
    ax.set_title(f"Чем дальше от жанра, тем хуже текст?\n"
                 f"ρ = {r.statistic:+.2f}, p = {r.pvalue:.3f}", fontsize=10)
    ax.grid(alpha=.25)

    ax = axes[2]
    share = D.groupby("профиль")["подозрительный"].mean().reindex(order) * 100
    c = [LO if v > D["подозрительный"].mean() * 100 else HI for v in share]
    ax.barh(range(len(order)), share, color=c)
    ax.axvline(D["подозрительный"].mean() * 100, c="#555", ls="--", lw=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontfamily="monospace")
    ax.set_xlabel("% текстов с хотя бы одной оценкой ниже 5")
    ax.set_title("Доля подозрительных\nштрих — средний уровень по выборке",
                 fontsize=10)
    ax.grid(axis="x", alpha=.25)
    fig.suptitle("Человеческие тексты, у которых размерность необычна для "
                 "своего жанра\nпрофиль — знаки отклонения по крупной, мелкой "
                 "и средней полосам, порог 10%", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(BASE, "figures", "outlier_profiles.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", os.path.relpath(path, BASE))


def base_mean(D):
    return D[D["профиль"] == "000"]["худшая оценка"].mean()


if __name__ == "__main__":
    main()
