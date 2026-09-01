"""The same map, but by deviation profile rather than by one band at a time.

Testing one band's sign asks whether that band matters with the other two left
free. The profile asks the finer question: what does a document look like when
the coarse band falls *and* the fine one holds, against one where all three
fall together. Nine profiles carry enough judged texts to compare.

Hinged at 10%: within that of the corpus mean a band counts as unremarkable,
and only the excess gives the profile its sign. The threshold is lower than the
20% used for selection because here it has to leave enough texts in each cell,
and because the cells are being compared with each other rather than against a
tail.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as st

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
TH = float(os.environ.get("HINGE", 10.0))
MIN_N = 8
AX = {"chars": "искажение записи слов", "web": "парсинг веб-страницы",
      "loss": "утрата содержания", "dup": "дублирование", "cut": "обрезка"}
FILES = ["coarse_sign.csv", "coarse_sign_мелкий.csv", "coarse_sign_средний.csv"]
CMAP = LinearSegmentedColormap.from_list(
    "dev", ["#2b6cb0", "#a8c4de", "#e6e6e3", "#eab3a6", "#c53030"])


def main():
    D = pd.concat([pd.read_csv(os.path.join(BASE, "results", f))
                   for f in FILES]).drop_duplicates("id")
    P = D[[f"откл_{b}" for b in B]].values
    H = np.sign(P) * np.clip(np.abs(P) - TH, 0, None)
    D["профиль"] = ["".join("0" if v == 0 else ("+" if v > 0 else "−")
                            for v in r) for r in H]
    n = D["профиль"].value_counts()
    keep = n[n >= MIN_N].index
    D = D[D["профиль"].isin(keep)]
    base = D[D["профиль"] == "000"]
    order = (D.groupby("профиль")["damage"].mean()
             .sort_values().index.tolist())

    M = np.array([[D[D["профиль"] == p][k].mean() - base[k].mean()
                   for p in order] for k in AX])
    Pv = np.array([[st.mannwhitneyu(D[D["профиль"] == p][k], base[k]).pvalue
                    if D[D["профиль"] == p][k].nunique() + base[k].nunique() > 2
                    else 1.0 for p in order] for k in AX])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.5, 8.8),
                                 height_ratios=[2.4, 1])
    # same scale as the generator maps, so the figures are comparable
    lim = float(os.environ.get("LIM", 1.6))
    im = ax.imshow(M, cmap=CMAP, vmin=-lim, vmax=lim, aspect="auto")
    labs = [f"{p}\nn={int(n[p])}" for p in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labs, fontsize=10, fontfamily="monospace")
    ax.set_yticks(range(len(AX)))
    ax.set_yticklabels(list(AX.values()), fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            m = "**" if Pv[i, j] < .01 else ("*" if Pv[i, j] < .05 else "")
            ax.text(j, i, f"{M[i, j]:+.2f}{m}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(M[i, j]) > lim * .55 else "#222")
    ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(AX), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Насколько сильнее дефект, чем у профиля 000\n"
                 "* p<0.05, ** p<0.01", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=.025, label="разница с 000, баллы")

    dm = [D[D["профиль"] == p]["damage"] for p in order]
    b0 = base["damage"].mean()
    c = ["#c53030" if st.mannwhitneyu(d, base["damage"]).pvalue < .05
         and d.mean() > b0 else "#c8d3e0" for d in dm]
    bx.bar(range(len(order)), [d.mean() for d in dm], color=c, width=.62)
    bx.axhline(b0, color="#555", ls="--", lw=1.2, label=f"профиль 000: {b0:.2f}")
    for i, d in enumerate(dm):
        p = st.mannwhitneyu(d, base["damage"]).pvalue
        m = "**" if p < .01 else ("*" if p < .05 else "")
        bx.annotate(f"{d.mean():.2f}{m}\n≥3: {(d >= 3).mean():.0%}",
                    (i, d.mean()), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5)
    bx.set_xticks(range(len(order)))
    bx.set_xticklabels(order, fontsize=11, fontfamily="monospace")
    bx.set_ylabel("общий урон, 0-5")
    bx.set_ylim(0, max(d.mean() for d in dm) * 1.5)
    bx.legend(frameon=False, fontsize=9, loc="upper left")
    bx.grid(axis="y", alpha=.25)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    fig.suptitle(f"Профиль отклонения против типа дефекта  "
                 f"(порядок полос: {', '.join(B)}; порог {TH:.0f}%)\n"
                 f"{len(D)} человеческих текстов COLING, оценки судьи по "
                 "дефектам сбора данных", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(BASE, "figures",
                        f"profiles_vs_defects_{TH:.0f}.png")
    fig.savefig(path, dpi=150)
    print("saved", os.path.relpath(path, BASE))
    pd.set_option("display.width", 200)
    print("\n" + D.groupby("профиль").agg(
        n=("damage", "size"), урон=("damage", "mean"),
        **{v[:12]: (k, "mean") for k, v in AX.items()}
    ).round(2).loc[order].to_string())


if __name__ == "__main__":
    main()
