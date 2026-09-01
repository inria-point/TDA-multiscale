"""Which band's failure means which kind of damage.

Three sign tests, one per band, each comparing texts far above and far below
the corpus mean against texts sitting on it. The interesting quantity is not
whether a group is damaged but which axis carries the damage, so the map shows
each group's departure from the control on every axis at once.

Diverging scale with a neutral grey midpoint, since zero here is a meaningful
origin -- "the same as an ordinary document" -- and the two directions mean
opposite things.
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
BASE = os.path.join(HERE, "..")

AX = {"chars": "искажение записи слов", "web": "парсинг веб-страницы",
      "loss": "утрата содержания", "dup": "дублирование", "cut": "обрезка"}
SETS = {"крупная": "coarse_sign.csv", "мелкая": "coarse_sign_мелкий.csv",
        "средняя": "coarse_sign_средний.csv"}
# blue for less than ordinary, grey for the same, red for more
CMAP = LinearSegmentedColormap.from_list(
    "dev", ["#2b6cb0", "#a8c4de", "#e6e6e3", "#eab3a6", "#c53030"])


def main():
    cols, cells, stars, dmg = [], [], [], []
    for band, fn in SETS.items():
        D = pd.read_csv(os.path.join(BASE, "results", fn))
        ctl = D[D["группа"] == "контроль"]
        for direc in ("вниз", "вверх"):
            g = D[D["группа"].str.endswith(direc)]
            cols.append(f"{band}\n{direc}")
            cells.append([g[k].mean() - ctl[k].mean() for k in AX])
            stars.append([st.mannwhitneyu(g[k], ctl[k]).pvalue
                          if g[k].nunique() + ctl[k].nunique() > 2 else 1.0
                          for k in AX])
            dmg.append((g["damage"].mean(), ctl["damage"].mean(),
                        st.mannwhitneyu(g["damage"], ctl["damage"]).pvalue,
                        (g["damage"] == 0).mean() * 100))
    M = np.array(cells).T
    P = np.array(stars).T

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(11.5, 8.6), height_ratios=[2.5, 1])
    lim = np.abs(M).max()
    im = ax.imshow(M, cmap=CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(AX)))
    ax.set_yticklabels(list(AX.values()), fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            mark = "**" if P[i, j] < 0.01 else ("*" if P[i, j] < 0.05 else "")
            ax.text(j, i, f"{M[i, j]:+.2f}{mark}", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(M[i, j]) > lim * .55 else "#222")
    ax.set_title("Насколько сильнее дефект, чем у обычного текста\n"
                 "* p<0.05, ** p<0.01 против контроля", fontsize=11)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(AX), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, fraction=.025, label="разница с контролем, баллы")

    x = np.arange(len(cols))
    vals = [d[0] for d in dmg]
    c = ["#c53030" if d[2] < .05 else "#c8d3e0" for d in dmg]
    bx.bar(x, vals, color=c, width=.62)
    bx.axhline(dmg[0][1], color="#555", ls="--", lw=1.2,
               label=f"контроль {dmg[0][1]:.2f}")
    for i, d in enumerate(dmg):
        mark = "**" if d[2] < .01 else ("*" if d[2] < .05 else "")
        bx.annotate(f"{d[0]:.2f}{mark}\nбез дефектов {d[3]:.0f}%",
                    (i, d[0]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5)
    bx.set_xticks(x)
    bx.set_xticklabels(cols, fontsize=10)
    bx.set_ylabel("общий урон, 0-5")
    bx.set_ylim(0, max(vals) * 1.45)
    bx.legend(frameon=False, fontsize=9, loc="upper left")
    bx.set_title("Общий урон по группам", fontsize=11)
    bx.grid(axis="y", alpha=.25)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    fig.suptitle("Провал полосы означает повреждение, подъём — нет; "
                 "какое именно повреждение, зависит от полосы\n"
                 "по 40 человеческих текстов COLING в каждой группе, "
                 "оценки судьи по дефектам сбора данных", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(BASE, "figures", "bands_vs_defects.png")
    fig.savefig(path, dpi=150)
    print("saved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
