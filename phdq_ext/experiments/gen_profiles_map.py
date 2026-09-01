"""Generator profiles against defect types, measured off the human control.

Two clusters, found by grouping the models on their three band means against
human text. They turn out to split by generation rather than by training
regime: everything from llama-1 onward in one, the OPT/bloom/GPT-J era in the
other, the latter sitting 56% above human on the fine band -- the signature of
repetition.

They occupy almost disjoint profile space, so each is drawn separately. The
comparison in every cell is against the same human control used throughout,
the 40 documents whose bands sit within 6% of the corpus mean, so generated and
human text are read on one ruler.
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
TH = int(os.environ.get("HINGE", 20))
MIN_N = 6
AX = {"chars": "искажение записи слов", "web": "парсинг веб-страницы",
      "loss": "утрата содержания", "dup": "дублирование", "cut": "обрезка"}
NAME = {1: "кластер 1 — llama и новее (n=20 моделей)",
        2: "кластер 2 — OPT, bloom, GPT-J (n=13 моделей)"}
CMAP = LinearSegmentedColormap.from_list(
    "dev", ["#2b6cb0", "#a8c4de", "#e6e6e3", "#eab3a6", "#c53030"])


def main():
    D = pd.read_csv(os.path.join(BASE, "results", "gen_judged.csv")).dropna(
        subset=["кластер"])
    H = pd.read_csv(os.path.join(BASE, "results", "coarse_sign.csv"))
    base = H[H["группа"] == "контроль"]
    col = f"профиль{TH}"

    # one colour scale for both clusters and for the human map, so the three
    # figures can be laid side by side and read against each other
    lim = float(os.environ.get("LIM", 1.6))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5),
                             height_ratios=[2.3, 1], sharey="row")
    for j, cl in enumerate((1, 2)):
        G = D[D["кластер"] == cl]
        n = G[col].value_counts()
        order = (G[G[col].isin(n[n >= MIN_N].index)].groupby(col)["damage"]
                 .mean().sort_values().index.tolist())
        M = np.array([[G[G[col] == p][k].mean() - base[k].mean()
                       for p in order] for k in AX])
        Pv = np.array([[st.mannwhitneyu(G[G[col] == p][k], base[k]).pvalue
                        if G[G[col] == p][k].nunique() + base[k].nunique() > 2
                        else 1.0 for p in order] for k in AX])
        ax = axes[0][j]
        im = ax.imshow(M, cmap=CMAP, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{p}\nn={int(n[p])}" for p in order],
                           fontsize=9.5, fontfamily="monospace")
        ax.set_yticks(range(len(AX)))
        if j == 0:
            ax.set_yticklabels(list(AX.values()), fontsize=10)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                m = "**" if Pv[a, b] < .01 else ("*" if Pv[a, b] < .05 else "")
                ax.text(b, a, f"{M[a, b]:+.2f}{m}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if abs(M[a, b]) > lim * .55 else "#222")
        ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(AX), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(NAME[cl], fontsize=11)


        bx = axes[1][j]
        dm = [G[G[col] == p]["damage"] for p in order]
        b0 = base["damage"].mean()
        c = ["#c53030" if st.mannwhitneyu(d, base["damage"]).pvalue < .05
             and d.mean() > b0 else "#c8d3e0" for d in dm]
        bx.bar(range(len(order)), [d.mean() for d in dm], color=c, width=.6)
        bx.axhline(b0, color="#555", ls="--", lw=1.2,
                   label=f"человеческий контроль: {b0:.2f}")
        for i, d in enumerate(dm):
            p = st.mannwhitneyu(d, base["damage"]).pvalue
            m = "**" if p < .01 else ("*" if p < .05 else "")
            bx.annotate(f"{d.mean():.2f}{m}\n≥3: {(d >= 3).mean():.0%}",
                        (i, d.mean()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
        bx.set_xticks(range(len(order)))
        bx.set_xticklabels(order, fontsize=10, fontfamily="monospace")
        if j == 0:
            bx.set_ylabel("общий урон, 0-5")
        bx.set_ylim(0, max(3.0, max(d.mean() for d in dm) * 1.35))
        bx.legend(frameon=False, fontsize=8.5, loc="upper left")
        bx.grid(axis="y", alpha=.25)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)

    fig.colorbar(im, ax=list(axes[0]), fraction=.022,
                 label="разница с человеческим контролем, баллы")
    fig.suptitle(f"Сгенерированный текст: профиль отклонения от человеческого "
                 f"среднего против типа дефекта (порог {TH}%)\n"
                 f"порядок полос {', '.join(B)}; сравнение с человеческим "
                 f"контролем 000; {len(D)} текстов, 33 модели", fontsize=12)
    fig.subplots_adjust(top=0.86, bottom=0.07, hspace=0.42)
    path = os.path.join(BASE, "figures", f"gen_profiles_{TH}.png")
    fig.savefig(path, dpi=150)
    print("saved", os.path.relpath(path, BASE))
    pd.set_option("display.width", 200)
    print(D.groupby(["кластер", col]).agg(
        n=("damage", "size"), урон=("damage", "mean"),
        **{v[:11]: (k, "mean") for k, v in AX.items()}
    ).round(2).to_string())


if __name__ == "__main__":
    main()
