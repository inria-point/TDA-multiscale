"""Human documents and both generator clusters on one sheet.

The same question asked of three populations: given a document whose bands
deviate in a particular pattern, what kind of damage does a reader find in it?
Everything is measured off one baseline -- the human control, forty documents
whose three bands sit within 6% of the corpus mean -- and drawn on one colour
scale, so a cell here means the same as a cell three panels over.

That is the point of putting them together. The profile that ranks worst is
--- in all three, but the row that lights up under it differs: character
corruption for scraped human text, duplication and truncation for a modern
generator, nothing at all for the OPT-era one.
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
LIM = 1.6
AX = {"chars": "искажение записи слов", "web": "парсинг веб-страницы",
      "loss": "утрата содержания", "dup": "дублирование", "cut": "обрезка"}
CMAP = LinearSegmentedColormap.from_list(
    "dev", ["#2b6cb0", "#a8c4de", "#e6e6e3", "#eab3a6", "#c53030"])


def hinge(D, cols):
    P = D[cols].values
    H = np.sign(P) * np.clip(np.abs(P) - TH, 0, None)
    return ["".join("0" if v == 0 else ("+" if v > 0 else "−") for v in r)
            for r in H]


def panels():
    """(title, frame, profile column, minimum cell size) for each population."""
    H = pd.concat([pd.read_csv(os.path.join(BASE, "results", f)) for f in
                   ("coarse_sign.csv", "coarse_sign_мелкий.csv",
                    "coarse_sign_средний.csv")]).drop_duplicates("id")
    H["профиль"] = hinge(H, [f"откл_{b}" for b in B])
    G = pd.read_csv(os.path.join(BASE, "results", "gen_judged.csv")).dropna(
        subset=["кластер"])
    G["профиль"] = hinge(G, [f"откл_{b}" for b in B])
    return [("человеческие тексты COLING", H, 8),
            ("генераторы: llama и новее", G[G["кластер"] == 1], 6),
            ("генераторы: OPT, bloom, GPT-J", G[G["кластер"] == 2], 6)], H


def main():
    sets, H = panels()
    base = pd.read_csv(os.path.join(BASE, "results", "coarse_sign.csv"))
    base = base[base["группа"] == "контроль"]
    b0 = base["damage"].mean()

    orders = []
    for _, D, minn in sets:
        n = D["профиль"].value_counts()
        keep = n[n >= minn].index
        orders.append((D[D["профиль"].isin(keep)].groupby("профиль")["damage"]
                       .mean().sort_values().index.tolist(), n))

    fig, axes = plt.subplots(
        2, 3, figsize=(23, 9.8), height_ratios=[2.3, 1], sharey="row",
        width_ratios=[len(o) for o, _ in orders])
    for j, ((title, D, _), (order, n)) in enumerate(zip(sets, orders)):
        M = np.array([[D[D["профиль"] == p][k].mean() - base[k].mean()
                       for p in order] for k in AX])
        Pv = np.array([[st.mannwhitneyu(D[D["профиль"] == p][k], base[k]).pvalue
                        if D[D["профиль"] == p][k].nunique()
                        + base[k].nunique() > 2 else 1.0
                        for p in order] for k in AX])
        ax = axes[0][j]
        im = ax.imshow(M, cmap=CMAP, vmin=-LIM, vmax=LIM, aspect="auto")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{p}\nn={int(n[p])}" for p in order],
                           fontsize=9, fontfamily="monospace")
        ax.set_yticks(range(len(AX)))
        if j == 0:
            ax.set_yticklabels(list(AX.values()), fontsize=10)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                m = "**" if Pv[a, b] < .01 else ("*" if Pv[a, b] < .05 else "")
                ax.text(b, a, f"{M[a, b]:+.2f}{m}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(M[a, b]) > LIM * .55 else "#222")
        ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(AX), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{title}   (n={len(D)})", fontsize=11)

        bx = axes[1][j]
        dm = [D[D["профиль"] == p]["damage"] for p in order]
        c = ["#c53030" if st.mannwhitneyu(d, base["damage"]).pvalue < .05
             and d.mean() > b0 else "#c8d3e0" for d in dm]
        bx.bar(range(len(order)), [d.mean() for d in dm], color=c, width=.62)
        bx.axhline(b0, color="#555", ls="--", lw=1.2)
        for i, d in enumerate(dm):
            p = st.mannwhitneyu(d, base["damage"]).pvalue
            m = "**" if p < .01 else ("*" if p < .05 else "")
            bx.annotate(f"{d.mean():.2f}{m}\n≥3: {(d >= 3).mean():.0%}",
                        (i, d.mean()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5)
        bx.set_xticks(range(len(order)))
        bx.set_xticklabels(order, fontsize=9.5, fontfamily="monospace")
        bx.set_ylim(0, 3.6)
        bx.grid(axis="y", alpha=.25)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        if j == 0:
            bx.set_ylabel("общий урон, 0-5")
            # as a legend entry, not an annotation: at 1.02 it sat on top of
            # the bar labels of the quietest profiles
            bx.plot([], [], color="#555", ls="--", lw=1.2,
                    label=f"человеческий контроль {b0:.2f}")
            bx.legend(frameon=False, fontsize=8.5, loc="upper left")


    fig.suptitle(
        f"Профиль отклонения против типа дефекта, порог {TH}%  "
        f"(порядок полос: {', '.join(B)})\n"
        "всё измерено от одного человеческого контроля и на одной цветовой "
        "шкале; * p<0.05, ** p<0.01", fontsize=13)
    # the colourbar gets an axis of its own: handed the panels instead, it
    # takes space back from subplots_adjust and clips the last column
    fig.subplots_adjust(top=0.85, bottom=0.07, left=0.098, right=0.915,
                        hspace=0.42, wspace=0.05)
    cax = fig.add_axes([0.928, 0.44, 0.008, 0.40])
    fig.colorbar(im, cax=cax, label="разница с человеческим контролем, баллы")
    path = os.path.join(BASE, "figures", f"all_profiles_{TH}.png")
    fig.savefig(path, dpi=150)
    print("saved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
