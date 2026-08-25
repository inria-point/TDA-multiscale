"""The pruned perturbation set in the PC basis, coloured by functional group.

The basis is the one fitted in pc_space.py over perturbations *and* generators
together, so these coordinates are the same numbers the generator map uses;
nothing is re-fitted here. Only the perturbations are drawn, and only the ones
that survive the pruning in taxonomy.py, so the picture can be read.

The n-gram points are drawn hollow: an n-gram sampler is a generative model,
not an edit of a given text, so it belongs on the map as a reference point but
not in a causal statement about text properties.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import taxonomy as T

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    P = pd.read_csv(os.path.join(BASE, "results", "pc_space.csv"), index_col=0)
    P = P[P["kind"] == "pert"].drop(columns="kind")
    P = P.loc[[k for k in T.KEEP if k in P.index]]
    P["name"] = [T.KEEP[i] for i in P.index]
    P["group"] = [T.group_of(n) for n in P["name"]]
    P = P.set_index("name")
    P.round(1).to_csv(os.path.join(BASE, "results", "pert_map.csv"))

    for a, b in [("PC1", "PC2"), ("PC2", "PC3")]:
        # two panels: the whole cloud, then the crowd around the origin, where
        # most of the weak perturbations sit and labels would otherwise collide
        zoom = P[(P[a].abs() < 110) & (P[b].abs() < 110)]
        fig, axes = plt.subplots(1, 2, figsize=(19, 9),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
        for ax, sel, title in [(axes[0], P, "весь набор"),
                               (axes[1], zoom, "центр крупным планом")]:
            ax.axhline(0, c="k", lw=1)
            ax.axvline(0, c="k", lw=1)
            ax.scatter(0, 0, s=760, marker="*", c="#1a7f37",
                       edgecolors="black", linewidths=0.9, zorder=6)
            texts = [ax.text(0, 0, "человек", fontsize=11, color="#1a7f37",
                             fontweight="bold")]
            for g, (label, c) in T.GROUPS.items():
                sub = sel[sel["group"] == g]
                if sub.empty:
                    continue
                hollow = g == "ngram"
                ax.scatter(sub[a], sub[b], s=95, zorder=3, label=label,
                           facecolors="none" if hollow else c,
                           edgecolors=c, linewidths=1.8 if hollow else 0.6)
                for name, r in sub.iterrows():
                    texts.append(ax.text(r[a], r[b], T.label_of(name),
                                         fontsize=9, color=c))
            adjust_text(texts, ax=ax, expand=(1.4, 1.7),
                        force_text=(0.7, 1.0),
                        arrowprops=dict(arrowstyle="-", color="#999999",
                                        lw=0.5))
            ax.set_xlabel(a)
            ax.set_ylabel(b)
            ax.set_title(title)
            ax.grid(alpha=0.25)
        # the legend goes on the full panel, which is the only one that
        # contains every group
        axes[0].legend(loc="upper right", fontsize=8.5,
                       framealpha=0.92)
        fig.suptitle("Пертурбации в базисе главных компонент; звезда — "
                     "неизменённый человеческий текст. "
                     "Полые кружки — н-граммный генератор: модель, а не правка")
        fig.tight_layout()
        path = os.path.join(BASE, "figures", f"pert_map_{a}_{b}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", path)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 60)
    print(P.sort_values(["group", "PC1"]).round(1).to_string())


if __name__ == "__main__":
    main()
