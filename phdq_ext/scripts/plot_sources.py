"""qPHD(q) curves by source within each genre, plus separability of the sources.

Two questions:
  * where on the q axis do the generators depart from human text
  * how far apart are they there, in units the noise can be judged against

Separability is reported as ROC-AUC of d_hat between human and each model at
each q: 0.5 means indistinguishable, 1.0 perfectly separable by that single
number. AUC is used rather than a difference of means because it is invariant
to the scale of d, which changes by a factor of five across q.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from data import GENRES, MODELS

BASE = os.path.join(os.path.dirname(__file__), "..")
MODE_ORDER = ["q_small", "q_large", "q0.5_range"]


def curves_by_source(df, mode, tag):
    """One panel per genre, one line per source."""
    sub = df[df["mode"] == mode]
    fig, axes = plt.subplots(1, len(GENRES), figsize=(4.2 * len(GENRES), 4.4),
                             sharey=True)
    for ax, genre in zip(axes, GENRES):
        g = sub[sub["genre"] == genre]
        for source, gg in g.groupby("source"):
            stats = gg.groupby("q")["d_hat"].agg(["mean", "sem"])
            style = dict(lw=2.4, zorder=5) if source == "human" else dict(lw=1.3)
            color = "black" if source == "human" else None
            line, = ax.plot(stats.index, stats["mean"], label=source,
                            color=color, **style)
            ax.fill_between(stats.index, stats["mean"] - stats["sem"],
                            stats["mean"] + stats["sem"], alpha=0.2,
                            color=line.get_color())
        ax.set_title(genre, fontsize=10)
        ax.set_xlabel("q")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("d_hat")
    axes[0].legend(fontsize=8)
    # SEM = sd/sqrt(n): precision of the group mean, not the spread of texts.
    # At n = 150 it is ~12x narrower than the text-to-text sd, so a tight band
    # here says the average is well located, not that single texts separate.
    fig.suptitle(f"{tag} — {mode}; полоса = ±стандартная ошибка среднего "
                 f"(не разброс текстов)")
    fig.tight_layout()
    path = os.path.join(BASE, "figures", f"{tag}_sources_{mode}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def auc_table(df):
    """ROC-AUC of d_hat, human vs each model, per genre / mode / q."""
    rows = []
    for (genre, mode, q), g in df.groupby(["genre", "mode", "q"]):
        human = g[g["source"] == "human"]["d_hat"].dropna()
        if len(human) < 5:
            continue
        for model in MODELS:
            m = g[g["source"] == model]["d_hat"].dropna()
            if len(m) < 5:
                continue
            y = np.r_[np.zeros(len(human)), np.ones(len(m))]
            s = np.r_[human.to_numpy(), m.to_numpy()]
            rows.append({"genre": genre, "mode": mode, "q": q, "model": model,
                         "auc": roc_auc_score(y, s), "n_human": len(human),
                         "n_model": len(m)})
    return pd.DataFrame(rows)


def plot_auc(auc, tag):
    """|AUC - 0.5| is the distance from 'indistinguishable', either direction."""
    for mode in MODE_ORDER:
        sub = auc[auc["mode"] == mode]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, len(GENRES), figsize=(4.2 * len(GENRES), 4.0),
                                 sharey=True)
        for ax, genre in zip(axes, GENRES):
            g = sub[sub["genre"] == genre]
            for model, gg in g.groupby("model"):
                gg = gg.sort_values("q")
                ax.plot(gg["q"], gg["auc"], marker="o", ms=3, label=model)
            ax.axhline(0.5, c="k", ls=":", lw=1)
            ax.set_title(genre, fontsize=10)
            ax.set_xlabel("q")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("ROC-AUC (human vs модель)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{tag} — различимость human и генераторов, {mode}")
        fig.tight_layout()
        path = os.path.join(BASE, "figures", f"{tag}_auc_{mode}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.csv))[0]
    df = pd.read_csv(args.csv)
    df = df[df["d_hat"] > 0]

    for mode in MODE_ORDER:
        curves_by_source(df, mode, tag)

    auc = auc_table(df)
    auc_path = os.path.join(BASE, "results", f"{tag}_auc.csv")
    auc.to_csv(auc_path, index=False)
    plot_auc(auc, tag)
    print("saved", auc_path)

    pd.set_option("display.width", 220)
    print("\n=== AUC, усреднённый по жанрам (0.5 = неразличимы)")
    for mode in MODE_ORDER:
        s = auc[auc["mode"] == mode]
        if s.empty:
            continue
        print(f"\n--- {mode}")
        print(s.pivot_table(index="model", columns="q", values="auc")
              .round(3).to_string())

    print("\n=== лучший q по различимости, для каждой пары жанр/модель")
    best = (auc.assign(dist=(auc["auc"] - 0.5).abs())
            .sort_values("dist", ascending=False)
            .groupby(["genre", "model"]).head(1)
            .sort_values(["genre", "model"]))
    print(best[["genre", "model", "mode", "q", "auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
