"""Every pair of versions against every band, one point per question.

The line plot shows the means moving and hides how little the individual texts
agree with that movement. Here each question is a point against the diagonal:
distance from the diagonal is what the rewrite did to that text, and the spread
around it is what any detector built on this would have to beat.
"""
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from halluc_result import CJK, LABEL, VERSIONS, bands
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
# the chain, so each row is one edit on a fixed starting point
PAIRS = [("raw", "polished"), ("polished", "chained"), ("raw", "chained")]
SHORT = {"raw": "выдумка", "polished": "выдумка, гладко",
         "chained": "гладко, факты исправлены",
         "fixed": "правка из исходника"}
ROW = ["полировка языка, факты те же",
       "правка фактов на гладком тексте",
       "обе правки вместе"]


def main():
    texts = {v: json.load(open(os.path.join(
        BASE, "results", f"halluc_{v}.json")))["answers"] for v in VERSIONS}
    bad = {q for q, t in texts["raw"].items() if len(CJK.findall(t)) > 20}
    D = pd.read_csv(os.path.join(BASE, "results", "halluc_qphd_L201.csv.gz"))
    D = D[(D["d_hat"] > 0) & ~D["qid"].isin(bad)]
    W = bands(D).pivot(index="qid", columns="version")

    grid(W, PAIRS, "halluc_scatter.png",
         "Каждая точка — один вопрос. Диагональ = версия ничего не изменила.\n"
         "строки: " + " / ".join(ROW))
    # the contrast the experiment exists for, on its own
    grid(W, [("polished", "chained")], "halluc_scatter_facts.png",
         "Гладкий текст до и после правки фактов — язык не менялся.\n"
         "Каждая точка — один вопрос; диагональ = размерность не сдвинулась")
    print(f"({len(W)} вопросов)")


def grid(W, pairs, fname, suptitle):
    fig, axes = plt.subplots(len(pairs), 3, squeeze=False,
                             figsize=(15.5, 5 * len(pairs)))
    for i, (a, b) in enumerate(pairs):
        for j, band in enumerate(BANDS):
            ax = axes[i][j]
            col = W[band]
            x, y = col[a], col[b]
            lo = min(x.min(), y.min()) * 0.97
            hi = max(x.max(), y.max()) * 1.03
            ax.plot([lo, hi], [lo, hi], color="#888", lw=1.2, zorder=1)
            up = (y > x)
            ax.scatter(x[up], y[up], s=26, color="#2f855a", alpha=.75,
                       linewidths=0, label=f"выросло: {up.sum()}")
            ax.scatter(x[~up], y[~up], s=26, color="#c53030", alpha=.75,
                       linewidths=0, label=f"упало: {(~up).sum()}")
            r = stats.spearmanr(x, y).statistic
            rel = ((y - x) / x * 100)
            p = stats.wilcoxon(y - x).pvalue
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"d: {SHORT[a]}")
            ax.set_ylabel(f"d: {SHORT[b]}")
            # the absolute level matters as much as the shift: a 2% move on
            # a d of 20 is a different claim from 2% on a d of 3.6
            ax.set_title(f"{band} масштаб\n"
                         f"среднее d: {x.mean():.2f} → {y.mean():.2f}"
                         f"  ({rel.mean():+.1f}%, медиана {rel.median():+.1f}%,"
                         f" p={p:.3f})\n"
                         f"связь между версиями ρ={r:.2f}", fontsize=9)
            ax.legend(fontsize=7.5, loc="upper left", frameon=False)
            ax.grid(alpha=.25)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.045 / len(pairs) * 3))
    path = os.path.join(BASE, "figures", fname)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
