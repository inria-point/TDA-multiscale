"""Does the dimension see invention once fluency is held level?

Two contrasts share the raw answer as baseline. Polished minus raw changes the
writing and keeps the invented content; corrected minus polished removes the
invention at nearly the same level of fluency. The second is the test.

Answers that drifted out of English are dropped: the small model occasionally
switches language mid-task, which whitespace word counts do not catch and
which would dominate any comparison.
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
from mech_props import NAMES as MECH, corpus_ranks, measure
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
VERSIONS = ["raw", "polished", "fixed"]
LABEL = {"raw": "выдумка, как есть", "polished": "та же выдумка, гладко",
         "fixed": "факты исправлены"}
CJK = re.compile(r"[　-鿿]")


def english_only(texts):
    bad = {q for q, t in texts["raw"].items()
           if len(CJK.findall(t)) > 20}
    if bad:
        print(f"отброшено за смену языка: {sorted(bad)}")
    return bad


def bands(D):
    out = {}
    for name, (mode, qmin) in BANDS.items():
        s = D[(D["mode"] == mode) & (D["q"] >= qmin)]
        out[name] = s.groupby(["version", "qid"])["d_hat"].mean()
    return pd.DataFrame(out).reset_index()


def main():
    texts = {v: json.load(open(os.path.join(
        BASE, "results", f"halluc_{v}.json")))["answers"] for v in VERSIONS}
    D = pd.read_csv(os.path.join(BASE, "results", "halluc_qphd_L201.csv.gz"))
    D = D[D["d_hat"] > 0]
    bad = english_only(texts)
    D = D[~D["qid"].isin(bad)]
    B = bands(D)
    W = B.pivot(index="qid", columns="version")
    print(f"\n{len(W)} вопросов")

    S = pd.read_csv(os.path.join(BASE, "results", "halluc_scores.csv"))
    S = S[~S["qid"].isin(bad)].set_index("qid")

    ranks = corpus_ranks([t for t in texts["raw"].values()])
    mech = {v: pd.DataFrame({q: measure(t, ranks) for q, t in texts[v].items()
                             if q not in bad}).T for v in VERSIONS}

    print("\n=== оценки судьи (среднее)")
    for k, ru in (("invention", "выдуманность"), ("fluency", "гладкость")):
        vals = [S[f"{v}_{k}"].mean() for v in VERSIONS if f"{v}_{k}" in S]
        print(f"  {ru:14s} " + "   ".join(
            f"{LABEL[v]}: {m:.1f}" for v, m in zip(VERSIONS, vals)))

    print("\n=== контроль: совпадают ли лексические статистики")
    keep = ["ttr", "word_entropy", "hapax_share", "mean_log_rank",
            "trigram_repeat"]
    ctl = pd.DataFrame({LABEL[v]: mech[v][keep].mean() for v in VERSIONS})
    ctl.index = [MECH[k] for k in keep]
    print(ctl.round(3).to_string())

    print("\n=== размерность d по полосам")
    rows = []
    for band in BANDS:
        col = W[band]
        for a, b in [("raw", "polished"), ("polished", "fixed"),
                     ("raw", "fixed")]:
            d = (col[b] - col[a]).dropna()
            rel = (d / col[a].reindex(d.index) * 100)
            t = stats.wilcoxon(d) if len(d) > 5 else None
            rows.append({
                "полоса": band, "контраст": f"{LABEL[a]} → {LABEL[b]}",
                "n": len(d), "d до": col[a].mean(), "d после": col[b].mean(),
                "сдвиг %": rel.mean(), "медиана %": rel.median(),
                "выросло у": (d > 0).mean(),
                "p": t.pvalue if t is not None else np.nan})
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 210)
    print(R.round(3).to_string(index=False))
    R.round(4).to_csv(os.path.join(BASE, "results", "halluc_result.csv"),
                      index=False)
    figure(W, R)


def figure(W, R):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, band in zip(axes, BANDS):
        col = W[band]
        for i, v in enumerate(VERSIONS):
            ax.scatter(np.full(len(col), i) + np.linspace(-.09, .09, len(col)),
                       col[v], s=18, alpha=.55, color="#2b6cb0", linewidths=0)
        for q in col.index:
            ax.plot(range(3), [col[v][q] for v in VERSIONS], color="#9aa5b1",
                    lw=.5, alpha=.5, zorder=0)
        ax.plot(range(3), [col[v].mean() for v in VERSIONS], color="#c53030",
                lw=2.5, marker="o", zorder=5)
        sub = R[R["полоса"] == band].set_index("контраст")
        key = f"{LABEL['polished']} → {LABEL['fixed']}"
        ax.set_title(f"{band} масштаб\n"
                     f"снятие выдумки при равной гладкости: "
                     f"{sub.loc[key, 'сдвиг %']:+.1f}%, p={sub.loc[key, 'p']:.2f}",
                     fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels([LABEL[v] for v in VERSIONS], fontsize=8.5)
        ax.set_ylabel("d")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Одни и те же ответы в трёх версиях; линия — один вопрос, "
                 "красная — среднее", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(BASE, "figures", "halluc.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
