"""Is there a fixed number of cluster-joining edges, or a fixed fraction?

The hypothesis: the long end contains a small, fixed count of edges -- about
four -- that hold genuinely separate clusters together, one of them the sink
bridge. Fixed count, not fixed share: the number is a property of the cloud's
structure, so it should not grow when L grows, whereas anything measured as a
percentile necessarily does.

That is the test, and it is a clean one. For a run at subsample size L, count
the edges whose removal detaches a component of at least k points. If the
structure is four clusters, the count stays near four at every L and the *share*
of edges falls as 1/L. If the long end is instead a continuum of stragglers, the
count grows roughly in proportion to L and the share stays flat.

Both the absolute threshold (>= 5 points) and the relative one (>= 5% of L) are
reported, since which of the two holds still is the whole question.

Before that, a check on the reasoning that motivated the guess: 38.7% of the ten
longest edges have no hapax endpoint, which was read as four cluster bridges. But
an edge with two repeated endpoints can perfectly well be a seam that detaches a
single point, so the size of what comes off is looked at directly.
"""
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from edge_taxonomy import COS_CUT, NORM_CUT, SKIP, corpus_ranks, token_class
from embedder import Embedder
from mst_picture import smaller_side
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 80))
SEEDS = int(os.environ.get("SEEDS", 3))
LS = [41, 81, 121, 201, 321, 401]
MIN_ABS = 5


def one(text, emb, ranks, u_sink, L, seed):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    full = Counter(toks[i] for i in keep)
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v, t = e[idx], [toks[i] for i in idx]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    hap = np.array([full[x] == 1 for x in t]) & ~sink
    cls = ["сток" if s_ else token_class(x, ranks) for x, s_ in zip(t, sink)]

    m = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    o = np.argsort(m.data)
    lens, ra, ca = m.data[o], m.row[o], m.col[o]
    small = smaller_side(ra, ca, lens, L)
    rel = lens / lens.mean()
    touches_sink = sink[ra] | sink[ca]
    # "detaches >= 5 points" alone is useless: a tree of 201 nodes is full of
    # subtrees that size, and half such edges sit in the short half. A bridge
    # has to be long AND hold a group, so the two conditions are conjoined and
    # several thresholds are reported side by side.
    big5 = small >= max(2, int(0.05 * L))
    big10 = small >= max(2, int(0.10 * L))
    return dict(
        L=L,
        **{"группа>=5%, длина>=1.2": int((big5 & (rel >= 1.2)).sum()),
           "группа>=5%, длина>=1.4": int((big5 & (rel >= 1.4)).sum()),
           "группа>=10%, длина>=1.2": int((big10 & (rel >= 1.2)).sum()),
           "группа>=10%, длина>=1.4": int((big10 & (rel >= 1.4)).sum()),
           "группа>=5 точек, длина>=1.4": int(((small >= MIN_ABS)
                                               & (rel >= 1.4)).sum()),
           # the question as asked: among the ten longest edges only, how many
           # actually hold a group. The denominator is fixed at ten, so this
           # cannot grow with L by construction -- if the count is stable the
           # structure is a fixed number of clusters, if it grows the long end
           # is holding more and more real pieces as the sample widens
           "из 10 длинных: группа>=5": int((small[-10:] >= MIN_ABS).sum()),
           "из 10 длинных: группа>=5%": int(
               (small[-10:] >= max(2, int(0.05 * L))).sum())},
        со_стоком=int((big5 & (rel >= 1.2) & touches_sink).sum()),
        рёбер=len(lens),
        # and the check on the reasoning behind the guess
        топ10_без_hapax=int((~hap[ra[-10:]] & ~hap[ca[-10:]]).sum()),
        топ10_без_hapax_лист=int((~hap[ra[-10:]] & ~hap[ca[-10:]]
                                  & (small[-10:] == 1)).sum()),
        топ10_без_hapax_группа=int((~hap[ra[-10:]] & ~hap[ca[-10:]]
                                    & (small[-10:] >= MIN_ABS)).sum()),
    ), (small, rel, hap, cls, ra, ca)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)

    rows, sides = [], []
    for i, s in enumerate(hum[:N_TEXTS]):
        for L in LS:
            for seed in range(SEEDS):
                r = one(s, emb, ranks, u_sink, L, 1000 * seed + i)
                if r is None:
                    continue
                rec, extra = r
                rec["текст"] = i
                rows.append(rec)
                if L == 201 and seed == 0:
                    small, rel, hap, cls, ra, ca = extra
                    for j in np.where(small >= MIN_ABS)[0]:
                        sides.append({"текст": i, "отн_длина": rel[j],
                                      "меньшая": small[j],
                                      "перцентиль": j / len(rel) * 100,
                                      "сток": ("сток" in (cls[ra[j]],
                                                          cls[ca[j]]))})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "bridge_count.csv"), index=False)
    pd.set_option("display.width", 220)

    print("\nсперва — проверка рассуждения, по которому взялась цифра 4\n")
    d = D[D["L"] == 201]
    print(f"  из 10 самых длинных рёбер без hapax-концов: "
          f"{d['топ10_без_hapax'].mean():.2f} штук")
    print(f"  из них отрывают одну точку:                 "
          f"{d['топ10_без_hapax_лист'].mean():.2f}")
    print(f"  из них отрывают группу >= {MIN_ABS}:                 "
          f"{d['топ10_без_hapax_группа'].mean():.2f}")

    print("\n\nглавный тест: растёт ли число перемычек с L\n")
    DEFS = ["из 10 длинных: группа>=5", "из 10 длинных: группа>=5%",
            "группа>=5%, длина>=1.2", "группа>=5%, длина>=1.4",
            "группа>=10%, длина>=1.2", "группа>=10%, длина>=1.4",
            "группа>=5 точек, длина>=1.4"]
    g = D.groupby("L")[DEFS + ["со_стоком"]].mean()
    g["прогонов"] = D.groupby("L").size()
    print("перемычек на прогон, при разных определениях перемычки\n")
    print(g.round(2).to_string())

    ln = np.log(g.index.values.astype(float))
    print("\nкак число растёт с L\n")
    for c in DEFS:
        b = np.polyfit(ln, np.log(g[c].values.clip(1e-9)), 1)[0]
        verdict = ("ПОСТОЯННО — фиксированное число" if abs(b) < 0.2 else
                   "растёт пропорционально L (это доля)" if b > 0.8 else
                   "растёт медленнее L")
        print(f"  {c:30s} ~ L^{b:+.3f}   {verdict}")

    if sides:
        S = pd.DataFrame(sides)
        print(f"\n\nчто это за перемычки при L = 201 "
              f"({len(S)} штук на {S['текст'].nunique()} текстов, "
              f"{len(S) / S['текст'].nunique():.1f} на текст)\n")
        print(f"  со стоком на конце:     {S['сток'].mean() * 100:.0f}%")
        print(f"  медианная отн. длина:   {S['отн_длина'].median():.2f}")
        print(f"  доля среди 10% длинных: "
              f"{(S['перцентиль'] >= 90).mean() * 100:.0f}%")
        print(f"  доля среди 50% коротких:"
              f" {(S['перцентиль'] < 50).mean() * 100:.0f}%")
        print(f"  медианный размер отрываемого: {S['меньшая'].median():.0f}")
        S.to_csv(os.path.join(BASE, "results", "bridge_sides.csv"), index=False)
    plot(g)


def plot(g):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    a.plot(g.index, g["из 10 длинных: группа>=5%"], "o-", color="#08306b",
           label="из 10 длинных: отрывают >= 5% точек")
    a.plot(g.index, g["группа>=5%, длина>=1.4"], "o-", color="#e6550d",
           label="все рёбра: группа >= 5% и длина >= 1.4")
    a.plot(g.index, g["со_стоком"], "o-", color="#333", label="со стоком")
    a.axhline(4, color="#999", ls=":", lw=1)
    a.text(LS[0], 4.15, "гипотеза: 4", fontsize=9, color="#666")
    a.set_xscale("log")
    a.set_xticks(LS)
    a.set_xticklabels([str(x) for x in LS])
    a.set_xlabel("размер выборки L")
    a.set_ylabel("перемычек на прогон")
    a.set_title("Фиксированное число или доля?")
    a.legend(fontsize=9)

    a = ax[1]
    for c, col in [("группа>=10%, длина>=1.2", "#08306b"),
                   ("группа>=10%, длина>=1.4", "#6baed6")]:
        a.plot(g.index, g[c], "o-", color=col, label=c)
    a.axhline(4, color="#999", ls=":", lw=1)
    a.set_xscale("log")
    a.set_xticks(LS)
    a.set_xticklabels([str(x) for x in LS])
    a.set_xlabel("размер выборки L")
    a.set_ylabel("перемычек на прогон")
    a.set_title("Строгое определение: отрывает >= 10% точек")
    a.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "bridge_count.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
