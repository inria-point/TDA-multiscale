"""What are the islands on the t-SNE map?

token_geometry.py draws the map and colours it by class, and the picture is the
same every time: one large diffuse continent, a few dozen small tight islands,
and one detached island of attention sinks. Colour says the continent is content
words and the islands are function words and punctuation, but that is a class
statement and the interesting question is finer -- is an island one token, or a
neighbourhood of similar tokens?

DBSCAN on the two-dimensional map, then the purity of each cluster with respect
to its commonest token. Nothing here depends on t-SNE being faithful: the map is
used only to find candidate groups, and purity is computed on the token labels,
which are exact.
"""
import os
import sys
from collections import Counter

import pandas as pd
from sklearn.cluster import DBSCAN

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "..")
EPS = 2.5
MIN_SAMPLES = 12


def main():
    d = pd.read_csv(os.path.join(BASE, "results", "token_tsne.csv"))
    d["остров"] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(
        d[["x", "y"]].values)
    core = d[d["остров"] >= 0]
    rows = []
    for c, g in core.groupby("остров"):
        tok, n = Counter(g["tok"]).most_common(1)[0]
        rows.append({"остров": c, "точек": len(g), "главный токен": tok,
                     "чистота, %": n / len(g) * 100,
                     "класс": Counter(g["cls"]).most_common(1)[0][0],
                     "норма": g["norm"].mean()})
    r = pd.DataFrame(rows).sort_values("точек", ascending=False)
    pd.set_option("display.width", 200)
    print(f"{len(d)} точек, {len(core)} в плотных группах "
          f"({len(core) / len(d) * 100:.0f}%), групп {len(r)}\n")
    print(r.round(1).to_string(index=False))

    # the continent is one huge cluster of near-zero purity; the islands are
    # small and nearly pure. Splitting at 100 points separates them cleanly.
    isl = r[r["точек"] < 100]
    big = r[r["точек"] >= 100]
    print(f"\nостровов меньше 100 точек: {len(isl)}, "
          f"медианная чистота {isl['чистота, %'].median():.0f}%")
    print(f"крупных групп: {len(big)}, "
          f"чистота {big['чистота, %'].round(0).tolist()}")
    print("\nдоля точек класса, не попавших ни в одну плотную группу, %")
    print((d[d["остров"] < 0]["cls"].value_counts()
           / d["cls"].value_counts() * 100).round(0).to_string())
    r.round(2).to_csv(os.path.join(BASE, "results", "tsne_islands.csv"),
                      index=False)


if __name__ == "__main__":
    main()
