"""Human texts grouped by which bands are unusual for their genre.

Sorting by one criterion at a time answers the wrong question: a text can be
odd on the middle band and ordinary on the other two, and lumping it with texts
that are odd everywhere hides that. So each text gets a hinged deviation --
anything within 10 per cent of its genre mean counts as zero, and only the
excess beyond that survives -- and the sign pattern of the three hinged values
is its profile.

The hinge does the work a clustering algorithm would otherwise be asked to do,
and does it with a boundary that means something: 000 is "unremarkable on every
band", and everything else names which bands are off and in what direction.
Comparison is within genre, since an arXiv abstract and a Reddit post have no
business sharing a baseline, and in per cent, since no human text in this
corpus is even close to ten absolute units from its genre mean.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
TH = float(os.environ.get("HINGE", 20.0))
# genre means make the comparison fair but leave the tails too thin to compare
# profiles against each other: at 20% no genre-relative profile reaches twenty
# texts. Against the corpus mean the same threshold leaves 209 texts out of
# norm in groups of 8 to 38, at the cost that genre itself now counts as
# deviation -- which the genre breakdown of each profile makes checkable.
BASELINE = os.environ.get("BASELINE", "corpus")
MIN_GROUP = int(os.environ.get("MIN_GROUP", 10))
PER_GROUP = int(os.environ.get("PER_GROUP", 40))


def hinge(T):
    if BASELINE == "corpus":
        for c in ["общая"] + B:
            T[f"pct_{c}"] = (T[c] - T[c].mean()) / T[c].mean() * 100
    P = T[[f"pct_{b}" for b in B]].values
    H = np.sign(P) * np.clip(np.abs(P) - TH, 0, None)
    T = T.copy()
    for i, b in enumerate(B):
        T[f"h_{b}"] = H[:, i]
    T["профиль"] = ["".join("0" if v == 0 else ("+" if v > 0 else "−")
                            for v in row) for row in H]
    T["сила"] = np.abs(H).sum(1)
    return T


def main():
    T = hinge(pd.read_csv(os.path.join(BASE, "results", "human_band_dev.csv")))
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    T["id"] = T["id"].astype(str)
    T = T.merge(pool[["id", "text"]], on="id", how="inner")

    n = T["профиль"].value_counts()
    keep = n[n >= MIN_GROUP].index
    rng = np.random.default_rng(0)
    parts = []
    for p in keep:
        g = T[T["профиль"] == p]
        # the strongest cases in each profile, so the group is what it claims;
        # the control is sampled at random since there is nothing to rank
        g = (g.sample(min(PER_GROUP, len(g)), random_state=0) if p == "000"
             else g.nlargest(min(PER_GROUP, len(g)), "сила"))
        parts.append(g)
    S = pd.concat(parts)
    path = os.path.join(BASE, "results", "outlier_sample.csv")
    S.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    print(f"база: {BASELINE}; порог {TH:.0f}%; "
          f"профилей с n>={MIN_GROUP}: {len(keep)}; "
          f"отобрано {len(S)} из {len(T)} текстов\n")
    r = S.groupby("профиль").agg(
        n=("id", "size"), сила=("сила", "mean"),
        **{b: (f"pct_{b}", "mean") for b in B})
    print(r.round(1).sort_values("сила", ascending=False).to_string())
    print("\nжанровый состав профилей (проверка, не ловим ли мы жанр):")
    for p in r.sort_values("n", ascending=False).index:
        g = S[S["профиль"] == p]
        print(f"  {p} n={len(g):3d}  " + ", ".join(
            f"{k}:{n}" for k, n in g["sub_source"].value_counts().head(4).items()))
    print(f"\nжанров: {S['sub_source'].nunique()}; "
          f"слов в тексте: медиана {S['text'].str.split().str.len().median():.0f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
