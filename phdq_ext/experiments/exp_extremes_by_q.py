"""Stage 1: what separates high-d texts from low-d texts, at each q separately.

For every (mode, q) the pool is split at the deciles of d_hat measured *at that
q*, and the two groups are compared on the whole feature battery. A feature
that separates the extremes there becomes a hypothesis: perturb it and see
whether d moves.

Two confounds are controlled. Domain is a strong predictor of both d and most
features, so effect sizes are also reported after removing domain means
(within-domain standardisation). Length is held out by computing every rate on
a fixed word budget.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from text_features import WORD, features

BASE = os.path.join(os.path.dirname(__file__), "..")


def corpus_ranks(texts):
    c = Counter()
    for t in texts:
        c.update(WORD.findall(t.lower()))
    return {w: i + 1 for i, (w, _) in enumerate(c.most_common())}


def build_features(pool):
    ranks = corpus_ranks(pool["text"])
    rows = []
    for _, r in pool.iterrows():
        f = features(r["text"], ranks=ranks)
        if not f:
            continue
        f["id"] = r["id"]
        f["sub_source"] = r["sub_source"]
        f["model"] = r["model"]
        f["is_human"] = r["is_human"]
        rows.append(f)
    return pd.DataFrame(rows)


def within_domain(df, cols, key="sub_source"):
    """Subtract the domain mean from each feature, so comparisons are internal."""
    out = df.copy()
    out[cols] = df.groupby(key)[cols].transform(lambda x: x - x.mean())
    return out


def separation(feat, d_by_id, cols, frac=0.1):
    """Standardised difference between the top and bottom decile of d_hat."""
    f = feat.copy()
    f["d_hat"] = f["id"].map(d_by_id)
    f = f.dropna(subset=["d_hat"])
    if len(f) < 60:
        return None
    lo_c, hi_c = f["d_hat"].quantile([frac, 1 - frac])
    lo, hi = f[f["d_hat"] <= lo_c], f[f["d_hat"] >= hi_c]
    rows = []
    for c in cols:
        a, b = lo[c].dropna(), hi[c].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"feature": c, "low": a.mean(), "high": b.mean(),
                     "sep": (b.mean() - a.mean()) / sd if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qphd", default=os.path.join(BASE, "results",
                                                   "coling_qphd_L201.csv.gz"))
    ap.add_argument("--pool", default=os.path.join(BASE, "..", "coling",
                                                   "pool.parquet"))
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    d = pd.read_csv(args.qphd)
    d = d[d["d_hat"] > 0]
    pool = pool[pool["id"].isin(d["id"].unique())]
    print(f"{len(pool)} текстов, {d['q'].nunique()} значений q", flush=True)

    feat = build_features(pool)
    cols = [c for c in feat.columns
            if c not in ("id", "sub_source", "model", "is_human")]
    feat.to_csv(os.path.join(BASE, "results", "coling_features.csv"), index=False)
    feat_wd = within_domain(feat, cols)
    print(f"{len(cols)} признаков", flush=True)

    out = []
    for (mode, q), g in d.groupby(["mode", "q"]):
        by_id = g.set_index("id")["d_hat"]
        for label, table in [("raw", feat), ("within_domain", feat_wd)]:
            s = separation(table, by_id, cols)
            if s is None:
                continue
            s["mode"], s["q"], s["control"] = mode, q, label
            out.append(s)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(os.path.join(BASE, "results", "coling_extremes_by_q.csv"),
               index=False)

    pd.set_option("display.width", 220)
    wd = res[res["control"] == "within_domain"]
    print("\n=== признаки, сильнее всего разделяющие крайние группы "
          "(within-domain, |sep|)\n")
    for mode in ["q_small", "q_large"]:
        s = wd[wd["mode"] == mode]
        piv = s.pivot_table(index="feature", columns="q", values="sep")
        qs = [c for c in [0.0, 0.2, 0.4, 0.6, 0.9] if c in piv.columns]
        piv = piv[qs]
        piv["|max|"] = piv.abs().max(1)
        print(f"--- {mode}")
        print(piv.sort_values("|max|", ascending=False).head(14).round(2).to_string())
        print()
    print(f"saved -> results/coling_extremes_by_q.csv")


if __name__ == "__main__":
    main()
