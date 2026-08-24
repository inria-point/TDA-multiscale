"""Stage 1b: what separates the extremes once the repetition axis is removed.

The raw extremes of d are dominated by degenerate generations — loops at one
end (hapax 0.22) and near-random word salad at the other (hapax 0.72), with
human text almost absent from both. Whatever those extremes teach is about
generation pathologies, which are already known to move d.

To reach secondary properties, two things are done. Every feature is
residualised on the repetition axis (the first principal component of the
diversity family), so that what remains is variation the dominant axis does not
explain. And the analysis is repeated on a healthy subset — human text plus
modern instruction-tuned models — where degenerate generations are rare.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from exp_extremes_by_q import separation, within_domain

BASE = os.path.join(os.path.dirname(__file__), "..")

DIVERSITY = ["hapax", "ttr", "content_ttr", "zipf_slope", "word_entropy",
             "top10_share", "top1_share", "bigram_repeat", "trigram_repeat"]
MODERN = ["human", "gpt4", "gpt4o", "gpt-3.5-turbo", "gpt-35", "llama3-8b",
          "llama3-70b", "mixtral-8x7b", "gemma-7b-it", "gemma2-9b-it",
          "text-davinci-003", "cohere", "davinci"]


def repetition_axis(feat, cols):
    """First PC of the diversity family, standardised."""
    x = feat[cols].apply(lambda c: (c - c.mean()) / (c.std() or 1)).fillna(0)
    u, s, vt = np.linalg.svd(x.values - x.values.mean(0), full_matrices=False)
    pc = u[:, 0] * s[0]
    # orient so that higher means more diverse
    if np.corrcoef(pc, feat["hapax"].fillna(feat["hapax"].mean()))[0, 1] < 0:
        pc = -pc
    return pc


def residualise(feat, cols, axis):
    """Remove the linear part explained by `axis` from every feature."""
    out = feat.copy()
    a = (axis - axis.mean()) / (axis.std() or 1)
    for c in cols:
        v = feat[c].astype(float)
        ok = v.notna()
        if ok.sum() < 30:
            continue
        beta = np.polyfit(a[ok], v[ok], 1)[0]
        out.loc[ok, c] = v[ok] - beta * a[ok]
    return out


def report(d, feat, cols, title, top=12):
    out = []
    for (mode, q), g in d.groupby(["mode", "q"]):
        s = separation(feat, g.set_index("id")["d_hat"], cols)
        if s is None:
            continue
        s["mode"], s["q"] = mode, q
        out.append(s)
    if not out:
        return None
    res = pd.concat(out, ignore_index=True)
    pd.set_option("display.width", 220)
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    for mode in ["q_small", "q_large"]:
        s = res[res["mode"] == mode]
        piv = s.pivot_table(index="feature", columns="q", values="sep")
        qs = [c for c in [0.0, 0.2, 0.4, 0.6, 0.9] if c in piv.columns]
        piv = piv[qs]
        piv["|max|"] = piv.abs().max(1)
        print(f"\n--- {mode}")
        print(piv.sort_values("|max|", ascending=False).head(top).round(2).to_string())
    return res


def main():
    feat = pd.read_csv(os.path.join(BASE, "results", "coling_features.csv"))
    d = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    d = d[d["d_hat"] > 0]
    cols = [c for c in feat.columns
            if c not in ("id", "sub_source", "model", "is_human")]

    axis = repetition_axis(feat, [c for c in DIVERSITY if c in feat])
    feat["rep_axis"] = axis
    keep = [c for c in cols if c not in DIVERSITY]

    res_a = report(d, within_domain(residualise(feat, keep, axis), keep),
                   cols=keep,
                   title="A. Весь пул, ось повторяемости вычтена, "
                         "внутри домена")

    healthy = feat[feat["model"].isin(MODERN)]
    dh = d[d["id"].isin(healthy["id"])]
    print(f"\nздоровое подмножество: {len(healthy)} текстов, "
          f"{healthy['model'].nunique()} моделей, "
          f"human {healthy['is_human'].mean():.0%}")
    res_b = report(dh, within_domain(healthy.copy(), cols), cols=cols,
                   title="B. Только human и современные модели, все признаки")

    for name, r in [("residual", res_a), ("healthy", res_b)]:
        if r is not None:
            r.to_csv(os.path.join(BASE, "results",
                                  f"coling_extremes_{name}.csv"), index=False)
    feat.to_csv(os.path.join(BASE, "results", "coling_features.csv"), index=False)


if __name__ == "__main__":
    main()
