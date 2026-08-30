"""What drives the coarse band: how many word types, or how they are spread?

Two candidates came out of the screenplay and the essay, and they disagree on
that pair. The screenplay raised the type-token ratio and lowered the entropy
of word use, because it added rare types while piling mass onto a handful of
frequent ones; the essay raised both. If the coarse band followed the count of
types the two would move the same way, and they move 43 points apart.

Every measure is taken as a paired shift against the same source texts, and
each is tested with the other held fixed, because type count and entropy are
themselves strongly related.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from text_features import WORD, features

BASE = os.path.join(HERE, "..")
FILES = [
    "perturbed_texts_coling.json", "perturbed_texts_colingv2.json",
    "perturbed_texts_loops.json", "perturbed_texts_ngram.json",
    "perturbed_texts_expand.json", "perturbed_texts_hapax.json",
    "perturbed_texts_pc2neg.json", "perturbed_texts_composed.json",
    "perturbed_texts_rarify4.json", "coling_style_edits.json",
    "coling_v2_edits.json", "coling_hard_edits.json",
    "coling_pc2_probes.json", "coling_collapse_probes.json",
    "coling_down_more2.json", "coling_syn.json", "coling_script_essay.json",
]
FE = ["ttr", "hapax", "word_entropy", "zipf_slope", "top10_share",
      "content_ratio", "bigram_repeat", "mean_sent_len"]
BANDS = ["крупный", "мелкий", "средний"]


def partial(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    orig = {f"coling::{r.id}": r.text for r in pool.itertuples()}
    texts = {}
    for fn in FILES:
        p = os.path.join(BASE, "results", fn)
        if os.path.exists(p):
            with open(p) as f:
                for k, v in json.load(f).items():
                    texts.setdefault(k, v)

    base, rows = {}, []
    for name, by in texts.items():
        acc = []
        for k, v in list(by.items())[:30]:
            if k not in orig:
                continue
            if k not in base:
                base[k] = features(orig[k])
            f = features(v)
            acc.append({c: f.get(c, np.nan) - base[k].get(c, np.nan)
                        for c in FE})
        if len(acc) >= 8:
            rec = pd.DataFrame(acc).mean().to_dict()
            rec["перт"] = name
            rows.append(rec)
    F = pd.DataFrame(rows).set_index("перт")

    P = pd.read_csv(os.path.join(BASE, "results", "bands_perts.csv"),
                    index_col=0)
    J = F.join(P[BANDS]).dropna(subset=BANDS + ["ttr", "word_entropy"])
    J.round(4).to_csv(os.path.join(BASE, "results", "coarse_drivers.csv"))

    print(f"по {len(J)} пертурбациям\n")
    print(f"rho(ttr, энтропия) = {spearmanr(J['ttr'], J['word_entropy'])[0]:+.2f}\n")
    st = lambda p: "*" if p < 0.01 else " "
    print(f"{'признак':16s} {'крупный':>10s} {'мелкий':>10s} {'средний':>10s}")
    for c in FE:
        cells = []
        for b in BANDS:
            r, p = spearmanr(J[c], J[b])
            cells.append(f"{r:+.2f}{st(p)}")
        print(f"{c:16s} {cells[0]:>10s} {cells[1]:>10s} {cells[2]:>10s}")

    print("\n--- частные, каждый при фиксированном другом")
    print(f"{'':16s} {'крупный':>10s} {'мелкий':>10s} {'средний':>10s}")
    for a, b_ in [("word_entropy", "ttr"), ("ttr", "word_entropy")]:
        cells = []
        for band in BANDS:
            r, p = partial(J[a], J[band], J[b_])
            cells.append(f"{r:+.2f}{st(p)}")
        print(f"{a:11s}|{b_:4s} {cells[0]:>10s} {cells[1]:>10s} "
              f"{cells[2]:>10s}")
    print("\n* p < 0.01")

    pd.set_option("display.width", 200)
    print("\nкрайние по крупной полосе:")
    show = ["ttr", "word_entropy", "top10_share", "крупный"]
    print(J.sort_values("крупный")[show].head(5).round(3).to_string())
    print(J.sort_values("крупный")[show].tail(5).round(3).to_string())


if __name__ == "__main__":
    main()
