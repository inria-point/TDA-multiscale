"""What text property does each principal component actually track?

The map places a perturbation by what it did to the dimension profile. This
script asks the converse question with numbers instead of by eye: measure 33
surface and lexical features in the perturbed texts, take the paired change
against the untouched source, and correlate that change across the 39 kept
perturbations with each principal coordinate.

A correlation here is over *perturbations*, not over texts: 39 points, each one
a perturbation's mean feature shift versus its mean position. It says "the
perturbations that move a text left on PC1 are the ones that raise repetition",
which is a statement about the axis, not yet about causality in a single text.

Spearman is used throughout: several features (repeat rates, bullet rate) are
zero for most perturbations and huge for a few, so a linear correlation would
be decided by one point.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import taxonomy as T
from text_features import WORD, features

BASE = os.path.join(os.path.dirname(__file__), "..")


def load_texts():
    """perturbation -> {text_id: text}, taking the first file that has it."""
    out = {}
    for fn in ["perturbed_texts_coling.json", "perturbed_texts_colingv2.json",
               "perturbed_texts_loops.json", "perturbed_texts_ngram.json",
               "perturbed_texts_ngramwide.json", "perturbed_texts_hard.json",
               "coling_style_edits.json", "coling_v2_edits.json",
               "coling_hard_edits.json", "coling_temps.json"]:
        path = os.path.join(BASE, "results", fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            if k in T.KEEP and k not in out:
                out[k] = v
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    originals = {f"coling::{r.id}": r.text for r in pool.itertuples()}
    counts = Counter()
    for t in pool[pool["is_human"]]["text"]:
        counts.update(WORD.findall(t.lower()))
    ranks = {w: i + 1 for i, (w, _) in enumerate(counts.most_common())}

    # local plausibility: share of adjacent word pairs never seen in the human
    # corpus. Separates "locally fine but globally repetitive" (a stutter: every
    # pair is a real phrase) from "locally broken" (shuffled words, a substituted
    # word, a spliced n-gram), which no rate metric distinguishes.
    seen = set()
    for t in pool[pool["is_human"]]["text"]:
        w = WORD.findall(t.lower())
        seen.update(zip(w, w[1:]))
    print(f"биграмм в корпусе: {len(seen)}")

    def bigram_oov(text):
        w = WORD.findall(text.lower())
        if len(w) < 2:
            return np.nan
        pairs = list(zip(w, w[1:]))
        return sum(1 for b in pairs if b not in seen) / len(pairs)

    texts = load_texts()
    base_cache = {}
    rows = []
    for pname, by_id in texts.items():
        deltas = []
        for key, new in by_id.items():
            k = key if key in originals else f"coling::{key}"
            old = originals.get(k)
            if old is None:
                continue
            if k not in base_cache:
                base_cache[k] = features(old, ranks)
                base_cache[k]["bigram_oov"] = bigram_oov(old)
            fo, fn_ = base_cache[k], features(new, ranks)
            if not fo or not fn_:
                continue
            fo = dict(fo, bigram_oov=bigram_oov(old))
            fn_ = dict(fn_, bigram_oov=bigram_oov(new))
            deltas.append({c: fn_[c] - fo[c] for c in fo if c in fn_})
        if len(deltas) >= 20:
            rec = pd.DataFrame(deltas).mean().to_dict()
            rec["perturbation"] = T.KEEP[pname]
            rec["n"] = len(deltas)
            rows.append(rec)
    F = pd.DataFrame(rows).set_index("perturbation")
    F.round(4).to_csv(os.path.join(BASE, "results", "feature_deltas.csv"))
    print(f"признаки измерены для {len(F)} пертурбаций "
          f"(медиана {int(F['n'].median())} текстов)")

    P = pd.read_csv(os.path.join(BASE, "results", "pert_map.csv"), index_col=0)
    J = F.join(P[["PC1", "PC2", "PC3"]], how="inner")
    feats = [c for c in F.columns if c != "n"]

    def partial(x, y, z):
        """Spearman of x and y with the influence of z taken out: correlate the
        residuals of the rank-regressions of x on z and y on z."""
        rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return spearmanr(ex, ey)

    out = []
    for c in feats:
        rec = {"feature": c}
        for pc in ["PC1", "PC2", "PC3"]:
            sub = J[[c, pc]].dropna()
            r, p = spearmanr(sub[c], sub[pc])
            rec[pc] = r
            rec[f"p_{pc}"] = p
        for pc in ["PC2", "PC3"]:
            sub = J[[c, pc, "PC1"]].dropna()
            r, p = partial(sub[c], sub[pc], sub["PC1"])
            rec[f"{pc}|PC1"] = r
            rec[f"p_{pc}|PC1"] = p
        out.append(rec)
    R = pd.DataFrame(out).set_index("feature")
    R.to_csv(os.path.join(BASE, "results", "property_axes.csv"))

    pd.set_option("display.width", 200)
    for pc in ["PC1", "PC2", "PC3", "PC2|PC1", "PC3|PC1"]:
        s = R[[pc, f"p_{pc}"]].copy()
        s = s.reindex(s[pc].abs().sort_values(ascending=False).index).head(9)
        note = " — с исключённым влиянием PC1" if "|" in pc else ""
        print(f"\n=== {pc}{note}")
        print(s.rename(columns={pc: "rho", f"p_{pc}": "p"}).round(3)
              .to_string())


if __name__ == "__main__":
    main()
