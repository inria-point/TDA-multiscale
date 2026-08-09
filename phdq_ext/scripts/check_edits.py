"""Did the rewrite actually do what it was asked?

Every LLM perturbation claims to move one property. If the model did not
execute the instruction, the qPHD result would be a measurement of nothing.
So each perturbation is checked against a metric that tracks its target
property, plus a length check, before any dimension is computed.

Word rarity is scored against frequencies of the corpus itself, so no external
frequency list is needed.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import GENRES, load

BASE = os.path.join(os.path.dirname(__file__), "..")
WORD = re.compile(r"[a-z']+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SUBORD = {"which", "that", "because", "although", "while", "whereas", "since",
          "whose", "whom", "though", "unless", "whereby", "wherein"}

# perturbation -> (metric, expected direction)
#
# Ideas and topics are scored by idea_spread, not by any word-count statistic.
# A word-based metric fails here: to hold the length while carrying fewer
# ideas the model restates the same claim in different words, which *raises*
# lexical diversity. idea_spread instead measures how far apart the sentences
# of a text sit in embedding space.
EXPECTED = {
    "lexical_diversity_up": ("ttr", +1),
    "lexical_diversity_down": ("ttr", -1),
    "ideas_up": ("idea_spread", +1),
    "ideas_down": ("idea_spread", -1),
    "ideas_up_v2": ("idea_spread", +1),
    "ideas_down_v2": ("idea_spread", -1),
    "topics_up": ("idea_spread", +1),
    "topics_down": ("idea_spread", -1),
    "words_common": ("mean_log_rank", -1),
    "words_rare": ("mean_log_rank", +1),
    "syntax_complex": ("mean_sent_len", +1),
    "syntax_simple": ("mean_sent_len", -1),
    "shorten_sentences": ("mean_sent_len", -1),
    "lengthen_sentences": ("mean_sent_len", +1),
    "add_linebreaks": ("linebreaks", +1),
    "break_at_commas": ("linebreaks", +1),
    "remove_linebreaks": ("linebreaks", -1),
    "add_typos": ("oov_rate", +1),
    "lowercase": ("upper_frac", -1),
    "strip_punctuation": ("punct_frac", -1),
}


def sentence_vectors(text, emb, cap=40, min_words=4):
    """Mean-pooled embedding per sentence, or None if too few sentences."""
    sents = [s for s in SENT_SPLIT.split(text.strip())
             if len(s.split()) >= min_words][:cap]
    if len(sents) < 3:
        return None
    return np.stack([emb.embed(s).mean(0) for s in sents])


def idea_spread(vecs, centre):
    """1 - mean pairwise cosine between sentences: higher = more distinct ideas.

    Vectors are centred first. ModernBERT's space is anisotropic — raw cosines
    between any two sentences sit near 0.93 — so without centring the measure
    has almost no dynamic range and the manipulations look like no-ops.
    """
    w = vecs - centre
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    sim = w @ w.T
    iu = np.triu_indices(len(w), 1)
    return 1.0 - float(sim[iu].mean())


def corpus_ranks():
    """Word -> frequency rank over all human texts (rank 1 = most frequent)."""
    counts = Counter()
    for genre in GENRES:
        for _, text in load(genre, "human"):
            counts.update(WORD.findall(text.lower()))
    return {w: i + 1 for i, (w, _) in enumerate(counts.most_common())}, counts


def metrics(text, ranks, vocab, budget=None):
    """Text statistics; `budget` truncates to a fixed number of words first.

    Rate metrics such as type-token ratio fall as a text gets longer (Heaps'
    law), so comparing a rewrite against its source only makes sense on an
    equal number of words. n_words is reported from the untruncated text.
    """
    all_words = WORD.findall(text.lower())
    n_total = len(all_words)
    if n_total == 0:
        return {}
    words = all_words[:budget] if budget else all_words
    n = len(words)
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    counts = Counter(words)
    content = [w for w in words if len(w) > 3]
    known = [ranks[w] for w in words if w in ranks]
    return {
        "n_words": n_total,
        "ttr": len(counts) / n,
        "hapax": sum(1 for c in counts.values() if c == 1) / n,
        "content_types": len(set(content)) / n,
        "mean_log_rank": float(np.mean(np.log(known))) if known else np.nan,
        "mean_sent_len": n / max(1, len(sents)),
        "linebreaks": text.count("\n") / max(1, len(sents)),
        "oov_rate": sum(1 for w in words if w not in vocab) / n,
        "upper_frac": sum(c.isupper() for c in text) / max(1, len(text)),
        "punct_frac": sum(c in ",;:—-()\"'.!?" for c in text) / max(1, len(text)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", default=os.path.join(BASE, "results",
                                                    "llm_edits.json"))
    ap.add_argument("--source", default="human")
    args = ap.parse_args()

    with open(args.edits) as f:
        edits = json.load(f)
    ranks, counts = corpus_ranks()
    vocab = set(counts)
    originals = {}
    for genre in GENRES:
        for text_id, text in load(genre, args.source):
            originals[f"{genre}::{text_id}"] = text

    # sentence vectors only where a semantic metric is actually needed
    needs_sem = {p for p, (m, _) in EXPECTED.items() if m == "idea_spread"}
    vecs, pool = {}, []
    if any(p in edits for p in needs_sem):
        from embedder import Embedder

        emb = Embedder()
        for pname in needs_sem & set(edits):
            for key, new in edits[pname].items():
                if key not in originals:
                    continue
                vb = vecs.setdefault(("orig", key),
                                     sentence_vectors(originals[key], emb))
                va = sentence_vectors(new, emb)
                vecs[(pname, key)] = va
                pool += [v for v in (vb, va) if v is not None]
    centre = np.concatenate(pool).mean(0) if pool else None

    rows = []
    for pname, by_id in edits.items():
        for key, new in by_id.items():
            old = originals.get(key)
            if old is None:
                continue
            genre, text_id = key.split("::", 1)
            budget = min(len(WORD.findall(old.lower())),
                         len(WORD.findall(new.lower())))
            if budget < 50:
                continue
            mo = metrics(old, ranks, vocab, budget=budget)
            mn = metrics(new, ranks, vocab, budget=budget)
            if not mo or not mn:
                continue
            row = {"perturbation": pname, "genre": genre, "text_id": text_id}
            for k in mo:
                row[f"{k}_before"] = mo[k]
                row[f"{k}_after"] = mn[k]
            vb, va = vecs.get(("orig", key)), vecs.get((pname, key))
            if centre is not None and vb is not None and va is not None:
                row["idea_spread_before"] = idea_spread(vb, centre)
                row["idea_spread_after"] = idea_spread(va, centre)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("нет данных")
        return
    df.to_csv(os.path.join(BASE, "results", "edit_fidelity_raw.csv"), index=False)

    out = []
    for pname, g in df.groupby("perturbation"):
        metric, want = EXPECTED.get(pname, (None, 0))
        rec = {
            "perturbation": pname,
            "n": len(g),
            "len_ratio": (g["n_words_after"] / g["n_words_before"]).median(),
        }
        if metric and f"{metric}_before" in g:
            pair = g[[f"{metric}_before", f"{metric}_after"]].dropna()
            if pair.empty:
                out.append(rec)
                continue
            before, after = pair[f"{metric}_before"], pair[f"{metric}_after"]
            delta = after - before
            rec.update({
                "metric": metric,
                "before": before.mean(),
                "after": after.mean(),
                "rel_change": (after.mean() - before.mean()) / abs(before.mean())
                if before.mean() else np.nan,
                # share of texts that moved the intended way
                "hit_rate": float((np.sign(delta) == want).mean()),
            })
        out.append(rec)
    res = pd.DataFrame(out).sort_values("hit_rate", na_position="last")
    res.to_csv(os.path.join(BASE, "results", "edit_fidelity.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nвыполнение инструкций: hit_rate — доля текстов, "
          "сдвинувшихся в нужную сторону\n")
    print(res.round(3).to_string(index=False))
    bad = res[res["hit_rate"] < 0.7].dropna(subset=["hit_rate"])
    if len(bad):
        print(f"\nслабо исполнены (hit_rate < 0.7): "
              f"{', '.join(bad['perturbation'])}")
    off = res[(res["len_ratio"] < 0.8) | (res["len_ratio"] > 1.25)]
    if len(off):
        print(f"\nдлина уехала больше чем на 20-25%: "
              f"{', '.join(off['perturbation'])}")


if __name__ == "__main__":
    main()
