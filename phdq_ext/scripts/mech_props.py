"""Ten counted properties, chosen to complement the judge rather than repeat it.

The judge rates what has to be read: whether words fit their places, whether
the text advances, whether it could have been written by someone. These are
the things that can be counted instead, and they are counted because the judge
is unreliable at exactly this -- asked how repetitive a text is, a model gives
an impression; a trigram tally gives a number.

Two of them exist because a pair of perturbations that no lexical statistic
separates behave oppositely. Collapsing a vocabulary to ten words and echoing
every phrase both produce heavy repetition, and both raise trigram_repeat; but
the echo puts its copies side by side and the collapse scatters them through
the document, and only the fine scale of the dimension notices. adjacent_share
is that distinction made countable.
"""
import re
from collections import Counter

import numpy as np

WORD = re.compile(r"[a-z']+")
TOKEN = re.compile(r"\S+")
SENT = re.compile(r"(?<=[.!?])\s+")
PUNCT = re.compile(r"[^\w\s]")
NEAR = 15

NAMES = {
    "ttr": "разнообразие словаря",
    "hapax_share": "доля однократных слов",
    "word_entropy": "энтропия словоупотребления",
    "mean_log_rank": "редкость лексики",
    "trigram_repeat": "повтор триграмм",
    "adjacent_share": "доля повторов вплотную",
    "punct_variety": "разнообразие пунктуации",
    "sent_len_cv": "разброс длин предложений",
    "opening_uniformity": "однотипность зачинов",
    "proper_noun_rate": "доля имён собственных",
}


def measure(text, ranks=None):
    words = WORD.findall(text.lower())
    if len(words) < 30:
        return {}
    counts = Counter(words)
    n = len(words)
    p = np.array(list(counts.values()), float) / n
    tri = list(zip(words, words[1:], words[2:]))
    tri_counts = Counter(tri)

    # where the copies of a repeated word stand relative to each other: a
    # stutter and a ten-word vocabulary repeat equally often and differently
    pos = {}
    for i, w in enumerate(words):
        pos.setdefault(w, []).append(i)
    near = total = 0
    for ii in pos.values():
        if len(ii) < 2:
            continue
        gaps = np.diff(ii)
        near += int((gaps < NEAR).sum())
        total += len(gaps)

    raw = TOKEN.findall(text)
    sents = [s for s in SENT.split(text.strip()) if s.split()]
    lens = np.array([len(s.split()) for s in sents], float)
    openings = Counter(" ".join(s.split()[:2]).lower() for s in sents)
    inner = [t for t in raw[1:] if t[:1].isupper()]

    out = {
        "ttr": len(counts) / n,
        "hapax_share": sum(1 for v in counts.values() if v == 1) / len(counts),
        "word_entropy": float(-(p * np.log(p)).sum()),
        "trigram_repeat": (sum(v for v in tri_counts.values() if v > 1)
                           / max(1, len(tri))),
        "adjacent_share": near / total if total else np.nan,
        "punct_variety": len(set(PUNCT.findall(text))),
        "sent_len_cv": float(lens.std() / lens.mean()) if len(lens) > 1
        else np.nan,
        "opening_uniformity": (max(openings.values()) / len(sents)
                               if sents else np.nan),
        "proper_noun_rate": len(inner) / max(1, len(raw)),
    }
    if ranks:
        known = [ranks[w] for w in words if w in ranks]
        out["mean_log_rank"] = (float(np.mean(np.log(known))) if known
                                else np.nan)
    return out


def corpus_ranks(texts):
    """Word -> frequency rank over a reference corpus, rank 1 the commonest."""
    counts = Counter()
    for t in texts:
        counts.update(WORD.findall(t.lower()))
    return {w: i + 1 for i, (w, _) in enumerate(counts.most_common())}
