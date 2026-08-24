"""A broad battery of text features, for generating hypotheses about qPHD.

The point is coverage: rather than guess in advance which property drives the
dimension at a given q, we measure many and let the extreme groups say which
ones separate. Features that survive become perturbation candidates.

Every rate is computed on a fixed budget of words, because rates such as the
type-token ratio fall as a text lengthens (Heaps' law) and would otherwise
report length rather than the property named.
"""
import math
import re
from collections import Counter

import numpy as np

WORD = re.compile(r"[A-Za-z']+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
BUDGET = 220  # words; every text in the pool has at least this many

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "there", "here", "he", "she",
    "they", "we", "you", "i", "his", "her", "their", "our", "your", "my", "not",
    "no", "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "must", "than", "then", "so", "such",
}
SUBORDINATORS = {
    "which", "that", "because", "although", "while", "whereas", "since",
    "whose", "whom", "though", "unless", "until", "whether", "wherein",
    "whereby", "after", "before", "when", "where",
}
HEDGES = {
    "may", "might", "could", "perhaps", "possibly", "likely", "generally",
    "often", "usually", "typically", "somewhat", "relatively", "arguably",
    "seems", "suggests", "appears", "potentially", "approximately",
}
CONNECTIVES = {
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "additionally", "consequently", "overall", "finally", "firstly",
    "secondly", "meanwhile", "nevertheless", "besides", "similarly",
}
PRONOUNS_1_2 = {"i", "me", "my", "mine", "we", "us", "our", "you", "your"}


def _entropy(counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in counts if c)


def _zipf_slope(counts):
    """Slope of log frequency against log rank: how skewed the word use is."""
    f = np.sort(np.array(sorted(counts, reverse=True), dtype=float))[::-1]
    f = f[f > 0]
    if f.size < 8:
        return np.nan
    r = np.arange(1, f.size + 1)
    b, _ = np.polyfit(np.log(r), np.log(f), 1)
    return float(b)


def _repeat_rate(seq, n):
    """Share of n-grams that are not unique."""
    if len(seq) < n + 1:
        return np.nan
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    c = Counter(grams)
    return 1.0 - len(c) / len(grams)


def _burstiness(x):
    """Coefficient of variation of a sequence: how uneven it is."""
    x = np.asarray(x, dtype=float)
    if x.size < 2 or x.mean() == 0:
        return np.nan
    return float(x.std() / x.mean())


def features(text, ranks=None, budget=BUDGET):
    """All scalar features of one text. `ranks` maps word -> corpus frequency rank."""
    all_words = WORD.findall(text.lower())
    if len(all_words) < 30:
        return {}
    words = all_words[:budget]
    n = len(words)
    counts = Counter(words)
    content = [w for w in words if w not in FUNCTION_WORDS]
    sents_all = [s for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    # sentence lengths, restricted to the same budget of words
    slen, used = [], 0
    for s in sents_all:
        k = len(WORD.findall(s.lower()))
        if used >= budget:
            break
        slen.append(min(k, budget - used))
        used += k
    slen = [x for x in slen if x > 0]
    chars = text[: sum(len(w) + 1 for w in words) + 200]
    nc = max(1, len(chars))

    f = {
        # --- lexical richness
        "ttr": len(counts) / n,
        "hapax": sum(1 for c in counts.values() if c == 1) / n,
        "top1_share": counts.most_common(1)[0][1] / n,
        "top10_share": sum(c for _, c in counts.most_common(10)) / n,
        "word_entropy": _entropy(list(counts.values())),
        "zipf_slope": _zipf_slope(list(counts.values())),
        "content_ratio": len(content) / n,
        "content_ttr": len(set(content)) / max(1, len(content)),
        # --- repetition at the phrase level
        "bigram_repeat": _repeat_rate(words, 2),
        "trigram_repeat": _repeat_rate(words, 3),
        # --- word shape
        "mean_word_len": float(np.mean([len(w) for w in words])),
        "long_word_rate": sum(1 for w in words if len(w) >= 8) / n,
        # --- sentences
        "mean_sent_len": float(np.mean(slen)) if slen else np.nan,
        "sd_sent_len": float(np.std(slen)) if len(slen) > 1 else np.nan,
        "sent_len_burst": _burstiness(slen),
        "n_sents_per_100w": 100 * len(slen) / n,
        # --- syntax markers
        "subordinator_rate": sum(1 for w in words if w in SUBORDINATORS) / n,
        "function_rate": sum(1 for w in words if w in FUNCTION_WORDS) / n,
        "connective_rate": sum(1 for w in words if w in CONNECTIVES) / n,
        "hedge_rate": sum(1 for w in words if w in HEDGES) / n,
        "pron12_rate": sum(1 for w in words if w in PRONOUNS_1_2) / n,
        # --- surface and typography
        "comma_rate": chars.count(",") / nc,
        "punct_rate": sum(c in ".,;:!?" for c in chars) / nc,
        "quote_rate": sum(c in "\"'“”" for c in chars) / nc,
        "paren_rate": sum(c in "()[]" for c in chars) / nc,
        "dash_rate": sum(c in "-—–" for c in chars) / nc,
        "digit_rate": sum(c.isdigit() for c in chars) / nc,
        "upper_rate": sum(c.isupper() for c in chars) / nc,
        "linebreak_per_100w": 100 * chars.count("\n") / n,
        # list markers anywhere, not only at a line start: many generations run
        # the list inside a paragraph ("because: 1. ... 2. ..."), which an
        # anchored pattern misses entirely
        "bullet_rate": len(re.findall(
            r"(?:^\s*|(?<=[.:;)]\s)|(?<=\n))(?:[-*\u2022\u2023]|\d+[.)]|[a-z][.)])\s",
            text, flags=re.M)) / n * 100,
        "md_heading_rate": len(re.findall(r"^\s*#{1,6}\s|\*\*[^*]+\*\*",
                                          text, flags=re.M)) / n * 100,
        "question_rate": chars.count("?") / max(1, len(slen)),
        "exclaim_rate": chars.count("!") / max(1, len(slen)),
    }
    if ranks:
        kn = [ranks[w] for w in words if w in ranks]
        if kn:
            f["mean_log_rank"] = float(np.mean(np.log(kn)))
            f["rare_word_rate"] = float(np.mean([r > 5000 for r in kn]))
    return f


def embedding_features(embeds, tokens=None):
    """Geometry of the token cloud that is not the dimension itself."""
    from scipy.spatial.distance import pdist, squareform

    e = embeds[:400]
    d = squareform(pdist(e))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    finite = d[np.isfinite(d)]
    centred = e - e.mean(0)
    norms = np.linalg.norm(centred, axis=1)
    # participation ratio: effective number of directions the cloud occupies
    s = np.linalg.svd(centred, compute_uv=False)
    v = s ** 2
    pr = float(v.sum() ** 2 / (v ** 2).sum()) if v.sum() > 0 else np.nan
    return {
        "nn_dist_mean": float(nn.mean()),
        "nn_dist_cv": float(nn.std() / nn.mean()),
        "pair_dist_mean": float(finite.mean()),
        "pair_dist_cv": float(finite.std() / finite.mean()),
        "cloud_radius_cv": float(norms.std() / norms.mean()),
        "participation_ratio": pr,
    }
