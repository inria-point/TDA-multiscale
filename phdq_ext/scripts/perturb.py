"""Mechanical text perturbations for the intervention study.

These need no model, so they are the controlled end of the experiment: each
one changes exactly one property, and the change is reversible on paper. The
LLM-driven rewrites (lexical diversity, number of ideas, register) go through
OpenRouter separately.

Two of these come from what the extreme-d texts looked like: low-d texts were
dialogue and verse with heavy line breaks and short sentences, high-d texts
were dense prose with long sentences. So sentence length and line breaks are
manipulated directly, to test whether they cause the effect or merely
accompany it.
"""
import random
import re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"\b\w+\b")


def sentences(text):
    return [s for s in SENT_SPLIT.split(text.strip()) if s.strip()]


CLAUSE_BREAK = {"and", "but", "or", "which", "while", "although", "because",
                "however", "though", "whereas", "since"}


def shorten_sentences(text, rng, min_words=4):
    """Split long sentences into short ones at clause boundaries.

    Cuts only after a comma or semicolon, or before a conjunction, so the
    result stays grammatical: this manipulates the number of sentence
    boundaries without scrambling syntax. A sentence with no clause boundary
    is left alone rather than cut mid-phrase.
    """
    out = []
    for sent in sentences(text):
        words = sent.split()
        chunks, cur = [], []
        for i, w in enumerate(words):
            starts_clause = (
                w.lower().strip(",;") in CLAUSE_BREAK
                and len(cur) >= min_words
                and len(words) - i >= min_words
            )
            if starts_clause:
                chunks.append(cur)
                cur = [w]
                continue
            cur.append(w)
            if w.endswith((",", ";")) and len(cur) >= min_words \
                    and len(words) - i - 1 >= min_words:
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)
        for ch in chunks:
            s = " ".join(ch).strip().rstrip(",;:")
            if not s:
                continue
            s = s[0].upper() + s[1:]
            if not s.endswith((".", "!", "?")):
                s += "."
            out.append(s)
    return " ".join(out)


def lengthen_sentences(text, rng, join=3):
    """Merge consecutive sentences into longer ones with commas.

    The inverse of shorten_sentences: same words, fewer boundaries.
    """
    sents = sentences(text)
    out = []
    for i in range(0, len(sents), join):
        group = sents[i : i + join]
        parts = []
        for j, s in enumerate(group):
            s = s.strip()
            if j > 0:
                s = s.rstrip(".!?")
                s = s[0].lower() + s[1:] if s else s
            else:
                s = s.rstrip(".!?")
            parts.append(s)
        merged = ", ".join(p for p in parts if p)
        if merged:
            out.append(merged + ".")
    return " ".join(out)


def add_linebreaks(text, rng, every=1):
    """Put a hard line break after every `every` sentences.

    Purely typographic: not a single character of the text itself changes.
    """
    sents = sentences(text)
    lines = [" ".join(sents[i : i + every]) for i in range(0, len(sents), every)]
    return "\n".join(lines)


def break_at_commas(text, rng):
    """Line break at every comma as well: verse-like layout, same words."""
    return re.sub(r",\s*", ",\n", add_linebreaks(text, rng, every=1))


def remove_linebreaks(text, rng):
    """Collapse all line breaks into single spaces."""
    return re.sub(r"\s*\n+\s*", " ", text).strip()


def shuffle_words(text, rng):
    """Shuffle all words, destroying syntax but keeping the exact vocabulary."""
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def shuffle_within_sentences(text, rng):
    """Shuffle words inside each sentence, keeping sentence boundaries."""
    out = []
    for sent in sentences(text):
        w = sent.split()
        rng.shuffle(w)
        out.append(" ".join(w))
    return " ".join(out)


def shuffle_sentences(text, rng):
    """Reorder whole sentences: local syntax intact, global coherence gone."""
    sents = sentences(text)
    rng.shuffle(sents)
    return " ".join(sents)


def add_typos(text, rng, rate=0.05):
    """Corrupt a fraction of words with a swap, drop, duplicate or replace."""
    words = text.split()
    for i, w in enumerate(words):
        core = WORD.search(w)
        if not core or len(core.group()) < 4 or rng.random() > rate:
            continue
        s = list(w)
        j = rng.randrange(len(s) - 1)
        op = rng.choice(["swap", "drop", "dup", "replace"])
        if op == "swap":
            s[j], s[j + 1] = s[j + 1], s[j]
        elif op == "drop":
            del s[j]
        elif op == "dup":
            s.insert(j, s[j])
        else:
            s[j] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        words[i] = "".join(s)
    return " ".join(words)


def lowercase(text, rng):
    """Drop capitalisation: removes a marker that correlated with low d."""
    return text.lower()


def strip_punctuation(text, rng):
    """Remove punctuation, keeping words and spaces.

    The mechanism behind its large effect is known: the deleted tokens sit 1.5x
    closer to their nearest neighbour than the rest (11.5 against 17.4), so
    they are the densest clusters in the cloud and the source of the shortest
    MST edges — exactly what the fine-scale regime (q_large) sums over.

    That explains the effect rather than diminishing it. The one thing to keep
    narrow is the claim: what is established is that removing punctuation
    raises measured d, not that punctuation as a feature of style governs
    dimension, since this also fragments words ("don't" -> "don t",
    "non-invasive" -> "non invasive") and erases sentence boundaries. A test of
    the wider claim would substitute or insert marks instead of deleting tokens.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def strip_punct_then_shuffle(text, rng):
    """Both local-structure manipulations at once, to test whether they share
    a channel.

    strip_punctuation and shuffle_words peak in the same regime (q_large near
    q = 0.7) and their effect profiles correlate at 0.61, which suggested both
    might act through one channel: dismantling local organisation. If so,
    applying both should land near the larger of the two rather than near
    their sum.

    Result: rejected in the working range. At q = 0.7 the pair gives +67.2%
    against +34.2% for the larger alone -- twice as much, which no single
    shared resource can produce. Under a multiplicative null (effects on a
    ratio quantity compose as products) the two are independent to within
    2 pp over q = 0.5-0.7. That last statement is null-dependent: a
    slope-additive null fits at q = 0 instead and overshoots badly later, and
    nothing in the geometry dictates which to use.

    Saturation does appear at the extreme: by q = 0.9 both nulls overpredict
    (by 27 and 57 pp), i.e. after the first manipulation the second has little
    left to destroy. That is robust to the choice of null, and is independent
    evidence that q_large at high q measures local organisation specifically.
    """
    return shuffle_words(strip_punctuation(text, rng), rng)


def strip_punct_then_shuffle_within(text, rng):
    """As above, with shuffling confined to sentences.

    Sentence boundaries are gone after stripping, so this collapses onto the
    plain version; kept for symmetry with the uncombined pair.
    """
    return shuffle_within_sentences(strip_punctuation(text, rng), rng)


PERTURBATIONS = {
    "identity": lambda t, rng: t,
    # sentence length -- suggested by the extreme-d texts
    "shorten_sentences": shorten_sentences,
    "lengthen_sentences": lengthen_sentences,
    # line breaks -- purely typographic
    "add_linebreaks": add_linebreaks,
    "break_at_commas": break_at_commas,
    "remove_linebreaks": remove_linebreaks,
    # word order
    "shuffle_words": shuffle_words,
    "shuffle_within_sentences": shuffle_within_sentences,
    "shuffle_sentences": shuffle_sentences,
    # surface noise
    "add_typos": add_typos,
    "lowercase": lowercase,
    "strip_punctuation": strip_punctuation,
    # composites: do the two local-structure manipulations share a channel?
    "strip_punct_then_shuffle": strip_punct_then_shuffle,
    "strip_punct_then_shuffle_within": strip_punct_then_shuffle_within,
}


def apply(name, text, seed=0, **kwargs):
    rng = random.Random(seed)
    return PERTURBATIONS[name](text, rng, **kwargs)
