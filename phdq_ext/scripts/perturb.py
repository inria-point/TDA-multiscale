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


# ---------------------------------------------------------------------------
# Stage-2 mechanical perturbations, from the COLING extreme-group hypotheses.
# Defects are included deliberately: the point is to tell a dimension drop
# caused by a style from one caused by a generation failure.
# ---------------------------------------------------------------------------

def _regroup(text, pattern):
    """Join existing sentences in groups of the given sizes.

    Only real sentence boundaries are used, so grammar is untouched and the
    only thing that changes is how evenly length is distributed. Cutting at a
    fixed word count instead would break phrases, and destroying syntax has a
    large effect of its own that would swamp the one under test.
    """
    sents = sentences(text)
    out, i, k = [], 0, 0
    while i < len(sents):
        n = pattern[k % len(pattern)]
        group = sents[i:i + n]
        if not group:
            break
        merged = group[0].rstrip(".!?")
        for extra in group[1:]:
            extra = extra.rstrip(".!?").strip()
            if extra:
                merged += ", " + extra[0].lower() + extra[1:]
        out.append(merged + ".")
        i += n
        k += 1
    return " ".join(out)


def burst_alternate(text, rng):
    """Alternate one short sentence with a long merged one: uneven lengths (H1)."""
    return _regroup(text, [1, 4])


def burst_flatten(text, rng, group=2.5):
    """Merge sentences into groups of equal *word count*, not equal count.

    Grouping a fixed number of sentences does not flatten anything: the sums
    inherit the variation of their parts. Packing greedily towards a target
    word count does, and the target is set to the same average as
    burst_alternate so the pair differs in unevenness alone, not in mean
    sentence length.
    """
    sents = sentences(text)
    if not sents:
        return text
    lens = [len(x.split()) for x in sents]
    target = group * (sum(lens) / len(lens))
    out, cur, cur_w = [], [], 0
    for s_, w in zip(sents, lens):
        cur.append(s_)
        cur_w += w
        if cur_w >= target:
            out.append(cur)
            cur, cur_w = [], 0
    if cur:
        out.append(cur)
    joined = []
    for group_ in out:
        merged = group_[0].rstrip(".!?")
        for extra in group_[1:]:
            extra = extra.rstrip(".!?").strip()
            if extra:
                merged += ", " + extra[0].lower() + extra[1:]
        joined.append(merged + ".")
    return " ".join(joined)


def drop_sentence_boundaries(text, rng):
    """Remove sentence-final punctuation: the text becomes one run-on.

    A defect, and the one the mean_sent_len feature was actually detecting in
    the corpus (flan_t5 outputs with no full stops at all).
    """
    return re.sub(r"\s+", " ", re.sub(r"[.!?]+(\s|$)", " ", text)).strip()


def loop_phrase(text, rng, span=12, repeats=4):
    """Repeat one phrase over and over, as a degenerate decoder does.

    The classic failure at the low-dimension end of the corpus: the same clause
    restated with small variations until the length is filled.
    """
    words = text.split()
    if len(words) < span * 2:
        return text
    start = rng.randrange(0, max(1, len(words) - span))
    phrase = " ".join(words[start:start + span])
    out, i = [], 0
    while i < len(words):
        out.extend(words[i:i + span])
        for _ in range(repeats):
            out.append(phrase)
        i += span * (repeats + 1)
    return " ".join(out[:len(words)])


def add_punctuation(text, rng, rate=0.18):
    """Insert commas and semicolons at plausible clause boundaries (H4).

    The opposite of strip_punctuation, and the direction the corpus could not
    supply: it only ever showed punctuation being lost.
    """
    out = []
    for sent in sentences(text):
        w = sent.split()
        new = []
        for i, x in enumerate(w):
            new.append(x)
            if (0 < i < len(w) - 2 and not x.endswith((",", ";", ".", ":"))
                    and rng.random() < rate):
                new[-1] = x + (";" if rng.random() < 0.25 else ",")
        out.append(" ".join(new))
    return " ".join(out)


def strip_digits(text, rng):
    """Remove digits, keeping everything else (H7, subtractive direction)."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", " ", text)).strip()


def add_digits(text, rng, rate=0.09):
    """Insert plausible numbers, dates and percentages (H7, additive)."""
    units = ["%", " million", " per cent", "", "", ""]
    out = []
    for w in text.split():
        out.append(w)
        if rng.random() < rate:
            if rng.random() < 0.4:
                out.append(f"({rng.randrange(1950, 2025)})")
            else:
                out.append(f"{rng.randrange(2, 99)}{rng.choice(units)}")
    return " ".join(out)


def capitalize_terms(text, rng, rate=0.12):
    """Upper-case some content words, as acronyms and proper names do (H8)."""
    out = []
    for w in text.split():
        core = WORD.search(w)
        if core and len(core.group()) > 3 and rng.random() < rate:
            w = w.upper()
        out.append(w)
    return out and " ".join(out) or text


def to_numbered_list(text, rng, per_item=2):
    """Reformat the same sentences as a numbered list (H6)."""
    sents = sentences(text)
    items = [" ".join(sents[i:i + per_item])
             for i in range(0, len(sents), per_item)]
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(items))


def drop_function_words(text, rng, rate=0.75):
    """Delete most function words: a telegraphic surface (H5)."""
    out = [w for w in text.split()
           if WORD.search(w) is None
           or WORD.search(w).group() not in FUNCTION_WORDS
           or rng.random() > rate]
    return " ".join(out)


FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "there", "here",
}

PERTURBATIONS.update({
    "burst_alternate": burst_alternate,
    "burst_flatten": burst_flatten,
    "drop_sentence_boundaries": drop_sentence_boundaries,
    "loop_phrase": loop_phrase,
    "add_punctuation": add_punctuation,
    "strip_digits": strip_digits,
    "add_digits": add_digits,
    "capitalize_terms": capitalize_terms,
    "to_numbered_list": to_numbered_list,
    "drop_function_words": drop_function_words,
})


# ---------------------------------------------------------------------------
# Degeneracy, closer to what a failing decoder actually produces.
#
# The original loop_phrase splices one globally chosen phrase through the whole
# text, cruder than reality: real loops are local and start partway in. The
# corpus shows the pattern -- flan_t5 emitting "Select option one / two /
# three ... salik account" with a drifting detail, GLM130B repeating "However,
# Tesco said it Tesco said it". So the collapse is parameterised instead, by
# where it begins, how long the repeated unit is and how often it repeats,
# which also makes a magnitude series possible.
# ---------------------------------------------------------------------------

def loop_tail(text, rng, keep=0.4, unit=3, drift=0.0):
    """Normal prose for the first `keep` of the text, then a collapse into
    repeating a short unit until the original length is filled.

    unit  -- words in the repeated fragment (1 = a single word)
    drift -- chance of re-picking the fragment, so the repetition wanders
             rather than being literally identical, as real loops do
    """
    words = text.split()
    n = len(words)
    cut = max(unit + 1, int(n * keep))
    out = list(words[:cut])
    frag = words[max(0, cut - unit):cut]
    while len(out) < n:
        if drift and rng.random() < drift:
            j = rng.randrange(0, max(1, cut - unit))
            frag = words[j:j + unit]
        out.extend(frag)
    return " ".join(out[:n])


def loop_local(text, rng, span=6, repeats=2):
    """Repeat what was just said, throughout: a drifting local stutter."""
    words = text.split()
    out, i = [], 0
    while i < len(words):
        chunk = words[i:i + span]
        out.extend(chunk)
        for _ in range(repeats):
            out.extend(chunk)
        i += span
    return " ".join(out[:len(words)])


def _tail(**kw):
    return lambda text, rng: loop_tail(text, rng, **kw)


def _local(**kw):
    return lambda text, rng: loop_local(text, rng, **kw)


PERTURBATIONS.update({
    "loop_tail_word": _tail(keep=0.6, unit=1),
    "loop_tail_short": _tail(keep=0.6, unit=3),
    "loop_tail_mid": _tail(keep=0.4, unit=3),
    "loop_tail_early": _tail(keep=0.2, unit=3),
    "loop_tail_drift": _tail(keep=0.4, unit=3, drift=0.25),
    "loop_local_1": _local(span=6, repeats=1),
    "loop_local_3": _local(span=6, repeats=3),
})


def collapse_vocabulary(text, rng, keep=60):
    """Rewrite the text using only its own `keep` most frequent content words.

    The mechanical counterpart of asking a model to lower lexical diversity.
    Asking did almost nothing on this corpus -- type-token ratio moved 0.60 to
    0.59 -- because a model will not strip terminology out of an abstract. Here
    the strength is a parameter rather than a request: every content word
    outside the kept vocabulary is replaced by the nearest kept word by length,
    so the token count and the function-word skeleton survive.
    """
    words = text.split()
    cores = [WORD.search(w) for w in words]
    content = [c.group().lower() for c, w in zip(cores, words)
               if c and c.group().lower() not in FUNCTION_WORDS
               and len(c.group()) > 3]
    if not content:
        return text
    from collections import Counter

    freq = [w for w, _ in Counter(content).most_common(keep)]
    if not freq:
        return text
    by_len = sorted(freq, key=len)
    out = []
    for w, c in zip(words, cores):
        if not c:
            out.append(w)
            continue
        low = c.group().lower()
        if low in FUNCTION_WORDS or len(low) <= 3 or low in freq:
            out.append(w)
            continue
        # nearest kept word by length keeps the surface rhythm intact
        repl = min(by_len, key=lambda k: (abs(len(k) - len(low)), k))
        out.append(w.replace(c.group(), repl))
    return " ".join(out)


PERTURBATIONS.update({
    "collapse_vocab_60": lambda t, rng: collapse_vocabulary(t, rng, keep=60),
    "collapse_vocab_50": lambda t, rng: collapse_vocabulary(t, rng, keep=50),
    "collapse_vocab_25": lambda t, rng: collapse_vocabulary(t, rng, keep=25),
    "collapse_vocab_10": lambda t, rng: collapse_vocabulary(t, rng, keep=10),
})


# ---------------------------------------------------------------------------
# Locally coherent nonsense, by sampling from an n-gram model of the corpus.
#
# The right edge of PC1 is held by the old base models (bloom_7b, opt_2.7b),
# roughly twice as far out as the strongest rewrite we can obtain. What they
# produce is locally plausible and globally incoherent, which is exactly what
# an n-gram model generates: every adjacent pair of words is real English
# because it was observed, while the text as a whole wanders across topics and
# so carries corpus-wide vocabulary rather than one document's.
#
# The order is the dial: order 2 wanders almost every word, order 5 copies long
# stretches verbatim and stays nearly coherent.
# ---------------------------------------------------------------------------

_NGRAM_CACHE = {}


def _build_ngram(order, corpus_texts, tag=""):
    key = (order, len(corpus_texts), tag)
    if key in _NGRAM_CACHE:
        return _NGRAM_CACHE[key]
    from collections import defaultdict

    table = defaultdict(list)
    for t in corpus_texts:
        w = t.split()
        for i in range(len(w) - order):
            table[tuple(w[i:i + order - 1])].append(w[i + order - 1])
    _NGRAM_CACHE[key] = table
    return table


def _corpus():
    """Human texts of the COLING pool, as the source of n-gram statistics."""
    import os

    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "..", "..", "coling",
                        "pool.parquet")
    p = pd.read_parquet(path)
    return p[p["is_human"]]["text"].tolist()


def ngram_generate(text, rng, order=3):
    """A text of the same length, sampled from the corpus n-gram model.

    Seeded from the opening of the source so the beginning stays on topic and
    the drift is visible against it.
    """
    table = _build_ngram(order, _corpus())
    words = text.split()
    n = len(words)
    out = list(words[:order - 1])
    for _ in range(n - len(out)):
        key = tuple(out[-(order - 1):])
        choices = table.get(key)
        if not choices:
            key = rng.choice(list(table.keys()))
            out.extend(key)
            continue
        out.append(rng.choice(choices))
    return " ".join(out[:n])


PERTURBATIONS.update({
    "ngram_2": lambda t, rng: ngram_generate(t, rng, order=2),
    "ngram_3": lambda t, rng: ngram_generate(t, rng, order=3),
    "ngram_4": lambda t, rng: ngram_generate(t, rng, order=4),
})


def _corpus_full(min_words=200, cap=20000):
    """The whole COLING dev split, not just the sampled pool.

    A wider vocabulary than the 650 pooled human texts can supply: the pooled
    n-grams reached PC1 = 262 where opt_2.7b sits at 454, and the shortfall is
    the diversity of the source material rather than of the mechanism.
    """
    import os

    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "..", "..", "coling",
                        "dev.parquet")
    d = pd.read_parquet(path, columns=["text", "model"])
    d = d[d["model"] == "human"]
    d = d[d["text"].str.split().str.len() >= min_words]
    return d["text"].head(cap).tolist()


def ngram_generate_wide(text, rng, order=3):
    """As ngram_generate, but with statistics from the full corpus."""
    table = _build_ngram(order, _corpus_full(), tag="wide")
    words = text.split()
    n = len(words)
    out = list(words[:order - 1])
    for _ in range(n - len(out)):
        key = tuple(out[-(order - 1):])
        choices = table.get(key)
        if not choices:
            key = rng.choice(list(table.keys()))
            out.extend(key)
            continue
        out.append(rng.choice(choices))
    return " ".join(out[:n])


PERTURBATIONS.update({
    "ngram_wide_2": lambda t, rng: ngram_generate_wide(t, rng, order=2),
    "ngram_wide_3": lambda t, rng: ngram_generate_wide(t, rng, order=3),
})


# ---------------------------------------------------------------------------
# Incoherence at the discourse scale rather than the word scale.
#
# n-gram generation breaks the text inside the sentence: adjacent word pairs
# are real but nothing is a statement. Splicing does the opposite -- every
# sentence is a real, fully coherent sentence written by a person, and only the
# sequence is meaningless. Comparing the two says whether the dimension
# responds to incoherence as such or to the scale at which it occurs.
# ---------------------------------------------------------------------------

_BANK = {}


def _sentence_bank(same_domain=None):
    import os

    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "..", "..", "coling",
                        "pool.parquet")
    p = pd.read_parquet(path)
    p = p[p["is_human"]]
    if same_domain:
        p = p[p["sub_source"] == same_domain]
    bank = []
    for t in p["text"]:
        bank.extend(s for s in sentences(t) if 6 <= len(s.split()) <= 45)
    return bank


def splice_sentences(text, rng, same_domain=None):
    """Fill the original length with sentences drawn from unrelated texts."""
    key = same_domain or "*"
    if key not in _BANK:
        _BANK[key] = _sentence_bank(same_domain)
    bank = _BANK[key]
    if not bank:
        return text
    target = len(text.split())
    out, n = [], 0
    while n < target:
        out.append(bank[rng.randrange(len(bank))])
        n += len(out[-1].split())
    return " ".join(" ".join(out).split()[:target])


def splice_pairs(text, rng):
    """Alternate a sentence of the original with a foreign one: half the
    discourse survives, so this sits between untouched and fully spliced."""
    if "*" not in _BANK:
        _BANK["*"] = _sentence_bank(None)
    bank = _BANK["*"]
    own = sentences(text)
    target = len(text.split())
    out, n, i = [], 0, 0
    while n < target and own:
        s = own[i % len(own)] if i % 2 == 0 else bank[rng.randrange(len(bank))]
        out.append(s)
        n += len(s.split())
        i += 1
    return " ".join(" ".join(out).split()[:target])


PERTURBATIONS.update({
    "splice_sentences": splice_sentences,
    "splice_pairs": splice_pairs,
})


# ---------------------------------------------------------------------------
# The mirror of collapse_vocabulary: instead of forcing the content words down
# onto the k most frequent ones, give every one of them a word that occurs
# nowhere else in the text.
#
# Replacements are matched on length *and* on frequency rank. Length alone
# would let the draw wander into rarer vocabulary than the original, and word
# rarity is what PC3 tracks; matching the rank band keeps the mean log rank of
# the text where it was, so the perturbation isolates type diversity.
#
# Function words, punctuation, word order and length in tokens are untouched,
# exactly as in collapse_vocabulary, so the pair differs in one thing only.
# ---------------------------------------------------------------------------

_VOCAB = {}


def _vocab_pool():
    """Corpus content words bucketed by length, each bucket in rank order."""
    if _VOCAB:
        return _VOCAB
    from collections import Counter, defaultdict

    counts = Counter()
    for t in _corpus():
        counts.update(WORD.findall(t.lower()))
    # a word must occur in at least three different documents to count as
    # vocabulary: the tail of a single-document count is typos, fragments and
    # identifiers, and drawing "rare words" from it would measure noise
    df = Counter()
    for t in _corpus():
        df.update(set(WORD.findall(t.lower())))
    rank = {w: i for i, (w, _) in enumerate(counts.most_common())}
    by_len = defaultdict(list)
    for w, r in sorted(rank.items(), key=lambda kv: kv[1]):
        if w in FUNCTION_WORDS or len(w) <= 3 or df[w] < 3:
            continue
        by_len[len(w)].append(w)
    _VOCAB.update(by_len=dict(by_len), rank=rank,
                  ranks={L: [rank[w] for w in ws] for L, ws in by_len.items()})
    return _VOCAB


def expand_vocabulary(text, rng, share=1.0, window=60):
    """Replace content words with fresh ones of the same length and rank band.

    share  -- fraction of content words to replace
    window -- how far along the rank-ordered bucket the draw may reach
    """
    import bisect

    pool = _vocab_pool()
    by_len, ranks, rank = pool["by_len"], pool["ranks"], pool["rank"]
    words = text.split()
    cores = [WORD.search(w) for w in words]
    own = {c.group().lower() for c in cores if c}
    used = set()
    out = []
    for w, c in zip(words, cores):
        if not c:
            out.append(w)
            continue
        low = c.group().lower()
        if low in FUNCTION_WORDS or len(low) <= 3 or rng.random() > share:
            out.append(w)
            continue
        L = len(low) if by_len.get(len(low)) else len(low) - 1
        bucket = by_len.get(L)
        if not bucket:
            out.append(w)
            continue
        rs = ranks[L]
        i0 = bisect.bisect_left(rs, rank.get(low, rs[len(rs) // 2]))
        cand = [x for x in bucket[max(0, i0 - window):i0 + window]
                if x not in own and x not in used]
        if not cand:
            out.append(w)
            continue
        repl = cand[rng.randrange(len(cand))]
        used.add(repl)
        if c.group()[0].isupper():
            repl = repl.capitalize()
        out.append(w.replace(c.group(), repl))
    return " ".join(out)


PERTURBATIONS.update({
    "expand_vocab_100": lambda t, rng: expand_vocabulary(t, rng, share=1.0),
    "expand_vocab_50": lambda t, rng: expand_vocabulary(t, rng, share=0.5),
})


# ---------------------------------------------------------------------------
# Two routes to a lower PC2 that are not loops.
#
# PC2 is the fine scale minus the coarse one, and in the whole set it only
# goes substantially negative for loops, which collapse the fine scale. Two
# non-loops reach it weakly and by the opposite mechanism -- asking for
# scientific terminology or for rarer words raises the *coarse* scale while
# the fine one barely moves (+3.9 against -3.7, and +3.6 against +0.8). One
# type of change is not a basis for a conclusion, so both routes get a
# mechanical perturbation with a dose.
#
# map_vocabulary is the route through the coarse scale. Unlike
# expand_vocabulary it maps *types*, not occurrences: every occurrence of a
# word gets the same replacement, so the repetition geometry that the fine
# scale is made of survives untouched and only the rarity of the vocabulary
# changes.
#
# interleave_marker is the route through the fine scale without repeating
# anything from the text: the original is kept whole and in order, and a
# constant foreign token is inserted every k words, which plants identical
# contexts at a controlled period.
# ---------------------------------------------------------------------------


def map_vocabulary(text, rng, band="rare", share=1.0, head=0.10, tail=0.30):
    """Replace content word *types* consistently with words from a rank band.

    band  -- "rare" draws from the tail of the frequency order, "common" from
             the head; length is matched so word length does not move
    share -- fraction of types remapped
    """
    pool = _vocab_pool()
    by_len = pool["by_len"]
    words = text.split()
    cores = [WORD.search(w) for w in words]
    types = []
    for c in cores:
        if not c:
            continue
        low = c.group().lower()
        if low not in FUNCTION_WORDS and len(low) > 3 and low not in types:
            types.append(low)
    if not types:
        return text

    mapping, used = {}, set(types)
    for low in types:
        if rng.random() > share:
            continue
        L = len(low) if by_len.get(len(low)) else len(low) - 1
        bucket = by_len.get(L)
        if not bucket:
            continue
        n = len(bucket)
        cand = (bucket[int((1 - tail) * n):] if band == "rare"
                else bucket[:max(1, int(head * n))])
        cand = [x for x in cand if x not in used]
        if not cand:
            continue
        repl = cand[rng.randrange(len(cand))]
        mapping[low] = repl
        used.add(repl)

    out = []
    for w, c in zip(words, cores):
        if not c or c.group().lower() not in mapping:
            out.append(w)
            continue
        repl = mapping[c.group().lower()]
        if c.group()[0].isupper():
            repl = repl.capitalize()
        out.append(w.replace(c.group(), repl))
    return " ".join(out)


def interleave_marker(text, rng, period=6, marker="item"):
    """Insert one constant token every `period` words, keeping the text whole.

    Nothing from the text is repeated: the identical contexts come from a
    foreign token planted at a fixed period, so the fine scale can be moved
    without the text degenerating into a repeat of itself.
    """
    words = text.split()
    out = []
    for i, w in enumerate(words):
        out.append(w)
        if (i + 1) % period == 0:
            out.append(marker)
    return " ".join(out[:len(words)])


def echo_partial(text, rng, span=6, repeats=1, share=0.25):
    """loop_local applied to only a share of the chunks, chosen at random.

    A full echo overshoots: it lands at PC1 -189 where the generators that
    share its PC2 sit at -54 to -106. Echoing part of the text moves the point
    along the same ray without leaving it.
    """
    words = text.split()
    out, i = [], 0
    while i < len(words):
        chunk = words[i:i + span]
        out.extend(chunk)
        if rng.random() < share:
            for _ in range(repeats):
                out.extend(chunk)
        i += span
    return " ".join(out[:len(words)])


PERTURBATIONS.update({
    "echo_p25": lambda t, rng: echo_partial(t, rng, share=0.25),
    "echo_p50": lambda t, rng: echo_partial(t, rng, share=0.50),
})


PERTURBATIONS.update({
    "rarify_types": lambda t, rng: map_vocabulary(t, rng, band="rare"),
    "rarify_types_50": lambda t, rng: map_vocabulary(t, rng, band="rare",
                                                     share=0.5),
    "commonize_types": lambda t, rng: map_vocabulary(t, rng, band="common"),
    "marker_p3": lambda t, rng: interleave_marker(t, rng, period=3),
    "marker_p6": lambda t, rng: interleave_marker(t, rng, period=6),
    "marker_p12": lambda t, rng: interleave_marker(t, rng, period=12),
})


# ---------------------------------------------------------------------------
# Swapping the once-used vocabulary.
#
# The long edges, which carry the coarse scale, are 64.8% edges with a hapax
# at one end -- against 0.4% at the short end. A word used once cannot form a
# short edge at all: a short edge needs a near-duplicate, and a hapax has no
# second occurrence of itself, while two *different* content words are never
# close (0.2% of short edges). So the once-used vocabulary is the coarse
# scale's own material, and the repeated vocabulary is the fine scale's.
#
# The perturbation acts on that material and nothing else: every content word
# occurring exactly once is replaced, every word occurring more than once is
# left alone. The count of hapax is therefore unchanged and so is the
# repetition skeleton; what changes is how far apart the once-used words are
# in meaning.
#
# The pair is the test. hapax_swap_wide draws each replacement from a
# different domain of the corpus, so the injected words span many subjects;
# hapax_swap_local draws them all from one domain, so they are as coherent as
# the originals. Same operation, same counts, same lengths -- only the spread
# differs. If the coarse scale is about the spread of once-used vocabulary,
# the first should raise it and the second should not.
# ---------------------------------------------------------------------------

_DOMAIN_VOCAB = {}


def _domain_vocab():
    """Content words of the human corpus, grouped by domain and by length."""
    if _DOMAIN_VOCAB:
        return _DOMAIN_VOCAB
    import os
    from collections import Counter, defaultdict

    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "..", "..", "coling",
                        "pool.parquet")
    p = pd.read_parquet(path)
    p = p[p["is_human"]]
    for dom, g in p.groupby("sub_source"):
        counts = Counter()
        for t in g["text"]:
            counts.update(WORD.findall(t.lower()))
        by_len = defaultdict(list)
        for w, c in counts.items():
            if w in FUNCTION_WORDS or len(w) <= 3 or c < 2:
                continue
            by_len[len(w)].append(w)
        if sum(len(v) for v in by_len.values()) > 200:
            _DOMAIN_VOCAB[dom] = dict(by_len)
    return _DOMAIN_VOCAB


def swap_hapax(text, rng, wide=True, share=1.0):
    """Replace once-used content words with once-used words from elsewhere.

    wide  -- draw each replacement from a different domain, or all from one
    share -- fraction of the hapax replaced
    """
    vocab = _domain_vocab()
    doms = sorted(vocab)
    if not doms:
        return text
    home = doms[rng.randrange(len(doms))]

    from collections import Counter

    words = text.split()
    cores = [WORD.search(w) for w in words]
    counts = Counter(c.group().lower() for c in cores if c)
    used = set(counts)
    out = []
    for w, c in zip(words, cores):
        low = c.group().lower() if c else None
        if (low is None or low in FUNCTION_WORDS or len(low) <= 3
                or counts[low] != 1 or rng.random() > share):
            out.append(w)
            continue
        dom = doms[rng.randrange(len(doms))] if wide else home
        bucket = vocab[dom].get(len(low)) or vocab[dom].get(len(low) - 1)
        if not bucket:
            out.append(w)
            continue
        cand = [x for x in bucket if x not in used]
        if not cand:
            out.append(w)
            continue
        repl = cand[rng.randrange(len(cand))]
        used.add(repl)
        if c.group()[0].isupper():
            repl = repl.capitalize()
        out.append(w.replace(c.group(), repl))
    return " ".join(out)


PERTURBATIONS.update({
    "hapax_swap_wide": lambda t, rng: swap_hapax(t, rng, wide=True),
    "hapax_swap_wide_50": lambda t, rng: swap_hapax(t, rng, wide=True,
                                                    share=0.5),
    "hapax_swap_local": lambda t, rng: swap_hapax(t, rng, wide=False),
})
