"""Canonical names, functional groups and the pruned set of perturbations.

Fifty-two perturbations accumulated over the project, many of them variants
tried and kept. A perturbation earns a place in the final set only if it is
distinguishable from every other one on *both* counts:

  * function -- what it does to the text, judged by construction, not by result
  * position -- its coordinates in the PC1-PC3 basis, by Euclidean distance

Two perturbations from the same functional group that land within DEDUP_R of
each other measure the same thing twice, so one is kept and the other dropped.
Two from *different* groups landing close is not redundancy: that is the
result, and both are kept (add_typos and words_rare are 18 apart; capitalize
and the news-with-dates style are 8 apart).

DEDUP_R = 30 is about a fifth of the mean profile length (164) and roughly
twice the spread between two runs of the same perturbation.

Names are `group/what_was_done_with_which_parameters`, with no spaces so they
stay legible as point labels on a plot, and with the parameters spelled out so
a dose series reads as a series.
"""

DEDUP_R = 30.0

# group -> (russian label, colour)
GROUPS = {
    "loop":    ("зацикливание", "#b7791f"),
    "vocab":   ("сужение словаря", "#975a16"),
    "deplete": ("LLM: обеднение содержания", "#805ad5"),
    "enrich":  ("LLM: обогащение содержания", "#2f855a"),
    "rarity":  ("LLM: частотность слов", "#38a169"),
    "style":   ("LLM: жанровый регистр", "#2b6cb0"),
    "shuffle": ("разрушение порядка", "#c53030"),
    "splice":  ("чужие предложения", "#dd6b20"),
    "ngram":   ("н-граммная генерация", "#6b46c1"),
    "syntax":  ("границы и служебные слова", "#4a5568"),
    "surface": ("типографика", "#a0aec0"),
}

# raw name -> canonical name.  Order inside a group is the dose order.
KEEP = {
    # --- degenerate repetition. tail: keep% of intact prose, then a fragment
    #     of N words repeated to fill the length. echo: every span repeated.
    "loop_local_1":        "loop/echo6w_x2",
    "loop_local_3":        "loop/echo6w_x4",
    "loop_tail_word":      "loop/tail_keep60_unit1w",
    "loop_tail_short":     "loop/tail_keep60_unit3w",
    "loop_tail_mid":       "loop/tail_keep40_unit3w",
    "loop_tail_drift":     "loop/tail_keep40_unit3w_drift",
    "loop_tail_early":     "loop/tail_keep20_unit3w",
    "loop_phrase":         "loop/phrase12w_x4",
    # --- content words replaced by the k most frequent words of the same text
    "collapse_vocab_60":   "vocab/top60",
    "collapse_vocab_25":   "vocab/top25",
    "collapse_vocab_10":   "vocab/top10",
    # the mirror: every content word replaced by a fresh one of the same
    # length and frequency band, so type diversity moves and rarity does not
    "expand_vocab_50":     "vocab/expand50",
    "expand_vocab_100":    "vocab/expand100",
    # --- LLM asked for less content
    "lexical_diversity_down_s": "deplete/lexdiv_down",
    "ideas_down_v2":            "deplete/ideas_down",
    # --- LLM asked for more content
    "lexical_diversity_up_s":   "enrich/lexdiv_up",
    "topics_up_hard":           "enrich/topics_up_hard",
    # --- LLM asked to shift word frequency
    "words_common_s":      "rarity/common_words",
    "words_rare_s":        "rarity/rare_words",
    # --- LLM asked for a genre register
    "style_simplified":             "style/simplified",
    "style_dialogue_interjections": "style/dialogue_interj",
    "style_news_telegraphic":       "style/telegraphic",
    "style_scientific_terms":       "style/scientific_terms",
    "style_bulletin_abbreviations": "style/bulletin_abbrev",
    "style_news_dates_suffixes":    "style/news_dates",
    "style_literary":               "style/literary_T0.7",
    # the temperature series is kept although it duplicates the base prompt by
    # the pruning rule: PC2-negative coverage is the scarce resource in this
    # set, and these are the only non-loop points that lower PC2 at all
    "style_literary_T1.3":          "style/literary_T1.3",
    "style_literary_T1.7":          "style/literary_T1.7",
    # --- word order destroyed
    "shuffle_words":       "shuffle/words",
    # --- sentences taken from unrelated texts
    "splice_pairs":        "splice/every2nd_sent",
    "splice_sentences":    "splice/all_sents",
    # --- sampled from an n-gram model of the human corpus
    "ngram_4":             "ngram/n4",
    "ngram_2":             "ngram/n2",
    # --- sentence boundaries and function words
    "burst_alternate":          "syntax/burst_1to4",
    "drop_sentence_boundaries": "syntax/no_sent_bounds",
    "drop_function_words":      "syntax/drop_funcwords",
    # --- typography only
    "lowercase":           "surface/lowercase",
    "to_numbered_list":    "surface/numbered_list",
    "add_digits":          "surface/digits_add",
    "add_punctuation":     "surface/punct_add",
    "strip_punctuation":   "surface/punct_strip",
    "add_typos":           "surface/typos",
    "capitalize_terms":    "surface/caps_terms",
}

# dropped name -> (kept name it duplicates, distance in PC space)
DROPPED = {
    "burst_flatten":             ("burst_alternate", 3.9),
    "shuffle_within_sentences":  ("shuffle_words", 4.2),
    "lexical_diversity_up_hard": ("lexical_diversity_up_s", 11.7),
    "ideas_up_v2":               ("lexical_diversity_up_s", 12.4),
    "ideas_up_hard":             ("lexical_diversity_up_s", 19.6),
    "topics_up_v2":              ("topics_up_hard", 24.7),
    "topics_down_v2":            ("lexical_diversity_down_s", 5.1),
    "strip_digits":              ("lowercase", 9.9),
    "ngram_wide_2":              ("ngram_2", 15.9),
    "ngram_wide_3":              ("ngram_2", 16.8),
    "ngram_3":                   ("ngram_2", 29.3),
}


def group_of(canonical):
    return canonical.split("/", 1)[0]


def label_of(canonical):
    return canonical.split("/", 1)[1]
