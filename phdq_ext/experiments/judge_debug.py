"""Run the judge over a spanning sample and show what each property separates.

The sample is deliberately one source text carried through every major
perturbation, plus a few untouched human texts. Holding the source fixed means
a difference between two rows is attributable to the operation rather than to
the text, which is what a prompt needs to be judged on: whether it separates
the things it was written to separate.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS, build, parse
from openrouter import complete

BASE = os.path.join(HERE, "..")
# Sonnet returns finish_reason "refusal" with zero output tokens on the most
# damaged inputs -- scrambled word order and collapsed vocabularies trigger it
# reliably -- and those are precisely the texts the judge exists to score. A
# single model must rate every text or the scores are not comparable, so the
# whole run uses the model that does not decline.
MODEL = "gemini-3.6-flash"

WANT = [
    "loop_local_3", "loop_phrase", "loop_tail_early", "echo_p25",
    "collapse_vocab_10", "expand_vocab_100", "shuffle_words",
    "ngram_2", "ngram_4", "splice_sentences", "marker_p3",
    "hapax_swap_wide", "inject_topic", "syn_none", "syn_max",
    "style_literary", "essay_one_topic", "script_events",
    "style_record_fields", "style_catechism", "style_code",
    "style_equations", "style_anaphora", "syntax_monotone",
    "syntax_varied", "syntax_simple", "syntax_complex",
    "context_locked", "terms_long", "add_punctuation", "strip_punctuation",
    "add_typos", "lowercase", "ideas_down_v2", "lexdiv_down_more",
    "literary_echo25",
]
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


def collect():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    orig = {f"coling::{r.id}": r.text for r in pool.itertuples()}
    texts = {}
    for fn in FILES:
        p = os.path.join(BASE, "results", fn)
        if os.path.exists(p):
            with open(p) as f:
                for k, v in json.load(f).items():
                    texts.setdefault(k, v)
    # a source text present in as many perturbations as possible
    counts = {}
    for name in WANT:
        for k in texts.get(name, {}):
            counts[k] = counts.get(k, 0) + 1
    key = max(counts, key=counts.get)
    print(f"опорный текст: {key}, покрыт {counts[key]} пертурбациями из "
          f"{len(WANT)}\n")

    sample = [("человек (опорный)", orig[key])]
    for i, k in enumerate(list(orig)[:2]):
        if k != key:
            sample.append((f"человек {i + 1}", orig[k]))
    for name in WANT:
        t = texts.get(name, {}).get(key)
        if t:
            sample.append((name, t))
    return sample


def score(name, text):
    out = complete(build(text), model=MODEL, provider="apiyi",
                   max_tokens=1200, temperature=0)
    return name, parse(out)


def main():
    sample = collect()
    rows = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(score, n, t): n for n, t in sample}
        for f in as_completed(futs):
            n, d = f.result()
            rows[n] = d
    D = pd.DataFrame([{k: v[0] for k, v in rows.get(n, {}).items()}
                      | {"вариант": n} for n, _ in sample]).set_index("вариант")
    R = pd.DataFrame([{k: v[1] for k, v in rows.get(n, {}).items()}
                      | {"вариант": n} for n, _ in sample]).set_index("вариант")
    D.to_csv(os.path.join(BASE, "results", "judge_debug.csv"))
    R.to_csv(os.path.join(BASE, "results", "judge_debug_reasons.csv"))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 60)
    print(D.to_string())
    print("\n" + "=" * 74)
    for k, p in PROPS.items():
        if k not in D:
            continue
        s = D[k].dropna().sort_values()
        if s.empty:
            print(f"\n{p['ru']} ({k}): все значения пусты")
            continue
        na = D[k].isna().sum()
        print(f"\n{p['ru']} ({k}): разброс {s.min():g}..{s.max():g}, "
              f"медиана {s.median():g}"
              + (f", null у {na}" if na else ""))
        print("  низ:  " + ", ".join(f"{i}={v:g}" for i, v in s.head(4).items()))
        mid = s.iloc[len(s) // 2 - 1:len(s) // 2 + 2]
        print("  сер.: " + ", ".join(f"{i}={v:g}" for i, v in mid.items()))
        print("  верх: " + ", ".join(f"{i}={v:g}" for i, v in s.tail(4).items()))


if __name__ == "__main__":
    main()
