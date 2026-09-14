"""The same geometry battery on the LLM rewrites, which are already on disk.

pert_geometry.py covers the mechanical perturbations: they are cheap, exact and
reproducible, but every one of them is an operation on the surface of the text.
The interesting half of the catalogue is the other one -- rewrites where a model
was asked to carry fewer ideas, to stay on one topic, to never repeat a content
word, to weave twenty foreign words in so that they belong. Those texts were
generated once and stored, so putting them through the same battery costs
nothing but the encoder.

Two of them matter more than the rest.

  inject_topic     the designed twin of hapax_swap_wide. The same twenty
                   foreign words, but written into the text by a model instead
                   of substituted mechanically. Lexical diversity, entropy and
                   top-10 share agree to the third decimal, and the coarse band
                   goes +2.1 instead of -19.2. If the cell coordinates are flat
                   for both, then whatever separates them is exactly the thing
                   the model is missing.

  ideas_down       fewer distinct thoughts at the same length: -26.1 coarse,
                   -43.0 middle, the largest purely semantic drop in the
                   catalogue, and nothing mechanical about it at all.

Two mechanical conditions are run alongside as anchors, from the same stored
files, so this table and the mechanical one can be laid against each other
despite the different text sample.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from coling_data import human_texts
from edge_taxonomy import corpus_ranks
from embedder import Embedder
from pert_geometry import measure
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 50))
SEEDS = int(os.environ.get("SEEDS", 3))
OUT = os.path.join(BASE, "results", os.environ.get("OUT", "pert_geometry_llm.csv"))

# label, key in the stored json, file holding it
PERTS = [
    # the designed pair with hapax_swap_wide
    ("тема вписана", "inject_topic", "coling_syn.json"),
    # meaning removed
    ("меньше мыслей", "ideas_down_v2", "coling_v2_edits.json"),
    ("одна тема", "topics_down_v2", "coling_v2_edits.json"),
    ("беднее словарь", "lexical_diversity_down_s", "coling_v2_edits.json"),
    ("без синонимов", "syn_none", "coling_syn.json"),
    ("термин в той же обёртке", "context_locked", "coling_pc2_probes.json"),
    # meaning added
    ("каждое предложение о своём", "topics_up_hard", "coling_hard_edits.json"),
    ("богаче словарь", "lexical_diversity_up_s", "coling_v2_edits.json"),
    ("ни одного слова дважды", "syn_max", "coling_syn.json"),
    ("редкие синонимы", "words_rare_s", "coling_v2_edits.json"),
    ("художественный стиль", "style_literary_T1.3", "coling_temps.json"),
    # form repeated, meaning kept
    ("анафора", "style_anaphora", "coling_collapse_probes.json"),
    ("поле: значение", "style_record_fields", "coling_collapse_probes.json"),
    # mechanical anchors, to tie this table to the other one
    ("перемешивание", "shuffle_words", "perturbed_texts_coling.json"),
    ("чужие хапаксы", "hapax_swap_wide", "perturbed_texts_hapax.json"),
]


def main():
    src = dict(human_texts(120, min_words=280))
    store = {}
    for label, key, fname in PERTS:
        with open(os.path.join(BASE, "results", fname)) as f:
            d = json.load(f)
        store[label] = d[key]
        print(f"{label:26s} {len(d[key]):4d} текстов  ({key})")

    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)
    print(f"\n{len(PERTS) + 1} условий x {N_TEXTS} текстов x {SEEDS} зёрен",
          flush=True)

    rows, done = [], 0
    for key, text in src.items():
        if done >= N_TEXTS:
            break
        got = False
        for label, txt in [("исходный", text)] + [
                (lab, store[lab].get(key)) for lab, _, _ in PERTS]:
            if txt is None:
                continue
            for seed in range(SEEDS):
                r = measure(txt, emb, ranks, u_sink, seed=1000 + seed,
                            key=f"pgl_{label}_{key[-8:]}")
                if r is None:
                    continue
                rows.append({"текст": key, "условие": label, "зерно": seed,
                             **r})
                got = True
        done += int(got)
        if done % 5 == 0 and got:
            print(f"  {done} текстов, {len(rows)} строк", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT, index=False)
    print(f"\n{OUT}: {len(D)} строк, "
          f"{D.groupby('условие')['текст'].nunique().min()}–"
          f"{D.groupby('условие')['текст'].nunique().max()} текстов на условие")


if __name__ == "__main__":
    main()
