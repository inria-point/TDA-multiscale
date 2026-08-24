"""COLING 2025 MGT detection corpus: loading and stratified sampling.

The corpus spans 41 generators from opt_125m to gpt4o across 25 domains, which
is what makes it usable both for hypothesis generation (a wide spread of texts)
and for the later diagnosis of individual generators.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
PARQUET = os.path.join(BASE, "..", "coling", "dev.parquet")


def load(min_words=260):
    """Texts long enough to survive tokenisation to >= L tokens."""
    d = pd.read_parquet(PARQUET)
    d["n_words"] = d["text"].str.split().str.len()
    d = d[d["n_words"] >= min_words].reset_index(drop=True)
    d["is_human"] = d["model"] == "human"
    return d


def sample_pool(d, n=2000, seed=0, max_share_domain=0.12, min_per_model=8):
    """A pool spread over domains and generators.

    Domain is capped so no single one dominates the extreme groups, and every
    generator with enough long texts is represented, so that the same pool can
    later be reused to characterise individual models.
    """
    rng = np.random.default_rng(seed)
    cap = int(n * max_share_domain)
    picked = []

    # floor per model first, so rare generators are not lost to the domain cap
    for model, g in d.groupby("model"):
        k = min(min_per_model, len(g))
        picked.append(g.sample(k, random_state=int(rng.integers(1e9))))
    out = pd.concat(picked)

    remaining = d.drop(out.index)
    for domain, g in remaining.groupby("sub_source"):
        have = int((out["sub_source"] == domain).sum())
        k = min(max(0, cap - have), len(g), max(1, n // d["sub_source"].nunique()))
        if k:
            out = pd.concat([out, g.sample(k, random_state=int(rng.integers(1e9)))])

    if len(out) > n:
        out = out.sample(n, random_state=seed)
    return out.reset_index(drop=True)


def human_texts(n=120, min_words=280, seed=0, pool_path=None):
    """Human texts from the pool, as (key, text), spread over domains.

    These are the base for the stage-2 interventions: the perturbations are
    applied to human writing, so that whatever a style does to the dimension is
    not confounded with the generator that produced the source.
    """
    import pandas as pd

    path = pool_path or os.path.join(BASE, "..", "coling", "pool.parquet")
    p = pd.read_parquet(path)
    p = p[p["is_human"] & (p["n_words"] >= min_words)]
    per = max(1, n // max(1, p["sub_source"].nunique()))
    out = (p.groupby("sub_source", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), per), random_state=seed)))
    if len(out) < n:
        rest = p.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)),
                                          random_state=seed)])
    out = out.head(n)
    return [(f"coling::{r['id']}", r["text"]) for _, r in out.iterrows()]
