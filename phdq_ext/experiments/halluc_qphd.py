"""qPHD for the three versions of each answer.

The design is two contrasts sharing one baseline. Polished minus raw holds the
invented content fixed and changes only the writing; corrected minus polished
holds the writing roughly fixed and removes the invention. If the bands respond
to invention at all, it has to show in the second and not be explained by the
first.

`raw` is the baseline each displacement is measured against, exactly as
`identity` is for the perturbations, so the numbers are on the same scale as
everything else in this project.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from qphd import qphd

BASE = os.path.join(HERE, "..")
L = cfg.L_DEFAULT
VERSIONS = ["raw", "polished", "fixed", "chained", "absurd"]


def load_versions():
    out = {}
    for v in VERSIONS:
        p = os.path.join(BASE, "results", f"halluc_{v}.json")
        with open(p) as f:
            out[v] = json.load(f)["answers"]
    return out


def main():
    texts = load_versions()
    emb = Embedder()
    qids = sorted(set.intersection(*(set(t) for t in texts.values())))
    print(f"{len(qids)} вопросов x {len(VERSIONS)} версий, L={L}", flush=True)

    rows, skipped, t0 = [], 0, time.time()
    for i, qid in enumerate(qids):
        # a question is kept only if all three versions survive the length
        # requirement, otherwise the pair is not a pair
        embeds = {}
        for v in VERSIONS:
            e = emb.embed_cached(texts[v][qid], cache_key=f"halluc_{v}_{qid}")
            if e.shape[0] < L:
                break
            embeds[v] = e
        if len(embeds) < len(VERSIONS):
            skipped += 1
            continue
        for v, e in embeds.items():
            rng = np.random.default_rng(42 + i)
            idx = rng.choice(e.shape[0], size=L, replace=False)
            df = qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                      **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, rng),
                                        replicates=cfg.REPLICATES))
            df["version"] = v
            df["qid"] = qid
            df["n_tokens"] = e.shape[0]
            rows.append(df)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(qids)}  {time.time() - t0:.0f}s", flush=True)

    D = pd.concat(rows)
    path = os.path.join(BASE, "results", f"halluc_qphd_L{L}.csv.gz")
    D.to_csv(path, index=False)
    print(f"\n{D.qid.nunique()} вопросов прошло, {skipped} коротких отброшено")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
