"""qPHD(q) curves per genre and per source, length-equalized.

For each (genre, source): take texts with >= L tokens, embed with ModernBERT,
subsample L token-embeddings, run qPHD in the three trimming modes under the
configuration fixed in config.py.

Note on the generator files: `text_ai` begins with a human prefix of ~60 words
that the model was asked to continue. It is kept by default, matching how the
dataset is built; --strip-prefix measures the continuation alone.

Writes tidy per-text results to results/qphd_L{L}{tag}.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from data import GENRES, SOURCES, load
from embedder import Embedder
from qphd import qphd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=cfg.L_DEFAULT)
    ap.add_argument("--n-texts", type=int, default=150, help="per genre+source")
    ap.add_argument("--replicates", type=int, default=cfg.REPLICATES)
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--genres", nargs="+", default=GENRES)
    ap.add_argument("--strip-prefix", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--legacy", action="store_true", help="the paper's settings")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.tag or ("_legacy" if args.legacy else "")
    out_path = os.path.join(RESULTS_DIR, f"qphd_L{args.L}{tag}.csv.gz")

    emb = Embedder()
    print(f"device={emb.device}  L={args.L}  n_texts={args.n_texts}", flush=True)

    all_rows = []
    for source in args.sources:
        for genre in args.genres:
            texts = load(genre, source, strip_prefix=args.strip_prefix)
            t0 = time.time()
            done = 0
            for i, (text_id, text) in enumerate(texts):
                if done >= args.n_texts:
                    break
                embeds = emb.embed_cached(text, cache_key=f"{source}_{genre}_{i}")
                if embeds.shape[0] < args.L:
                    continue
                rng = np.random.default_rng(args.seed + i)
                idx = rng.choice(embeds.shape[0], size=args.L, replace=False)
                if args.legacy:
                    df = qphd(embeds[idx], q_list=cfg.Q_GRID, rng=rng, **cfg.LEGACY)
                else:
                    df = qphd(
                        embeds[idx],
                        q_list=cfg.Q_GRID,
                        rng=rng,
                        **cfg.qphd_kwargs(
                            L=args.L,
                            pool=cfg.make_pool(embeds, args.L, rng),
                            replicates=args.replicates,
                        ),
                    )
                df["genre"] = genre
                df["source"] = source
                df["text_id"] = text_id
                all_rows.append(df)
                done += 1
            print(f"{source:18s} {genre:20s} {done:4d} texts "
                  f"({time.time() - t0:.0f}s)", flush=True)
            pd.concat(all_rows).to_csv(out_path, index=False)  # checkpoint

    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
