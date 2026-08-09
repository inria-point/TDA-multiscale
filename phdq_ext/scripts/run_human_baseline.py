"""Stage 1: qPHD(q) curves for human texts, 4 genres, length-equalized.

For each genre: take texts with >= L tokens, embed with ModernBERT,
subsample L token-embeddings, run qPHD in 3 trimming modes.
Writes tidy per-text results to results/human_L{L}.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from embedder import Embedder
from qphd import qphd

GENRES = ["xsum", "academic_abstracts", "amazon_reviews", "writingprompts"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_completion")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
Q_LIST = tuple(np.round(np.arange(0, 0.901, 0.05), 2))
# 8 log-spaced subsample sizes over [0.2*L, L]; see variant_sweep.py
LOG8 = tuple(np.round(np.logspace(np.log10(0.2), 0.0, 8), 3))


def load_texts(genre, source="human"):
    path = os.path.join(DATA_DIR, genre, f"{source}.jsonl")
    with open(path) as f:
        return [json.loads(line)["abstract"] for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=cfg.L_DEFAULT,
                    help="tokens per text (subsample)")
    ap.add_argument("--n-texts", type=int, default=100, help="texts per genre")
    ap.add_argument("--replicates", type=int, default=32)
    ap.add_argument("--source", default="human")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="suffix for the output file")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="paper settings: bootstrap sampling, 5 uniform n, pool = the L-subset",
    )
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.tag or ("legacy" if args.legacy else "v2")
    out_path = os.path.join(RESULTS_DIR, f"{args.source}_L{args.L}_{tag}.csv")

    emb = Embedder()
    print(f"device={emb.device}  L={args.L}  n_texts={args.n_texts}", flush=True)

    all_rows = []
    for genre in GENRES:
        texts = load_texts(genre, args.source)
        t_start = time.time()
        done = 0
        for i, text in enumerate(texts):
            if done >= args.n_texts:
                break
            embeds = emb.embed_cached(text, cache_key=f"{args.source}_{genre}_{i}")
            if embeds.shape[0] < args.L:
                continue
            rng = np.random.default_rng(args.seed + i)
            idx = rng.choice(embeds.shape[0], size=args.L, replace=False)
            if args.legacy:
                df = qphd(embeds[idx], q_list=Q_LIST, rng=rng, **cfg.LEGACY)
            else:
                df = qphd(
                    embeds[idx],
                    q_list=Q_LIST,
                    rng=rng,
                    **cfg.qphd_kwargs(
                        L=args.L,
                        pool=cfg.make_pool(embeds, args.L, rng),
                        replicates=args.replicates,
                    ),
                )
            df["genre"] = genre
            df["text_idx"] = i
            all_rows.append(df)
            done += 1
            if done % 20 == 0:
                dt = time.time() - t_start
                print(f"  {genre}: {done}/{args.n_texts}  ({dt:.0f}s)", flush=True)
        print(f"{genre}: done {done} texts in {time.time()-t_start:.0f}s", flush=True)
        pd.concat(all_rows).to_csv(out_path, index=False)  # checkpoint per genre

    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
