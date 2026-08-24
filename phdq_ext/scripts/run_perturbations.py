"""qPHD under controlled text perturbations.

Each perturbation is applied to the same set of source texts, so every
comparison is paired: the same text with and without the change. That removes
between-text variance, which we measured to be the larger of the two
components, and makes small effects visible.

Only mechanical perturbations are run here (perturb.py); the LLM rewrites go
through generate_llm_edits.py and land in the same table via --texts-json.

Writes results/perturb_L{L}{tag}.csv with a `perturbation` column.
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
from data import GENRES, load
from embedder import Embedder
from perturb import PERTURBATIONS, apply
from qphd import qphd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=cfg.L_DEFAULT)
    ap.add_argument("--n-texts", type=int, default=80, help="per genre")
    ap.add_argument("--genres", nargs="+", default=GENRES)
    ap.add_argument("--source", default="human")
    ap.add_argument("--corpus", choices=["flat", "coling"], default="flat")
    ap.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS))
    ap.add_argument("--texts-json", default=None,
                    help="precomputed edits: {perturbation: {text_id: text}}")
    ap.add_argument("--replicates", type=int, default=cfg.REPLICATES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump-only", action="store_true",
                    help="write the perturbed texts and skip the qPHD pass")
    args = ap.parse_args()

    extra = {}
    if args.texts_json:
        with open(args.texts_json) as f:
            extra = json.load(f)
        print(f"loaded {len(extra)} perturbations from {args.texts_json}",
              flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"perturb_L{args.L}{args.tag}.csv.gz")
    # every perturbed text is also written out, in the same shape as
    # llm_edits.json, so the mechanical ones can be read and not just rerun
    texts_path = os.path.join(RESULTS_DIR, f"perturbed_texts{args.tag}.json")
    written = {}
    emb = Embedder()

    if args.corpus == "coling":
        from coling_data import human_texts

        genres_iter = [("coling", human_texts(args.n_texts, min_words=280))]
    else:
        genres_iter = [(g, load(g, args.source)) for g in args.genres]

    rows = []
    for genre, texts in genres_iter:
        # a perturbation may shorten the text below L, so require some slack
        chosen = []
        for i, (text_id, text) in enumerate(texts):
            if len(chosen) >= args.n_texts:
                break
            e = emb.embed_cached(text, cache_key=f"{args.source}_{genre}_{i}")
            if e.shape[0] < int(1.3 * args.L):
                continue
            chosen.append((i, text_id, text))
        print(f"{genre}: {len(chosen)} texts", flush=True)

        for pname in args.perturbations:
            t0 = time.time()
            done = skipped = 0
            for i, text_id, text in chosen:
                if pname in extra:
                    lookup = text_id if args.corpus == "coling" else f"{genre}::{text_id}"
                    new = extra[pname].get(lookup)
                    if new is None:
                        skipped += 1
                        continue
                else:
                    new = apply(pname, text, seed=args.seed + i)
                wkey = text_id if args.corpus == "coling" else f"{genre}::{text_id}"
                written.setdefault(pname, {})[wkey] = new
                if args.dump_only:
                    done += 1
                    continue
                embeds = emb.embed_cached(
                    new, cache_key=f"pert_{pname}_{args.source}_{genre}_{i}"
                )
                if embeds.shape[0] < args.L:
                    skipped += 1
                    continue
                rng = np.random.default_rng(args.seed + i)
                idx = rng.choice(embeds.shape[0], size=args.L, replace=False)
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
                df["perturbation"] = pname
                df["text_id"] = text_id
                df["n_tokens"] = embeds.shape[0]
                rows.append(df)
                done += 1
            print(f"  {pname:26s} {done:4d} texts"
                  f"{f' ({skipped} skipped)' if skipped else ''}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            if rows:
                pd.concat(rows).to_csv(out_path, index=False)
            with open(texts_path, "w") as f:
                json.dump(written, f, ensure_ascii=False)

    if rows:
        print(f"saved -> {out_path}", flush=True)
    print(f"saved -> {texts_path}", flush=True)


if __name__ == "__main__":
    main()
