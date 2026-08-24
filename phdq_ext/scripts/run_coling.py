"""qPHD over a COLING pool, one row per (text, mode, q).

Same calibrated configuration as everywhere else (config.py), so the numbers
are comparable with the FLAT runs.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from embedder import Embedder
from qphd import qphd

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(BASE, "..", "coling", "pool.parquet"))
    ap.add_argument("--L", type=int, default=cfg.L_DEFAULT)
    ap.add_argument("--replicates", type=int, default=cfg.REPLICATES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    if args.limit:
        pool = pool.head(args.limit)
    out_path = os.path.join(BASE, "results", f"coling_qphd_L{args.L}{args.tag}.csv.gz")

    emb = Embedder()
    print(f"device={emb.device}  texts={len(pool)}  L={args.L}", flush=True)

    rows, skipped, t0 = [], 0, time.time()
    for i, r in pool.iterrows():
        e = emb.embed_cached(r["text"], cache_key=f"coling_{r['id']}")
        if e.shape[0] < args.L:
            skipped += 1
            continue
        rng = np.random.default_rng(args.seed + i)
        idx = rng.choice(e.shape[0], size=args.L, replace=False)
        df = qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                  **cfg.qphd_kwargs(L=args.L,
                                    pool=cfg.make_pool(e, args.L, rng),
                                    replicates=args.replicates))
        for col in ["id", "model", "sub_source", "source", "is_human", "n_words"]:
            df[col] = r[col]
        df["n_tokens"] = e.shape[0]
        rows.append(df)
        if len(rows) % 200 == 0:
            print(f"  {len(rows)}/{len(pool)}  ({time.time()-t0:.0f}s)", flush=True)
            pd.concat(rows).to_csv(out_path, index=False)

    pd.concat(rows).to_csv(out_path, index=False)
    print(f"готово: {len(rows)} текстов, пропущено {skipped}, "
          f"{time.time()-t0:.0f}s\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
