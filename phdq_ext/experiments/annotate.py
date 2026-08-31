"""Annotate individual texts: ten judged properties, ten counted ones, three bands.

Sampling is stratified by perturbation rather than by position in the band
space, because the point of moving to individual texts is to see whether a
relation that holds between perturbations also holds within one. That needs
replication inside each operation, which a sample spread evenly over the space
would not guarantee.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS, build, parse
from judge_debug import FILES as TEXT_FILES
from mech_props import NAMES as MECH, corpus_ranks, measure
from openrouter import complete
from three_bands import BANDS, SUF

BASE = os.path.join(HERE, "..")
MODEL = "gemini-3.6-flash"
PER_PERT = int(os.environ.get("PER_PERT", 30))
N_PERT = int(os.environ.get("N_PERT", 40))


def load_texts():
    out = {}
    for fn in TEXT_FILES:
        p = os.path.join(BASE, "results", fn)
        if os.path.exists(p):
            with open(p) as f:
                for k, v in json.load(f).items():
                    out.setdefault(k, {}).update(v)
    return out


def main():
    T = pd.read_csv(os.path.join(BASE, "results", f"per_text_bands{SUF}.csv"))
    texts = load_texts()
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    orig = {str(r.id): r.text for r in pool.itertuples()}
    ranks = corpus_ranks([r.text for r in pool.itertuples() if r.is_human])

    have = T[T["perturbation"].isin(texts)]
    counts = have.groupby("perturbation").size()
    keep = counts[counts >= PER_PERT].index
    rng = np.random.default_rng(0)
    chosen = sorted(rng.permutation(list(keep))[:N_PERT])
    rows = []
    for p in chosen:
        sub = have[have["perturbation"] == p]
        sub = sub.sample(min(PER_PERT, len(sub)), random_state=0)
        for _, r in sub.iterrows():
            tid = str(r["text_id"])
            body = texts[p].get(tid) or texts[p].get(f"coling::{tid}")
            if body:
                rows.append((p, tid, body, r))
    print(f"{len(chosen)} пертурбаций x до {PER_PERT} текстов = {len(rows)} "
          f"вызовов", flush=True)

    done, t0 = [0], time.time()

    def one(job):
        p, tid, body, r = job
        out = complete(build(body), model=MODEL, provider="apiyi",
                       max_tokens=1200, temperature=0)
        got = parse(out)
        rec = {"perturbation": p, "text_id": tid}
        rec |= {k: v[0] for k, v in got.items()}
        rec |= {f"why_{k}": v[1] for k, v in got.items()}
        rec |= measure(body, ranks)
        base = orig.get(tid)
        if base:
            b = measure(base, ranks)
            rec |= {f"src_{k}": v for k, v in b.items()}
        rec |= {n: r[n] for n in BANDS}
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"  {done[0]}/{len(rows)}  {time.time() - t0:.0f}s",
                  flush=True)
        return rec

    out = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(one, j) for j in rows]
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as exc:
                print("  ошибка:", str(exc)[:90], flush=True)
    D = pd.DataFrame(out)
    path = os.path.join(BASE, "results", f"annotated{SUF}.csv")
    D.to_csv(path, index=False)
    scored = D[list(PROPS)].notna().all(axis=1).sum()
    print(f"\nsaved {path}: {len(D)} строк, полностью оценено {scored}")


if __name__ == "__main__":
    main()
