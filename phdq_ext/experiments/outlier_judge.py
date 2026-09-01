"""Score the profile groups blind, and see whether a reader agrees.

The judge is given the text and nothing else -- no genre, no deviation, no
profile -- so the scores are an independent opinion about whether these texts
are damaged. That is the whole point: the geometry flagged them without reading
them, and the question is whether the flag means anything.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS, build, parse
from mech_props import NAMES as MECH, corpus_ranks, measure
from openrouter import complete

BASE = os.path.join(HERE, "..")
MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.6-flash")


def main():
    S = pd.read_csv(os.path.join(BASE, "results", "outlier_sample.csv"))
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    ranks = corpus_ranks([r.text for r in pool.itertuples() if r.is_human])
    print(f"{len(S)} текстов, {S['профиль'].nunique()} профилей", flush=True)
    done, t0 = [0], time.time()

    def one(row):
        out = complete(build(row.text), model=MODEL, provider="apiyi",
                       max_tokens=1200, temperature=0)
        got = parse(out)
        rec = {"id": row.id}
        rec |= {k: v[0] for k, v in got.items()}
        rec |= {f"why_{k}": v[1] for k, v in got.items()}
        rec |= measure(row.text, ranks)
        done[0] += 1
        if done[0] % 40 == 0:
            print(f"  {done[0]}/{len(S)}  {time.time() - t0:.0f}s", flush=True)
        return rec

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, r) for r in S.itertuples()]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as exc:
                print("  ошибка:", str(exc)[:110], flush=True)
    J = pd.DataFrame(rows)
    J["id"] = J["id"].astype(str)
    S["id"] = S["id"].astype(str)
    D = S.merge(J, on="id", how="left")
    path = os.path.join(BASE, "results", "outlier_judged.csv")
    D.drop(columns=["text"]).to_csv(path, index=False)
    scored = D[list(PROPS)].notna().any(axis=1).sum()
    print(f"\nsaved {path}: {len(D)} строк, оценено {scored}")


if __name__ == "__main__":
    main()
