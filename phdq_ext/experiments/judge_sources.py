"""Score the untouched sources, so a judged property can be read as a shift.

An absolute score confounds the operation with the document: a perturbed text
rated 6 may have started at 9 or at 6. The mechanical measures already showed
that pairing is what makes a property track the bands, and the same correction
was never available for the judge because only perturbed texts were scored.
This scores the 104 originals once each; every perturbed row then carries
`d_<property>` = its score minus its own source's.
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
from openrouter import complete
from three_bands import SUF

BASE = os.path.join(HERE, "..")
MODEL = "gemini-3.6-flash"


def main():
    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    orig = {str(r.id): r.text for r in pool.itertuples()}
    ids = [t for t in D["text_id"].astype(str).unique() if t in orig]
    print(f"{len(ids)} исходников к оценке", flush=True)

    done, t0 = [0], time.time()

    def one(tid):
        out = complete(build(orig[tid]), model=MODEL, provider="apiyi",
                       max_tokens=1200, temperature=0)
        got = parse(out)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  {done[0]}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
        return {"text_id": tid, **{f"base_{k}": v[0] for k, v in got.items()}}

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, t): t for t in ids}
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as exc:
                print("  ошибка:", str(exc)[:90], flush=True)
    S = pd.DataFrame(rows)
    path = os.path.join(BASE, "results", f"judge_sources{SUF}.csv")
    S.to_csv(path, index=False)
    print(f"\nsaved {path}: {len(S)} исходников")
    print(S[[f"base_{k}" for k in PROPS if f"base_{k}" in S]]
          .describe().loc[["mean", "std", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
