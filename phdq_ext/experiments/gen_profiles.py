"""The same profile map for generated text, split by training regime.

Human documents are damaged by a pipeline; generated ones are damaged, if at
all, by the generator. Whether the same axes apply is the question, and the two
regimes have to be kept apart because they sit in different places to begin
with: base models run above human text on the fine and middle bands, 4.12 and
15.28 against 3.50 and 14.56, instruction-tuned ones below, 3.02 and 12.97.
Pooling them would make every deviation a statement about which regime produced
the text.

Deviation is therefore taken within regime, and the sampling is stratified by
profile so the cells have something in them.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from defect_judge import build, check_quotes, parse
from openrouter import complete
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.6-flash")
PER_PROFILE = int(os.environ.get("PER_PROFILE", 14))
MIN_N = 6


def main():
    T = pd.read_csv(os.path.join(BASE, "results", "gen_bands.csv"))
    T["id"] = T["id"].astype(str)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    T = T.merge(pool[["id", "text"]], on="id")
    # within regime, since the regimes start from different places
    for g, idx in T.groupby("группа").groups.items():
        for b in B:
            m = T.loc[idx, b].mean()
            T.loc[idx, f"откл_{b}"] = (T.loc[idx, b] - m) / m * 100

    picks = []
    for th in (20.0,):
        P = T[[f"откл_{b}" for b in B]].values
        H = np.sign(P) * np.clip(np.abs(P) - th, 0, None)
        T["профиль"] = ["".join("0" if v == 0 else ("+" if v > 0 else "−")
                                for v in r) for r in H]
    for (grp, prof), g in T.groupby(["группа", "профиль"]):
        if len(g) >= MIN_N:
            picks.append(g.sample(min(PER_PROFILE, len(g)), random_state=0))
    S = pd.concat(picks).drop_duplicates("id")
    print(f"{len(S)} текстов: " + ", ".join(
        f"{k} {v}" for k, v in S["группа"].value_counts().items()), flush=True)
    print("профилей:", S.groupby("группа")["профиль"].nunique().to_dict(),
          flush=True)
    done, t0 = [0], time.time()

    def one(r):
        out = complete(build(r.text), model=MODEL, provider="apiyi",
                       max_tokens=1400, temperature=0)
        rec = parse(out)
        done[0] += 1
        if done[0] % 30 == 0:
            print(f"  {done[0]}/{len(S)}  {time.time() - t0:.0f}s", flush=True)
        return {"id": r.id, "группа": r.группа, "model": r.model,
                "sub_source": r.sub_source,
                **{f"откл_{b}": getattr(r, f"откл_{b}") for b in B},
                **rec, "плохие цитаты": "; ".join(check_quotes(rec, r.text))}

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, r) for r in S.itertuples()]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as exc:
                print("  ошибка:", str(exc)[:110], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "gen_judged.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + D.groupby("группа")[
        ["loss", "web", "dup", "chars", "cut", "damage", "nature", "human"]
    ].mean().round(2).to_string())
    print(f"\nнеподтверждённых цитат: {(D['плохие цитаты'] != '').sum()}/{len(D)}")


if __name__ == "__main__":
    main()
