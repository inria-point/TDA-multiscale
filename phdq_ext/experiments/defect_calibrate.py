"""Try the defect judge on three sets before trusting it anywhere.

A judge is only worth running once it is known to fire on damage, to stay quiet
on clean text, and to agree with a reading done by hand. Those are three
different failures and each needs its own set:

  damaged   texts we broke ourselves -- it must find them
  clean     the ten controls from the blind read -- it must stay quiet
  marked    the twenty blind-read texts -- it must agree with the manual marks

Quotations are verified against the document, so a score justified by text that
is not there counts as a failure of the judge rather than as evidence.
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
from defect_judge import OPS, build, check_quotes, parse
from judge_debug import FILES as TEXT_FILES
from openrouter import complete

BASE = os.path.join(HERE, "..")
MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.6-flash")
PER_PERT = 4


def damaged():
    known = {}
    for fn in TEXT_FILES:
        p = os.path.join(BASE, "results", fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for name, by_id in json.load(f).items():
                known.setdefault(name, {}).update(by_id)
    out = []
    for name in ("shuffle_words", "loop_tail_early", "collapse_vocab_60",
                 "ngram_2", "add_typos", "strip_punctuation"):
        for tid in sorted(known.get(name, {}))[:PER_PERT]:
            out.append({"набор": f"порча:{name}", "метка": f"{name}/{tid[:8]}",
                        "text": known[name][tid]})
    return out


def blind():
    K = pd.read_csv(os.path.join(BASE, "results", "blind_read_key.csv"))
    K["id"] = K["id"].astype(str)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    K = K.merge(pool[["id", "text"]], on="id")
    return [{"набор": f"слепые:{r['группа']}", "метка": r["метка"],
             "text": r["text"]} for _, r in K.iterrows()]


def main():
    jobs = damaged() + blind()
    print(f"{len(jobs)} документов: "
          f"{sum(j['набор'].startswith('порча') for j in jobs)} испорченных, "
          f"{sum(j['набор'].startswith('слепые') for j in jobs)} из слепого "
          f"чтения", flush=True)
    done, t0 = [0], time.time()

    def one(j):
        out = complete(build(j["text"]), model=MODEL, provider="apiyi",
                       max_tokens=1400, temperature=0)
        rec = parse(out)
        bad = check_quotes(rec, j["text"])
        done[0] += 1
        if done[0] % 15 == 0:
            print(f"  {done[0]}/{len(jobs)}  {time.time() - t0:.0f}s",
                  flush=True)
        return {"набор": j["набор"], "метка": j["метка"], **rec,
                "цитаты не подтвердились": "; ".join(bad)}

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as exc:
                print("  ошибка:", str(exc)[:110], flush=True)
    D = pd.DataFrame(rows).sort_values(["набор", "метка"])
    D.to_csv(os.path.join(BASE, "results", "defect_calibrate.csv"), index=False)

    pd.set_option("display.width", 220)
    keys = list(OPS) + ["damage", "nature", "human"]
    print("\n=== средние по наборам")
    G = D.groupby("набор")[keys].mean()
    G["n"] = D.groupby("набор").size()
    G["выбросить, %"] = D.groupby("набор")["verdict"].apply(
        lambda s: s.str.startswith("drop").mean() * 100)
    print(G.round(1).to_string())
    n_bad = (D["цитаты не подтвердились"] != "").sum()
    print(f"\nответов с неподтверждённой цитатой: {n_bad}/{len(D)}")
    if n_bad:
        print(D[D["цитаты не подтвердились"] != ""]
              [["метка", "цитаты не подтвердились"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
