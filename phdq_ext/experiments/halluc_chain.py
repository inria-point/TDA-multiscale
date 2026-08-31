"""The correction as a second step on the polished text, not a sibling of it.

The first attempt derived both rewrites from the original independently, so
"polished versus corrected" compared two rewrites that differ in their facts
and in every incidental choice the two instructions made about wording. Chaining
makes the second contrast a single edit on a fixed starting point: the same
polished sentences go in, and only the false claims come out.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import halluc_fix
import halluc_judge as J
from openrouter import complete

BASE = os.path.join(HERE, "..")
MODEL = os.environ.get("FIXER_MODEL", "gemini-3.6-flash")


def main():
    with open(os.path.join(BASE, "results", "halluc_polished.json")) as f:
        polished = json.load(f)["answers"]
    Q = pd.read_csv(os.path.join(BASE, "results", "halluc_questions.csv")
                    ).set_index("qid")
    done, t0 = [0], time.time()

    def one(qid):
        # the reference answer is deliberately not passed this time: on five
        # questions MMLU's key is wrong out of context and the instruction to
        # stay consistent with it made the model assert a falsehood
        out = complete(halluc_fix.build(polished[qid]), model=MODEL,
                       provider="apiyi", max_tokens=2500, temperature=0.3)
        got = J.parse(complete(J.build(out), model=MODEL, provider="apiyi",
                               max_tokens=900, temperature=0))
        done[0] += 1
        if done[0] % 15 == 0:
            print(f"  {done[0]}/{len(polished)}  {time.time() - t0:.0f}s",
                  flush=True)
        rec = {"qid": qid, "n_words_chained": len(out.split())}
        for k in J.PROPS:
            if k in got:
                rec[f"chained_{k}"], rec[f"why_chained_{k}"] = got[k]
        return rec, out

    rows, texts = [], {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(one, q) for q in polished]
        for f in as_completed(futs):
            try:
                rec, out = f.result()
                rows.append(rec)
                texts[rec["qid"]] = out
            except Exception as exc:
                print("  ошибка:", str(exc)[:120], flush=True)

    C = pd.DataFrame(rows).sort_values("qid")
    with open(os.path.join(BASE, "results", "halluc_chained.json"), "w") as f:
        json.dump({"model": MODEL, "answers": texts}, f, ensure_ascii=False,
                  indent=1)
    path = os.path.join(BASE, "results", "halluc_scores.csv")
    D = pd.read_csv(path)
    D = D.drop(columns=[c for c in D.columns if c.startswith("chained")],
               errors="ignore").merge(C, on="qid", how="left")
    D.to_csv(path, index=False)

    print("\nчетыре версии, средние оценки:")
    for k, meta in J.PROPS.items():
        cols = [f"{v}_{k}" for v in ("raw", "polished", "chained", "fixed")]
        if all(c in D for c in cols):
            m = D[cols].mean()
            print(f"  {meta['ru']:22s} как есть {m[cols[0]]:.1f}  "
                  f"гладко {m[cols[1]]:.1f}  гладко+правка {m[cols[2]]:.1f}  "
                  f"(прежняя ветка {m[cols[3]]:.1f})")
    r = D["n_words_chained"] / D["n_words_polished"]
    print(f"\nдлина после правки к гладкой: медиана {r.median():.2f}")


if __name__ == "__main__":
    main()
