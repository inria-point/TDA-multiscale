"""Stage two: score the invented version, repair it, score the repair.

Scoring happens before and after so the repair can be shown to have worked --
if invention does not fall, there is no contrast to measure and the qPHD pass
would be a waste.
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
JUDGE = os.environ.get("JUDGE_MODEL", "gemini-3.6-flash")
FIXER = os.environ.get("FIXER_MODEL", "gemini-3-pro-preview")
WORKERS = int(os.environ.get("WORKERS", 6))


def score(text):
    out = complete(J.build(text), model=JUDGE, provider="apiyi",
                   max_tokens=900, temperature=0)
    return J.parse(out)


def main():
    Q = pd.read_csv(os.path.join(BASE, "results", "halluc_questions.csv")
                    ).set_index("qid")
    with open(os.path.join(BASE, "results", "halluc_raw.json")) as f:
        raw = json.load(f)["answers"]
    print(f"{len(raw)} ответов; судья {JUDGE}, правит {FIXER}", flush=True)

    done, t0 = [0], time.time()

    def one(qid):
        text = raw[qid]
        before = score(text)
        fixed = complete(
            halluc_fix.build(text, Q.loc[qid, "question"],
                             Q.loc[qid, "answer_text"]),
            model=FIXER, provider="apiyi", max_tokens=2000, temperature=0.3)
        after = score(fixed)
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"  {done[0]}/{len(raw)}  {time.time() - t0:.0f}s",
                  flush=True)
        rec = {"qid": qid, "subject": Q.loc[qid, "subject"],
               "n_words_raw": len(text.split()),
               "n_words_fixed": len(fixed.split())}
        for k in J.PROPS:
            if k in before:
                rec[f"raw_{k}"], rec[f"why_raw_{k}"] = before[k]
            if k in after:
                rec[f"fixed_{k}"], rec[f"why_fixed_{k}"] = after[k]
        return rec, fixed

    rows, fixed_texts = [], {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, q) for q in raw]
        for f in as_completed(futs):
            try:
                rec, fixed = f.result()
                rows.append(rec)
                fixed_texts[rec["qid"]] = fixed
            except Exception as exc:
                print("  ошибка:", str(exc)[:120], flush=True)

    D = pd.DataFrame(rows).sort_values("qid")
    D.to_csv(os.path.join(BASE, "results", "halluc_scores.csv"), index=False)
    with open(os.path.join(BASE, "results", "halluc_fixed.json"), "w") as f:
        json.dump({"model": FIXER, "answers": fixed_texts}, f,
                  ensure_ascii=False, indent=1)

    pd.set_option("display.width", 200)
    cols = [c for c in D.columns if c.startswith(("raw_", "fixed_", "n_words"))]
    print("\n" + D[cols].describe().loc[["mean", "std", "min", "max"]]
          .round(1).to_string())
    for k in J.PROPS:
        if f"raw_{k}" in D and f"fixed_{k}" in D:
            d = (D[f"fixed_{k}"] - D[f"raw_{k}"]).dropna()
            print(f"\n{J.PROPS[k]['ru']}: {D[f'raw_{k}'].mean():.1f} -> "
                  f"{D[f'fixed_{k}'].mean():.1f}  (сдвиг {d.mean():+.1f}, "
                  f"упало у {(d < 0).mean():.0%})")
    r = D["n_words_fixed"] / D["n_words_raw"]
    print(f"\nдлина после/до: медиана {r.median():.2f}, "
          f"вне 0.9-1.1: {((r < 0.9) | (r > 1.1)).mean():.0%}")


if __name__ == "__main__":
    main()
