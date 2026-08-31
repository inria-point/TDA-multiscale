"""The control arm: same polish, invented content left in place.

Repairing the facts also raised fluency from 5.5 to 8.8, so the corrected text
differs from the original in two ways at once and any band movement could be
either. This arm changes only the writing: every claim is carried over as
stated, however wrong. Corrected minus polished then isolates invention with
fluency held at the same level, which is the contrast the experiment was for.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import halluc_judge as J
from openrouter import complete

BASE = os.path.join(HERE, "..")
MODEL = os.environ.get("FIXER_MODEL", "gemini-3.6-flash")

PROMPT = """Below is a long answer written by a small language model. The
writing is uneven: awkward phrasing, repetition, weak connectives, padding.

Rewrite it so that it reads as well-written prose, and change nothing else.

This text is material for a study of writing quality, so its content must
survive intact. Some of its claims are factually wrong. Carry every claim over
exactly as it stands, including the wrong ones: the same facts, the same
figures, dates, names, mechanisms and attributions, the same conclusions,
asserted with the same confidence. Do not correct anything, do not hedge
anything that was asserted, do not add caveats, and do not signal that
anything is doubtful.

Preserve the length to within five per cent, the paragraph structure, the
formatting, and the order in which topics appear. Improve only sentence
construction, word choice, transitions and rhythm.

Output only the rewritten text.
---
ORIGINAL:
{text}
"""


def main():
    with open(os.path.join(BASE, "results", "halluc_raw.json")) as f:
        raw = json.load(f)["answers"]
    done, t0 = [0], time.time()

    def one(qid):
        text = raw[qid]
        out = complete(PROMPT.format(text=text), model=MODEL, provider="apiyi",
                       max_tokens=2500, temperature=0.3)
        got = J.parse(complete(J.build(out), model=MODEL, provider="apiyi",
                               max_tokens=900, temperature=0))
        done[0] += 1
        if done[0] % 15 == 0:
            print(f"  {done[0]}/{len(raw)}  {time.time() - t0:.0f}s",
                  flush=True)
        rec = {"qid": qid, "n_words_polished": len(out.split())}
        for k in J.PROPS:
            if k in got:
                rec[f"polished_{k}"], rec[f"why_polished_{k}"] = got[k]
        return rec, out

    rows, texts = [], {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(one, q) for q in raw]
        for f in as_completed(futs):
            try:
                rec, out = f.result()
                rows.append(rec)
                texts[rec["qid"]] = out
            except Exception as exc:
                print("  ошибка:", str(exc)[:120], flush=True)

    P = pd.DataFrame(rows).sort_values("qid")
    with open(os.path.join(BASE, "results", "halluc_polished.json"), "w") as f:
        json.dump({"model": MODEL, "answers": texts}, f, ensure_ascii=False,
                  indent=1)
    D = pd.read_csv(os.path.join(BASE, "results", "halluc_scores.csv"))
    D = D.merge(P, on="qid", how="left")
    D.to_csv(os.path.join(BASE, "results", "halluc_scores.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nтри версии, средние оценки:")
    for k, meta in J.PROPS.items():
        cols = [f"raw_{k}", f"polished_{k}", f"fixed_{k}"]
        if all(c in D for c in cols):
            m = D[cols].mean()
            print(f"  {meta['ru']:22s} выдумка {m[cols[0]]:.1f}  "
                  f"полировка {m[cols[1]]:.1f}  правка {m[cols[2]]:.1f}")
    r = D["n_words_polished"] / D["n_words_raw"]
    print(f"\nдлина полировки к исходнику: медиана {r.median():.2f}")


if __name__ == "__main__":
    main()
