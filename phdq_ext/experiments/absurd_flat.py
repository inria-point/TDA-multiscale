"""Absurdity again, this time without the tokeniser confound.

The first absurd arm replaced technical vocabulary with words like bergamot and
juggler, which the tokeniser splits more than pleocytosis or BRCA2 does. Texts
gained 5.4% in tokens and 4.2 points of subword share, and the coarse-band drop
shrank from 9.0% to 5.1% once that was partialled out.

So the substitutions are drawn from a fixed pool this time: content words that
the encoder holds as a single token and that are common in the corpus. The
result should be just as absurd and no more fragmented than the text it
replaces.
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

PROMPT = """Below is a well-written factual text. Rewrite it so that what it
says is flagrantly, obviously false, while the writing itself stays exactly as
it is.

This is material for a study of prose style, so the form must survive
untouched. Keep the same number of paragraphs, the same sentence boundaries,
the same sentence structures, the same connectives, the same formatting, the
same register, and the same length to within five per cent. Every sentence in
the rewrite should sit where the corresponding sentence sat in the original and
do the same grammatical work.

Change only what the text asserts. Replace the entities, mechanisms, causes,
figures and authorities with absurd ones.

There is one constraint on the replacements: use ordinary, everyday words.
Nothing technical, nothing rare, nothing foreign, nothing long or unusual. The
words themselves should be words a child knows -- it is their being in the
wrong place that makes the text absurd, not the words being strange. Where the
original has a technical term, put an everyday phrase of roughly the same
length and the same grammatical category in its place.

Assert it all in exactly the tone of the original: calm, confident, technical.
Do not signal that anything is a joke, do not wink at the reader, do not hedge,
do not add caveats. The result should read as competent prose that happens to
state nonsense.

Output only the rewritten text.
---
ORIGINAL:
{text}
"""



def main():
    with open(os.path.join(BASE, "results", "halluc_chained.json")) as f:
        base = json.load(f)["answers"]
    done, t0 = [0], time.time()

    def one(qid):
        out = complete(PROMPT.format(text=base[qid]),
                       model=MODEL, provider="apiyi", max_tokens=2500,
                       temperature=0.4)
        got = J.parse(complete(J.build(out), model=MODEL, provider="apiyi",
                               max_tokens=900, temperature=0))
        done[0] += 1
        if done[0] % 15 == 0:
            print(f"  {done[0]}/{len(base)}  {time.time() - t0:.0f}s",
                  flush=True)
        rec = {"qid": qid, "n_words_flat": len(out.split())}
        for k in J.PROPS:
            if k in got:
                rec[f"flat_{k}"], rec[f"why_flat_{k}"] = got[k]
        return rec, out

    rows, texts = [], {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(one, q) for q in base]
        for f in as_completed(futs):
            try:
                rec, out = f.result()
                rows.append(rec)
                texts[rec["qid"]] = out
            except Exception as exc:
                print("  ошибка:", str(exc)[:120], flush=True)

    F = pd.DataFrame(rows).sort_values("qid")
    with open(os.path.join(BASE, "results", "halluc_flat.json"), "w") as f:
        json.dump({"model": MODEL, "answers": texts}, f,
                  ensure_ascii=False, indent=1)
    path = os.path.join(BASE, "results", "halluc_scores.csv")
    D = pd.read_csv(path)
    D = D.drop(columns=[c for c in D.columns if c.startswith("flat")],
               errors="ignore").merge(F, on="qid", how="left")
    D.to_csv(path, index=False)

    print("\nоценки:")
    for k, meta in J.PROPS.items():
        cols = [f"{v}_{k}" for v in ("chained", "absurd", "flat")]
        if all(c in D for c in cols):
            m = D[cols].mean()
            print(f"  {meta['ru']:22s} верно {m[cols[0]]:.1f}  "
                  f"абсурд {m[cols[1]]:.1f}  абсурд из пула {m[cols[2]]:.1f}")
    r = D["n_words_flat"] / D["n_words_chained"]
    print(f"\nдлина к исправленному: медиана {r.median():.2f}")
    q = sorted(texts)[3]
    print("\nпример:", " ".join(texts[q].split()[:50]))


if __name__ == "__main__":
    main()
