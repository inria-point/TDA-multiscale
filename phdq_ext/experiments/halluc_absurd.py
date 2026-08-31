"""The reverse direction: keep the prose, make the content absurd.

Removing plausible invention moved nothing, but plausible invention is close to
the truth in vocabulary as well as in form -- a wrong enzyme is still an enzyme.
This arm goes the other way and as far as it can: the corrected text keeps its
sentences, its rhythm and its confidence, and its factual content is replaced
with statements that no reader could take seriously. If the bands miss this
too, they are not reading content at all.

It runs from the corrected version, so the pair differs in one step, as the
chain does elsewhere.
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
figures and authorities with absurd ones: a patient may become a bergamot tree,
an enzyme may become a brass instrument, a treaty may have been signed by a
weather system. Keep each replacement in the same grammatical category and of
roughly the same length as what it replaces, so a noun phrase becomes a noun
phrase of similar size.

Assert it all in exactly the tone of the original: calm, confident, technical.
Do not signal that anything is a joke, do not wink at the reader, do not hedge,
do not add caveats or notes. The result should read as competent prose that
happens to state nonsense.

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
        out = complete(PROMPT.format(text=base[qid]), model=MODEL,
                       provider="apiyi", max_tokens=2500, temperature=0.4)
        got = J.parse(complete(J.build(out), model=MODEL, provider="apiyi",
                               max_tokens=900, temperature=0))
        done[0] += 1
        if done[0] % 15 == 0:
            print(f"  {done[0]}/{len(base)}  {time.time() - t0:.0f}s",
                  flush=True)
        rec = {"qid": qid, "n_words_absurd": len(out.split())}
        for k in J.PROPS:
            if k in got:
                rec[f"absurd_{k}"], rec[f"why_absurd_{k}"] = got[k]
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

    A = pd.DataFrame(rows).sort_values("qid")
    with open(os.path.join(BASE, "results", "halluc_absurd.json"), "w") as f:
        json.dump({"model": MODEL, "answers": texts}, f, ensure_ascii=False,
                  indent=1)
    path = os.path.join(BASE, "results", "halluc_scores.csv")
    D = pd.read_csv(path)
    D = D.drop(columns=[c for c in D.columns if c.startswith("absurd")],
               errors="ignore").merge(A, on="qid", how="left")
    D.to_csv(path, index=False)

    print("\nпять версий, средние оценки:")
    for k, meta in J.PROPS.items():
        cols = [f"{v}_{k}" for v in ("raw", "polished", "chained", "absurd")]
        if all(c in D for c in cols):
            m = D[cols].mean()
            print(f"  {meta['ru']:22s} как есть {m[cols[0]]:.1f}  "
                  f"гладко {m[cols[1]]:.1f}  исправлено {m[cols[2]]:.1f}  "
                  f"абсурд {m[cols[3]]:.1f}")
    r = D["n_words_absurd"] / D["n_words_chained"]
    print(f"\nдлина абсурда к исправленному: медиана {r.median():.2f}")
    print("\nпример:")
    q = sorted(texts)[3]
    print("  ИСПРАВЛЕНО:", " ".join(base[q].split()[:55]))
    print("  АБСУРД   :", " ".join(texts[q].split()[:55]))


if __name__ == "__main__":
    main()
