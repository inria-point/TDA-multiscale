"""Stage one: long answers from the small model, which is where the nonsense is.

The prompt asks for reasoning at length rather than for the answer, because
that is where a 1.5B model invents -- the verdict is often right by luck while
the supporting detail is fabricated. Correctness of the final answer is not
what this experiment measures.
"""
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from small_gen import MODEL, generate

BASE = os.path.join(HERE, "..")
WORDS = int(os.environ.get("WORDS", 600))


def prompt_for(q):
    return (f"{q}\n\nAnswer this in about {WORDS} words. Explain your "
            f"reasoning in full: give the relevant background, name the "
            f"specific facts, figures, dates, mechanisms or authorities that "
            f"bear on it, weigh the alternatives, and say what follows. Be "
            f"concrete and specific throughout.")


def main():
    Q = pd.read_csv(os.path.join(BASE, "results", "halluc_questions.csv"))
    out, t0 = {}, time.time()
    for i, r in Q.iterrows():
        txt = generate(prompt_for(r["question"]), max_new_tokens=1000,
                       temperature=0.8, seed=i)
        out[r["qid"]] = txt
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(Q)}  {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(BASE, "results", "halluc_raw.json")
    with open(path, "w") as f:
        json.dump({"model": MODEL, "answers": out}, f, ensure_ascii=False,
                  indent=1)
    n = pd.Series({k: len(v.split()) for k, v in out.items()})
    print(f"\n{len(out)} ответов, слов: медиана {n.median():.0f}, "
          f"мин {n.min()}, макс {n.max()}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
