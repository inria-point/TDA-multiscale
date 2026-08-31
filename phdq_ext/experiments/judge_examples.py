"""Write out the judged texts themselves next to their scores.

A score with a stated reason can still be wrong about the text; only the text
settles it. This assembles one section per variant -- the text as the judge
saw it, then every score with the sentence that justified it -- so a rating can
be checked against its object rather than against its own explanation.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS, build
from judge_debug import collect

BASE = os.path.join(HERE, "..")


def main():
    sample = dict(collect())
    D = pd.read_csv(os.path.join(BASE, "results", "judge_debug.csv"),
                    index_col=0)
    R = pd.read_csv(os.path.join(BASE, "results", "judge_debug_reasons.csv"),
                    index_col=0)

    lines = ["# Тексты и оценки судьи", "",
             "Один исходный текст, проведённый через все пертурбации. "
             "Судья видел ровно то, что приведено ниже (обрезка на 420 слов).",
             ""]
    order = list(D.index)
    for name in order:
        text = sample.get(name, "")
        shown = " ".join(text.split()[:420])
        lines += [f"## {name}", "", "```", shown, "```", "",
                  "| свойство | оценка | почему |", "|---|---|---|"]
        for k, p in PROPS.items():
            if k not in D.columns:
                continue
            v = D.loc[name, k]
            v = "—" if pd.isna(v) else f"**{v:g}**"
            why = R.loc[name, k] if k in R.columns else ""
            why = "" if pd.isna(why) else str(why).replace("|", "/")
            lines.append(f"| {p['ru']} | {v} | {why} |")
        lines.append("")

    path = os.path.join(BASE, "results", "judge_examples.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {path}: {len(order)} вариантов, "
          f"{sum(len(l) for l in lines) // 1024} КБ")


if __name__ == "__main__":
    main()
