"""Twenty texts to read without knowing which is which.

Every automatic check so far asks about properties someone specified in
advance, and the objection to all of them is the same: a defect nobody named is
a defect nobody measured. Reading is the only method without that limit.

Ten texts with the largest deviation from the corpus mean and ten with the
smallest, shuffled together and stripped of every label. The key goes to a
separate file so that whoever reads the texts -- including whoever generated
them -- can record a judgement before seeing it.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from three_bands import BANDS as _B

B = list(_B)
BASE = os.path.join(HERE, "..")
N_EACH = 10
SEED = 20260901


def main():
    T = pd.read_csv(os.path.join(BASE, "results", "human_band_dev.csv"))
    T["id"] = T["id"].astype(str)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    T = T.merge(pool[["id", "text"]], on="id")
    T["макс_откл"] = T[[f"all_{b}" for b in B]].abs().max(axis=1)

    hi = T.nlargest(N_EACH, "макс_откл").assign(группа="аутлаер")
    lo = T.nsmallest(N_EACH, "макс_откл").assign(группа="обычный")
    S = pd.concat([hi, lo]).sample(frac=1, random_state=SEED).reset_index(
        drop=True)
    S["метка"] = [f"T{i + 1:02d}" for i in range(len(S))]

    key = S[["метка", "группа", "id", "sub_source", "макс_откл", "профиль"]
            + [f"all_{b}" for b in B]]
    key.round(1).to_csv(os.path.join(BASE, "results", "blind_read_key.csv"),
                        index=False)

    out = [
        "# Двадцать текстов вслепую",
        "",
        "Десять из них размерность считает аномальными для корпуса, десять — "
        "обычными. Порядок перемешан, пометок нет.",
        "",
        "Читая, отмечай не «нравится / не нравится», а **дефект данных**: "
        "обрыв, склейка разнородных кусков, форматный мусор, дублирование "
        "внутри текста, следы разметки или навигации, смена языка, "
        "обрубленное начало или конец, шаблонность, всё прочее.",
        "",
        "## Куда записывать",
        "",
        "| текст | дефект есть? | какой | это аутлаер? |",
        "|---|---|---|---|",
    ]
    out += [f"| {m} |  |  |  |" for m in S["метка"]]
    out += ["", "Ключ лежит в `results/blind_read_key.csv` — не открывай, "
                "пока не разметишь.", "", "---", ""]
    for _, r in S.iterrows():
        w = len(r["text"].split())
        out += [f"## {r['метка']}  ·  {w} слов", "", "```text",
                r["text"].strip(), "```", ""]
    path = os.path.join(BASE, "results", "blind_read.md")
    with open(path, "w") as f:
        f.write("\n".join(out))

    print(f"{len(S)} текстов, {S['text'].str.split().str.len().sum()} слов "
          f"всего, медиана {S['text'].str.split().str.len().median():.0f}")
    print(f"отклонения: аутлаеры {hi['макс_откл'].min():.0f}-"
          f"{hi['макс_откл'].max():.0f}%, "
          f"обычные {lo['макс_откл'].min():.1f}-{lo['макс_откл'].max():.1f}%")
    print(f"жанры: {S['sub_source'].value_counts().to_dict()}")
    print(f"\nsaved {path}")
    print("ключ: results/blind_read_key.csv (не открывать до разметки)")


if __name__ == "__main__":
    main()
