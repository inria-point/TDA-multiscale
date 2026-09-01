"""The inverse experiment: corrupt only the tokenisation of clean documents.

Repairing the damaged documents returned a fifth of the coarse band. That is
consistent with two readings -- the repair was incomplete, or the corruption
only ever accounted for a fifth. Going the other way settles it: take the
control documents, whose bands sit within 6% of the corpus mean, apply exactly
the corruption the damaged ones carry, and see how far the band falls.

The corruption is the inverse of the repair and is applied at three strengths,
because the damaged documents differ in how thoroughly they were mangled: some
carry five spaces before punctuation, some carry thirty-four.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from detok_repair import bands_of, letters
from embedder import Embedder
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
CLITIC = re.compile(r"\b(\w+?)(n't|'s|'re|'ve|'ll|'d|'m)\b")
PUNCT = re.compile(r"(\w)([,.;:!?])")


def corrupt(t, share=1.0, seed=0):
    """Split clitics off and put a space before punctuation, as a tokeniser
    left half-done would. `share` selects what fraction of sites are hit."""
    rng = np.random.default_rng(seed)

    def maybe(m, joiner):
        return joiner(m) if rng.random() < share else m.group(0)

    t = CLITIC.sub(lambda m: maybe(m, lambda x: f"{x.group(1)} {x.group(2)}"), t)
    t = PUNCT.sub(lambda m: maybe(m, lambda x: f"{x.group(1)} {x.group(2)}"), t)
    t = re.sub(r"(\w)-(\w)",
               lambda m: maybe(m, lambda x: f"{x.group(1)} - {x.group(2)}"), t)
    return t


def main():
    R = pd.read_csv(os.path.join(BASE, "results", "detok_repair.csv"))
    ctl = R[R["группа"] == "контроль"]["id"].astype(str).tolist()
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    S = pool[pool["id"].isin(ctl)]
    emb = Embedder()
    print(f"{len(S)} контрольных документов, три силы порчи", flush=True)

    rows = []
    for i, r in enumerate(S.itertuples()):
        base = bands_of(r.text, emb, seed=i)
        if not base:
            continue
        rec = {"id": r.id, **{f"чистый_{k}": v for k, v in base.items()}}
        for share in (0.33, 0.66, 1.0):
            bad = corrupt(r.text, share, seed=i)
            assert letters(bad) == letters(r.text)
            v = bands_of(bad, emb, seed=i)
            if v:
                rec |= {f"порча{share:.2f}_{k}": x for k, x in v.items()}
                rec[f"вставок{share:.2f}"] = len(bad) - len(r.text)
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(S)}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "detok_break.csv"), index=False)

    from scipy import stats as st
    print(f"\n{len(D)} документов; поток букв не изменён\n")
    print(f"{'сила':>6s} {'вставок':>8s}  " + "  ".join(f"{b:>18s}" for b in B))
    for share in (0.33, 0.66, 1.0):
        line = f"{share:6.2f} {D[f'вставок{share:.2f}'].mean():8.0f}  "
        for b in B:
            x, y = D[f"чистый_{b}"], D[f"порча{share:.2f}_{b}"]
            rel = ((y - x) / x * 100)
            line += (f"{rel.mean():+6.1f}% p={st.wilcoxon(y - x).pvalue:.3f}".rjust(20))
        print(line)
    print("\nдля сравнения: у повреждённых документов крупная полоса "
          f"{R[R['группа'] != 'контроль']['до_крупный'].mean():.2f} против "
          f"{D['чистый_крупный'].mean():.2f} у чистых, "
          f"то есть на {(R[R['группа'] != 'контроль']['до_крупный'].mean() / D['чистый_крупный'].mean() - 1) * 100:.0f}%")


if __name__ == "__main__":
    main()
