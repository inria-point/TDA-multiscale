"""Everything the coarse band is computed from, for one document, before and
after the formatting repair.

The claim to be checked is that detokenisation destroys once-used words and
that the far end of the MST is built on them. Both halves are visible directly:
the list of once-used words in each version, and the list of the long edges
with the tokens they join.
"""
import os
import re
import sys
from collections import Counter

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from detok_repair import repair
from embedder import Embedder
from long_edge_types import SKIP, token_class

BASE = os.path.join(HERE, "..")
WORD = re.compile(r"[a-z']+")
FRAC = 0.30


def hapax_tokens(text, emb):
    """Once-used *tokens*, not words: the MST is built on tokens, so that is
    the level at which uniqueness is either kept or destroyed."""
    _, toks = emb.embed(text, return_tokens=True)
    keep = [t for t in toks if t not in SKIP]
    c = Counter(keep)
    return {t for t, n in c.items() if n == 1}, len(keep)


def long_edges(text, emb, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed_cached(text), None
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    t = [toks[i] for i in idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    sel = o[-max(1, int(FRAC * len(o))):]
    full = Counter(toks[j] for j in keep)
    rows = []
    for k in sel[::-1]:
        a, b = m.row[k], m.col[k]
        rows.append((m.data[k], t[a], t[b], token_class(t[a]), token_class(t[b]),
                     full[t[a]] == 1, full[t[b]] == 1,
                     abs(int(idx[a]) - int(idx[b]))))
    return rows


def main():
    import pandas as pd

    H = pd.concat([pd.read_csv(os.path.join(BASE, "results", f)) for f in
                   ("coarse_sign.csv", "coarse_sign_мелкий.csv",
                    "coarse_sign_средний.csv")]).drop_duplicates("id")
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    H["id"] = H["id"].astype(str)
    R = pd.read_csv(os.path.join(BASE, "results", "detok_repair.csv"))
    R["id"] = R["id"].astype(str)
    M = H.merge(pool[["id", "text"]], on="id").merge(
        R[["id", "до_крупный", "после_крупный"]], on="id")
    want = os.environ.get("TEXT_MATCH", "Alexander won")
    row = M[M["text"].str.contains(want, regex=False)].iloc[0]
    a, b = row["text"], repair(row["text"])
    emb = Embedder()

    ha, na = hapax_tokens(a, emb)
    hb, nb = hapax_tokens(b, emb)
    out = [f"# Один текст до и после починки форматирования",
           "",
           f"[{row['sub_source']}]  крупная полоса "
           f"{row['до_крупный']:.2f} → {row['после_крупный']:.2f} "
           f"({(row['после_крупный'] / row['до_крупный'] - 1) * 100:+.1f}%)",
           "", "## Однократные слова", "",
           f"до: {len(ha)} из {na} токенов ({len(ha)/na:.1%})   "
           f"после: {len(hb)} из {nb} ({len(hb)/nb:.1%})", "",
           "**появились после правки** (были разбиты на куски):", "",
           "`" + "`, `".join(sorted(hb - ha)) + "`", "",
           "**исчезли после правки** (обрывки, ставшие частью слов):", "",
           "`" + "`, `".join(sorted(ha - hb)) + "`", ""]
    for name, txt in (("ДО", a), ("ПОСЛЕ", b)):
        out += [f"## Длинные рёбра, {name} (верхние 30% MST)", "",
                "| длина | токен A | токен B | класс A | класс B | однокр. | разрыв |",
                "|---|---|---|---|---|---|---|"]
        for ln, ta, tb, ca, cb, oa, ob, gap in long_edges(txt, emb):
            mark = ("A" if oa else "") + ("B" if ob else "") or "—"
            out.append(f"| {ln:.1f} | `{ta}` | `{tb}` | {ca} | {cb} | {mark} "
                       f"| {gap} |")
        out.append("")
    p = os.path.join(BASE, "results", "one_text_edges.md")
    open(p, "w").write("\n".join(out))
    print("saved", os.path.relpath(p, BASE))
    print(f"\nоднократных токенов: до {len(ha)}/{na} ({len(ha)/na:.1%}), "
          f"после {len(hb)}/{nb} ({len(hb)/nb:.1%})")
    print("появились:", ", ".join(sorted(hb - ha)))
    print("исчезли:  ", ", ".join(sorted(ha - hb)))


if __name__ == "__main__":
    main()
