"""Does the graph tighten when the content stops meaning anything?

The prediction under test is that the coarse band falls because rare words stop
standing apart: with nothing to be rare *within*, an unusual token is no longer
anchored to a subject it belongs to, and it sinks into the mass with everything
else. That is a claim about the MST, not about word counts, and it makes three
checkable predictions for the long edges.

  hapax share      should fall: once-used words no longer sit at the far end
  endpoint rank    should move toward the text's own average: the long edges
                   stop being anchored on distinctive vocabulary
  spread of edge   should fall: a tighter graph has a narrower distribution of
  lengths          edge lengths, which is what a lower dimension looks like

The comparison is the corrected text against its absurd twin, which share every
sentence, so nothing here is explained by how the two are written.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from halluc_result import CJK
from long_edge_types import SKIP, stats, token_class
from mech_props import corpus_ranks

BASE = os.path.join(HERE, "..")
PAIR = tuple(os.environ.get("PAIR", "chained,flat").split(","))
RU = {"chained": "факты верны", "absurd": "абсурд, странные слова",
      "flat": "абсурд, обычные слова"}
# the coarse band averages d over q = 0.5 ... 0.9, i.e. over keeping the
# longest 50% down to the longest 10% of edges -- a mean of 30%. Reading the
# graph at a single 20% was inside that window but not centred on it, so the
# sweep covers the whole band and the headline number is taken at 0.30.
FRACS = (0.10, 0.20, 0.30, 0.40, 0.50)
KEY = "хотя бы один hapax, %"


def spread(text, emb, L=cfg.L_DEFAULT, seed=0):
    """How concentrated the MST edge lengths are -- the 'tightness' itself."""
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    w = minimum_spanning_tree(d).tocoo().data
    return {"разброс длин рёбер (cv)": float(w.std() / w.mean()),
            "длинные к медиане": float(np.mean(np.sort(w)[-len(w) // 5:])
                                       / np.median(w)),
            "средняя длина ребра": float(w.mean())}


def main():
    texts = {v: json.load(open(os.path.join(
        BASE, "results", f"halluc_{v}.json")))["answers"] for v in PAIR}
    bad = {q for q, t in json.load(open(os.path.join(
        BASE, "results", "halluc_raw.json")))["answers"].items()
        if len(CJK.findall(t)) > 20}
    qids = sorted(set(texts[PAIR[0]]) & set(texts[PAIR[1]]) - bad)
    emb = Embedder()
    ranks = corpus_ranks(list(texts["chained"].values()))

    rows = []
    for i, q in enumerate(qids):
        rec = {}
        ok = True
        for v in PAIR:
            sp = spread(texts[v][q], emb, seed=i)
            if sp is None:
                ok = False
                break
            rec |= {f"{v}::{k}@0.20": val for k, val in sp.items()}
            for fr in FRACS:
                s = stats(texts[v][q], emb, ranks, seed=i, frac=fr)
                if s is None:
                    ok = False
                    break
                rec |= {f"{v}::{k}@{fr:.2f}": val for k, val in s.items()}
            if not ok:
                break
        if ok:
            rec["qid"] = q
            rows.append(rec)
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(qids)}", flush=True)
    D = pd.DataFrame(rows).set_index("qid")
    D.to_csv(os.path.join(BASE, "results",
                          f"absurd_edges_{PAIR[1]}.csv"))

    keys = [k.split("::", 1)[1] for k in D.columns
            if k.startswith(f"{PAIR[0]}::")]
    out = []
    for k in keys:
        a, b = D[f"{PAIR[0]}::{k}"], D[f"{PAIR[1]}::{k}"]
        d = (b - a).dropna()
        frac = k.rsplit("@", 1)[1] if "@" in k else ""
        out.append({"признак": k.rsplit("@", 1)[0], "доля рёбер": frac,
                    RU[PAIR[0]]: a.mean(), RU[PAIR[1]]: b.mean(),
                    "сдвиг": d.mean(), "упало у": (d < 0).mean(),
                    "p": st.wilcoxon(d).pvalue if len(d) > 5 else np.nan})
    R = pd.DataFrame(out)
    pd.set_option("display.width", 210)
    pd.set_option("display.max_rows", 200)
    print(f"\n{len(D)} пар. Крупная полоса усредняет по доле длинных рёбер "
          f"от 0.10 до 0.50, центр 0.30\n")
    print("=== доля длинных рёбер, где хотя бы один конец hapax")
    h = R[R["признак"] == KEY].set_index("доля рёбер")
    print(h[[RU[PAIR[0]], RU[PAIR[1]], "сдвиг", "упало у", "p"]]
          .round(3).to_string())
    print("\n=== доля длинных рёбер между кусками ОДНОГО слова")
    ww = R[R["признак"] == "куски одного слова, %"].set_index("доля рёбер")
    print(ww[[RU[PAIR[0]], RU[PAIR[1]], "сдвиг", "упало у", "p"]]
          .round(3).to_string())
    print("\n=== всё остальное, на центре полосы (0.30)")
    print(R[(R["доля рёбер"].isin(["0.30", "0.20"])) & (R["признак"] != KEY)]
          .round(3).to_string(index=False))
    R.round(4).to_csv(os.path.join(
        BASE, "results", f"absurd_edges_summary_{PAIR[1]}.csv"), index=False)


if __name__ == "__main__":
    main()
