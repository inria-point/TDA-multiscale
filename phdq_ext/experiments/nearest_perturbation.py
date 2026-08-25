"""Which perturbation each generator's profile is nearest to, without a basis.

Quadrant membership turned out to be unstable: the components themselves barely
move when the set changes (5 degrees), but the centre does, and a quadrant is
defined by the sign of a coordinate relative to that centre. Points near the
origin therefore change quadrant without changing at all.

Distances between profiles have no such dependence. This works in the raw
49-dimensional space and reports, for each generator, the perturbations closest
to it in shape and the margin over the next unrelated one, so the claim
"generators resemble mild looping" can be checked rather than eyeballed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pc_space import generator_profiles, perturbation_profiles

BASE = os.path.join(os.path.dirname(__file__), "..")


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b) / (na * nb) if na and nb else np.nan


def main():
    P = perturbation_profiles(os.path.join(BASE, "results",
                                           "perturb_L201_coling_all.csv.gz"))
    G, n = generator_profiles(os.path.join(BASE, "results",
                                           "coling_qphd_L201.csv.gz"))
    cols = [c for c in P.columns if c in G.columns]
    P, G = P[cols].fillna(0), G[cols].fillna(0)

    rows = []
    for m in G.index:
        y = G.loc[m].values
        sims = sorted(((cosine(y, P.loc[p].values), p) for p in P.index),
                      reverse=True)
        top = sims[:3]
        best_loop = next((s for s, p in sims if "loop" in p), np.nan)
        best_other = next((s for s, p in sims if "loop" not in p), np.nan)
        rows.append({
            "model": m, "n": int(n.get(m, 0)),
            # every number here belongs to a PAIR (model, perturbation), not
            # to the model: a cosine needs two vectors. The names say so.
            "ближайшая пертурбация": top[0][1],
            "cos(модель, ближайшая)": top[0][0],
            "max cos(модель, петля)": best_loop,
            "max cos(модель, не-петля)": best_other,
            "перевес петли": best_loop - best_other,
            "тройка": ", ".join(f"{p} {s:+.2f}" for s, p in top),
        })
    res = pd.DataFrame(rows).sort_values("перевес петли", ascending=False)
    res.to_csv(os.path.join(BASE, "results", "nearest_perturbation.csv"),
               index=False)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 62)
    print("каждое число — косинус между профилем МОДЕЛИ (строка) и профилем\n"
          "ПЕРТУРБАЦИИ (названа рядом), в исходном 49-мерном пространстве\n")
    print(res[["model", "n", "ближайшая пертурбация", "cos(модель, ближайшая)",
               "max cos(модель, петля)", "max cos(модель, не-петля)",
               "перевес петли"]].round(2).to_string(index=False))
    k = int((res["перевес петли"] > 0).sum())
    print(f"\nу {k} моделей из {len(res)} ближайшая по форме — зацикливание")
    print("\nтройка ближайших:")
    for _, r in res.iterrows():
        print(f"  {r['model']:18s} {r['тройка']}")


if __name__ == "__main__":
    main()
