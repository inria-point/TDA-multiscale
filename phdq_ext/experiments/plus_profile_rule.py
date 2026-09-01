"""Three bands above human normal at once: who does that?

Nobody, almost, except one generation of models. Across 626 human documents,
918 from llama-1 onward and 395 from the OPT/bloom/GPT-J era, the profile +++
at a 20% hinge occurs 4 times, 0 times and 116 times respectively. As a rule it
reads: all three bands more than 20% above the human mean means the text came
from an OPT-era model, at 97% precision.

It detects one generation, not machine text in general -- the modern cluster
never triggers it -- and it needs the human mean of a comparable corpus. What
makes it worth writing down is that it needs nothing else: no training set, no
reference model, three numbers computed from the document itself.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)


def profile(df, cols, th):
    P = df[cols].values
    H = np.sign(P) * np.clip(np.abs(P) - th, 0, None)
    return np.array(["".join("0" if v == 0 else ("+" if v > 0 else "−")
                             for v in r) for r in H])


def main():
    H = pd.read_csv(os.path.join(BASE, "results", "human_band_dev.csv"))
    G = pd.read_csv(os.path.join(BASE, "results", "gen_bands.csv"))
    C = pd.read_csv(os.path.join(BASE, "results", "gen_clusters.csv")
                    ).set_index("model")["кластер"]
    G["кластер"] = G["model"].map(C)
    n_old = int((G["кластер"] == 2).sum())

    rows = []
    for th in (10, 15, 20, 25, 30):
        ph = profile(H, [f"all_{b}" for b in B], th)
        pg = profile(G, [f"откл_{b}" for b in B], th)
        hit_h = int((ph == "+++").sum())
        hit_new = int(((pg == "+++") & (G["кластер"] == 1).values).sum())
        tp = int(((pg == "+++") & (G["кластер"] == 2).values).sum())
        hit = hit_h + hit_new + tp
        rows.append({"порог, %": th, "люди": hit_h, "новые модели": hit_new,
                     "старые модели": tp,
                     "точность": tp / hit if hit else np.nan,
                     "полнота": tp / n_old})
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 180)
    print(f"профиль +++ как правило (люди {len(H)}, новые "
          f"{int((G['кластер'] == 1).sum())}, старые {n_old})\n")
    print(T.round(3).to_string(index=False))
    T.round(4).to_csv(os.path.join(BASE, "results", "plus_profile_rule.csv"),
                      index=False)


if __name__ == "__main__":
    main()
