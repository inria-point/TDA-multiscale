"""Project new perturbations into the basis that is already fitted.

Refitting the SVD with the new points in it would move the axes, and a shift
along PC2 could then not be told apart from PC2 having turned towards the new
points. The components are therefore taken from the existing decomposition and
the new profiles are only multiplied by them.

Usage: project_new.py <results csv.gz> [perturbation ...]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from pc_space import (ALL_MODES, full_profile, generator_profiles, paired,
                      perturbation_profiles, profiles)

BASE = os.path.join(HERE, "..")


def basis():
    """The components as fitted in pc_space.py, plus the column order."""
    pert = perturbation_profiles(os.path.join(BASE, "results",
                                              "perturb_L201_coling_all.csv.gz"))
    gen, _ = generator_profiles(os.path.join(BASE, "results",
                                             "coling_qphd_L201.csv.gz"))
    cols = [c for c in pert.columns if c in gen.columns]
    X = pd.concat([pert[cols], gen[cols]]).fillna(0)
    _, _, vt = np.linalg.svd(X.values, full_matrices=False)
    return vt[:3], cols


def main():
    path = sys.argv[1]
    want = sys.argv[2:] or None
    vt, cols = basis()

    d = pd.read_csv(path)
    p = paired(d[d["d_hat"] > 0])
    new = full_profile({m: profiles(p, m) for m in ALL_MODES})
    if want:
        new = new.loc[[i for i in new.index if i in want]]
    new = new.reindex(columns=cols).fillna(0)
    P = pd.DataFrame(new.values @ vt.T, index=new.index,
                     columns=["PC1", "PC2", "PC3"])
    P["|v|"] = np.linalg.norm(P.values, axis=1)

    ref = pd.read_csv(os.path.join(BASE, "results", "pert_map.csv"),
                      index_col=0)
    pd.set_option("display.width", 200)
    print("новые точки в старом базисе\n")
    print(P.round(1).sort_values("PC2").to_string())
    print("\nразмах старого набора для сравнения")
    print(ref[["PC1", "PC2", "PC3"]].agg(["min", "max"]).round(0).to_string())
    out = os.path.join(BASE, "results",
                       os.path.basename(path).split(".")[0] + "_projected.csv")
    P.round(1).to_csv(out)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
