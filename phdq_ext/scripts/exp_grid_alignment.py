"""Can the rounding artefact be avoided by aligning the n-grid instead?

If L is fixed, the grid can be chosen so that q*m is an integer for every q on
the grid and every m = n-1, removing the need to round at all. With a q-step of
0.05 (q = j/20) it suffices that every m be a multiple of 20.

This compares four arms on the same texts:
  aligned_discrete   aligned grid, integer trimming  -- rounding "not needed"
  aligned_frac       aligned grid, fractional trimming
  standard_frac      log grid on [0.2L, L], fractional trimming  (the config)
  standard_discrete  log grid, integer trimming                  (the artefact)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from data import GENRES, load
from embedder import Embedder
from qphd import qphd

BASE = os.path.join(os.path.dirname(__file__), "..")

# m multiples of 20, log-spaced as closely as the alignment allows
M_ALIGNED = np.array([60, 80, 100, 120, 140, 180, 220, 260])
L_ALIGNED = 261  # so that the largest m = L-1 = 260 is also a multiple of 20
FRAC_ALIGNED = tuple((M_ALIGNED + 1) / L_ALIGNED)

Q_LIST = tuple(np.round(np.arange(0, 0.901, 0.05), 2))

ARMS = [
    ("aligned_floor", L_ALIGNED, FRAC_ALIGNED, "floor"),
    ("aligned_round", L_ALIGNED, FRAC_ALIGNED, "round"),
    ("aligned_frac", L_ALIGNED, FRAC_ALIGNED, "fractional"),
    ("standard_floor", 256, cfg.LOG8, "floor"),
    ("standard_round", 256, cfg.LOG8, "round"),
    ("standard_frac", 256, cfg.LOG8, "fractional"),
]


def roughness(y):
    """Mean absolute second difference: how far the curve is from smooth."""
    return float(np.abs(np.diff(np.asarray(y), 2)).mean())


def main():
    emb = Embedder()
    clouds = []
    for genre in GENRES:
        got = 0
        for i, (tid, text) in enumerate(load(genre, "human")):
            if got >= 6:
                break
            e = emb.embed_cached(text, cache_key=f"human_{genre}_{i}")
            if e.shape[0] < 600:
                continue
            clouds.append((f"{genre}_{i}", e))
            got += 1
    print(f"{len(clouds)} texts", flush=True)

    rows = []
    for name, e in clouds:
        for seed in range(3):
            for arm, L, fracs, trim in ARMS:
                rng = np.random.default_rng(10007 * seed + hash(name) % 9973)
                pts = e[rng.choice(e.shape[0], L, replace=False)]
                df = qphd(
                    pts,
                    q_list=Q_LIST,
                    modes=["q_small"],
                    n_fraction_list=fracs,
                    replicates=16,
                    replace=False,
                    trim=trim,
                    pool=cfg.make_pool(e, L, rng),
                    rng=rng,
                )
                df["arm"] = arm
                df["text"] = name
                rows.append(df)
        print(f"  {name} done", flush=True)

    d = pd.concat(rows)
    d = d[d["d_hat"] > 0]
    d.to_csv(os.path.join(BASE, "results", "exp_grid_alignment.csv"), index=False)

    piv = d.pivot_table(index="q", columns="arm", values="d_hat")
    piv = piv[[a[0] for a in ARMS]]
    pd.set_option("display.width", 200)
    print("\nmean d_hat, q_small\n")
    print(piv.round(3).to_string())
    print("\nroughness (mean |2nd difference|):")
    for c in piv.columns:
        print(f"  {c:20s} {roughness(piv[c]):.4f}")

    # where integer trimming misses despite the alignment
    print("\nq*m that binary floating point turns into a wrong floor:")
    for q in Q_LIST:
        for m in M_ALIGNED:
            t = q * m
            if abs(t - round(t)) < 1e-9 and int(np.floor(t)) != round(t):
                print(f"  q={q}  m={m}:  q*m={t!r}  floor={int(np.floor(t))}"
                      f"  expected={round(t)}")


if __name__ == "__main__":
    main()
