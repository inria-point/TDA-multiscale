"""Is one text's dimension a property of the text or of the sample?

qPHD draws L tokens from a text at random. A 300-word text hands over almost
everything; a 1700-word text hands over a sixth, and a different sixth each
time. If d moves as much between two draws from one text as it does between two
texts, then "this document deviates 25% from the corpus" is a statement about
the draw, and no selection rule built on it can work.

So: the same texts, several seeds each, and the within-text spread put beside
the between-text spread. Length goes in the same table, since it is the other
thing that could be producing the deviations by itself.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
import numpy as _np
from embedder import Embedder
from qphd import qphd
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
N_TEXTS = int(os.environ.get("N_TEXTS", 60))
N_SEEDS = int(os.environ.get("N_SEEDS", 6))


def bands_of(df):
    out = {}
    for name, (mode, qmin) in BANDS.items():
        qmax = 0.5 if mode == "q0.5_range" else 0.9
        s = df[(df["mode"] == mode) & df["q"].between(qmin, qmax)
               & (df["d_hat"] > 0)]
        out[name] = float(s["d_hat"].mean()) if len(s) else np.nan
    return out


def one(text, emb, seed, L=cfg.L_DEFAULT):
    """Exactly the pipeline's own call, with the seed as the only difference."""
    e = emb.embed_cached(text)
    if e.shape[0] < L:
        return None
    rng = _np.random.default_rng(seed)
    idx = rng.choice(e.shape[0], size=L, replace=False)
    return qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, rng),
                                  replicates=10))


def main():
    T = pd.read_csv(os.path.join(BASE, "results", "human_band_dev.csv"))
    T["id"] = T["id"].astype(str)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    T = T.merge(pool[["id", "text"]], on="id")
    T["макс_откл"] = T[[f"all_{b}" for b in B]].abs().max(axis=1)
    # half from the tail, half from the middle: the question is whether the
    # tail is stable, and the middle says what stability looks like
    S = pd.concat([T.nlargest(N_TEXTS // 2, "макс_откл"),
                   T.nsmallest(N_TEXTS // 2, "макс_откл")])
    emb = Embedder()

    rows = []
    for i, r in enumerate(S.itertuples()):
        for seed in range(N_SEEDS):
            df = one(r.text, emb, seed=1000 + seed)
            if df is None:
                continue
            rows.append({"id": r.id, "seed": seed,
                         "слов": len(r.text.split()),
                         "хвост": r.макс_откл > 20, **bands_of(df)})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(S)}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "d_stability.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"\n{S['id'].nunique()} текстов x {N_SEEDS} зёрен\n")
    out = []
    for b in B:
        within = D.groupby("id")[b].std().mean()
        between = D.groupby("id")[b].mean().std()
        rel = D.groupby("id")[b].std() / D.groupby("id")[b].mean() * 100
        out.append({"полоса": b, "разброс внутри текста (sd)": within,
                    "разброс между текстами (sd)": between,
                    "доля шума": within / between,
                    "внутри, % от d": rel.mean()})
    print(pd.DataFrame(out).round(2).to_string(index=False))
    print("\nсвязь d с длиной текста:")
    m = D.groupby("id").agg({**{b: "mean" for b in B}, "слов": "first",
                             "хвост": "first"})
    from scipy import stats as st
    for b in B:
        r = st.spearmanr(m["слов"], m[b])
        print(f"  {b:8s} rho={r.statistic:+.2f}  p={r.pvalue:.4f}")
    print("\nдлина в хвосте против середины:")
    print(m.groupby("хвост")["слов"].describe()[["count", "mean", "50%"]]
          .round(0).to_string())


if __name__ == "__main__":
    main()
