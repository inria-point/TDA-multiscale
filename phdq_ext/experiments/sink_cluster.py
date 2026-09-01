"""A point mass inside the cloud, and what it does to the fine band.

edge_taxonomy.py left one anomaly. Every cell behaves monotonically along the
scale except the very first bin, the 5% shortest edges, which alone carries
different-token pairs -- `.` next to `Ġthe` at a distance of 1 in a space where
two ordinary tokens stand 34 apart. Their norms gave it away: 26.2 where the
text's tokens average 37.7.

Those tokens are not a tail of the norm distribution, they are a spike at its
bottom: the 0.5th and the 1st percentile of the norm are the same number. They
are 2.4% of all tokens, they appear in every text without exception, they are
spread evenly through it, and they are almost entirely `Ġthe`, `.`, `Ġto`, `,`,
`Ċ` -- the emptiest tokens in the language. Mutual cosine inside the group is
0.987, and the direction is the same in every text (cosine with the group's
centre 0.993, against 0.233 for an ordinary token).

That is an attention sink: positions the model parks on when it has nothing to
attend to, whose hidden state collapses onto one fixed direction of fixed small
length. It is a property of ModernBERT, not of the text.

For qPHD this matters because a point mass is not a neutral passenger. Every
sink token is within a couple of units of every other, so a text with k of them
contributes on the order of k ultra-short MST edges -- and k varies from 6 to 31
across texts. The fine band is read off exactly these shortest edges.

So: identify the sinks by direction, then compute the three bands twice on the
same texts, with them and without them, several seeds each so the answer is not
buried in the +-8% resampling noise.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from qphd import qphd
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
N_TEXTS = int(os.environ.get("N_TEXTS", 24))
N_SEEDS = int(os.environ.get("N_SEEDS", 4))
REPLICATES = int(os.environ.get("REPLICATES", 16))
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
# membership by direction, not by norm: the norm cut is what found the group,
# but a direction is what defines it, and it generalises to a text whose overall
# scale happens to differ
NORM_CUT = 30.0
COS_CUT = 0.80


def sink_direction(texts, emb):
    """Mean direction of the low-norm tokens, over a sample of texts."""
    acc = []
    for s in texts:
        e = emb.embed(s)
        n = np.linalg.norm(e, axis=1)
        acc.append(e[n < NORM_CUT])
    v = np.concatenate(acc)
    u = v.mean(0)
    return u / np.linalg.norm(u), len(v)


def bands_of(df):
    out = {}
    for name, (mode, qmin) in BANDS.items():
        qmax = 0.5 if mode == "q0.5_range" else 0.9
        s = df[(df["mode"] == mode) & df["q"].between(qmin, qmax)
               & (df["d_hat"] > 0)]
        out[name] = float(s["d_hat"].mean()) if len(s) else np.nan
    return out


def run(e, seed, L=cfg.L_DEFAULT):
    if e.shape[0] < L:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(e.shape[0], size=L, replace=False)
    return qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, rng),
                                  replicates=REPLICATES))


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    u, n_seen = sink_direction(hum[300:340], emb)
    print(f"направление стока оценено по {n_seen} токенам 40 текстов\n",
          flush=True)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        e, toks = emb.embed(s, return_tokens=True)
        keep = np.array([k for k, t in enumerate(toks) if t not in SKIP])
        e = e[keep]
        nrm = np.linalg.norm(e, axis=1)
        cos = (e @ u) / nrm
        is_sink = (cos > COS_CUT) & (nrm < NORM_CUT)
        if e.shape[0] - is_sink.sum() < cfg.L_DEFAULT:
            continue
        for seed in range(N_SEEDS):
            a = run(e, seed=1000 + seed)
            b = run(e[~is_sink], seed=1000 + seed)
            if a is None or b is None:
                continue
            rows.append({"текст": i, "seed": seed, "токенов": e.shape[0],
                         "стоков": int(is_sink.sum()),
                         "доля стоков, %": is_sink.mean() * 100,
                         **{f"со стоками: {k}": v
                            for k, v in bands_of(a).items()},
                         **{f"без стоков: {k}": v
                            for k, v in bands_of(b).items()}})
        print(f"  {i + 1}/{N_TEXTS}  стоков {int(is_sink.sum())} "
              f"из {e.shape[0]}", flush=True)

    D = pd.DataFrame(rows)
    out = os.path.join(BASE, "results", "sink_cluster.csv")
    D.to_csv(out, index=False)
    per = D.groupby("текст").mean(numeric_only=True)
    pd.set_option("display.width", 220)

    print(f"\n{len(per)} текстов, {N_SEEDS} зёрен усреднено, "
          f"replicates={REPLICATES}")
    print(f"стоков на текст: {per['стоков'].mean():.1f} "
          f"({per['доля стоков, %'].mean():.2f}% токенов), "
          f"разброс {per['стоков'].min():.0f}-{per['стоков'].max():.0f}\n")

    from scipy.stats import wilcoxon
    tbl = []
    for k in B:
        a, b = per[f"со стоками: {k}"], per[f"без стоков: {k}"]
        rel = (b - a) / a * 100
        st = wilcoxon(a, b)
        tbl.append({"полоса": k, "со стоками": a.mean(), "без стоков": b.mean(),
                    "сдвиг, %": rel.mean(), "сд сдвига": rel.std(),
                    "p": st.pvalue})
    tbl = pd.DataFrame(tbl).set_index("полоса")
    print("удаление стоков (2.4% точек) — что делается с полосами\n")
    print(tbl.round(4).to_string())
    tbl.round(5).to_csv(os.path.join(BASE, "results", "sink_bands.csv"))

    # does the number of sinks predict the band by itself?
    print("\nкорреляция числа стоков с полосой (со стоками):")
    for k in B:
        r = np.corrcoef(per["доля стоков, %"], per[f"со стоками: {k}"])[0, 1]
        print(f"  {k:10s} r = {r:+.3f}")


if __name__ == "__main__":
    main()
