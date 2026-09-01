"""Does the norm carry any of the dimension, or is it all angle?

edge_taxonomy.py showed that the mean norm of an edge's endpoints is flat along
the scale -- 38.6 at the 5th percentile of length, 37.0 at the 95th, a 4% drop
while the length itself triples. The cosine does all the work, falling 0.97 to
0.67 over the same range. That says the cloud is a thin shell of radius ~37 and
qPHD is measuring angles on it.

If so, projecting every token onto the unit sphere should leave the bands
essentially unchanged. That is a real prediction and it is cheap to test, so it
is worth testing rather than asserting: the norms do carry a frequency gradient
(r = -0.34 with log rank), and a small radial spread can still matter to an MST
when the angular spread is small.

Four clouds on the same texts, same seeds, paired:

  сырой              what every result in the project is computed on
  нормированный      v / ||v||, scaled back to the mean norm so the numbers stay
                     on a comparable scale -- d is scale-invariant, but keeping
                     the radius makes the edge lengths readable
  без стоков         the attention-sink point mass dropped (sink_cluster.py)
  без стоков + норм. both
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
from sink_cluster import COS_CUT, NORM_CUT, bands_of, sink_direction
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
N_TEXTS = int(os.environ.get("N_TEXTS", 40))
N_SEEDS = int(os.environ.get("N_SEEDS", 6))
REPLICATES = int(os.environ.get("REPLICATES", 32))
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
VARIANTS = ["сырой", "нормированный", "без стоков", "без стоков + норм."]


def run(e, seed, L=cfg.L_DEFAULT):
    if e.shape[0] < L:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(e.shape[0], size=L, replace=False)
    return qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, rng),
                                  replicates=REPLICATES))


def normed(e):
    """Unit sphere, rescaled to the cloud's own mean radius."""
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / n * n.mean()


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    u, n_seen = sink_direction(hum[300:340], emb)
    print(f"направление стока по {n_seen} токенам\n", flush=True)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        e, toks = emb.embed(s, return_tokens=True)
        e = e[[k for k, t in enumerate(toks) if t not in SKIP]]
        nrm = np.linalg.norm(e, axis=1)
        keep = ~(((e @ u) / nrm > COS_CUT) & (nrm < NORM_CUT))
        clouds = {"сырой": e, "нормированный": normed(e),
                  "без стоков": e[keep], "без стоков + норм.": normed(e[keep])}
        if min(c.shape[0] for c in clouds.values()) < cfg.L_DEFAULT:
            continue
        for seed in range(N_SEEDS):
            rec = {"текст": i, "seed": seed}
            ok = True
            for name, c in clouds.items():
                df = run(c, seed=2000 + seed)
                if df is None:
                    ok = False
                    break
                rec.update({f"{name}: {k}": v for k, v in bands_of(df).items()})
            if ok:
                rows.append(rec)
        print(f"  {i + 1}/{N_TEXTS}", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "space_variants.csv"), index=False)
    per = D.groupby("текст").mean(numeric_only=True)
    pd.set_option("display.width", 220)

    from scipy.stats import wilcoxon
    out = []
    for k in B:
        base = per[f"сырой: {k}"]
        row = {"полоса": k, "сырой": base.mean()}
        for v in VARIANTS[1:]:
            x = per[f"{v}: {k}"]
            row[v] = x.mean()
            row[f"{v}, %"] = ((x - base) / base * 100).mean()
            row[f"{v}, p"] = wilcoxon(base, x).pvalue
        out.append(row)
    out = pd.DataFrame(out).set_index("полоса")
    print(f"\n{len(per)} текстов, {N_SEEDS} зёрен усреднено\n")
    print(out.round(4).to_string())
    out.round(5).to_csv(os.path.join(BASE, "results", "space_variants_bands.csv"))


if __name__ == "__main__":
    main()
