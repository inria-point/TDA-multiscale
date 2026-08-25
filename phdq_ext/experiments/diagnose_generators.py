"""Stage 4: read a generator's qPHD profile through the perturbation map.

Each perturbation gives a profile: what it does to d at every q. A generator
also has a profile — its texts against human texts in the same domains. If the
perturbation profiles span the space, a generator's profile can be expressed in
them, and the weights say what the generator's output behaves like: a style, a
defect, or a mixture.

Two readings are produced and they answer different questions.

  attribution   least squares of the generator profile on the perturbation
                profiles. Says which manipulations reproduce its shape.
  feature shift how the generator's texts differ from human texts on the
                feature battery, in units of the human spread. Says what is
                actually different about the writing.

Agreement between the two is the check: an attribution to "lexical repetition"
should come with a measured drop in type-token ratio, or it is an artefact of
an ill-conditioned fit.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_property_map import DEFECTS, paired, profiles

BASE = os.path.join(os.path.dirname(__file__), "..")
MODES = ["q_small", "q_large"]
ALL_MODES = ["q_small", "q_large", "q0.5_range"]


def generator_profiles(qphd, min_texts=25):
    """Mean d per (model, mode, q), expressed against human text in the same
    domains, so that a generator's favourite domains do not masquerade as its
    style."""
    d = qphd[qphd["d_hat"] > 0].copy()
    human = (d[d["is_human"]]
             .groupby(["sub_source", "mode", "q"])["d_hat"].mean()
             .rename("h"))
    d = d.join(human, on=["sub_source", "mode", "q"])
    d = d.dropna(subset=["h"])
    d["rel"] = (d["d_hat"] - d["h"]) / d["h"] * 100
    n = d[(d["mode"] == "q_small") & (d["q"] == 0)].groupby("model").size()
    keep = n[n >= min_texts].index
    out = {}
    for mode in MODES:
        s = d[(d["mode"] == mode) & d["model"].isin(keep) & (~d["is_human"])]
        out[mode] = s.pivot_table(index="model", columns="q", values="rel")
    return out, n


def stack(profiles_by_mode, index=None):
    """One long vector per row: q_small and q_large concatenated.

    Using both regimes doubles the number of equations, and more importantly
    the two carry different information — several manipulations that look
    alike in one regime separate in the other.
    """
    parts = []
    for mode in MODES:
        m = profiles_by_mode[mode]
        if index is not None:
            m = m.reindex(index)
        m = m.copy()
        m.columns = [f"{mode}:{c}" for c in m.columns]
        parts.append(m)
    return pd.concat(parts, axis=1)


def dedupe(basis, thresh=0.985):
    """Drop perturbations whose profile duplicates one already kept.

    shuffle_words and shuffle_within_sentences, for instance, correlate at
    almost one; keeping both makes the basis ill-conditioned without adding
    information.
    """
    keep, merged = [], {}
    v = basis.fillna(0)
    for name in basis.index:
        x = v.loc[name].values
        hit = None
        for k in keep:
            y = v.loc[k].values
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            if denom and float(x @ y) / denom > thresh:
                hit = k
                break
        if hit:
            merged.setdefault(hit, []).append(name)
        else:
            keep.append(name)
    return basis.loc[keep], merged


def split_level_shape(row):
    """Separate the overall level of an effect from how it varies with q.

    Comparing raw profiles compares levels: any two effects that are positive
    everywhere correlate at nearly one, which is why every generator first
    appeared to match everything. The shape — the deviation from a constant —
    is what distinguishes a manipulation that acts uniformly from one that
    acts only at small or only at large q.
    """
    v = np.nan_to_num(row.values.astype(float))
    level = float(v.mean())
    shape = v - level
    norm = float(np.linalg.norm(shape))
    return level, (shape / norm if norm else shape), norm


def match(gen_row, basis, top=3):
    """Which perturbations the generator resembles in shape, level apart."""
    g_level, g_shape, g_norm = split_level_shape(gen_row)
    rows = []
    for name, row in basis.iterrows():
        b_level, b_shape, b_norm = split_level_shape(row)
        if b_norm == 0 or g_norm == 0:
            continue
        rows.append((name, float(g_shape @ b_shape), b_level))
    rows.sort(key=lambda r: -r[1])
    return rows[:top], g_level, g_norm


def main():
    qphd = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    pert = pd.read_csv(os.path.join(BASE, "results", "perturb_L201_coling.csv.gz"))
    pert = pert[pert["d_hat"] > 0]
    p = paired(pert)

    gen, n = generator_profiles(qphd)
    print(f"генераторов с достаточной выборкой: {len(gen['q_small'])}", flush=True)

    basis_by_mode = {m: profiles(p, m) for m in MODES}
    basis = stack(basis_by_mode)
    basis, merged = dedupe(basis)
    if merged:
        print("\nобъединены как неразличимые по форме:")
        for k, v in merged.items():
            print(f"  {k} <- {', '.join(v)}")

    G = stack(gen, index=gen["q_small"].index)
    rows = []
    for model in G.index:
        top, level, shape_norm = match(G.loc[model], basis)
        rows.append({
            "model": model, "n": int(n.get(model, 0)),
            "уровень, %": level,
            "форма": shape_norm / np.sqrt(len(basis.columns)),
            "похоже на": "; ".join(f"{k} {c:+.2f}" for k, c, _ in top),
            "лучшее": top[0][0] if top else "",
            "косинус": top[0][1] if top else np.nan,
            "дефект?": top[0][0] in DEFECTS if top else False,
        })
    res = pd.DataFrame(rows).sort_values("уровень, %")
    res.to_csv(os.path.join(BASE, "results", "generator_diagnosis.csv"),
               index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 68)
    print("\n=== на что похож профиль генератора (обе шкалы q сразу)")
    print(res[["model", "n", "уровень, %", "форма", "лучшее", "косинус",
               "дефект?"]].round(2).to_string(index=False))
    print("\n=== три ближайших по форме")
    for _, r in res.iterrows():
        print(f"  {r['model']:18s} {r['похоже на']}")

    # картинка: профили генераторов по q, все три режима
    d = qphd[qphd["d_hat"] > 0].copy()
    human = (d[d["is_human"]].groupby(["sub_source", "mode", "q"])["d_hat"]
             .mean().rename("h"))
    d = d.join(human, on=["sub_source", "mode", "q"]).dropna(subset=["h"])
    d["rel"] = (d["d_hat"] - d["h"]) / d["h"] * 100
    keep = n[n >= 25].index.difference(["human"])

    fig, axes = plt.subplots(1, len(ALL_MODES), figsize=(6.4 * len(ALL_MODES), 5.6),
                             sharey=True)
    # colour by the sign of the deviation, so the two families are visible
    for ax, mode in zip(axes, ALL_MODES):
        M = (d[(d["mode"] == mode) & d["model"].isin(keep)]
             .pivot_table(index="model", columns="q", values="rel"))
        base = M[[c for c in M.columns if c <= 0.3]].mean(axis=1)
        for model in base.sort_values().index:
            ax.plot(M.columns, M.loc[model], lw=1.2, alpha=0.85,
                    color="#c53030" if base[model] > 0 else "#2b6cb0",
                    label=model)
        ax.axhline(0, c="k", lw=1.2)
        ax.set_title(mode, fontsize=11)
        ax.set_xlabel("q")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("d относительно человека в том же домене, %")
    axes[-1].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle("Профили генераторов: красные выше человека на малых q, "
                 "синие ниже\nпересечение нуля по q — отдельная ось, "
                 "не сводимая к уровню")
    fig.tight_layout()
    path = os.path.join(BASE, "figures", "generator_profiles.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
