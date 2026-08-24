"""Stage 3: the map from a text property to what it does to the dimension.

Two things are read off the same run and then joined.

  what a perturbation DID to the text   — every perturbed text is re-scored on
      the feature battery and compared with its own source, so a style named
      "scientific register" is described by the properties it actually carries
      rather than by the properties it was asked to carry.

  what it did to the DIMENSION          — the paired change in d at every q,
      relative to the untouched arm of the same text.

The result is a profile per perturbation over q, which is what makes styles
and generation defects separable: two manipulations can move d by the same
amount and still differ in where along q they move it.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from text_features import features

BASE = os.path.join(os.path.dirname(__file__), "..")

DEFECTS = {"loop_phrase", "drop_sentence_boundaries", "strip_punctuation",
           "shuffle_words", "shuffle_within_sentences", "add_typos",
           "strip_digits", "lowercase", "drop_function_words"}


def paired(df):
    key = ["text_id", "mode", "q"]
    base = (df[df["perturbation"] == "identity"]
            .set_index(key)["d_hat"].rename("d0"))
    out = df[df["perturbation"] != "identity"].join(base, on=key)
    out = out.dropna(subset=["d0"])
    out["rel"] = (out["d_hat"] - out["d0"]) / out["d0"]
    out["delta"] = out["d_hat"] - out["d0"]
    return out


def profiles(p, mode):
    """perturbation x q matrix of mean relative change."""
    s = p[p["mode"] == mode]
    return s.pivot_table(index="perturbation", columns="q", values="rel") * 100


def counts(p):
    """Texts behind each perturbation. The style arms are much smaller than the
    mechanical ones — the OpenRouter key ran out of monthly quota partway — so
    the sample size has to travel with the numbers."""
    return (p[p["q"] == p["q"].min()]
            .groupby("perturbation")["text_id"].nunique().rename("n"))


def feature_shifts(texts_path, feat_cols=None):
    """How each perturbation moved every feature, in units of the source sd."""
    import json

    with open(texts_path) as f:
        texts = json.load(f)
    base = {k: features(v) for k, v in texts.get("identity", {}).items()}
    if not base:
        return None
    ref = pd.DataFrame(base).T
    cols = feat_cols or [c for c in ref.columns if ref[c].dtype != object]
    sd = ref[cols].std().replace(0, np.nan)

    rows = []
    for pname, by_key in texts.items():
        if pname == "identity":
            continue
        diffs = []
        for k, t in by_key.items():
            if k not in base:
                continue
            f = features(t)
            if not f:
                continue
            diffs.append({c: f.get(c, np.nan) - base[k].get(c, np.nan)
                          for c in cols})
        if diffs:
            m = pd.DataFrame(diffs).mean() / sd
            m["perturbation"] = pname
            rows.append(m)
    return pd.DataFrame(rows).set_index("perturbation")


def heatmap(mat, title, path, vmax=None, cmap="RdBu_r"):
    if mat.empty:
        return
    z = linkage(np.nan_to_num(mat.values), method="average")
    order = leaves_list(z)
    m = mat.iloc[order]
    vmax = vmax or np.nanpercentile(np.abs(m.values), 97)
    fig, ax = plt.subplots(figsize=(1.0 + 0.42 * m.shape[1], 0.34 * len(m) + 1.8))
    im = ax.imshow(m.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([f"{c:g}" if isinstance(c, (int, float)) else str(c)
                        for c in m.columns], fontsize=7,
                       rotation=90 if m.columns.dtype == object else 0)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=7.5)
    for i, name in enumerate(m.index):
        if name in DEFECTS:
            ax.get_yticklabels()[i].set_color("#b7791f")
    ax.set_xlabel("q")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    src = os.path.join(BASE, "results", "perturb_L201_coling.csv.gz")
    d = pd.read_csv(src)
    d = d[d["d_hat"] > 0]
    p = paired(d)
    print(f"{p['perturbation'].nunique()} пертурбаций, "
          f"{p['text_id'].nunique()} текстов", flush=True)

    n = counts(p)
    print("\nтекстов на пертурбацию:")
    for k, v in n.sort_values().items():
        print(f"  {k:32s} {v:4d}")

    for mode in ["q_small", "q_large", "q0.5_range"]:
        mat = profiles(p, mode)
        if mat.empty:
            continue
        mat.join(n).to_csv(os.path.join(BASE, "results", f"map_{mode}.csv"))
        heatmap(mat, f"Изменение d, % — {mode}\n(жёлтым выделены дефекты)",
                os.path.join(BASE, "figures", f"map_{mode}.png"))

    tp = os.path.join(BASE, "results", "perturbed_texts_coling.json")
    if os.path.exists(tp):
        fs = feature_shifts(tp)
        if fs is not None:
            fs.to_csv(os.path.join(BASE, "results", "map_feature_shifts.csv"))
            keep = fs.abs().max().sort_values(ascending=False).head(16).index
            heatmap(fs[keep], "Что пертурбация сделала с текстом\n"
                              "(сдвиг признака в единицах sd оригинала)",
                    os.path.join(BASE, "figures", "map_feature_shifts.png"),
                    cmap="PuOr_r")
            pd.set_option("display.width", 240)
            print("\n=== свойства, которые сдвинул каждый стиль (топ-6 по |сдвигу|)")
            for name in [i for i in fs.index if i.startswith("style_")]:
                r = fs.loc[name].dropna().sort_values(key=abs, ascending=False)
                top = ", ".join(f"{k} {v:+.1f}" for k, v in r.head(6).items())
                print(f"  {name:32s} {top}")


if __name__ == "__main__":
    main()
