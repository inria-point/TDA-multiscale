"""A short interpretable linear model per band, chosen by held-out perturbations.

Features are all paired shifts -- each text against its own untouched source --
because that is the form in which both the counted and the judged properties
carry signal. They are standardised, so a coefficient reads as "per cent of
band displacement per one standard deviation of this property", and the sizes
are comparable across features measured in different units.

Selection is forward stepwise on cross-validated R2, and the folds are grouped
by perturbation: every candidate is scored on operations the model has never
seen. That is deliberately harsher than random folds, which would let a model
memorise what each operation does to the bands and then recognise it from any
of its side effects. A feature only survives here if it generalises to a kind
of damage that was absent from training.

Each band is fitted twice. The pooled fit answers "what distinguishes these
texts"; the within fit subtracts each perturbation's own mean from both sides
first and answers the harder question, "given that two texts underwent the same
operation, what says which one moved further".
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS
from judge_shift import load
from mech_props import NAMES as MECH
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")
MAX_FEATURES = 6
MIN_GAIN = 0.004      # a feature must buy this much held-out R2 to be kept
N_SPLITS = 5


def features(D):
    """Paired shifts only, with a Russian label and a family for each."""
    cols, label, family = [], {}, {}
    for m, ru in MECH.items():
        c = f"d_{m}"
        if c in D:
            cols.append(c)
            label[c] = ru
            family[c] = "механика"
    for k, meta in PROPS.items():
        c = f"d_{k}"
        if c in D:
            cols.append(c)
            label[c] = meta["ru"]
            family[c] = "судья"
    return cols, label, family


def forward(X, y, groups, names):
    cv = GroupKFold(n_splits=N_SPLITS)
    chosen, history = [], []
    best = 0.0
    while len(chosen) < MAX_FEATURES:
        scores = {}
        for c in names:
            if c in chosen:
                continue
            s = cross_val_score(LinearRegression(), X[chosen + [c]], y,
                                groups=groups, cv=cv, scoring="r2").mean()
            scores[c] = s
        c = max(scores, key=scores.get)
        if scores[c] - best < MIN_GAIN:
            break
        best = scores[c]
        chosen.append(c)
        history.append((c, best))
    return chosen, history


def demean(df, cols, by):
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def fit(D, band, cols, label, family, within):
    sub = D[[band, "perturbation"] + cols].dropna()
    if within:
        sub = demean(sub, [band] + cols, "perturbation")
    X = sub[cols]
    sd = X.std().replace(0, 1)
    Xs = (X - X.mean()) / sd
    y = sub[band]
    chosen, hist = forward(Xs, y, sub["perturbation"], cols)
    if not chosen:
        return None, None
    m = LinearRegression().fit(Xs[chosen], y)
    cv = GroupKFold(n_splits=N_SPLITS)
    held = cross_val_score(m, Xs[chosen], y, groups=sub["perturbation"],
                           cv=cv, scoring="r2").mean()
    rows = [{"полоса": band, "режим": "внутри" if within else "все тексты",
             "признак": label[c], "тип": family[c],
             "коэффициент": b, "1 sd признака": sd[c],
             "шаг": i + 1, "R2 накопл.": hist[i][1]}
            for i, (c, b) in enumerate(zip(chosen, m.coef_))]
    info = {"полоса": band, "режим": "внутри" if within else "все тексты",
            "n": len(sub), "признаков": len(chosen),
            "R2 на своих": m.score(Xs[chosen], y),
            "R2 на чужих пертурбациях": held}
    return pd.DataFrame(rows), info


def main():
    D = load()
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]
    cols, label, family = features(D)
    print(f"кандидатов: {len(cols)} "
          f"({sum(v == 'механика' for v in family.values())} механических, "
          f"{sum(v == 'судья' for v in family.values())} судейских)\n")

    tabs, infos = [], []
    for band in BANDS:
        for within in (False, True):
            t, i = fit(D, band, cols, label, family, within)
            if t is not None:
                tabs.append(t)
                infos.append(i)
    T = pd.concat(tabs, ignore_index=True)
    T.round(3).to_csv(os.path.join(BASE, "results", f"band_models{SUF}.csv"),
                      index=False)
    I = pd.DataFrame(infos)
    pd.set_option("display.width", 220)
    print(I.round(2).to_string(index=False))
    for band in BANDS:
        for mode in ("все тексты", "внутри"):
            g = T[(T["полоса"] == band) & (T["режим"] == mode)]
            if g.empty:
                continue
            print(f"\n=== {band} масштаб, {mode}")
            print(g[["шаг", "признак", "тип", "коэффициент", "1 sd признака",
                     "R2 накопл."]].round(2).to_string(index=False))
    figure(T)


def figure(T):
    modes = ["все тексты", "внутри"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for j, band in enumerate(BANDS):
        for i, mode in enumerate(modes):
            ax = axes[i][j]
            g = T[(T["полоса"] == band) & (T["режим"] == mode)]
            g = g.iloc[::-1]
            if g.empty:
                ax.axis("off")
                continue
            c = ["#b7791f" if t == "судья" else "#2b6cb0" for t in g["тип"]]
            ax.barh(range(len(g)), g["коэффициент"], color=c)
            ax.set_yticks(range(len(g)))
            ax.set_yticklabels(g["признак"], fontsize=9)
            ax.axvline(0, c="k", lw=1)
            ax.set_title(f"{band} масштаб — {mode}\n"
                         f"R² на чужих пертурбациях "
                         f"{g['R2 накопл.'].iloc[0]:.2f}", fontsize=10)
            ax.set_xlabel("% сдвига полосы на 1 sd признака")
            ax.grid(axis="x", alpha=0.25)
    h = [plt.Line2D([], [], color="#2b6cb0", lw=8, label="механика"),
         plt.Line2D([], [], color="#b7791f", lw=8, label="судья")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Короткая линейная модель каждой полосы. Признаки — сдвиги "
                 "относительно собственного исходника, стандартизованные;\n"
                 "отбор пошаговый по R² на отложенных пертурбациях",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    path = os.path.join(BASE, "figures", f"band_models{SUF}.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
