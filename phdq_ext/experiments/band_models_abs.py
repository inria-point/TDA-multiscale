"""The same models on absolute values instead of paired shifts.

Two versions, because "absolute" can mean two different things.

A -- absolute properties, displacement as target. The text is described by what
it is rather than by how far it moved from its source, but we still ask how far
it moved. This isolates the value of pairing on the feature side alone, and the
target is unchanged so R2 is directly comparable to the paired models.

B -- absolute properties, absolute dimension as target. No source enters
anywhere: given a text, predict its d in each band. The 104 untouched sources
join as ordinary rows, since a model of dimension should cover human text too.
B is the harder and more useful question -- a detector never sees the original.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from band_models import MAX_FEATURES, MIN_GAIN, N_SPLITS, demean
from judge import PROPS
from judge_shift import load
from mech_props import NAMES as MECH
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")


def absolute_features(D):
    cols, label, family = [], {}, {}
    for m, ru in MECH.items():
        if m in D:
            cols.append(m)
            label[m] = ru
            family[m] = "механика"
    for k, meta in PROPS.items():
        if k in D:
            cols.append(k)
            label[k] = meta["ru"]
            family[k] = "судья"
    return cols, label, family


def with_sources(D, cols):
    """Add the untouched sources as ordinary rows for the absolute target."""
    A = pd.read_csv(os.path.join(BASE, "results",
                                 f"per_text_bands_abs{SUF}.csv"))
    A["text_id"] = A["text_id"].astype(str).str.replace("^coling::", "",
                                                        regex=True)
    # D carries the relative bands under the same names; drop them first or
    # the merge silently keeps the displacement as the absolute target
    P = D.drop(columns=BANDS).merge(A, on=["perturbation", "text_id"],
                                    how="inner")
    src = D.drop_duplicates("text_id").copy()
    ren = {f"src_{m}": m for m in MECH if f"src_{m}" in src}
    ren |= {f"base_{k}": k for k in PROPS if f"base_{k}" in src}
    src = src[["text_id"] + list(ren)].rename(columns=ren)
    src["perturbation"] = "identity"
    src = src.merge(A[A["perturbation"] == "identity"],
                    on=["perturbation", "text_id"], how="inner")
    keep = ["perturbation", "text_id"] + cols + BANDS
    return pd.concat([P[keep], src[keep]], ignore_index=True)


def forward(X, y, names):
    cv = KFold(N_SPLITS, shuffle=True, random_state=0)
    chosen, hist, best = [], [], 0.0
    while len(chosen) < MAX_FEATURES:
        sc = {c: cross_val_score(LinearRegression(), X[chosen + [c]], y, cv=cv,
                                 scoring="r2").mean()
              for c in names if c not in chosen}
        c = max(sc, key=sc.get)
        if sc[c] - best < MIN_GAIN:
            break
        best = sc[c]
        chosen.append(c)
        hist.append(best)
    return chosen, hist


def run(sub, target, cols, label, family, tag, mode):
    X = sub[cols]
    sd = X.std().replace(0, 1)
    Xs = (X - X.mean()) / sd
    y = sub[target]
    chosen, hist = forward(Xs, y, cols)
    if not chosen:
        return None, None
    m = LinearRegression().fit(Xs[chosen], y)
    pred = cross_val_predict(m, Xs[chosen], y,
                             cv=KFold(N_SPLITS, shuffle=True, random_state=0))
    err, base = np.abs(y - pred), np.abs(y - y.mean())
    rows = [{"вариант": tag, "полоса": target, "режим": mode,
             "шаг": i + 1, "признак": label[c], "тип": family[c],
             "коэффициент": b, "R2 накопл.": hist[i]}
            for i, (c, b) in enumerate(zip(chosen, m.coef_))]
    info = {"вариант": tag, "полоса": target, "режим": mode, "n": len(y),
            "разброс цели (sd)": y.std(),
            "R2": 1 - (err ** 2).sum() / (base ** 2).sum(),
            "ошибка: медиана": err.median(),
            "без модели: медиана": base.median(),
            "ошибка: 90-й": err.quantile(0.9),
            "без модели: 90-й": base.quantile(0.9)}
    return pd.DataFrame(rows), info


def main():
    D = load()
    cols, label, family = absolute_features(D)
    tabs, infos = [], []

    # A: absolute properties -> displacement
    for band in BANDS:
        sub = D[[band, "perturbation"] + cols].dropna()
        for mode in ("все тексты", "внутри"):
            s = demean(sub, [band] + cols, "perturbation") if mode == "внутри" \
                else sub
            t, i = run(s, band, cols, label, family, "A: сдвиг полосы", mode)
            if t is not None:
                tabs.append(t)
                infos.append(i)

    # B: absolute properties -> absolute dimension
    W = with_sources(D, cols).dropna()
    print(f"вариант B: {len(W)} строк, из них исходников "
          f"{(W.perturbation == 'identity').sum()}")
    for band in BANDS:
        t, i = run(W, band, cols, label, family, "B: сама размерность",
                   "все тексты")
        if t is not None:
            tabs.append(t)
            infos.append(i)

    T = pd.concat(tabs, ignore_index=True)
    T.round(3).to_csv(os.path.join(BASE, "results",
                                   f"band_models_abs{SUF}.csv"), index=False)
    I = pd.DataFrame(infos)
    pd.set_option("display.width", 240)
    print("\n" + I.round(2).to_string(index=False))
    for _, r in I.iterrows():
        g = T[(T["вариант"] == r["вариант"]) & (T["полоса"] == r["полоса"]) &
              (T["режим"] == r["режим"])]
        print(f"\n=== {r['вариант']} | {r['полоса']} | {r['режим']}")
        print(g[["шаг", "признак", "тип", "коэффициент", "R2 накопл."]]
              .round(2).to_string(index=False))
    figure(T, I)


def panel(ax, g, r, unit, title):
    """One coefficient chart, captioned with what the fit is actually worth."""
    g = g.iloc[::-1]
    c = ["#b7791f" if t == "судья" else "#2b6cb0" for t in g["тип"]]
    ax.barh(range(len(g)), g["коэффициент"], color=c)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g["признак"], fontsize=9)
    ax.axvline(0, c="k", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{unit} на 1 sd признака")
    ax.grid(axis="x", alpha=0.25)
    # R2 alone hides the scale, so the median miss is given against the
    # do-nothing baseline that always predicts the average
    gain = 1 - r["ошибка: медиана"] / r["без модели: медиана"]
    ax.text(0.98, 0.04,
            f"R² {r['R2']:.2f}\n"
            f"типичный промах {r['ошибка: медиана']:.2f} {unit}\n"
            f"без модели {r['без модели: медиана']:.2f} → выигрыш {gain:.0%}\n"
            f"худшая десятая: {r['ошибка: 90-й']:.2f} против "
            f"{r['без модели: 90-й']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e6" if gain < 0.15
                      else "#eef7ee", ec="#bbb", lw=0.8))


def figure(T, I):
    B = I[I["вариант"].str.startswith("B")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, band in zip(axes, BANDS):
        g = T[(T["вариант"].str.startswith("B")) & (T["полоса"] == band)]
        r = B[B["полоса"] == band].iloc[0]
        panel(ax, g, r, "единиц d",
              f"{band} масштаб — сама размерность d")
    h = [plt.Line2D([], [], color="#2b6cb0", lw=8, label="механика"),
         plt.Line2D([], [], color="#b7791f", lw=8, label="судья")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Абсолютные признаки против абсолютной размерности: "
                 "ни один исходник не участвует\n"
                 "879 текстов, из них 104 неизменённых", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = os.path.join(BASE, "figures", f"band_models_abs{SUF}.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
