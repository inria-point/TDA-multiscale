"""What the lexical model is, drawn out.

It is one ordinary least-squares fit per band. Nothing is selected, tuned or
weighted: all ten counted properties go in, as paired shifts, and the band
comes out. The point of keeping it dumb is that the residual should be
"whatever no word count explains", and a model with choices in it would leave a
residual shaped by those choices instead.

Fitted on the annotated sample: 1125 perturbed texts, 40 perturbations, each
text measured against its own unperturbed source. Predictions used anywhere
else are out-of-fold, so a text never contributes to the model that explains
it.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS
from judge_shift import load
from mech_props import NAMES as MECH
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")
# one line each, so the figure can be read without the source
WHAT = {
    "ttr": "разных слов / всего слов",
    "hapax_share": "слов, встреченных ровно раз / всего",
    "word_entropy": "энтропия распределения частот слов",
    "mean_log_rank": "средний log частотного ранга слова по корпусу",
    "trigram_repeat": "доля повторяющихся троек слов",
    "adjacent_share": "повторов одного слова ближе 15 токенов / всех повторов",
    "punct_variety": "число различных знаков препинания",
    "sent_len_cv": "разброс длин предложений / средняя длина",
    "opening_uniformity": "доля предложений с самым частым зачином",
    "proper_noun_rate": "слов с заглавной не в начале предложения / всего",
}


def main():
    D = load()
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]
    mech = [f"d_{m}" for m in MECH if f"d_{m}" in D]
    judge = [f"d_{k}" for k in PROPS if f"d_{k}" in D]
    D = D.dropna(subset=BANDS + mech + judge).reset_index(drop=True)
    X = D[mech]
    sd = X.std().replace(0, 1)
    Xs = (X - X.mean()) / sd
    cv = KFold(5, shuffle=True, random_state=0)

    print(f"обучающая выборка: {len(D)} текстов, "
          f"{D['perturbation'].nunique()} пертурбаций, "
          f"{D['text_id'].nunique()} исходников")
    print(f"признаков: {len(mech)}, все входят, отбора нет\n")
    print("признак: что считается, sd сдвига по выборке")
    for m in MECH:
        if f"d_{m}" in D:
            print(f"  {MECH[m]:28s} {WHAT[m]:52s} sd={sd[f'd_{m}']:.3f}")

    coefs, preds = {}, {}
    print("\nкоэффициенты (% сдвига полосы на 1 sd признака) и качество\n")
    for band in BANDS:
        y = D[band]
        m = LinearRegression().fit(Xs, y)
        p = cross_val_predict(LinearRegression(), Xs, y, cv=cv)
        coefs[band] = pd.Series(m.coef_, index=[MECH[c[2:]] for c in mech])
        preds[band] = p
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"{band:8s} R2 вне складки {r2:.2f}   типичный промах "
              f"{np.abs(y - p).median():.1f}% против "
              f"{np.abs(y - y.mean()).median():.1f}% без модели")
    C = pd.DataFrame(coefs)
    print("\n" + C.round(2).to_string())
    C.round(3).to_csv(os.path.join(BASE, "results", f"mech_model{SUF}.csv"))
    figure(D, C, preds)


def figure(D, C, preds):
    fig = plt.figure(figsize=(16.5, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    for j, band in enumerate(BANDS):
        ax = fig.add_subplot(gs[0, j])
        y, p = D[band], preds[band]
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], c="#888", lw=1.2)
        ax.scatter(p, y, s=9, alpha=.4, color="#2b6cb0", linewidths=0)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ax.set_title(f"{band} масштаб\nR² вне складки {r2:.2f}; "
                     f"остаток sd {(y - p).std():.1f}% "
                     f"при sd полосы {y.std():.1f}%", fontsize=10)
        ax.set_xlabel("предсказано лексикой, %")
        ax.set_ylabel("измерено, %")
        ax.grid(alpha=.25)
    order = C.abs().max(axis=1).sort_values().index
    for j, band in enumerate(BANDS):
        ax = fig.add_subplot(gs[1, j])
        v = C[band].reindex(order)
        ax.barh(range(len(v)), v, color=["#c53030" if x < 0 else "#2b6cb0"
                                         for x in v])
        ax.set_yticks(range(len(v)))
        ax.set_yticklabels(v.index if j == 0 else [], fontsize=8)
        ax.axvline(0, c="k", lw=1)
        ax.set_xlabel("% на 1 sd признака")
        ax.set_title(f"{band} масштаб", fontsize=10)
        ax.grid(axis="x", alpha=.25)
    fig.suptitle(
        "Лексическая модель: обычная линейная регрессия полосы по десяти "
        "счётным свойствам текста.\nПризнаки — сдвиги относительно "
        "собственного исходника; отбора признаков нет, все десять входят; "
        f"{len(D)} текстов, {D['perturbation'].nunique()} пертурбаций",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(BASE, "figures", f"mech_model{SUF}.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
