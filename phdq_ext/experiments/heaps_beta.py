"""Объясняет ли показатель Хипса β сдвиги крупной полосы?

report_1.pdf, теорема B.3: каркасная полоса имеет показатель κ_fr = β(1 − α/d_g),
то есть зависит ровно от двух величин — β (рост числа типов с числом токенов,
N ~ n^β) и d_g (размерность каркаса). Наши пертурбации меняют словарь, значит
меняют и β. Прежде чем вводить в модель неоднородность каркаса, надо проверить,
не объясняется ли всё уже имеющимся β.

β считается по той же сетке подгонки, что и qPHD: для каждого n′ из сетки берётся
среднее число различных типов в подвыборке, и β — наклон log N против log n′.
Эмбеддинги не нужны, только токены.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from coling_data import human_texts
from edge_taxonomy import SKIP
from embedder import MODEL_NAME
from pert_geometry import PERTS as MECH
from pert_geometry_llm import PERTS as LLM
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS, SEEDS, REPS = 50, 3, 20


def beta_of(toks, grid):
    """β и q* по одной выборке L токенов."""
    out = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(len(toks), size=cfg.L_DEFAULT, replace=False)
        t = [toks[j] for j in idx]
        r2 = np.random.default_rng(7)
        Ns = []
        for n in grid:
            v = [len(set(t[j] for j in r2.choice(len(t), size=n, replace=False)))
                 for _ in range(REPS)]
            Ns.append(np.mean(v))
        b = np.polyfit(np.log(grid), np.log(Ns), 1)[0]
        N = Ns[-1]
        out.append((b, (cfg.L_DEFAULT - N) / (cfg.L_DEFAULT - 1)))
    return float(np.mean([x[0] for x in out])), float(np.mean([x[1] for x in out]))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    grid = np.array(cfg.aligned_grid(cfg.L_DEFAULT))
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            t = [x for x in tok.tokenize(txt) if x not in SKIP]
            if len(t) < cfg.L_DEFAULT:
                continue
            b, q = beta_of(t, grid)
            rows.append(("механические", str(i), label, b, q))
        if (i + 1) % 10 == 0:
            print(f"  механические: {i+1}", flush=True)
    src = dict(human_texts(120, min_words=280))
    store = {}
    for label, key, fname in LLM:
        with open(os.path.join(BASE, "results", fname)) as f:
            store[label] = json.load(f)[key]
    for n, (k, text) in enumerate(src.items()):
        if n >= N_TEXTS:
            break
        for label, txt in [("исходный", text)] + [
                (lab, store[lab].get(k)) for lab, _, _ in LLM]:
            if txt is None:
                continue
            t = [x for x in tok.tokenize(txt) if x not in SKIP]
            if len(t) < cfg.L_DEFAULT:
                continue
            b, q = beta_of(t, grid)
            rows.append(("LLM", k, label, b, q))
    D = pd.DataFrame(rows, columns=["набор", "текст", "условие", "beta", "q*"])
    D.to_csv(os.path.join(BASE, "results", "heaps_beta.csv"), index=False)

    B = pd.concat([pd.read_csv(os.path.join(BASE, "results", f)).set_index(
        "условие") for f in ("pert_geometry_shifts.csv",
                             "pert_geometry_llm_shifts.csv")])
    B = B[~B.index.duplicated(keep="first")]
    sh = {}
    for nab, g in D.groupby("набор"):
        for c in ("beta", "q*"):
            piv = g.pivot_table(index="текст", columns="условие", values=c)
            if "исходный" not in piv:
                continue
            for cond in piv.columns:
                if cond == "исходный":
                    continue
                m = piv["исходный"].notna() & piv[cond].notna()
                if m.sum() < 8:
                    continue
                sh.setdefault(cond, {})[c] = (
                    (piv.loc[m, cond] - piv.loc[m, "исходный"])).median()
    S = pd.DataFrame(sh).T.join(B[["крупный", "мелкий", "средний"]]).dropna(
        subset=["крупный"])
    S.to_csv(os.path.join(BASE, "results", "heaps_beta_shifts.csv"))
    print(f"\nβ и q* у неиспорченных текстов: "
          f"β = {D[D['условие']=='исходный']['beta'].mean():.3f}, "
          f"q* = {D[D['условие']=='исходный']['q*'].mean():.3f}\n")
    mech = S.index.isin(pd.read_csv(os.path.join(
        BASE, "results", "pert_geometry_shifts.csv"))["условие"])
    print(f"{'':28s}{'все':>16s}{'дефектные':>16s}{'смысловые':>16s}")
    for c in ("beta", "q*"):
        line = f"  крупная × Δ{c:14s}"
        for m in (np.ones(len(S), bool), mech, ~mech):
            g = S[m]
            line += f"{pearsonr(g[c], g['крупный'])[0]:+8.2f}/{spearmanr(g[c], g['крупный'])[0]:+7.2f}"
        print(line)
    print("\nСдвиги по условиям:\n")
    print(S[["beta", "q*", "крупный", "мелкий", "средний"]].round(3)
          .sort_values("крупный").to_string())


if __name__ == "__main__":
    main()
