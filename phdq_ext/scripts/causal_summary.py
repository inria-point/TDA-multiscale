"""Rank text properties by how strong the causal evidence for their effect is.

Not every measured effect supports the same claim. A perturbation that raises d
may be acting through the property it names, or merely through the fact that a
model rewrote the text — every LLM rewrite raises d on its own (finding 8).
Three things distinguish the two:

  reversal     the two arms move d in opposite directions. A generic rewrite
               effect is common to both arms and cannot produce a reversal, so
               a sign flip is evidence the named property is doing the work.
  replication  independent runs on two different rewriting models agree in
               sign. Guards against a quirk of one model's prose.
  mechanical   a direct, model-free manipulation of the same property points
               the same way. The strongest form of control available here.

Effect size is Cohen's dz of the paired differences: mean over sd of the
per-text change, so it says how reliably the change is seen, not just how big
the percentage is.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

# property -> arms per model, and the mechanical manipulation that bears on it
PROPERTIES = {
    "лексическое разнообразие": {
        "arms": {"gemini": ("lexical_diversity_up", "lexical_diversity_down"),
                 "sonnet": ("lexical_diversity_up_s", "lexical_diversity_down_s")},
        "mechanical": None,
    },
    "редкость слов": {
        "arms": {"gemini": ("words_rare", "words_common"),
                 "sonnet": ("words_rare_s", "words_common_s")},
        "mechanical": None,
    },
    "число идей": {
        "arms": {"gemini": ("ideas_up", "ideas_down"),
                 "sonnet": ("ideas_up_v2", "ideas_down_v2")},
        "mechanical": None,
    },
    "разнообразие тем": {
        "arms": {"gemini": ("topics_up", "topics_down"),
                 "sonnet": ("topics_up_v2", "topics_down_v2")},
        "mechanical": None,
    },
    "сложность синтаксиса": {
        "arms": {"gemini": ("syntax_complex", "syntax_simple"),
                 "sonnet": ("syntax_complex_s", "syntax_simple_s")},
        "mechanical": ("lengthen_sentences", "shorten_sentences"),
    },
    "порядок слов": {
        "arms": {},
        "mechanical": ("shuffle_words", None),
    },
    "порядок предложений": {
        "arms": {},
        "mechanical": ("shuffle_sentences", None),
    },
    "пунктуация": {
        "arms": {},
        "mechanical": ("strip_punctuation", None),
    },
    "опечатки": {
        "arms": {},
        "mechanical": ("add_typos", None),
    },
    "переносы строк": {
        "arms": {},
        "mechanical": ("add_linebreaks", None),
    },
    "регистр": {
        "arms": {},
        "mechanical": ("lowercase", None),
    },
}


def paired(df):
    key = ["genre", "text_id", "mode", "q"]
    base = (df[df["perturbation"] == "identity"]
            .set_index(key)["d_hat"].rename("d0"))
    out = df[df["perturbation"] != "identity"].join(base, on=key)
    out = out.dropna(subset=["d0"])
    out["delta"] = out["d_hat"] - out["d0"]
    out["rel"] = out["delta"] / out["d0"]
    return out


def arm_effect(p, name):
    """Peak paired effect of one perturbation: (dz, rel, mode, q) or None."""
    g = p[p["perturbation"] == name]
    if g.empty:
        return None
    rows = []
    for (mode, q), gg in g.groupby(["mode", "q"]):
        sd = gg["delta"].std()
        if sd > 0:
            rows.append((gg["delta"].mean() / sd, gg["rel"].mean(), mode, q))
    if not rows:
        return None
    return max(rows, key=lambda r: abs(r[0]))


def plot_ranking(res):
    """Properties ranked by |dz|, marked by what kind of evidence backs them."""
    rows = []
    for _, r in res.iterrows():
        dz = r.get("dz_макс")
        mech = r.get("механ_dz")
        if pd.notna(dz):
            kind = ("разворот на двух моделях"
                    if r.get("sonnet_разворот") and r.get("gemini_разворот")
                    else "разворот на одной модели")
            rows.append((r["свойство"], abs(dz), kind))
        elif pd.notna(mech):
            rows.append((r["свойство"], abs(mech), "механический контроль"))
    if not rows:
        return
    rows.sort(key=lambda x: x[1])
    names, vals, kinds = zip(*rows)
    colours = {"разворот на двух моделях": "#2b6cb0",
               "разворот на одной модели": "#63b3ed",
               "механический контроль": "#dd6b20"}

    fig, ax = plt.subplots(figsize=(9, 0.52 * len(rows) + 2))
    ax.barh(names, vals, color=[colours[k] for k in kinds])
    for thr, label in [(0.26, "58%"), (1.18, "89%"), (1.93, "99%")]:
        ax.axvline(thr, c="grey", ls=":", lw=1)
        ax.text(thr, -0.8, label, ha="center", fontsize=7, color="grey")
    ax.set_xlabel("|dz| — во сколько раз сдвиг больше своего разброса по текстам\n"
                  "(пунктир: доля текстов, сдвинувшихся в одну сторону)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles, colours.keys(), fontsize=8, loc="lower right")
    ax.set_title("Что причинно влияет на размерность")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(BASE, "figures", "causal_ranking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?",
                    default=os.path.join(BASE, "results",
                                         "perturb_L201_all.csv.gz"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["d_hat"] > 0]
    p = paired(df)

    rows = []
    for prop, spec in PROPERTIES.items():
        rec = {"свойство": prop}
        signs, peaks = [], []
        for model, (up, down) in spec["arms"].items():
            eu, ed = arm_effect(p, up), arm_effect(p, down)
            if eu is None or ed is None:
                continue
            rec[f"{model}_up"] = eu[1]
            rec[f"{model}_down"] = ed[1]
            # a reversal means the arms move d in opposite directions
            rec[f"{model}_разворот"] = bool(np.sign(eu[1]) != np.sign(ed[1]))
            signs.append(np.sign(eu[1] - ed[1]))
            peaks.append(max(abs(eu[0]), abs(ed[0])))
        if spec["mechanical"]:
            mu, md = spec["mechanical"]
            em = arm_effect(p, mu)
            if em is not None:
                rec["механ_rel"] = em[1]
                rec["механ_dz"] = em[0]
                rec["механ_где"] = f"{em[2]} q={em[3]}"
                if md:
                    emd = arm_effect(p, md)
                    if emd is not None:
                        rec["механ_разворот"] = bool(
                            np.sign(em[1]) != np.sign(emd[1]))
        if signs:
            rec["воспроизв"] = bool(len(set(signs)) == 1) if len(signs) > 1 else None
            rec["dz_макс"] = max(peaks)
        rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(BASE, "results", "causal_summary.csv"), index=False)
    plot_ranking(res)
    pd.set_option("display.width", 250)

    print("=== LLM-манипуляции: пиковый эффект каждого плеча, доля от d\n")
    cols = [c for c in ["свойство", "gemini_up", "gemini_down", "gemini_разворот",
                        "sonnet_up", "sonnet_down", "sonnet_разворот",
                        "воспроизв", "dz_макс"] if c in res]
    print(res[res["dz_макс"].notna()][cols].round(3).to_string(index=False))

    print("\n=== механические манипуляции: контроль без модели\n")
    m = res[res.get("механ_rel").notna()] if "механ_rel" in res else pd.DataFrame()
    if len(m):
        mc = [c for c in ["свойство", "механ_rel", "механ_dz", "механ_где",
                          "механ_разворот"] if c in m]
        print(m[mc].round(3).sort_values("механ_dz", key=abs, ascending=False)
              .to_string(index=False))


if __name__ == "__main__":
    main()
