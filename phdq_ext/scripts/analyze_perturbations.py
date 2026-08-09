"""Which text properties move qPHD, and at which q.

Every perturbation is compared against `identity` on the *same* text, so the
effect is a within-text difference. Between-text variance was measured to be
the larger component, and pairing removes it entirely.

Effects are reported in two ways:
  delta      d_perturbed - d_identity, in units of d
  effect     the same divided by the sd of the paired differences (Cohen's dz)
             -- how reliably the change is seen, not just how large it is
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

GROUPS = {
    "лексика": ["lexical_diversity_up", "lexical_diversity_down",
                "words_rare", "words_common"],
    "идеи и темы": ["ideas_up", "ideas_down", "topics_up", "topics_down"],
    "синтаксис": ["syntax_complex", "syntax_simple",
                  "lengthen_sentences", "shorten_sentences"],
    "типографика": ["add_linebreaks", "break_at_commas", "lowercase",
                    "strip_punctuation"],
    "разрушение": ["shuffle_words", "shuffle_within_sentences",
                   "shuffle_sentences", "add_typos"],
}


def paired(df):
    """Join every perturbation to identity on (genre, text_id, mode, q)."""
    key = ["genre", "text_id", "mode", "q"]
    base = (df[df["perturbation"] == "identity"]
            .set_index(key)["d_hat"].rename("d0"))
    out = df[df["perturbation"] != "identity"].join(base, on=key)
    out = out.dropna(subset=["d0"])
    out["delta"] = out["d_hat"] - out["d0"]
    out["rel"] = out["delta"] / out["d0"]
    return out


def summarize(p):
    rows = []
    for (pert, mode, q), g in p.groupby(["perturbation", "mode", "q"]):
        d = g["delta"]
        sd = d.std()
        rows.append({
            "perturbation": pert, "mode": mode, "q": q, "n": len(d),
            "d0": g["d0"].mean(), "delta": d.mean(),
            "rel": g["rel"].mean(),
            "effect": d.mean() / sd if sd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_groups(s, mode, tag):
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(4.1 * len(GROUPS), 4.4),
                             sharey=True)
    sub = s[s["mode"] == mode]
    for ax, (title, perts) in zip(axes, GROUPS.items()):
        for pert in perts:
            g = sub[sub["perturbation"] == pert].sort_values("q")
            if g.empty:
                continue
            ax.plot(g["q"], g["rel"] * 100, marker="o", ms=3, label=pert)
        ax.axhline(0, c="k", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("q")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("изменение d относительно оригинала, %")
    fig.suptitle(f"{tag} — {mode}: что двигает размерность")
    fig.tight_layout()
    path = os.path.join(BASE, "figures", f"{tag}_effects_{mode}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?",
                    default=os.path.join(BASE, "results", "perturb_L201.csv.gz"))
    ap.add_argument("--tag", default="perturb")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["d_hat"] > 0]
    p = paired(df)
    s = summarize(p)
    s.to_csv(os.path.join(BASE, "results", f"{args.tag}_effects.csv"), index=False)

    for mode in ["q_small", "q_large", "q0.5_range"]:
        if (s["mode"] == mode).any():
            plot_groups(s, mode, args.tag)

    pd.set_option("display.width", 240)
    for mode in ["q_small", "q_large"]:
        sub = s[s["mode"] == mode]
        if sub.empty:
            continue
        print(f"\n{'='*100}\n=== {mode}: изменение d, % от оригинала")
        piv = sub.pivot(index="perturbation", columns="q", values="rel") * 100
        cols = [c for c in [0.0, 0.2, 0.4, 0.6, 0.9] if c in piv.columns]
        print(piv[cols].round(1).sort_values(cols[0]).to_string())

    print(f"\n{'='*100}\n=== где эффект максимален (по |effect|), для каждой модификации")
    best = (s.assign(a=s["effect"].abs())
            .sort_values("a", ascending=False)
            .groupby("perturbation").head(1)
            .sort_values("a", ascending=False))
    print(best[["perturbation", "mode", "q", "d0", "delta", "rel", "effect"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
