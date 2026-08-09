"""Plots and tables for qPHD(q) curves.

Input: tidy csv from run_human_baseline.py (columns: mode, q, d_hat, r2,
genre, text_idx, ...). Produces:
  figures/<tag>_curves.png      - 3 panels (one per mode), lines = genres
  results/<tag>_table_<mode>.csv - mean +- std of d_hat at selected q
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
MODE_ORDER = ["q_small", "q_large", "q0.5_range"]
TABLE_Q = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--group", default="genre", help="column to group curves by")
    args = ap.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.csv))[0]
    df = pd.read_csv(args.csv)
    df = df[df["d_hat"] > 0]  # drop degenerate fits

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, mode in zip(axes, MODE_ORDER):
        sub = df[df["mode"] == mode]
        for key, grp in sub.groupby(args.group):
            stats = grp.groupby("q")["d_hat"].agg(["mean", "std"])
            ax.plot(stats.index, stats["mean"], marker="o", ms=3, label=key)
            ax.fill_between(
                stats.index,
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.15,
            )
        ax.set_title(mode)
        ax.set_xlabel("q")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("d_hat (qPHD)")
    axes[0].legend(fontsize=9)
    fig.suptitle(tag)
    fig.tight_layout()
    fig_path = os.path.join(BASE, "figures", f"{tag}_curves.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print("saved", fig_path)

    # tables: mean +- std at selected q, one file per mode
    for mode in MODE_ORDER:
        sub = df[(df["mode"] == mode) & (df["q"].isin(TABLE_Q))]
        stats = (
            sub.groupby([args.group, "q"])["d_hat"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats["cell"] = stats.apply(
            lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1
        )
        table = stats.pivot(index=args.group, columns="q", values="cell")
        out = os.path.join(BASE, "results", f"{tag}_table_{mode}.csv")
        table.to_csv(out)
        print("saved", out)
        print(f"--- {mode}")
        print(table.to_string())


if __name__ == "__main__":
    main()
