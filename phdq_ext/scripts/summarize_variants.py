"""Rank qPHD estimator variants by measurement quality."""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        BASE, "results", "variant_sweep_L256.csv"
    )
    d = pd.read_csv(path)
    pd.set_option("display.width", 240)

    for mode in ["q_small", "q_large"]:
        s = d[d["mode"] == mode]
        print(f"\n{'='*100}\n=== {mode}: relative measurement noise cv_within")
        print(s.pivot(index="variant", columns="q", values="cv_within")
              .round(3).sort_values(0.3).to_string())
        print(f"\n=== {mode}: mean d_hat (shows how far each variant moves the value)")
        print(s.pivot(index="variant", columns="q", values="mean_d")
              .round(2).to_string())
        print(f"\n=== {mode}: rel_resid (deviation from the power law)")
        print(s.pivot(index="variant", columns="q", values="rel_resid")
              .round(4).to_string())

    print(f"\n{'='*100}\n=== overall: cv_within averaged over q<=0.6, both modes")
    core = d[(d["q"] <= 0.6) & (d["mode"] != "q0.5_range")]
    summ = core.groupby("variant").agg(
        cv_within=("cv_within", "mean"),
        rel_resid=("rel_resid", "mean"),
        rel_dse=("rel_dse", "mean"),
    ).sort_values("cv_within")
    base = summ.loc["baseline", "cv_within"]
    summ["noise_vs_baseline"] = (summ["cv_within"] / base).round(2)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
