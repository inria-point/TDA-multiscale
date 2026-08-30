"""The catalogue: every perturbation, what it does, and its three-band effect.

Sorted so that the strongest movers of each band come first within their sign
group, because the table is meant to be read as "I need the middle band down
without touching the coarse one -- what have I got".
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
from descriptions import DESC
from three_bands import BANDS as BAND_DEF
from three_bands import SUF, VARIANT

BASE = os.path.join(HERE, "..")
BANDS = list(BAND_DEF)
RANGES = "\n".join(f"* **{n}** — `{m}`, среднее по q >= {q}"
                   for n, (m, q) in BAND_DEF.items())


def main():
    P = pd.read_csv(os.path.join(BASE, "results", f"bands_perts{SUF}.csv"),
                    index_col=0)
    P["описание"] = [DESC.get(i, "") for i in P.index]
    P["полос"] = [sum(c != "0" for c in o) for o in P["octant"]]
    P["сила"] = P[BANDS].abs().max(axis=1)
    out = P[["описание", "octant"] + BANDS + ["полос", "сила"]]
    out = out.sort_values(["полос", "octant", "сила"],
                          ascending=[True, True, False])
    out.round(1).to_csv(os.path.join(BASE, "results", f"catalogue{SUF}.csv"))

    lines = ["| пертурбация | что делает | крупный | мелкий | средний |",
             "|---|---|---|---|---|"]
    for k in [1, 2, 3]:
        sub = out[out["полос"] == k]
        if sub.empty:
            continue
        head = {1: "однополосные", 2: "двухполосные",
                3: "трёхполосные"}[k]
        lines.append(f"| **{head}** | | | | |")
        for name, r in sub.iterrows():
            def f(v):
                return f"**{v:+.1f}**" if abs(v) >= 3 else f"{v:+.1f}"
            lines.append(f"| `{name}` | {r['описание']} | {f(r['крупный'])} "
                         f"| {f(r['мелкий'])} | {f(r['средний'])} |")
    md = "\n".join(lines)
    path = os.path.join(BASE, "results", f"catalogue{SUF}.md")
    with open(path, "w") as fh:
        fh.write(f"# Каталог пертурбаций (нарезка `{VARIANT}`)\n\n"
                 "Сдвиг d в процентах против неизменённого человеческого "
                 "текста, парно по одним и тем же текстам. Полоса — среднее "
                 "относительного сдвига по своему диапазону q:\n\n"
                 + RANGES + "\n\n"
                 "`q_small` отбрасывает q самых коротких рёбер MST и "
                 "оставляет длинные, `q_large` наоборот, `q0.5_range` — окно "
                 "шириной 0.5.\n\n"
                 "Жирным выделено превышающее 3%: ниже этого порога сдвиг не "
                 "отличается от шума прогона.\n\n" + md + "\n")
    print(f"saved {path}  ({len(out)} пертурбаций)")
    pd.set_option("display.width", 210)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_colwidth", 52)
    print(out.round(1).to_string())


if __name__ == "__main__":
    main()
