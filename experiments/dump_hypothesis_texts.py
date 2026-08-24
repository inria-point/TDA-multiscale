"""Three texts at each pole of every hypothesis, for reading before designing
the perturbations. Each feature is shown at the q where it separates best."""
import os, sys, textwrap
import numpy as np, pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
# feature -> (mode, q, где эффект максимален)
HYP = [
    ("sent_len_burst",   "q_small", 0.9, "H1 неровность длин предложений"),
    ("mean_sent_len",    "q_large", 0.9, "H2 длина предложений"),
    ("long_word_rate",   "q_small", 0.9, "H3 длина слов"),
    ("punct_rate",       "q_small", 0.6, "H4 плотность пунктуации"),
    ("content_ratio",    "q_small", 0.9, "H5 доля знаменательных слов"),
    ("bullet_rate",      "q_small", 0.4, "H6 списочное форматирование"),
    ("digit_rate",       "q_small", 0.6, "H7 цифры и числа"),
    ("upper_rate",       "q_small", 0.6, "H8 заглавные и аббревиатуры"),
]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    chars = int(sys.argv[2]) if len(sys.argv) > 2 else 420
    feat = pd.read_csv(os.path.join(BASE, "results", "coling_features.csv"))
    dd = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    dd = dd[dd["d_hat"] > 0]

    for f, mode, q, title in HYP:
        d = dd[(dd["mode"] == mode) & (np.isclose(dd["q"], q))]
        m = feat.merge(d[["id", "d_hat"]], on="id").merge(
            pool[["id", "text"]], on="id")
        for c in [f, "d_hat"]:
            m[c + "_z"] = m.groupby("sub_source")[c].transform(
                lambda x: (x - x.mean()) / (x.std() or 1))
        print("\n" + "=" * 100)
        print(f"{title}   [{f}]   {mode}, q={q}")
        print("=" * 100)
        for label, sign in [("(+) высокая d, высокий признак", +1),
                            ("(−) низкая d, низкий признак", -1)]:
            sel = m.loc[(sign * (m[f + "_z"] + m["d_hat_z"])).nlargest(k).index]
            print(f"\n{label}")
            for _, r in sel.iterrows():
                print(f"\n  [{r['model']} / {r['sub_source']}]  "
                      f"d={r['d_hat']:.1f}  {f}={r[f]:.3f}")
                body = " ".join(str(r["text"]).split())[:chars]
                print(textwrap.fill(body, 94, initial_indent="   ",
                                    subsequent_indent="   "))


if __name__ == "__main__":
    main()
