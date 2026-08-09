"""Pull the texts at the low / middle / high end of d_hat for one (mode, q).

Motivation: d_hat spread is small when long MST edges are trimmed away
(q_large) and large when they are kept (q_small at high q). So the variance
seems to live in the long edges. This dumps the extreme texts with their
lengths and a few descriptive stats to eyeball what drives it.
"""
import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data import load

BASE = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(BASE, "cache", "embeds")


def token_len(source, genre, idx):
    """Token count of the full text, read from the embedding cache."""
    path = os.path.join(CACHE, f"{source}_{genre}_{idx}.npz")
    if not os.path.exists(path):
        return np.nan
    return int(np.load(path)["embeds"].shape[0])


def text_stats(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    n = len(words)
    if n == 0:
        return {}
    counts = Counter(words)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "n_words": n,
        "ttr": len(counts) / n,  # type-token ratio
        "hapax": sum(1 for w, c in counts.items() if c == 1) / n,
        "top1_freq": counts.most_common(1)[0][1] / n,
        "mean_wlen": float(np.mean([len(w) for w in words])),
        "n_sents": len(sents),
        "mean_slen": n / max(1, len(sents)),
        "digit_frac": sum(c.isdigit() for c in text) / max(1, len(text)),
        "upper_frac": sum(c.isupper() for c in text) / max(1, len(text)),
        "punct_frac": sum(c in ",;:—-()\"'" for c in text) / max(1, len(text)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--mode", default="q_small")
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=10, help="texts per bucket")
    ap.add_argument("--source", default="human")
    ap.add_argument("--genre", default=None, help="restrict to one genre")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[(df["mode"] == args.mode) & (np.isclose(df["q"], args.q)) & (df["d_hat"] > 0)]
    if args.genre:
        df = df[df["genre"] == args.genre]
    df = df.sort_values("d_hat").reset_index(drop=True)

    n = len(df)
    mid = n // 2
    buckets = {
        "low": df.iloc[: args.k],
        "mid": df.iloc[mid - args.k // 2 : mid + (args.k + 1) // 2],
        "high": df.iloc[n - args.k :],
    }

    texts_by_genre = {g: dict(load(g, args.source)) for g in df["genre"].unique()}

    rows = []
    for bucket, sub in buckets.items():
        for _, r in sub.iterrows():
            text = texts_by_genre[r["genre"]][int(r["text_idx"])]
            rows.append(
                {
                    "bucket": bucket,
                    "genre": r["genre"],
                    "idx": int(r["text_idx"]),
                    "d_hat": round(r["d_hat"], 2),
                    "r2": round(r["r2"], 3),
                    "n_tokens": token_len(args.source, r["genre"], int(r["text_idx"])),
                    **text_stats(text),
                }
            )
    out = pd.DataFrame(rows)

    tag = f"{args.source}_{args.mode}_q{args.q}"
    csv_path = os.path.join(BASE, "results", f"extremes_{tag}.csv")
    out.to_csv(csv_path, index=False)

    cols = ["bucket", "genre", "idx", "d_hat", "r2", "n_tokens", "n_words",
            "ttr", "hapax", "mean_wlen", "mean_slen", "digit_frac", "upper_frac"]
    pd.set_option("display.width", 200)
    print(f"=== {args.mode} @ q={args.q}, source={args.source}, n_texts={n}")
    print(out[cols].round(3).to_string(index=False))
    print("\n=== bucket means")
    print(out.groupby("bucket", sort=False).mean(numeric_only=True).round(3).to_string())
    print("\n=== correlation of d_hat with each stat (all texts in bucket dump)")
    corr = out.drop(columns=["bucket", "genre", "idx"]).corr(numeric_only=True)["d_hat"]
    print(corr.drop("d_hat").round(3).sort_values().to_string())

    # dump the actual texts for reading
    txt_path = os.path.join(BASE, "results", f"extremes_{tag}.txt")
    with open(txt_path, "w") as f:
        for bucket, sub in buckets.items():
            for _, r in sub.iterrows():
                text = texts_by_genre[r["genre"]][int(r["text_idx"])]
                f.write(f"\n{'='*90}\n[{bucket}] {r['genre']} #{int(r['text_idx'])} "
                        f"d_hat={r['d_hat']:.2f} "
                        f"n_tokens={token_len(args.source, r['genre'], int(r['text_idx']))}\n"
                        f"{'='*90}\n{text}\n")
    print(f"\nsaved {csv_path}\nsaved {txt_path}")


if __name__ == "__main__":
    main()
