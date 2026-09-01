"""Does the sign of the coarse band say what kind of document it found?

The hypothesis under test: a coarse band well above normal marks a text that is
merely unusual and fine, while a coarse band below normal marks real damage.
Twenty texts read by hand pointed that way -- the three that rose were one clean
dense encyclopedia lead and two with cosmetic defects, the two that fell were a
formula wreck and a detokenised post -- but three against two is not a test.

Deviation is taken against the corpus mean, not the genre mean. Normalising by
genre looked like the careful choice and is the wrong one here: there are no
genre artefacts, only data artefacts that happen to land on a whole subcorpus.
reddit_eli5 carries 8.94 spaces before punctuation per hundred words against
roughly zero everywhere else, because all of it was parsed off reddit the same
way, and its coarse band sits 15 points below the rest. Subtracting the genre
mean would subtract exactly the defect we are looking for. BASELINE=genre runs
the genre-relative version for comparison.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from defect_judge import OPS, build, check_quotes, parse
from openrouter import complete
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.6-flash")
PER_GROUP = int(os.environ.get("PER_GROUP", 40))
# which band's sign is being tested; the coarse one is the hypothesis, the
# other two are the check that the effect is not simply "any band is unusual"
BAND = os.environ.get("BAND", "крупный")


def sample():
    T = pd.read_csv(os.path.join(BASE, "results", "human_band_dev.csv"))
    T["id"] = T["id"].astype(str)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    T = T.merge(pool[["id", "text"]], on="id")
    pre = "pct_" if os.environ.get("BASELINE") == "genre" else "all_"
    T["центр"] = T[[f"{pre}{b}" for b in B]].abs().max(axis=1)
    parts = [T.nlargest(PER_GROUP, f"{pre}{BAND}").assign(группа=f"{BAND} вверх"),
             T.nsmallest(PER_GROUP, f"{pre}{BAND}").assign(группа=f"{BAND} вниз"),
             T.nsmallest(PER_GROUP, "центр").assign(группа="контроль")]
    return pd.concat(parts).drop_duplicates("id")


def main():
    S = sample()
    print(f"{len(S)} текстов: " + ", ".join(
        f"{k} {v}" for k, v in S["группа"].value_counts().items()), flush=True)
    done, t0 = [0], time.time()

    def one(r):
        out = complete(build(r.text), model=MODEL, provider="apiyi",
                       max_tokens=1400, temperature=0)
        rec = parse(out)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  {done[0]}/{len(S)}  {time.time() - t0:.0f}s", flush=True)
        return {"id": r.id, "группа": r.группа, "sub_source": r.sub_source,
                **{f"откл_{b}": getattr(r, f"all_{b}") for b in B},
                **rec, "плохие цитаты": "; ".join(check_quotes(rec, r.text))}

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, r) for r in S.itertuples()]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as exc:
                print("  ошибка:", str(exc)[:110], flush=True)
    D = pd.DataFrame(rows)
    suf = ("_genre" if os.environ.get("BASELINE") == "genre" else "")
    suf += "" if BAND == "крупный" else f"_{BAND}"
    D.to_csv(os.path.join(BASE, "results", f"coarse_sign{suf}.csv"),
             index=False)

    from scipy import stats as st
    pd.set_option("display.width", 220)
    keys = list(OPS) + ["damage", "nature", "human"]
    G = D.groupby("группа")[keys].mean()
    G["n"] = D.groupby("группа").size()
    G["выбросить %"] = D.groupby("группа")["verdict"].apply(
        lambda s: s.str.startswith("drop").mean() * 100)
    G["урон>0 %"] = D.groupby("группа")["damage"].apply(lambda s: (s > 0).mean() * 100)
    print("\n" + G.round(2).to_string())
    ctl = D[D["группа"] == "контроль"]["damage"]
    print("\nпротив контроля:")
    for g in (f"{BAND} вверх", f"{BAND} вниз"):
        a = D[D["группа"] == g]["damage"]
        print(f"  {g:14s} урон {a.mean():.2f} против {ctl.mean():.2f}, "
              f"p={st.mannwhitneyu(a, ctl).pvalue:.3f}")
    a = D[D["группа"] == f"{BAND} вверх"]["damage"]
    b = D[D["группа"] == f"{BAND} вниз"]["damage"]
    print(f"  вверх против вниз: {a.mean():.2f} против {b.mean():.2f}, "
          f"p={st.mannwhitneyu(a, b).pvalue:.3f}")
    print(f"\nнеподтверждённых цитат: {(D['плохие цитаты'] != '').sum()}/{len(D)}")


if __name__ == "__main__":
    main()
