"""Ask the judge what a cluster has in common, and check it is not confabulating.

Per-text scores found nothing, but they ask a narrow question -- ten fixed
properties, one text at a time. A shared property might be something none of
the ten names, and might only be visible across several texts at once. So the
judge is shown five texts from one profile and asked what principle put them
together.

A model asked "what do these have in common" will always answer. The control is
therefore built in: a third of the clusters are random texts drawn across
profiles, and the judge is told plainly that some clusters are meaningless and
that saying so is a correct answer. If the confidence it reports is no higher
on real profiles than on random ones, its explanations are worth nothing, and
that comparison is the actual result -- the prose is only readable evidence.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from openrouter import complete

BASE = os.path.join(HERE, "..")
MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.6-flash")
K = 5                    # texts per cluster
REPEATS = 5              # subsamples per profile
N_CONTROL = 15
# no truncation. The first run capped texts at 320 words, and the only
# property the judge ever found was "truncated mid-sentence at the end" -- it
# was describing the cap, in random clusters as readily as in real ones.
CAP = None

PROMPT = """A filtering system for problematic training data has grouped the
{k} texts below into one cluster. Your task is to say what principle it used.

Be warned: this system is being evaluated, and some of its clusters are
spurious -- the texts in them have nothing in common beyond being text. Saying
so is a correct and valuable answer, not a failure. Do not manufacture a shared
property to be helpful.

Ignore what the texts are about. Subject matter is not a defect, and five texts
on unrelated topics are not thereby a cluster. Look instead for shared
properties of how they are written or how they came to exist: formatting
damage, truncation, scraping artefacts, repetition, boilerplate, register,
sentence construction, vocabulary, structural conventions.

Answer in exactly four lines, each beginning with its tag:

PRINCIPLE: one sentence naming the shared property, or "none" if there is none
EVIDENCE: one sentence pointing at what in the texts shows it
DEFECT: yes if the shared property is a defect that would justify filtering
these texts out of a training corpus, no if it is a neutral characteristic
CONFIDENCE: a whole number 1-10 for how sure you are that a real shared
property exists at all, where 1 means these texts are simply unrelated and 10
means the shared property is unmistakable

---
{texts}"""


def parse(out):
    r = {}
    for tag in ("PRINCIPLE", "EVIDENCE", "DEFECT", "CONFIDENCE"):
        m = re.search(rf"{tag}\s*:\s*(.+)", out)
        r[tag.lower()] = m.group(1).strip() if m else ""
    try:
        r["confidence"] = int(re.search(r"\d+", r["confidence"]).group())
    except (AttributeError, ValueError):
        r["confidence"] = np.nan
    r["defect"] = r["defect"].lower().startswith("yes")
    return r


def _damaged(jobs, per=3):
    """Clusters drawn from perturbations whose defect is not in doubt."""
    from judge_debug import FILES as TEXT_FILES

    known = {}
    for fn in TEXT_FILES:
        fp = os.path.join(BASE, "results", fn)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for name, by_id in json.load(f).items():
                known.setdefault(name, {}).update(by_id)
    body = {}
    rng = np.random.default_rng(7)
    for name in ("shuffle_words", "loop_tail_early", "collapse_vocab_60",
                 "ngram_2", "add_typos"):
        if name not in known:
            continue
        ids = sorted(known[name])
        for i in range(per):
            pick = rng.choice(ids, size=K, replace=False)
            tag = [f"{name}::{t}" for t in pick]
            body |= {k: known[name][k.split("::", 1)[1]] for k in tag}
            jobs.append((f"ПОРЧА:{name}", i, tag))
    return body


def main():
    S = pd.read_csv(os.path.join(BASE, "results", "outlier_sample.csv"))
    S["id"] = S["id"].astype(str)
    rng = np.random.default_rng(0)
    jobs = []
    for p, g in S.groupby("профиль"):
        for i in range(REPEATS):
            jobs.append((p, i, g.sample(min(K, len(g)),
                                        random_state=100 + i)["id"].tolist()))
    for i in range(N_CONTROL):
        jobs.append(("СЛУЧАЙНЫЙ", i,
                     S.sample(K, random_state=500 + i)["id"].tolist()))
    # the most sensitive form of the question: ignore which bands are off and
    # draw from everything the filter flags, so the probe has the whole tail to
    # find a shared property in rather than one profile at a time
    ex = S[S["профиль"] != "000"]
    for i in range(N_CONTROL):
        jobs.append(("ВСЕ ВНЕ НОРМЫ", i,
                     ex.sample(K, random_state=900 + i)["id"].tolist()))
    strong = ex.nlargest(max(K * 3, 15), "сила")
    for i in range(N_CONTROL):
        jobs.append(("САМЫЕ КРАЙНИЕ", i,
                     strong.sample(K, random_state=1300 + i)["id"].tolist()))
    body = dict(zip(S["id"], S["text"]))
    # positive control: clusters of texts we damaged ourselves. If the probe
    # cannot name a shared property in five shuffled or looped texts, a null on
    # the profiles says nothing about the profiles.
    body |= _damaged(jobs)
    print(f"{len(jobs)} кластеров: {S['профиль'].nunique()} профилей x "
          f"{REPEATS} + {N_CONTROL} случайных", flush=True)
    done, t0 = [0], time.time()

    def one(job):
        p, i, ids = job
        texts = "\n\n".join(
            f"=== TEXT {n + 1} ===\n"
            + (" ".join(body[t].split()[:CAP]) if CAP else body[t].strip())
            for n, t in enumerate(ids))
        out = complete(PROMPT.format(k=len(ids), texts=texts), model=MODEL,
                       provider="apiyi", max_tokens=700, temperature=0)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]}/{len(jobs)}  {time.time() - t0:.0f}s",
                  flush=True)
        return {"профиль": p, "подвыборка": i, "ids": ";".join(ids),
                **parse(out)}

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as exc:
                print("  ошибка:", str(exc)[:110], flush=True)
    D = pd.DataFrame(rows).sort_values(["профиль", "подвыборка"])
    path = os.path.join(BASE, "results", "cluster_probe.csv")
    D.to_csv(path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 150)
    D["сказал none"] = D["principle"].str.lower().str.startswith("none")
    R = D.groupby("профиль").agg(
        n=("confidence", "size"), уверенность=("confidence", "mean"),
        **{"назвал дефектом": ("defect", "mean"),
           "сказал «нет общего»": ("сказал none", "mean")})
    ctrl = R.loc["СЛУЧАЙНЫЙ"] if "СЛУЧАЙНЫЙ" in R.index else None
    print("\n=== уверенность судьи, что общее свойство вообще есть\n")
    print(R.round(2).sort_values("уверенность", ascending=False).to_string())
    if ctrl is not None:
        real = D[D["профиль"] != "СЛУЧАЙНЫЙ"]["confidence"]
        rnd = D[D["профиль"] == "СЛУЧАЙНЫЙ"]["confidence"]
        from scipy import stats as st
        print(f"\nнастоящие профили {real.mean():.2f} против случайных "
              f"{rnd.mean():.2f}; p = "
              f"{st.mannwhitneyu(real.dropna(), rnd.dropna()).pvalue:.3f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
