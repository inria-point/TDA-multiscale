"""Long-tail questions with a known answer, as seeds for confabulation.

MMLU is used not as a benchmark but as a supply of factual questions whose
correct answer is recorded, which is what the repair stage needs: the strong
model is told the answer rather than asked for it, so the corrected text does
not depend on the strong model happening to know an obscure fact either.

Subjects are chosen for being long-tail. A 1.5B model asked about the causes of
the French Revolution will be roughly right; asked about a specific virus, a
minor treaty or a point of canon law it will produce fluent invention, which is
the material this experiment is about.
"""
import glob
import os
import re

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
SUBJECTS = ["virology", "prehistory", "world_religions", "human_aging",
            "global_facts", "medical_genetics", "jurisprudence",
            "high_school_european_history", "astronomy", "anatomy",
            "college_medicine", "professional_medicine", "nutrition",
            "security_studies", "sociology"]
N = int(os.environ.get("N_Q", 70))
LETTERS = "ABCD"


def main():
    p = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--cais--mmlu/snapshots/*/all/"
        "test-*.parquet"))[0]
    d = pd.read_parquet(p)
    d = d[d["subject"].isin(SUBJECTS)].copy()
    # a question that only makes sense next to its options ("which of the
    # following", "all of the above") cannot be turned into an essay prompt
    # also drop items that only make sense inside a textbook -- "in this
    # chapter's Senior View" is not a question anyone can write an essay about
    bad = re.compile(r"following|above|NOT|except|this chapter|this text|"
                     r"the passage|the author|Senior View|the case study", re.I)
    d = d[~d["question"].str.contains(bad)]
    d = d[d["question"].str.split().str.len().between(6, 60)]
    per = max(1, N // len(SUBJECTS))
    take = pd.concat([g.sample(min(per, len(g)), random_state=0)
                      for _, g in d.groupby("subject")])
    take = take.sample(min(N, len(take)), random_state=0).reset_index(drop=True)
    take["answer_text"] = [c[a] for c, a in zip(take["choices"],
                                                take["answer"])]
    take["qid"] = [f"q{i:03d}" for i in range(len(take))]
    out = take[["qid", "subject", "question", "answer_text"]]
    path = os.path.join(BASE, "results", "halluc_questions.csv")
    out.to_csv(path, index=False)
    print(f"{len(out)} вопросов из {out.subject.nunique()} предметов")
    print(out.subject.value_counts().to_string())
    print("\nпримеры:")
    for _, r in out.head(5).iterrows():
        print(f"  [{r.subject}] {r.question}")
        print(f"     ответ: {r.answer_text}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
