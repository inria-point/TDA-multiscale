"""LLM rewrites: one text property changed at a time.

Each prompt targets a single property and holds everything else fixed —
length above all, since d depends on the number of tokens. The rewrites are
paired: the same source text under every condition, so the comparison is
within-text.

Writes results/llm_edits.json as {perturbation: {"genre::text_id": text}},
which run_perturbations.py consumes via --texts-json. The key carries the
genre because text ids restart at zero in every genre file.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from data import GENRES, load
from openrouter import complete

BASE = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"

SYSTEM = (
    "You rewrite text for a controlled experiment. Follow the single requested "
    "change exactly and change nothing else. Keep the same genre, register and "
    "language (English).\n\n"
    "LENGTH IS A HARD CONSTRAINT. The rewrite must contain the stated number of "
    "words. Never compress the text: if the change would make it shorter, "
    "expand what remains — elaborate, restate, add detail — until the length is "
    "met. A rewrite that is short is a failed rewrite, however well it follows "
    "the other instruction.\n\n"
    "Output only the rewritten text: no preamble, no commentary, no markdown, "
    "no quotation marks around it."
)

PROMPTS = {
    "lexical_diversity_up": (
        "Increase the lexical diversity of the text: avoid repeating any content "
        "word, use a different word each time an idea recurs. Do not add new "
        "ideas and do not change the meaning."
    ),
    "lexical_diversity_down": (
        "Decrease the lexical diversity of the text: reuse the same small set of "
        "content words as often as possible instead of varying them. Do not "
        "remove ideas and do not change the meaning."
    ),
    "ideas_up": (
        "Increase the number of distinct ideas in the text: add further "
        "specifics, consequences and details, so that more separate claims are "
        "made in the same space. Stay on the same topic."
    ),
    "ideas_down": (
        "Decrease the number of distinct ideas in the text: keep only one or two "
        "claims and restate and elaborate them, so that fewer separate ideas "
        "fill the same space. Stay on the same topic."
    ),
    "topics_up": (
        "Increase the topical variety of the text: let it move between several "
        "different subjects rather than staying on one, while keeping the same "
        "genre and style."
    ),
    "topics_down": (
        "Decrease the topical variety of the text: make it stay narrowly on a "
        "single subject throughout, while keeping the same genre and style."
    ),
    "words_common": (
        "Replace rare and specialised words with the most common everyday words "
        "that carry the same meaning. Keep proper names. Do not change the "
        "ideas or the sentence structure."
    ),
    "words_rare": (
        "Replace common everyday words with rarer, more specialised or more "
        "literary synonyms. Keep proper names. Do not change the ideas or the "
        "sentence structure."
    ),
    "syntax_complex": (
        "Make the syntax more complex: use subordinate clauses, participial "
        "phrases and embedded structures, combining short sentences into long "
        "ones. Keep exactly the same content and vocabulary where possible."
    ),
    "syntax_simple": (
        "Make the syntax simpler: use short main clauses in subject-verb-object "
        "order, no subordination. Keep exactly the same content and vocabulary "
        "where possible."
    ),
}


def build_prompt(instruction, text):
    n = len(text.split())
    lo, hi = int(n * 0.95), int(n * 1.1)
    return (
        f"{instruction}\n\n"
        f"The text below is {n} words long. Your rewrite must be between {lo} "
        f"and {hi} words — count them. Do not stop early.\n\n"
        f"TEXT:\n{text}\n\nREWRITTEN TEXT ({lo}-{hi} words):"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-texts", type=int, default=50, help="per genre")
    ap.add_argument("--genres", nargs="+", default=GENRES)
    ap.add_argument("--source", default="human")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--perturbations", nargs="+", default=list(PROMPTS))
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(BASE, "results", "llm_edits.json"))
    args = ap.parse_args()

    jobs = []
    for genre in args.genres:
        picked = 0
        for text_id, text in load(genre, args.source):
            if picked >= args.n_texts:
                break
            if len(text.split()) < args.min_words:
                continue
            for pname in args.perturbations:
                jobs.append((pname, genre, text_id, text))
            picked += 1
    print(f"{len(jobs)} requests, model={args.model}, {args.workers} workers",
          flush=True)

    edits = {p: {} for p in args.perturbations}
    lock = threading.Lock()
    done = [0]
    failures = []

    def run(job):
        pname, genre, text_id, text = job
        out = complete(
            build_prompt(PROMPTS[pname], text),
            model=args.model,
            system=SYSTEM,
            temperature=args.temperature,
            max_tokens=max(1024, int(2.2 * len(text.split()))),
        )
        return pname, f"{genre}::{text_id}", out.strip()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                pname, key, out = fut.result()
                with lock:
                    edits[pname][key] = out
            except Exception as exc:
                failures.append((job[0], job[2], str(exc)[:120]))
            with lock:
                done[0] += 1
                if done[0] % 100 == 0:
                    print(f"  {done[0]}/{len(jobs)}  ошибок: {len(failures)}",
                          flush=True)
                    with open(args.out, "w") as f:
                        json.dump(edits, f, ensure_ascii=False)

    with open(args.out, "w") as f:
        json.dump(edits, f, ensure_ascii=False)
    print(f"\nsaved -> {args.out}")
    for p in args.perturbations:
        print(f"  {p:24s} {len(edits[p]):4d} текстов")
    if failures:
        print(f"\n{len(failures)} ошибок, первые пять:")
        for f_ in failures[:5]:
            print("  ", f_)


if __name__ == "__main__":
    main()
