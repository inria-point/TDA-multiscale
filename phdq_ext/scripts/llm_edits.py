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
    # v1 asked for "more" and "fewer" ideas in relative terms, which the model
    # satisfied by rewriting in general. v2 states the target as an absolute
    # property of every sentence, which is checkable sentence by sentence.
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
    "ideas_up_v2": (
        "Rewrite the text so that EVERY sentence introduces a new fact or a "
        "distinct thought. No sentence may restate, paraphrase or elaborate "
        "what an earlier sentence has already said. Each sentence must add "
        "information that cannot be inferred from the others. Stay on the same "
        "topic and in the same genre."
    ),
    "ideas_down_v2": (
        "Rewrite the text so that it expresses ONE single idea, restated again "
        "and again in different words. Take the central claim of the original "
        "and paraphrase it repeatedly — different vocabulary, different "
        "sentence shapes, different angles on the same thought — but never "
        "introduce a second distinct fact or claim. Every sentence must say "
        "essentially the same thing as every other sentence."
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


PROMPTS.update({
    # Strengthened variants. The instruction is the same in kind, stated with a
    # quantitative target instead of a comparative, because "more diverse" has
    # no ceiling a model can aim at while "no content word twice" does. This is
    # the single-call route to a larger magnitude; the other is to apply the
    # ordinary operator to its own output, which raises the magnitude without
    # touching the direction at all.
    "lexical_diversity_up_hard": (
        "Rewrite so that NO content word appears more than twice in the entire "
        "text. Every repeated notion must be named by a different word each "
        "time it recurs: use synonyms, hypernyms, paraphrases. Function words "
        "may repeat freely. Do not add or remove ideas."
    ),
    "ideas_up_hard": (
        "Rewrite so that EVERY CLAUSE carries a fact not stated anywhere else "
        "in the text. Not merely every sentence — every clause. Nothing may be "
        "restated, summarised or elaborated. If a sentence has three clauses "
        "it must carry three separate pieces of information."
    ),
    "topics_up_hard": (
        "Rewrite so that EVERY SENTENCE belongs to a different subject area "
        "from the one before it — different field, different domain of life, "
        "different register of knowledge. The text must not be assignable to "
        "any single topic at all, while still reading as continuous prose."
    ),
    # --- stage 2: styles named for what they are, not for a property to move.
    # Each is a register a human writer would recognise; the properties they
    # carry (sentence length, term repetition, abbreviations, numbers) are
    # measured afterwards rather than dictated, so the mapping from style to
    # dimension is read off rather than assumed.
    "style_news_telegraphic": (
        "Rewrite as a newswire dispatch in telegraphic style: short declarative "
        "sentences, facts first, no commentary, no transitions, minimal "
        "adjectives. Attribute claims briefly ('officials said'). Keep the same "
        "events and details."
    ),
    "style_simplified": (
        "Rewrite in deliberately simple language, as for a reader with a small "
        "vocabulary: common everyday words only, one idea per sentence, no "
        "subordinate clauses, no technical terms. Keep all the content."
    ),
    "style_news_dates_suffixes": (
        "Rewrite as a dense news report saturated with specifics: give dates, "
        "years, quantities and named institutions wherever the content allows, "
        "and prefer nominalised forms with suffixes (-ation, -ment, -ity, "
        "-ance) over plain verbs. Invent no facts that contradict the text."
    ),
    "style_dialogue_interjections": (
        "Rewrite as everyday spoken dialogue between two people, with "
        "interjections and fillers (oh, well, hmm, you know, right, I mean), "
        "contractions, short turns and interruptions. The same content must be "
        "conveyed through what they say."
    ),
    "style_literary": (
        "Rewrite as polished literary and essayistic prose: varied rhythm, "
        "concrete imagery, balanced periodic sentences, an authorial voice. "
        "Keep the same subject matter and length."
    ),
    "style_bulletin_abbreviations": (
        "Rewrite as a terse information bulletin dense with abbreviations and "
        "acronyms: introduce an acronym for every institution or repeated "
        "multi-word term and then use it throughout. Keep the same content."
    ),
    "style_scientific_terms": (
        "Rewrite in an academic scientific register: introduce precise "
        "technical terminology and named concepts, and then REUSE the same "
        "terms consistently rather than varying the wording, as a research "
        "paper does. Add abbreviations for repeated terms. Keep the same "
        "content and length."
    ),
    # topics_down executed weakly (0.70) on the small model; reworded in the
    # same absolute, sentence-checkable style that fixed the ideas pair
    "topics_up_v2": (
        "Rewrite the text so that it ranges over SEVERAL clearly different "
        "subjects. Consecutive sentences should concern different domains, so "
        "that the text as a whole cannot be assigned to one topic. Keep the "
        "same genre and register."
    ),
    "topics_down_v2": (
        "Rewrite the text so that it stays on ONE narrow subject throughout. "
        "Every sentence must concern that same single subject; never bring in "
        "another domain, example or aside from elsewhere. Keep the same genre "
        "and register."
    ),
})

# the remaining pairs executed well already (0.92-1.00); they are rerun on the
# stronger model unchanged, so that every contrast is within one model
for _p in ["lexical_diversity_up", "lexical_diversity_down",
           "words_rare", "words_common",
           "syntax_complex", "syntax_simple"]:
    PROMPTS[_p + "_s"] = PROMPTS[_p]


# Iterated variants, for raising the magnitude without changing the direction.
#
# Iteration only compounds when the instruction is *comparative*. The absolute
# forms used elsewhere -- "every sentence must introduce a new fact" -- are
# idempotent: once the text satisfies the condition a second pass has nothing
# to do and returns it unchanged. These are therefore phrased against the text
# as received, so each pass pushes beyond wherever the previous one stopped.
PROMPTS.update({
    "lexical_diversity_up_more": (
        "This text has already been rewritten once for lexical variety. Push it "
        "FURTHER than it currently is: find every content word that still "
        "occurs more than once and replace all but one occurrence with a "
        "different word. Judge against the text as given, not against some "
        "ideal. Do not add or remove ideas."
    ),
    "ideas_up_more": (
        "This text has already been rewritten once to carry more distinct "
        "ideas. Push it FURTHER than it currently is: find every sentence that "
        "still restates, elaborates or merely illustrates another and replace "
        "it with a sentence carrying a fact stated nowhere else. Judge against "
        "the text as given."
    ),
    "topics_up_more": (
        "This text has already been rewritten once for topical variety. Push it "
        "FURTHER than it currently is: find every pair of adjacent sentences "
        "that still share a subject area and move one of them to a different "
        "field. Judge against the text as given."
    ),
})


# ---------------------------------------------------------------------------
# Probes for the second component.
#
# PC2 is fine-scale dimension minus coarse-scale dimension. The reading being
# tested: it measures how many distinct *contexts* a given token appears in,
# holding the token inventory fixed. A contextual embedding of the same word
# spreads out when its surroundings vary and collapses onto itself when they
# do not, which is a fine-scale effect; the topic, which sets the coarse scale,
# is untouched by either.
#
# Everything below therefore holds vocabulary and subject matter constant on
# purpose and varies only how sentences are built. That is what makes the test
# separable from PC1: if the reading is right these should spread along PC2
# while staying in a narrow band on PC1.
#
# Prediction registered before running:
#   syntax_varied, syntax_complex  -> PC2 up
#   syntax_monotone, terms_long, context_locked -> PC2 down
#   all six -> |PC1| well below the vocabulary perturbations
PROMPTS.update({
    "syntax_monotone": (
        "Rewrite so that every sentence is built the same way: subject, then "
        "verb, then object, then one prepositional phrase, in that order. No "
        "subordination, no clauses joined by conjunctions, and every sentence "
        "about the same length. Keep the same content words and the same "
        "subject matter — change only how the sentences are built."
    ),
    "syntax_varied": (
        "Rewrite so that no two consecutive sentences are built the same way. "
        "Alternate deliberately: active and passive; opening with the subject, "
        "with a subordinate clause, with a participial phrase, with a "
        "prepositional phrase; short and long; statement and rhetorical "
        "question. Keep the same content words and the same subject matter — "
        "change only how the sentences are built."
    ),
    "context_locked": (
        "Rewrite so that every term occurring more than once always appears "
        "inside exactly the same surrounding phrase. If the first mention is "
        "'the proposed model, trained on the full corpus', then every later "
        "mention must repeat that wording verbatim rather than shortening it "
        "or rephrasing it. Keep the same content and the same length."
    ),
    "terms_long": (
        "Rewrite in the manner of a legal or regulatory document: replace "
        "every abbreviation, acronym and short noun with its full multi-word "
        "designation, and repeat that full designation in every subsequent "
        "mention without ever shortening it. Keep the same content and length."
    ),
})


# ---------------------------------------------------------------------------
# Structure-preserving ways to collapse the short edges.
#
# Every perturbation in the set that pulls the short MST edges together is a
# loop, and every loop destroys the text, so "low dimension implies a broken
# text" was left standing by construction rather than by evidence. Measuring
# real source code and real numeric tables at equal length shows it is false:
# code collapses three times as many edges as prose and a table collapses
# seven times as many (24.4% against 3.3%, the same as the harshest echo),
# while both are perfectly well-formed texts of their genre.
#
# The two do it differently, and both routes are worth having as rewrites of
# the same human text. A table repeats a field label inside an identical frame
# once per row -- identical context, distant position. Code repeats an
# identifier a few tokens away -- identical context and adjacent. Anaphora is
# the prose form of the first, a catechism the prose form of the second.
PROMPTS.update({
    "style_record_fields": (
        "Rewrite the content as a list of structured records. Every record "
        "must use the same field labels in the same order, spelled the same "
        "way every time, one field per line, in the form 'Label: value'. Do "
        "not vary the labels or abbreviate them after the first record. Keep "
        "the same content."
    ),
    "style_anaphora": (
        "Rewrite so that every sentence opens with the same three or four "
        "words, repeated verbatim, in the manner of rhetorical anaphora. The "
        "text must still read as deliberate, well-formed prose and carry the "
        "same content."
    ),
    "style_catechism": (
        "Rewrite as a sequence of question and answer pairs. Every question "
        "must be phrased with the same opening formula, and every answer must "
        "begin by restating the subject of its question word for word. Keep "
        "the same content."
    ),
    "style_code": (
        "Rewrite the content as a documented Python module: functions, "
        "variable assignments, and docstrings that carry the meaning of the "
        "original. Reuse the same identifiers throughout rather than "
        "inventing synonyms. The result must be syntactically valid Python."
    ),
    "style_equations": (
        "Rewrite the content as a mathematical derivation: numbered "
        "equations, symbols defined once and then reused, and short "
        "connecting sentences between steps. Reuse the same symbols "
        "throughout rather than introducing new notation for the same "
        "quantity."
    ),
})


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
    ap.add_argument("--corpus", choices=["flat", "coling"], default="flat")
    ap.add_argument("--input-json", default=None,
                    help="feed a previous run's output back in: applies the "
                         "same operator again, which raises the magnitude "
                         "without changing its direction")
    ap.add_argument("--input-key", default=None,
                    help="which perturbation of --input-json to use as input; "
                         "defaults to the one being generated")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--provider", default=None,
                    help="openrouter | apiyi | openai")
    ap.add_argument("--perturbations", nargs="+", default=list(PROMPTS))
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(BASE, "results", "llm_edits.json"))
    args = ap.parse_args()

    prev = {}
    if args.input_json:
        with open(args.input_json) as f:
            prev = json.load(f)

    jobs = []
    if args.corpus == "coling":
        from coling_data import human_texts

        for key, text in human_texts(args.n_texts, min_words=args.min_words):
            for pname in args.perturbations:
                src = text
                if prev:
                    base = args.input_key or pname.rsplit("_x", 1)[0]
                    src = prev.get(base, {}).get(key)
                    if src is None:
                        continue
                jobs.append((pname, None, key, src))
    else:
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
            provider=args.provider,
        )
        key = text_id if genre is None else f"{genre}::{text_id}"
        return pname, key, out.strip()

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
