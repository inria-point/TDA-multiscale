"""Unified loader for the FLAT-style completion dataset.

human.jsonl        -> {"title", "abstract", "word_count"}
<model>.jsonl      -> {"_idx", "model", "title", "text_ai", "prefix", ...}
                      text_ai = human prefix + model continuation
"""
import json
import os

GENRES = ["xsum", "academic_abstracts", "amazon_reviews", "writingprompts"]
MODELS = ["gpt-5-nano", "gemini-2.5-flash", "deepseek-v4-flash", "qwen3.6-flash"]
SOURCES = ["human"] + MODELS
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data_completion")
)


def load(genre, source, strip_prefix=False):
    """Return list of (text_id, text). text_id is stable across sources."""
    path = os.path.join(DATA_DIR, genre, f"{source}.jsonl")
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if source == "human":
                out.append((i, d["abstract"]))
            else:
                text = d["text_ai"]
                if strip_prefix and text.startswith(d["prefix"]):
                    text = text[len(d["prefix"]) :].lstrip()
                out.append((d["_idx"], text))
    return out
