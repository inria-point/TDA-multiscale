"""Long-form answers from a small local model, which is where the nonsense is.

A 1.5B model asked for a detailed account of an obscure fact will produce
fluent, well-formed, confidently wrong prose. That is exactly the material the
experiment needs: the defect is factual and nothing else, so if the bands move
they are responding to something no lexical statistic sees.

Runs on the local machine rather than through the gateway because nothing
smaller than a 30B mixture is offered there, and the point is the small model.
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SMALL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
_state = {}


def _load():
    if "m" not in _state:
        tok = AutoTokenizer.from_pretrained(MODEL)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        # .to(dev) rather than device_map, which would pull in accelerate for
        # a model that fits on one device anyway
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
        m.to(dev).eval()
        _state.update(m=m, tok=tok)
    return _state["m"], _state["tok"]


# left to itself the model answers with headings and numbered lists, and we
# already know that rigidity of form moves all three bands -- that would
# confound the comparison with the corrected version, which a strong model
# writes as prose
SYSTEM = ("You are writing for a print magazine that does not use any "
          "formatting. Write continuous flowing paragraphs of prose. Never "
          "use headings, bullet points, numbered lists, bold or italic "
          "markers, or section titles.")


@torch.no_grad()
def generate(prompt, max_new_tokens=700, temperature=0.8, seed=0,
             system=SYSTEM):
    m, tok = _load()
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(m.device)
    torch.manual_seed(seed)
    out = m.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                     temperature=temperature, top_p=0.95,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(generate(sys.argv[1] if len(sys.argv) > 1 else
                   "Write 300 words on the Treaty of Kars."))
