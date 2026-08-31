"""Two scales for a small model's long answer: invention and fluency.

Both are needed and they are meant to come apart. A 1.5B model writing about an
obscure topic produces prose that reads well and asserts things that are not
so; if the two scales moved together the experiment would be measuring
brokenness again, which we have already measured to death.

Invention is about the supporting detail, not the verdict. A text that reaches
the right conclusion through invented mechanisms, dates and authorities scores
high on invention; a text that reaches the wrong conclusion from sound and
correctly stated premises scores low.
"""
import re

SCALE = """Use a 1-10 scale. Anchors:
  1-2  none of this present at all
  3-4  present but slight
  5    the level at which a reader would first notice it
  7    clearly present, the level of an ordinary careless draft
  9-10 pervasive, the dominant feature of the text
"""

PROPS = {
    "invention": dict(
        ru="выдуманность деталей",
        prompt="""How much of the supporting detail in this text is invented,
false or incoherent?

Judge the DETAIL, not the final verdict. Look at named facts, figures, dates,
mechanisms, technical terms, cited authorities and causal claims. Count a claim
against the text when it is factually wrong, when it is a real term used with
the wrong meaning, when a mechanism is described in a way that does not work,
or when a sentence asserts something that contradicts another sentence.

  1-2  every specific claim is sound
  3-4  one or two shaky details in an otherwise sound account
  5    a reader with domain knowledge would stop at something wrong
  7    several confident claims are false or use terms wrongly
  9-10 the specifics are largely fabricated"""),
    "fluency": dict(
        ru="гладкость языка",
        prompt="""How fluent and well-formed is the language, ignoring whether
anything said is true?

Judge sentence construction, grammar, connectives, paragraph flow and register.
A text can be entirely false and perfectly fluent; score only the writing.

  1-2  broken: agreement errors, sentences that do not parse
  3-4  awkward, repetitive, visibly machine-like
  5    readable but flat and padded
  7    competent prose, the level of an ordinary human draft
  9-10 well-controlled, varied, would pass an editor"""),
}

FRAME = """You are scoring a text written by a small language model answering a
factual question at length. Score it as it is: do not rewrite it, do not
comment on its usefulness, and do not withhold a score.

{scale}
For each property give exactly two lines:
REASON: <one sentence, citing something specific from the text>
SCORE: <integer 1-10>
"""


def build(text, cap_words=700):
    body = " ".join(text.split()[:cap_words])
    parts = [FRAME.format(scale=SCALE)]
    for i, (k, meta) in enumerate(PROPS.items(), 1):
        parts.append(f"\nPROPERTY {i} ({k}):\n{meta['prompt']}")
    parts.append("\n---\nTEXT:\n" + body)
    parts.append("\n---\nNow give, in this exact order and nothing else:\n" +
                 "\n".join(f"{k}\nREASON: ...\nSCORE: ..." for k in PROPS))
    return "\n".join(parts)


def parse(out):
    got = {}
    for k in PROPS:
        m = re.search(rf"{k}\s*\n\s*REASON:\s*(.+?)\n\s*SCORE:\s*(\d+)",
                      out, re.I | re.S)
        if m:
            got[k] = (int(m.group(2)), m.group(1).strip()[:300])
    if not got:  # some models drop the property headers and just answer twice
        pairs = re.findall(r"REASON:\s*(.+?)\n\s*SCORE:\s*(\d+)", out,
                           re.I | re.S)
        for k, (why, sc) in zip(PROPS, pairs):
            got[k] = (int(sc), why.strip()[:300])
    return got
