"""Rewrite the small model's answer with the same shape and true content.

The whole experiment rests on this stage changing one thing. If the rewrite is
shorter, better organised or written in a different register, then any band
movement is explained by that and says nothing about invention. So the
instruction is mostly about what to preserve, and the result is checked
afterwards against length, vocabulary spread and entropy before anything is
concluded.
"""
PROMPT = """Below is a long answer written by a small language model. It reads
reasonably but much of its specific content is invented, wrong, or incoherent.

Rewrite it so that every factual claim is true and every piece of reasoning is
sound, while changing as little else as possible.

Preserve exactly:
- the length, to within five per cent of the original word count
- the paragraph structure: same number of paragraphs, same order of topics
- the formatting: if the original uses headings or numbered lists, use the same
  ones; if it is continuous prose, keep continuous prose
- the register and sentence rhythm: same level of formality, similar sentence
  lengths, the same rhetorical moves and connectives
- the stance: if the original hedges, hedge; if it asserts, assert

Change only:
- claims that are false, and terms used with the wrong meaning
- mechanisms, figures, dates and attributions that are wrong
- sentences that contradict each other or do not follow

Do not add caveats, corrections, apologies or notes about what was changed. Do
not improve the organisation. Do not make it more concise. Output only the
rewritten text.

{reference}
---
ORIGINAL:
{text}
"""


def build(text, question=None, answer=None):
    ref = ""
    if question and answer:
        # the reference is supplied rather than asked for, so the repair does
        # not depend on the strong model happening to know an obscure fact
        ref = (f"For reference, the question was: {question}\n"
               f"The established answer is: {answer}\n"
               f"Keep the text consistent with that.\n")
    return PROMPT.format(reference=ref, text=text)
