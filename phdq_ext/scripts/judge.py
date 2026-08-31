"""LLM-judged text properties, one call per text scoring all of them.

The judge exists to measure what counting cannot. Yesterday's controlled
contrast established this need precisely: twenty foreign words substituted
mechanically move the coarse band -19.2, the same twenty worked into the text
by a model move it +2.1, and every lexical statistic -- type-token ratio, word
entropy, repeated-trigram share -- is identical to three decimals across the
pair. Whatever the coarse band responds to, no count sees it.

Design decisions that matter:

Anchored five-point scales rather than 0-100. Language models are poorly
calibrated on wide numeric ranges and drift between calls; explicit verbal
anchors for each level hold them steady. Counts are used where a count is the
natural answer.

Complexity is conditional. A text can look syntactically or semantically
elaborate purely because it is broken -- shuffled words produce long stretches
without a finite verb, an n-gram sampler produces sentences that sound
sophisticated and mean nothing. Both complexity judgements are therefore
returned as null when the text falls below a legibility floor, so that
gibberish never scores as complex.

Literacy is severity-weighted rather than counted per sentence. A missing
comma and a sentence that dissolves mid-clause are not one defect each.

Scores are absolute, not comparative. We need a value per text to correlate
against that same text's coordinates; a ranking would give only order.
"""

# Two fixed external anchors rather than one, on a ten-point scale.
#
# Five points gave too little room below the human level, which is where most
# of the set lives: a shuffled text and a looped text both landed at the floor
# with nothing between them. Ten points with 5 and 7 pinned to describable
# states -- "the least a text can be and still be a text" and "an ordinary
# human piece" -- calibrate the judge against something outside itself, which
# a bare 1-to-5 range does not.
#
# Every scale runs the same direction: more of the named quality is a higher
# number. Literacy is therefore stated positively rather than as a count of
# violations, so that 5 and 7 mean the same thing on every property.
SCALE = (
    "Answer with a whole number from 1 to 10, using these fixed reference "
    "points:\n"
    "  5 = the minimum a text can score and still read as a text at all; "
    "below 5 this aspect of it reads as noise\n"
    "  7 = an ordinary human-written text in which this quality is "
    "moderately present\n"
    "Scores of 8 to 10 mean the quality is unusually strong, 1 to 4 that the "
    "text is degraded on this dimension. A text written by a person for a "
    "real purpose sits at 5 or above on every scale; below 5 is reserved for "
    "texts that have been damaged. Use the whole range."
)

# template and domains are outside that scheme: an untouched human text has no
# template at all, so "minimally acceptable" has no meaning there, and a count
# of subjects is a count.
SCALE_PRESENCE = (
    "Answer with a whole number from 1 to 10, where 1 means continuous prose "
    "with no repeating frame whatever, 5 means a frame is clearly present but "
    "governs only part of the text, and 10 means the text is nothing but a "
    "filled-in template. An ordinary human-written article scores 1 or 2."
)

FRAME = (
    "The text below comes from a controlled experiment in which human writing "
    "was deliberately damaged in various ways. It may be scrambled, "
    "repetitive, machine-generated, stripped of punctuation or otherwise "
    "degraded, and it may be nonsense. That is expected and is exactly what "
    "you are being asked to measure. Always return scores; never decline.\n\n"
)

PROPS = {
    "integration": dict(
        ru="уместность слов", kind="scale",
        prompt=(
            "How well does each content word fit the place it stands in? "
            "Consider whether nouns, verbs and adjectives are the ones the "
            "sense of that sentence calls for, or whether they look "
            "substituted, arbitrary, or borrowed from an unrelated subject.\n"
            "  1-4 = content words are largely arbitrary; many could be "
            "swapped for any other word of the same part of speech\n"
            "  5 = enough words fit that the subject can still be made out\n"
            "  7 = a competent article in which nearly every word fits, with a "
            "few approximate or generic choices where a sharper word "
            "existed\n"
            "  8-10 = every content word is exactly the word that place "
            "requires"
        )),
    "coherence": dict(
        ru="связность изложения", kind="scale",
        prompt=(
            "Does the text read as one connected piece that advances? A text "
            "circling back to restate itself is not connected in this sense, "
            "however smooth its individual sentences are.\n"
            "  1-4 = unrelated statements; order could be shuffled without "
            "loss\n"
            "  5 = adjacent sentences connect, but the whole does not hold "
            "together\n"
            "  7 = a well-organised piece whose paragraphs follow one another, "
            "with the occasional abrupt transition or aside\n"
            "  8-10 = a single sustained argument or narration advancing from "
            "first sentence to last"
        )),
    "domains": dict(
        ru="тематический разброс", kind="scale",
        prompt=(
            "How much ground does the text cover? Rate the spread itself, "
            "not whether the material is well handled: a text that lurches "
            "between unrelated fields covers a great deal of ground however "
            "badly it joins them, and belongs high on this scale. Whether "
            "the words fit their places is judged separately.\n"
            "  1-4 = the text never moves at all, because it has been "
            "damaged into immobility: one phrase repeated, or word order "
            "destroyed so that no subject can be followed\n"
            "  5 = one topic treated flatly, stated and left as it stands\n"
            "  7 = one subject examined from several sides, with related "
            "matters brought in as they bear on it -- a paper that states a "
            "problem, a method and its consequences is here\n"
            "  8-10 = conspicuous for fast, unexpected changes of subject; "
            "at 10 consecutive sentences belong to unrelated fields"
        )),
    "novelty": dict(
        ru="скорость подачи нового", kind="scale",
        prompt=(
            "How fast does the text introduce material not already present?\n"
            "  1-4 = one point restated over and over; almost nothing is "
            "added after the opening\n"
            "  5 = advances, but slowly; most sentences elaborate what has "
            "been said\n"
            "  7 = a text examining one subject from several angles, each "
            "sentence contributing something without restating the last\n"
            "  8-10 = nearly every sentence brings something unconnected to "
            "the one before it: new events, facts or subjects in rapid "
            "succession"
        )),
    "literacy": dict(
        ru="грамотность", kind="scale",
        prompt=(
            "How sound is the text as written English? Weigh defects by "
            "severity rather than counting them: a missing comma is not the "
            "equal of a clause that dissolves halfway through. Count stray "
            "tokens, fragments and mechanical debris as severe.\n"
            "  1-4 = large parts are not well-formed English at all\n"
            "  5 = readable but repeatedly broken; passages must be reread or "
            "guessed at\n"
            "  7 = prose with occasional comma errors, an awkward phrase or a "
            "stray agreement slip, none of which impede reading\n"
            "  8-10 = clean, publishable prose"
        )),
    "template": dict(
        ru="жёсткость формы", kind="scale",
        prompt=(
            "How rigidly is the text organised by a repeating form? Every "
            "real text has some shape -- paragraphs, a lead and a body, a "
            "problem followed by a method and a result -- so ordinary prose "
            "is not at the bottom of this scale. What rises on it is the "
            "degree to which one frame is repeated and governs the writing: "
            "numbered or bulleted items; labelled fields such as 'Name:' or "
            "'Date:'; speaker names set off as in a script; a fixed "
            "question-and-answer pattern; or a phrasal frame repeated with "
            "substitutions, where sentence after sentence opens the same way "
            "and differs only in what is slotted in.\n"
            "  1-4 = no discernible organisation at all, as when the order "
            "of words or sentences has been destroyed\n"
            "  5 = continuous prose whose only shape is the conventional one "
            "of its genre\n"
            "  7 = a clear recurring structure visible within prose: "
            "sections, a report with a standing shape, paragraphs that open "
            "alike\n"
            "  8-10 = governed by an explicit repeating frame, and at 10 the "
            "text is nothing but a filled-in template"
        )),
    "punctuation": dict(
        ru="пунктуация", kind="scale",
        prompt=(
            "Judge the punctuation as punctuation, not by quantity: whether "
            "the marks present are the ones the sentences require, and "
            "whether required marks are missing. Judge against the sentences "
            "actually present -- a text whose sentences have collapsed cannot "
            "have good punctuation even if the marks were inherited intact. "
            "Genre conventions count as correct; a screenplay or a table of "
            "records is not penalised for punctuating unlike an essay.\n"
            "  1-4 = absent, arbitrary, or unattached to any sentence "
            "structure\n"
            "  5 = frequently wrong, missing or excessive, but the sentence "
            "boundaries can still be found\n"
            "  7 = sentence boundaries all marked, commas mostly where needed, "
            "the occasional missing or superfluous mark\n"
            "  8-10 = every mark earns its place and nothing needed is missing"
        )),
    "syntax_complexity": dict(
        ru="синтаксическая сложность", kind="scale_cond",
        prompt=(
            "How elaborate is the sentence construction? Consider "
            "subordination, embedding, participial and relative clauses, and "
            "variety of sentence shape.\n"
            "  1-4 = short main clauses in one repeated pattern, as in a "
            "children's reader or a screenplay of bare actions\n"
            "  5 = simple sentences with little variation, as in casual "
            "messaging or a forum reply\n"
            "  7 = a news report: mainly simple and compound sentences, with a "
            "relative or subordinate clause every few sentences\n"
            "  8-10 = frequent subordination, varied shape, and at the top "
            "long periodic sentences with several levels of embedding\n"
            "IMPORTANT: answer null instead of a number if the text is too "
            "damaged for the question to mean anything -- if word order is "
            "destroyed or sentences do not hold together, apparent complexity "
            "is an artefact and must not be scored."
        )),
    "semantic_complexity": dict(
        ru="смысловая сложность", kind="scale_cond",
        prompt=(
            "How demanding is the content? Consider abstraction, density of "
            "argument, how much the reader must hold in mind, and whether "
            "claims depend on one another.\n"
            "  1-4 = a plain sequence of concrete facts or events, as in a "
            "chronicle or a list of actions\n"
            "  5 = simple description with little abstraction, as in a "
            "product review or a personal anecdote\n"
            "  7 = an encyclopedia entry: facts organised around a topic, some "
            "abstraction, no argument the reader must track across "
            "paragraphs\n"
            "  8-10 = layered argument requiring attention, and at the top "
            "sustained abstraction that cannot be skimmed\n"
            "IMPORTANT: answer null instead of a number if the text is too "
            "damaged for the question to mean anything -- unconnected words "
            "can look profound while meaning nothing, and must not be scored."
        )),
    "naturalness": dict(
        ru="естественность как текста", kind="scale",
        prompt=(
            "Could a person have written this, in some genre, for some "
            "purpose? Judge against the standards of whatever genre it "
            "appears to belong to -- a screenplay, a table of records and a "
            "novel are each natural in their own terms.\n"
            "  1-4 = no genre or purpose accounts for this text\n"
            "  5 = recognisable as writing, but odd or artificial "
            "throughout\n"
            "  7 = plainly a real text of a real genre, unremarkable, with "
            "nothing in it that a reader would call strange\n"
            "  8-10 = an accomplished specimen of a real genre"
        )),
}


def build(text, cap_words=420):
    """One prompt scoring every property, each with a stated reason.

    The reason is required before the score, in a fixed two-line block per
    property. This makes every rating auditable -- a score whose justification
    is wrong can be seen to be wrong, which a bare number cannot -- and asking
    for the ground before the verdict tends to steady the ratings themselves.
    """
    words = text.split()
    body = " ".join(words[:cap_words])
    items = []
    for i, (key, p) in enumerate(PROPS.items(), 1):
        tail = {"count": "", "presence": " " + SCALE_PRESENCE}.get(
            p["kind"], " " + SCALE)
        items.append(f"{i}. {key.upper()}\n{p['prompt']}{tail}")
    listing = "\n\n".join(items)
    blocks = "\n\n".join(
        f"{k.upper()}\nREASON: <one sentence, citing what in the text decides "
        f"it>\nSCORE: <number{' or null' if PROPS[k]['kind'] == 'scale_cond' else ''}>"
        for k in PROPS)
    return (
        FRAME
        + "You are rating one text on several independent properties. Judge "
        "the text on its own terms; do not compare it with anything else, and "
        "do not let your rating on one property pull another with it.\n\n"
        f"{listing}\n\n"
        "OUTPUT FORMAT. Reply with exactly these blocks, in this order, "
        "nothing before and nothing after. Each REASON is a single sentence "
        "on its own line; each SCORE is a bare number on its own line.\n\n"
        f"{blocks}\n\n"
        f"TEXT:\n{body}\n\nOUTPUT:"
    )


def parse(out):
    """Pull {key: (score, reason)} out of the block format."""
    import re

    got = {}
    for k in PROPS:
        m = re.search(rf"{k.upper()}\s*\n\s*REASON:\s*(.*?)\s*\n\s*"
                      rf"SCORE:\s*([-\d.]+|null|N/?A)", out,
                      re.I | re.S)
        if not m:
            continue
        reason, raw = m.group(1).strip(), m.group(2).strip().lower()
        val = None if raw in ("null", "na", "n/a") else float(raw)
        got[k] = (val, reason)
    return got
