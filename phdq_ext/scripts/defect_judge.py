"""A judge for what a pipeline did to a document, not for how well it is written.

The previous judge scored ten aspects of prose quality and found nothing. On
twenty texts read by hand it gave 8 out of 10 to a document largely composed of
{\\displaystyle s}, and separated outliers from controls by 0.4 points. It was
measuring the wrong thing: a scraped page with its formulas torn out reads
beautifully and is useless, while a rambling forum answer full of typos is
excellent data.

Three design choices, each answering a specific way that attempt failed.

The list is of *operations* rather than of artefacts. A pipeline can only lose
material, break the document's unity, duplicate, corrupt the writing or
truncate; every concrete trace -- MathML residue, a moderator footer, `do n't`,
a flattened table -- is one of those five. Insertion and concatenation were
separate at first and are now one score, because they are the same failure seen
from two sides: a reader does not care whether the foreign block arrived by
insertion or by joining, only whether the document still reads as one thing.
Naming the operations rather than the traces is what lets the same judge work on
a corpus whose artefacts we have never seen.

Every non-zero score must carry a verbatim quotation, and the quotation is
checked against the document afterwards. A model asked for an impression will
always supply one; a model asked to point at the offending characters either can
or cannot.

Human writing is scored on its own axis and excluded from the damage total.
Conflating "a person wrote this carelessly" with "a pipeline damaged this" is
precisely what sank the previous judge.
"""
import re

FRAME = """You are auditing documents scraped from the web and processed into a
machine-learning training corpus. Find the damage the pipeline did to the
document. Do not judge how well it is written.

This distinction is the entire task. A rambling forum comment full of typos,
slang and missing commas is undamaged -- a person wrote it that way and it is
good training data. A polished encyclopedia article whose formulas have been
replaced by rendering junk is damaged, however well its sentences read.

Score each operation from 0 to 3. Every score above 0 must be justified by
quoting the offending text verbatim, up to fifteen words. If you cannot quote
it, the score is 0."""

# The anchors are deliberately blatant: the judge needs to learn where the
# levels sit, and calibrating it on borderline cases teaches it nothing.
OPS = {
    "loss": ("утрата", """LOSS -- content or structure was removed.

  Structure counts. A list whose delimiters are gone, its items running
  together as one paragraph, has lost as much as a list whose items are gone:
  the words survive and the document no longer says what they are.

  1 = a pointer to something that is not there ("see the table below", and no
      table follows); a few boundaries lost
  2 = a heading, a colon or an enumeration opening onto nothing, so the
      document promises material it never delivers; or an enumeration flattened
      into running prose so that entries collide -- "Foo, 2019 (Composer: X)
      Premiere: Y Bar, 2018 (Composer: Z) Premiere: W"
  3 = whole sections gone: a thought begins, breaks off, and the next sentence
      is about an unrelated subject; or a table reduced to an unreadable run of
      cell values"""),
    "unity": ("цельность", """UNITY -- does the document hold together as one
readable object, or has it come apart? Material foreign to it and material
joined onto it are the same failure seen from two sides, and are scored
together.
  0 = one object: everything in it belongs to it
  1 = one object with a blemish: an isolated leftover tag, a signature, one
      abrupt transition. A reader passes over it without losing the thread
  2 = noticeably composite: an appended standard block, a second document run
      on after the first, several separate pieces strung together. A reader has
      to skip past material that is not part of what they are reading
  3 = it does not hold together at all: a container of unrelated fragments with
      no single document in it

  Machine-generated material counts here even when it genuinely sat on the
  source page: event feeds, catalogues, listings, navigation, automated
  summaries. The question is not whether the page was intact but whether the
  document is usable as text."""),
    "dup": ("дублирование", """DUP -- the same content appears more than once.
  1 = a phrase or sentence repeats
  2 = a paragraph repeats verbatim
  3 = a large part of the document is a copy of another part

  Do not score rhetorical repetition a writer chose, such as an anaphora or a
  refrain. Score only repetition that looks mechanical."""),
    "chars": ("искажение символов", """CHARS -- the way words are written was
corrupted, independently of what they say: lost or added spaces, spaces before
punctuation, clitics split off as separate words, words fused across a
boundary, hard line breaks inside sentences, stray characters, broken encoding,
a class of element mangled by the parser wherever it occurs.

Score by how far it spreads and what it costs the reader, not by how ugly any
one instance is.
  1 = isolated: a handful of instances in the whole document; reading is
      unaffected
  2 = throughout the document, systematically -- the same failure at every
      sentence boundary, at every apostrophe, at every formula; reading slows
      but the text is still readable
  3 = readability is destroyed: the words can no longer be reliably made
      out"""),
    "cut": ("обрезка", """CUT -- the document's boundaries were lost.
  1 = the ending is unfinished but the document is substantially whole
  2 = it breaks off mid-sentence, or begins mid-sentence
  3 = neither beginning nor end is present: a fragment cut from the middle"""),
}

# "not a defect" was read as a licence: the judge answered "a mixed document by
# nature" and closed the case on a match report glued to a live event feed.
NATURE = """Record the form of the document. This is description only and moves
the damage score in neither direction.
  1 = continuous prose
  2 = prose with formal apparatus: headings, citations, quoted blocks
  3 = predominantly an enumeration: list, table, catalogue, running commentary
  4 = a template: a form, a questionnaire, an automatically generated page
  5 = a mixture: a prose section together with an enumerated one"""

HUMAN = """How the person wrote. This is NOT damage and does NOT enter the
damage total.
  0 = careful standard writing
  1 = informal but controlled
  2 = conversational: slang, missing commas, typos, loose sentences
  3 = barely organised: stream of thought, heavy misspelling, no structure"""

FORMAT = """Answer in exactly these lines, each beginning with its tag. Use "-"
where no quotation is required.

LOSS: <0-3> | <verbatim quote or ->
UNITY: <0-3> | <verbatim quote or ->
DUP: <0-3> | <verbatim quote or ->
CHARS: <0-3> | <verbatim quote or ->
CUT: <0-3> | <verbatim quote or ->
NATURE: <1-5>
HUMAN: <0-3>
DAMAGE: <0-5, overall severity of pipeline damage, from the six scores above
only; 0 = untouched, 5 = destroyed as data>
VERDICT: keep | keep-with-caveats | drop
WHY: <one line>"""


def build(text):
    return "\n\n".join([FRAME] + [v[1] for v in OPS.values()]
                       + [NATURE, HUMAN, FORMAT,
                          "---\nDOCUMENT:\n" + text.strip()])


TAGS = {k: k.upper() for k in OPS} | {"nature": "NATURE", "human": "HUMAN",
                                      "damage": "DAMAGE"}


def parse(out):
    r = {}
    for key, tag in TAGS.items():
        m = re.search(rf"^{tag}\s*:\s*(\d+)(?:\s*\|\s*(.*))?$", out,
                      re.M | re.I)
        if not m:
            continue
        r[key] = int(m.group(1))
        if key in OPS:
            q = (m.group(2) or "").strip()
            r[f"q_{key}"] = "" if q in ("-", "") else q
    for tag, key in (("VERDICT", "verdict"), ("WHY", "why")):
        m = re.search(rf"^{tag}\s*:\s*(.+)$", out, re.M | re.I)
        r[key] = m.group(1).strip() if m else ""
    r["verdict"] = r.get("verdict", "").lower()
    return r


def check_quotes(rec, text):
    """Did the quotations actually occur in the document?

    The requirement is only worth having if it is enforced: an unverifiable
    quotation is the same confabulation the scores were meant to replace.
    """
    norm = " ".join(text.split()).lower()
    bad = []
    for k in OPS:
        if rec.get(k, 0) > 0:
            # the model often echoes the template's dash before the quote,
            # and quotes it with whatever punctuation it likes
            q = " ".join(rec.get(f"q_{k}", "").split()).lower()
            q = q.lstrip("-–— ").strip('"“”\'`…')
            if not q:
                bad.append(f"{k}: нет цитаты")
            elif q not in norm:
                bad.append(f"{k}: цитата не найдена")
    return bad
