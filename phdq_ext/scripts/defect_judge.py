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
it, the score is 0.

Charge each piece of damage to one axis only -- the one that names what
happened. A stripped link that leaves a stray space before a full stop is WEB,
not also CHARS."""

# The anchors are deliberately blatant: the judge needs to learn where the
# levels sit, and calibrating it on borderline cases teaches it nothing.
OPS = {
    "loss": ("утрата содержания", """LOSS -- part of what the document said is
gone. Not formatting and not markup, which belong to WEB: meaning the document
promises and does not deliver.
  1 = a pointer to material that is not there ("see the table below", and no
      table follows)
  2 = a heading, a colon or an enumeration opening onto nothing
  3 = whole sections gone: a thought begins, breaks off, and the next sentence
      is about an unrelated subject"""),
    "web": ("парсинг веб-страницы", """WEB -- damage from turning a web page
into a document: delimiters lost, service blocks kept, separate pieces of the
page run together, markup left behind. This axis names where the damage came
from; the axes below describe other kinds.
  1 = small: list or paragraph separators gone so entries run together, an
      isolated leftover tag or bracket where a link was, a signature or a
      single service line. Nothing here obstructs understanding
  2 = medium: pieces of the page that are not the document have been kept or
      joined on -- a moderator footer, a navigation block, several separate
      answers on one topic strung into one body, an automated feed appended to
      an article. The document is still followable and its sense recoverable,
      but a reader has to skip past material that is not part of it
  3 = large: there is more junk than document. Menus, boilerplate and fragments
      of unrelated pages crowd out the text, and what the document was meant to
      say can no longer be made out

  Machine-generated blocks count even when they genuinely sat on the source
  page: event feeds, catalogues, listings, navigation, automated summaries. The
  question is not whether the page was intact but whether the document is
  usable as text."""),
    "dup": ("дублирование", """DUP -- the same content appears more than once.
  1 = a phrase or sentence repeats
  2 = a paragraph repeats verbatim
  3 = a large part of the document is a copy of another part

  Do not score rhetorical repetition a writer chose, such as an anaphora or a
  refrain. Score only repetition that looks mechanical."""),
    "chars": ("искажение записи слов", """CHARS -- the words are all there but
written wrongly: clitics split off (`do n't`), spaces lost between sentences
(`film.The`), spaces added around apostrophes and punctuation by a tokeniser,
hard line breaks inside sentences, stray characters, broken encoding.

Ordinary html residue -- footers, stripped links, lost separators -- is WEB.
CHARS is for systematic corruption that goes beyond it: formulas that fell out
or were replaced by rendering junk, a broken encoding turning text to mojibake,
a tokeniser applied and not undone, an element class the parser mangled
wherever it occurs. The mark of CHARS is that the same failure repeats at every
instance of its kind.

Score by extent, counting instances rather than judging ugliness.
  1 = isolated: fewer than about five instances in the document
  2 = systematic: most occurrences of the affected element are damaged -- every
      apostrophe, every sentence boundary, every line end. Reading slows but
      the text is readable
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

WEB: <0-3> | <verbatim quote or ->
LOSS: <0-3> | <verbatim quote or ->
DUP: <0-3> | <verbatim quote or ->
CHARS: <0-3> | <verbatim quote or ->
CUT: <0-3> | <verbatim quote or ->
NATURE: <1-5>
HUMAN: <0-3>
DAMAGE: <0-5, overall severity of pipeline damage, from the scores above only
and never from how well the document is written. Only the two ends were
anchored at first and the answers piled up at the bottom, so every level is
named:
  0 = nothing found; the document arrived intact
  1 = isolated blemishes; a reader would pass over them without noticing
  2 = clearly present but local -- one appended block, a handful of corrupted
      spots. The document is fully usable
  3 = systematic: the defect runs through the whole document, at every
      apostrophe, every formula, every sentence boundary, or an intruding block
      takes up a real share of the text. Still usable, but a model trained on
      this would learn the defect along with the content
  4 = severe: the damage competes with the content. Much of what is here should
      not be here, and what should be is hard to follow
  5 = useless as data: what the document meant to say can no longer be
      recovered>
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
