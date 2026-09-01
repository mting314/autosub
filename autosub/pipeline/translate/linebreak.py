"""Netflix-style line breaking for translated subtitles.

Where a subtitle may be broken is a grammatical question, not an arithmetic one.
Breaking at the midpoint of a string reliably strands articles from their nouns
and verbs from their complements, which reads as broken mid-thought.

This encodes the Netflix Timed Text Style Guide line-break rules (the de-facto
industry standard; BBC's guidelines and the subtitle-segmentation literature —
Karakanta et al., MuST-Cinema — agree):

    Break: after punctuation, before conjunctions, before prepositions.
    NEVER separate: article/adjective from noun, first from last name, verb from
    its subject pronoun, a prepositional verb from its preposition, a verb from
    an auxiliary, reflexive pronoun, or negation.

Two engines pick the boundary:

- ``dep`` (preferred): a spaCy dependency parse. A break is only allowed where
  the two sides are separate subtrees, which structurally guarantees none of the
  "never separate" pairs are split.
- ``regex``: a dependency-free approximation (sentence ends, commas,
  comma-gated coordinators, no infinitive "to"). Used automatically when spaCy
  or its model is not installed, so the pipeline degrades instead of failing.

Ported from ``scripts/autosplit_long_lines.py`` in the subtitling-projects repo,
which applies the same rules as a post-hoc QC pass.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# Netflix caps an English subtitle at 42 characters per line, two lines maximum.
# 42 assumes a full-width bottom-centred subtitle; when the script's own layout
# gives a line more or less room than that, capacity_for_style() is used instead
# so the budget matches the box the text actually renders into.
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2

# Mean glyph advance as a fraction of font size, measured by rendering mixed-case
# English through libass at Arial 54 (24.2px per character).
_CHAR_WIDTH_RATIO = 0.45

# A break must leave at least this share of the line on each side, so a
# grammatically valid boundary cannot produce a one-word orphan line.
MIN_SHARE = 0.3

COORD = ("and", "but", "so", "or", "yet", "nor")
SUBORD = (
    "because",
    "when",
    "while",
    "since",
    "although",
    "though",
    "which",
    "where",
    "who",
    "that",
    "if",
)
COMPLEMENTIZERS = {
    "that",
    "how",
    "why",
    "what",
    "whether",
    "if",
    "when",
    "where",
    "who",
}
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])|(?<=\.\.\.)\s+")

# Dependency labels that bind two tokens into one tight unit — a break must never
# fall between them. This is the Netflix "never separate" list, structurally.
HARD_TIGHT = {
    "det",
    "poss",
    "predet",
    "amod",
    "compound",
    "nummod",
    "flat",
    "fixed",
    "aux",
    "auxpass",
    "cop",
    "neg",
    "prt",
    "case",
    "nsubj",
    "nsubjpass",
    "csubj",
    "csubjpass",
    "expl",
    "dobj",
    "dative",
    "attr",
    "acomp",
    "oprd",
    "agent",
    "nmod",
    "appos",
    "pcomp",
    "pobj",
    "quantmod",
}

# Cognition/speech verbs that take a (often that-less) complement; ending a line
# on one strands the complement ("I really felt | they wrote...").
COMPLEMENT_VERBS = {
    "felt",
    "feel",
    "feels",
    "think",
    "thinks",
    "thought",
    "know",
    "knows",
    "knew",
    "believe",
    "believed",
    "realize",
    "realized",
    "said",
    "say",
    "says",
    "hope",
    "hoped",
    "wish",
    "wished",
    "guess",
    "suppose",
    "mean",
    "meant",
    "see",
    "saw",
    "notice",
    "noticed",
    "find",
    "found",
    "heard",
    "hear",
    "wonder",
    "wondered",
    "assume",
    "assumed",
    "decided",
    "remember",
    "remembered",
    "imagine",
    "imagined",
    "figured",
    "worried",
}

_TAG_RE = re.compile(r"\{[^}]*\}")


def strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


@lru_cache(maxsize=1)
def load_nlp():
    """Load the spaCy English model, or None when it is unavailable.

    Cached because loading the model costs ~1s and every line needs it.
    """
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception as exc:
        logger.info(
            "spaCy/en_core_web_sm unavailable (%s); using the regex line-break engine.",
            exc.__class__.__name__,
        )
        return None


def _candidates_dep(nlp, prose: str) -> list[tuple[int, float]]:
    """Allowed break offsets from a dependency parse, with a preference penalty."""
    doc = nlp(prose)
    toks = list(doc)
    if len(toks) < 4:
        return []

    out: list[tuple[int, float]] = []
    for k in range(1, len(toks)):
        tk = toks[k]
        idx = tk.idx

        if tk.tag_ == "TO":  # never break before an infinitive marker
            continue
        if (
            tk.dep_ == "cc" or tk.pos_ == "CCONJ" or tk.text.lower() in COORD
        ) and toks[k - 1].text != ",":  # a coordinator needs a preceding comma
            continue
        if toks[k - 1].text.lower() in COORD:  # no dangling 'and'/'but' at line end
            continue

        crossing = [t for t in doc if min(t.head.i, t.i) < k <= max(t.head.i, t.i)]
        forbid = False
        for t in crossing:
            if t.dep_ in HARD_TIGHT:
                forbid = True
                break
            if t.dep_ in ("ccomp", "xcomp"):  # verb | bare complement
                first = min(t.subtree, key=lambda x: x.i)
                if (
                    first.i == k
                    and tk.dep_ != "mark"
                    and tk.tag_ != "WDT"
                    and tk.text.lower() not in COMPLEMENTIZERS
                ):
                    forbid = True
                    break
        if forbid:
            continue

        prev = toks[k - 1]
        penalty = 0.0
        if prev.is_punct or prev.text in ".!?":
            penalty -= 100
        if tk.dep_ in (
            "mark",
            "advcl",
            "cc",
            "conj",
            "relcl",
            "parataxis",
            "prep",
            "advmod",
        ):
            penalty -= 30
        penalty += len(crossing) * 5
        out.append((idx, penalty))
    return out


def _candidates_regex(prose: str) -> list[tuple[int, float]]:
    """Allowed break offsets without a parser: punctuation and connectives."""
    out: list[tuple[int, float]] = []
    for m in SENT_END.finditer(prose):
        out.append((m.end(), -100.0))
    for m in re.finditer(r", ", prose):
        out.append((m.end(), -60.0))
    for c in COORD:
        for m in re.finditer(rf", {c} ", prose):
            out.append((m.start() + 2, -40.0))
    for c in SUBORD:
        for m in re.finditer(rf"\b{c} ", prose):
            if m.start():
                out.append((m.start(), -20.0))
    return out


def _break_offsets(prose: str, nlp=None) -> list[tuple[int, float]]:
    """Every grammatically safe break offset in prose, with preference penalties."""
    if nlp is not None:
        try:
            return _candidates_dep(nlp, prose)
        except Exception as exc:  # a parse failure must not sink the pipeline
            logger.warning("Dependency parse failed (%s); falling back to regex.", exc)
    return _candidates_regex(prose)


def _normalized_with_offsets(text: str) -> tuple[str, str, list[int]]:
    """Normalise whitespace while keeping track of where the visible text sits.

    Returns the tagged text with existing breaks and runs of whitespace collapsed
    to single spaces, the visible characters alone, and for each visible
    character its index in the tagged string. Breaks are chosen against what the
    viewer reads but applied to the tagged string, so inline formatting such as
    the italics on a Japanese term survives the wrap.
    """
    tagged: list[str] = []
    visible: list[str] = []
    offsets: list[int] = []
    tagged_len = 0
    pending_space = False
    i, n = 0, len(text)

    while i < n:
        if text[i] == "{":
            close = text.find("}", i)
            if close == -1:
                break
            # Emit a pending space before the block so the tag stays attached to
            # the word it formats rather than drifting in front of the space.
            if pending_space and visible:
                tagged.append(" ")
                visible.append(" ")
                offsets.append(tagged_len)
                tagged_len += 1
                pending_space = False
            block = text[i : close + 1]
            tagged.append(block)
            tagged_len += len(block)
            i = close + 1
            continue
        if text.startswith("\\N", i):
            pending_space = True
            i += 2
            continue
        char = text[i]
        if char.isspace():
            pending_space = True
            i += 1
            continue
        if pending_space and visible:
            tagged.append(" ")
            visible.append(" ")
            offsets.append(tagged_len)
            tagged_len += 1
        pending_space = False
        tagged.append(char)
        visible.append(char)
        offsets.append(tagged_len)
        tagged_len += 1
        i += 1

    return "".join(tagged), "".join(visible), offsets


def visible_length(text: str) -> int:
    """Number of characters the viewer actually sees, ignoring override blocks."""
    return len(_normalized_with_offsets(text)[1])


def best_break(
    prose: str,
    nlp=None,
    max_chars: int = MAX_CHARS_PER_LINE,
    require_fit: bool = True,
    min_share: float = MIN_SHARE,
) -> int | None:
    """Pick the best offset to break prose into two parts.

    With require_fit, only offsets leaving both sides within max_chars are
    considered, so the result is a legal two-line subtitle. Without it, the
    break merely has to be grammatical — used when the text is too long for two
    lines and has to become two separate events instead.

    Breaks in the balanced middle of the line are strongly preferred: a
    grammatical break that leaves a one-word first line still reads badly, so
    lopsided breaks are only used when nothing else is available.
    """
    total = len(prose)
    candidates = _break_offsets(prose, nlp)
    if not candidates:
        return None

    balanced, lopsided = [], []
    for idx, penalty in candidates:
        left = prose[:idx].strip()
        right = prose[idx:].strip()
        if len(left) < 2 or len(right) < 2:
            continue
        if require_fit and (len(left) > max_chars or len(right) > max_chars):
            continue
        # An odd number of quote marks on the left means the break falls inside
        # a quotation ('That "' | 'one" represents...').
        if left.count('"') % 2:
            continue
        # Prefer a grammatically strong break, then a balanced one.
        scored = (penalty + abs(idx - total / 2) * 0.1, idx)
        if min_share * total <= idx <= (1 - min_share) * total:
            balanced.append(scored)
        else:
            lopsided.append(scored)

    viable = balanced or lopsided
    if not viable:
        return None
    return min(viable)[1]


def normalized_text(text: str) -> str:
    """The text with existing breaks and repeated whitespace collapsed."""
    return _normalized_with_offsets(text)[0]


def wrap_line(
    text: str, nlp=None, max_chars: int = MAX_CHARS_PER_LINE
) -> str | None:
    """Lay text out as one or two lines within max_chars, breaking grammatically.

    Returns the text with a single ``\\N`` inserted, the text unchanged when it
    already fits on one line, or None when no legal two-line layout exists (the
    caller should split it into two events instead).
    """
    tagged, prose, offsets = _normalized_with_offsets(text)
    if not prose:
        return tagged
    if len(prose) <= max_chars:
        return tagged

    idx = best_break(prose, nlp, max_chars, require_fit=True)
    if idx is None:
        return None
    cut = offsets[idx]
    return f"{tagged[:cut].rstrip()}\\N{tagged[cut:].lstrip()}"


def split_text(
    text: str, nlp=None, max_chars: int = MAX_CHARS_PER_LINE
) -> tuple[str, str] | None:
    """Cut text into two pieces for two separate events, or None if unsafe.

    Used when the text cannot be laid out as two lines. Prefers a break that
    leaves the first piece fully laid out, and otherwise takes the most
    grammatical break available.
    """
    tagged, prose, offsets = _normalized_with_offsets(text)
    if not prose:
        return None

    idx = best_break(prose, nlp, max_chars * MAX_LINES, require_fit=True)
    if idx is None:
        idx = best_break(prose, nlp, max_chars, require_fit=False)
    if idx is None:
        return None

    cut = offsets[idx]
    return tagged[:cut].rstrip(), tagged[cut:].lstrip()


def capacity_for_style(
    play_res_x: int | None,
    margin_l: int | None,
    margin_r: int | None,
    font_size: float | None,
) -> int:
    """Characters that fit on one line of the box this style renders into.

    Netflix's 42 assumes a full-width subtitle. A positioned line — a radio
    overlay slot, say — has its own width, and breaking at 42 there splits lines
    that would comfortably fit. Falls back to the Netflix figure when the script
    does not pin down the geometry.
    """
    if not play_res_x or not font_size:
        return MAX_CHARS_PER_LINE
    usable = play_res_x - (margin_l or 0) - (margin_r or 0)
    if usable <= 0:
        return MAX_CHARS_PER_LINE
    return max(20, int(usable / (font_size * _CHAR_WIDTH_RATIO)))
