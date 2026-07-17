"""Deterministic line-break reflow for translated subtitles.

The translator produces one English line per Japanese line, with the line
boundaries frozen upstream at the Format stage based on *Japanese* cues
(clause-final particles like けど/から/そして followed by a pause). Because those
connectives are clause-*final* in Japanese but clause-*initial* in English, the
1:1 positional translation strands them at the end of a line ("...refreshing,
but"). This module re-decides where the English breaks, with English awareness:

1. Group consecutive lines the model translated as one flowing sentence.
2. Re-split each group's English across the same time slots at natural
   boundaries (after punctuation / before a connective), balanced by each
   slot's display duration, never ending a piece on a dangling function word.

The number of lines in and out is identical, so all downstream machinery
(timing, bilingual embedding, checkpointing) is untouched. A guardrail
guarantees only whitespace is redistributed and casing adjusted — no words are
added, dropped, or altered — otherwise the original split is kept verbatim.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# A re-split engine takes a batch of sentence groups — each a (pieces, durations)
# pair — and returns a proposed re-split (or None to leave it alone) per group,
# aligned to the input. Batching lets the LLM engine serve every group in one
# call; the deterministic engine just maps over the batch.
ReflowGroup = "tuple[list[str], list[float]]"
Resplitter = Callable[[list["tuple[list[str], list[float]]"]], list["list[str] | None"]]

# A piece must never *end* on one of these bare words — that is the dangling
# connective/preposition/article/auxiliary that reads as broken mid-thought.
# (Words carrying trailing punctuation are always allowed to end a piece.)
NO_END = {
    # coordinating / subordinating conjunctions
    "and", "but", "or", "nor", "so", "yet", "for", "because", "though",
    "although", "while", "if", "when", "since", "as", "than", "that",
    "whether", "unless", "until", "before", "after", "once",
    # prepositions
    "of", "to", "in", "on", "at", "by", "with", "from", "into", "onto",
    "upon", "about", "over", "under", "between", "among", "through",
    "during", "without", "within", "toward", "towards", "per",
    # articles / determiners / possessives
    "a", "an", "the", "my", "your", "his", "her", "its", "our", "their",
    "this", "these", "those",
    # auxiliaries / copulas
    "is", "was", "are", "were", "be", "been", "being", "will", "would",
    "can", "could", "should", "may", "might", "must", "do", "does", "did",
    "has", "have", "had", "not",
}

# Breaking *before* one of these words is a natural clause boundary.
START_WORDS = {
    "and", "but", "or", "nor", "so", "yet", "because", "though", "although",
    "while", "if", "when", "since", "as", "than", "that", "whether", "unless",
    "until", "before", "after", "once", "which", "who", "where",
}

# Words safe to lowercase when they land mid-sentence after a re-join (they are
# never proper nouns, so "..., But then" -> "..., but then").
SAFE_LOWER = {
    "and", "but", "or", "nor", "so", "yet", "because", "though", "although",
    "while", "if", "when", "since", "as", "than", "that", "in", "on", "at",
    "of", "to", "the", "a", "an", "with", "from", "by",
}

_TERMINAL_PUNCT = (".", "!", "?", "…")
_CLAUSE_PUNCT = (",", ";", ":", ")", '"', "”", *_TERMINAL_PUNCT)
# Trailing characters stripped before classifying a word (quotes/brackets).
_TRAIL_STRIP = "".join(_CLAUSE_PUNCT) + "'’"

# Bonuses, in character-equivalent units, that pull a chosen break toward a
# higher-quality boundary even when it is slightly farther from the target.
_PUNCT_BONUS = 25.0
_BEFORE_START_BONUS = 12.0

LONG_GAP_S = 1.5  # matches the format chunker's long-pause split threshold


def reflow_line_breaks(
    texts: list[str],
    durations_s: list[float],
    boundaries: set[int] | None = None,
    resplitter: Resplitter | None = None,
) -> list[str]:
    """Re-split translated lines at natural English boundaries.

    Args:
        texts: translated lines, one per subtitle event, in order.
        durations_s: display duration (seconds) of each line's time slot.
        boundaries: indices ``i`` where a hard group-break must occur *before*
            line ``i`` (speaker change, corner boundary, large time gap).
        resplitter: engine that proposes a re-split per sentence group. Defaults
            to the deterministic engine. An alternative (e.g. LLM-backed) engine
            is held to the same guardrail and quality gate, so it can only
            improve on the original split, never corrupt or churn it.

    Returns:
        A new list of the same length with text redistributed across each
        sentence group's slots. Lines outside a multi-line group are unchanged.
    """
    if len(texts) != len(durations_s):
        raise ValueError(
            f"texts ({len(texts)}) and durations_s ({len(durations_s)}) length mismatch"
        )
    boundaries = boundaries or set()
    n = len(texts)
    if n == 0:
        return list(texts)
    if resplitter is None:
        resplitter = _deterministic_batch

    groups = [g for g in _group_indices(texts, boundaries) if len(g) >= 2]
    batch = [([texts[i] for i in g], [durations_s[i] for i in g]) for g in groups]
    proposals = resplitter(batch) if batch else []
    if len(proposals) != len(groups):
        logger.warning(
            "Reflow engine returned %d proposals for %d groups; skipping reflow.",
            len(proposals),
            len(groups),
        )
        return list(texts)

    result = list(texts)
    reflowed_groups = 0
    for group, (pieces, _durs), new_pieces in zip(groups, batch, proposals):
        if not _accept_resplit(new_pieces, pieces):
            continue
        for slot, i in enumerate(group):
            result[i] = new_pieces[slot]
        reflowed_groups += 1

    if reflowed_groups:
        logger.info(
            "Line-break reflow adjusted %d sentence group(s).", reflowed_groups
        )
    return result


def _deterministic_batch(
    batch: list[tuple[list[str], list[float]]],
) -> list[list[str] | None]:
    """Default engine: apply the deterministic re-split to each group."""
    return [_resplit_group(pieces, durs) for pieces, durs in batch]


def _accept_resplit(new_pieces: list[str] | None, pieces: list[str]) -> bool:
    """Gate a proposed re-split against the original.

    A proposal is accepted only when it preserves the text (whitespace/casing
    aside) and either scores strictly higher or is a casing-only change. This
    guarantees reflow never churns an already-clean break into a worse or equal
    one — regardless of which engine produced the proposal.
    """
    if new_pieces is None or not _is_preserving(new_pieces, pieces):
        return False
    if new_pieces == pieces:
        return False
    strictly_better = _split_score(new_pieces) > _split_score(pieces)
    casing_only = _word_partition(new_pieces) == _word_partition(pieces)
    return strictly_better or casing_only


def _group_indices(texts: list[str], boundaries: set[int]) -> list[list[int]]:
    """Group consecutive line indices that form one flowing sentence.

    A new group starts before line ``i`` when the previous English line already
    ended a sentence (terminal punctuation) or a hard boundary is marked there.
    """
    groups: list[list[int]] = []
    current: list[int] = [0]
    for i in range(1, len(texts)):
        if i in boundaries or _ends_sentence(texts[i - 1]):
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    groups.append(current)
    return groups


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip().rstrip('"”\')’')
    return stripped.endswith(_TERMINAL_PUNCT)


def _resplit_group(pieces: list[str], durs: list[float]) -> list[str] | None:
    """Re-split a group's concatenated English into ``len(pieces)`` slots.

    Returns None if it cannot produce that many non-empty pieces, signalling the
    caller to keep the original split.
    """
    k = len(pieces)
    words = " ".join(p.strip() for p in pieces).split()
    if len(words) < k:
        return None

    # Lowercase safe function words that ended up mid-sentence after re-joining.
    # Single-letter tokens are left alone so labels like "A"/"B" sections and the
    # pronoun "I" are never clobbered.
    for idx in range(1, len(words)):
        w = words[idx]
        core = w.rstrip(_TRAIL_STRIP).lower()
        if len(core) > 1 and core in SAFE_LOWER and w[:1].isupper():
            words[idx] = w[0].lower() + w[1:]

    n = len(words)
    # Cumulative character length up to and including each word (single-spaced).
    cum_chars: list[int] = []
    running = 0
    for j, w in enumerate(words):
        running += len(w) + (1 if j > 0 else 0)
        cum_chars.append(running)
    total_chars = cum_chars[-1]
    total_dur = sum(durs)

    cuts: list[int] = []  # index of the last word of each piece (except last)
    prev = -1
    cum_dur = 0.0
    for m in range(k - 1):
        cum_dur += durs[m]
        if total_dur > 0:
            target = total_chars * (cum_dur / total_dur)
        else:
            target = total_chars * ((m + 1) / k)
        remaining_pieces = k - m - 1  # pieces still needed after this cut
        lo = prev + 1
        hi = n - 1 - remaining_pieces  # leave >=1 word for each remaining piece
        if lo > hi:
            return None
        j = _best_cut(words, cum_chars, target, lo, hi)
        if j is None:
            return None
        cuts.append(j)
        prev = j

    bounds = [-1, *cuts, n - 1]
    out: list[str] = []
    for a in range(k):
        seg = words[bounds[a] + 1 : bounds[a + 1] + 1]
        if not seg:
            return None
        out.append(" ".join(seg))
    return out


def _best_cut(
    words: list[str], cum_chars: list[int], target: float, lo: int, hi: int
) -> int | None:
    """Pick the word index (inclusive end of a piece) with the best score.

    Score rewards proximity to the character target and penalises poor break
    quality; candidates that would end a piece on a bare function word are
    excluded entirely.
    """
    best_j: int | None = None
    best_score = float("inf")
    for j in range(lo, hi + 1):
        if _forbidden_end(words[j]):
            continue
        score = abs(cum_chars[j] - target)
        score -= _quality_bonus(words, j)
        if score < best_score:
            best_score = score
            best_j = j
    return best_j


def _forbidden_end(word: str) -> bool:
    """A piece may not end on a bare (unpunctuated) function word."""
    if word[-1:] in _CLAUSE_PUNCT:
        return False
    return word.lower() in NO_END


def _quality_bonus(words: list[str], j: int) -> float:
    bonus = 0.0
    if words[j].rstrip('"”\')’').endswith(_CLAUSE_PUNCT):
        bonus += _PUNCT_BONUS
    if j + 1 < len(words):
        nxt = words[j + 1].lower().strip(_TRAIL_STRIP)
        if nxt in START_WORDS:
            bonus += _BEFORE_START_BONUS
    return bonus


def _word_partition(pieces: list[str]) -> list[int]:
    """Word count per piece — identical partitions differ only in casing."""
    return [len(p.split()) for p in pieces]


def _split_score(pieces: list[str]) -> float:
    """Rate the quality of a split by the strength of its internal boundaries.

    Higher is better. Ending a line on terminal or clause punctuation is
    rewarded; ending on a bare function word is heavily penalised; starting the
    next line on a connective (a natural clause opener) earns a small bonus.
    """
    total = 0.0
    for a in range(len(pieces) - 1):
        end_words = pieces[a].split()
        start_words = pieces[a + 1].split()
        if not end_words or not start_words:
            continue
        total += _boundary_score(end_words[-1], start_words[0])
    return total


def _boundary_score(end_word: str, start_word: str) -> float:
    score = 0.0
    trimmed = end_word.rstrip('"”\')’')
    if trimmed.endswith(_TERMINAL_PUNCT):
        score += 3.0
    elif trimmed.endswith((",", ";", ":")):
        score += 2.0
    elif _forbidden_end(end_word):
        score -= 10.0
    if start_word.lower().strip(_TRAIL_STRIP) in START_WORDS:
        score += 1.0
    return score


def _is_preserving(new_pieces: list[str], old_pieces: list[str]) -> bool:
    """Guarantee reflow only redistributed whitespace and adjusted casing."""
    return _canonical(new_pieces) == _canonical(old_pieces)


def _canonical(pieces: list[str]) -> str:
    return "".join(" ".join(pieces).split()).lower()
