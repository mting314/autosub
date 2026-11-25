from __future__ import annotations

import logging
import re
from pathlib import Path

from autosub.core.schemas import SubtitleCue, SubtitleDocument
from autosub.core.speaker_map import (
    build_slot_lookup,
    build_style_name_lookup,
    style_name_for_speaker,
)
from autosub.pipeline.format.generator import build_ass_script, render_ass_document
from autosub.pipeline.format.timing import (
    _close_screen_gaps,
    _offset_replacement_spans,
    _original_text_length,
    _roles_compatible_for_merge,
    check_display_invariants,
    slot_key_for_speaker,
)
from autosub.pipeline.translate.linebreak import (
    MAX_CHARS_PER_LINE,
    capacity_for_style,
    load_nlp,
    normalized_text,
    wrap_line,
)

logger = logging.getLogger(__name__)

# Enough to see the shape of a regression without burying the rest of the log.
_MAX_LOGGED_VIOLATIONS = 20

QUOTE_CHARS = {'"', "“", "”"}
LINE_BREAK_RE = re.compile(r"(\\N|\\n|\r\n|\n|\r)")
# Written into the line by the translator, which is asked to prepend it so segment
# boundaries are visible while translating. Translate strips it before measuring
# line widths; repeated here so a document from an older run can be fixed by
# re-running this stage rather than the whole pipeline.
CORNER_MARKER_RE = re.compile(r"^((?:\{[^}]*\})*)\s*\[CORNER:[^\]]*\]\s*")


class _CueSpan:
    """Adapter exposing the start_ms/end_ms that the timing passes work in."""

    def __init__(self, cue: SubtitleCue):
        self.cue = cue
        self.start_ms = round(cue.start_time * 1000)
        self.end_ms = round(cue.end_time * 1000)


def _strip_corner_markers(document: SubtitleDocument) -> int:
    stripped = 0
    for cue in document.cues:
        for field in ("translated_text", "final_text"):
            value = getattr(cue, field, None)
            if not value:
                continue
            cleaned = CORNER_MARKER_RE.sub(r"\1", value)
            if cleaned != value:
                setattr(cue, field, cleaned)
                stripped += 1
    return stripped


def postprocess_subtitles(
    input_json_path: Path,
    output_json_path: Path | None = None,
    output_ass_path: Path | None = None,
    extensions_config: dict | None = None,
    bilingual: bool = True,
    speaker_map: dict[str, dict] | None = None,
    min_duration_ms: int = 500,
    min_gap_ms: int | None = None,
) -> bool:
    """Produce the file that ships, and make it satisfy the on-screen rules.

    This is the stage that owns the finished output, so re-running it on an older
    document is also how such a document is brought up to date — no
    re-transcribing, no re-translating, and any manual QC in the text survives.
    """
    if output_json_path is None:
        output_json_path = input_json_path.with_name("postprocessed.json")
    if output_ass_path is None:
        output_ass_path = output_json_path.with_suffix(".ass")

    document = SubtitleDocument.model_validate_json(
        input_json_path.read_text(encoding="utf-8")
    )
    if document.stage != "translated":
        raise ValueError(
            f"postprocess expects stage='translated', got {document.stage!r}"
        )
    processed = document.model_copy(deep=True)
    processed.stage = "postprocessed"
    for cue in processed.cues:
        cue.final_text = cue.final_text or cue.translated_text

    modified = False
    stripped = _strip_corner_markers(processed)
    if stripped:
        modified = True
        logger.info(
            "Removed the [CORNER: ...] prefix from %d line(s); the boundary is "
            "carried by cue.corner and the rendered corner comments.",
            stripped,
        )

    gap_floor = min_duration_ms if min_gap_ms is None else min_gap_ms
    slot_lookup = build_slot_lookup(speaker_map)

    merged_flashes = _merge_same_slot_flash_cues(
        processed,
        speaker_map=speaker_map,
        min_duration_ms=min_duration_ms,
        min_gap_ms=gap_floor,
        bilingual=bilingual,
    )
    if merged_flashes:
        modified = True
        logger.info(
            "Merged %d chained same-slot flash/fragment cue(s).",
            merged_flashes,
        )

    rewrapped = _rewrap_overlong_cues(
        processed,
        speaker_map=speaker_map,
        bilingual=bilingual,
    )
    if rewrapped:
        modified = True
        logger.info(
            "Re-wrapped %d cue(s) to fit the current slot style width.",
            rewrapped,
        )

    repaired_overlaps = _repair_cross_slot_turn_overlaps(
        processed.cues,
        speaker_map=speaker_map,
        min_duration_ms=min_duration_ms,
    )
    if repaired_overlaps:
        modified = True
        logger.info(
            "Adjusted %d cross-slot turn transition(s) to eliminate artificial overlaps and hold across backchannels.",
            repaired_overlaps,
        )
    extensions_config = extensions_config or {}
    radio_discourse_config = extensions_config.get("radio_discourse", {})
    if radio_discourse_config.get("enabled"):
        if _apply_radio_discourse_postprocess(processed):
            modified = True
            logger.info("Postprocessing modified subtitle document.")

    # Close gaps too short to read as a pause, across every slot. This only
    # settles once translation has finished splitting and reflowing cues, which
    # is why it lives here rather than in apply_timing_rules.
    #
    # This moves ends and never starts. A start has to land on the speech or the
    # subtitle reads as out of sync; an end does not, so holding a line until
    # the next one appears costs nothing a viewer can see.
    spans = [_CueSpan(cue) for cue in processed.cues if (cue.final_text or "").strip()]
    before = [span.end_ms for span in spans]
    _close_screen_gaps(spans, gap_floor, slot_lookup=slot_lookup)
    extended = 0
    for span, was in zip(spans, before):
        if span.end_ms != was:
            span.cue.end_time = span.end_ms / 1000.0
            extended += 1
    if extended:
        modified = True
        logger.info("Extended %d line(s) to close sub-%dms gaps.", extended, gap_floor)

    # Whatever the repairs could not settle is a regression upstream, so say so
    # rather than leaving it silent.
    violations = check_display_invariants(
        processed.cues,
        speaker_map=speaker_map,
        min_duration_ms=min_duration_ms,
        min_gap_ms=gap_floor,
    )
    if violations:
        logger.warning(
            "%d display invariant violation(s) remain in the finished document:",
            len(violations),
        )
        for violation in violations[:_MAX_LOGGED_VIOLATIONS]:
            logger.warning("  %s", violation)
        if len(violations) > _MAX_LOGGED_VIOLATIONS:
            logger.warning(
                "  ... and %d more.", len(violations) - _MAX_LOGGED_VIOLATIONS
            )

    logger.info(f"Writing postprocessed JSON to {output_json_path}...")
    output_json_path.write_text(processed.model_dump_json(indent=2), encoding="utf-8")

    logger.info(f"Writing postprocessed subtitles to {output_ass_path}...")
    render_ass_document(
        processed,
        output_ass_path,
        mode="bilingual" if bilingual else "final",
        speaker_map=speaker_map,
    )
    return modified


def _apply_radio_discourse_postprocess(document: SubtitleDocument) -> bool:
    modified = False
    for cue in document.cues:
        if cue.role != "listener_mail":
            continue
        if not cue.final_text:
            continue

        quoted = _ensure_quoted(cue.final_text)
        if quoted != cue.final_text:
            cue.final_text = quoted
            modified = True
    return modified


def _ensure_quoted(text: str) -> str:
    if _is_wrapped_in_quotes(text):
        return _normalize_quote_edges(text)
    return _normalize_quote_edges(f'"{text}"')


def _is_wrapped_in_quotes(text: str) -> bool:
    stripped = text.strip()
    return (
        len(stripped) >= 2
        and stripped[0] in QUOTE_CHARS
        and stripped[-1] in QUOTE_CHARS
    )


def _collapse_outer_duplicate_quotes(text: str) -> str:
    leading_whitespace_len = len(text) - len(text.lstrip())
    trailing_whitespace_len = len(text) - len(text.rstrip())
    trailing_start = (
        len(text) - trailing_whitespace_len if trailing_whitespace_len else len(text)
    )

    prefix = text[:leading_whitespace_len]
    core = text[leading_whitespace_len:trailing_start]
    suffix = text[trailing_start:]

    while (
        len(core) >= 4
        and core[0] in QUOTE_CHARS
        and core[1] in QUOTE_CHARS
        and core[-1] in QUOTE_CHARS
        and core[-2] in QUOTE_CHARS
    ):
        core = core[1:-1]

    return f"{prefix}{core}{suffix}"


def _normalize_quote_edges(text: str) -> str:
    collapsed = _collapse_outer_duplicate_quotes(text)
    return _collapse_duplicate_visual_line_quotes(collapsed)


def _collapse_duplicate_visual_line_quotes(text: str) -> str:
    parts = LINE_BREAK_RE.split(text)
    if not parts:
        return text

    # Only collapse the display edges of a multi-line cue; duplicated quotes next
    # to interior line breaks are preserved because they belong to inner content.
    parts[0] = _collapse_leading_quote_run(parts[0])
    parts[-1] = _collapse_trailing_quote_run(parts[-1])
    return "".join(parts)


def _collapse_leading_quote_run(text: str) -> str:
    leading_whitespace_len = len(text) - len(text.lstrip())
    prefix = text[:leading_whitespace_len]
    core = text[leading_whitespace_len:]

    while len(core) >= 2 and core[0] in QUOTE_CHARS and core[1] in QUOTE_CHARS:
        core = core[1:]

    return f"{prefix}{core}"


def _collapse_trailing_quote_run(text: str) -> str:
    trailing_whitespace_len = len(text) - len(text.rstrip())
    suffix = text[-trailing_whitespace_len:] if trailing_whitespace_len else ""
    core = (
        text[: len(text) - trailing_whitespace_len] if trailing_whitespace_len else text
    )

    while len(core) >= 2 and core[-1] in QUOTE_CHARS and core[-2] in QUOTE_CHARS:
        core = core[:-1]

    return f"{core}{suffix}"


_TERMINAL_PUNCT = (".", "?", "!", "…", '"', "”", "—")


def _style_capacities_for_document(
    document: SubtitleDocument,
    *,
    bilingual: bool,
    speaker_map: dict[str, dict] | None,
) -> dict[str, int]:
    script = build_ass_script(
        document,
        mode="bilingual" if bilingual else "final",
        speaker_map=speaker_map,
    )
    play_res_x = None
    for key, value in getattr(script, "scriptInfo", []) or []:
        if key == "PlayResX":
            try:
                play_res_x = int(value)
            except (TypeError, ValueError):
                pass
            break
    return {
        style.name: capacity_for_style(
            play_res_x,
            getattr(style, "marginL", None),
            getattr(style, "marginR", None),
            getattr(style, "fontSize", None),
            getattr(style, "fontName", None),
        )
        for style in getattr(script, "styles", []) or []
    }


def _merge_cue_into(left: SubtitleCue, right: SubtitleCue, wrapped_text: str) -> None:
    separator = " " if left.source_text and right.source_text else ""
    right_orig_offset = _original_text_length(
        left.source_text, left.replacement_spans
    ) + len(separator)
    right_replaced_offset = len(left.source_text) + len(separator)
    left.replacement_spans.extend(
        _offset_replacement_spans(
            right.replacement_spans,
            orig_offset=right_orig_offset,
            replaced_offset=right_replaced_offset,
        )
    )
    left.source_text = f"{left.source_text}{separator}{right.source_text}".strip()
    if left.normalized_source_text or right.normalized_source_text:
        left_norm = left.normalized_source_text or left.source_text
        right_norm = right.normalized_source_text or right.source_text
        left.normalized_source_text = f"{left_norm}{separator}{right_norm}".strip()
    if separator and left.words:
        left.words[-1] = left.words[-1].model_copy(
            update={"word": left.words[-1].word + separator}
        )
    left.words.extend(right.words)
    left.end_time = max(left.end_time, right.end_time)
    left.translated_text = wrapped_text
    left.final_text = wrapped_text


def _merge_same_slot_flash_cues(
    document: SubtitleDocument,
    *,
    speaker_map: dict[str, dict] | None,
    min_duration_ms: int,
    min_gap_ms: int,
    bilingual: bool,
) -> int:
    """Merge consecutive same-slot cues when keeping them split creates a flash.

    Catches three patterns that survive or arise from translation/reflow:
    1. A sentence split across two adjacent same-slot cues where the first cue
       lacks terminal punctuation and the tail cue is under 1.0s.
    2. Two back-to-back minimum-duration cues (both <= min_duration_ms + 50ms)
       in the same slot that flash two 1-2 word phrases in rapid succession.
    3. A minimum-duration cue whose underlying transcribed speech was an
       ultra-short <= 150ms single-token fragment padded with silence.
    """
    cues = document.cues
    if len(cues) < 2:
        return 0

    slot_lookup = build_slot_lookup(speaker_map)
    style_names = build_style_name_lookup(speaker_map)
    capacities = _style_capacities_for_document(
        document, bilingual=bilingual, speaker_map=speaker_map
    )
    nlp = load_nlp()
    merged_count = 0

    i = 0
    while i < len(cues) - 1:
        a = cues[i]
        a_text = normalized_text((a.final_text or a.translated_text or "").strip())
        if not a_text:
            i += 1
            continue

        a_slot = slot_key_for_speaker(a.speaker, slot_lookup)
        # Find the immediately next cue in the same slot (only if no other cue in
        # the same slot intervenes and they are adjacent or only separated by a
        # brief concurrent cue in another slot).
        b_idx = None
        for j in range(i + 1, len(cues)):
            cand = cues[j]
            if not (cand.final_text or cand.translated_text or "").strip():
                continue
            if slot_key_for_speaker(cand.speaker, slot_lookup) == a_slot:
                b_idx = j
                break
        if b_idx is None:
            i += 1
            continue

        b = cues[b_idx]
        b_text = normalized_text((b.final_text or b.translated_text or "").strip())
        gap_ms = round((b.start_time - a.end_time) * 1000)
        if (
            gap_ms < 0
            or gap_ms >= min_gap_ms
            or not _roles_compatible_for_merge(a.role, b.role)
            or b.corner
        ):
            i += 1
            continue

        a_dur_ms = round((a.end_time - a.start_time) * 1000)
        b_dur_ms = round((b.end_time - b.start_time) * 1000)
        b_raw_ms = (
            round((b.words[-1].end_time - b.words[0].start_time) * 1000)
            if b.words
            else b_dur_ms
        )

        unclosed_sentence_tail = (
            not a_text.endswith(_TERMINAL_PUNCT) and b_dur_ms < 1000
        )
        double_min_flash = (
            a_dur_ms <= min_duration_ms + 50 and b_dur_ms <= min_duration_ms + 50
        )
        padded_micro_fragment = (
            b_dur_ms <= min_duration_ms + 50 and b_raw_ms <= 150
        )

        if not (unclosed_sentence_tail or double_min_flash or padded_micro_fragment):
            i += 1
            continue

        style_name = style_name_for_speaker(a.speaker, style_names)
        max_chars = capacities.get(style_name, MAX_CHARS_PER_LINE)
        combined = f"{a_text} {b_text}".strip()
        wrapped = wrap_line(combined, nlp=nlp, max_chars=max_chars)
        if wrapped is None and len(a_text) <= max_chars and len(b_text) <= max_chars:
            wrapped = f"{a_text}\\N{b_text}"
        if wrapped is None:
            i += 1
            continue

        _merge_cue_into(a, b, wrapped)
        cues.pop(b_idx)
        if document.chunk_boundaries:
            updated_bounds: list[int] = []
            for bound in document.chunk_boundaries:
                adj = bound - 1 if bound > b_idx else bound
                if not updated_bounds or updated_bounds[-1] != adj:
                    updated_bounds.append(adj)
            document.chunk_boundaries = updated_bounds
        merged_count += 1
        # Stay at index i in case another fragment chains onto the merged cue.

    return merged_count


def _rewrap_overlong_cues(
    document: SubtitleDocument,
    *,
    speaker_map: dict[str, dict] | None,
    bilingual: bool,
) -> int:
    """Re-wrap cues whose existing lines exceed their current slot style capacity."""
    if not speaker_map:
        return 0
    style_names = build_style_name_lookup(speaker_map)
    capacities = _style_capacities_for_document(
        document, bilingual=bilingual, speaker_map=speaker_map
    )
    nlp = load_nlp()
    rewrapped = 0
    for cue in document.cues:
        text = (cue.final_text or cue.translated_text or "").strip()
        if not text:
            continue
        style_name = style_name_for_speaker(cue.speaker, style_names)
        max_chars = capacities.get(style_name, MAX_CHARS_PER_LINE)
        parts = text.split("\\N")
        if any(len(part) > max_chars for part in parts):
            wrapped = wrap_line(text, nlp=nlp, max_chars=max_chars)
            if wrapped is not None and wrapped != text:
                cue.final_text = wrapped
                cue.translated_text = wrapped
                rewrapped += 1
    return rewrapped


def _repair_cross_slot_turn_overlaps(
    cues: list[SubtitleCue],
    *,
    speaker_map: dict[str, dict] | None,
    min_duration_ms: int,
) -> int:
    """Repair artificial cross-slot overlaps and hold primary speakers across backchannels.

    Uses word-level timestamps (`cue.words`) to distinguish genuine overlapping
    speech from 1-5 frame artificial overlaps manufactured by symmetric padding
    or per-slot midpoint gap snapping.
    """
    if len(cues) < 2:
        return 0

    slot_lookup = build_slot_lookup(speaker_map)
    active = [c for c in cues if (c.final_text or c.translated_text or "").strip()]
    unique_slots = {
        slot_key_for_speaker(c.speaker, slot_lookup) for c in active if c.speaker
    }
    if len(unique_slots) <= 1 or not any(c.words for c in active):
        return 0

    def s_key(cue: SubtitleCue) -> str | None:
        return slot_key_for_speaker(cue.speaker, slot_lookup)

    def raw_bounds_ms(cue: SubtitleCue) -> tuple[int, int]:
        if cue.words:
            return (
                round(cue.words[0].start_time * 1000),
                round(cue.words[-1].end_time * 1000),
            )
        return (round(cue.start_time * 1000), round(cue.end_time * 1000))

    slot_groups: dict[str | None, list[SubtitleCue]] = {}
    for cue in active:
        slot_groups.setdefault(s_key(cue), []).append(cue)
    prev_in_slot: dict[str, SubtitleCue] = {}
    next_in_slot: dict[str, SubtitleCue] = {}
    for grp in slot_groups.values():
        grp.sort(key=lambda c: (c.start_time, c.end_time))
        for idx, cue in enumerate(grp):
            if idx > 0:
                prev_in_slot[cue.id] = grp[idx - 1]
            if idx + 1 < len(grp):
                next_in_slot[cue.id] = grp[idx + 1]

    adjusted = 0
    held_pairs: set[tuple[str, str]] = set()

    # 1. Bridge [A1] [B_short] [A2] cross-slot backchannels
    for i in range(len(active) - 2):
        a1, b, a2 = active[i], active[i + 1], active[i + 2]
        if (
            s_key(a1) is not None
            and s_key(a1) == s_key(a2)
            and s_key(a1) != s_key(b)
            and next_in_slot.get(a1.id) is a2
        ):
            a1_raw_s, a1_raw_e = raw_bounds_ms(a1)
            b_raw_s, b_raw_e = raw_bounds_ms(b)
            a2_raw_s, a2_raw_e = raw_bounds_ms(a2)
            b_raw_dur = b_raw_e - b_raw_s
            raw_gap_a = a2_raw_s - a1_raw_e

            b_text = (b.final_text or b.translated_text or b.source_text or "").strip()
            a1_text = (a1.final_text or a1.translated_text or a1.source_text or "").strip()
            b_is_long_question = b_text.endswith(("?", "？")) and len(b_text) > 16
            a1_unfinished = a1_text.endswith((",", "...", "…", "--", "—", "、", "けど", "つつ", "から", "て", "で"))
            max_b_dur = 1200 if not b_is_long_question else 400
            max_gap_a = 1800 if a1_unfinished else 1350

            if (
                b_raw_dur <= max_b_dur
                and -250 <= raw_gap_a <= max_gap_a
                and b_raw_s >= a1_raw_s
            ):
                changed = False
                a2_start_ms = round(a2.start_time * 1000)
                a2_end_ms = round(a2.end_time * 1000)
                if a2_start_ms < a2_raw_s and a2_end_ms - a2_raw_s >= min_duration_ms:
                    a2.start_time = a2_raw_s / 1000.0
                    a2_start_ms = a2_raw_s
                    changed = True

                a1_end_ms = round(a1.end_time * 1000)
                if a1_end_ms < a2_start_ms:
                    a1.end_time = a2_start_ms / 1000.0
                    changed = True

                b_start_ms = round(b.start_time * 1000)
                b_end_ms = round(b.end_time * 1000)
                if b_start_ms < a1_raw_e:
                    b_prev = prev_in_slot.get(b.id)
                    floor_ms = max(
                        a1_raw_e,
                        round(b_prev.end_time * 1000) if b_prev else 0,
                    )
                    target_b_start = max(
                        floor_ms, min(b_raw_s, a2_start_ms - min_duration_ms)
                    )
                    b_next = next_in_slot.get(b.id)
                    ceil_ms = (
                        round(b_next.start_time * 1000)
                        if b_next
                        else target_b_start + min_duration_ms
                    )
                    target_b_end = min(
                        ceil_ms,
                        max(target_b_start + min_duration_ms, min(b_end_ms, a2_start_ms)),
                    )
                    if target_b_end - target_b_start >= min_duration_ms:
                        b.start_time = target_b_start / 1000.0
                        b.end_time = target_b_end / 1000.0
                        b_start_ms = target_b_start
                        b_end_ms = target_b_end
                        changed = True

                # If B's post-speech hang spills past A2's start (or stops within
                # min_duration_ms before A2's start), snap B's end to A2's start so
                # A1 and B transition in sync with A2 rather than staggering.
                if (
                    b_end_ms > b_raw_e
                    and b_end_ms > a2_start_ms
                    and a2_start_ms - b_start_ms >= min_duration_ms
                    and b_raw_e <= a2_start_ms + 150
                ):
                    b.end_time = a2_start_ms / 1000.0
                    changed = True

                held_pairs.add((a1.id, b.id))
                held_pairs.add((b.id, a2.id))
                if changed:
                    adjusted += 1

    # 2. Eliminate artificial padding overlaps on sequential non-overlapping turns
    for i in range(len(active) - 1):
        prev_cue = active[i]
        curr_cue = active[i + 1]
        if s_key(prev_cue) == s_key(curr_cue):
            continue
        if (prev_cue.id, curr_cue.id) in held_pairs:
            continue

        prev_raw_s, prev_raw_e = raw_bounds_ms(prev_cue)
        curr_raw_s, curr_raw_e = raw_bounds_ms(curr_cue)
        if curr_raw_s < prev_raw_e - 150:
            # Genuine overlapping speech in the audio; preserve the overlap.
            continue

        prev_start_ms = round(prev_cue.start_time * 1000)
        prev_end_ms = round(prev_cue.end_time * 1000)
        curr_start_ms = round(curr_cue.start_time * 1000)
        curr_end_ms = round(curr_cue.end_time * 1000)
        if prev_end_ms <= curr_start_ms:
            continue

        changed = False
        # Clamp prev_cue's post-speech padding so its stop aligns with curr_cue's start
        if prev_end_ms > curr_start_ms and curr_start_ms - prev_start_ms >= min_duration_ms:
            prev_cue.end_time = curr_start_ms / 1000.0
            prev_end_ms = curr_start_ms
            changed = True
        elif prev_end_ms > prev_raw_e and prev_end_ms > curr_raw_s:
            new_prev_end = max(prev_raw_e, curr_raw_s)
            prev_cue.end_time = new_prev_end / 1000.0
            prev_end_ms = new_prev_end
            if prev_end_ms - prev_start_ms < min_duration_ms:
                p_prev = prev_in_slot.get(prev_cue.id)
                left_bound = round(p_prev.end_time * 1000) if p_prev else 0
                prev_start_ms = max(left_bound, prev_end_ms - min_duration_ms)
                prev_cue.start_time = prev_start_ms / 1000.0
                if prev_end_ms - prev_start_ms < min_duration_ms:
                    prev_end_ms = prev_start_ms + min_duration_ms
                    prev_cue.end_time = prev_end_ms / 1000.0
            changed = True

        # Shift curr_cue's pre-speech lead-in so it does not overlap prev_cue
        if curr_start_ms < prev_end_ms and curr_start_ms < curr_raw_s:
            new_curr_start = max(prev_end_ms, min(curr_raw_s, prev_end_ms))
            curr_cue.start_time = new_curr_start / 1000.0
            curr_start_ms = new_curr_start
            if curr_end_ms - curr_start_ms < min_duration_ms:
                c_next = next_in_slot.get(curr_cue.id)
                right_bound = (
                    round(c_next.start_time * 1000)
                    if c_next
                    else curr_start_ms + min_duration_ms
                )
                curr_cue.end_time = min(right_bound, curr_start_ms + min_duration_ms) / 1000.0
            changed = True

        if changed:
            adjusted += 1

    return adjusted
