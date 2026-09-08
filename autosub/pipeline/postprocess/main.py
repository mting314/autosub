from __future__ import annotations

import logging
import re
from pathlib import Path

from autosub.core.schemas import SubtitleCue, SubtitleDocument
from autosub.pipeline.format.generator import render_ass_document
from autosub.core.speaker_map import build_slot_lookup
from autosub.pipeline.format.timing import (
    _close_slot_gaps,
    check_display_invariants,
    slot_key_for_speaker,
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
) -> None:
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

    stripped = _strip_corner_markers(processed)
    if stripped:
        logger.info(
            "Removed the [CORNER: ...] prefix from %d line(s); the boundary is "
            "carried by cue.corner and the rendered corner comments.",
            stripped,
        )

    extensions_config = extensions_config or {}
    radio_discourse_config = extensions_config.get("radio_discourse", {})
    if radio_discourse_config.get("enabled"):
        if _apply_radio_discourse_postprocess(processed):
            logger.info("Postprocessing modified subtitle document.")

    # Close gaps too short to read as a pause, within each slot. This only
    # settles once translation has finished splitting and reflowing cues, which
    # is why it lives here rather than in apply_timing_rules.
    #
    # Scoped to one box on purpose. A hand-off between slots leaves no visible
    # hole — one box empties as another fills, and the backdrop bars are there
    # regardless — so closing those would only hold the previous speaker's line
    # past where they stopped talking, at real cost to how well the subtitle
    # tracks the audio.
    gap_floor = min_duration_ms if min_gap_ms is None else min_gap_ms
    slot_lookup = build_slot_lookup(speaker_map)
    by_slot: dict[object, list[_CueSpan]] = {}
    for cue in processed.cues:
        if (cue.final_text or "").strip():
            key = slot_key_for_speaker(cue.speaker, slot_lookup)
            by_slot.setdefault(key, []).append(_CueSpan(cue))

    extended = 0
    for spans in by_slot.values():
        before = [span.end_ms for span in spans]
        _close_slot_gaps(spans, gap_floor)
        for span, was in zip(spans, before):
            if span.end_ms != was:
                span.cue.end_time = span.end_ms / 1000.0
                extended += 1
    if extended:
        logger.info(
            "Extended %d line(s) to close sub-%dms gaps within a slot.",
            extended,
            gap_floor,
        )

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
