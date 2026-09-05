import hashlib
import json
import logging
import re
import traceback
from pathlib import Path

import pyass

from autosub.core.config import PROJECT_ID
from autosub.core.llm import ReasoningEffort
from autosub.core.schemas import SubtitleCue, SubtitleDocument
from autosub.core.speaker_map import build_style_name_lookup, style_name_for_speaker
from autosub.pipeline.format.generator import (
    build_ass_script,
    render_ass_document,
)
from autosub.pipeline.translate.chunker import make_chunks
from autosub.pipeline.translate.linebreak import (
    MAX_CHARS_PER_LINE,
    capacity_for_style,
    load_nlp,
    normalized_text,
    split_text,
    strip_tags,
    visible_length,
    wrap_line,
)

logger = logging.getLogger(__name__)

# Guard against pathological recursion when a line keeps failing to fit.
_MAX_SPLIT_DEPTH = 4


def _compute_cue_fingerprint(
    cues: list[SubtitleCue], chunk_size: int, corner_boundaries: list[int] | None
) -> str:
    """Hash translation inputs and cue metadata to detect stale checkpoints."""
    h = hashlib.sha256()
    h.update(str(chunk_size).encode())
    h.update(b"\x00")
    for b in corner_boundaries or []:
        h.update(str(b).encode())
        h.update(b"\x00")
    h.update(b"\x01")
    for cue in cues:
        source_text = cue.normalized_source_text or cue.source_text
        payload = {
            "text": source_text,
            "start_time": cue.start_time,
            "end_time": cue.end_time,
            "speaker": cue.speaker,
            "role": cue.role,
            "corner": cue.corner,
        }
        h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()


def translate_subtitles(
    input_json_path: Path,
    output_json_path: Path,
    output_ass_path: Path | None = None,
    engine: str = "vertex",
    system_prompt: str | None = None,
    target_lang: str = "en",
    source_lang: str = "ja",
    bilingual: bool = True,
    model: str | None = None,
    location: str = "global",
    provider: str = "google-vertex",
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM,
    reasoning_budget_tokens: int | None = None,
    reasoning_dynamic: bool | None = None,
    chunk_size: int = 0,
    debug: bool = False,
    retry_chunks: list[int] | None = None,
    log_dir: Path | None = None,
    reflow: bool = True,
    reflow_engine: str = "deterministic",
    reflow_model: str | None = None,
    speaker_map: dict[str, dict] | None = None,
    min_duration_ms: int = 500,
) -> None:
    """
    Reads a formatted subtitle JSON document, translates cue source text, and
    writes a translated JSON document plus a rendered ASS byproduct.

    Corner boundaries are read directly from structured cue metadata.
    """
    if output_ass_path is None:
        output_ass_path = output_json_path.with_suffix(".ass")

    logger.info(f"Loading '{input_json_path}' for translation...")
    document = SubtitleDocument.model_validate_json(
        input_json_path.read_text(encoding="utf-8")
    )
    if document.stage != "formatted":
        raise ValueError(f"translate expects stage='formatted', got {document.stage!r}")

    cues_to_translate = []
    texts_to_translate = []
    for cue in document.cues:
        source_text = cue.normalized_source_text or cue.source_text
        if source_text.strip():
            cues_to_translate.append(cue)
            texts_to_translate.append(source_text)

    if not texts_to_translate:
        logger.warning("No subtitle text found to translate. Exiting.")
        translated_document = document.model_copy(deep=True)
        translated_document.stage = "translated"
        translated_document.chunk_boundaries = []
        output_json_path.write_text(
            translated_document.model_dump_json(indent=2), encoding="utf-8"
        )
        render_ass_document(
            translated_document,
            output_ass_path,
            mode="bilingual" if bilingual else "translated",
            speaker_map=speaker_map,
        )
        return

    llm_trace_path: Path | None = None

    if engine == "vertex":
        from autosub.pipeline.translate.translator import VertexTranslator

        if provider in {"google-vertex", "anthropic-vertex"} and not PROJECT_ID:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in the environment.")

        llm_trace_path = output_ass_path.with_suffix(".llm_trace.jsonl")
        if llm_trace_path.exists():
            llm_trace_path.unlink()
            logger.info("Removed previous LLM trace file.")

        translator = VertexTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
            model=model,
            location=location,
            provider=provider,
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
            reasoning_dynamic=reasoning_dynamic,
            trace_path=llm_trace_path,
        )
    elif engine == "cloud-v3":
        from autosub.pipeline.translate.api import CloudTranslationTranslator

        if not PROJECT_ID:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in the environment.")

        translator = CloudTranslationTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unknown translation engine: {engine}")

    checkpoint_path = output_json_path.with_suffix(".checkpoint.json")
    error_path = output_json_path.with_suffix(".error.txt")

    if error_path.exists():
        error_path.unlink()
        logger.info("Removed previous translation error file.")

    corner_boundaries = _extract_corner_boundaries_from_cues(document)
    if corner_boundaries:
        logger.info(
            f"Found {len(corner_boundaries)} corner boundaries at dialogue indices {corner_boundaries}"
        )

    splits: set[int] = set()
    try:
        if chunk_size > 0:
            translated_texts, splits = _translate_chunked(
                translator,
                cues_to_translate,
                chunk_size,
                checkpoint_path,
                corner_boundaries=corner_boundaries or None,
                retry_chunks=retry_chunks,
                log_dir=log_dir,
            )
        else:
            translated_texts = translator.translate_cues(cues_to_translate)
    except Exception as exc:
        _write_error_report(error_path, exc)
        logger.error(f"Wrote translation error details to {error_path}.")
        raise

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file.")

    if len(translated_texts) != len(cues_to_translate):
        raise ValueError(
            f"Translation API expected {len(cues_to_translate)} translations, but got {len(translated_texts)}"
        )

    if reflow:
        translated_texts = _reflow_translations(
            translated_texts,
            cues_to_translate,
            corner_boundaries,
            engine=reflow_engine,
            provider=provider,
            location=location,
            model=reflow_model,
        )

    logger.info("Applying translations to subtitle document...")
    translated_document = document.model_copy(deep=True)
    translated_document.stage = "translated"
    cue_by_id = {cue.id: cue for cue in translated_document.cues}
    for source_cue, translated_text in zip(
        cues_to_translate, translated_texts, strict=True
    ):
        cue_by_id[source_cue.id].translated_text = translated_text

    # Lay the translation out inside each speaker's box before serialising, so the
    # document carries the wraps and splits rather than only the .ass. A script is
    # built first purely for its per-style capacities, which do not exist until the
    # styles have been generated.
    # splits index into cues_to_translate, which has the empty cues filtered out,
    # so record the boundaries by cue id and resolve them to indices afterwards.
    boundary_cue_ids = (
        {cues_to_translate[split].id for split in splits} if debug else set()
    )
    capacities = _line_capacities(
        build_ass_script(
            translated_document,
            mode="bilingual" if bilingual else "translated",
            speaker_map=speaker_map,
        )
    )
    cue_count = len(translated_document.cues)
    translated_document.cues = _apply_line_breaks_to_cues(
        translated_document.cues,
        capacities,
        style_names=build_style_name_lookup(speaker_map),
        min_duration_ms=min_duration_ms,
    )
    if len(translated_document.cues) != cue_count:
        logger.info(
            "Line breaking split %d cue(s) that could not fit two lines.",
            len(translated_document.cues) - cue_count,
        )
    translated_document.chunk_boundaries = _chunk_boundary_indices(
        translated_document.cues, boundary_cue_ids
    )

    logger.info(f"Writing translated JSON to {output_json_path}...")
    output_json_path.write_text(
        translated_document.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info(f"Writing translated .ass file to {output_ass_path}...")
    render_ass_document(
        translated_document,
        output_ass_path,
        mode="bilingual" if bilingual else "translated",
        speaker_map=speaker_map,
    )

    if llm_trace_path is not None and llm_trace_path.exists():
        logger.info(f"Wrote LLM trace to {llm_trace_path}.")

    logger.info("Translation complete!")


def _reflow_translations(
    translated_texts: list[str],
    cues_to_translate: list[SubtitleCue],
    corner_boundaries: list[int] | None,
    engine: str = "deterministic",
    provider: str = "google-vertex",
    location: str = "global",
    model: str | None = None,
) -> list[str]:
    """Re-split translated lines at natural English boundaries.

    Derives per-line display durations and hard group-break indices (speaker
    change, corner boundary, long time gap) from the cues, then delegates to
    the reflow. The ``llm`` engine uses a cheap model to choose break points
    (falling back to the deterministic engine per group); ``deterministic`` (the
    default) needs no API calls. Any failure is non-fatal: the original
    translations are returned unchanged.
    """
    from autosub.pipeline.translate.reflow import LONG_GAP_S, reflow_line_breaks

    try:
        durations_s: list[float] = []
        boundaries: set[int] = set(corner_boundaries or [])
        for i, cue in enumerate(cues_to_translate):
            durations_s.append(max(0.0, cue.end_time - cue.start_time))
            if i == 0:
                continue
            prev = cues_to_translate[i - 1]
            if cue.speaker != prev.speaker:
                boundaries.add(i)
            elif cue.start_time - prev.end_time > LONG_GAP_S:
                boundaries.add(i)

        resplitter = None
        if engine == "llm":
            from autosub.pipeline.translate.reflow_llm import build_llm_resplitter

            logger.info("Using LLM line-break reflow engine.")
            resplitter = build_llm_resplitter(
                project_id=PROJECT_ID,
                model=model,  # None -> splitter's cheap flash-lite default
                location=location,
                provider=provider,
            )
        return reflow_line_breaks(
            translated_texts, durations_s, boundaries, resplitter=resplitter
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Line-break reflow skipped due to error: {exc}")
        return translated_texts


def _extract_corner_boundaries_from_cues(document: SubtitleDocument) -> list[int]:
    boundaries: list[int] = []
    dialogue_idx = 0
    # A corner on an empty cue attaches to the next translatable cue, matching
    # the old behavior where corner Comments preceded their dialogue event.
    pending_corner_cue_id: str | None = None
    for cue in document.cues:
        source_text = cue.normalized_source_text or cue.source_text
        if not source_text.strip():
            if cue.corner:
                pending_corner_cue_id = cue.id
            continue
        if cue.corner or pending_corner_cue_id is not None:
            boundaries.append(dialogue_idx)
            pending_corner_cue_id = None
        dialogue_idx += 1
    if pending_corner_cue_id is not None:
        logger.warning(
            "Dropping corner boundary on empty cue %s: no later dialogue cue to attach it to.",
            pending_corner_cue_id,
        )
    return boundaries


def _write_error_report(error_path: Path, exc: Exception) -> None:
    error_path.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )


def _load_checkpoint(checkpoint_path: Path, fingerprint: str) -> dict[int, list[str]]:
    """Load and validate completed chunk results from checkpoint file.

    Returns dict[int, list[str]] mapping chunk index to translated strings.
    JSON serializes int keys as strings, so they are converted back on load.
    Invalid entries are skipped with a warning.
    Discards the checkpoint if the fingerprint doesn't match (input changed).
    """
    if not checkpoint_path.exists():
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint, starting fresh: {e}")
        return {}

    if not isinstance(data, dict):
        logger.warning("Checkpoint is not a JSON object, starting fresh.")
        return {}

    # Validate fingerprint
    if "_fingerprint" not in data:
        logger.warning("Legacy checkpoint without fingerprint, discarding.")
        return {}
    if data["_fingerprint"] != fingerprint:
        logger.warning(
            "Checkpoint fingerprint mismatch (input or chunking config changed), "
            "discarding stale checkpoint."
        )
        return {}

    chunks_data = data.get("chunks", {})
    if not isinstance(chunks_data, dict):
        logger.warning("Checkpoint 'chunks' is not a dict, starting fresh.")
        return {}

    validated: dict[int, list[str]] = {}
    for k, v in chunks_data.items():
        try:
            chunk_idx = int(k)
        except (ValueError, TypeError):
            logger.warning(f"Skipping checkpoint entry with non-integer key: {k!r}")
            continue

        if chunk_idx < 0:
            logger.warning(f"Skipping checkpoint entry with negative key: {chunk_idx}")
            continue

        if not isinstance(v, list) or not v:
            logger.warning(
                f"Skipping checkpoint entry {chunk_idx}: value must be a non-empty list."
            )
            continue

        if not all(isinstance(s, str) for s in v):
            logger.warning(
                f"Skipping checkpoint entry {chunk_idx}: list contains non-string elements."
            )
            continue

        validated[chunk_idx] = v

    return validated


def _save_checkpoint(
    checkpoint_path: Path, completed: dict[int, list[str]], fingerprint: str
) -> None:
    """Save completed chunk results to checkpoint file."""
    payload = {"_fingerprint": fingerprint, "chunks": completed}
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _translate_chunked(
    translator,
    cues: list[SubtitleCue],
    chunk_size: int,
    checkpoint_path: Path,
    corner_boundaries: list[int] | None = None,
    retry_chunks: list[int] | None = None,
    log_dir: Path | None = None,
) -> tuple[list[str], set[int]]:
    """Split texts into chunks, translate each once, and merge results."""
    texts = [cue.normalized_source_text or cue.source_text for cue in cues]
    chunks, splits = make_chunks(texts, chunk_size, corner_boundaries=corner_boundaries)
    fingerprint = _compute_cue_fingerprint(cues, chunk_size, corner_boundaries)

    # Set up structured log directory
    chunks_dir = None
    token_summary_path = None
    system_prompt_path = None
    if log_dir:
        chunks_dir = log_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        token_summary_path = log_dir / "token_summary.tsv"
        system_prompt_path = log_dir / "system_prompt.txt"
        # Write header if new file
        if not token_summary_path.exists():
            token_summary_path.write_text(
                "chunk\tlines\tprompt\tcandidates\tthoughts\ttotal\n",
                encoding="utf-8",
            )
    completed = _load_checkpoint(checkpoint_path, fingerprint)

    # Remove specified chunks from checkpoint to force re-translation
    if retry_chunks and completed:
        for idx in retry_chunks:
            chunk_num = idx - 1  # user-facing is 1-based
            if chunk_num in completed:
                del completed[chunk_num]
                logger.info(f"Cleared checkpoint for chunk {idx} — will re-translate.")
            else:
                logger.warning(f"Chunk {idx} not in checkpoint — nothing to retry.")
        _save_checkpoint(checkpoint_path, completed, fingerprint)

    if completed:
        logger.info(
            f"Resuming from checkpoint: {len(completed)}/{len(chunks)} chunks already completed."
        )

    logger.info(
        f"Translating {len(texts)} subtitle lines "
        f"in {len(chunks)} chunks of up to {chunk_size}..."
    )

    line_offset = 0
    for chunk_idx, chunk in enumerate(chunks):
        cue_chunk = cues[line_offset : line_offset + len(chunk)]
        if len(cue_chunk) != len(chunk):
            raise ValueError("Cue chunking lost alignment with text chunks.")

        line_start = line_offset + 1
        line_end = line_offset + len(chunk)

        if chunk_idx in completed:
            logger.info(f"  Chunk {chunk_idx + 1}/{len(chunks)} — skipped (checkpoint)")
            line_offset += len(chunk)
            continue

        first = chunk[0][:40] + "..." if len(chunk[0]) > 40 else chunk[0]
        last = chunk[-1][:40] + "..." if len(chunk[-1]) > 40 else chunk[-1]
        logger.info(
            f"  Chunk {chunk_idx + 1}/{len(chunks)} "
            f"(lines {line_start}-{line_end}, {len(chunk)} lines)"
        )
        logger.info(f"    first: {first}")
        logger.info(f"    last:  {last}")
        results = translator.translate_cues(cue_chunk)
        completed[chunk_idx] = results
        _save_checkpoint(checkpoint_path, completed, fingerprint)

        # Write structured log files per chunk
        if chunks_dir and hasattr(translator, "last_diagnostics"):
            chunk_num = f"{chunk_idx + 1:02d}"

            # Write system prompt once
            if system_prompt_path and (
                chunk_idx == 0 or not system_prompt_path.exists()
            ):
                if hasattr(translator, "last_system_instruction"):
                    system_prompt_path.write_text(
                        translator.last_system_instruction, encoding="utf-8"
                    )

            if hasattr(translator, "last_input"):
                (chunks_dir / f"chunk_{chunk_num}_input.json").write_text(
                    translator.last_input, encoding="utf-8"
                )
            if hasattr(translator, "last_output"):
                (chunks_dir / f"chunk_{chunk_num}_output.json").write_text(
                    translator.last_output, encoding="utf-8"
                )

            diag = translator.last_diagnostics
            if diag.thinking_text:
                (chunks_dir / f"chunk_{chunk_num}_thinking.txt").write_text(
                    diag.thinking_text, encoding="utf-8"
                )

            if token_summary_path:
                with open(token_summary_path, "a", encoding="utf-8") as tsv:
                    tsv.write(
                        f"{chunk_idx + 1}\t{len(chunk)}\t"
                        f"{diag.prompt_token_count}\t{diag.candidates_token_count}\t"
                        f"{diag.thoughts_token_count}\t{diag.total_token_count}\n"
                    )

        line_offset += len(chunk)

    # Reassemble in order
    all_translated: list[str] = []
    for chunk_idx in range(len(chunks)):
        all_translated.extend(completed[chunk_idx])

    return all_translated, splits


_LEADING_TAGS_RE = re.compile(r"^((?:\{[^}]*\})+)")


def _split_leading_tags(text: str) -> tuple[str, str]:
    """Separate leading ASS override blocks from the visible text.

    The format stage writes each line's slot position as a leading \\pos tag. The
    translator only ever returns prose, so the tags have to be carried across by
    hand or every line falls back to the style default and the slot layout is lost.
    """
    match = _LEADING_TAGS_RE.match(text)
    if not match:
        return "", text
    return match.group(1), text[match.end() :]


def _line_capacities(script: pyass.Script) -> dict[str, int]:
    """Characters that fit on one line, per style, from the script's own layout.

    A positioned line has whatever width its style's margins leave it, which for
    an overlay slot is not the full-width figure Netflix's 42 assumes.
    """
    play_res_x = None
    for key, value in script.scriptInfo:
        if key == "PlayResX":
            try:
                play_res_x = int(value)
            except (TypeError, ValueError):
                pass
            break

    capacities: dict[str, int] = {}
    for style in getattr(script, "styles", []) or []:
        capacities[style.name] = capacity_for_style(
            play_res_x,
            getattr(style, "marginL", None),
            getattr(style, "marginR", None),
            getattr(style, "fontSize", None),
            getattr(style, "fontName", None),
        )
    return capacities


def _chunk_boundary_indices(
    cues: list[SubtitleCue], boundary_cue_ids: set[str]
) -> list[int]:
    """Locate each chunk boundary after line breaking may have inserted cues.

    Boundaries are stored as indices into the cue list, so a split anywhere ahead
    of one shifts it. Match on the originating cue id instead and take the first
    piece it produced. Cue ids are fixed-width, so no id is a prefix of another
    and the child-id check cannot match the wrong cue.
    """
    indices: list[int] = []
    for boundary_id in boundary_cue_ids:
        prefix = f"{boundary_id}-"
        for index, cue in enumerate(cues):
            if cue.id == boundary_id or cue.id.startswith(prefix):
                indices.append(index)
                break
    return sorted(indices)


def _apply_line_breaks_to_cues(
    cues: list[SubtitleCue],
    capacities: dict[str, int] | None = None,
    style_names: dict[str, str] | None = None,
    min_duration_ms: int = 0,
) -> list[SubtitleCue]:
    """Lay every cue out within its box, at most two lines.

    A cue that will not fit two lines becomes two consecutive cues, cut at a
    grammatical boundary.

    This runs before the document is serialised, so the JSON carries the layout.
    Doing it to the rendered events instead would strand the result in the
    translated .ass: postprocess re-renders from the JSON and would silently drop
    every wrap and split back out of the file that actually ships.
    """
    nlp = load_nlp()
    capacities = capacities or {}
    processed: list[SubtitleCue] = []

    for cue in cues:
        if not strip_tags(cue.translated_text or "").strip():
            # Nothing to lay out. Keep the cue so document indices stay meaningful;
            # the renderer already leaves textless cues out of the .ass.
            processed.append(cue)
            continue
        style = style_name_for_speaker(cue.speaker, style_names)
        max_chars = capacities.get(style, MAX_CHARS_PER_LINE)
        processed.extend(_lay_out_cue(cue, nlp, max_chars, min_duration_ms))

    return processed


def _lay_out_cue(
    cue: SubtitleCue,
    nlp,
    max_chars: int,
    min_duration_ms: int = 0,
    depth: int = 0,
) -> list[SubtitleCue]:
    """Fit one cue into two lines, splitting it into two cues if it will not."""
    tags, body = _split_leading_tags(cue.translated_text or "")

    wrapped = wrap_line(body, nlp, max_chars)
    if wrapped is not None:
        return [cue.model_copy(update={"translated_text": tags + wrapped})]

    def leave_over_length(reason: str) -> list[SubtitleCue]:
        # Leaving the line long is better than breaking it badly; the QC pass
        # flags it for rewording.
        normalized = normalized_text(body)
        logger.warning("%s: %r", reason, strip_tags(normalized)[:60])
        return [cue.model_copy(update={"translated_text": tags + normalized})]

    parts = split_text(body, nlp, max_chars) if depth < _MAX_SPLIT_DEPTH else None
    if parts is None:
        return leave_over_length("No safe line break, leaving it over length")

    part1, part2 = parts

    # Divide the display time in proportion to how much text each piece carries.
    # The QC pass can snap the cut to a real silence later; the transcript is not
    # available here. Never let the cut leave the cue's own span, or a piece would
    # end before it starts and libass would drop it.
    total_ms = round((cue.end_time - cue.start_time) * 1000)
    if total_ms <= 0:
        return [cue]

    seen1 = visible_length(part1)
    share = seen1 / max(1, seen1 + visible_length(part2))
    mid_ms = min(max(round(total_ms * share), 0), total_ms)

    # A split that puts either half on screen for less than the minimum duration
    # trades one over-long line for two flashes, which is the worse of the two.
    if min_duration_ms and min(mid_ms, total_ms - mid_ms) < min_duration_ms:
        return leave_over_length(
            f"Splitting would leave a line under {min_duration_ms}ms, "
            "leaving it over length"
        )

    first, second = _split_cue(
        cue, cue.start_time + mid_ms / 1000.0, tags + part1, tags + part2
    )
    return _lay_out_cue(first, nlp, max_chars, min_duration_ms, depth + 1) + _lay_out_cue(
        second, nlp, max_chars, min_duration_ms, depth + 1
    )


def _split_cue(
    cue: SubtitleCue, mid: float, first_text: str, second_text: str
) -> tuple[SubtitleCue, SubtitleCue]:
    """Cut one cue in two at mid, giving each half its share of the translation.

    Only the translation was split. The source stays whole on the first half
    rather than being cut at a guessed offset, so a bilingual render shows the
    Japanese once, over the opening half of the line it belongs to.

    Child ids extend the parent's, which keeps an unsplit cue's id identical to
    the one the format stage assigned and makes a split traceable back to it.
    """
    first = cue.model_copy(
        update={
            "id": f"{cue.id}-1",
            "end_time": mid,
            "translated_text": first_text,
            "words": [word for word in cue.words if word.end_time <= mid],
        }
    )
    second = cue.model_copy(
        update={
            "id": f"{cue.id}-2",
            "start_time": mid,
            "translated_text": second_text,
            "source_text": "",
            "normalized_source_text": None,
            "replacement_spans": [],
            "words": [word for word in cue.words if word.end_time > mid],
        }
    )
    return first, second
