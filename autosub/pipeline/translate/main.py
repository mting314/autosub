import json
import logging
import re
import time
from pathlib import Path

import pyass
from autosub.core.config import PROJECT_ID
from autosub.pipeline.translate.chunker import make_chunks

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds


def translate_subtitles(
    input_ass_path: Path,
    output_ass_path: Path,
    engine: str = "vertex",
    system_prompt: str | None = None,
    target_lang: str = "en",
    source_lang: str = "ja",
    bilingual: bool = True,
    model: str = "gemini-3-flash-preview",
    location: str = "global",
    chunk_size: int = 0,
    corner_names: list[str] | None = None,
    corner_cues: list[str] | None = None,
    debug: bool = False,
    retry_chunks: list[int] | None = None,
    log_dir: Path | None = None,
) -> None:
    """
    Reads an original .ass file, translates the dialogue events, and outputs a new .ass file.

    Args:
        input_ass_path: Path to the original (e.g., Japanese) .ass file.
        output_ass_path: Path to save the translated .ass file.
        engine: Translation engine ('vertex' or 'cloud-v3').
        system_prompt: Optional instructions for the LLM.
        target_lang: Language code to translate to.
        source_lang: Original language code.
        bilingual: If True, keep the original text above the translated text. If False, replace completely.
        model: Vertex model name for LLM translation.
        location: Vertex region for LLM translation.
        chunk_size: If > 0, split into chunks of this size with retry logic.
        corner_names: Valid corner names from the profile. If set, only these are accepted as markers.
        corner_cues: Corner cue phrases from the profile for intelligent chunk boundary detection.
    """
    logger.info(f"Loading '{input_ass_path}' for translation...")

    with open(input_ass_path, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    # Extract only the dialogue text
    # We maintain a reference list to easily write the translations back to the correct events
    events_to_translate = []
    texts_to_translate = []

    for event in script.events:
        # Skip Comment events so they aren't sent to the LLM
        if isinstance(event, pyass.Event) and event.format == pyass.EventFormat.COMMENT:
            continue
        if isinstance(event, pyass.Event) and event.text:
            # We don't want to translate raw .ass tags.
            # In a robust implementation, we'd strip {\\tags} before translating.
            # Pyass has an event.text property which returns the raw text. Let's grab the raw string representation of parts.
            raw_text = event.text

            if raw_text.strip():
                events_to_translate.append(event)
                texts_to_translate.append(raw_text)

    if not texts_to_translate:
        logger.warning("No subtitle text found to translate. Exiting.")
        return

    if not PROJECT_ID:
        raise ValueError("AUTOSUB_PROJECT_ID is not set in the environment.")

    if engine == "vertex":
        from autosub.pipeline.translate.translator import VertexTranslator

        translator = VertexTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
            model=model,
            location=location,
        )
    elif engine == "cloud-v3":
        from autosub.pipeline.translate.api import CloudTranslationTranslator

        translator = CloudTranslationTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unknown translation engine: {engine}")

    checkpoint_path = output_ass_path.with_suffix(".checkpoint.json")

    splits: set[int] = set()
    if chunk_size > 0:
        translated_texts, splits = _translate_chunked(
            translator, texts_to_translate, chunk_size, checkpoint_path,
            corner_cues=corner_cues,
            retry_chunks=retry_chunks,
            log_dir=log_dir,
        )
    else:
        translated_texts = _translate_with_retry(translator, texts_to_translate)

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file.")

    if len(translated_texts) != len(events_to_translate):
        raise ValueError(
            f"Translation API expected {len(events_to_translate)} translations, but got {len(translated_texts)}"
        )

    logger.info("Applying translations to subtitle events...")

    # Parse corner markers and build the final event list
    corner_marker_re = re.compile(r"^\[CORNER:\s*(.+?)\]\s*")
    new_events: list[pyass.Event] = []
    translated_event_set = set(id(e) for e in events_to_translate)

    # Walk all events in order, preserving non-translated events (e.g. Comments)
    # in place while applying translations and inserting corner markers.
    valid_corners = set(corner_names) if corner_names else None
    corners_found: list[str] = []
    last_corner: str | None = None
    event_idx = 0
    for event in script.events:
        if id(event) not in translated_event_set:
            # Non-dialogue event (Comment, etc.) — keep as-is
            new_events.append(event)
            continue

        # Match this event to its translation by index
        original_text = texts_to_translate[event_idx]
        translated_text = translated_texts[event_idx]

        # Insert debug comment at artificial chunk boundaries
        if debug and event_idx in splits:
            debug_comment = pyass.Event(
                format=pyass.EventFormat.COMMENT,
                start=event.start,
                end=event.end,
                style=event.style,
                effect="",
                text="[autosub] Chunk boundary — review translation around this line",
            )
            new_events.append(debug_comment)

        event_idx += 1

        # Check for corner marker in translation
        corner_match = corner_marker_re.match(translated_text)
        if corner_match:
            corner_name = corner_match.group(1)
            # Always strip the marker tag from the translated text
            translated_text = translated_text[corner_match.end():]

            # Skip if not a profile-defined corner
            if valid_corners and corner_name not in valid_corners:
                logger.debug(f"  Ignoring unknown corner at line {event_idx}: {corner_name}")
            # Skip consecutive duplicate (e.g. chunk boundary re-detection)
            elif corner_name == last_corner:
                logger.debug(f"  Skipping duplicate corner at line {event_idx}: {corner_name}")
            else:
                comment = pyass.Event(
                    format=pyass.EventFormat.COMMENT,
                    start=event.start,
                    end=event.end,
                    style=event.style,
                    effect="",
                    text=f"=== Corner: {corner_name} ===",
                )
                new_events.append(comment)
                corners_found.append(corner_name)
                last_corner = corner_name
                logger.info(f"  Corner detected at line {event_idx}: {corner_name}")

        # Update the event with the new text
        if bilingual:
            # Create a stacked visual format: Original (small/translucent) on top, Translated (large) on bottom
            # Formatting uses standard ASS override tags
            # Since Pyass parses parts, we assign a new EventText part containing our constructed string.
            # Technically, Pyass will treat override tags as standard text if we don't parse them,
            # but Aegisub/video players handle raw string output perfectly fine.
            event.text = f"{{\\\\fs24\\\\a6}}{original_text}{{\\\\N}}{{\\\\fs48\\\\a2}}{translated_text}"
        else:
            event.text = translated_text

        new_events.append(event)

    script.events = new_events

    if corners_found:
        logger.info(
            f"Inserted {len(corners_found)} corner marker(s): {', '.join(corners_found)}"
        )

    logger.info(f"Writing translated .ass file to {output_ass_path}...")
    with open(output_ass_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)

    logger.info("Translation complete!")


def _translate_with_retry(translator, texts: list[str], label: str = "") -> list[str]:
    """Translate texts with exponential backoff retry."""
    prefix = f"  {label}: " if label else ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return translator.translate(texts)
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"{prefix}Attempt {attempt} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"{prefix}Failed after {MAX_RETRIES} attempts: {e}"
                )
                raise
    return []  # unreachable


def _load_checkpoint(checkpoint_path: Path) -> dict[int, list[str]]:
    """Load and validate completed chunk results from checkpoint file.

    Returns dict[int, list[str]] mapping chunk index to translated strings.
    JSON serializes int keys as strings, so they are converted back on load.
    Invalid entries are skipped with a warning.
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
        logger.warning(f"Checkpoint is not a JSON object, starting fresh.")
        return {}

    validated: dict[int, list[str]] = {}
    for k, v in data.items():
        try:
            chunk_idx = int(k)
        except (ValueError, TypeError):
            logger.warning(f"Skipping checkpoint entry with non-integer key: {k!r}")
            continue

        if chunk_idx < 0:
            logger.warning(f"Skipping checkpoint entry with negative key: {chunk_idx}")
            continue

        if not isinstance(v, list) or not v:
            logger.warning(f"Skipping checkpoint entry {chunk_idx}: value must be a non-empty list.")
            continue

        if not all(isinstance(s, str) for s in v):
            logger.warning(f"Skipping checkpoint entry {chunk_idx}: list contains non-string elements.")
            continue

        validated[chunk_idx] = v

    return validated


def _save_checkpoint(
    checkpoint_path: Path, completed: dict[int, list[str]]
) -> None:
    """Save completed chunk results to checkpoint file."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(completed, f, ensure_ascii=False, indent=2)


def _translate_chunked(
    translator,
    texts: list[str],
    chunk_size: int,
    checkpoint_path: Path,
    corner_cues: list[str] | None = None,
    retry_chunks: list[int] | None = None,
    log_dir: Path | None = None,
) -> tuple[list[str], set[int]]:
    """Split texts into chunks, translate each with retry logic, and merge results."""
    chunks, splits = make_chunks(texts, chunk_size, corner_cues=corner_cues)

    # Set up structured log directory
    chunks_dir = None
    token_summary_path = None
    if log_dir:
        chunks_dir = log_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        token_summary_path = log_dir / "token_summary.tsv"
        # Write header if new file
        if not token_summary_path.exists():
            token_summary_path.write_text(
                "chunk\tlines\tprompt\tcandidates\tthoughts\ttotal\n",
                encoding="utf-8",
            )
    completed = _load_checkpoint(checkpoint_path)

    # Remove specified chunks from checkpoint to force re-translation
    if retry_chunks and completed:
        for idx in retry_chunks:
            chunk_num = idx - 1  # user-facing is 1-based
            if chunk_num in completed:
                del completed[chunk_num]
                logger.info(f"Cleared checkpoint for chunk {idx} — will re-translate.")
            else:
                logger.warning(f"Chunk {idx} not in checkpoint — nothing to retry.")
        _save_checkpoint(checkpoint_path, completed)

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
        line_start = line_offset + 1
        line_end = line_offset + len(chunk)

        if chunk_idx in completed:
            logger.info(
                f"  Chunk {chunk_idx + 1}/{len(chunks)} — skipped (checkpoint)"
            )
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
        results = _translate_with_retry(
            translator, chunk, label=f"Chunk {chunk_idx + 1}"
        )
        completed[chunk_idx] = results
        _save_checkpoint(checkpoint_path, completed)

        # Write structured log files per chunk
        if chunks_dir and hasattr(translator, "last_diagnostics"):
            chunk_num = f"{chunk_idx + 1:02d}"

            # Write system prompt once
            if chunk_idx == 0 or not (log_dir / "system_prompt.txt").exists():
                if hasattr(translator, "last_system_instruction"):
                    (log_dir / "system_prompt.txt").write_text(
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
