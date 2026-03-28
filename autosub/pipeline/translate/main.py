import json
import logging
import traceback
from pathlib import Path

import pyass
from autosub.core.config import PROJECT_ID
from autosub.core.llm import ReasoningEffort

logger = logging.getLogger(__name__)


def translate_subtitles(
    input_ass_path: Path,
    output_ass_path: Path,
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
        reasoning_effort: Provider-agnostic reasoning effort for LLM translation.
        reasoning_budget_tokens: Optional token-budget override for LLM reasoning.
        reasoning_dynamic: Whether to request dynamic reasoning budget when supported.
        chunk_size: If > 0, split into chunks of this size.
    """
    logger.info(f"Loading '{input_ass_path}' for translation...")

    with open(input_ass_path, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    # Extract only the dialogue text
    # We maintain a reference list to easily write the translations back to the correct events
    events_to_translate = []
    texts_to_translate = []

    for event in script.events:
        # standard Dialogue events or non-commented events
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

    llm_trace_path: Path | None = None

    if engine == "vertex":
        from autosub.pipeline.translate.translator import VertexTranslator

        if provider == "google-vertex" and not PROJECT_ID:
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

    checkpoint_path = output_ass_path.with_suffix(".checkpoint.json")
    error_path = output_ass_path.with_suffix(".error.txt")

    if error_path.exists():
        error_path.unlink()
        logger.info("Removed previous translation error file.")

    try:
        if chunk_size > 0:
            translated_texts = _translate_chunked(
                translator, texts_to_translate, chunk_size, checkpoint_path
            )
        else:
            translated_texts = translator.translate(texts_to_translate)
    except Exception as exc:
        _write_error_report(error_path, exc)
        logger.error(f"Wrote translation error details to {error_path}.")
        raise

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file.")

    if len(translated_texts) != len(events_to_translate):
        raise ValueError(
            f"Translation API expected {len(events_to_translate)} translations, but got {len(translated_texts)}"
        )

    logger.info("Applying translations to subtitle events...")

    # Update the events with the new text
    for i, event in enumerate(events_to_translate):
        original_text = texts_to_translate[i]
        translated_text = translated_texts[i]

        if bilingual:
            # Create a stacked visual format: Original (small/translucent) on top, Translated (large) on bottom
            # Formatting uses standard ASS override tags
            new_text_str = f"{{\\\\fs24\\\\a6}}{original_text}{{\\\\N}}{{\\\\fs48\\\\a2}}{translated_text}"

            # Since Pyass parses parts, we assign a new EventText part containing our constructed string.
            # Technically, Pyass will            # overide tags as standard text if we don't parse them,
            # but Aegisub/video players handle raw string output perfectly fine.
            event.text = new_text_str
        else:
            event.text = translated_text

    logger.info(f"Writing translated .ass file to {output_ass_path}...")
    with open(output_ass_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)

    if llm_trace_path is not None and llm_trace_path.exists():
        logger.info(f"Wrote LLM trace to {llm_trace_path}.")

    logger.info("Translation complete!")


def _write_error_report(error_path: Path, exc: Exception) -> None:
    error_path.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )


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
        logger.warning("Checkpoint is not a JSON object, starting fresh.")
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


def _save_checkpoint(checkpoint_path: Path, completed: dict[int, list[str]]) -> None:
    """Save completed chunk results to checkpoint file."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(completed, f, ensure_ascii=False, indent=2)


def _translate_chunked(
    translator, texts: list[str], chunk_size: int, checkpoint_path: Path
) -> list[str]:
    """Split texts into chunks, translate each once, and merge results."""
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    completed = _load_checkpoint(checkpoint_path)

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
        results = translator.translate(chunk)
        completed[chunk_idx] = results
        _save_checkpoint(checkpoint_path, completed)
        line_offset += len(chunk)

    # Reassemble in order
    all_translated: list[str] = []
    for chunk_idx in range(len(chunks)):
        all_translated.extend(completed[chunk_idx])

    return all_translated
