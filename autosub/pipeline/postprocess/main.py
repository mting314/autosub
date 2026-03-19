from __future__ import annotations

import logging
from pathlib import Path

import pyass

logger = logging.getLogger(__name__)

BILINGUAL_TRANSLATION_TAG = r"{\fs48\a2}"
QUOTE_CHARS = {'"', "“", "”"}


def postprocess_subtitles(
    input_ass_path: Path,
    output_ass_path: Path | None = None,
    extensions_config: dict | None = None,
    bilingual: bool = True,
) -> None:
    if output_ass_path is None:
        output_ass_path = input_ass_path

    if not extensions_config:
        return

    with open(input_ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    modified = False

    radio_discourse_config = extensions_config.get("radio_discourse", {})
    if radio_discourse_config.get("enabled"):
        modified = _apply_radio_discourse_postprocess(script, bilingual) or modified

    if not modified:
        return

    logger.info(f"Writing postprocessed subtitles to {output_ass_path}...")
    with open(output_ass_path, "w", encoding="utf-8") as handle:
        pyass.dump(script, handle)


def _apply_radio_discourse_postprocess(script: pyass.Script, bilingual: bool) -> bool:
    modified = False
    for event in script.events:
        if not isinstance(event, pyass.Event) or not event.text:
            continue
        if event.name != "listener_mail":
            continue

        quoted = _quote_listener_mail_text(event.text, bilingual=bilingual)
        if quoted != event.text:
            event.text = quoted
            modified = True
    return modified


def _quote_listener_mail_text(text: str, bilingual: bool) -> str:
    if bilingual and BILINGUAL_TRANSLATION_TAG in text:
        prefix, translated = text.rsplit(BILINGUAL_TRANSLATION_TAG, 1)
        return f"{prefix}{BILINGUAL_TRANSLATION_TAG}{_ensure_quoted(translated)}"

    return _ensure_quoted(text)


def _ensure_quoted(text: str) -> str:
    stripped = text.strip()
    if (
        len(stripped) >= 2
        and stripped[0] in QUOTE_CHARS
        and stripped[-1] in QUOTE_CHARS
    ):
        return text
    return f'"{text}"'
