"""Per-project configuration discovered next to the input audio.

Format (project.toml):

    title = "My Event Title"
    vocab = ["term1", "term2"]

    [glossary]
    "ソース" = "translation"

    [[speakers.cast]]
    name = "Voice Actor Name"
    character = "Character Name"
    color = "#22DDBB"

The cast list, if present, overrides the profile's [[speakers.cast]] for tools
that consume the resolved cast (see `resolve_cast`).
"""

import logging
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_CONFIG_FILENAME = "project.toml"


def load_project_config(reference_path: Path | None) -> dict | None:
    """Auto-discover and load project.toml from the directory of the given file.

    Returns the parsed dict, or None if no file is found.
    """
    if reference_path is None:
        return None
    candidate = Path(reference_path).resolve().parent / PROJECT_CONFIG_FILENAME
    if not candidate.exists():
        return None
    return _read_toml(candidate)


def _read_toml(path: Path) -> dict | None:
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except Exception as exc:
        logger.error(f"Failed to parse {path}: {exc}")
        return None
    logger.info(f"Loaded project config from {path}")
    return data


def merge_glossary(profile_glossary: dict, project_config: dict | None) -> dict:
    """Return a copy of profile_glossary with project [glossary] entries layered on top."""
    merged = dict(profile_glossary or {})
    if not project_config:
        return merged
    extra = project_config.get("glossary")
    if isinstance(extra, dict):
        merged.update(extra)
    return merged


def merge_vocab(profile_vocab: list[str], project_config: dict | None) -> list[str]:
    """Return profile_vocab with project vocab appended (preserving order, deduped)."""
    seen: set[str] = set()
    merged: list[str] = []
    for term in list(profile_vocab or []):
        if term not in seen:
            merged.append(term)
            seen.add(term)
    if project_config:
        extra = project_config.get("vocab", [])
        if isinstance(extra, list):
            for term in extra:
                if isinstance(term, str) and term not in seen:
                    merged.append(term)
                    seen.add(term)
    return merged


def build_title_prompt(project_config: dict | None) -> str | None:
    """Return a prompt fragment about the canonical event title, or None."""
    if not project_config:
        return None
    title = project_config.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    title = title.strip()
    return (
        f'Canonical event title: "{title}". '
        "If a near-match to this title appears in the source (transcribed phonetically "
        "or with dropped/altered syllables), render it exactly as the canonical form. "
        "Do not propagate transcription corruption into the translation."
    )


def resolve_cast(
    profile_cast: list[dict] | None,
    project_config: dict | None,
) -> list[dict]:
    """Resolve the speaker cast for downstream tools that consume cast information.

    Project-level [[speakers.cast]] (if present) wins over profile-level cast.
    Returns a list of {"name", "character", "color"} dicts (any may be missing).
    """
    project_cast = None
    if project_config:
        speakers_block = project_config.get("speakers")
        if isinstance(speakers_block, dict):
            candidate = speakers_block.get("cast")
            if isinstance(candidate, list):
                project_cast = candidate
    if project_cast is not None:
        return [_normalize_cast_entry(entry) for entry in project_cast]
    if profile_cast:
        return [_normalize_cast_entry(entry) for entry in profile_cast]
    return []


def _normalize_cast_entry(entry: object) -> dict:
    if not isinstance(entry, dict):
        return {"name": str(entry), "character": None, "color": None}
    return {
        "name": entry.get("name"),
        "character": entry.get("character"),
        "color": entry.get("color"),
    }
