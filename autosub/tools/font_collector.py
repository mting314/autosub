"""Collect the fonts a subtitle script asks for and attach them to it.

libass resolves a style's Fontname against whatever fontconfig offers on the
machine doing the render. On any other machine the name silently resolves to a
different face, and the burn comes out in the wrong font — usually at a
noticeably different size, because metrics differ between faces.

Attaching the fonts to the script fixes that at the source. ASS has a ``[Fonts]``
section for exactly this (it is what Aegisub's Fonts Collector writes with
"Attach fonts to subtitle file"), libass reads it, and the file then renders the
same anywhere without needing ``fontsdir`` or anything installed.

Which fonts a script needs is read from the script itself — every style's
Fontname plus any inline ``\\fn`` override — so this works for whatever font a
project happens to use rather than a hardcoded list.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

FONT_SUFFIXES = (".ttf", ".otf", ".ttc", ".otc")

# Searched after the script's own directory. Covers the usual per-user and
# system locations on macOS, Linux and Windows.
_SYSTEM_FONT_DIRS = (
    "~/Library/Fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    "~/.fonts",
    "~/.local/share/fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "C:/Windows/Fonts",
    "~/AppData/Local/Microsoft/Windows/Fonts",
)

_STYLE_RE = re.compile(r"^Style:\s*([^,]*),\s*([^,]*),", re.MULTILINE)
_INLINE_FN_RE = re.compile(r"\\fn([^\\}]+)")


def fonts_required(ass_text: str) -> set[str]:
    """Every font family the script asks for, by name.

    Reads each style's Fontname and any inline ``\\fn`` override. A leading ``@``
    marks a vertical-writing variant of the same family, so it is stripped.
    """
    names: set[str] = set()
    for _style_name, font_name in _STYLE_RE.findall(ass_text):
        cleaned = font_name.strip().lstrip("@")
        if cleaned:
            names.add(cleaned)
    for match in _INLINE_FN_RE.findall(ass_text):
        cleaned = match.strip().lstrip("@")
        if cleaned:
            names.add(cleaned)
    return names


def _describe(path: Path) -> tuple[str, str] | None:
    """The (family, style) a font file declares, or None if unreadable."""
    try:
        from PIL import ImageFont

        family, style = ImageFont.truetype(str(path), 12).getname()
        return (family or "").strip(), (style or "").strip()
    except Exception:
        return None


def _candidate_dirs(script_dir: Path, extra_dirs: Iterable[Path] | None) -> list[Path]:
    dirs = [script_dir]
    for extra in extra_dirs or ():
        dirs.append(Path(extra))
    for raw in _SYSTEM_FONT_DIRS:
        dirs.append(Path(raw).expanduser())
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        try:
            resolved = d.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def find_font_file(
    name: str, script_dir: Path, extra_dirs: Iterable[Path] | None = None
) -> Path | None:
    """Locate the file providing font family ``name``.

    Matched against each candidate's declared family, and against
    "Family Style" so a style-qualified name like "Lato ExtraBold" resolves to
    the face that declares family "Lato", style "ExtraBold". The script's own
    directory wins, which is where projects already drop bundled fonts.
    """
    wanted = name.strip().lower()
    for directory in _candidate_dirs(script_dir, extra_dirs):
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in FONT_SUFFIXES or not path.is_file():
                continue
            described = _describe(path)
            if described is None:
                continue
            family, style = described
            if wanted in {
                family.lower(),
                f"{family} {style}".strip().lower(),
                path.stem.lower(),
            }:
                return path
    return None


def uuencode(data: bytes) -> list[str]:
    """Encode bytes the way ASS attachments are encoded.

    Three bytes become four characters, each a 6-bit group offset by 33, and a
    trailing partial group emits one character per byte plus one. Lines wrap at
    80 characters.
    """
    values: list[int] = []
    for offset in range(0, len(data), 3):
        chunk = data[offset : offset + 3]
        padded = chunk + b"\x00" * (3 - len(chunk))
        packed = (padded[0] << 16) | (padded[1] << 8) | padded[2]
        groups = [
            (packed >> 18) & 63,
            (packed >> 12) & 63,
            (packed >> 6) & 63,
            packed & 63,
        ]
        values.extend(groups[: len(chunk) + 1])
    text = "".join(chr(v + 33) for v in values)
    return [text[i : i + 80] for i in range(0, len(text), 80)]


# Section boundaries have to be matched against the known ASS section names, not
# "a line in brackets". The attachment alphabet runs from ASCII 33 to 96, so
# encoded font data contains plenty of lines that both start with "[" and end
# with "]" — about one in every four thousand — and treating those as headers
# ends the section early and leaves orphaned font data behind.
_KNOWN_SECTIONS = {
    "[script info]",
    "[v4 styles]",
    "[v4+ styles]",
    "[v4++ styles]",
    "[events]",
    "[fonts]",
    "[graphics]",
    "[aegisub project garbage]",
    "[aegisub extradata]",
}


def _is_section_header(line: str) -> bool:
    return line.strip().lower() in _KNOWN_SECTIONS


def strip_fonts_section(text: str) -> str:
    """Remove an existing ``[Fonts]`` section, leaving the rest untouched."""
    out: list[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        if _is_section_header(line):
            skipping = line.strip().lower() == "[fonts]"
            if skipping:
                continue
        if not skipping:
            out.append(line)
    return "".join(out)


def _attachment_name(path: Path, index: int) -> str:
    """Aegisub names attachments ``<stem>_<n><suffix>``; match that."""
    return f"{path.stem}_{index}{path.suffix.lower()}"


def build_fonts_section(font_paths: Iterable[Path]) -> str:
    parts = ["[Fonts]"]
    for index, path in enumerate(font_paths):
        parts.append(f"fontname: {_attachment_name(path, index)}")
        parts.extend(uuencode(path.read_bytes()))
    return "\n".join(parts) + "\n"


def embed_fonts(
    ass_path: Path,
    output_path: Path | None = None,
    extra_dirs: Iterable[Path] | None = None,
) -> tuple[list[Path], list[str]]:
    """Attach every font the script needs, returning (embedded, missing).

    Any existing ``[Fonts]`` section is replaced, so this is safe to re-run.
    """
    ass_path = Path(ass_path)
    output_path = Path(output_path) if output_path else ass_path
    text = ass_path.read_text(encoding="utf-8-sig")

    # Drop a previous [Fonts] section so re-running does not stack attachments.
    text = strip_fonts_section(text)

    embedded: list[Path] = []
    missing: list[str] = []
    for name in sorted(fonts_required(text)):
        found = find_font_file(name, ass_path.parent, extra_dirs)
        if found is None:
            missing.append(name)
            logger.warning(
                "Font %r not found; the burn will fall back to another face.", name
            )
        else:
            embedded.append(found)
            logger.info("Attaching %s for %r", found.name, name)

    if embedded:
        section = "\n" + build_fonts_section(embedded)
        if "\n[Events]" in text:
            text = text.replace("\n[Events]", section + "\n[Events]", 1)
        else:
            text = text.rstrip("\n") + "\n" + section
    output_path.write_text(text, encoding="utf-8")
    return embedded, missing
