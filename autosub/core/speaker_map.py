import logging
import tomllib
from pathlib import Path

import pyass

from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)


def _resolve_avatar(avatar: str, map_path: Path) -> str:
    """Locate an avatar named in a speaker map.

    A path that resolves from the working directory is used as-is, which is how the
    existing maps written relative to the repo root behave. Otherwise it is tried
    again relative to the speaker map itself, so a map can point at assets next to it
    and keep working when the pipeline runs from somewhere else — a git worktree, or a
    remote box the map was copied onto.
    """
    candidate = Path(avatar)
    if candidate.exists():
        return str(candidate)

    beside_map = map_path.parent / candidate
    if beside_map.exists():
        return str(beside_map)

    # Neither location has it. Keep the original so the caller reports the path the
    # map actually asked for, rather than one this function invented.
    return str(candidate)


def load_speaker_map(path: Path) -> dict[str, dict]:
    """Load a speaker_map.toml file.

    Expected format:
        [speakers."0"]
        name = "Suzuki Minori"
        character = "Ena Shinonome"
        color = "#FFA0A0"
        slot = 1
        avatar = "assets/avatars/minoringo.png"

        [speakers."1"]
        name = "Sato Hinata"
        character = "Mizuki Akiyama"
        color = "#A0D0FF"
        slot = 2
        avatar = "assets/avatars/hinata.png"

    Returns {"0": {"name": ..., "character": ..., "color": ..., "slot": ..., "avatar": ...}, ...}
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        data = tomllib.loads(f.read())

    speakers = data.get("speakers", {})
    result = {}
    default_slot = 1
    for label, entry in speakers.items():
        slot_val = entry.get("slot")
        if slot_val is None:
            slot_val = default_slot
            default_slot += 1
        else:
            slot_val = int(slot_val)

        avatar_path = entry.get("avatar")
        if avatar_path:
            avatar_path = _resolve_avatar(avatar_path, path)

        result[str(label)] = {
            "name": entry.get("name", str(label)),
            "character": entry.get("character"),
            "color": entry.get("color"),
            "slot": slot_val,
            "avatar": avatar_path,
        }
    return result


def build_style_name_lookup(speaker_map: dict[str, dict] | None) -> dict[str, str]:
    """Map every raw diarization label and speaker name to its ASS style name.

    A cue's style name is the mapped speaker's name, reachable from either the raw
    label the diarizer emitted or the name it was already remapped to — the format
    stage rewrites labels in place, so both spellings turn up depending on how far
    down the pipeline a document has travelled.
    """
    lookup: dict[str, str] = {}
    if not speaker_map:
        return lookup
    for label, entry in speaker_map.items():
        name = entry.get("name", label)
        if not name:
            continue
        lookup[str(label)] = name
        lookup[str(name)] = name
    return lookup


def style_name_for_speaker(
    speaker: str | None, style_names: dict[str, str] | None
) -> str:
    """Resolve the ASS style a cue renders with, mirroring the generator."""
    name = speaker or "Default"
    if style_names:
        name = style_names.get(name, name)
    return name or "Default"


def build_slot_lookup(speaker_map: dict[str, dict] | None) -> dict[str, int]:
    """Map every raw diarization label and speaker name to its on-screen slot.

    Speaker maps are many-to-one: several diarization labels routinely resolve to
    the same person, and therefore to the same subtitle slot. Anything that reasons
    about what shares a box on screen must key off the slot, not the raw label.
    """
    lookup: dict[str, int] = {}
    if not speaker_map:
        return lookup
    for label, entry in speaker_map.items():
        slot = entry.get("slot")
        if slot is None:
            continue
        lookup[str(label)] = int(slot)
        name = entry.get("name")
        if name:
            lookup[str(name)] = int(slot)
    return lookup


# Subtitle text metrics the slot geometry is sized against.
SLOT_FONT_SIZE = 70
SLOT_FONT_NAME = "Lato ExtraBold"
# Two lines is the design maximum, enforced by linebreak.MAX_LINES.
SLOT_MAX_LINES = 2
SLOT_LINE_HEIGHT = 1.2
# Breathing room between the backdrop bar's edge and the text inside it. Also the
# gap between the avatar card and the start of the bar.
SLOT_TEXT_PAD = 20
SLOT_CANVAS_MARGIN_RIGHT = 30


def calculate_speaker_slot_layout(
    slot: int,
    total_slots: int = 3,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
    card_width: int = 260,
    card_margin_left: int = 30,
    text_gap: int = 40,
    font_size: int = SLOT_FONT_SIZE,
) -> dict[str, int]:
    """Calculate 2D coordinates for avatar cards and subtitle text for a given speaker slot.

    Slots are 1-indexed (1 = top, 2 = middle, 3 = bottom, etc.).

    This is the single source of the overlay's geometry: the PNG generator draws
    the cards and backdrop bars from it, and the ASS styles take their margins
    from it. They have to agree, and they only agree if they are computed once.

    The subtitle anchors at ``text_top`` rather than at the slot's centre. libass
    ignores MarginV for the middle alignments, so a centred slot can only be
    expressed with a \\pos tag baked into the event text — which is what made
    retagging a speaker in Aegisub move the colour but not the line. Top
    anchoring is expressible in the style alone, so the style is the only thing
    a human has to change.
    """
    total_slots = max(1, total_slots)
    slot_index = max(0, slot - 1)
    slot_height = canvas_height / total_slots

    card_h = int(slot_height * 0.75)
    center_y = int((slot_index + 0.5) * slot_height)
    card_top_y = center_y - (card_h // 2)

    text_x = card_margin_left + card_width + text_gap

    # The bar has to hold the tallest thing that can land in it: two full lines.
    bar_height = round(
        SLOT_MAX_LINES * font_size * SLOT_LINE_HEIGHT + 2 * SLOT_TEXT_PAD
    )
    bar_x1 = text_x - SLOT_TEXT_PAD
    bar_x2 = canvas_width - SLOT_CANVAS_MARGIN_RIGHT
    bar_y1 = center_y - bar_height // 2

    return {
        "slot": slot,
        "text_x": text_x,
        "text_y": center_y,
        "text_top": bar_y1 + SLOT_TEXT_PAD,
        "text_margin_right": canvas_width - bar_x2 + SLOT_TEXT_PAD,
        "font_size": font_size,
        "card_x": card_margin_left,
        "card_y": card_top_y,
        "card_width": card_width,
        "card_height": card_h,
        "bar_x1": bar_x1,
        "bar_y1": bar_y1,
        "bar_x2": bar_x2,
        "bar_y2": bar_y1 + bar_height,
        "bar_height": bar_height,
    }


def slot_style(
    style_name: str, outline_color: "pyass.Color | None", layout: dict[str, int]
) -> "pyass.Style":
    """Build the ASS style for one overlay slot.

    White fill with the character's colour in the outline, matching the
    convention these projects already use. Putting the colour in the fill
    instead leaves dark characters unreadable on the slot's dark backdrop: navy
    #172B80 scores 1.17:1 against it, well under the 3.0 WCAG floor for large
    text, and its black outline does not rescue it.

    The slot's position lives here, in the margins, and nowhere else. Every
    caller that writes slot styles goes through this so a line cannot end up
    styled as one speaker but positioned as another.
    """
    return pyass.Style(
        name=style_name,
        fontName=SLOT_FONT_NAME,
        fontSize=layout["font_size"],
        isBold=True,
        primaryColor=pyass.Color(r=255, g=255, b=255, a=0),
        outlineColor=outline_color or pyass.Color(r=255, g=255, b=255, a=0),
        backColor=pyass.Color(r=0, g=0, b=0, a=0),
        outline=2.5,
        shadow=1.5,
        alignment=pyass.Alignment.TOP_LEFT,
        marginL=layout["text_x"],
        marginR=layout["text_margin_right"],
        marginV=layout["text_top"],
    )


def slot_styles_for_map(speaker_map: dict[str, dict]) -> dict[str, "pyass.Style"]:
    """Build one positioned style per named speaker that has a slot.

    Speaker maps are many-to-one, so several labels collapse onto one style.
    Speakers without a slot are skipped: they are not part of the overlay and
    keep whatever styling they already had.
    """
    slots = [
        int(entry["slot"])
        for entry in speaker_map.values()
        if entry.get("slot") is not None
    ]
    total_slots = max(slots) if slots else 1

    styles: dict[str, pyass.Style] = {}
    for entry in speaker_map.values():
        name = entry.get("name")
        slot = entry.get("slot")
        if not name or slot is None or name in styles:
            continue
        color = hex_to_pyass_color(entry["color"]) if entry.get("color") else None
        styles[name] = slot_style(
            name,
            color,
            calculate_speaker_slot_layout(slot=int(slot), total_slots=total_slots),
        )
    return styles


def build_speaker_prompt(speaker_map: dict[str, dict]) -> str:
    """Build a prompt fragment describing the speakers in this recording.

    The framing matters as much as the list. Given only "Name (voice of
    Character)", translators drift into reading the show as the characters
    talking, and render first-person lines as if the role were speaking.
    """
    lines = ["Speakers in this recording:"]
    has_characters = False
    for entry in speaker_map.values():
        name = entry["name"]
        character = entry.get("character")
        if character:
            has_characters = True
            lines.append(f"- {name} (voice of {character})")
        else:
            lines.append(f"- {name}")

    if has_characters:
        lines.append("")
        # Kept free of gendered pronouns: this runs for whatever cast the speaker
        # map names, unlike the per-show prompt files which know theirs.
        lines.append(
            "These hosts are the voice actors themselves, appearing as themselves. "
            "They are not in character. When a host says 'I', they mean themselves: "
            "their own life, work, opinions and experiences, not their character's. "
            "A character name refers to a role they play, so treat it as a third "
            "party being talked about, never as the speaker."
        )
    return "\n".join(lines)


def remap_speaker_labels(
    lines: list[SubtitleLine], speaker_map: dict[str, dict]
) -> None:
    """Replace raw API speaker labels with character names. Mutates in place."""
    for line in lines:
        if line.speaker and line.speaker in speaker_map:
            line.speaker = speaker_map[line.speaker]["name"]


def hex_to_pyass_color(hex_color: str) -> pyass.Color:
    """Convert '#RRGGBB' hex string to pyass.Color."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return pyass.Color(r=r, g=g, b=b, a=0)
