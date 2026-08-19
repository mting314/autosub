import logging
import tomllib
from pathlib import Path

import pyass

from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)


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
            avatar_path = str(Path(avatar_path))

        result[str(label)] = {
            "name": entry.get("name", str(label)),
            "character": entry.get("character"),
            "color": entry.get("color"),
            "slot": slot_val,
            "avatar": avatar_path,
        }
    return result


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


def calculate_speaker_slot_layout(
    slot: int,
    total_slots: int = 3,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
    card_width: int = 260,
    card_margin_left: int = 30,
    text_gap: int = 40,
) -> dict[str, int]:
    """Calculate 2D coordinates for avatar cards and subtitle text for a given speaker slot.

    Slots are 1-indexed (1 = top, 2 = middle, 3 = bottom, etc.).
    """
    total_slots = max(1, total_slots)
    slot_index = max(0, slot - 1)
    slot_height = canvas_height / total_slots

    card_h = int(slot_height * 0.75)
    center_y = int((slot_index + 0.5) * slot_height)
    card_top_y = center_y - (card_h // 2)

    text_x = card_margin_left + card_width + text_gap

    return {
        "slot": slot,
        "text_x": text_x,
        "text_y": center_y,
        "card_x": card_margin_left,
        "card_y": card_top_y,
        "card_width": card_width,
        "card_height": card_h,
    }


def build_speaker_prompt(speaker_map: dict[str, dict]) -> str:
    """Build a prompt fragment describing the speakers in this recording."""
    lines = ["Speakers in this recording:"]
    for entry in speaker_map.values():
        name = entry["name"]
        character = entry.get("character")
        if character:
            lines.append(f"- {name} (voice of {character})")
        else:
            lines.append(f"- {name}")
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
