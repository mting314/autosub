from pathlib import Path
from typing import List
import pyass

from autosub.core.schemas import SubtitleLine
from autosub.core.speaker_map import hex_to_pyass_color, remap_speaker_labels


def generate_ass_file(
    lines: List[SubtitleLine],
    output_path: Path,
    speaker_map: dict[str, dict] | None = None,
):
    """
    Converts a list of SubtitleLine objects into a pyass Script and saves it to disk.
    Automatically generates unique styles per speaker.

    If speaker_map is provided, uses character names and specified colors
    instead of raw API labels with auto-colors.
    """
    # Remap raw speaker labels to character names before building styles
    if speaker_map:
        remap_speaker_labels(lines, speaker_map)

    # 1. Identify unique speakers and generate styles
    unique_speakers = {line.speaker if line.speaker else "Default" for line in lines}

    # Pre-defined array of subtle color tints for up to a few speakers
    auto_colors = [
        pyass.Color(r=255, g=255, b=255, a=0),  # White
        pyass.Color(r=255, g=255, b=200, a=0),  # Light Yellow
        pyass.Color(r=200, g=255, b=255, a=0),  # Light Cyan
        pyass.Color(r=255, g=200, b=255, a=0),  # Light Magenta
        pyass.Color(r=200, g=255, b=200, a=0),  # Light Green
    ]

    # Build lookups from speaker_map (raw label / name → slot, color, mapped name)
    map_colors: dict[str, pyass.Color] = {}
    map_slots: dict[str, int] = {}
    raw_to_name: dict[str, str] = {}

    if speaker_map:
        for label, entry in speaker_map.items():
            spk_name = entry.get("name", label)
            raw_to_name[label] = spk_name
            raw_to_name[spk_name] = spk_name
            if entry.get("color"):
                color_val = hex_to_pyass_color(entry["color"])
                map_colors[label] = color_val
                map_colors[spk_name] = color_val
            if entry.get("slot") is not None:
                map_slots[label] = entry["slot"]
                map_slots[spk_name] = entry["slot"]

    total_slots = (
        max([s for s in map_slots.values() if s is not None] or [1])
        if map_slots
        else len(unique_speakers)
    )

    styles = []
    speakerOriginToStyleMap = {}

    for i, speaker_name in enumerate(sorted(unique_speakers)):
        resolved_name = raw_to_name.get(speaker_name, speaker_name)
        style_name = resolved_name if resolved_name else "Default"
        c = map_colors.get(
            speaker_name,
            map_colors.get(resolved_name, auto_colors[i % len(auto_colors)]),
        )

        slot = map_slots.get(speaker_name, map_slots.get(resolved_name))
        if slot is not None:
            from autosub.core.speaker_map import calculate_speaker_slot_layout

            layout = calculate_speaker_slot_layout(slot=slot, total_slots=total_slots)
            st = pyass.Style(
                name=style_name,
                fontName="Arial",
                fontSize=44,
                isBold=True,
                primaryColor=c,
                outlineColor=pyass.Color(r=0, g=0, b=0, a=0),
                backColor=pyass.Color(r=0, g=0, b=0, a=0),
                outline=2.0,
                shadow=2.0,
                alignment=pyass.Alignment.CENTER_LEFT,
                marginL=layout["text_x"],
                marginV=0,
            )
        else:
            st = pyass.Style(
                name=style_name,
                fontName="Arial",
                fontSize=48,
                isBold=True,
                primaryColor=c,
                outlineColor=pyass.Color(r=0, g=0, b=0, a=0),
                backColor=pyass.Color(r=0, g=0, b=0, a=0),
                outline=2.0,
                shadow=2.0,
                alignment=pyass.Alignment.BOTTOM,
                marginV=20,
            )
        styles.append(st)
        speakerOriginToStyleMap[speaker_name] = style_name
        speakerOriginToStyleMap[resolved_name] = style_name

    # 2. Convert SubtitleLines into pyass Events
    pyass_events: List[pyass.Event] = []

    for line in lines:
        assigned_speaker = line.speaker if line.speaker else "Default"
        resolved_name = raw_to_name.get(assigned_speaker, assigned_speaker)
        assigned_style = speakerOriginToStyleMap.get(
            resolved_name, speakerOriginToStyleMap.get(assigned_speaker, "Default")
        )
        event_name = line.role or (resolved_name if resolved_name else "")

        slot = map_slots.get(assigned_speaker, map_slots.get(resolved_name))
        if slot is not None:
            from autosub.core.speaker_map import calculate_speaker_slot_layout

            layout = calculate_speaker_slot_layout(slot=slot, total_slots=total_slots)
            event_text = f"{{\\pos({layout['text_x']},{layout['text_y']})}}{line.text}"
        else:
            event_text = line.text

        if line.corner:
            pyass_events.append(
                pyass.Event(
                    format=pyass.EventFormat.COMMENT,
                    start=pyass.timedelta(seconds=line.start_time),
                    end=pyass.timedelta(seconds=line.end_time),
                    style=assigned_style,
                    effect="corner",
                    text=f"=== Corner: {line.corner} ===",
                )
            )

        pyass_events.append(
            pyass.Event(
                start=pyass.timedelta(seconds=line.start_time),
                end=pyass.timedelta(seconds=line.end_time),
                style=assigned_style,
                name=event_name,
                text=event_text,
            )
        )

    # 3. Create the pyass Script container
    script = pyass.Script(styles=styles, events=pyass_events)
    script.scriptInfo.append(("PlayResX", "1920"))
    script.scriptInfo.append(("PlayResY", "1080"))

    # 4. Dump to disk
    with open(output_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)
