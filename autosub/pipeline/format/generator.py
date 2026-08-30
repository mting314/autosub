from pathlib import Path
from typing import Literal, NamedTuple
import pyass

from autosub.core.schemas import SubtitleCue, SubtitleDocument, SubtitleLine
from autosub.core.speaker_map import (
    build_slot_lookup,
    hex_to_pyass_color,
    remap_speaker_labels,
)

# Script resolution the styles are authored against (1080p). Must be written into the
# header — see the note in _script_from_entries.
PLAY_RES_X = 1920
PLAY_RES_Y = 1080


AssRenderMode = Literal["source", "translated", "bilingual", "final"]


class _AssEntry(NamedTuple):
    text: str
    start_time: float
    end_time: float
    speaker: str | None = None
    role: str | None = None
    corner: str | None = None


def generate_ass_file(
    lines: list[SubtitleLine],
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

    write_ass_script(
        _script_from_entries(
            [_line_to_entry(line) for line in lines], speaker_map=speaker_map
        ),
        output_path,
    )


def build_ass_script(
    document: SubtitleDocument,
    *,
    mode: AssRenderMode,
    chunk_boundaries: list[int] | set[int] | None = None,
    speaker_map: dict[str, dict] | None = None,
) -> pyass.Script:
    """Build the ASS script for a document without writing it.

    Split out from render_ass_document so callers that need a layout pass over
    the finished events (line breaking, which needs per-style capacities that
    only exist once styles are built) can modify the script before it is
    written.
    """
    entries = [
        _AssEntry(
            text=_cue_text_for_mode(cue, mode),
            start_time=cue.start_time,
            end_time=cue.end_time,
            speaker=cue.speaker,
            role=cue.role,
            corner=cue.corner,
        )
        for cue in document.cues
    ]

    script = _script_from_entries(entries, speaker_map=speaker_map)
    boundaries = (
        document.chunk_boundaries if chunk_boundaries is None else chunk_boundaries
    )
    if boundaries:
        script.events = _insert_chunk_boundary_comments(script.events, set(boundaries))
    return script


def render_ass_document(
    document: SubtitleDocument,
    output_path: Path,
    *,
    mode: AssRenderMode,
    chunk_boundaries: list[int] | set[int] | None = None,
    speaker_map: dict[str, dict] | None = None,
) -> None:
    """Render a structured subtitle document into an ASS byproduct."""
    write_ass_script(
        build_ass_script(
            document,
            mode=mode,
            chunk_boundaries=chunk_boundaries,
            speaker_map=speaker_map,
        ),
        output_path,
    )


def write_ass_script(script: pyass.Script, output_path: Path) -> None:
    # Auto-link project background overlay for Aegisub if present.
    output_dir = Path(output_path).parent
    if (output_dir / "radio_background_with_overlay.png").exists():
        script.scriptInfo.append(("Video File", "radio_background_with_overlay.png"))

    with open(output_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)


def _cue_text_for_mode(cue: SubtitleCue, mode: AssRenderMode) -> str:
    source = cue.normalized_source_text or cue.source_text
    translated = cue.translated_text or source
    final = cue.final_text or translated

    if mode == "source":
        return source
    if mode == "translated":
        return translated
    if mode == "final":
        return final
    if mode == "bilingual":
        return rf"{{\fs24\a6}}{source}{{\N}}{{\fs48\a2}}{final}"
    raise ValueError(f"Unknown ASS render mode: {mode}")


def _line_to_entry(line: SubtitleLine) -> _AssEntry:
    return _AssEntry(
        text=line.text,
        start_time=line.start_time,
        end_time=line.end_time,
        speaker=line.speaker,
        role=line.role,
        corner=line.corner,
    )


def _script_from_entries(
    entries: list[_AssEntry],
    speaker_map: dict[str, dict] | None = None,
) -> pyass.Script:
    # 1. Identify unique speakers and generate styles
    unique_speakers = {
        entry.speaker if entry.speaker else "Default" for entry in entries
    }

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
        map_slots.update(build_slot_lookup(speaker_map))

    total_slots = (
        max([s for s in map_slots.values() if s is not None] or [1])
        if map_slots
        else len(unique_speakers)
    )

    styles = []
    speaker_origin_to_style_map = {}
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
            # White fill with the character's colour in the outline, matching the
            # convention these projects already use. Putting the colour in the
            # fill instead leaves dark characters unreadable on the slot's dark
            # backdrop: navy #172B80 scores 1.17:1 against it, well under the 3.0
            # WCAG floor for large text, and its black outline does not rescue it.
            st = pyass.Style(
                name=style_name,
                fontName="Lato ExtraBold",
                fontSize=70,
                isBold=True,
                primaryColor=pyass.Color(r=255, g=255, b=255, a=0),
                outlineColor=c,
                backColor=pyass.Color(r=0, g=0, b=0, a=0),
                outline=2.5,
                shadow=1.5,
                alignment=pyass.Alignment.CENTER_LEFT,
                marginL=layout["text_x"],
                marginR=80,
                marginV=0,
            )
        else:
            st = pyass.Style(
                name=style_name,
                fontName="Arial",
                fontSize=54,
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
        speaker_origin_to_style_map[speaker_name] = style_name
        speaker_origin_to_style_map[resolved_name] = style_name

    # 2. Convert entries into pyass Events
    pyass_events: list[pyass.Event] = []

    for entry in entries:
        assigned_speaker = entry.speaker if entry.speaker else "Default"
        resolved_name = raw_to_name.get(assigned_speaker, assigned_speaker)
        assigned_style = speaker_origin_to_style_map.get(
            resolved_name,
            speaker_origin_to_style_map.get(assigned_speaker, "Default"),
        )
        event_name = entry.role or (resolved_name if resolved_name else "")

        slot = map_slots.get(assigned_speaker, map_slots.get(resolved_name))
        if slot is not None:
            from autosub.core.speaker_map import calculate_speaker_slot_layout

            layout = calculate_speaker_slot_layout(slot=slot, total_slots=total_slots)
            event_text = f"{{\\pos({layout['text_x']},{layout['text_y']})}}{entry.text}"
        else:
            event_text = entry.text

        if entry.corner:
            pyass_events.append(
                pyass.Event(
                    format=pyass.EventFormat.COMMENT,
                    start=pyass.timedelta(seconds=entry.start_time),
                    end=pyass.timedelta(seconds=entry.end_time),
                    style=assigned_style,
                    effect="corner",
                    text=f"=== Corner: {entry.corner} ===",
                )
            )

        pyass_events.append(
            pyass.Event(
                start=pyass.timedelta(seconds=entry.start_time),
                end=pyass.timedelta(seconds=entry.end_time),
                style=assigned_style,
                name=event_name,
                text=event_text,
            )
        )

    # 3. Create the pyass Script container
    script = pyass.Script(styles=styles, events=pyass_events)

    # pyass omits PlayResX/PlayResY. Without them libass falls back to a 384x288 script
    # canvas and scales that up to the video frame, so a Fontsize-100 style renders ~5x
    # too large and every line wraps. Pin the resolution the styles are authored against.
    script.scriptInfo.append(("PlayResX", str(PLAY_RES_X)))
    script.scriptInfo.append(("PlayResY", str(PLAY_RES_Y)))
    return script


def _insert_chunk_boundary_comments(
    events: list[pyass.Event],
    chunk_boundaries: set[int],
) -> list[pyass.Event]:
    new_events: list[pyass.Event] = []
    dialogue_idx = 0
    for event in events:
        if isinstance(event, pyass.Event) and event.format != pyass.EventFormat.COMMENT:
            if dialogue_idx in chunk_boundaries:
                new_events.append(
                    pyass.Event(
                        format=pyass.EventFormat.COMMENT,
                        start=event.start,
                        end=event.end,
                        style=event.style,
                        effect="",
                        text="[autosub] Chunk boundary - review translation around this line",
                    )
                )
            dialogue_idx += 1
        new_events.append(event)
    return new_events
