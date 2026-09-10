from PIL import Image

from autosub.core.speaker_map import calculate_speaker_slot_layout
from autosub.tools.overlay_generator import generate_radio_overlay_image


def test_calculate_speaker_slot_layout():
    # Slot 1 of 3 on 1920x1080 canvas
    layout1 = calculate_speaker_slot_layout(
        slot=1, total_slots=3, canvas_width=1920, canvas_height=1080
    )
    assert layout1["slot"] == 1
    assert layout1["text_x"] == 330  # 30 + 260 + 40
    assert layout1["text_y"] == 180  # 1080 / 3 * 0.5
    assert layout1["card_x"] == 30

    # Slot 2 of 3
    layout2 = calculate_speaker_slot_layout(
        slot=2, total_slots=3, canvas_width=1920, canvas_height=1080
    )
    assert layout2["slot"] == 2
    assert layout2["text_y"] == 540  # 1080 / 3 * 1.5

    # Slot 3 of 3
    layout3 = calculate_speaker_slot_layout(
        slot=3, total_slots=3, canvas_width=1920, canvas_height=1080
    )
    assert layout3["slot"] == 3
    assert layout3["text_y"] == 900  # 1080 / 3 * 2.5


def test_subtitle_text_box_sits_inside_its_backdrop_bar():
    """The PNG bar and the ASS style margins must come from one calculation.

    They are drawn by different modules into different file formats; if they
    drift, the subtitle renders outside the bar that is supposed to back it.
    """
    from autosub.core.speaker_map import (
        SLOT_LINE_HEIGHT,
        SLOT_MAX_LINES,
        slot_style,
    )

    canvas_w, canvas_h = 1920, 1080
    for slot in (1, 2, 3):
        layout = calculate_speaker_slot_layout(
            slot=slot, total_slots=3, canvas_width=canvas_w, canvas_height=canvas_h
        )
        style = slot_style(f"S{slot}", None, layout)

        # Left and right edges of the text box fall inside the bar.
        assert style.marginL >= layout["bar_x1"]
        assert canvas_w - style.marginR <= layout["bar_x2"]

        # Two full lines, growing down from the top-anchored margin, still fit.
        text_bottom = style.marginV + SLOT_MAX_LINES * style.fontSize * SLOT_LINE_HEIGHT
        assert style.marginV >= layout["bar_y1"]
        assert text_bottom <= layout["bar_y2"]

    # Slots do not collide with each other.
    bars = [
        calculate_speaker_slot_layout(slot=s, total_slots=3, canvas_height=canvas_h)
        for s in (1, 2, 3)
    ]
    for upper, lower in zip(bars, bars[1:]):
        assert upper["bar_y2"] < lower["bar_y1"]


def test_generate_radio_overlay_image(tmp_path):
    toml_content = """\
[speakers."0"]
name = "Date Sayuri"
character = "Kanon"
color = "#FF9E00"
slot = 1

[speakers."1"]
name = "Liyuu"
character = "Keke"
color = "#00A3E0"
slot = 2
"""
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text(toml_content, encoding="utf-8")

    out_png = tmp_path / "overlay.png"
    result_path = generate_radio_overlay_image(
        map_file, out_png, canvas_size=(1920, 1080)
    )

    assert result_path.exists()
    assert result_path == out_png

    # Verify PNG image properties
    with Image.open(out_png) as img:
        assert img.size == (1920, 1080)
        assert img.mode == "RGBA"


def test_missing_avatar_warns_instead_of_silently_blanking(tmp_path, caplog):
    """A blank card is easy to miss; an unresolvable avatar path must be logged."""
    import logging

    from autosub.tools.overlay_generator import generate_radio_overlay_image

    speaker_map = {
        "0": {
            "name": "Someone",
            "character": "A Character",
            "color": "#5DD1CA",
            "slot": 1,
            "avatar": "relative/path/that/does/not/exist.png",
        }
    }
    with caplog.at_level(logging.WARNING):
        generate_radio_overlay_image(speaker_map, tmp_path / "overlay.png")

    assert "not found" in caplog.text
    assert "Someone" in caplog.text


def test_avatarless_speaker_warns(tmp_path, caplog):
    import logging

    from autosub.tools.overlay_generator import generate_radio_overlay_image

    speaker_map = {"0": {"name": "Someone", "color": "#5DD1CA", "slot": 1}}
    with caplog.at_level(logging.WARNING):
        generate_radio_overlay_image(speaker_map, tmp_path / "overlay.png")

    assert "No avatar set" in caplog.text
