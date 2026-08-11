from pathlib import Path
from PIL import Image
import pytest

from autosub.core.speaker_map import calculate_speaker_slot_layout, load_speaker_map
from autosub.tools.overlay_generator import generate_radio_overlay_image


def test_calculate_speaker_slot_layout():
    # Slot 1 of 3 on 1920x1080 canvas
    layout1 = calculate_speaker_slot_layout(slot=1, total_slots=3, canvas_width=1920, canvas_height=1080)
    assert layout1["slot"] == 1
    assert layout1["text_x"] == 410  # 50 + 300 + 60
    assert layout1["text_y"] == 180  # 1080 / 3 * 0.5
    assert layout1["card_x"] == 50

    # Slot 2 of 3
    layout2 = calculate_speaker_slot_layout(slot=2, total_slots=3, canvas_width=1920, canvas_height=1080)
    assert layout2["slot"] == 2
    assert layout2["text_y"] == 540  # 1080 / 3 * 1.5

    # Slot 3 of 3
    layout3 = calculate_speaker_slot_layout(slot=3, total_slots=3, canvas_width=1920, canvas_height=1080)
    assert layout3["slot"] == 3
    assert layout3["text_y"] == 900  # 1080 / 3 * 2.5


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
    result_path = generate_radio_overlay_image(map_file, out_png, canvas_size=(1920, 1080))

    assert result_path.exists()
    assert result_path == out_png

    # Verify PNG image properties
    with Image.open(out_png) as img:
        assert img.size == (1920, 1080)
        assert img.mode == "RGBA"
