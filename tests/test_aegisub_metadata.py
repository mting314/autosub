"""Tests for Aegisub Project Garbage metadata injection."""

import pyass

from autosub.core.schemas import SubtitleLine
from autosub.pipeline.format.generator import generate_ass_file, inject_aegisub_metadata


def _line(text, start=0.0, end=1.0):
    return SubtitleLine(text=text, start_time=start, end_time=end)


def test_inject_metadata_adds_section(tmp_path):
    ass_path = tmp_path / "test.ass"
    video_path = tmp_path / "video.mkv"
    video_path.touch()

    generate_ass_file([_line("hello")], ass_path)
    inject_aegisub_metadata(ass_path, video_path)

    content = ass_path.read_text(encoding="utf-8")
    assert "[Aegisub Project Garbage]" in content
    assert "Video File: video.mkv" in content
    assert "Audio File: video.mkv" in content


def test_inject_metadata_before_styles(tmp_path):
    ass_path = tmp_path / "test.ass"
    video_path = tmp_path / "video.mkv"
    video_path.touch()

    generate_ass_file([_line("hello")], ass_path)
    inject_aegisub_metadata(ass_path, video_path)

    content = ass_path.read_text(encoding="utf-8")
    garbage_pos = content.index("[Aegisub Project Garbage]")
    styles_pos = content.index("[V4+ Styles]")
    assert garbage_pos < styles_pos


def test_inject_metadata_relative_path(tmp_path):
    sub_dir = tmp_path / "subs"
    sub_dir.mkdir()
    ass_path = sub_dir / "test.ass"
    video_path = tmp_path / "video.mkv"
    video_path.touch()

    generate_ass_file([_line("hello")], ass_path)
    inject_aegisub_metadata(ass_path, video_path)

    content = ass_path.read_text(encoding="utf-8")
    # Video is one level up from the ASS file
    assert "Video File: ../video.mkv" in content


def test_inject_metadata_preserves_events(tmp_path):
    ass_path = tmp_path / "test.ass"
    video_path = tmp_path / "video.mkv"
    video_path.touch()

    generate_ass_file([_line("hello"), _line("world", 1.0, 2.0)], ass_path)
    inject_aegisub_metadata(ass_path, video_path)

    # pyass can still load the file
    with open(ass_path, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    assert len(script.events) == 2
    assert script.events[0].text == "hello"
    assert script.events[1].text == "world"


def test_inject_metadata_same_directory(tmp_path):
    ass_path = tmp_path / "test.ass"
    video_path = tmp_path / "video.mkv"
    video_path.touch()

    generate_ass_file([_line("hello")], ass_path)
    inject_aegisub_metadata(ass_path, video_path)

    content = ass_path.read_text(encoding="utf-8")
    assert "Video File: video.mkv" in content
