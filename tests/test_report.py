"""Tests for the HTML review report generator."""

from datetime import timedelta
from pathlib import Path

import pyass
import pytest

from autosub.pipeline.report.analysis import LineReport, analyze_lines
from autosub.pipeline.report.main import _escape, _format_timestamp, generate_report


def _make_event(start_s: float, end_s: float, text: str, style: str = "Default") -> pyass.Event:
    return pyass.Event(
        format=pyass.EventFormat.DIALOGUE,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        style=style,
        name="",
        marginL=0,
        marginR=0,
        marginV=0,
        effect="",
        text=text,
    )


def _make_ass(events: list[pyass.Event]) -> pyass.Script:
    script = pyass.Script()
    script.styles.append(pyass.Style(name="Default"))
    script.events.extend(events)
    return script


class TestAnalyzeLines:
    def test_detects_short_translation(self):
        jp = [_make_event(0, 5, "これは長いテストの文章です")]
        en = [_make_event(0, 5, "Short")]
        lines, stats = analyze_lines(jp, en)
        assert "short" in lines[0].issues
        assert stats.issue_counts.get("short", 0) == 1

    def test_detects_long_translation(self):
        jp = [_make_event(0, 5, "短い文です")]  # 4 chars, meets min_jp_chars_for_long
        en = [_make_event(0, 5, "This is a very long translation that goes on and on and on")]
        lines, stats = analyze_lines(jp, en)
        assert "long" in lines[0].issues
        assert stats.issue_counts.get("long", 0) == 1

    def test_no_long_issue_for_very_short_jp(self):
        """JP text < min_jp_chars_for_long should not trigger long."""
        jp = [_make_event(0, 5, "あ")]
        en = [_make_event(0, 5, "Oh, I see, that makes sense")]
        lines, _ = analyze_lines(jp, en)
        assert "long" not in lines[0].issues

    def test_detects_zero_duration(self):
        jp = [_make_event(10.0, 10.05, "テスト")]
        en = [_make_event(10.0, 10.05, "Test")]
        lines, stats = analyze_lines(jp, en)
        assert "zero_duration" in lines[0].issues
        assert stats.issue_counts.get("zero_duration", 0) == 1

    def test_detects_large_gap(self):
        jp = [_make_event(0, 5, "最初"), _make_event(60, 65, "次")]
        en = [_make_event(0, 5, "First"), _make_event(60, 65, "Next")]
        lines, stats = analyze_lines(jp, en)
        assert "large_gap" in lines[0].issues
        assert stats.issue_counts.get("large_gap", 0) == 1
        # Second line should not have large_gap (no line after it)
        assert "large_gap" not in lines[1].issues

    def test_no_issues_for_normal_line(self):
        jp = [_make_event(0, 3, "普通のテスト文章です")]
        en = [_make_event(0, 3, "This is normal test text")]
        lines, stats = analyze_lines(jp, en)
        assert lines[0].issues == set()
        assert all(v == 0 for v in stats.issue_counts.values()) if stats.issue_counts else True

    def test_stats_computation(self):
        jp = [
            _make_event(0, 3, "テスト"),
            _make_event(3, 6, "もう一つ"),
        ]
        en = [
            _make_event(0, 3, "Test"),
            _make_event(3, 6, "Another one"),
        ]
        lines, stats = analyze_lines(jp, en)
        assert stats.line_count == 2
        assert stats.jp_char_count == len("テスト") + len("もう一つ")
        assert stats.en_char_count == len("Test") + len("Another one")
        assert stats.en_jp_ratio == round(stats.en_char_count / stats.jp_char_count, 2)

    def test_mismatched_line_count(self):
        jp = [_make_event(0, 3, "一"), _make_event(3, 6, "二")]
        en = [_make_event(0, 3, "One")]
        lines, stats = analyze_lines(jp, en)
        assert stats.line_count == 1
        assert len(lines) == 1

    def test_index_is_one_based(self):
        jp = [_make_event(0, 3, "テスト")]
        en = [_make_event(0, 3, "Test")]
        lines, _ = analyze_lines(jp, en)
        assert lines[0].index == 1


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0) == "0:00:00.00"

    def test_seconds(self):
        assert _format_timestamp(5.5) == "0:00:05.50"

    def test_minutes(self):
        assert _format_timestamp(125.3) == "0:02:05.30"

    def test_hours(self):
        assert _format_timestamp(3661.12) == "1:01:01.12"


class TestEscapeHtml:
    def test_escapes_angle_brackets(self):
        assert "&lt;" in _escape("<script>")
        assert "&gt;" in _escape("</script>")

    def test_escapes_ampersand(self):
        assert "&amp;" in _escape("A & B")

    def test_escapes_quotes(self):
        assert "&quot;" in _escape('say "hello"')


class TestGenerateReport:
    def test_creates_valid_html(self, tmp_path: Path):
        jp_events = [_make_event(0, 3, "テスト文"), _make_event(3, 6, "次の行")]
        en_events = [_make_event(0, 3, "Test text"), _make_event(3, 6, "Next line")]

        jp_ass = tmp_path / "original.ass"
        en_ass = tmp_path / "translated.ass"
        out_html = tmp_path / "report.html"

        with open(jp_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(jp_events), f)
        with open(en_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(en_events), f)

        generate_report(jp_ass, en_ass, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "Translation Comparison" in html_content
        assert "テスト文" in html_content
        assert "Test text" in html_content
        assert "次の行" in html_content
        assert "Next line" in html_content
        assert "<table>" in html_content

    def test_with_video(self, tmp_path: Path):
        jp_events = [_make_event(0, 3, "テスト")]
        en_events = [_make_event(0, 3, "Test")]

        jp_ass = tmp_path / "original.ass"
        en_ass = tmp_path / "translated.ass"
        out_html = tmp_path / "report.html"
        video = tmp_path / "video.mkv"
        video.touch()

        with open(jp_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(jp_events), f)
        with open(en_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(en_events), f)

        generate_report(jp_ass, en_ass, out_html, video_path=video)

        html_content = out_html.read_text(encoding="utf-8")
        assert "video.mkv" in html_content
        assert '<video id="video"' in html_content

    def test_without_video(self, tmp_path: Path):
        jp_events = [_make_event(0, 3, "テスト")]
        en_events = [_make_event(0, 3, "Test")]

        jp_ass = tmp_path / "original.ass"
        en_ass = tmp_path / "translated.ass"
        out_html = tmp_path / "report.html"

        with open(jp_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(jp_events), f)
        with open(en_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(en_events), f)

        generate_report(jp_ass, en_ass, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "<video" not in html_content

    def test_custom_title(self, tmp_path: Path):
        jp_events = [_make_event(0, 3, "テスト")]
        en_events = [_make_event(0, 3, "Test")]

        jp_ass = tmp_path / "original.ass"
        en_ass = tmp_path / "translated.ass"
        out_html = tmp_path / "report.html"

        with open(jp_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(jp_events), f)
        with open(en_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(en_events), f)

        generate_report(jp_ass, en_ass, out_html, title="My Custom Title")

        html_content = out_html.read_text(encoding="utf-8")
        assert "My Custom Title" in html_content

    def test_escapes_html_in_subtitle_text(self, tmp_path: Path):
        jp_events = [_make_event(0, 3, "<script>alert('xss')</script>")]
        en_events = [_make_event(0, 3, "A & B \"quoted\"")]

        jp_ass = tmp_path / "original.ass"
        en_ass = tmp_path / "translated.ass"
        out_html = tmp_path / "report.html"

        with open(jp_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(jp_events), f)
        with open(en_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(en_events), f)

        generate_report(jp_ass, en_ass, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "<script>alert" not in html_content
        assert "&lt;script&gt;" in html_content

    def test_issue_filter_buttons_present(self, tmp_path: Path):
        jp_events = [_make_event(10.0, 10.05, "ゼロ長テスト文章です")]
        en_events = [_make_event(10.0, 10.05, "zero")]

        jp_ass = tmp_path / "original.ass"
        en_ass = tmp_path / "translated.ass"
        out_html = tmp_path / "report.html"

        with open(jp_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(jp_events), f)
        with open(en_ass, "w", encoding="utf-8") as f:
            pyass.dump(_make_ass(en_events), f)

        generate_report(jp_ass, en_ass, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "Zero duration" in html_content
        assert "Short translations" in html_content
