"""Tests for the HTML review report generator."""

from pathlib import Path

import pytest

from autosub.core.schemas import SubtitleCue, SubtitleDocument
from autosub.pipeline.report.analysis import analyze_cues
from autosub.pipeline.report.main import _escape, _format_timestamp, generate_report


def _make_cue(
    index: int,
    start: float,
    end: float,
    source: str,
    translated: str,
    *,
    role: str | None = None,
    speaker: str | None = None,
    final: str | None = None,
) -> SubtitleCue:
    return SubtitleCue(
        id=f"cue-{index:05d}",
        start_time=start,
        end_time=end,
        source_text=source,
        translated_text=translated,
        final_text=final,
        role=role,
        speaker=speaker,
    )


def _write_translated_document(path: Path, cues: list[SubtitleCue]) -> None:
    document = SubtitleDocument(stage="translated", cues=cues)
    path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


class TestAnalyzeCues:
    def test_detects_short_translation(self):
        cues = [_make_cue(1, 0, 5, "これは長いテストの文章です", "Short")]
        lines, stats = analyze_cues(cues)
        assert "short" in lines[0].issues
        assert stats.issue_counts.get("short", 0) == 1

    def test_detects_long_translation(self):
        cues = [
            _make_cue(
                1,
                0,
                5,
                "短い文です",
                "This is a very long translation that goes on and on and on",
            )
        ]
        lines, stats = analyze_cues(cues)
        assert "long" in lines[0].issues
        assert stats.issue_counts.get("long", 0) == 1

    def test_no_long_issue_for_very_short_jp(self):
        cues = [_make_cue(1, 0, 5, "あ", "Oh, I see, that makes sense")]
        lines, _ = analyze_cues(cues)
        assert "long" not in lines[0].issues

    def test_detects_zero_duration(self):
        cues = [_make_cue(1, 10.0, 10.05, "テスト", "Test")]
        lines, stats = analyze_cues(cues)
        assert "zero_duration" in lines[0].issues
        assert stats.issue_counts.get("zero_duration", 0) == 1

    def test_detects_large_gap(self):
        cues = [
            _make_cue(1, 0, 5, "最初", "First"),
            _make_cue(2, 60, 65, "次", "Next"),
        ]
        lines, stats = analyze_cues(cues)
        assert "large_gap" in lines[0].issues
        assert stats.issue_counts.get("large_gap", 0) == 1
        assert "large_gap" not in lines[1].issues

    def test_no_issues_for_normal_line(self):
        cues = [_make_cue(1, 0, 3, "普通のテスト文章です", "This is normal test text")]
        lines, stats = analyze_cues(cues)
        assert lines[0].issues == set()
        assert (
            all(v == 0 for v in stats.issue_counts.values())
            if stats.issue_counts
            else True
        )

    def test_stats_computation(self):
        cues = [
            _make_cue(1, 0, 3, "テスト", "Test"),
            _make_cue(2, 3, 6, "もう一つ", "Another one"),
        ]
        lines, stats = analyze_cues(cues)
        assert stats.line_count == 2
        assert stats.jp_char_count == len("テスト") + len("もう一つ")
        assert stats.en_char_count == len("Test") + len("Another one")
        assert stats.en_jp_ratio == round(stats.en_char_count / stats.jp_char_count, 2)

    def test_index_is_one_based(self):
        cues = [_make_cue(1, 0, 3, "テスト", "Test")]
        lines, _ = analyze_cues(cues)
        assert lines[0].index == 1

    def test_cue_id_is_preserved(self):
        cues = [_make_cue(42, 0, 3, "テスト", "Test")]
        lines, _ = analyze_cues(cues)
        assert lines[0].cue_id == "cue-00042"

    def test_uses_role_for_style(self):
        cues = [
            _make_cue(1, 0, 3, "テスト", "Test", role="listener_mail", speaker="rino")
        ]
        lines, _ = analyze_cues(cues)
        assert lines[0].style == "listener_mail"

    def test_falls_back_to_speaker_when_no_role(self):
        cues = [_make_cue(1, 0, 3, "テスト", "Test", speaker="rino")]
        lines, _ = analyze_cues(cues)
        assert lines[0].style == "rino"

    def test_default_style_when_no_role_or_speaker(self):
        cues = [_make_cue(1, 0, 3, "テスト", "Test")]
        lines, _ = analyze_cues(cues)
        assert lines[0].style == "Default"

    def test_prefers_final_text_over_translated(self):
        cues = [
            _make_cue(
                1,
                0,
                3,
                "テスト",
                translated="raw translation",
                final="post-processed translation",
            )
        ]
        lines, _ = analyze_cues(cues)
        assert lines[0].en_text == "post-processed translation"

    def test_prefers_normalized_source_over_raw(self):
        cue = SubtitleCue(
            id="cue-00001",
            start_time=0,
            end_time=3,
            source_text="raw",
            normalized_source_text="normalized",
            translated_text="translation",
        )
        lines, _ = analyze_cues([cue])
        assert lines[0].jp_text == "normalized"


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
        cues = [
            _make_cue(1, 0, 3, "テスト文", "Test text"),
            _make_cue(2, 3, 6, "次の行", "Next line"),
        ]
        input_json = tmp_path / "translated.json"
        out_html = tmp_path / "report.html"
        _write_translated_document(input_json, cues)

        generate_report(input_json, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "Translation Comparison" in html_content
        assert "テスト文" in html_content
        assert "Test text" in html_content
        assert "次の行" in html_content
        assert "Next line" in html_content
        assert "<table>" in html_content

    def test_with_video(self, tmp_path: Path):
        cues = [_make_cue(1, 0, 3, "テスト", "Test")]
        input_json = tmp_path / "translated.json"
        out_html = tmp_path / "report.html"
        video = tmp_path / "video.mkv"
        video.touch()
        _write_translated_document(input_json, cues)

        generate_report(input_json, out_html, video_path=video)

        html_content = out_html.read_text(encoding="utf-8")
        assert "video.mkv" in html_content
        assert '<video id="video"' in html_content

    def test_without_video(self, tmp_path: Path):
        cues = [_make_cue(1, 0, 3, "テスト", "Test")]
        input_json = tmp_path / "translated.json"
        out_html = tmp_path / "report.html"
        _write_translated_document(input_json, cues)

        generate_report(input_json, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "<video" not in html_content

    def test_custom_title(self, tmp_path: Path):
        cues = [_make_cue(1, 0, 3, "テスト", "Test")]
        input_json = tmp_path / "translated.json"
        out_html = tmp_path / "report.html"
        _write_translated_document(input_json, cues)

        generate_report(input_json, out_html, title="My Custom Title")

        html_content = out_html.read_text(encoding="utf-8")
        assert "My Custom Title" in html_content

    def test_escapes_html_in_subtitle_text(self, tmp_path: Path):
        cues = [
            _make_cue(1, 0, 3, "<script>alert('xss')</script>", 'A & B "quoted"'),
        ]
        input_json = tmp_path / "translated.json"
        out_html = tmp_path / "report.html"
        _write_translated_document(input_json, cues)

        generate_report(input_json, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "<script>alert" not in html_content
        assert "&lt;script&gt;" in html_content

    def test_issue_filter_buttons_present(self, tmp_path: Path):
        cues = [_make_cue(1, 10.0, 10.05, "ゼロ長テスト文章です", "zero")]
        input_json = tmp_path / "translated.json"
        out_html = tmp_path / "report.html"
        _write_translated_document(input_json, cues)

        generate_report(input_json, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "Zero duration" in html_content
        assert "Short translations" in html_content

    def test_accepts_postprocessed_document(self, tmp_path: Path):
        cues = [
            _make_cue(
                1, 0, 3, "テスト", translated="raw", final='"polished translation"'
            )
        ]
        document = SubtitleDocument(stage="postprocessed", cues=cues)
        input_json = tmp_path / "postprocessed.json"
        input_json.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        out_html = tmp_path / "report.html"

        generate_report(input_json, out_html)

        html_content = out_html.read_text(encoding="utf-8")
        assert "polished translation" in html_content
        assert ">raw<" not in html_content

    def test_rejects_formatted_document(self, tmp_path: Path):
        cues = [_make_cue(1, 0, 3, "テスト", "")]
        document = SubtitleDocument(stage="formatted", cues=cues)
        input_json = tmp_path / "formatted.json"
        input_json.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        out_html = tmp_path / "report.html"

        with pytest.raises(ValueError, match="report expects"):
            generate_report(input_json, out_html)
