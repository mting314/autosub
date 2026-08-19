from unittest.mock import MagicMock, patch

import pytest
import autosub.pipeline.translate.main as translate_main_module
import autosub.pipeline.translate.translator as translator_module

from autosub.pipeline.translate.main import (
    _compute_fingerprint,
    _translate_chunked,
    _load_checkpoint,
    _save_checkpoint,
    _write_error_report,
    translate_subtitles,
)


class FakeTranslator:
    """Translator that returns prefixed text."""

    def translate(self, texts: list[str]) -> list[str]:
        return [f"translated:{t}" for t in texts]


class FailNTimesTranslator:
    """Translator that fails N times then succeeds."""

    def __init__(self, fail_count: int):
        self.fail_count = fail_count
        self.attempts = 0

    def translate(self, texts: list[str]) -> list[str]:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ConnectionError("Server disconnected without sending a response.")
        return [f"translated:{t}" for t in texts]


# --- Error report tests ---


def test_write_error_report_includes_traceback(tmp_path):
    error_path = tmp_path / "translated.error.txt"

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        _write_error_report(error_path, exc)

    report = error_path.read_text(encoding="utf-8")
    assert "Traceback" in report
    assert "RuntimeError: boom" in report


# --- _translate_chunked tests ---


def test_chunked_splits_and_merges(tmp_path):
    translator = FakeTranslator()
    texts = [f"line{i}" for i in range(5)]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result, splits = _translate_chunked(
        translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
    )

    assert result == [f"translated:line{i}" for i in range(5)]
    # Checkpoint should still exist (caller is responsible for cleanup)
    assert checkpoint_path.exists()


def test_chunked_fails_fast_on_error(tmp_path):
    translator = FailNTimesTranslator(fail_count=1)
    texts = ["a", "b", "c"]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    with pytest.raises(ConnectionError):
        _translate_chunked(
            translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
        )

    assert translator.attempts == 1
    assert not checkpoint_path.exists()


def test_chunked_preserves_order(tmp_path):
    translator = FakeTranslator()
    texts = [f"line{i}" for i in range(10)]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result, splits = _translate_chunked(
        translator, texts, chunk_size=3, checkpoint_path=checkpoint_path
    )

    assert len(result) == 10
    for i in range(10):
        assert result[i] == f"translated:line{i}"


# --- Checkpoint tests ---


def test_save_and_load_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "test.checkpoint.json"
    data = {0: ["a", "b"], 1: ["c", "d"]}
    fp = "test_fingerprint"

    _save_checkpoint(checkpoint_path, data, fp)
    loaded = _load_checkpoint(checkpoint_path, fp)

    assert loaded == data


def test_load_checkpoint_missing_file(tmp_path):
    checkpoint_path = tmp_path / "nonexistent.json"
    assert _load_checkpoint(checkpoint_path, "any") == {}


def test_load_checkpoint_corrupt_file(tmp_path):
    checkpoint_path = tmp_path / "corrupt.json"
    checkpoint_path.write_text("not valid json{{{")
    assert _load_checkpoint(checkpoint_path, "any") == {}


def test_load_checkpoint_not_a_dict(tmp_path):
    checkpoint_path = tmp_path / "bad.json"
    checkpoint_path.write_text('["a", "b"]')
    assert _load_checkpoint(checkpoint_path, "any") == {}


def test_load_checkpoint_skips_non_integer_keys(tmp_path):
    checkpoint_path = tmp_path / "bad_keys.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a"], "foo": ["b"], "1": ["c"]}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"], 1: ["c"]}


def test_load_checkpoint_skips_negative_keys(tmp_path):
    checkpoint_path = tmp_path / "neg.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"-1": ["a"], "0": ["b"]}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["b"]}


def test_load_checkpoint_skips_empty_lists(tmp_path):
    checkpoint_path = tmp_path / "empty.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a"], "1": []}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_list_values(tmp_path):
    checkpoint_path = tmp_path / "bad_vals.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a"], "1": "not a list", "2": 42}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_string_elements(tmp_path):
    checkpoint_path = tmp_path / "bad_elems.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a", "b"], "1": [1, 2]}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a", "b"]}


def test_chunked_resumes_from_checkpoint(tmp_path):
    """Simulate a previous run that completed chunks 0 and 1, then resume."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d", "e", "f"]
    fp = _compute_fingerprint(texts, chunk_size=2, corner_boundaries=None)

    # Pre-populate checkpoint with chunks 0 and 1 already done
    existing = {
        0: ["translated:a", "translated:b"],
        1: ["translated:c", "translated:d"],
    }
    _save_checkpoint(checkpoint_path, existing, fp)

    # Track which texts the translator actually receives
    translated_inputs = []
    original_translate = FakeTranslator.translate

    def tracking_translate(self, texts):
        translated_inputs.extend(texts)
        return original_translate(self, texts)

    translator = FakeTranslator()
    translator.translate = lambda texts: tracking_translate(translator, texts)

    result, splits = _translate_chunked(
        translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
    )

    # Should only translate chunk 2 (lines e, f)
    assert translated_inputs == ["e", "f"]
    # Full result should include checkpointed + new
    assert result == [
        "translated:a",
        "translated:b",
        "translated:c",
        "translated:d",
        "translated:e",
        "translated:f",
    ]


def test_checkpoint_saved_after_each_chunk(tmp_path):
    """Verify checkpoint file is written after each chunk completes."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d"]

    translator = FakeTranslator()

    with patch("autosub.pipeline.translate.main._save_checkpoint") as mock_save:
        mock_save.side_effect = lambda path, completed, fp: _save_checkpoint(
            path, completed, fp
        )
        _translate_chunked(
            translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
        )
        # Should be called once per chunk
        assert mock_save.call_count == 2


def test_translate_subtitles_sets_llm_trace_path(tmp_path, monkeypatch):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(input_ass_path, output_ass_path, engine="vertex")

    assert captured["trace_path"] == output_ass_path.with_suffix(".llm_trace.jsonl")


def test_translate_subtitles_allows_anthropic_without_google_project(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path,
        output_ass_path,
        engine="vertex",
        provider="anthropic",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "anthropic"
    assert captured["model"] is None


def test_translate_subtitles_allows_anthropic_vertex_with_google_project(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path,
        output_ass_path,
        engine="vertex",
        provider="anthropic-vertex",
    )

    assert captured["project_id"] == "test-project"
    assert captured["provider"] == "anthropic-vertex"
    assert captured["model"] is None


def test_translate_subtitles_allows_openai_without_google_project(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path,
        output_ass_path,
        engine="vertex",
        provider="openai",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "openai"
    assert captured["model"] is None


def test_translate_subtitles_allows_openrouter_without_google_project(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path,
        output_ass_path,
        engine="vertex",
        provider="openrouter",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "openrouter"
    assert captured["model"] is None


def test_translate_subtitles_requires_google_project_for_anthropic_vertex(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT is not set"):
        translate_subtitles(
            input_ass_path,
            output_ass_path,
            engine="vertex",
            provider="anthropic-vertex",
        )


def test_translate_subtitles_writes_error_file_on_failure(tmp_path, monkeypatch):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    class FailingVertexTranslator:
        def __init__(self, **kwargs):
            pass

        def translate(self, texts: list[str]) -> list[str]:
            raise RuntimeError("translation exploded")

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FailingVertexTranslator)

    with pytest.raises(RuntimeError, match="translation exploded"):
        translate_subtitles(input_ass_path, output_ass_path, engine="vertex")

    error_path = output_ass_path.with_suffix(".error.txt")
    report = error_path.read_text(encoding="utf-8")
    assert "Traceback" in report
    assert "RuntimeError: translation exploded" in report


def test_chunked_all_checkpointed_skips_translation(tmp_path):
    """If all chunks are in the checkpoint, no translation calls should be made."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d"]
    fp = _compute_fingerprint(texts, chunk_size=2, corner_boundaries=None)

    existing = {
        0: ["translated:a", "translated:b"],
        1: ["translated:c", "translated:d"],
    }
    _save_checkpoint(checkpoint_path, existing, fp)

    translator = MagicMock()

    result, splits = _translate_chunked(
        translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
    )

    translator.translate.assert_not_called()
    assert result == ["translated:a", "translated:b", "translated:c", "translated:d"]


# --- Fingerprint tests ---


def test_load_checkpoint_fingerprint_mismatch(tmp_path):
    """Checkpoint with wrong fingerprint is discarded."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    _save_checkpoint(checkpoint_path, {0: ["a"]}, "fingerprint_aaa")
    result = _load_checkpoint(checkpoint_path, "fingerprint_bbb")
    assert result == {}


def test_load_checkpoint_legacy_format_discarded(tmp_path):
    """Old-format checkpoint (no _fingerprint) is discarded."""
    checkpoint_path = tmp_path / "legacy.json"
    import json

    json.dump({"0": ["a", "b"], "1": ["c"]}, open(checkpoint_path, "w"))
    result = _load_checkpoint(checkpoint_path, "any_fingerprint")
    assert result == {}


def test_fingerprint_changes_with_texts():
    fp1 = _compute_fingerprint(["a", "b", "c"], chunk_size=2, corner_boundaries=None)
    fp2 = _compute_fingerprint(["b", "c"], chunk_size=2, corner_boundaries=None)
    assert fp1 != fp2


def test_fingerprint_changes_with_chunk_size():
    texts = ["a", "b", "c", "d"]
    fp1 = _compute_fingerprint(texts, chunk_size=2, corner_boundaries=None)
    fp2 = _compute_fingerprint(texts, chunk_size=3, corner_boundaries=None)
    assert fp1 != fp2


def test_translate_preserves_slot_pos_tags(tmp_path, monkeypatch):
    """The slot \\pos written by the format stage must survive translation."""
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\pos(330,540)}こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            pass

        def translate(self, texts: list[str]) -> list[str]:
            # Translators return prose, dropping any override tags they were sent.
            return ["Hello there"] * len(texts)

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path, output_ass_path, engine="vertex", bilingual=False
    )

    written = output_ass_path.read_text(encoding="utf-8")
    assert "{\\pos(330,540)}Hello there" in written


def test_split_leading_tags_separates_override_blocks():
    from autosub.pipeline.translate.main import _split_leading_tags

    assert _split_leading_tags("{\\pos(1,2)}hi") == ("{\\pos(1,2)}", "hi")
    assert _split_leading_tags("{\\pos(1,2)}{\\b1}hi") == ("{\\pos(1,2)}{\\b1}", "hi")
    assert _split_leading_tags("plain") == ("", "plain")
    # Tags that are not leading stay part of the body.
    assert _split_leading_tags("hi {\\i1}there") == ("", "hi {\\i1}there")


# --- Netflix line-break rules ---


def _wrap(text):
    from autosub.pipeline.translate.linebreak import wrap_line

    return wrap_line(text, nlp=None)


def test_wrapped_lines_stay_within_two_lines_and_the_char_cap():
    """Netflix budget: at most two lines, at most 42 characters each."""
    test_cases = [
        "Short line under threshold.",
        "Well then, let's start the radio show wearing our matching brooches.",
        "They want us to tell them our favorite green things.",
        "Line 1\\NLine 2\\NLine 3\\NLine 4 should be re-balanced.",
        "That groundless feeling makes me like you even more.",
    ]
    for text in test_cases:
        wrapped = _wrap(text)
        if wrapped is None:
            continue  # no legal layout; the caller splits it into two events
        for line in wrapped.split("\\N"):
            assert len(line.strip()) <= 42, f"{line!r} exceeds 42 chars"
        assert len(wrapped.split("\\N")) <= 2, f"too many lines for {text!r}"


def test_short_line_is_left_on_one_line():
    assert _wrap("Thanks for having me.") == "Thanks for having me."


def test_break_prefers_punctuation_over_the_midpoint():
    """The old midpoint split broke 'Shiki-san\'s | birthday'; the comma is correct."""
    wrapped = _wrap("Tomorrow is Shiki-san's birthday, isn't it?")
    assert wrapped == "Tomorrow is Shiki-san's birthday,\\Nisn't it?"


def test_never_breaks_before_an_infinitive_marker():
    from autosub.pipeline.translate.linebreak import best_break

    prose = "We really wanted to find something green to give her today"
    idx = best_break(prose, nlp=None, max_chars=42, require_fit=False)
    assert idx is None or not prose[idx:].startswith("to ")


def test_never_strands_a_coordinator_without_a_comma():
    from autosub.pipeline.translate.linebreak import best_break

    prose = "She picked the brooch and I paid for the wrapping at the counter"
    idx = best_break(prose, nlp=None, max_chars=42, require_fit=False)
    if idx is not None:
        assert not prose[:idx].strip().endswith(("and", "but", "or"))


def test_unbreakable_long_line_returns_none_for_event_splitting():
    """A long run with no legal boundary must not be broken arbitrarily."""
    assert _wrap("aaaaaaaaaa " * 12) is None


def test_inline_override_tags_survive_wrapping():
    """Japanese terms are italicised inline; wrapping must not strip the tags."""
    from autosub.pipeline.translate.linebreak import wrap_line

    text = r"She called it {\i1}oshi{\i0} which is the one you support the most, apparently"
    wrapped = wrap_line(text, nlp=None, max_chars=62)
    assert wrapped is not None
    assert r"{\i1}" in wrapped and r"{\i0}" in wrapped
    assert r"\N" in wrapped


def test_wrapping_measures_visible_text_not_tag_characters():
    """A short line dressed in tags must not be broken as though it were long."""
    from autosub.pipeline.translate.linebreak import visible_length, wrap_line

    text = r"{\i1}{\b1}{\fs48}Short line.{\r}"
    assert visible_length(text) == len("Short line.")
    assert "\\N" not in (wrap_line(text, nlp=None, max_chars=20) or "")


def test_split_text_keeps_tags_and_splits_on_visible_text():
    from autosub.pipeline.translate.linebreak import split_text

    parts = split_text(
        r"She called it {\i1}oshi{\i0} which is the one you support, apparently",
        nlp=None,
        max_chars=20,
    )
    assert parts is not None
    assert r"{\i1}" in parts[0] + parts[1]


def test_zero_duration_event_is_not_split():
    """Splitting a zero-length event would put the cut past its own end."""
    import pyass
    from autosub.pipeline.translate.main import _lay_out_event

    event = pyass.Event(
        format=pyass.EventFormat.DIALOGUE,
        style="S",
        start=pyass.timedelta(seconds=5),
        end=pyass.timedelta(seconds=5),
        text="This sentence is far too long to fit, and it needs a split.",
    )
    out = _lay_out_event(event, None, 20)
    assert len(out) == 1
    assert out[0].start <= out[0].end
