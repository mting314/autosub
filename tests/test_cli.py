import autosub.cli as cli_module
from typer.testing import CliRunner

from autosub.cli import app


runner = CliRunner()


def test_transcribe_help_omits_speakers_option():
    result = runner.invoke(app, ["transcribe", "--help"])

    assert result.exit_code == 0
    assert "--speakers" not in result.output


def test_translate_help_uses_chunk_size_only():
    result = runner.invoke(app, ["translate", "--help"])

    assert result.exit_code == 0
    assert "--chunk-size" in result.output
    assert "--no-chunk" not in result.output
    assert "--model" in result.output
    assert "--llm-provider" in result.output
    assert "--vertex-model" in result.output
    assert "--vertex-location" in result.output


def test_run_help_hides_advanced_translation_knobs():
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--engine" not in result.output
    assert "--vertex-model" not in result.output
    assert "--vertex-location" not in result.output
    assert "--vertex-reasoning-budget" not in result.output
    assert "--vertex-reasoning-dynamic" not in result.output
    assert "--speakers" not in result.output
    assert "--no-chunk" not in result.output
    assert "--model" in result.output
    assert "--llm-provider" in result.output
    assert "--vertex-reasoning-effort" in result.output
    assert "--chunk-size" in result.output
    assert "--start" in result.output
    assert "--end" in result.output


def test_translate_model_infers_provider_and_engine(tmp_path, monkeypatch):
    input_ass = tmp_path / "original.ass"
    input_ass.write_text(
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

    def fake_translate_subtitles(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        cli_module.translate_module, "translate_subtitles", fake_translate_subtitles
    )

    result = runner.invoke(
        app, ["translate", str(input_ass), "--model", "claude-sonnet-4-6"]
    )

    assert result.exit_code == 0
    assert captured["engine"] == "vertex"
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4-6"


def test_translate_model_rejects_cloud_v3(tmp_path):
    input_ass = tmp_path / "original.ass"
    input_ass.write_text(
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

    result = runner.invoke(
        app,
        ["translate", str(input_ass), "--engine", "cloud-v3", "--model", "gpt-5-mini"],
    )

    assert result.exit_code == 2
    assert "--model cannot be used with --engine cloud-v3" in result.output


def test_run_model_infers_provider(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module.transcribe_main, "transcribe", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.format_module, "format_subtitles", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.postprocess_module,
        "postprocess_subtitles",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(app, ["run", str(video_path), "--model", "gpt-5-mini"])

    assert result.exit_code == 0
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-5-mini"
