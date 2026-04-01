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
    assert "--vertex-reasoning-effort" in result.output
    assert "--chunk-size" in result.output
    assert "--start" in result.output
    assert "--end" in result.output
