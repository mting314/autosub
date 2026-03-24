import pytest
from scripts import freeze_frame_video as script


def test_resolve_job_paths_creates_default_output_for_non_subtitle_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=None,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    assert paths.input_video == input_video.resolve()
    assert paths.output_video.suffix == ".mkv"
    assert "freeze_1080p" in paths.output_video.name
    assert paths.frame_image.suffix == ".png"
    assert paths.subtitle is None


def test_resolve_job_paths_creates_default_output_for_subtitle_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    subtitle = tmp_path / "sample.ass"
    input_video.write_bytes(b"video")
    subtitle.write_text("[Script Info]\n", encoding="utf-8")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=subtitle,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    assert paths.output_video.suffix == ".mp4"
    assert paths.subtitle == subtitle.resolve()


def test_resolve_job_paths_enforces_correct_suffix_based_on_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    # Non-subtitle mode requires .mkv
    with pytest.raises(ValueError, match="requires .mkv output"):
        script.resolve_job_paths(
            input_video=input_video,
            subtitle=None,
            output=tmp_path / "bad.mp4",
            frame_image=None,
            upscaled_image=None,
        )

    # Subtitle mode requires .mp4
    subtitle = tmp_path / "sample.ass"
    subtitle.write_text("[Script Info]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires .mp4 output"):
        script.resolve_job_paths(
            input_video=input_video,
            subtitle=subtitle,
            output=tmp_path / "bad.mkv",
            frame_image=None,
            upscaled_image=None,
        )


def test_build_subtitles_filter_escapes_windows_style_path(tmp_path):
    subtitle = tmp_path / "dir[name]" / "line,one;two's.ass"
    subtitle.parent.mkdir()
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    filter_value = script.escape_filter_path(subtitle)

    assert "\\[" in filter_value
    assert "\\]" in filter_value
    assert "\\," in filter_value
    assert "\\;" in filter_value


def test_build_video_command_uses_requested_codecs(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=None,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    command = script.build_final_command(
        paths=paths,
        overwrite=True,
        fps=30,
        crf=18,
        preset="medium",
        audio_bitrate="192k",
    )

    rendered = " ".join(command)

    assert "-vcodec" in command
    assert command[command.index("-vcodec") + 1] == "libx264"
    assert "-acodec" in command
    assert command[command.index("-acodec") + 1] == "aac"
    assert "-b:a" in command
    assert command[command.index("-b:a") + 1] == "192k"
    assert paths.output_video.as_posix() in rendered

    inputs = [command[i + 1] for i, v in enumerate(command) if v == "-i"]
    assert paths.input_video.as_posix() in inputs
    assert paths.upscaled_image.as_posix() in inputs


def test_build_video_command_adds_subtitle_filter_for_burn_in_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    subtitle = tmp_path / "sample.ass"
    input_video.write_bytes(b"video")
    subtitle.write_text("[Script Info]\n", encoding="utf-8")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=subtitle,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    command = script.build_final_command(
        paths=paths,
        overwrite=True,
        fps=30,
        crf=18,
        preset="medium",
        audio_bitrate="192k",
    )
    rendered = " ".join(command)

    assert "subtitles=" in rendered
    assert "-movflags" in command
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert paths.output_video.as_posix() in rendered


def test_ensure_output_paths_requires_overwrite_for_existing_files(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")
    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=None,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )
    paths.frame_image.write_bytes(b"png")

    with pytest.raises(FileExistsError, match="--overwrite"):
        script.ensure_output_paths(paths, overwrite=False)
