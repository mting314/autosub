from pathlib import Path
from threading import Barrier
import time
from types import SimpleNamespace

from autosub.core.schemas import (
    TranscribedWord,
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
)
from autosub.pipeline.transcribe import main as transcribe_main


def _offset(seconds: float) -> SimpleNamespace:
    whole_seconds = int(seconds)
    nanos = int(round((seconds - whole_seconds) * 1_000_000_000))
    return SimpleNamespace(seconds=whole_seconds, nanos=nanos)


def _make_local_response(
    word: str, start_time: float, end_time: float
) -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        words=[
                            SimpleNamespace(
                                word=word,
                                start_offset=_offset(start_time),
                                end_offset=_offset(end_time),
                            )
                        ]
                    )
                ]
            )
        ]
    )


def test_transcribe_merges_multiple_ranges_concurrently(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    output_path = tmp_path / "transcript.json"
    created_audio_paths: list[Path] = []
    barrier = Barrier(2)

    monkeypatch.setattr(transcribe_main, "PROJECT_ID", "project-id")

    def fake_extract_audio(
        input_video_path: Path,
        start_time: str | None = None,
        end_time: str | None = None,
        opus: bool = False,
    ) -> Path:
        audio_path = tmp_path / f"audio_{start_time}_{end_time}.wav"
        audio_path.write_text(start_time or "full", encoding="utf-8")
        created_audio_paths.append(audio_path)
        return audio_path

    monkeypatch.setattr(transcribe_main.audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        transcribe_main.audio, "get_audio_duration", lambda audio_path: 5.0
    )

    def fake_transcribe_local_file(
        audio_content: bytes,
        project_id: str,
        language_code: str = "ja-JP",
        vocabulary: list[str] | None = None,
        num_speakers: int | None = None,
        model: str = "chirp_3",
    ) -> SimpleNamespace:
        segment_id = audio_content.decode("utf-8")
        barrier.wait(timeout=2)
        if segment_id == "0":
            time.sleep(0.1)
        return _make_local_response(f"word-{segment_id}", 0.1, 0.3)

    monkeypatch.setattr(
        transcribe_main.api, "transcribe_local_file", fake_transcribe_local_file
    )

    result = transcribe_main.transcribe(
        video_path,
        output_path,
        time_ranges=[("15", "20"), ("0", "5")],
    )

    assert [word.word for word in result.words] == ["word-0", "word-15"]
    assert [word.start_time for word in result.words] == [0.1, 15.1]
    assert len(result.segments) == 2
    assert result.metadata is not None
    assert result.metadata.backend == "chirp_3"
    assert output_path.exists()
    assert all(not path.exists() for path in created_audio_paths)


def test_transcribe_fails_if_any_segment_fails(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    output_path = tmp_path / "transcript.json"
    created_audio_paths: list[Path] = []
    barrier = Barrier(2)

    monkeypatch.setattr(transcribe_main, "PROJECT_ID", "project-id")

    def fake_extract_audio(
        input_video_path: Path,
        start_time: str | None = None,
        end_time: str | None = None,
        opus: bool = False,
    ) -> Path:
        audio_path = tmp_path / f"audio_{start_time}_{end_time}.wav"
        audio_path.write_text(start_time or "full", encoding="utf-8")
        created_audio_paths.append(audio_path)
        return audio_path

    monkeypatch.setattr(transcribe_main.audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        transcribe_main.audio, "get_audio_duration", lambda audio_path: 5.0
    )

    def fake_transcribe_local_file(
        audio_content: bytes,
        project_id: str,
        language_code: str = "ja-JP",
        vocabulary: list[str] | None = None,
        num_speakers: int | None = None,
        model: str = "chirp_3",
    ) -> SimpleNamespace:
        segment_id = audio_content.decode("utf-8")
        barrier.wait(timeout=2)
        if segment_id == "15":
            raise RuntimeError("segment boom")
        return _make_local_response(f"word-{segment_id}", 0.1, 0.3)

    monkeypatch.setattr(
        transcribe_main.api, "transcribe_local_file", fake_transcribe_local_file
    )

    try:
        transcribe_main.transcribe(
            video_path,
            output_path,
            time_ranges=[("0", "5"), ("15", "20")],
        )
        assert False, "Expected transcription to fail when one segment fails."
    except RuntimeError as exc:
        assert "segment 2" in str(exc)
        assert "segment boom" in str(exc)

    assert not output_path.exists()
    assert all(not path.exists() for path in created_audio_paths)


def test_transcribe_whisperx_does_not_require_google_project_or_gcs(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    output_path = tmp_path / "transcript.json"
    created_audio_paths: list[Path] = []

    monkeypatch.setattr(transcribe_main, "PROJECT_ID", None)

    def fake_extract_audio(
        input_video_path: Path,
        start_time: str | None = None,
        end_time: str | None = None,
        opus: bool = False,
    ) -> Path:
        audio_path = tmp_path / "audio.wav"
        audio_path.write_text("fake", encoding="utf-8")
        created_audio_paths.append(audio_path)
        return audio_path

    monkeypatch.setattr(transcribe_main.audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        transcribe_main.audio, "get_audio_duration", lambda audio_path: 125.0
    )
    monkeypatch.setattr(
        transcribe_main.gcs,
        "upload_to_gcs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("WhisperX should not upload to GCS.")
        ),
    )
    monkeypatch.setattr(
        transcribe_main.whisperx_backend,
        "transcribe_file",
        lambda *args, **kwargs: TranscriptionResult(
            words=[TranscribedWord(word="hello", start_time=0.25, end_time=0.75)],
            segments=[
                TranscriptionSegment(
                    text="hello",
                    start_time=0.25,
                    end_time=0.75,
                    words=[
                        TranscribedWord(
                            word="hello",
                            start_time=0.25,
                            end_time=0.75,
                        )
                    ],
                    kind="sentence",
                )
            ],
            metadata=TranscriptionMetadata(
                backend="whisperx",
                language="ja",
                model="large-v2",
            ),
        ),
    )

    result = transcribe_main.transcribe(
        video_path,
        output_path,
        time_ranges=[("15", "20")],
        transcription_backend="whisperx",
    )

    assert [word.word for word in result.words] == ["hello"]
    assert [word.start_time for word in result.words] == [15.25]
    assert result.segments[0].text == "hello"
    assert result.segments[0].start_time == 15.25
    assert result.metadata is not None
    assert result.metadata.backend == "whisperx"
    assert output_path.exists()
    assert all(not path.exists() for path in created_audio_paths)


def test_transcribe_whisperx_rejects_unknown_backend(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    output_path = tmp_path / "transcript.json"

    try:
        transcribe_main.transcribe(
            video_path,
            output_path,
            transcription_backend="unknown-backend",
        )
        assert False, "Expected unsupported backends to fail."
    except ValueError as exc:
        assert "Unsupported transcription backend" in str(exc)


# --- Bogus timestamp clamping tests ---


def test_clamp_bogus_zero_end_offset():
    start, end = transcribe_main._clamp_word_timestamps(5.0, 0.0, 1080.0)
    assert start == 5.0
    assert end == 5.0


def test_clamp_end_exceeds_chunk_duration():
    start, end = transcribe_main._clamp_word_timestamps(100.0, 1200.0, 1080.0)
    assert start == 100.0
    assert end == 100.0


def test_clamp_start_exceeds_chunk_duration():
    start, end = transcribe_main._clamp_word_timestamps(1200.0, 50.0, 1080.0)
    assert start == 50.0
    assert end == 50.0


def test_clamp_both_exceed_chunk_duration():
    start, end = transcribe_main._clamp_word_timestamps(1200.0, 1300.0, 1080.0)
    assert start == 0.0
    assert end == 0.0


def test_clamp_negative_end_offset():
    start, end = transcribe_main._clamp_word_timestamps(5.0, -1.0, 1080.0)
    assert start == 5.0
    assert end == 5.0


def test_clamp_end_less_than_start():
    start, end = transcribe_main._clamp_word_timestamps(10.0, 8.0, 1080.0)
    assert start == 10.0
    assert end == 10.0


def test_clamp_no_chunk_duration_still_fixes_negative():
    start, end = transcribe_main._clamp_word_timestamps(5.0, -1.0, 0.0)
    assert start == 5.0
    assert end == 5.0


def test_clamp_valid_timestamps_unchanged():
    start, end = transcribe_main._clamp_word_timestamps(5.0, 6.0, 1080.0)
    assert start == 5.0
    assert end == 6.0


def test_clamp_without_chunk_duration():
    """end < start is caught even without chunk_duration."""
    start, end = transcribe_main._clamp_word_timestamps(1853.28, 1080.0)
    assert start == 1853.28
    assert end == 1853.28


def test_clamp_chirp_internal_chunk_boundary_pattern():
    """Chirp returns previous 18-min chunk boundary as end_time for later words.

    Real example: word at 1853.28s (30:53) gets end_time=1080.0 (18:00),
    the boundary of Chirp's internal chunk 1.
    """
    # Chunk 2 word with end_time at chunk 1 boundary
    start, end = transcribe_main._clamp_word_timestamps(1853.28, 1080.0)
    assert end == start

    # Chunk 3 word with end_time at chunk 2 boundary
    start, end = transcribe_main._clamp_word_timestamps(2160.16, 2160.0)
    assert end == start

    # Chunk 4 word with end_time at chunk 3 boundary — end > start so not bogus
    start, end = transcribe_main._clamp_word_timestamps(3239.68, 3240.0)
    assert start == 3239.68
    assert end == 3240.0  # valid: end > start


def test_parse_words_clamps_bogus_timestamps():
    """Chirp 3 bogus end_time=0 should be clamped to start_time before offset."""
    results = [
        SimpleNamespace(
            alternatives=[
                SimpleNamespace(
                    words=[
                        SimpleNamespace(
                            word="テスト",
                            start_offset=_offset(100.0),
                            end_offset=_offset(0.0),  # bogus
                        ),
                        SimpleNamespace(
                            word="正常",
                            start_offset=_offset(101.0),
                            end_offset=_offset(102.0),  # normal
                        ),
                    ]
                )
            ]
        )
    ]
    words = transcribe_main._parse_words(
        results, offset_seconds=1080.0, chunk_duration=1080.0
    )
    # Bogus word: end clamped to start (100.0), then offset applied
    assert words[0].start_time == 1180.0
    assert words[0].end_time == 1180.0
    # Normal word: unchanged except offset
    assert words[1].start_time == 1181.0
    assert words[1].end_time == 1182.0


def test_parse_words_clamps_without_chunk_duration():
    """Clamping works even when chunk_duration is not provided."""
    results = [
        SimpleNamespace(
            alternatives=[
                SimpleNamespace(
                    words=[
                        SimpleNamespace(
                            word="ます",
                            start_offset=_offset(1853.28),
                            end_offset=_offset(1080.0),  # Chirp chunk boundary
                        ),
                    ]
                )
            ]
        )
    ]
    words = transcribe_main._parse_words(results, offset_seconds=0.0)
    assert words[0].start_time == 1853.28
    assert words[0].end_time == 1853.28  # clamped to start


def test_chirp_2_is_a_supported_backend():
    """DEFAULT_TRANSCRIPTION_BACKEND is chirp_3, so listing it left chirp_2 out."""
    from autosub.pipeline.transcribe.main import SUPPORTED_TRANSCRIPTION_BACKENDS

    assert SUPPORTED_TRANSCRIPTION_BACKENDS == {
        "chirp_2",
        "chirp_3",
        "whisperx",
        "gemini",
    }


def test_transcribe_gemini_routes_and_offsets_segments(tmp_path, monkeypatch):
    from autosub.core.schemas import (
        TranscriptionMetadata,
        TranscriptionResult,
        TranscriptionSegment,
    )
    from autosub.pipeline.transcribe import main as transcribe_main

    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    audio_path = tmp_path / "extracted.wav"
    audio_path.write_text("wav", encoding="utf-8")
    out_path = tmp_path / "transcript.json"

    monkeypatch.setattr(transcribe_main, "PROJECT_ID", "test-proj")
    monkeypatch.setattr(transcribe_main.audio, "extract_audio", lambda *a, **k: audio_path)
    monkeypatch.setattr(transcribe_main.audio, "get_audio_duration", lambda *a, **k: 25.0)

    captured: dict = {}

    def fake_gemini_transcribe(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return TranscriptionResult(
            words=[],
            segments=[
                TranscriptionSegment(
                    text="こんにちは。",
                    start_time=1.5,
                    end_time=3.2,
                    words=[],
                    speaker="Tomomi Jiena Sumi",
                    kind="segment",
                )
            ],
            metadata=TranscriptionMetadata(
                backend="gemini", language="ja-JP", model="gemini-4-carbon"
            ),
        )

    monkeypatch.setattr(transcribe_main.gemini_backend, "transcribe_file", fake_gemini_transcribe)

    res = transcribe_main.transcribe(
        video_path,
        out_path,
        transcription_backend="gemini",
        start_time="00:01:00",
        end_time="00:01:25",
        gemini_model="gemini-4-carbon",
        speaker_map={"0": {"name": "Tomomi Jiena Sumi", "character": "An Shiraishi"}},
    )

    assert captured["model_name"] == "gemini-4-carbon"
    assert res.metadata.backend == "gemini"
    assert len(res.segments) == 1
    assert res.segments[0].start_time == 61.5
    assert res.segments[0].end_time == 63.2
    assert res.segments[0].speaker == "Tomomi Jiena Sumi"
    assert out_path.exists()


def test_gemini_backend_prompt_and_corner_dedup():
    from autosub.core.schemas import TranscriptionSegment
    from autosub.pipeline.transcribe import gemini_backend

    corners = [
        {"name": "Opening Talk", "description": "Intro chat", "cues": ["こんばんは"]},
        {"name": "Ending", "description": "Closing", "cues": ["バイバイ"]},
    ]
    prompt = gemini_backend._build_chunk_prompt(
        chunk_idx=1,
        chunk_count=4,
        chunk_start=180.0,
        chunk_end=360.0,
        total_duration=720.0,
        language_code="ja-JP",
        speaker_entries=[{"name": "Kent Ito", "character": "Toya Aoyagi"}],
        num_speakers=2,
        vocabulary=["セカラジ"],
        classify_roles=True,
        corners=corners,
    )
    assert "chunk 2/4" in prompt
    assert "Discourse role (`role` field):" in prompt
    assert 'Program segments (`corner` field):' in prompt
    assert '"Opening Talk"' in prompt

    valid = {"opening talk": "Opening Talk", "ending": "Ending"}
    assert gemini_backend._canonicalize_corner(" opening talk ", valid) == "Opening Talk"
    assert gemini_backend._canonicalize_corner("Invalid Corner", valid) is None

    segs = [
        TranscriptionSegment(text="a", start_time=0.0, end_time=1.0, corner="Opening Talk"),
        TranscriptionSegment(text="b", start_time=1.0, end_time=2.0, corner=None),
        TranscriptionSegment(text="c", start_time=2.0, end_time=3.0, corner="Opening Talk"),
        TranscriptionSegment(text="d", start_time=3.0, end_time=4.0, corner="Ending"),
    ]
    deduped = gemini_backend._dedup_segment_corners(segs)
    assert [s.corner for s in deduped] == ["Opening Talk", None, None, "Ending"]
