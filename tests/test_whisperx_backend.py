from pathlib import Path
from types import SimpleNamespace

from autosub.pipeline.transcribe import whisperx_backend


def test_transcribe_file_passes_language_to_load_model_and_transcribe(monkeypatch):
    captured: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, audio, batch_size, language):
            captured["transcribe_audio"] = audio
            captured["transcribe_batch_size"] = batch_size
            captured["transcribe_language"] = language
            return {
                "language": language,
                "segments": [{"words": [{"word": "hello", "start": 1.0, "end": 2.0}]}],
            }

    def fake_load_model(model_name, device, compute_type, language):
        captured["load_model_name"] = model_name
        captured["load_model_device"] = device
        captured["load_model_compute_type"] = compute_type
        captured["load_model_language"] = language
        return FakeModel()

    def fake_load_audio(audio_path):
        captured["load_audio_path"] = audio_path
        return "audio-bytes"

    def fake_load_align_model(language_code, device):
        captured["align_language_code"] = language_code
        captured["align_device"] = device
        return "align-model", "align-metadata"

    def fake_align(
        segments, align_model, metadata, audio, device, return_char_alignments
    ):
        captured["align_segments"] = segments
        captured["align_model"] = align_model
        captured["align_metadata"] = metadata
        captured["align_audio"] = audio
        captured["align_device_call"] = device
        captured["align_return_chars"] = return_char_alignments
        return {"segments": segments}

    fake_whisperx = SimpleNamespace(
        load_model=fake_load_model,
        load_audio=fake_load_audio,
        load_align_model=fake_load_align_model,
        align=fake_align,
    )
    monkeypatch.setattr(
        whisperx_backend, "_load_whisperx_module", lambda: fake_whisperx
    )

    words = whisperx_backend.transcribe_file(
        Path("audio.wav"),
        language_code="ja-JP",
        model_name="large-v2",
        device="cuda",
        compute_type="float16",
        batch_size=8,
    )

    assert captured["load_model_language"] == "ja"
    assert captured["transcribe_language"] == "ja"
    assert captured["align_language_code"] == "ja"
    assert words[0].word == "hello"
    assert words[0].start_time == 1.0
    assert words[0].end_time == 2.0
