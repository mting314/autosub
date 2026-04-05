from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any
import warnings

from autosub.core.schemas import (
    TranscribedWord,
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)


def _load_whisperx_module() -> Any:
    try:
        warnings.filterwarnings(
            "ignore",
            message=r".*torchcodec is not installed correctly so built-in audio decoding will fail\..*",
            category=UserWarning,
        )
        return importlib.import_module("whisperx")
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX transcription requires the optional `whisperx` package. "
            "Install it before using `--backend whisperx`."
        ) from exc


def _normalize_language_code(language_code: str) -> str:
    return language_code.split("-", 1)[0].lower()


def _resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    for env_var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def _word_from_mapping(
    word_info: dict[str, Any], segment_speaker: str | None = None
) -> TranscribedWord | None:
    word = str(word_info.get("word") or "").strip()
    start_time = word_info.get("start")
    end_time = word_info.get("end")
    if not word or start_time is None or end_time is None:
        return None

    return TranscribedWord(
        word=word,
        start_time=float(start_time),
        end_time=float(end_time),
        speaker=word_info.get("speaker") or segment_speaker,
        confidence=(
            float(word_info["score"]) if word_info.get("score") is not None else None
        ),
    )


def _extract_transcription_segments(
    result: dict[str, Any],
) -> tuple[list[TranscriptionSegment], list[TranscribedWord]]:
    segments: list[TranscriptionSegment] = []
    words: list[TranscribedWord] = []

    for segment in result.get("segments", []):
        segment_speaker = segment.get("speaker")
        segment_words: list[TranscribedWord] = []
        for word_info in segment.get("words", []):
            parsed = _word_from_mapping(word_info, segment_speaker=segment_speaker)
            if parsed is not None:
                segment_words.append(parsed)
                words.append(parsed)
        segment_start = segment.get("start")
        segment_end = segment.get("end")
        if segment_start is None or segment_end is None:
            continue
        segments.append(
            TranscriptionSegment(
                text=str(segment.get("text") or "").strip(),
                start_time=float(segment_start),
                end_time=float(segment_end),
                words=segment_words,
                speaker=segment_speaker,
                confidence=(
                    float(segment["avg_logprob"])
                    if segment.get("avg_logprob") is not None
                    else None
                ),
                kind="sentence",
            )
        )

    if words:
        return segments, words

    for word_info in result.get("word_segments", []):
        parsed = _word_from_mapping(word_info)
        if parsed is not None:
            words.append(parsed)

    return segments, words


def _with_metadata(
    *,
    words: list[TranscribedWord],
    segments: list[TranscriptionSegment],
    language: str,
    model_name: str,
) -> TranscriptionResult:
    return TranscriptionResult(
        words=words,
        segments=segments,
        metadata=TranscriptionMetadata(
            backend="whisperx",
            language=language,
            model=model_name,
        ),
    )


def transcribe_file(
    audio_path: Path,
    *,
    language_code: str = "ja-JP",
    model_name: str = "large-v2",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 16,
    diarize: bool = False,
    hf_token: str | None = None,
    num_speakers: int | None = None,
) -> TranscriptionResult:
    whisperx = _load_whisperx_module()
    whisper_language = _normalize_language_code(language_code)

    logger.info(
        "Running WhisperX transcription with model=%s device=%s compute_type=%s",
        model_name,
        device,
        compute_type,
    )
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=whisper_language,
    )
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=whisper_language)

    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", whisper_language),
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if diarize:
        resolved_hf_token = _resolve_hf_token(hf_token)
        if not resolved_hf_token:
            raise RuntimeError(
                "WhisperX diarization requires a Hugging Face token. Set "
                "`HF_TOKEN`/`HUGGINGFACE_TOKEN` or pass `--whisper-hf-token`."
            )

        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=resolved_hf_token,
            device=device,
        )
        diarize_kwargs: dict[str, int] = {}
        if num_speakers is not None:
            diarize_kwargs["min_speakers"] = num_speakers
            diarize_kwargs["max_speakers"] = num_speakers
        diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
        aligned = whisperx.assign_word_speakers(diarize_segments, aligned)

    segments, words = _extract_transcription_segments(aligned)
    if not words:
        raise RuntimeError(
            "WhisperX completed but did not return any aligned word timestamps."
        )
    return _with_metadata(
        words=words,
        segments=segments,
        language=str(
            aligned.get("language") or result.get("language") or whisper_language
        ),
        model_name=model_name,
    )
