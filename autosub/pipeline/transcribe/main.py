import tempfile
from pathlib import Path
import logging
from typing import Any, cast
from uuid import uuid4

from autosub.pipeline.transcribe import audio, gcs, api
from autosub.pipeline.transcribe.audio import MAX_CHUNK_MINUTES
from autosub.core.schemas import TranscribedWord, TranscriptionResult
from autosub.core.config import GCS_BUCKET, PROJECT_ID
from autosub.core.utils import parse_timestamp

logger = logging.getLogger(__name__)


def _duration_seconds(value: Any) -> float:
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(cast(Any, total_seconds()))

    seconds = float(getattr(value, "seconds", 0))
    nanos = float(getattr(value, "nanos", 0))
    return seconds + (nanos / 1_000_000_000)


def transcribe(
    video_path: Path,
    output_json_path: Path,
    language_code: str = "ja-JP",
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    language_codes: list[str] | None = None,
) -> TranscriptionResult:
    """
    End-to-end transcription of a video file:
    1. Extracts audio
    2. Splits into chunks if longer than Chirp 3's word-timestamp limit (~18 min)
    3. Uploads to GCS and calls Chirp 3 BatchRecognize (or sync for < 1min)
    4. Parses results and saves to disk
    """
    if not PROJECT_ID:
        raise ValueError("AUTOSUB_PROJECT_ID is not set in the environment.")

    # Build language_codes list: use explicit list if provided, else wrap single code
    if not language_codes:
        language_codes = [language_code]

    logger.info(f"Extracting audio from {video_path}...")
    audio_path = audio.extract_audio(video_path, start_time, end_time)

    duration = audio.get_audio_duration(audio_path)
    logger.info(f"Audio duration: {duration:.2f} seconds")

    words_data = []
    offset = parse_timestamp(start_time) if start_time else 0.0

    try:
        if duration > 60:
            if not GCS_BUCKET:
                raise ValueError(
                    "AUTOSUB_GCS_BUCKET environment variable must be set for videos longer than 1 minute."
                )

            max_chunk_seconds = MAX_CHUNK_MINUTES * 60

            if duration > max_chunk_seconds:
                # Chunked transcription for long audio
                with tempfile.TemporaryDirectory() as tmp_dir:
                    chunks = audio.split_audio(
                        audio_path, max_chunk_seconds, Path(tmp_dir)
                    )
                    logger.info(
                        f"Split audio into {len(chunks)} chunks "
                        f"({MAX_CHUNK_MINUTES} min each)"
                    )

                    for chunk_idx, (chunk_path, chunk_start) in enumerate(chunks):
                        gcs_dest = (
                            f"autosub_staging/{uuid4()}_{chunk_path.name}"
                        )
                        logger.info(
                            f"  Chunk {chunk_idx + 1}/{len(chunks)}: "
                            f"uploading to {gcs_dest}..."
                        )
                        gcs_uri = gcs.upload_to_gcs(GCS_BUCKET, chunk_path, gcs_dest)

                        try:
                            response = api.transcribe_uri(
                                gcs_uri, PROJECT_ID, language_codes,
                                vocabulary, num_speakers,
                            )
                            words_data.extend(
                                _parse_batch_response(
                                    response, gcs_uri,
                                    time_offset=chunk_start + offset,
                                )
                            )
                        finally:
                            logger.info(f"  Cleaning up {gcs_dest}...")
                            gcs.delete_from_gcs(GCS_BUCKET, gcs_uri)
            else:
                # Single-file GCS workflow
                gcs_dest = f"autosub_staging/{uuid4()}_{audio_path.name}"
                logger.info(f"Uploading audio to {gcs_dest}...")
                gcs_uri = gcs.upload_to_gcs(GCS_BUCKET, audio_path, gcs_dest)

                try:
                    response = api.transcribe_uri(
                        gcs_uri, PROJECT_ID, language_codes,
                        vocabulary, num_speakers,
                    )
                    words_data.extend(
                        _parse_batch_response(response, gcs_uri, time_offset=offset)
                    )
                finally:
                    logger.info(f"Cleaning up GCS staging file {gcs_uri}...")
                    gcs.delete_from_gcs(GCS_BUCKET, gcs_uri)

        else:
            # Fast synchronous workflow (< 1 min)
            with open(audio_path, "rb") as f:
                audio_content = f.read()

            response = api.transcribe_local_file(
                audio_content, PROJECT_ID, language_codes,
                vocabulary, num_speakers,
            )
            for result in response.results:
                for alt in result.alternatives:
                    for w in alt.words:
                        words_data.append(
                            TranscribedWord(
                                word=w.word,
                                start_time=_duration_seconds(w.start_offset) + offset,
                                end_time=_duration_seconds(w.end_offset) + offset,
                                speaker=w.speaker_label
                                if hasattr(w, "speaker_label")
                                else None,
                            )
                        )

    finally:
        # Always clean up the local extracted audio
        if audio_path.exists():
            audio_path.unlink()

    # Serialize to Pydantic and save
    final_result = TranscriptionResult(words=words_data)
    logger.info(
        f"Found {len(words_data)} transcribed words. Saving to {output_json_path}..."
    )

    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(final_result.model_dump_json(indent=2))

    return final_result


def _parse_batch_response(
    response, gcs_uri: str, time_offset: float = 0.0
) -> list[TranscribedWord]:
    """Parse a BatchRecognizeResponse into TranscribedWord objects."""
    words = []
    for result in response.results[gcs_uri].inline_result.transcript.results:
        for alt in result.alternatives:
            for w in alt.words:
                words.append(
                    TranscribedWord(
                        word=w.word,
                        start_time=_duration_seconds(w.start_offset) + time_offset,
                        end_time=_duration_seconds(w.end_offset) + time_offset,
                        speaker=w.speaker_label
                        if hasattr(w, "speaker_label")
                        else None,
                    )
                )
    return words
