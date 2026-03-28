import subprocess
import tempfile
import logging
import ffmpeg
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CHUNK_MINUTES = 18  # stay under Chirp 3's 20-min word-timestamp limit


def extract_audio(
    video_path: Path, start_time: str | None = None, end_time: str | None = None
) -> Path:
    """
    Extracts the audio track from a video file as a 16kHz mono WAV file,
    which is optimized for Chirp 3 API transcription.
    Returns the Path to the temporary audio file.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    temp_dir = Path(tempfile.gettempdir())
    output_audio_path = temp_dir / f"{video_path.stem}_audio.wav"

    try:
        input_args = {}
        if start_time:
            input_args["ss"] = start_time
        if end_time:
            input_args["to"] = end_time

        (
            ffmpeg.input(str(video_path), **input_args)
            .output(
                str(output_audio_path),
                acodec="pcm_s16le",  # 16-bit PCM
                ac=1,  # Mono channel
                ar="16k",  # 16kHz sample rate
                loglevel="error",  # Suppress verbose ffmpeg output
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
        raise RuntimeError(f"Failed to extract audio: {error_message}") from e

    return output_audio_path


def split_audio(
    audio_path: Path, chunk_seconds: float, output_dir: Path
) -> list[tuple[Path, float]]:
    """Split audio into non-overlapping chunks for Chirp 3's word-timestamp limit.

    Returns list of (chunk_path, chunk_start_seconds) tuples.
    """
    duration = get_audio_duration(audio_path)

    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    chunks = []
    start = 0.0
    chunk_idx = 0

    while start < duration:
        chunk_path = output_dir / f"chunk_{chunk_idx:03d}.wav"
        chunk_duration = min(chunk_seconds, duration - start)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(chunk_duration),
                "-acodec", "copy",
                str(chunk_path),
            ],
            capture_output=True,
            check=True,
        )
        logger.info(
            f"  Chunk {chunk_idx}: {start:.0f}s - {start + chunk_duration:.0f}s "
            f"({chunk_duration:.0f}s)"
        )
        chunks.append((chunk_path, start))

        start += chunk_seconds
        chunk_idx += 1

    return chunks


def get_audio_duration(audio_path: Path) -> float:
    """Returns the duration of the audio file in seconds using ffprobe."""
    try:
        probe = ffmpeg.probe(str(audio_path))
        return float(probe["format"]["duration"])
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "Unknown ffprobe error"
        raise RuntimeError(f"Failed to probe audio duration: {error_message}") from e
