from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from autosub.core.ffmpeg_bin import FFMPEG_BIN
from autosub.core.schemas import (
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
)
from autosub.pipeline.transcribe import audio as audio_utils

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_TRANSCRIBE_MODEL = "gemini-4-carbon"
DEFAULT_GEMINI_CHUNK_SECONDS = 60.0
SILENCE_SEARCH_WINDOW_SECONDS = 12.0
MAX_CONCURRENT_GEMINI_CHUNKS = 6
MAX_CHUNK_ATTEMPTS = 3
CHUNK_REQUEST_TIMEOUT_MS = 180_000
VALID_ROLES = {"host", "listener_mail", "host_meta"}


class _UtteranceSchema(BaseModel):
    start_time: float
    end_time: float
    speaker: str
    text: str


class _TranscriptChunkSchema(BaseModel):
    utterances: list[_UtteranceSchema]


class _AnnotatedUtteranceSchema(BaseModel):
    start_time: float
    end_time: float
    speaker: str
    text: str
    role: Literal["host", "listener_mail", "host_meta"] | None = None
    corner: str | None = None


class _AnnotatedTranscriptChunkSchema(BaseModel):
    utterances: list[_AnnotatedUtteranceSchema]


def _detect_silences(audio_path: Path) -> list[tuple[float, float]]:
    """Detect quiet intervals using ffmpeg silencedetect."""
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        "silencedetect=noise=-32dB:d=0.35",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = proc.stderr or ""
    starts: list[float] = []
    intervals: list[tuple[float, float]] = []
    for line in output.splitlines():
        m_start = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m_start:
            starts.append(float(m_start.group(1)))
            continue
        m_end = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m_end and starts:
            intervals.append((starts.pop(0), float(m_end.group(1))))
    return intervals


def _plan_chunk_boundaries(
    duration: float,
    chunk_seconds: float,
    silences: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return (start, end) intervals split on pauses near chunk_seconds multiples."""
    if duration <= chunk_seconds + 15.0:
        return [(0.0, duration)]

    boundaries: list[float] = [0.0]
    cursor = 0.0
    while duration - cursor > chunk_seconds + 15.0:
        target = cursor + chunk_seconds
        best_mid = target
        best_dist = SILENCE_SEARCH_WINDOW_SECONDS + 1.0
        for s_start, s_end in silences:
            mid = 0.5 * (s_start + s_end)
            if mid <= cursor + 30.0 or mid >= duration - 15.0:
                continue
            dist = abs(mid - target)
            if dist <= SILENCE_SEARCH_WINDOW_SECONDS and dist < best_dist:
                best_dist = dist
                best_mid = mid
        boundaries.append(best_mid)
        cursor = best_mid
    boundaries.append(duration)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _extract_mp3_chunk(
    audio_path: Path, start: float, end: float, out_path: Path
) -> None:
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{max(0.1, end - start):.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-loglevel",
        "error",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to encode MP3 chunk [{start:.2f}, {end:.2f}]: {proc.stderr}"
        )


def _speaker_entries(speaker_map: dict[str, dict] | None) -> list[dict[str, str]]:
    if not speaker_map:
        return []
    seen: set[str] = set()
    entries: list[dict[str, str]] = []
    for entry in speaker_map.values():
        name = str(entry.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        entries.append(
            {
                "name": name,
                "character": str(entry.get("character") or "").strip(),
            }
        )
    return entries


def _build_speaker_alias_map(entries: list[dict[str, str]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for entry in entries:
        name = entry["name"]
        aliases[name.lower()] = name
        if entry["character"]:
            aliases[entry["character"].lower()] = name
    return aliases


def _canonicalize_speaker(raw_speaker: str, aliases: dict[str, str]) -> str:
    cleaned = raw_speaker.strip()
    if not cleaned:
        return raw_speaker
    if cleaned.lower() in aliases:
        return aliases[cleaned.lower()]
    for alias_key, canonical in aliases.items():
        if alias_key in cleaned.lower():
            return canonical
    return cleaned


def _canonicalize_corner(
    raw_corner: str | None, valid_corners: dict[str, str]
) -> str | None:
    if not raw_corner or not valid_corners:
        return None
    cleaned = raw_corner.strip()
    if not cleaned:
        return None
    return valid_corners.get(cleaned.lower())


def _dedup_segment_corners(
    segments: list[TranscriptionSegment],
) -> list[TranscriptionSegment]:
    """Collapse consecutive duplicate corner transitions across chunk boundaries."""
    last_corner: str | None = None
    deduped: list[TranscriptionSegment] = []
    for seg in segments:
        if seg.corner is not None and seg.corner == last_corner:
            deduped.append(seg.model_copy(update={"corner": None}))
        else:
            deduped.append(seg)
            if seg.corner is not None:
                last_corner = seg.corner
    return deduped


def _build_chunk_prompt(
    *,
    chunk_idx: int,
    chunk_count: int,
    chunk_start: float,
    chunk_end: float,
    total_duration: float,
    language_code: str,
    speaker_entries: list[dict[str, str]],
    num_speakers: int | None,
    vocabulary: list[str] | None,
    classify_roles: bool = False,
    corners: list[dict] | None = None,
) -> str:
    chunk_duration = chunk_end - chunk_start
    if speaker_entries:
        speaker_lines = []
        allowed_names = []
        for entry in speaker_entries:
            allowed_names.append(f'"{entry["name"]}"')
            desc = (
                f' (voice of {entry["character"]})' if entry["character"] else ""
            )
            speaker_lines.append(f'- "{entry["name"]}"{desc}')
        speaker_block = (
            "Speakers in this recording:\n"
            + "\n".join(speaker_lines)
            + f"\nLabel `speaker` strictly as one of: {', '.join(allowed_names)}.\n"
        )
    elif num_speakers and num_speakers > 0:
        labels = ", ".join(f'"Speaker {i + 1}"' for i in range(num_speakers))
        speaker_block = (
            f"There are up to {num_speakers} speakers. "
            f"Label `speaker` consistently as {labels}.\n"
        )
    else:
        speaker_block = (
            'Label distinct speakers consistently as "Speaker 1", "Speaker 2", etc.\n'
        )

    vocab_block = ""
    if vocabulary:
        terms = ", ".join(vocabulary[:100])
        vocab_block = f"Domain vocabulary / spellings to preserve when spoken: {terms}.\n"

    extra_blocks = ""
    if classify_roles:
        extra_blocks += (
            "\nDiscourse role (`role` field):\n"
            '- "host": host live speech, banter, reactions, or monologue.\n'
            '- "listener_mail": text from a listener letter/email/message being read aloud (use vocal reading cadence + content cues).\n'
            '- "host_meta": host framing around listener mail (introducing the sender/radio name or closing phrases like "といただきました").\n'
            '- If a host echoes/answers a listener greeting immediately afterward, the host reply is "host", not "listener_mail".\n'
        )

    if corners:
        seg_lines = []
        for seg in corners:
            line = f'- "{seg["name"]}": {seg.get("description", "")}'
            if seg.get("cues"):
                line += f" (cues: {', '.join(seg['cues'])})"
            seg_lines.append(line)
        extra_blocks += (
            "\nProgram segments (`corner` field):\n"
            + "\n".join(seg_lines)
            + "\nCorner rules:\n"
            "- Set `corner` to the exact segment name ONLY on the FIRST utterance where the program genuinely transitions into that segment (listening for both spoken cues and audio jingles/BGM shifts).\n"
            "- Set `corner` to null on all continuing lines within an ongoing segment.\n"
            "- Passing mentions or previews of another segment (e.g. mentioning the Ending earlier in the show) are NOT transitions — leave `corner` null.\n"
        )

    return (
        f"Transcribe chunk {chunk_idx + 1}/{chunk_count} "
        f"[{chunk_start:.0f}s - {chunk_end:.0f}s of {total_duration:.0f}s total] "
        f"of this {language_code} audio verbatim into subtitle-sized spoken turns "
        "with accurate timestamps and speaker labels.\n\n"
        f"{speaker_block}\n"
        "Requirements:\n"
        "1. Segment into natural spoken clauses/turns (typically 1-6 seconds each). "
        "Do NOT merge long multi-sentence passages into single blocks.\n"
        f"2. Output `start_time` and `end_time` in float seconds relative to the start of this clip "
        f"(0.00 to {chunk_duration:.2f}), tightly aligned to actual speech onset and offset.\n"
        "3. Combine acoustic voice identity with conversational semantics: a person addressed by name "
        "or asked a question is usually NOT the speaker of that line (the addressee speaks next); "
        "self-introductions identify the speaker; separate quick reactions and overlapping backchannels "
        "onto their own turns with the voice that spoke them.\n"
        "4. Use natural punctuation (for Japanese: 。！？、). "
        "If an instrumental/sung music segment plays with no host talking over it, skip sung song lyrics "
        "and do not hallucinate dialogue.\n"
        f"{vocab_block}{extra_blocks}"
    )


def _call_gemini_stream(
    *,
    client: genai.Client,
    model_name: str,
    audio_bytes: bytes,
    prompt: str,
    chunk_duration: float,
    annotated: bool = False,
) -> list[_AnnotatedUtteranceSchema]:
    schema = _AnnotatedTranscriptChunkSchema if annotated else _TranscriptChunkSchema
    parts: list[str] = []
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        ),
    ):
        if chunk.text:
            parts.append(chunk.text)

    raw_text = "".join(parts).strip()
    if not raw_text:
        raise RuntimeError(
            f"Gemini model '{model_name}' returned an empty transcription response."
        )
    parsed = schema.model_validate_json(raw_text)
    cleaned: list[_AnnotatedUtteranceSchema] = []
    for utt in parsed.utterances:
        text = utt.text.strip()
        if not text:
            continue
        start = max(0.0, min(float(utt.start_time), chunk_duration))
        end = max(0.0, min(float(utt.end_time), chunk_duration))
        if end <= start:
            continue
        cleaned.append(
            _AnnotatedUtteranceSchema(
                start_time=start,
                end_time=end,
                speaker=utt.speaker.strip(),
                text=text,
                role=getattr(utt, "role", None),
                corner=getattr(utt, "corner", None),
            )
        )
    return cleaned


def _transcribe_chunk_with_retries(
    *,
    project_id: str,
    location: str,
    model_name: str,
    chunk_idx: int,
    chunk_count: int,
    chunk_path: Path,
    chunk_start: float,
    chunk_end: float,
    total_duration: float,
    language_code: str,
    speaker_entries: list[dict[str, str]],
    aliases: dict[str, str],
    num_speakers: int | None,
    vocabulary: list[str] | None,
    classify_roles: bool = False,
    corners: list[dict] | None = None,
    valid_corners: dict[str, str] | None = None,
) -> list[TranscriptionSegment]:
    chunk_duration = chunk_end - chunk_start
    annotated = bool(classify_roles or corners)
    prompt = _build_chunk_prompt(
        chunk_idx=chunk_idx,
        chunk_count=chunk_count,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        total_duration=total_duration,
        language_code=language_code,
        speaker_entries=speaker_entries,
        num_speakers=num_speakers,
        vocabulary=vocabulary,
        classify_roles=classify_roles,
        corners=corners,
    )
    audio_bytes = chunk_path.read_bytes()

    last_exc: Exception | None = None
    for attempt in range(1, MAX_CHUNK_ATTEMPTS + 1):
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=types.HttpOptions(timeout=CHUNK_REQUEST_TIMEOUT_MS),
        )
        t0 = time.monotonic()
        try:
            logger.info(
                "Gemini chunk %d/%d [%.1fs - %.1fs] starting (attempt %d/%d)...",
                chunk_idx + 1,
                chunk_count,
                chunk_start,
                chunk_end,
                attempt,
                MAX_CHUNK_ATTEMPTS,
            )
            utterances = _call_gemini_stream(
                client=client,
                model_name=model_name,
                audio_bytes=audio_bytes,
                prompt=prompt,
                chunk_duration=chunk_duration,
                annotated=annotated,
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "Gemini chunk %d/%d finished in %.1fs (%d segments).",
                chunk_idx + 1,
                chunk_count,
                elapsed,
                len(utterances),
            )
            corner_map = valid_corners or {}
            return [
                TranscriptionSegment(
                    text=utt.text,
                    start_time=utt.start_time + chunk_start,
                    end_time=utt.end_time + chunk_start,
                    words=[],
                    speaker=_canonicalize_speaker(utt.speaker, aliases),
                    role=(
                        utt.role
                        if classify_roles and utt.role in VALID_ROLES
                        else None
                    ),
                    corner=_canonicalize_corner(utt.corner, corner_map),
                    confidence=None,
                    kind="segment",
                )
                for utt in utterances
            ]
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Gemini chunk %d/%d failed on attempt %d/%d: %s",
                chunk_idx + 1,
                chunk_count,
                attempt,
                MAX_CHUNK_ATTEMPTS,
                exc,
            )
            if attempt < MAX_CHUNK_ATTEMPTS:
                time.sleep(min(20.0, 4.0 * attempt))
        finally:
            try:
                client.close()
            except Exception:
                logger.debug("Failed to close Vertex client cleanly.", exc_info=True)

    raise RuntimeError(
        f"Gemini chunk {chunk_idx + 1} failed after {MAX_CHUNK_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


def transcribe_file(
    audio_path: Path,
    *,
    project_id: str,
    language_code: str = "ja-JP",
    model_name: str = DEFAULT_GEMINI_TRANSCRIBE_MODEL,
    location: str = "global",
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    speaker_map: dict[str, dict] | None = None,
    chunk_seconds: float = DEFAULT_GEMINI_CHUNK_SECONDS,
    checkpoint_dir: Path | None = None,
    classify_roles: bool = False,
    corners: list[dict] | None = None,
    source_media_path: Path | None = None,
    youtube_url: str | None = None,
    time_offset: float = 0.0,
) -> TranscriptionResult:
    """Transcribe an audio file using a multimodal Gemini model on Vertex AI."""
    duration = audio_utils.get_audio_duration(audio_path)
    silences = _detect_silences(audio_path) if duration > chunk_seconds else []
    planned = _plan_chunk_boundaries(duration, chunk_seconds, silences)
    speaker_entries = _speaker_entries(speaker_map)
    aliases = _build_speaker_alias_map(speaker_entries)
    valid_corners = {
        str(c["name"]).strip().lower(): str(c["name"]).strip()
        for c in (corners or [])
        if c.get("name")
    }

    logger.info(
        "Gemini (%s) transcription planned %d chunk(s) across %.1fs of audio.",
        model_name,
        len(planned),
        duration,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    chunk_segments: dict[int, list[TranscriptionSegment]] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        chunk_jobs: list[tuple[int, float, float, Path]] = []
        for idx, (start, end) in enumerate(planned):
            ckpt_file = (
                checkpoint_dir / f"chunk_{idx:03d}.json" if checkpoint_dir else None
            )
            if ckpt_file and ckpt_file.exists():
                try:
                    cached = json.loads(ckpt_file.read_text(encoding="utf-8"))
                    if (
                        abs(float(cached.get("start", -1)) - start) < 0.05
                        and abs(float(cached.get("end", -1)) - end) < 0.05
                    ):
                        chunk_segments[idx] = [
                            TranscriptionSegment(**s) for s in cached["segments"]
                        ]
                        logger.info(
                            "Gemini chunk %d/%d loaded from checkpoint (%d segments).",
                            idx + 1,
                            len(planned),
                            len(chunk_segments[idx]),
                        )
                        continue
                except Exception:
                    logger.debug("Ignoring stale/corrupt checkpoint %s", ckpt_file)
            mp3_path = tmp_path / f"chunk_{idx:03d}.mp3"
            _extract_mp3_chunk(audio_path, start, end, mp3_path)
            chunk_jobs.append((idx, start, end, mp3_path))

        max_workers = min(MAX_CONCURRENT_GEMINI_CHUNKS, max(1, len(chunk_jobs)))
        if chunk_jobs:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _transcribe_chunk_with_retries,
                        project_id=project_id,
                        location=location,
                        model_name=model_name,
                        chunk_idx=idx,
                        chunk_count=len(planned),
                        chunk_path=mp3_path,
                        chunk_start=start,
                        chunk_end=end,
                        total_duration=duration,
                        language_code=language_code,
                        speaker_entries=speaker_entries,
                        aliases=aliases,
                        num_speakers=num_speakers,
                        vocabulary=vocabulary,
                        classify_roles=classify_roles,
                        corners=corners,
                        valid_corners=valid_corners,
                    ): (idx, start, end)
                    for idx, start, end, mp3_path in chunk_jobs
                }
                for future in as_completed(futures):
                    idx, start, end = futures[future]
                    segs = future.result()
                    chunk_segments[idx] = segs
                    if checkpoint_dir is not None:
                        ckpt_file = checkpoint_dir / f"chunk_{idx:03d}.json"
                        ckpt_file.write_text(
                            json.dumps(
                                {
                                    "start": start,
                                    "end": end,
                                    "segments": [s.model_dump() for s in segs],
                                },
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )

    all_segments: list[TranscriptionSegment] = []
    for idx in sorted(chunk_segments):
        all_segments.extend(chunk_segments[idx])
    all_segments.sort(key=lambda seg: (seg.start_time, seg.end_time))
    if corners:
        all_segments = _dedup_segment_corners(all_segments)

    from autosub.pipeline.transcribe.align import (
        align_segments_to_vtt,
        ensure_companion_vtt,
        snap_segments_to_waveform_onsets,
    )

    lookup_path = source_media_path or audio_path
    companion_vtt = ensure_companion_vtt(
        lookup_path,
        youtube_url=youtube_url,
        lang=language_code,
    )
    if companion_vtt is not None:
        all_segments = align_segments_to_vtt(
            all_segments,
            companion_vtt,
            chunk_boundaries=planned,
            time_offset=time_offset,
        )
    else:
        all_segments = snap_segments_to_waveform_onsets(all_segments, audio_path)

    if checkpoint_dir is not None and checkpoint_dir.exists():
        for p in checkpoint_dir.glob("chunk_*.json"):
            p.unlink(missing_ok=True)
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass

    all_words = [w for seg in all_segments for w in seg.words]
    return TranscriptionResult(
        words=all_words,
        segments=all_segments,
        metadata=TranscriptionMetadata(
            backend="gemini",
            language=language_code,
            model=model_name,
        ),
    )
