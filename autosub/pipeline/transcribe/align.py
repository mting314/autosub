from __future__ import annotations

import array
import difflib
import glob
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess

from autosub.core.ffmpeg_bin import FFMPEG_BIN
from autosub.core.schemas import (
    SubtitleCue,
    SubtitleDocument,
    TranscribedWord,
    TranscriptionSegment,
)
from autosub.core.speaker_map import build_slot_lookup
from autosub.pipeline.format.generator import render_ass_document
from autosub.pipeline.format.timing import (
    _close_screen_gaps,
    check_display_invariants,
    slot_key_for_speaker,
)
from autosub.pipeline.postprocess.main import (
    _CueSpan,
    _repair_cross_slot_turn_overlaps,
)

logger = logging.getLogger(__name__)

_VTT_HEADER_RE = re.compile(
    r"(\d\d:\d\d:\d\d\.\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d\.\d\d\d)"
)
_VTT_INLINE_TAG_RE = re.compile(r"<(\d\d:\d\d:\d\d\.\d\d\d)><c>(.*?)</c>")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_STRIP_PUNCT_RE = re.compile(r"\[.*?\]|[、。！？!?「」『』（）()・…\s〜ー\-]")
_YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)
_YOUTUBE_ID_ONLY_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_NOTES_VIDEO_ID_RE = re.compile(
    r"(?:Video ID|video_id)\**\s*[:=]\s*`?([A-Za-z0-9_-]{11})`?", re.IGNORECASE
)
_BRACKET_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]$")


def _parse_vtt_ts(ts: str) -> float:
    h, m, rest = ts.split(":")
    sec, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def to_phonetic_hiragana(text: str) -> str:
    """Normalize Japanese text to hiragana + lowercase ASCII without punctuation."""
    cleaned = _STRIP_PUNCT_RE.sub("", text or "")
    out: list[str] = []
    for ch in cleaned:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch.lower())
    return "".join(out)


def _base_stem(media_or_doc_path: Path) -> str:
    stem = media_or_doc_path.stem
    for suffix in (
        "_postprocessed",
        "_translated",
        "_formatted",
        "_transcript",
        "_final",
        "_original",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_companion_vtt(media_or_doc_path: Path) -> Path | None:
    """Discover a companion YouTube auto-caption ``.vtt`` beside a video or subtitle file."""
    parent = media_or_doc_path.parent
    if not parent.is_dir():
        return None
    stem = _base_stem(media_or_doc_path)

    candidates = [
        parent / f"{stem}.auto-captions.ja.vtt",
        parent / f"{stem}.ja.vtt",
        parent / f"{stem}.vtt",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    # Only accept captions named after this media/doc stem; never fall back to an
    # arbitrary .vtt in the folder (it may belong to a different episode).
    vtts = sorted(parent.glob(f"{glob.escape(stem)}*.vtt"))
    return vtts[0] if vtts else None


def resolve_youtube_url(
    media_or_doc_path: Path | None = None,
    explicit_url: str | None = None,
) -> str | None:
    """Resolve a YouTube watch URL from an explicit URL/ID, project notes.md, or filename."""
    if explicit_url and explicit_url.strip():
        raw = explicit_url.strip()
        if _YOUTUBE_ID_ONLY_RE.match(raw):
            return f"https://www.youtube.com/watch?v={raw}"
        m = _YOUTUBE_URL_RE.search(raw)
        if m:
            return f"https://www.youtube.com/watch?v={m.group(1)}"
        return raw

    if media_or_doc_path is None:
        return None

    m_br = _BRACKET_ID_RE.search(_base_stem(media_or_doc_path))
    if m_br:
        return f"https://www.youtube.com/watch?v={m_br.group(1)}"

    parent = media_or_doc_path.parent
    if parent.is_dir():
        for doc_name in ("notes.md", "project.toml", "README.md"):
            doc_file = parent / doc_name
            if not doc_file.is_file():
                continue
            try:
                content = doc_file.read_text(encoding="utf-8")
            except OSError:
                continue
            m_url = _YOUTUBE_URL_RE.search(content)
            if m_url:
                return f"https://www.youtube.com/watch?v={m_url.group(1)}"
            m_id = _NOTES_VIDEO_ID_RE.search(content)
            if m_id:
                return f"https://www.youtube.com/watch?v={m_id.group(1)}"

    return None


def fetch_youtube_vtt(
    target_dir: Path,
    stem: str,
    youtube_url: str,
    *,
    lang: str = "ja",
) -> Path | None:
    """Download YouTube auto-captions (`.vtt`) via `yt-dlp` without downloading the video."""
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        logger.warning("yt-dlp not found in PATH; cannot fetch YouTube VTT captions.")
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    sub_lang = lang.split("-")[0].lower() if lang else "ja"
    out_tmpl = str(target_dir / f"{stem}.auto-captions")
    cmd = [
        yt_dlp,
        "--no-update",
        "--no-playlist",
        "--skip-download",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang",
        sub_lang,
        "--sub-format",
        "vtt",
        "-o",
        out_tmpl,
        youtube_url,
    ]
    logger.info("Fetching YouTube %s VTT captions via yt-dlp from %s...", sub_lang, youtube_url)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        logger.warning(
            "yt-dlp VTT fetch failed (exit %d): %s",
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip()[-300:],
        )
        return None

    written = sorted(target_dir.glob(f"{glob.escape(stem)}.auto-captions*.vtt"))
    if not written:
        logger.warning("yt-dlp exited cleanly but wrote no %s captions for %s.", sub_lang, youtube_url)
        return None
    return written[0]


def ensure_companion_vtt(
    media_or_doc_path: Path,
    *,
    youtube_url: str | None = None,
    lang: str = "ja",
) -> Path | None:
    """Return an existing companion `.vtt` or fetch it from YouTube via `yt-dlp`."""
    existing = find_companion_vtt(media_or_doc_path)
    if existing is not None:
        return existing

    resolved_url = resolve_youtube_url(media_or_doc_path, explicit_url=youtube_url)
    if not resolved_url:
        return None

    return fetch_youtube_vtt(
        media_or_doc_path.parent,
        _base_stem(media_or_doc_path),
        resolved_url,
        lang=lang,
    )


def parse_vtt_words(vtt_path: Path) -> list[TranscribedWord]:
    """Extract word-level ``TranscribedWord`` timestamps from a YouTube VTT file."""
    vtt_text = vtt_path.read_text(encoding="utf-8")
    raw_tokens: list[tuple[float, float, str]] = []
    last_untagged_line: str | None = None

    for block in vtt_text.split("\n\n"):
        m_hdr = _VTT_HEADER_RE.search(block)
        if not m_hdr:
            continue
        bst = _parse_vtt_ts(m_hdr.group(1))
        ben = _parse_vtt_ts(m_hdr.group(2))
        if ben - bst < 0.05:
            continue
        lines = [
            line
            for line in block.splitlines()
            if "-->" not in line
            and not line.startswith(("WEBVTT", "Kind:", "Language:"))
        ]
        if not lines:
            continue
        last_line = lines[-1]
        if "<c>" in last_line:
            last_untagged_line = None
            block_tokens: list[tuple[float, str]] = []
            first_part = re.split(r"<\d\d:\d\d:\d\d\.\d\d\d>", last_line)[0]
            first_tok = _HTML_TAG_RE.sub("", first_part).strip()
            if first_tok:
                block_tokens.append((bst, first_tok))
            for tm, tok in _VTT_INLINE_TAG_RE.findall(last_line):
                cleaned_tok = _HTML_TAG_RE.sub("", tok).strip()
                if cleaned_tok:
                    block_tokens.append((_parse_vtt_ts(tm), cleaned_tok))
            for k, (t_tok, tok_str) in enumerate(block_tokens):
                t_next = block_tokens[k + 1][0] if k + 1 < len(block_tokens) else ben
                raw_tokens.append((t_tok, max(t_tok + 0.04, t_next), tok_str))
        else:
            cleaned_line = _HTML_TAG_RE.sub("", last_line).strip()
            if cleaned_line and cleaned_line != last_untagged_line:
                last_untagged_line = cleaned_line
                sub_toks = (
                    re.findall(r"[^、。！？!?…\s][、。！？!?…]*", cleaned_line)
                    or [cleaned_line]
                )
                span_end = min(ben, bst + max(0.45, 0.22 * len(cleaned_line)))
                span_dur = max(0.06, span_end - bst)
                weights = [
                    1.0
                    + (
                        0.65
                        if any(ch in st for ch in "。！？!?…")
                        else (0.35 if "、" in st else 0.0)
                    )
                    for st in sub_toks
                ]
                total_w = sum(weights) or 1.0
                cursor = bst
                for st, w in zip(sub_toks, weights):
                    dur = span_dur * (w / total_w)
                    raw_tokens.append((cursor, cursor + dur, st))
                    cursor += dur

    words: list[TranscribedWord] = []
    for idx, (t_val, t_max_end, tok) in enumerate(raw_tokens):
        next_t = raw_tokens[idx + 1][0] if idx + 1 < len(raw_tokens) else t_max_end
        cap_dur = max(0.45, 0.22 * len(tok))
        end_t = min(t_val + cap_dur, t_max_end, max(t_val + 0.04, next_t))
        words.append(
            TranscribedWord(
                word=tok,
                start_time=round(t_val, 3),
                end_time=round(end_t, 3),
            )
        )
    return words


def _align_items_in_windows(
    texts: list[str],
    starts: list[float],
    ends: list[float],
    speakers: list[str | None],
    vtt_words: list[TranscribedWord],
    windows: list[tuple[float, float]],
) -> tuple[list[float], list[float], list[list[TranscribedWord]]]:
    """Phonetic sequence alignment + piecewise-monotone time warp within windows."""
    vtt_chars: list[tuple[float, float, str, int]] = []
    for w_idx, w in enumerate(vtt_words):
        h = to_phonetic_hiragana(w.word)
        if not h:
            continue
        dur = max(0.06, w.end_time - w.start_time)
        for k, ch in enumerate(h):
            f0 = k / len(h)
            f1 = (k + 1) / len(h)
            vtt_chars.append(
                (w.start_time + f0 * dur, w.start_time + f1 * dur, ch, w_idx)
            )

    new_starts = list(starts)
    new_ends = list(ends)
    item_words: list[list[TranscribedWord]] = [[] for _ in texts]

    if not vtt_chars or not texts:
        return new_starts, new_ends, item_words

    for cs, ce in windows:
        c_indices = [
            i for i, st in enumerate(starts) if cs - 0.05 <= st < ce + 0.05
        ]
        if not c_indices:
            continue

        cue_chars: list[tuple[int, str]] = []
        cue_spans: dict[int, tuple[int, int]] = {}
        for ci in c_indices:
            h = to_phonetic_hiragana(texts[ci])
            s_pos = len(cue_chars)
            # Exclude 1-3 char backchannels (はい, うん, ああ) from SequenceMatcher
            # so overlapping backchannels cannot steal false matches from main speech.
            if len(h) >= 4:
                for ch in h:
                    cue_chars.append((ci, ch))
            cue_spans[ci] = (s_pos, len(cue_chars))
        cue_str = "".join(ch for _, ch in cue_chars)

        v_indices = [
            j for j, vc in enumerate(vtt_chars) if cs - 2.0 <= vc[0] <= ce + 2.0
        ]
        v_sub = "".join(vtt_chars[j][2] for j in v_indices)

        sm = difflib.SequenceMatcher(None, cue_str, v_sub, autojunk=False)
        cue_char_to_vtt: dict[int, int] = {}
        for a, b, size in sm.get_matching_blocks():
            if size >= 3:
                for k in range(size):
                    cue_char_to_vtt[a + k] = v_indices[b + k]

        # Build monotone anchor curve strictly from onset points (gt_start, vt_start)
        # so a long outer segment's end_time never jumps ahead of an inner backchannel.
        anchors: list[tuple[float, float]] = [(cs, cs)]
        for ci in c_indices:
            sp, ep_pos = cue_spans[ci]
            if ep_pos - sp < 4:
                continue
            first_hits = [
                cue_char_to_vtt[p]
                for p in range(sp, min(ep_pos, sp + 3))
                if p in cue_char_to_vtt
            ]
            if first_hits:
                vt = vtt_chars[first_hits[0]][0]
                gt = starts[ci]
                if (
                    abs(gt - vt) < 12.0
                    and vt > anchors[-1][1]
                    and gt > anchors[-1][0]
                ):
                    anchors.append((gt, vt))
        if ce > anchors[-1][0] and ce > anchors[-1][1]:
            anchors.append((ce, ce))

        def warp(t: float) -> float:
            if t <= anchors[0][0]:
                return anchors[0][1] + (t - anchors[0][0])
            for (g0, v0), (g1, v1) in zip(anchors[:-1], anchors[1:]):
                if g0 <= t <= g1:
                    if g1 == g0:
                        return v0
                    return v0 + (t - g0) * (v1 - v0) / (g1 - g0)
            g0, v0 = anchors[-1]
            return v0 + (t - g0)

        for ci in c_indices:
            orig_dur = max(0.45, ends[ci] - starts[ci])
            sp, ep_pos = cue_spans[ci]
            h_len = max(1, ep_pos - sp)
            w_st = warp(starts[ci])
            w_en = warp(ends[ci])
            hit_positions = [p for p in range(sp, ep_pos) if p in cue_char_to_vtt]
            hits = [cue_char_to_vtt[p] for p in hit_positions]
            # Last matched position within the cue text (not the VTT char index).
            last_hit_pos = hit_positions[-1] if hit_positions else sp
            if hits and (
                sp in cue_char_to_vtt
                or (sp + 1) in cue_char_to_vtt
                or (sp + 2) in cue_char_to_vtt
            ):
                cand_st = vtt_chars[hits[0]][0]
                if abs(cand_st - w_st) <= 2.5:
                    w_st = cand_st
            if hits and (
                (ep_pos - 1) in cue_char_to_vtt
                or (ep_pos - 2) in cue_char_to_vtt
                or (last_hit_pos - sp >= max(3, int(0.55 * h_len)))
            ):
                rem_chars = max(0, (ep_pos - 1) - last_hit_pos)
                cand_en = vtt_chars[hits[-1]][1] + rem_chars * 0.12
                cand_dur = cand_en - w_st
                natural_dur = 0.04 * h_len <= cand_dur <= max(2.5, 0.30 * h_len)
                if cand_en > w_st + 0.15 and (
                    abs(cand_en - w_en) <= 2.5 or natural_dur
                ):
                    w_en = cand_en
            if w_en <= w_st + 0.25:
                w_en = w_st + max(min(orig_dur, 2.5), min(4.0, 0.11 * h_len))

            raw_st = round(max(0.0, w_st), 2)
            raw_en = round(max(raw_st + 0.25, w_en), 2)
            new_starts[ci] = raw_st
            new_ends[ci] = raw_en

            matched_w_indices = sorted({vtt_chars[h][3] for h in hits})
            if matched_w_indices:
                w_list = [
                    vtt_words[w_i].model_copy(update={"speaker": speakers[ci]})
                    for w_i in matched_w_indices
                    if raw_st - 0.3 <= vtt_words[w_i].start_time <= raw_en + 0.3
                ]
            else:
                w_list = []
            if not w_list:
                tok = (texts[ci] or "…").strip()
                w_list = [
                    TranscribedWord(
                        word=tok,
                        start_time=raw_st,
                        end_time=raw_en,
                        speaker=speakers[ci],
                    )
                ]
            item_words[ci] = w_list

    return new_starts, new_ends, item_words


def _default_windows(
    starts: list[float], ends: list[float], window_seconds: float = 180.0
) -> list[tuple[float, float]]:
    if not starts:
        return []
    max_t = max(ends) if ends else max(starts)
    windows: list[tuple[float, float]] = []
    cur = 0.0
    while cur < max_t + 0.1:
        windows.append((cur, cur + window_seconds))
        cur += window_seconds
    return windows


def align_segments_to_vtt(
    segments: list[TranscriptionSegment],
    vtt_path: Path,
    chunk_boundaries: list[tuple[float, float]] | None = None,
    *,
    time_offset: float = 0.0,
) -> list[TranscriptionSegment]:
    """Phonetic-align ``TranscriptionSegment`` items against a YouTube ``.vtt`` file."""
    if not segments:
        return []
    vtt_words = parse_vtt_words(vtt_path)
    if not vtt_words:
        return list(segments)

    if time_offset != 0.0:
        shifted_words: list[TranscribedWord] = []
        for w in vtt_words:
            if w.end_time <= time_offset:
                continue
            shifted_words.append(
                w.model_copy(
                    update={
                        "start_time": round(max(0.0, w.start_time - time_offset), 3),
                        "end_time": round(max(0.06, w.end_time - time_offset), 3),
                    }
                )
            )
        vtt_words = shifted_words
        if not vtt_words:
            return list(segments)

    texts = [s.text for s in segments]
    starts = [s.start_time for s in segments]
    ends = [s.end_time for s in segments]
    speakers = [s.speaker for s in segments]
    windows = chunk_boundaries or _default_windows(starts, ends)

    new_starts, new_ends, seg_words = _align_items_in_windows(
        texts, starts, ends, speakers, vtt_words, windows
    )
    aligned: list[TranscriptionSegment] = []
    for seg, st, en, words in zip(segments, new_starts, new_ends, seg_words):
        aligned.append(
            seg.model_copy(
                update={
                    "start_time": st,
                    "end_time": en,
                    "words": words,
                }
            )
        )
    logger.info(
        "Aligned %d segment(s) against VTT word timeline (%s).",
        len(aligned),
        vtt_path.name,
    )
    return aligned


def snap_segments_to_waveform_onsets(
    segments: list[TranscriptionSegment],
    audio_path: Path,
    *,
    max_early_shift: float = 0.85,
    silence_db: float = -41.0,
    speech_db: float = -33.0,
) -> list[TranscriptionSegment]:
    """Snap late LLM segment start times to physical PCM speech onsets after pauses."""
    if not segments or not audio_path.is_file():
        return list(segments)

    cmd = [
        FFMPEG_BIN,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "1000",
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        logger.debug("Waveform decode skipped for onset snapping: %s", audio_path)
        return list(segments)

    pcm = array.array("h", proc.stdout)
    frame_samples = 20  # 20ms @ 1000 Hz (50 frames/sec)
    db_frames: list[float] = []
    for off in range(0, len(pcm) - frame_samples + 1, frame_samples):
        window = pcm[off : off + frame_samples]
        rms = math.sqrt(sum(x * x for x in window) / frame_samples) / 32768.0
        db_frames.append(20.0 * math.log10(max(rms, 1e-5)))

    snapped: list[TranscriptionSegment] = []
    prev_end = 0.0
    for seg in segments:
        st = seg.start_time
        en = seg.end_time
        search_floor = max(prev_end, st - max_early_shift)
        f_start = max(0, int(search_floor * 50))
        f_pred = min(len(db_frames) - 1, int(st * 50))

        # Look for a quiet frame followed by a clear speech onset before the predicted start
        best_st = st
        saw_silence = False
        for f_idx in range(f_start, f_pred + 1):
            db = db_frames[f_idx]
            if db <= silence_db:
                saw_silence = True
            elif saw_silence and db >= speech_db:
                best_st = round(f_idx / 50.0, 2)
                break

        shift = st - best_st
        new_st = best_st
        # If the whole short utterance was shifted late into a silence gap, pull end_time too
        if shift > 0.15 and (en - st) <= 2.5:
            new_en = round(max(new_st + 0.35, en - shift), 2)
        else:
            new_en = round(max(new_st + 0.35, en), 2)

        prev_end = new_en
        words = list(seg.words) if seg.words else [
            TranscribedWord(
                word=seg.text.strip() or "…",
                start_time=new_st,
                end_time=new_en,
                speaker=seg.speaker,
            )
        ]
        snapped.append(
            seg.model_copy(
                update={
                    "start_time": new_st,
                    "end_time": new_en,
                    "words": words,
                }
            )
        )
    return snapped


def align_subtitle_document(
    target_json_path: Path,
    *,
    media_path: Path | None = None,
    ass_path: Path | None = None,
    vtt_path: Path | None = None,
    youtube_url: str | None = None,
    speaker_map: dict[str, dict] | None = None,
    min_duration_ms: int = 500,
    lead_out_ms: int = 500,
) -> SubtitleDocument:
    """Re-align a SubtitleDocument (and optional .ass) against YouTube VTT or waveform timing."""
    doc = SubtitleDocument.model_validate_json(
        target_json_path.read_text(encoding="utf-8")
    )
    cues = doc.cues
    if not cues:
        return doc

    # Preserve any manual Aegisub edits from the .ass file when counts match 1-to-1
    if ass_path is not None and ass_path.is_file():
        ass_dialogues = [
            line
            for line in ass_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Dialogue:")
        ]
        if len(ass_dialogues) == len(cues):
            for cue, dline in zip(cues, ass_dialogues):
                txt = dline.split(",", 9)[9]
                cue.final_text = txt
                cue.translated_text = txt

    resolved_vtt = (
        vtt_path
        or (
            ensure_companion_vtt(media_path, youtube_url=youtube_url)
            if media_path
            else None
        )
        or ensure_companion_vtt(target_json_path, youtube_url=youtube_url)
    )

    # Group split sibling cues (cue-XXXXXXXX-1, cue-XXXXXXXX-2, ...) by base ID
    # and partition their parent Japanese source_text proportionally by English length,
    # even when another speaker's backchannel sits between the split siblings.
    eff_source = [c.source_text or "" for c in cues]
    split_groups: dict[str, list[int]] = {}
    for idx, c in enumerate(cues):
        m_id = re.match(r"^(cue-\d+)-\d+$", c.id)
        if m_id:
            split_groups.setdefault(m_id.group(1), []).append(idx)

    for idxs in split_groups.values():
        if len(idxs) > 1 and eff_source[idxs[0]]:
            full_jp = eff_source[idxs[0]]
            weights = [
                max(1, len((cues[i].final_text or cues[i].translated_text or "").strip()))
                for i in idxs
            ]
            total_w = sum(weights)
            cursor = 0
            for k, i in enumerate(idxs):
                if k == len(idxs) - 1:
                    eff_source[i] = full_jp[cursor:]
                else:
                    take = max(1, round(len(full_jp) * weights[k] / total_w))
                    eff_source[i] = full_jp[cursor : cursor + take]
                    cursor = min(len(full_jp) - 1, cursor + take)

    for i in range(len(cues) - 1):
        if eff_source[i] and not eff_source[i + 1] and cues[i].speaker == cues[i + 1].speaker:
            w0 = max(1, len((cues[i].final_text or cues[i].translated_text or "").strip()))
            w1 = max(1, len((cues[i + 1].final_text or cues[i + 1].translated_text or "").strip()))
            split_idx = max(1, min(len(eff_source[i]) - 1, round(len(eff_source[i]) * w0 / (w0 + w1))))
            full = eff_source[i]
            eff_source[i] = full[:split_idx]
            eff_source[i + 1] = full[split_idx:]

    internal_split_first_indices = {
        i for idxs in split_groups.values() if len(idxs) > 1 for i in idxs[:-1]
    }
    lead_out_s = lead_out_ms / 1000.0
    min_dur_s = min_duration_ms / 1000.0

    if resolved_vtt is not None and resolved_vtt.is_file():
        vtt_words = parse_vtt_words(resolved_vtt)
        starts = [c.start_time for c in cues]
        ends = [c.end_time for c in cues]
        speakers = [c.speaker for c in cues]
        windows = _default_windows(starts, ends, window_seconds=180.0)
        new_starts, new_ends, item_words = _align_items_in_windows(
            eff_source, starts, ends, speakers, vtt_words, windows
        )
        for idx, (cue, st, en, words) in enumerate(
            zip(cues, new_starts, new_ends, item_words)
        ):
            ext = 0.0 if idx in internal_split_first_indices else lead_out_s
            cue.start_time = st
            cue.end_time = round(max(st + min_dur_s, en + ext), 2)
            cue.words = words
        for idxs in split_groups.values():
            if len(idxs) > 1:
                for i_cur, i_nxt in zip(idxs[:-1], idxs[1:]):
                    if cues[i_nxt].start_time > cues[i_cur].start_time + min_dur_s:
                        cues[i_cur].end_time = cues[i_nxt].start_time
    elif media_path is not None and media_path.is_file():
        temp_segs = [
            TranscriptionSegment(
                text=src or c.final_text or "",
                start_time=c.start_time,
                end_time=c.end_time,
                speaker=c.speaker,
                words=list(c.words),
            )
            for c, src in zip(cues, eff_source)
        ]
        snapped = snap_segments_to_waveform_onsets(temp_segs, media_path)
        for cue, seg in zip(cues, snapped):
            cue.start_time = seg.start_time
            cue.end_time = round(
                max(seg.start_time + min_dur_s, seg.end_time + lead_out_s), 2
            )
            cue.words = list(seg.words)

    slot_lookup = build_slot_lookup(speaker_map)
    by_slot: dict[str | None, list[SubtitleCue]] = {}
    for c in cues:
        by_slot.setdefault(slot_key_for_speaker(c.speaker, slot_lookup), []).append(c)

    for scues in by_slot.values():
        scues.sort(key=lambda x: (x.start_time, x.end_time))
        for i in range(len(scues) - 1):
            a, b = scues[i], scues[i + 1]
            if b.start_time < a.start_time + min_dur_s:
                b.start_time = round(a.start_time + min_dur_s, 2)
                if b.end_time < b.start_time + min_dur_s:
                    b.end_time = round(b.start_time + min_dur_s, 2)
            if a.end_time > b.start_time or 0 < (b.start_time - a.end_time) < min_dur_s:
                a.end_time = b.start_time

    cues.sort(key=lambda x: (x.start_time, x.end_time))
    _repair_cross_slot_turn_overlaps(
        cues, speaker_map=speaker_map, min_duration_ms=min_duration_ms
    )

    for scues in by_slot.values():
        scues.sort(key=lambda x: (x.start_time, x.end_time))
        for i, c in enumerate(scues):
            if round((c.end_time - c.start_time) * 1000) < min_duration_ms:
                next_st = (
                    scues[i + 1].start_time
                    if i + 1 < len(scues)
                    else c.start_time + min_dur_s
                )
                if next_st - c.start_time >= min_dur_s:
                    c.end_time = round(c.start_time + min_dur_s, 2)
                else:
                    prev_en = scues[i - 1].end_time if i > 0 else 0.0
                    c.start_time = round(max(prev_en, next_st - min_dur_s), 2)
                    c.end_time = round(c.start_time + min_dur_s, 2)
        for i in range(len(scues) - 1):
            a, b = scues[i], scues[i + 1]
            a_text = (a.final_text or a.translated_text or a.source_text or "").strip()
            a_unfinished = a_text.endswith(
                (",", "...", "…", "--", "—", "、", "けど", "つつ", "から", "て", "で")
            )
            gap_ms = round((b.start_time - a.end_time) * 1000)
            if gap_ms < 0 or (0 < gap_ms < min_duration_ms) or (
                a_unfinished and 0 < gap_ms <= 1200
            ):
                a.end_time = b.start_time

    spans = [_CueSpan(c) for c in cues if (c.final_text or c.translated_text or "").strip()]
    _close_screen_gaps(spans, min_duration_ms, slot_lookup=slot_lookup)
    for sp in spans:
        sp.cue.start_time = round(sp.start_ms / 1000.0, 2)
        sp.cue.end_time = round(sp.end_ms / 1000.0, 2)

    violations = check_display_invariants(
        cues,
        speaker_map=speaker_map,
        min_duration_ms=min_duration_ms,
        min_gap_ms=min_duration_ms,
    )
    if violations:
        logger.warning("%d display invariant violation(s) after align.", len(violations))

    target_json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    if ass_path is not None:
        mode = "final" if any(c.final_text or c.translated_text for c in cues) else "source"
        render_ass_document(doc, ass_path, mode=mode, speaker_map=speaker_map)
    return doc
