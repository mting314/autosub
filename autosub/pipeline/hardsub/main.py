"""Hardsub stage: burn an .ass subtitle into a video, trim to segments, concat.

Requires ffmpeg built with libass.

- Burns the subtitle with the libass ``ass=`` filter (re-encodes video).
- Trims to one or more ``(start, end)`` segments; gaps between segments are
  dropped. Multiple segments burn independently (bounded parallelism) then
  concatenate with stream copy.
- Warns if a concat join contains a black interval (a fade/join flash).

Segments use **input-seek**: ``-ss <start>`` is placed *before* ``-i`` so ffmpeg
jumps straight to the segment (accurate seek, decodes only from the preceding
keyframe) instead of decoding + libass-rendering the entire lead-in and
discarding it. Input-seek resets the output timeline to zero at ``start``, so the
subtitle is time-shifted by ``-start`` per segment to stay aligned. The
whole-video case (no segments) needs no seek or shift.

The ``ass=`` filter can't handle spaces/special characters in the subtitle path,
so shifted subtitles are written to a clean temp dir, and the whole-video case
symlinks the original to a clean temp path; the filter argument is additionally
escaped for Windows drive paths. Input/output paths pass as argv, so spaces
there are fine.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pyass

from autosub.core.utils import parse_timestamp

logger = logging.getLogger(__name__)

# Keep ffmpeg's stderr minimal (drop the per-second progress spam) so failure
# output stays readable and stderr never floods a pipe.
_QUIET = ["-nostats", "-loglevel", "warning"]


def _resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. The hardsub stage requires ffmpeg built "
            "with libass (present in the autosub Docker image)."
        )
    return exe


def _escape_filter_path(path: Path | str) -> str:
    """Forward-slash + colon-escape a path for an ffmpeg filter option value.

    So Windows paths (``C:\\Users\\...``) aren't misread by the option parser.
    A no-op for POSIX temp paths (no ``\\`` or ``:``).
    """
    return str(path).replace("\\", "/").replace(":", r"\:")


def _ass_filter_arg(path: Path | str, fonts_dir: Path | str | None = None) -> str:
    """Build the libass ``ass=`` filter argument for a path.

    When ``fonts_dir`` is given, ``:fontsdir=`` points libass at a directory of
    bundled fonts. This is REQUIRED for portable/remote rendering: without it
    libass only consults the host's fontconfig, so a style's Fontname (e.g.
    ``Lato ExtraBold``) silently falls back to a different face on any machine
    that lacks that exact font — the subtitles then render in the wrong font and
    often much larger. Bundling the ``.ass``'s own fonts makes the burn match
    Aegisub byte-for-byte regardless of what's installed.
    """
    arg = f"ass={_escape_filter_path(path)}"
    if fonts_dir is not None:
        arg += f":fontsdir={_escape_filter_path(fonts_dir)}"
    return arg


def hardsub_video(
    video_path: Path | str,
    ass_path: Path | str,
    output_path: Path | str,
    segments: list[tuple[str, str]] | None = None,
    *,
    crf: int = 18,
    preset: str = "medium",
    detect_black: bool = True,
    fonts_dir: Path | str | None = None,
) -> None:
    """Burn ``ass_path`` into ``video_path``, trim to ``segments``, write output.

    ``segments`` is a list of ``(start, end)`` timestamp strings (ffmpeg-parsable,
    e.g. "00:09:45" or "585"). Empty/None hardsubs the whole video.

    ``fonts_dir`` is a directory of fonts handed to libass via ``fontsdir`` so the
    ``.ass``'s Fontnames resolve to bundled fonts instead of the host's fontconfig.
    Defaults to the subtitle's own directory (drop the ``.ttf``/``.otf`` next to
    the ``.ass``). Pass ``fonts_dir=""``/a nonexistent dir to disable.
    """
    ffmpeg = _resolve_ffmpeg()
    video_path = Path(video_path)
    ass_path = Path(ass_path)
    output_path = Path(output_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not ass_path.is_file():
        raise FileNotFoundError(f"subtitle not found: {ass_path}")

    # Resolve the libass font directory. An explicit ``fonts_dir`` is honored as
    # given; otherwise auto-default to the subtitle's own folder, but ONLY when it
    # actually holds bundled fonts — so behavior is unchanged for episodes without
    # any (and existing fontconfig-based rendering still works).
    if fonts_dir is not None:
        fonts_dir = Path(fonts_dir)
        fonts_dir = fonts_dir if fonts_dir.is_dir() else None
    else:
        parent = ass_path.parent
        has_fonts = any(
            p.is_file() and p.suffix.lower() in (".ttf", ".otf") for p in parent.iterdir()
        )
        fonts_dir = parent if has_fonts else None
    if fonts_dir is not None:
        logger.info("Using fonts from %s", fonts_dir)

    segs = list(segments or [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joins: list[float] = []

    with tempfile.TemporaryDirectory(prefix="autosub_hardsub_") as tmp:
        tmpdir = Path(tmp)

        if not segs:
            # Whole video: no seek, no shift — the ass= filter can't take spaces,
            # so burn via a clean symlink to the original subtitle.
            subs_link = tmpdir / "subs.ass"
            try:
                subs_link.symlink_to(ass_path.resolve())
            except OSError:
                shutil.copyfile(ass_path, subs_link)
            logger.info("Hardsubbing %s (whole video)...", video_path.name)
            _run(_burn_whole_cmd(ffmpeg, video_path, subs_link, output_path, crf, preset, fonts_dir), "hardsub burn")
        elif len(segs) == 1:
            start, end = segs[0]
            logger.info("Hardsubbing %s (%s–%s)...", video_path.name, start, end)
            cmd = _prepare_segment(ffmpeg, video_path, ass_path, output_path, start, end, crf, preset, tmpdir, 0, fonts_dir)
            _run(cmd, "hardsub burn")
        else:
            logger.info("Hardsubbing %s across %d segments...", video_path.name, len(segs))
            parts = _burn_segments_parallel(ffmpeg, video_path, ass_path, segs, crf, preset, tmpdir, fonts_dir)
            _concat(ffmpeg, parts, output_path, tmpdir)
            joins = _join_offsets(segs)

    logger.info("Hardsubbed video written to %s", output_path)
    if detect_black and joins:
        _warn_on_black(ffmpeg, output_path, joins)


def _join_offsets(segs: list[tuple[str, str]]) -> list[float]:
    """Output-timeline offsets (seconds) where consecutive segments are joined."""
    offsets: list[float] = []
    cumulative = 0.0
    for start, end in segs[:-1]:
        cumulative += parse_timestamp(end) - parse_timestamp(start)
        offsets.append(cumulative)
    return offsets


def _shift_ass(src_ass: Path, offset_seconds: float, dst_ass: Path) -> None:
    """Write ``src_ass`` with every event shifted earlier by ``offset_seconds``.

    Events ending before the new zero are dropped; events straddling it are
    clamped to start at 0. Used so subtitles align after an input-seek resets the
    output timeline to zero at the segment start.
    """
    with open(src_ass, encoding="utf-8") as handle:
        script = pyass.load(handle)
    offset = timedelta(seconds=offset_seconds)
    zero = timedelta(0)
    kept = []
    for event in script.events:
        new_start = event.start - offset
        new_end = event.end - offset
        if new_end <= zero:
            continue
        event.start = new_start if new_start > zero else zero
        event.end = new_end
        kept.append(event)
    script.events = kept
    with open(dst_ass, "w", encoding="utf-8") as handle:
        pyass.dump(script, handle)


def _prepare_segment(ffmpeg, video, ass_path, out, start, end, crf, preset, tmpdir, idx, fonts_dir=None) -> list[str]:
    """Shift the subtitle for this segment and return the input-seek burn command."""
    start_s = parse_timestamp(start)
    duration = parse_timestamp(end) - start_s
    if duration <= 0:
        raise ValueError(f"segment end must be after start: {start}–{end}")
    shifted = tmpdir / f"seg_{idx:03d}.ass"
    _shift_ass(ass_path, start_s, shifted)
    return _seg_burn_cmd(ffmpeg, video, shifted, out, start, duration, crf, preset, fonts_dir)


def _burn_whole_cmd(ffmpeg, video, subs_link, out, crf, preset, fonts_dir=None) -> list[str]:
    return [
        ffmpeg, "-y", *_QUIET, "-i", str(video), "-vf", _ass_filter_arg(subs_link, fonts_dir),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-c:a", "aac", str(out),
    ]


def _seg_burn_cmd(ffmpeg, video, shifted_ass, out, start, duration, crf, preset, fonts_dir=None) -> list[str]:
    # -ss BEFORE -i = input seek (fast, accurate): decode only from the keyframe
    # preceding `start`, not from 0. The subtitle is already shifted to match the
    # zero-based post-seek timeline.
    return [
        ffmpeg, "-y", *_QUIET, "-ss", start, "-i", str(video),
        "-vf", _ass_filter_arg(shifted_ass, fonts_dir), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-c:a", "aac", str(out),
    ]


def _burn_segments_parallel(ffmpeg, video, ass_path, segs, crf, preset, tmpdir, fonts_dir=None) -> list[Path]:
    """Burn each segment concurrently, bounded to the CPU count.

    Each burn runs via ``subprocess.run`` (which fully drains its own stderr, so
    there's no PIPE-buffer stall), inside a ThreadPoolExecutor capped at
    ``os.cpu_count()`` so many segments don't oversubscribe the machine.
    """
    parts = [tmpdir / f"part_{i:03d}.mp4" for i in range(len(segs))]
    workers = min(len(segs), os.cpu_count() or 2)

    def _burn(i: int) -> None:
        start, end = segs[i]
        logger.info("  segment %d/%d: %s–%s", i + 1, len(segs), start, end)
        cmd = _prepare_segment(ffmpeg, video, ass_path, parts[i], start, end, crf, preset, tmpdir, i, fonts_dir)
        _run(cmd, f"hardsub segment {i + 1}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_burn, i) for i in range(len(segs))]
        for future in as_completed(futures):
            future.result()  # re-raise the first segment failure
    return parts


def _concat(ffmpeg, parts: list[Path], out: Path, tmpdir: Path) -> None:
    concat_file = tmpdir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    _run(
        [ffmpeg, "-y", *_QUIET, "-f", "concat", "-safe", "0", "-i", str(concat_file),
         "-c", "copy", str(out)],
        "concat segments",
    )


def _run(cmd: list[str], desc: str) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg {desc} failed:\n{proc.stderr.decode(errors='replace')[-600:]}"
        )


def _warn_on_black(ffmpeg: str, out: Path, joins: list[float], window: float = 2.0) -> None:
    """Scan a short window around each concat join for black frames.

    Only the joins (where concatenation can introduce a flash) are scanned, using
    input-seek — not the whole output — so this stays cheap on long videos.
    """
    blacks: list[str] = []
    for offset in joins:
        start = max(0.0, offset - window)
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-i", str(out),
             "-t", f"{2 * window:.3f}", "-vf", "blackdetect=d=0.05:pix_th=0.10",
             "-an", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        blacks += [
            line for line in proc.stderr.decode(errors="replace").splitlines()
            if "black_start" in line
        ]
    if blacks:
        logger.warning(
            "Detected %d black interval(s) near concat joins (possible flash — "
            "check segment boundaries):",
            len(blacks),
        )
        for line in blacks[:5]:
            logger.warning("  %s", line.strip())
