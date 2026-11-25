from typing import List, NamedTuple, Optional

from autosub.core.schemas import (
    ReplacementSpan,
    SubtitleCue,
    SubtitleLine,
    TranscribedWord,
)
from autosub.core.speaker_map import build_slot_lookup

# There is deliberately no "breathing room" constant for two lines sharing a slot.
# A short gap blinks the box empty for a few frames and reads as a flash, so a
# same-slot pair either chains exactly or is far enough apart to be a real pause.


def _original_text_length(text: str, spans: list[ReplacementSpan]) -> int:
    delta = sum(
        (span.replaced_end - span.replaced_start) - (span.orig_end - span.orig_start)
        for span in spans
    )
    return len(text) - delta


def _offset_replacement_spans(
    spans: list[ReplacementSpan],
    *,
    orig_offset: int,
    replaced_offset: int,
) -> list[ReplacementSpan]:
    return [
        ReplacementSpan(
            orig_start=span.orig_start + orig_offset,
            orig_end=span.orig_end + orig_offset,
            replaced_start=span.replaced_start + replaced_offset,
            replaced_end=span.replaced_end + replaced_offset,
        )
        for span in spans
    ]


class SegmentMS:
    """Helper wrapper to manipulate timestamps in raw milliseconds to avoid float drift."""

    def __init__(self, line: SubtitleLine):
        self.text = line.text
        self.speaker = line.speaker
        self.role = line.role
        self.corner = line.corner
        self.words: list[TranscribedWord] = list(line.words)
        self.replacement_spans: list[ReplacementSpan] = list(line.replacement_spans)
        self.start_ms = int(round(line.start_time * 1000))
        self.end_ms = int(round(line.end_time * 1000))
        if self.words:
            self.raw_start_ms = int(round(self.words[0].start_time * 1000))
            self.raw_end_ms = int(round(self.words[-1].end_time * 1000))
        else:
            self.raw_start_ms = self.start_ms
            self.raw_end_ms = self.end_ms

    def to_subtitle_line(self) -> SubtitleLine:
        return SubtitleLine(
            text=self.text,
            start_time=self.start_ms / 1000.0,
            end_time=self.end_ms / 1000.0,
            speaker=self.speaker,
            role=self.role,
            corner=self.corner,
            words=list(self.words),
            replacement_spans=list(self.replacement_spans),
        )


def _merge_segment_into(left: SegmentMS, right: SegmentMS) -> None:
    """Merge ``right`` into ``left``, preserving words and replacement_spans offsets."""
    separator = " " if left.text and right.text else ""
    if hasattr(left, "replacement_spans") and hasattr(right, "replacement_spans"):
        right_orig_offset = _original_text_length(
            left.text, left.replacement_spans
        ) + len(separator)
        right_replaced_offset = len(left.text) + len(separator)
        left.replacement_spans.extend(
            _offset_replacement_spans(
                right.replacement_spans,
                orig_offset=right_orig_offset,
                replaced_offset=right_replaced_offset,
            )
        )
    left.text = f"{left.text}{separator}{right.text}".strip()
    left.end_ms = max(left.end_ms, right.end_ms)
    if hasattr(left, "raw_end_ms") and hasattr(right, "raw_end_ms"):
        left.raw_end_ms = max(left.raw_end_ms, right.raw_end_ms)
    if hasattr(left, "words") and hasattr(right, "words"):
        if separator and left.words:
            # Keep concatenated word text aligned with left.text so
            # char-position walks in find_split_time stay exact.
            left.words[-1] = left.words[-1].model_copy(
                update={"word": left.words[-1].word + separator}
            )
        left.words.extend(right.words)


class ProposedExtension:
    def __init__(self):
        self.lead_in = 0
        self.lead_out = 0


def _roles_compatible_for_merge(left_role: str | None, right_role: str | None) -> bool:
    """
    Allow merge when roles are identical or unavailable.

    When radio_discourse has already labeled lines, merging across role boundaries
    would collapse distinct speaker functions such as listener mail into host/meta.
    """
    if left_role is None or right_role is None:
        return True
    return left_role == right_role


def _get_prev_keyframe(time_ms: int, keyframes: List[int]) -> Optional[int]:
    """Finds the closest keyframe before or at the given time."""
    for k in reversed(keyframes):
        if k <= time_ms:
            return k
    return None


def _get_next_keyframe(time_ms: int, keyframes: List[int]) -> Optional[int]:
    """Finds the closest keyframe after or at the given time."""
    for k in keyframes:
        if k >= time_ms:
            return k
    return None


def _apply_min_duration_padding(
    segments: List[SegmentMS],
    keyframes: List[int],
    video_duration_ms: Optional[int],
    min_duration_ms: int,
) -> List[SegmentMS]:
    """Pass 1: Minimum Duration (Analyze and Commit)"""
    extensions = [ProposedExtension() for _ in segments]

    # Phase A: Analyze
    for i, seg in enumerate(segments):
        duration = seg.end_ms - seg.start_ms
        if duration < min_duration_ms:
            shortfall = min_duration_ms - duration
            lead_in = shortfall // 2
            lead_out = shortfall - lead_in
            extensions[i].lead_in = lead_in
            extensions[i].lead_out = lead_out

    # Distribute gap spaces fairly
    for i in range(len(segments) - 1):
        ext_left = extensions[i].lead_out
        ext_right = extensions[i + 1].lead_in
        gap = segments[i + 1].start_ms - segments[i].end_ms
        gap = max(0, gap)

        # Keyframe bounds within this gap
        gap_kf = [
            k for k in keyframes if segments[i].end_ms < k < segments[i + 1].start_ms
        ]

        if gap_kf:
            max_left = gap_kf[0] - segments[i].end_ms
            max_right = segments[i + 1].start_ms - gap_kf[-1]
            extensions[i].lead_out = min(max_left, extensions[i].lead_out)
            extensions[i + 1].lead_in = min(max_right, extensions[i + 1].lead_in)
        else:
            if ext_left + ext_right > gap:
                # Proportional 50/50 division
                if ext_left > 0 and ext_right > 0:
                    extensions[i].lead_out = gap // 2
                    extensions[i + 1].lead_in = gap - (gap // 2)
                elif ext_left > 0:
                    extensions[i].lead_out = gap
                elif ext_right > 0:
                    extensions[i + 1].lead_in = gap

    # Check boundaries against video start/end
    for i, seg in enumerate(segments):
        max_lead_in = seg.start_ms
        kf_before = _get_prev_keyframe(seg.start_ms, keyframes)
        if kf_before is not None:
            max_lead_in = min(max_lead_in, seg.start_ms - kf_before)
        if i == 0:
            extensions[i].lead_in = min(extensions[i].lead_in, max_lead_in)

        max_lead_out = float("inf")
        if video_duration_ms is not None:
            max_lead_out = video_duration_ms - seg.end_ms
        kf_after = _get_next_keyframe(seg.end_ms, keyframes)
        if kf_after is not None:
            max_lead_out = min(max_lead_out, kf_after - seg.end_ms)
        if i == len(segments) - 1:
            if max_lead_out != float("inf"):
                extensions[i].lead_out = min(extensions[i].lead_out, int(max_lead_out))

    # Commit extensions
    for i, seg in enumerate(segments):
        duration = seg.end_ms - seg.start_ms
        if duration >= min_duration_ms:
            continue

        shortfall = min_duration_ms - duration
        ext = extensions[i]

        actual_lead_in = ext.lead_in
        actual_lead_out = ext.lead_out

        # Recalculate segment specific bounds to avoid python loop scope bleed
        cur_max_lead_in = seg.start_ms
        kf_before = _get_prev_keyframe(seg.start_ms, keyframes)
        if kf_before is not None:
            cur_max_lead_in = min(cur_max_lead_in, seg.start_ms - kf_before)

        cur_max_lead_out = float("inf")
        if video_duration_ms is not None:
            cur_max_lead_out = video_duration_ms - seg.end_ms
        kf_after = _get_next_keyframe(seg.end_ms, keyframes)
        if kf_after is not None:
            cur_max_lead_out = min(cur_max_lead_out, kf_after - seg.end_ms)

        total_padding = actual_lead_in + actual_lead_out
        if total_padding < shortfall:
            # Shift the burden right
            if i < len(segments) - 1:
                right_gap = segments[i + 1].start_ms - segments[i].end_ms
                kf_gap = [
                    k
                    for k in keyframes
                    if segments[i].end_ms < k < segments[i + 1].start_ms
                ]
                if kf_gap:
                    right_gap = kf_gap[0] - segments[i].end_ms
                available_right = right_gap - extensions[i + 1].lead_in
            else:
                available_right = cur_max_lead_out

            if available_right > actual_lead_out:
                additional = min(
                    available_right - actual_lead_out, shortfall - total_padding
                )
                if additional != float("inf"):
                    actual_lead_out += int(additional)

            # Shift the burden left
            if i > 0:
                left_gap = segments[i].start_ms - segments[i - 1].end_ms
                kf_gap = [
                    k
                    for k in keyframes
                    if segments[i - 1].end_ms < k < segments[i].start_ms
                ]
                if kf_gap:
                    left_gap = segments[i].start_ms - kf_gap[-1]
                available_left = left_gap - extensions[i - 1].lead_out
            else:
                available_left = cur_max_lead_in

            total_padding = actual_lead_in + actual_lead_out
            if available_left > actual_lead_in and total_padding < shortfall:
                additional = min(
                    available_left - actual_lead_in, shortfall - total_padding
                )
                if additional != float("inf"):
                    actual_lead_in += int(additional)

        # Commit
        seg.start_ms -= int(actual_lead_in)
        seg.end_ms += int(actual_lead_out)
        extensions[i].lead_in = int(actual_lead_in)
        extensions[i].lead_out = int(actual_lead_out)

    # Merging
    merged_segments = []
    skip_next = False
    for i in range(len(segments)):
        if skip_next:
            skip_next = False
            continue

        seg = segments[i]
        duration = seg.end_ms - seg.start_ms
        if duration < min_duration_ms:
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                if (
                    _roles_compatible_for_merge(seg.role, next_seg.role)
                    and not next_seg.corner
                ):
                    # Merge with the next segment when semantic role stays compatible.
                    _merge_segment_into(seg, next_seg)
                    skip_next = True
            else:
                pass  # Final Segment edge case
        merged_segments.append(seg)
    return merged_segments


def _apply_gap_snapping(
    segments: List[SegmentMS],
    keyframes: List[int],
    snap_threshold_ms: int,
    conditional_snap_threshold_ms: int,
    min_gap_ms: int = 0,
) -> List[SegmentMS]:
    """Pass 2: Gaps"""
    for i in range(len(segments) - 1):
        prev_seg = segments[i]
        next_seg = segments[i + 1]

        gap = next_seg.start_ms - prev_seg.end_ms
        if gap <= 0:
            continue

        if gap < snap_threshold_ms:
            kfs_in_gap = [
                k for k in keyframes if prev_seg.end_ms < k < next_seg.start_ms
            ]
            if kfs_in_gap:
                # Small Gap with Keyframe
                prev_seg.end_ms = kfs_in_gap[0]
                next_seg.start_ms = kfs_in_gap[0]
            else:
                # Meeting in the middle
                half_gap = gap // 2
                prev_seg.end_ms += half_gap
                next_seg.start_ms -= gap - half_gap
        elif snap_threshold_ms <= gap < conditional_snap_threshold_ms:
            kfs_in_gap = [
                k for k in keyframes if prev_seg.end_ms < k < next_seg.start_ms
            ]
            if kfs_in_gap:
                # Multiple Keyframes in Gap: hold until the first cut and resume
                # at the last, so the blank lines up with the visual interlude —
                # but only when that blank is long enough to read as deliberate.
                # Otherwise it is just a flash, and both sides snap to one cut.
                if kfs_in_gap[-1] - kfs_in_gap[0] >= min_gap_ms:
                    prev_seg.end_ms = kfs_in_gap[0]
                    next_seg.start_ms = kfs_in_gap[-1]
                else:
                    prev_seg.end_ms = kfs_in_gap[0]
                    next_seg.start_ms = kfs_in_gap[0]
            else:
                # Conditional Gap (no keyframes) - snap standard
                half_gap = gap // 2
                prev_seg.end_ms += half_gap
                next_seg.start_ms -= gap - half_gap
    return segments


def _apply_micro_snapping(
    segments: List[SegmentMS],
    keyframes: List[int],
    micro_snap_threshold: int,
    video_duration_ms: Optional[int],
) -> List[SegmentMS]:
    """Pass 3: Micro-Snapping"""
    for i, seg in enumerate(segments):
        start_kf = _get_prev_keyframe(seg.start_ms, keyframes)
        if start_kf is not None:
            dist = seg.start_ms - start_kf
            if 0 < dist <= micro_snap_threshold:
                prev_end = segments[i - 1].end_ms if i > 0 else 0
                if start_kf >= prev_end:
                    seg.start_ms = start_kf

        end_kf = _get_next_keyframe(seg.end_ms, keyframes)
        if end_kf is not None:
            dist = end_kf - seg.end_ms
            if 0 < dist <= micro_snap_threshold:
                next_start = (
                    segments[i + 1].start_ms
                    if i < len(segments) - 1
                    else (video_duration_ms or float("inf"))
                )
                if end_kf <= next_start:
                    seg.end_ms = end_kf
    return segments


def _apply_interjection_merging(
    segments: List[SegmentMS],
    interjection_max_duration_ms: int,
    interjection_merge_threshold_ms: int,
    interjection_gap_threshold_ms: int,
) -> List[SegmentMS]:
    """Pass 0: Speaker-aware interjection handling.

    Detects the pattern [A] [B_short] [A] where B is a brief interjection
    (e.g. "yeah", "mmhmm") interrupting speaker A's continuous thought.
    Merges or extends A's lines across the interjection so there's no visual gap.
    B's interjection is left untouched (overlaps in .ass output).
    """
    i = 0
    while i < len(segments) - 2:
        prev_seg = segments[i]
        mid_seg = segments[i + 1]
        next_seg = segments[i + 2]

        # Check the A-B-A pattern
        if (
            prev_seg.speaker
            and prev_seg.speaker == next_seg.speaker
            and prev_seg.speaker != mid_seg.speaker
        ):
            mid_duration = mid_seg.end_ms - mid_seg.start_ms
            span_gap = next_seg.start_ms - prev_seg.end_ms

            if (
                mid_duration <= interjection_max_duration_ms
                and span_gap > 0
                and span_gap <= interjection_gap_threshold_ms
            ):
                if (
                    span_gap <= interjection_merge_threshold_ms
                    and _roles_compatible_for_merge(prev_seg.role, next_seg.role)
                    and not next_seg.corner
                ):
                    # MERGE: combine A's lines into one
                    _merge_segment_into(prev_seg, next_seg)
                    segments.pop(i + 2)
                    # Don't advance i — check if another interjection follows
                    continue
                else:
                    # EXTEND: close the gap between A's lines (meet in middle)
                    gap = next_seg.start_ms - prev_seg.end_ms
                    half_gap = gap // 2
                    prev_seg.end_ms += half_gap
                    next_seg.start_ms -= gap - half_gap
        i += 1

    return segments


def slot_key_for_speaker(
    speaker: Optional[str], slot_lookup: dict[str, int]
) -> Optional[str]:
    """Identify the box a line renders in, falling back to the raw label."""
    if speaker is None:
        return None
    slot = slot_lookup.get(str(speaker))
    return f"slot:{slot}" if slot is not None else speaker


def _prevent_slot_overlaps(segments, min_duration_ms: int = 0):
    """Ensure no two segments that share an on-screen slot are visible at once.

    Callers must pass a group of segments that all render in the same box. Two
    events at the same position do not stack, they draw on top of each other, so
    the earlier one is truncated to clear the later one.

    Truncating is preferred, but not at the cost of a flash. Cutting a line below
    min_duration_ms is worse than starting the next one slightly late, so the
    later line is delayed instead, and only when neither will fit are the two
    merged. Accepts anything exposing text/start_ms/end_ms, so cue timing can be
    re-checked after translation through the same rules.
    """
    resolved = []
    for segment in sorted(segments, key=lambda seg: (seg.start_ms, seg.end_ms)):
        if not resolved:
            resolved.append(segment)
            continue

        previous = resolved[-1]
        if segment.start_ms >= previous.end_ms:
            resolved.append(segment)
            continue

        # Chain exactly: the earlier line ends on the frame the later one starts.
        # Leaving a small gap here would blink the box empty for a few frames,
        # which reads as a flash — worse than either a clean swap or a real pause.
        truncated_end = segment.start_ms
        if (
            truncated_end > previous.start_ms
            and truncated_end - previous.start_ms >= min_duration_ms
        ):
            previous.end_ms = truncated_end
            resolved.append(segment)
            continue

        if min_duration_ms:
            delayed_start = previous.end_ms
            if segment.end_ms - delayed_start >= min_duration_ms:
                segment.start_ms = delayed_start
                resolved.append(segment)
                continue

        # Neither line can give way without becoming a flash. Merging keeps both
        # texts readable instead of stacking glyphs.
        _merge_segment_into(previous, segment)
    return resolved


def _screen_gaps(spans) -> List[tuple]:
    """Periods with no text anywhere, as (gap_ms, ends_at, resumes_at).

    Slots are independent boxes, so lines in different slots can be on screen at
    once and a gap only exists where the merged coverage of every line has a
    hole. Comparing consecutive lines pairwise would invent gaps wherever
    concurrent speech happens to be ordered awkwardly.
    """
    order = sorted(spans, key=lambda item: (item.start_ms, item.end_ms))
    merged: List[List[int]] = []
    for span in order:
        if merged and span.start_ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span.end_ms)
        else:
            merged.append([span.start_ms, span.end_ms])
    return [
        (later[0] - earlier[1], earlier[1], later[0])
        for earlier, later in zip(merged, merged[1:])
    ]


def _span_speaker(span) -> Optional[str]:
    if hasattr(span, "speaker"):
        return span.speaker
    if hasattr(span, "cue") and hasattr(span.cue, "speaker"):
        return span.cue.speaker
    return None


def _close_screen_gaps(
    spans,
    min_gap_ms: int,
    slot_lookup: Optional[dict[str, int]] = None,
):
    """Close gaps too short to read as a pause, within and across every slot.

    Only ever moves an END, never a START, and that asymmetry is the whole
    reason this is safe. A start has to land on the speech or the subtitle reads
    as out of sync; an end does not, and holding a line a little longer is
    ordinary subtitling practice — it is what minimum-duration rules do too.

    Closes sub-min_gap_ms gaps within each individual slot first (even if another
    slot is concurrently active on screen), then closes global screen-empty gaps.
    """
    if min_gap_ms <= 0 or len(spans) < 2:
        return spans

    # 1. Close sub-min_gap_ms flash gaps within each slot, even when another slot
    # is concurrently active on screen.
    lookup = slot_lookup or {}
    groups: dict[Optional[str], list] = {}
    for span in spans:
        key = slot_key_for_speaker(_span_speaker(span), lookup)
        groups.setdefault(key, []).append(span)

    for group in groups.values():
        group.sort(key=lambda item: (item.start_ms, item.end_ms))
        for earlier, later in zip(group, group[1:]):
            gap = later.start_ms - earlier.end_ms
            if 0 < gap < min_gap_ms:
                earlier.end_ms = later.start_ms

    # 2. Close global screen gaps (where no slot has text on screen).
    for gap, ends_at, resumes_at in _screen_gaps(spans):
        if 0 < gap < min_gap_ms:
            for span in spans:
                if span.end_ms == ends_at:
                    span.end_ms = resumes_at
    return spans


class DisplayViolation(NamedTuple):
    """One way the finished document breaks an on-screen timing rule."""

    kind: str  # "duration" | "overlap" | "gap"
    cue_id: str
    start_time: float
    detail: str

    def __str__(self) -> str:
        stamp = f"{int(self.start_time // 60):02d}:{self.start_time % 60:05.2f}"
        return f"{stamp} [{self.kind}] {self.cue_id}: {self.detail}"


def check_display_invariants(
    cues: List[SubtitleCue],
    *,
    speaker_map: Optional[dict[str, dict]] = None,
    min_duration_ms: int = 500,
    min_gap_ms: Optional[int] = None,
    text_field: str = "final_text",
) -> List[DisplayViolation]:
    """Report where a finished document breaks the on-screen timing rules.

    1. no line on screen for less than min_duration_ms,
    2. never two lines sharing a slot at once,
    3. no gap within a slot short enough to read as a flash — a same-slot pair
       either chains exactly or is at least min_gap_ms apart, never in between.

    This deliberately reports rather than repairs. apply_timing_rules already owns
    these rules and enforces them with far more context — keyframes, video
    duration, the whole timing model — so a second pass that quietly re-times cues
    would duplicate the decision and could override a considered one with a
    blunter version of itself. Worse, it would absorb upstream bugs silently: the
    50ms gap this module used to insert after an overlap fix went unnoticed
    precisely because it looked like something a repair pass would tidy up.

    A violation here means something upstream regressed, which is worth surfacing
    rather than papering over.

    min_gap_ms defaults to min_duration_ms: a gap too short to have been a
    readable line is also too short to register as a deliberate pause.
    """
    if not cues:
        return []

    gap_floor = min_duration_ms if min_gap_ms is None else min_gap_ms
    slot_lookup = build_slot_lookup(speaker_map)

    groups: dict[Optional[str], List[SubtitleCue]] = {}
    for cue in cues:
        # A cue that renders nothing cannot flash, overlap or be too brief.
        if not (getattr(cue, text_field, None) or "").strip():
            continue
        key = slot_key_for_speaker(cue.speaker, slot_lookup)
        groups.setdefault(key, []).append(cue)

    violations: List[DisplayViolation] = []
    reported_gap_windows: set[tuple[int, int]] = set()
    for group in groups.values():
        group.sort(key=lambda cue: (cue.start_time, cue.end_time))

        for cue in group:
            shown_ms = round((cue.end_time - cue.start_time) * 1000)
            if shown_ms < min_duration_ms:
                violations.append(
                    DisplayViolation(
                        "duration",
                        cue.id,
                        cue.start_time,
                        f"on screen {shown_ms}ms, under the {min_duration_ms}ms floor",
                    )
                )

        # Overlap and same-slot flash gaps stay per-slot: two lines in different
        # boxes at once is the point of the overlay, not a defect, and a blink in
        # one box is still a flash even while the other box is holding text.
        for earlier, later in zip(group, group[1:]):
            earlier_end_ms = round(earlier.end_time * 1000)
            later_start_ms = round(later.start_time * 1000)
            gap = later_start_ms - earlier_end_ms
            if gap < 0:
                violations.append(
                    DisplayViolation(
                        "overlap",
                        earlier.id,
                        earlier.start_time,
                        f"overlaps {later.id} by {-gap}ms in the same slot",
                    )
                )
            elif 0 < gap < gap_floor:
                reported_gap_windows.add((earlier_end_ms, later_start_ms))
                violations.append(
                    DisplayViolation(
                        "gap",
                        earlier.id,
                        earlier_end_ms / 1000.0,
                        f"{gap}ms gap before {later.id} in the same slot — too "
                        f"short to read as a pause",
                    )
                )

    # Gaps are also measured across every slot, over the merged coverage of the
    # rendering cues: a hand-off leaves the screen textless just as a same-slot
    # gap does. Closing one only ever lengthens the earlier line, which is
    # harmless — it is the start that has to land on the speech.
    class _Span(NamedTuple):
        start_ms: int
        end_ms: int

    spans = [
        _Span(round(c.start_time * 1000), round(c.end_time * 1000))
        for group in groups.values()
        for c in group
    ]
    ends_at_cues: dict[int, List[SubtitleCue]] = {}
    for group in groups.values():
        for c in group:
            ends_at_cues.setdefault(round(c.end_time * 1000), []).append(c)

    for gap, ends_at, resumes_at in _screen_gaps(spans):
        if 0 < gap < gap_floor and (ends_at, resumes_at) not in reported_gap_windows:
            matching_cues = ends_at_cues.get(ends_at, [])
            cue = matching_cues[-1] if matching_cues else None
            violations.append(
                DisplayViolation(
                    "gap",
                    cue.id if cue else "?",
                    ends_at / 1000.0,
                    f"{gap}ms with no text on screen before the next line — too "
                    f"short to read as a pause",
                )
            )

    violations.sort(key=lambda item: (item.start_time, item.kind))
    return violations


def _merge_micro_fragments_before_padding(
    segments: List[SegmentMS],
    snap_threshold_ms: int,
    max_merged_chars: int = 42,
) -> List[SegmentMS]:
    """Merge ultra-short speaker micro-fragments with an adjacent line before padding.

    When diarized speech contains a tiny 80-150ms single-token utterance or a
    pair of sub-250ms utterances separated by <= snap_threshold_ms, padding each
    fragment with 350-420ms of silence inflates them into standalone 500ms cues
    that no longer look short to the post-padding merge pass.
    """
    if len(segments) < 2:
        return segments

    merged: List[SegmentMS] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        while i + 1 < len(segments) and seg.speaker is not None:
            next_seg = segments[i + 1]
            gap = next_seg.start_ms - seg.end_ms
            seg_dur = seg.end_ms - seg.start_ms
            next_dur = next_seg.end_ms - next_seg.start_ms
            combined_len = len(seg.text) + (1 if seg.text and next_seg.text else 0) + len(next_seg.text)
            if (
                0 <= gap <= snap_threshold_ms
                and _roles_compatible_for_merge(seg.role, next_seg.role)
                and not next_seg.corner
                and combined_len <= max_merged_chars
                and (
                    seg_dur <= 150
                    or next_dur <= 150
                    or (seg_dur <= 250 and next_dur <= 250)
                )
            ):
                _merge_segment_into(seg, next_seg)
                i += 1
            else:
                break
        merged.append(seg)
        i += 1
    return merged


def _resolve_cross_slot_transitions(
    segments: List[SegmentMS],
    slot_lookup: dict[str, int],
    min_duration_ms: int,
    video_duration_ms: Optional[int] = None,
) -> List[SegmentMS]:
    """Coordinate turn hand-offs and backchannels across speaker slots.

    1. Bridges [A1] [B_short] [A2] backchannels so Speaker A's box holds across
       Speaker B's brief nod (no 500ms empty-box blink in A's slot), while A2
       starts on Speaker A's actual resumption speech onset.
    2. Eliminates artificial 1-5 frame cross-slot overlaps manufactured by
       symmetric lead-in/lead-out padding when the two speakers' raw speech
       did not overlap.
    """
    if len(segments) < 2:
        return segments

    def s_key(seg: SegmentMS) -> Optional[str]:
        return slot_key_for_speaker(seg.speaker, slot_lookup)

    # Index neighbours within each slot so clamping never creates a same-slot overlap.
    slot_groups: dict[Optional[str], List[SegmentMS]] = {}
    for seg in segments:
        slot_groups.setdefault(s_key(seg), []).append(seg)
    prev_in_slot: dict[int, SegmentMS] = {}
    next_in_slot: dict[int, SegmentMS] = {}
    for grp in slot_groups.values():
        grp.sort(key=lambda s: (s.start_ms, s.end_ms))
        for idx, seg in enumerate(grp):
            if idx > 0:
                prev_in_slot[id(seg)] = grp[idx - 1]
            if idx + 1 < len(grp):
                next_in_slot[id(seg)] = grp[idx + 1]

    # 1. Detect [A1] [B_short] [A2] cross-slot backchannels
    held_across_backchannel: set[tuple[int, int]] = set()
    for i in range(len(segments) - 2):
        a1, b, a2 = segments[i], segments[i + 1], segments[i + 2]
        if (
            s_key(a1) is not None
            and s_key(a1) == s_key(a2)
            and s_key(a1) != s_key(b)
            and next_in_slot.get(id(a1)) is a2
        ):
            b_raw_dur = b.raw_end_ms - b.raw_start_ms
            raw_gap_a = a2.raw_start_ms - a1.raw_end_ms
            b_text = (b.text or "").strip()
            a1_text = (a1.text or "").strip()
            b_is_long_question = b_text.endswith(("?", "？")) and len(b_text) > 16
            a1_unfinished = a1_text.endswith((",", "...", "…", "--", "—", "、", "けど", "つつ", "から", "て", "で"))
            max_b_dur = 1200 if not b_is_long_question else 400
            max_gap_a = 1800 if a1_unfinished else 1350
            if (
                b_raw_dur <= max_b_dur
                and -250 <= raw_gap_a <= max_gap_a
                # B must start (near) after A1's speech; a backchannel spoken
                # mid-A1 is genuine overlap and must not be pushed later.
                and b.raw_start_ms >= a1.raw_end_ms - 250
            ):
                # Restore A2's start to its speech onset if gap snapping pulled it early
                if a2.start_ms < a2.raw_start_ms and a2.end_ms - a2.raw_start_ms >= min_duration_ms:
                    a2.start_ms = a2.raw_start_ms
                # Hold A1 across B_short to meet A2 with zero blink in A's slot
                if a1.end_ms < a2.start_ms:
                    a1.end_ms = a2.start_ms
                # Ensure B_short does not start before A1's raw speech finishes
                if b.start_ms < a1.raw_end_ms:
                    b_prev = prev_in_slot.get(id(b))
                    floor_ms = max(a1.raw_end_ms, b_prev.end_ms if b_prev else 0)
                    b.start_ms = max(floor_ms, min(b.raw_start_ms, a2.start_ms - min_duration_ms))
                    if b.start_ms < floor_ms:
                        b.start_ms = floor_ms
                    b_next = next_in_slot.get(id(b))
                    ceil_ms = b_next.start_ms if b_next else (video_duration_ms or (b.start_ms + min_duration_ms))
                    if b.end_ms - b.start_ms < min_duration_ms:
                        b.end_ms = min(ceil_ms, b.start_ms + min_duration_ms)
                held_across_backchannel.add((id(a1), id(b)))
                held_across_backchannel.add((id(b), id(a2)))

    # 2. Remove artificial padding overlaps on sequential non-overlapping turns
    for i in range(len(segments) - 1):
        prev_seg = segments[i]
        curr_seg = segments[i + 1]
        if s_key(prev_seg) == s_key(curr_seg):
            continue
        if (id(prev_seg), id(curr_seg)) in held_across_backchannel:
            continue
        # Only repair when raw speech did not overlap
        if curr_seg.raw_start_ms < prev_seg.raw_end_ms:
            continue
        if prev_seg.end_ms <= curr_seg.start_ms:
            continue

        # Clamp prev_seg post-speech padding if it spilled past curr_seg's raw start
        if prev_seg.end_ms > prev_seg.raw_end_ms and prev_seg.end_ms > curr_seg.raw_start_ms:
            prev_seg.end_ms = max(prev_seg.raw_end_ms, curr_seg.raw_start_ms)
            if prev_seg.end_ms - prev_seg.start_ms < min_duration_ms:
                p_prev = prev_in_slot.get(id(prev_seg))
                left_bound = p_prev.end_ms if p_prev else 0
                prev_seg.start_ms = max(left_bound, prev_seg.end_ms - min_duration_ms)

        # Shift curr_seg pre-speech lead-in so it starts at or after prev_seg.end_ms
        if curr_seg.start_ms < prev_seg.end_ms and curr_seg.start_ms < curr_seg.raw_start_ms:
            new_start = min(curr_seg.raw_start_ms, max(curr_seg.start_ms, prev_seg.end_ms))
            if new_start < prev_seg.end_ms:
                new_start = prev_seg.end_ms
            curr_seg.start_ms = new_start
            if curr_seg.end_ms - curr_seg.start_ms < min_duration_ms:
                c_next = next_in_slot.get(id(curr_seg))
                right_bound = c_next.start_ms if c_next else (video_duration_ms or (curr_seg.start_ms + min_duration_ms))
                curr_seg.end_ms = min(right_bound, curr_seg.start_ms + min_duration_ms)

    segments.sort(key=lambda seg: (seg.start_ms, seg.end_ms))
    return segments


def apply_timing_rules(
    lines: List[SubtitleLine],
    keyframes_ms: Optional[List[int]] = None,
    video_duration_ms: Optional[int] = None,
    min_duration_ms: int = 500,
    snap_threshold_ms: int = 250,
    conditional_snap_threshold_ms: int = 500,
    interjection_max_duration_ms: int = 1000,
    interjection_merge_threshold_ms: int = 1500,
    interjection_gap_threshold_ms: int = 2000,
    min_gap_ms: Optional[int] = None,
    per_speaker: bool = False,
    speaker_map: Optional[dict[str, dict]] = None,
) -> List[SubtitleLine]:
    """Applies advanced timing rules to subtitle lines.

    When per_speaker is True or multiple distinct speakers are present,
    timing rules run independently per speaker to allow concurrent overlapping dialogue.

    Grouping is by on-screen slot rather than raw speaker label. Speaker maps are
    many-to-one, so several diarization labels commonly share one slot; grouping by
    label would let those labels overlap inside a single box.
    """

    if not lines:
        return []

    keyframes = sorted(keyframes_ms) if keyframes_ms else []

    slot_lookup = build_slot_lookup(speaker_map)

    def slot_key(speaker: Optional[str]) -> Optional[str]:
        return slot_key_for_speaker(speaker, slot_lookup)

    # Check if multi-speaker timing is needed
    unique_slots = {slot_key(line.speaker) for line in lines if line.speaker}
    if per_speaker or len(unique_slots) > 1:
        # Group lines by the slot they render in
        speaker_groups: dict[Optional[str], List[SubtitleLine]] = {}
        for line in lines:
            speaker_groups.setdefault(slot_key(line.speaker), []).append(line)

        all_processed: List[SegmentMS] = []
        for spk, spk_lines in speaker_groups.items():
            spk_segments = [SegmentMS(line) for line in spk_lines]
            spk_segments = _merge_micro_fragments_before_padding(
                spk_segments, snap_threshold_ms
            )
            spk_segments = _apply_interjection_merging(
                spk_segments,
                interjection_max_duration_ms,
                interjection_merge_threshold_ms,
                interjection_gap_threshold_ms,
            )
            spk_segments = _apply_min_duration_padding(
                spk_segments, keyframes, video_duration_ms, min_duration_ms
            )
            spk_segments = _apply_gap_snapping(
                spk_segments,
                keyframes,
                snap_threshold_ms,
                conditional_snap_threshold_ms,
                min_duration_ms if min_gap_ms is None else min_gap_ms,
            )
            spk_segments = _apply_micro_snapping(
                spk_segments, keyframes, snap_threshold_ms, video_duration_ms
            )
            spk_segments = _prevent_slot_overlaps(spk_segments, min_duration_ms)
            all_processed.extend(spk_segments)

        all_processed.sort(key=lambda seg: (seg.start_ms, seg.end_ms))
        segments = _resolve_cross_slot_transitions(
            all_processed,
            slot_lookup=slot_lookup,
            min_duration_ms=min_duration_ms,
            video_duration_ms=video_duration_ms,
        )
    else:
        segments = [SegmentMS(line) for line in lines]
        segments = _merge_micro_fragments_before_padding(
            segments, snap_threshold_ms
        )
        segments = _apply_interjection_merging(
            segments,
            interjection_max_duration_ms,
            interjection_merge_threshold_ms,
            interjection_gap_threshold_ms,
        )
        segments = _apply_min_duration_padding(
            segments, keyframes, video_duration_ms, min_duration_ms
        )
        segments = _apply_gap_snapping(
            segments,
            keyframes,
            snap_threshold_ms,
            conditional_snap_threshold_ms,
            min_duration_ms if min_gap_ms is None else min_gap_ms,
        )
        segments = _apply_micro_snapping(
            segments, keyframes, snap_threshold_ms, video_duration_ms
        )
        segments = _prevent_slot_overlaps(segments, min_duration_ms)

    # Screen-level gaps are deliberately not closed here. They can only be settled
    # once translation has finished reshaping the cues, so postprocess owns that
    # rule; closing it here too would mean the same decision in two places.

    # Final Bounds Check
    for seg in segments:
        if seg.start_ms < 0:
            seg.start_ms = 0
        if video_duration_ms is not None and seg.end_ms > video_duration_ms:
            seg.end_ms = video_duration_ms

    return [seg.to_subtitle_line() for seg in segments]
