from typing import List, Optional

from autosub.core.schemas import ReplacementSpan, SubtitleLine, TranscribedWord
from autosub.core.speaker_map import build_slot_lookup

# Breathing room left between two lines that share the same on-screen slot.
SLOT_OVERLAP_GAP_MS = 50


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
                if _roles_compatible_for_merge(seg.role, next_seg.role):
                    # Merge with the next segment when semantic role stays compatible.
                    separator = " " if seg.text and next_seg.text else ""
                    right_orig_offset = _original_text_length(
                        seg.text, seg.replacement_spans
                    ) + len(separator)
                    right_replaced_offset = len(seg.text) + len(separator)
                    seg.replacement_spans.extend(
                        _offset_replacement_spans(
                            next_seg.replacement_spans,
                            orig_offset=right_orig_offset,
                            replaced_offset=right_replaced_offset,
                        )
                    )
                    seg.text = f"{seg.text}{separator}{next_seg.text}".strip()
                    seg.end_ms = next_seg.end_ms
                    if separator and seg.words:
                        # Keep concatenated word text aligned with seg.text so
                        # char-position walks in find_split_time stay exact.
                        seg.words[-1] = seg.words[-1].model_copy(
                            update={"word": seg.words[-1].word + separator}
                        )
                    seg.words.extend(next_seg.words)
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
                # Multiple Keyframes in Gap
                prev_seg.end_ms = kfs_in_gap[0]
                next_seg.start_ms = kfs_in_gap[-1]
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
                if span_gap <= interjection_merge_threshold_ms:
                    # MERGE: combine A's lines into one
                    prev_seg.text = f"{prev_seg.text} {next_seg.text}".strip()
                    prev_seg.end_ms = next_seg.end_ms
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


def _prevent_slot_overlaps(segments: List[SegmentMS]) -> List[SegmentMS]:
    """Ensure no two segments that share an on-screen slot are visible at once.

    Callers must pass a group of segments that all render in the same box. Two
    events at the same position do not stack, they draw on top of each other, so
    the earlier one is truncated to clear the later one.
    """
    resolved: List[SegmentMS] = []
    for segment in sorted(segments, key=lambda seg: (seg.start_ms, seg.end_ms)):
        if not resolved:
            resolved.append(segment)
            continue

        previous = resolved[-1]
        if segment.start_ms >= previous.end_ms:
            resolved.append(segment)
            continue

        truncated_end = segment.start_ms - SLOT_OVERLAP_GAP_MS
        if truncated_end > previous.start_ms:
            previous.end_ms = truncated_end
            resolved.append(segment)
        else:
            # Starts land on top of each other, so there is nothing to truncate
            # to. Merging keeps both texts readable instead of stacking glyphs.
            previous.text = f"{previous.text} {segment.text}".strip()
            previous.end_ms = max(previous.end_ms, segment.end_ms)
    return resolved


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
        """Identify the box a line renders in, falling back to the raw label."""
        if speaker is None:
            return None
        slot = slot_lookup.get(str(speaker))
        return f"slot:{slot}" if slot is not None else speaker

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
            )
            spk_segments = _apply_micro_snapping(
                spk_segments, keyframes, snap_threshold_ms, video_duration_ms
            )
            spk_segments = _prevent_slot_overlaps(spk_segments)
            all_processed.extend(spk_segments)

        all_processed.sort(key=lambda seg: (seg.start_ms, seg.end_ms))
        segments = all_processed
    else:
        segments = [SegmentMS(line) for line in lines]
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
            segments, keyframes, snap_threshold_ms, conditional_snap_threshold_ms
        )
        segments = _apply_micro_snapping(
            segments, keyframes, snap_threshold_ms, video_duration_ms
        )
        segments = _prevent_slot_overlaps(segments)

    # Final Bounds Check
    for seg in segments:
        if seg.start_ms < 0:
            seg.start_ms = 0
        if video_duration_ms is not None and seg.end_ms > video_duration_ms:
            seg.end_ms = video_duration_ms

    return [seg.to_subtitle_line() for seg in segments]
