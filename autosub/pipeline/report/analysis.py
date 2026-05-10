from __future__ import annotations

import logging
from dataclasses import dataclass, field

from autosub.core.schemas import SubtitleCue

logger = logging.getLogger(__name__)


@dataclass
class LineReport:
    index: int  # 1-based
    cue_id: str
    start_seconds: float
    end_seconds: float
    style: str
    jp_text: str
    en_text: str
    issues: set[str] = field(default_factory=set)


@dataclass
class ReportStats:
    line_count: int
    jp_char_count: int
    en_char_count: int
    en_jp_ratio: float
    issue_counts: dict[str, int]


def _cue_style(cue: SubtitleCue) -> str:
    return cue.role or cue.speaker or "Default"


def _cue_source_text(cue: SubtitleCue) -> str:
    return cue.normalized_source_text or cue.source_text


def _cue_translated_text(cue: SubtitleCue) -> str:
    return cue.final_text or cue.translated_text or ""


def analyze_cues(
    cues: list[SubtitleCue],
    *,
    short_ratio: float = 0.5,
    long_ratio: float = 2.5,
    min_jp_chars_for_long: int = 4,
    max_short_en_chars: int = 10,
    zero_duration_threshold: float = 0.1,
    large_gap_threshold: float = 30.0,
) -> tuple[list[LineReport], ReportStats]:
    lines: list[LineReport] = []
    total_jp_chars = 0
    total_en_chars = 0
    issue_counts: dict[str, int] = {}

    for i, cue in enumerate(cues):
        start = cue.start_time
        end = cue.end_time
        duration = end - start
        jp_text = _cue_source_text(cue)
        en_text = _cue_translated_text(cue)
        style = _cue_style(cue)

        jp_len = len(jp_text)
        en_len = len(en_text)
        total_jp_chars += jp_len
        total_en_chars += en_len

        issues: set[str] = set()

        if duration < zero_duration_threshold:
            issues.add("zero_duration")

        ratio = en_len / max(jp_len, 1)
        if ratio < short_ratio and en_len < max_short_en_chars:
            issues.add("short")

        if ratio > long_ratio and jp_len >= min_jp_chars_for_long:
            issues.add("long")

        if i + 1 < len(cues):
            next_start = cues[i + 1].start_time
            gap = next_start - end
            if gap > large_gap_threshold:
                issues.add("large_gap")

        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        lines.append(
            LineReport(
                index=i + 1,
                cue_id=cue.id,
                start_seconds=start,
                end_seconds=end,
                style=style,
                jp_text=jp_text,
                en_text=en_text,
                issues=issues,
            )
        )

    en_jp_ratio = total_en_chars / max(total_jp_chars, 1)

    stats = ReportStats(
        line_count=len(cues),
        jp_char_count=total_jp_chars,
        en_char_count=total_en_chars,
        en_jp_ratio=round(en_jp_ratio, 2),
        issue_counts=issue_counts,
    )

    return lines, stats
