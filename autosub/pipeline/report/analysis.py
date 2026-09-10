from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pyass

logger = logging.getLogger(__name__)


@dataclass
class LineReport:
    index: int  # 1-based
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


def _dialogue_events(script: pyass.Script) -> list[pyass.Event]:
    return [e for e in script.events if e.format != pyass.EventFormat.COMMENT]


def analyze_lines(
    jp_events: list[pyass.Event],
    en_events: list[pyass.Event],
    *,
    short_ratio: float = 0.5,
    long_ratio: float = 2.5,
    min_jp_chars_for_long: int = 4,
    max_short_en_chars: int = 10,
    zero_duration_threshold: float = 0.1,
    large_gap_threshold: float = 30.0,
) -> tuple[list[LineReport], ReportStats]:
    if len(jp_events) != len(en_events):
        logger.warning(
            "Line count mismatch: %d JP events vs %d EN events. "
            "Pairing by index up to the shorter list.",
            len(jp_events),
            len(en_events),
        )

    pair_count = min(len(jp_events), len(en_events))
    lines: list[LineReport] = []
    total_jp_chars = 0
    total_en_chars = 0
    issue_counts: dict[str, int] = {}

    for i in range(pair_count):
        jp_ev = jp_events[i]
        en_ev = en_events[i]

        start = jp_ev.start.total_seconds()
        end = jp_ev.end.total_seconds()
        duration = end - start
        jp_text = jp_ev.text
        en_text = en_ev.text
        style = jp_ev.style

        jp_len = len(jp_text)
        en_len = len(en_text)
        total_jp_chars += jp_len
        total_en_chars += en_len

        issues: set[str] = set()

        # Zero/near-zero duration
        if duration < zero_duration_threshold:
            issues.add("zero_duration")

        # Short translation
        ratio = en_len / max(jp_len, 1)
        if ratio < short_ratio and en_len < max_short_en_chars:
            issues.add("short")

        # Long translation
        if ratio > long_ratio and jp_len >= min_jp_chars_for_long:
            issues.add("long")

        # Large gap to next line
        if i + 1 < pair_count:
            next_start = jp_events[i + 1].start.total_seconds()
            gap = next_start - end
            if gap > large_gap_threshold:
                issues.add("large_gap")

        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        lines.append(
            LineReport(
                index=i + 1,
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
        line_count=pair_count,
        jp_char_count=total_jp_chars,
        en_char_count=total_en_chars,
        en_jp_ratio=round(en_jp_ratio, 2),
        issue_counts=issue_counts,
    )

    return lines, stats
