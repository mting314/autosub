from __future__ import annotations

import html
import logging
import os
from pathlib import Path

import pyass

from autosub.pipeline.report.analysis import (
    LineReport,
    ReportStats,
    _dialogue_events,
    analyze_lines,
)
from autosub.pipeline.report.template import HTML_BODY_TEMPLATE, HTML_HEAD, HTML_SCRIPT

logger = logging.getLogger(__name__)

ISSUE_META: dict[str, tuple[str, str]] = {
    "short": ("Short translations", "btn-orange"),
    "long": ("Long translations", "btn-yellow"),
    "zero_duration": ("Zero duration", "btn-red"),
    "large_gap": ("Large gaps", "btn-purple"),
}

SEVERITY_CLASS: dict[str, str] = {
    "short": "severity-orange",
    "long": "severity-yellow",
    "zero_duration": "severity-red",
    "large_gap": "severity-purple",
}


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _build_stats_html(stats: ReportStats) -> str:
    cards = [
        (str(stats.line_count), "Lines"),
        (str(stats.jp_char_count), "JP Characters"),
        (str(stats.en_char_count), "EN Characters"),
        (str(stats.en_jp_ratio), "EN/JP Ratio"),
    ]
    parts = ['<div class="stats-grid">']
    for value, label in cards:
        parts.append(
            f'  <div class="stat-card">'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _build_filter_buttons(stats: ReportStats) -> str:
    buttons: list[str] = []
    for issue_type, (label, btn_class) in ISSUE_META.items():
        count = stats.issue_counts.get(issue_type, 0)
        if count > 0:
            buttons.append(
                f'<button class="filter-btn {btn_class}" '
                f'data-filter="{issue_type}" onclick="toggleFilter(this)">'
                f"{label} ({count})</button>"
            )
    return "\n  ".join(buttons)


def _build_video_section(
    video_path: Path | None, html_output_path: Path
) -> str:
    if video_path is None:
        return ""
    try:
        rel = os.path.relpath(video_path, html_output_path.parent)
    except ValueError:
        rel = str(video_path)
    escaped_src = _escape(rel)
    return (
        '<div id="player-section">\n'
        '  <div id="video-wrap">\n'
        '    <video id="video" controls preload="metadata">\n'
        f'      <source src="{escaped_src}">\n'
        "    </video>\n"
        "  </div>\n"
        '  <div id="now-playing">\n'
        '    <div id="now-header">\n'
        '      <span id="now-line-num"></span>\n'
        '      <span id="now-time"></span>\n'
        "    </div>\n"
        '    <div id="now-jp" class="now-text">Click a line to preview</div>\n'
        '    <div id="now-en" class="now-text"></div>\n'
        "  </div>\n"
        "</div>"
    )


def _build_table_rows(lines: list[LineReport]) -> str:
    rows: list[str] = []
    for line in lines:
        severity_classes = " ".join(
            SEVERITY_CLASS[issue] for issue in sorted(line.issues) if issue in SEVERITY_CLASS
        )
        issue_attrs = " ".join(
            f'data-issue-{issue}="1"' for issue in sorted(line.issues)
        )

        duration = line.end_seconds - line.start_seconds
        jp_escaped = _escape(line.jp_text)
        en_escaped = _escape(line.en_text)
        start_ts = _format_timestamp(line.start_seconds)
        end_ts = _format_timestamp(line.end_seconds)

        row = (
            f'<tr class="{severity_classes}" {issue_attrs} '
            f'data-start="{line.start_seconds:.3f}" '
            f'data-end="{line.end_seconds:.3f}" '
            f'data-jp="{jp_escaped}" data-en="{en_escaped}" '
            f'onclick="onRowClick(this)">'
            f"<td>{line.index}</td>"
            f"<td>{start_ts}</td>"
            f"<td>{end_ts}</td>"
            f"<td>{duration:.2f}s</td>"
            f"<td>{_escape(line.style)}</td>"
            f'<td class="jp-text">{jp_escaped}</td>'
            f'<td class="en-text">{en_escaped}</td>'
            f"<td>{len(line.jp_text)}/{len(line.en_text)}</td>"
            f"</tr>"
        )
        rows.append(row)
    return "\n".join(rows)


def generate_report(
    original_ass_path: Path,
    translated_ass_path: Path,
    output_html_path: Path,
    *,
    video_path: Path | None = None,
    title: str | None = None,
) -> None:
    with open(original_ass_path, encoding="utf-8-sig") as f:
        jp_script = pyass.load(f)
    with open(translated_ass_path, encoding="utf-8-sig") as f:
        en_script = pyass.load(f)

    jp_events = _dialogue_events(jp_script)
    en_events = _dialogue_events(en_script)

    lines, stats = analyze_lines(jp_events, en_events)

    if title is None:
        title = original_ass_path.stem

    head = HTML_HEAD.format(title=_escape(title))
    stats_html = _build_stats_html(stats)
    filter_buttons = _build_filter_buttons(stats)
    video_section = _build_video_section(video_path, output_html_path)
    table_rows = _build_table_rows(lines)

    body = HTML_BODY_TEMPLATE.format(
        title=_escape(title),
        stats_html=stats_html,
        video_section=video_section,
        filter_buttons=filter_buttons,
        table_rows=table_rows,
    )

    full_html = head + body + HTML_SCRIPT

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(full_html, encoding="utf-8")
    logger.info("Report saved to %s", output_html_path)
