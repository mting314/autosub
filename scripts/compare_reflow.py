#!/usr/bin/env python3
"""Eyeball the deterministic line-break reflow on real, already-translated data.

Zero LLM cost: replays the raw model output saved in a `--save-log` run's
`chunks/*_output.json` through the production reflow path, recovering real
per-line timing and speaker by aligning the chunk inputs against the episode's
`_original.ass`. Prints a focused BEFORE/AFTER diff of only the sentence groups
that reflow changed.

Usage:
    uv run python scripts/compare_reflow.py "<path to a *_logs*/chunks dir>" [original.ass]

If the original.ass is omitted, it is auto-detected as the sibling
`*_original.ass` of the log directory; failing that, equal line durations are
assumed (grouping still works, only duration weighting is approximated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyass

from autosub.pipeline.translate.reflow import (
    LONG_GAP_S,
    _forbidden_end,
    reflow_line_breaks,
)


def _load_chunks(chunks_dir: Path) -> tuple[list[str], list[str]]:
    """Reassemble (ja_inputs, en_outputs) across all chunks, in order."""
    ja: list[str] = []
    en: list[str] = []
    out_files = sorted(chunks_dir.glob("chunk_*_output.json"))
    if not out_files:
        sys.exit(f"No chunk_*_output.json found in {chunks_dir}")
    for out_file in out_files:
        in_file = Path(str(out_file).replace("_output.json", "_input.json"))
        outs = json.loads(out_file.read_text(encoding="utf-8"))
        outs = outs if isinstance(outs, list) else outs.get("subtitle_translations", [])
        ins = json.loads(in_file.read_text(encoding="utf-8")) if in_file.exists() else []
        en.extend(o.get("translated", "") for o in outs)
        ja.extend(i.get("text", "") for i in ins)
    return ja, en


def _load_original(original_ass: Path) -> list[pyass.Event]:
    with open(original_ass, encoding="utf-8") as handle:
        script = pyass.load(handle)
    events = []
    for ev in script.events:
        if not isinstance(ev, pyass.Event):
            continue
        if ev.format == pyass.EventFormat.COMMENT:
            continue
        if ev.text and ev.text.strip():
            events.append(ev)
    return events


def _align(ja: list[str], events: list[pyass.Event]) -> int | None:
    """Return the offset where the ja slice appears contiguously in events."""
    if not ja or not events:
        return None
    ja_all = [e.text for e in events]
    first = ja[0]
    for start in range(len(ja_all) - len(ja) + 1):
        if ja_all[start] == first and ja_all[start : start + len(ja)] == ja:
            return start
    return None


def _derive(events: list[pyass.Event]) -> tuple[list[float], set[int]]:
    durations: list[float] = []
    boundaries: set[int] = set()
    for i, ev in enumerate(events):
        durations.append(max(0.0, (ev.end - ev.start).total_seconds()))
        if i == 0:
            continue
        prev = events[i - 1]
        if ev.style != prev.style:
            boundaries.add(i)
        elif (ev.start - prev.end).total_seconds() > LONG_GAP_S:
            boundaries.add(i)
    return durations, boundaries


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    chunks_dir = Path(sys.argv[1])
    ja, en = _load_chunks(chunks_dir)

    if len(sys.argv) >= 3:
        original_ass: Path | None = Path(sys.argv[2])
    else:
        parent = chunks_dir.parent.parent
        matches = sorted(parent.glob("*_original.ass"))
        original_ass = matches[0] if matches else None

    durations: list[float]
    boundaries: set[int]
    if original_ass and original_ass.exists():
        events = _load_original(original_ass)
        offset = _align(ja, events)
        if offset is not None:
            sliced = events[offset : offset + len(en)]
            durations, boundaries = _derive(sliced)
            print(f"Aligned to {original_ass.name} at event offset {offset} "
                  f"(real timing + speaker boundaries).\n")
        else:
            durations = [2.0] * len(en)
            boundaries = set()
            print("Could not align to original.ass; using equal durations.\n")
    else:
        durations = [2.0] * len(en)
        boundaries = set()
        print("No original.ass; using equal durations.\n")

    reflowed = reflow_line_breaks(en, durations, boundaries)

    # Report: group changed lines into contiguous runs for readable context.
    changed = [i for i in range(len(en)) if en[i] != reflowed[i]]
    if not changed:
        print("No lines changed by reflow.")
        return

    runs: list[list[int]] = []
    for i in changed:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])

    dangling_before = sum(
        1 for t in en if t.split() and _forbidden_end(t.split()[-1])
    )
    dangling_after = sum(
        1 for t in reflowed if t.split() and _forbidden_end(t.split()[-1])
    )

    print(f"{len(changed)} line(s) changed across {len(runs)} sentence group(s).")
    print(f"Dangling line-endings: {dangling_before} -> {dangling_after}\n")
    print("=" * 78)
    for run in runs:
        lo, hi = run[0], run[-1]
        for i in range(lo, hi + 1):
            print(f"  BEFORE [{i}] {en[i]}")
        print("  " + "-" * 40)
        for i in range(lo, hi + 1):
            print(f"  AFTER  [{i}] {reflowed[i]}")
        print("=" * 78)


if __name__ == "__main__":
    main()
