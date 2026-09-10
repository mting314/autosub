#!/usr/bin/env python3
"""Measure whether the strengthened translation prompt reduces dangling line-ends.

Re-translates the Japanese inputs from a saved --save-log run using the CURRENT
translator prompt (no profile context, matching the baseline run's saved
system_prompt.txt), and compares dangling line-endings against the saved raw
output. This isolates the prompt as the only variable.

Usage:
    uv run python scripts/test_prompt_reflow.py "<chunks dir>" [chunk_nums...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from autosub.core.config import PROJECT_ID
from autosub.core.llm import ReasoningEffort
from autosub.pipeline.translate.reflow import _forbidden_end
from autosub.pipeline.translate.translator import VertexTranslator


def _danglers(lines: list[str]) -> list[str]:
    out = []
    for t in lines:
        words = t.split()
        if words and _forbidden_end(words[-1]):
            out.append(t)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    chunks_dir = Path(sys.argv[1])
    wanted = sys.argv[2:]  # optional list of chunk numbers like 03 05 06

    out_files = sorted(chunks_dir.glob("chunk_*_output.json"))
    if wanted:
        out_files = [
            f for f in out_files if any(f"chunk_{w}" in f.name for w in wanted)
        ]

    translator = VertexTranslator(
        project_id=PROJECT_ID,
        system_prompt=None,  # match baseline: base rules only, no profile context
        model="gemini-3-flash-preview",
        provider="google-vertex",
        reasoning_effort=ReasoningEffort.LOW,  # avoid thinking-explosion on this branch
    )

    base_total = new_total = 0
    for out_file in out_files:
        in_file = Path(str(out_file).replace("_output.json", "_input.json"))
        ins = json.loads(in_file.read_text(encoding="utf-8"))
        ja = [i["text"] for i in ins]
        baseline = [
            o["translated"] for o in json.loads(out_file.read_text(encoding="utf-8"))
        ]

        new = translator.translate(ja)

        b = _danglers(baseline)
        n = _danglers(new)
        base_total += len(b)
        new_total += len(n)
        print(
            f"\n=== {out_file.name}: baseline danglers={len(b)}  new-prompt danglers={len(n)} ==="
        )
        for t in b:
            print(f"  BASELINE dangles: {t}")
        for t in n:
            print(f"  NEW-PROMPT dangles: {t}")

    print(
        f"\n########  TOTAL dangling line-ends: baseline={base_total}  new-prompt={new_total}  ########"
    )


if __name__ == "__main__":
    main()
