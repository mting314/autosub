"""LLM-backed re-split engine for line-break reflow.

The deterministic engine (`reflow._resplit_group`) handles the common cases —
dangling connectives, mid-phrase breaks near punctuation — but has no syntactic
understanding, so duration weighting can occasionally pick an awkward break. This
engine asks a cheap model (flash-lite, same tier as the classification
extensions) to split each sentence group into its slots at natural boundaries.

It is a drop-in `reflow.Resplitter`: it takes the whole batch of sentence groups
and serves them in a single LLM call. Every proposal it returns still passes
through reflow's text-preservation guardrail and quality gate, so the LLM can
only improve on the original split — never corrupt or churn it. Any failure
degrades gracefully to "no proposal" (the deterministic result or original split
is kept).
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.pipeline.translate.reflow import Resplitter, _deterministic_batch

logger = logging.getLogger(__name__)


class ReflowSplit(BaseModel):
    id: int
    pieces: list[str]


class ReflowSplitter(BaseStructuredLLM):
    DEFAULT_MODELS = {
        "google-vertex": "gemini-2.5-flash-lite",
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
    }
    _DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash-lite"

    # Reflow output is small (just the split pieces) and lite models reject the
    # 65536 default, so cap the output window well below any lite model's limit.
    _MAX_OUTPUT_TOKENS = 8192

    def __init__(
        self,
        *,
        project_id: str | None,
        model: str | None = None,
        location: str = "global",
        provider: str = "google-vertex",
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.LOW,
        provider_options: dict[str, object] | None = None,
        **kwargs,
    ):
        resolved_model = model or self.DEFAULT_MODELS.get(
            provider, self._DEFAULT_GOOGLE_MODEL
        )
        # Scope a small max_tokens to this splitter (overrides the global default)
        # so lite models accept the request; caller options still win.
        options = {"max_tokens": self._MAX_OUTPUT_TOKENS}
        if provider_options:
            options.update(provider_options)
        super().__init__(
            project_id=project_id,
            model=resolved_model,
            location=location,
            provider=provider,
            reasoning_effort=reasoning_effort,
            provider_options=options,
            **kwargs,
        )

    def _system_instruction(self) -> str:
        return (
            "You are a subtitle line-break editor. Each input item is one English "
            "sentence that must be shown across a fixed number of subtitle lines "
            "with fixed timing.\n\n"
            "For each item, split 'text' into EXACTLY 'slots' pieces and return "
            "them in order as 'pieces'.\n\n"
            "Rules:\n"
            "1. Do NOT change any words, spelling, casing, or punctuation. Only "
            "decide where the line breaks fall. The pieces concatenated with "
            "single spaces must equal the original text exactly.\n"
            "2. Break only at natural boundaries: after punctuation, or before a "
            "conjunction/preposition that starts a new clause.\n"
            "3. NEVER end a piece on a dangling conjunction, preposition, or "
            "article (e.g. 'but', 'and', 'so', 'to', 'of', 'the'). Move such a "
            "word to the start of the next piece.\n"
            "4. Balance the pieces by 'durations' (seconds): a line shown longer "
            "may hold more text; a very short slot should hold less.\n"
            "5. Every piece must be non-empty.\n\n"
            "Output: valid JSON only, one item per input id, each with 'id' and "
            "'pieces' (a list of exactly 'slots' strings)."
        )

    def resplit_batch(
        self, batch: list[tuple[list[str], list[float]]]
    ) -> list[list[str] | None]:
        payload = [
            {
                "id": i,
                "text": " ".join(p.strip() for p in pieces),
                "slots": len(pieces),
                "durations": [round(d, 2) for d in durs],
            }
            for i, (pieces, durs) in enumerate(batch)
        ]
        contents = json.dumps(payload, ensure_ascii=False, indent=2)

        results, _diag = self._run_structured_output(
            user_prompt=contents,
            system_prompt=self._system_instruction(),
            output_type=list[ReflowSplit],
            operation_name="LLM line-break reflow",
            output_name="reflow_splits",
        )

        by_id = {r.id: r.pieces for r in results}
        proposals: list[list[str] | None] = []
        for i, (pieces, _durs) in enumerate(batch):
            got = by_id.get(i)
            # Enforce the slot count here; text/quality are enforced by reflow's
            # guardrail and quality gate downstream.
            proposals.append(got if got is not None and len(got) == len(pieces) else None)
        return proposals


def build_llm_resplitter(
    *,
    project_id: str | None,
    model: str | None = None,
    location: str = "global",
    provider: str = "google-vertex",
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.LOW,
) -> Resplitter:
    """Return a reflow Resplitter backed by the LLM, degrading to deterministic.

    If the LLM call fails or returns nothing usable for a group, that group
    falls back to the deterministic re-split, so enabling the LLM engine never
    does worse than the deterministic engine.
    """
    splitter = ReflowSplitter(
        project_id=project_id,
        model=model,
        location=location,
        provider=provider,
        reasoning_effort=reasoning_effort,
    )

    def _resplitter(batch: list[tuple[list[str], list[float]]]) -> list[list[str] | None]:
        deterministic = _deterministic_batch(batch)
        try:
            llm = splitter.resplit_batch(batch)
        except Exception as exc:
            logger.warning(
                "LLM reflow engine failed (%s: %s); using deterministic re-split.",
                type(exc).__name__,
                exc,
            )
            return deterministic
        # Prefer the LLM proposal per group; fall back to deterministic where the
        # LLM declined or returned an unusable shape.
        return [
            llm[i] if llm[i] is not None else deterministic[i]
            for i in range(len(batch))
        ]

    return _resplitter
